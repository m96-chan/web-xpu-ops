import { create, globals } from "webgpu";

/**
 * Runs a WGSL compute shader and reads its outputs back.
 *
 * Node-native, through Dawn — no browser, no page to serve, no secure-context
 * dance. An earlier version of this drove headless Chrome over Playwright and
 * every part of that turned out to be avoidable.
 */

/** One entry of a shader's `@group(0)` layout, in binding order. */
export type Binding =
  | { kind: "storage"; data: Float32Array | Int32Array | Uint32Array }
  | { kind: "out"; type: "f32" | "i32" | "u32"; length: number }
  /**
   * Storage the kernel uses but nobody supplies or inspects: no upload, no
   * readback, contents undefined (zero in practice).
   *
   * It exists for the roofline calibration, which streams hundreds of megabytes
   * to find out how fast this device moves memory. Through `storage` that would
   * upload the buffer, and through `out` it would copy it back — either one puts
   * a transfer of the same size next to the thing being timed, which is exactly
   * what must not happen when the measurement *is* the transfer rate.
   */
  | { kind: "scratch"; length: number }
  | { kind: "uniform"; data: ArrayBuffer };

export interface Dispatch {
  code: string;
  entry?: string;
  bindings: Binding[];
  workgroups: [number] | [number, number] | [number, number, number];
}

export interface Runner {
  run(dispatch: Dispatch): Promise<(Float32Array | Int32Array | Uint32Array)[]>;
  /**
   * Seconds of GPU time for the dispatch, or null when this device cannot say.
   *
   * Read from a timestamp query written around the compute pass, not from a
   * clock on the host. Wall-clock here measures buffer creation, submission and
   * the round trip waiting for a mapped readback — about a millisecond on this
   * machine, which is several times a real dispatch and swamps exactly the
   * quantity being measured. Measured while building the roofline: a wall-clock
   * slope reported 5.3 TB/s on a card whose ceiling is 1.8.
   *
   * `timestamp-query` is optional, and the devices most in need of an honest
   * ceiling advertise the fewest features. Null rather than a guess is the point
   * — rule 9 says an unmeasured figure must say so, and a fabricated one is
   * worse than none because it looks authoritative.
   */
  time(dispatch: Dispatch): Promise<number | null>;
  destroy(): void;
}

/**
 * The reason a shader could not be used, or null when it compiled.
 *
 * Split out as a pure function so it can be tested without a device. The half
 * that needs a GPU — whether Dawn reports a bad shader at all — is platform
 * behaviour and is measured in `harness/README` notes and issue #46. The half
 * that is ours is this: given messages, do we refuse. That distinction matters
 * because provoking a real compile failure crashes this binding in roughly four
 * runs in five, so an end-to-end test of it cannot be kept green, while this
 * can.
 */
export function compilationFailure(
  messages: readonly Pick<GPUCompilationMessage, "type" | "lineNum" | "linePos" | "message">[],
): string | null {
  const errors = messages.filter((message) => message.type === "error");
  if (errors.length === 0) return null;
  const where = (m: (typeof errors)[number]) => `${m.lineNum}:${m.linePos}: ${m.message}`;
  return `shader failed to compile\n${errors.map(where).join("\n")}`;
}

/** Resolves to null when no adapter is available, so suites can skip. */
export async function createRunner(): Promise<Runner | null> {
  Object.assign(globalThis, globals);
  const gpu = create([]) as GPU;
  const adapter = await gpu.requestAdapter();
  if (!adapter) return null;
  // Asked for explicitly, because the defaults are far below what the hardware
  // offers: this adapter allows a 2 GB storage binding but a device requested
  // with no limits caps them at 128 MiB. The roofline calibration streams more
  // than that on purpose — a buffer that fits in last-level cache measures cache
  // bandwidth, which is a real number about the wrong thing.
  //
  // Capped at what the adapter reports rather than requested blindly, since
  // asking for more than a device supports fails outright, and the weakest
  // devices are the ones this library most needs to keep working on.
  const wanted = 512 * 1024 * 1024;
  const timestamps = adapter.features.has("timestamp-query");
  const device = await adapter.requestDevice({
    requiredFeatures: timestamps ? ["timestamp-query"] : [],
    requiredLimits: {
      maxStorageBufferBindingSize: Math.min(wanted, adapter.limits.maxStorageBufferBindingSize),
      maxBufferSize: Math.min(wanted, adapter.limits.maxBufferSize),
    },
  });

  /**
   * Compiled modules, keyed by source.
   *
   * Validation costs an await, and a test file dispatches the same shader many
   * times over. Paying it per dispatch adds latency to every one of them, and
   * on this binding elapsed time before a dispatch is itself destabilising
   * (issue #49) — so the check is done once per distinct shader rather than
   * once per call.
   */
  const compiled = new Map<string, GPUShaderModule>();

  /**
   * Scratch buffers, kept and reused rather than reallocated per dispatch.
   *
   * Repeated large allocations abort this binding — measured: a 64 MiB scratch
   * buffer allocated and freed twice in one file kills the worker on the second
   * one, and 256 MiB kills it just as reliably (issue #51). The roofline
   * calibration needs the same large buffer several times over to take a slope,
   * so allocating per call is not merely wasteful here, it does not survive.
   *
   * Reuse is safe precisely because `scratch` is defined as storage nobody
   * supplies and nobody inspects — there is no content to carry over and no
   * reader to surprise. Keyed by binding position as well as size so two scratch
   * bindings in one dispatch never alias onto the same buffer.
   */
  const scratch = new Map<string, GPUBuffer>();

  /**
   * One dispatch. Returns the outputs, and the GPU time when asked for it.
   *
   * `run` and `time` share this so a timed dispatch is the same dispatch, not a
   * second code path that might diverge from the one under test.
   */
  async function dispatch(
    { code, entry = "main", bindings, workgroups }: Dispatch,
    clock: boolean,
  ): Promise<{ outputs: (Float32Array | Int32Array | Uint32Array)[]; seconds: number | null }> {
      // Checked before a single buffer exists. A dispatch that fails after
      // `queue.writeBuffer` leaves writes queued against buffers that are then
      // abandoned unsubmitted, and that was measured to destabilise this
      // binding; validating first means the failure path allocates nothing.
      //
      // Why it matters at all: without this a shader that never ran still
      // produces a readback — of the zeros its output buffer was created with.
      // Where the expected value contains zeros that is a passing test over a
      // kernel that did nothing, and since every correctness claim here is "it
      // agrees with the reference", a silent no-op reading as agreement is the
      // one failure that could hollow out all of them at once. Issue #46.
      //
      // Compilation is read through `getCompilationInfo` rather than an error
      // scope: both report it, but wrapping module creation in a scope was
      // measured at 1 pass in 5 here against 5 in 5 for this.
      let module = compiled.get(code);
      if (!module) {
        module = device.createShaderModule({ code });
        const failure = compilationFailure((await module.getCompilationInfo()).messages);
        if (failure) throw new Error(failure);
        compiled.set(code, module);
      }

      const created: GPUBuffer[] = [];
      const outputs: { spec: Extract<Binding, { kind: "out" }>; buffer: GPUBuffer }[] = [];

      const bound = bindings.map((binding, index) => {
        if (binding.kind === "out") {
          const buffer = device.createBuffer({
            size: binding.length * 4,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
          });
          created.push(buffer);
          outputs.push({ spec: binding, buffer });
          return buffer;
        }
        if (binding.kind === "scratch") {
          const key = `${index}:${binding.length}`;
          let buffer = scratch.get(key);
          if (!buffer) {
            buffer = device.createBuffer({
              size: Math.max(4, binding.length * 4),
              usage: GPUBufferUsage.STORAGE,
            });
            scratch.set(key, buffer);
          }
          // Deliberately not pushed to `created`: it outlives this dispatch.
          return buffer;
        }
        const bytes =
          binding.kind === "uniform"
            ? new Uint8Array(binding.data)
            : new Uint8Array(binding.data.buffer, binding.data.byteOffset, binding.data.byteLength);
        const buffer = device.createBuffer({
          // Uniform bindings have a 16-byte minimum.
          size: Math.max(binding.kind === "uniform" ? 16 : 4, bytes.byteLength),
          usage:
            binding.kind === "uniform"
              ? GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
              : GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        });
        device.queue.writeBuffer(buffer, 0, bytes);
        created.push(buffer);
        return buffer;
      });

      // The other route in: `layout: "auto"` omits a binding the shader never
      // mentions, and the bind group then fails validation. Not hypothetical —
      // it is why `attention` is two WGSL files rather than one file with two
      // entry points.
      device.pushErrorScope("validation");
      const pipeline = device.createComputePipeline({
        layout: "auto",
        compute: { module, entryPoint: entry },
      });
      const bindGroup = device.createBindGroup({
        layout: pipeline.getBindGroupLayout(0),
        entries: bound.map((buffer, binding) => ({ binding, resource: { buffer } })),
      });
      const invalid = await device.popErrorScope();
      if (invalid) throw new Error(`dispatch is not valid: ${invalid.message}`);

      // Written around the pass itself, so the number excludes buffer creation,
      // submission and the readback round trip — all of which dwarf a real
      // dispatch on this machine.
      const timing =
        clock && timestamps
          ? {
              set: device.createQuerySet({ type: "timestamp", count: 2 }),
              resolved: device.createBuffer({
                size: 16,
                usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC,
              }),
              readable: device.createBuffer({
                size: 16,
                usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
              }),
            }
          : null;

      const encoder = device.createCommandEncoder();
      const pass = encoder.beginComputePass(
        timing ? { timestampWrites: { querySet: timing.set, beginningOfPassWriteIndex: 0, endOfPassWriteIndex: 1 } } : undefined,
      );
      pass.setPipeline(pipeline);
      pass.setBindGroup(0, bindGroup);
      pass.dispatchWorkgroups(...(workgroups as [number, number?, number?]));
      pass.end();

      const staging = outputs.map(({ spec, buffer }) => {
        const read = device.createBuffer({
          size: spec.length * 4,
          usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
        });
        encoder.copyBufferToBuffer(buffer, 0, read, 0, spec.length * 4);
        created.push(read);
        return read;
      });
      if (timing) encoder.resolveQuerySet(timing.set, 0, 2, timing.resolved, 0);
      if (timing) encoder.copyBufferToBuffer(timing.resolved, 0, timing.readable, 0, 16);
      device.queue.submit([encoder.finish()]);

      const results: (Float32Array | Int32Array | Uint32Array)[] = [];
      for (const [index, read] of staging.entries()) {
        await read.mapAsync(GPUMapMode.READ);
        // The mapped range dies with the buffer, so copy before unmapping.
        const bytes = read.getMappedRange().slice(0);
        read.unmap();
        const type = outputs[index]!.spec.type;
        results.push(
          type === "i32"
            ? new Int32Array(bytes)
            : type === "u32"
              ? new Uint32Array(bytes)
              : new Float32Array(bytes),
        );
      }
      let seconds: number | null = null;
      if (timing) {
        await timing.readable.mapAsync(GPUMapMode.READ);
        const stamps = new BigUint64Array(timing.readable.getMappedRange().slice(0));
        timing.readable.unmap();
        // Timestamps are nanoseconds. A query set can return zeros when the
        // driver declines to serve it, and zero is not a duration.
        const elapsed = stamps[1]! - stamps[0]!;
        seconds = elapsed > 0n ? Number(elapsed) / 1e9 : null;
        timing.set.destroy();
        timing.resolved.destroy();
        timing.readable.destroy();
      }
      for (const buffer of created) buffer.destroy();
      return { outputs: results, seconds };
    }

  return {
    async run(spec) {
      return (await dispatch(spec, false)).outputs;
    },
    async time(spec) {
      if (!timestamps) return null;
      return (await dispatch(spec, true)).seconds;
    },
    destroy() {
      for (const buffer of scratch.values()) buffer.destroy();
      scratch.clear();
      device.destroy();
    },
  };
}

/** Packs a params struct of mixed u32 / i32 / f32 into a uniform buffer. */
export function params(fields: ["u32" | "i32" | "f32", number][]): ArrayBuffer {
  const buffer = new ArrayBuffer(Math.max(16, fields.length * 4));
  const view = new DataView(buffer);
  fields.forEach(([kind, value], index) => {
    if (kind === "f32") view.setFloat32(index * 4, value, true);
    else if (kind === "i32") view.setInt32(index * 4, value, true);
    else view.setUint32(index * 4, value, true);
  });
  return buffer;
}
