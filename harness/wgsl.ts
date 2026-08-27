import { create, globals } from "webgpu";
import { bindingTypeMismatch, kernelName, storageElementTypes } from "./binding-types.js";

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

/**
 * Instances and adapters this process has created, kept reachable on purpose.
 *
 * The Dawn Node binding does not keep the `GPU` instance alive from a
 * `GPUDevice`. Nothing in the returned object graph refers back to it, so once
 * the instance goes out of scope the collector is free to take it — and a
 * device whose instance has been collected starts failing dispatches. The
 * failure is not a clean error: the process aborts with a glibc/futex
 * `std::system_error`, segfaults, or hangs, and which one it does varies run to
 * run. That is the signature of a race, and it is why this looked like flake.
 *
 * Measured, A-B-A, on `ops/gqa/wgsl.test.ts` (46 GPU cases, RTX 5090, driver
 * 610.57.04, `webgpu` 0.4.0, Node v25.6.1, otherwise-idle GPU):
 *
 *   - without this array: **0/5 runs completed** (four produced no test result
 *     at all, one hung)
 *   - with it:            **5/5 runs green**, 46/46 each
 *   - reverted again:     **0/4**
 *
 * The only difference between those conditions is this reference. No dispatch,
 * allocation or ordering changed.
 *
 * This is why issues #38, #49, #68 and #107 all resisted diagnosis: they are
 * the same bug seen from different angles. #49 ("a test that takes more than a
 * few milliseconds before its first dispatch kills the worker") and #107
 * ("real-model-scale CPU work before a dispatch crashes the binding") both
 * describe *more time and more allocation for the collector to run in*. #68 ("a
 * file dies once it holds too many dispatches") describes the same pressure
 * from allocation volume. And the four vitest pool configurations that were
 * tried and did not help could not have helped: vitest calls a test body as a
 * function, so an instance created inside one is unreachable the moment that
 * body returns, whatever the pool does.
 *
 * Credit: the hypothesis and the minimal repro came from the voxshot session,
 * which isolated it to `nested` vs `nested-keep` in 35 lines with no model and
 * no weights.
 *
 * Held as an array rather than a single binding because `createRunner` can be
 * called more than once in a run (`harness/timing.test.ts` calls it directly),
 * and every instance has to outlive its device. Nothing reads this array;
 * being reachable is its whole job — `harness/instance-retention.test.ts`
 * exists to stop it being deleted as dead code.
 */
const retainedInstances: unknown[] = [];

/** Resolves to null when no adapter is available, so suites can skip. */
export async function createRunner(): Promise<Runner | null> {
  Object.assign(globalThis, globals);
  const gpu = create([]) as GPU;
  const adapter = await gpu.requestAdapter();
  retainedInstances.push(gpu, adapter);
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
  // **The adapter's own ceiling, not a number chosen here.**
  //
  // This was 512 MiB, and that constant was the reason `examples/zimage-vae`
  // could not decode at the resolution the model is trained for: a 1024 image's
  // last up block needs a 512 MiB binding and a 1 GB buffer, and Dawn's own
  // error said so — "this adapter supports a higher maxBufferSize of
  // 1099511627776, which can be specified in requiredLimits". The device was
  // refusing what the hardware offers because this line did not ask for it.
  //
  // Capped at what the adapter reports, since asking for more than a device
  // supports fails `requestDevice` outright, and the weakest devices are the
  // ones this library most needs to keep working on. `Math.min` below is what
  // does the capping; there is nothing left for a constant to decide.
  const timestamps = adapter.features.has("timestamp-query");
  const device = await adapter.requestDevice({
    requiredFeatures: timestamps ? ["timestamp-query"] : [],
    requiredLimits: {
      maxStorageBufferBindingSize: adapter.limits.maxStorageBufferBindingSize,
      maxBufferSize: adapter.limits.maxBufferSize,
      // Same reason as the two above, and found the same way: a tiled matmul
      // wanting 17,408 B of workgroup storage was refused while Dawn's own
      // message said "This adapter supports a higher
      // maxComputeWorkgroupStorageSize of 49152, which can be specified in
      // requiredLimits". The spec default is 16,384 — enough for a 16x16 tile
      // and not for the register-tiled shapes #177 is about, so a sweep run
      // against the default measures only the shapes that were already there.
      maxComputeWorkgroupStorageSize: adapter.limits.maxComputeWorkgroupStorageSize,
      maxComputeInvocationsPerWorkgroup: adapter.limits.maxComputeInvocationsPerWorkgroup,
      // X only. Every kernel here declares a one-dimensional workgroup, so Y
      // and Z stay at their defaults rather than being raised for nothing.
      maxComputeWorkgroupSizeX: adapter.limits.maxComputeWorkgroupSizeX,
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

      // Issue #221. Checked here, before a single buffer exists, because the
      // whole class of bug this catches produces no error later: WebGPU copies
      // whatever bytes it is given into whatever the kernel declared, and an
      // `Int32Array` in an `array<f32>` reads back as denormals (#217). Every
      // op's `wgsl.test.ts` goes through this function, so every op is covered
      // by having tests at all rather than by anyone remembering to check.
      const mismatch = bindingTypeMismatch(
        kernelName(code),
        storageElementTypes(code),
        bindings.map((b) =>
          b.kind === "storage" ? b.data : b.kind === "out" ? b.type : null,
        ),
      );
      if (mismatch) throw new Error(mismatch);

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
        // `bytes` is always backed by a plain `ArrayBuffer` (constructed just
        // above from either a typed-array binding's own `.buffer` or a
        // uniform's `ArrayBuffer`), never a `SharedArrayBuffer` — but
        // `@webgpu/types`' `GPUAllowSharedBufferSource` and TypeScript 5.7's
        // generic `Uint8Array<ArrayBufferLike>` can't express that, so this is
        // a type-level cast, not a runtime one. `as any` rather than naming
        // `BufferSource`/`GPUAllowSharedBufferSource` explicitly: those types
        // only resolve where `DOM` lib is loaded, and this file's own
        // `tsconfig.json` deliberately has no `DOM` lib (see
        // `tsconfig.build.json`'s note on why `ops/`/`harness` stay browser-lib-free) —
        // the mismatch is invisible there and only surfaced once
        // `examples/llm-demo/tsconfig.json` (issue #106) became the first
        // program in this repo to type-check `@webgpu/types` together with
        // `DOM`. `browser-runtime.ts#run` has the identical cast for the
        // identical reason, in a file that *does* have `DOM` lib available.
        device.queue.writeBuffer(buffer, 0, bytes as any);
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
