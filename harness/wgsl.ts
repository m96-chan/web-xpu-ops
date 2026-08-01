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
  | { kind: "uniform"; data: ArrayBuffer };

export interface Dispatch {
  code: string;
  entry?: string;
  bindings: Binding[];
  workgroups: [number] | [number, number] | [number, number, number];
}

export interface Runner {
  run(dispatch: Dispatch): Promise<(Float32Array | Int32Array | Uint32Array)[]>;
  destroy(): void;
}

/** Resolves to null when no adapter is available, so suites can skip. */
export async function createRunner(): Promise<Runner | null> {
  Object.assign(globalThis, globals);
  const gpu = create([]) as GPU;
  const adapter = await gpu.requestAdapter();
  if (!adapter) return null;
  const device = await adapter.requestDevice();

  return {
    async run({ code, entry = "main", bindings, workgroups }) {
      const created: GPUBuffer[] = [];
      const outputs: { spec: Extract<Binding, { kind: "out" }>; buffer: GPUBuffer }[] = [];

      const bound = bindings.map((binding) => {
        if (binding.kind === "out") {
          const buffer = device.createBuffer({
            size: binding.length * 4,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
          });
          created.push(buffer);
          outputs.push({ spec: binding, buffer });
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

      // Without these two checks a shader that never ran still produces a
      // readback — of the zeros its output buffer was created with. Where the
      // expected value contains zeros that is a passing test over a kernel that
      // did nothing, and since every correctness claim here is "it agrees with
      // the reference", a silent no-op reading as agreement is the one failure
      // that could hollow out all of them at once. Issue #46.
      //
      // Nothing is destroyed on these paths on purpose. Tearing buffers down
      // while the device is in an error state crashed this binding in 4 of 5
      // runs; leaving them to the device's own teardown is stable in 5 of 5.
      // The dispatch is over either way, so the buffers are short-lived.
      const module = device.createShaderModule({ code });

      // Compilation is read through `getCompilationInfo` rather than an error
      // scope. Both report it, but wrapping module creation in a scope was
      // measured at 1 pass in 5 here, while this is 5 in 5.
      const compilation = await module.getCompilationInfo();
      const failures = compilation.messages.filter((message) => message.type === "error");
      if (failures.length > 0) {
        const where = (m: GPUCompilationMessage) => `${m.lineNum}:${m.linePos}: ${m.message}`;
        throw new Error(`shader failed to compile\n${failures.map(where).join("\n")}`);
      }

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

      const encoder = device.createCommandEncoder();
      const pass = encoder.beginComputePass();
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
      for (const buffer of created) buffer.destroy();
      return results;
    },
    destroy() {
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
