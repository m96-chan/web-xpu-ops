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

      const pipeline = device.createComputePipeline({
        layout: "auto",
        compute: { module: device.createShaderModule({ code }), entryPoint: entry },
      });
      const bindGroup = device.createBindGroup({
        layout: pipeline.getBindGroupLayout(0),
        entries: bound.map((buffer, binding) => ({ binding, resource: { buffer } })),
      });

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
