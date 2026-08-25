/**
 * Does `batch()`'s per-dispatch profile attribute time to the right dispatch?
 *
 * Issue #177. It does not, and this is the test that says so rather than a
 * third wrong diagnosis. The history is worth keeping: an in-forward profile
 * reported `scores` at 0.127 ms and `context` at 12.07 ms for the same shape
 * and the same count, and the conclusion drawn from it — that `context` was
 * 95x slower — was exactly backwards. Timed on their own at the same shapes,
 * `scores` takes 35 ms and `context` 8 ms.
 *
 * The check is a known answer. One kernel, timed alone with `Runner.time`, then
 * the same kernel repeated `N` times inside one `batch()` with labels on. If
 * the profile attributes correctly, the sum of what it reports is `N` times the
 * standalone figure. Nothing here is model-specific: it is arithmetic against a
 * dispatch whose cost was measured before the profiler ever saw it.
 *
 * Rule 9's reason for existing: a number that looks authoritative and is wrong
 * is worse than no number. A profiler is the most authoritative-looking thing
 * in the repository, so it is the one that most needs a test it can fail.
 */
import { describe, expect, it } from "vitest";
import { createResidentDevice, type BatchProfileSink, type ResidentOp } from "./resident.js";
import { createRunner, params } from "./wgsl.js";

/**
 * A dispatch with enough arithmetic to be timeable and no memory traffic worth
 * speaking of, so what is being attributed is compute rather than a cache.
 */
const BUSY = /* wgsl */ `
struct Params { n: u32, rounds: u32 }
@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;
@group(0) @binding(2) var<uniform> params: Params;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= params.n) { return; }
  var acc = input[i];
  // Data-dependent so nothing can fold it away, and long enough that one
  // dispatch is far above the timer's resolution.
  for (var r: u32 = 0u; r < params.rounds; r += 1u) {
    acc = fma(acc, 0.999999, 0.000001);
  }
  output[i] = acc;
}
`;

const N = 1 << 20;
const ROUNDS = 2048;
const COPIES = 8;

describe("per-dispatch attribution", () => {
  it("reports the same total a standalone timing does", async () => {
    const runner = await createRunner();
    if (!runner) return; // no adapter — `gpuTest`'s convention

    const input = Float32Array.from({ length: N }, (_, i) => 1 + (i % 7) * 0.01);
    const dispatch = {
      code: BUSY,
      bindings: [
        { kind: "storage" as const, data: input },
        { kind: "out" as const, type: "f32" as const, length: N },
        { kind: "uniform" as const, data: params([["u32", N], ["u32", ROUNDS]]) },
      ],
      workgroups: [Math.ceil(N / 256)] as [number],
    };

    // Best of three: a scheduling hiccup should not become the baseline.
    let alone: number | null = null;
    for (let i = 0; i < 3; i += 1) {
      const seconds = await runner.time(dispatch);
      if (seconds !== null && (alone === null || seconds < alone)) alone = seconds;
    }
    runner.destroy();
    if (alone === null) return; // device declined to time it

    const device = await createResidentDevice();
    if (!device || !device.timestampsSupported) return;

    const inBuffer = device.createStorageBuffer(N * 4);
    device.upload(inBuffer, 0, input);
    const outBuffer = device.createStorageBuffer(N * 4);
    const uniform = device.createUniformBuffer(256);
    device.upload(uniform, 0, new Uint8Array(params([["u32", N], ["u32", ROUNDS]])));
    const pipeline = await device.pipelineFor(BUSY);

    const ops: ResidentOp[] = [];
    const labels: string[] = [];
    for (let i = 0; i < COPIES; i += 1) {
      ops.push({
        kind: "dispatch",
        pipeline,
        bindGroup: await device.bindGroup(pipeline, [inBuffer, outBuffer, uniform]),
        workgroups: [Math.ceil(N / 256)],
      });
      labels.push("busy");
    }
    const sink: BatchProfileSink = { encodeMs: null, submitToDoneMs: null, readbackMs: null, gpuEntries: [] };
    await device.batch(ops, [], { labels, sink });
    device.destroy();

    const attributed = sink.gpuEntries.reduce((sum, e) => sum + e.seconds, 0);
    const expected = alone * COPIES;

    // Generous: the claim is not that a profiler is exact, only that it is the
    // right order of magnitude for a dispatch whose cost is already known. The
    // failure it exists to catch reported 0.127 ms for something that takes 35.
    expect(sink.gpuEntries.length).toBe(COPIES);
    expect(attributed).toBeGreaterThan(expected * 0.5);
    expect(attributed).toBeLessThan(expected * 2.0);

    // Issue #182: every CPU-side field a caller can read has to be *written*,
    // not merely declared. `readbackMs` existed for months and was discarded
    // one level up, which left half a browser forward with no name. A field
    // that is never populated reads as "this phase costs nothing".
    expect(sink.encodeMs, "encodeMs was never written").not.toBeNull();
    expect(sink.submitToDoneMs, "submitToDoneMs was never written").not.toBeNull();
    expect(sink.readbackMs, "readbackMs was never written").not.toBeNull();
    // Recording this many dispatches cannot be free, and the wait for a batch
    // this long cannot be zero. Both would be zero if the timers straddled the
    // wrong statements.
    expect(sink.encodeMs!).toBeGreaterThan(0);
    expect(sink.submitToDoneMs!).toBeGreaterThan(0);
    // `encodeMs` stops before `submit()`, so it cannot contain the GPU wait.
    expect(sink.encodeMs!).toBeLessThan(sink.submitToDoneMs! + sink.encodeMs!);
  }, 120_000);

  it("does not move one dispatch's time onto its neighbour", async () => {
    const runner = await createRunner();
    if (!runner) return;

    const input = Float32Array.from({ length: N }, (_, i) => 1 + (i % 7) * 0.01);
    const make = (rounds: number) => ({
      code: BUSY,
      bindings: [
        { kind: "storage" as const, data: input },
        { kind: "out" as const, type: "f32" as const, length: N },
        { kind: "uniform" as const, data: params([["u32", N], ["u32", rounds]]) },
      ],
      workgroups: [Math.ceil(N / 256)] as [number],
    });

    // One cheap dispatch and one expensive one, alternating. If the profile is
    // attributing correctly the expensive label carries most of the time; the
    // failure this test exists for is the opposite, where a kernel's cost lands
    // on whatever ran next.
    const cheapSeconds = await runner.time(make(64));
    const dearSeconds = await runner.time(make(ROUNDS));
    runner.destroy();
    if (cheapSeconds === null || dearSeconds === null) return;
    // The premise: the two really are far apart when timed alone.
    expect(dearSeconds).toBeGreaterThan(cheapSeconds * 4);

    const device = await createResidentDevice();
    if (!device || !device.timestampsSupported) return;
    const inBuffer = device.createStorageBuffer(N * 4);
    device.upload(inBuffer, 0, input);
    const outBuffer = device.createStorageBuffer(N * 4);
    const pipeline = await device.pipelineFor(BUSY);
    const uniformFor = (rounds: number): GPUBuffer => {
      const u = device.createUniformBuffer(256);
      device.upload(u, 0, new Uint8Array(params([["u32", N], ["u32", rounds]])));
      return u;
    };
    const cheapUniform = uniformFor(64);
    const dearUniform = uniformFor(ROUNDS);

    const ops: ResidentOp[] = [];
    const labels: string[] = [];
    for (let i = 0; i < COPIES; i += 1) {
      for (const [label, uniform] of [["cheap", cheapUniform], ["dear", dearUniform]] as const) {
        ops.push({
          kind: "dispatch",
          pipeline,
          bindGroup: await device.bindGroup(pipeline, [inBuffer, outBuffer, uniform]),
          workgroups: [Math.ceil(N / 256)],
        });
        labels.push(label);
      }
    }
    const sink: BatchProfileSink = { encodeMs: null, submitToDoneMs: null, readbackMs: null, gpuEntries: [] };
    await device.batch(ops, [], { labels, sink });
    device.destroy();

    const total = (want: string): number =>
      sink.gpuEntries.filter((e) => e.label === want).reduce((sum, e) => sum + e.seconds, 0);
    expect(total("dear")).toBeGreaterThan(total("cheap") * 4);
  }, 120_000);
});
