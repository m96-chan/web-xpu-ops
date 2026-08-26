/**
 * `ResidentDevice.reclaim()` against the card, at the sizes R2V stages.
 *
 * Issue #213. A script rather than a test: the behaviour only appears within a
 * few gigabytes of the device's ceiling, so it needs a card with 32 GB free and
 * about a minute — neither of which belongs in `npm test`.
 *
 * **It is judged by a readback, not by an error.** An out-of-memory
 * `createBuffer` does not throw: it returns an *invalid* buffer, and the
 * failure surfaces later at `createBindGroup` or not at all. Two earlier
 * versions of this measurement read an error flag instead and both lied — one
 * reported every stage a success, the other reported a 2.08 GB ceiling that was
 * a leftover rejection from the round before. So each stage adds its first
 * buffer to its last on the device and every value must come back 2.
 *
 *     npx tsx harness/verify-reclaim.ts            # with reclaim, expected green
 *     npx tsx harness/verify-reclaim.ts --without  # the mutation, expected red
 *
 * `--without` is the half that makes the other half mean anything: it skips the
 * `reclaim()` call and nothing else. If it passes, this script is not measuring
 * what it claims to.
 */
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { createResidentDevice, type ResidentDevice } from "./resident.js";
import { params } from "./wgsl.js";

const without = process.argv.includes("--without");

const ELEMENTWISE = readFileSync(
  fileURLToPath(new URL("../ops/elementwise/wgsl/kernel.wgsl", import.meta.url)), "utf8");

const CHUNK = 64 << 20;
const WG = 256;
/** What `examples/h3-ref2v-web` has to hold, one stage at a time. */
const STAGES: [string, number][] = [
  ["conditioner", 25.78e9],
  ["DiT", 20.66e9],
  ["decoder", 2.43e9],
  ["conditioner again", 25.78e9],
];

const device = await createResidentDevice();
if (!device) {
  console.error("verify-reclaim: no adapter");
  process.exit(2);
}

async function stage(resident: ResidentDevice, label: string, bytes: number): Promise<boolean> {
  const buffers: GPUBuffer[] = [];
  const ones = new Float32Array(CHUNK / 4).fill(1);
  let held = 0;
  const at = performance.now();
  while (held < bytes) {
    const size = Math.min(CHUNK, bytes - held);
    const buffer = resident.createStorageBuffer(size);
    resident.upload(buffer, 0, ones.subarray(0, size / 4));
    buffers.push(buffer);
    held += size;
  }
  const first = buffers[0]!;
  const last = buffers[buffers.length - 1]!;
  const count = Math.min(first.size, last.size) / 4;
  const out = resident.createStorageBuffer(count * 4);
  const uniform = resident.createUniformBuffer(128);
  resident.upload(uniform, 0, new Uint8Array(params([["u32", count], ["u32", 0]])));
  const staging = resident.createStorageBuffer(
    count * 4, GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST);
  const release = (): void => {
    for (const buffer of [...buffers, out, uniform, staging]) buffer.destroy();
  };

  let values: Float32Array;
  try {
    const pipeline = await resident.pipelineFor(ELEMENTWISE);
    const [read] = await resident.batch([{
      kind: "dispatch",
      pipeline,
      bindGroup: await resident.bindGroup(pipeline, [first, last, out, uniform]),
      workgroups: [Math.ceil(count / WG)],
    }], [{ staging, source: out, sourceOffset: 0, length: count, type: "f32" }]);
    values = read as Float32Array;
  } catch (error) {
    console.log(`  ${label}: ${(held / 1e9).toFixed(2)} GB — FAILED, ${String(error).slice(0, 110)}`);
    release();
    return false;
  }
  let wrong = 0;
  for (let i = 0; i < values.length; i += 1) if (values[i] !== 2) wrong += 1;
  console.log(
    `  ${label}: ${(held / 1e9).toFixed(2)} GB in ${((performance.now() - at) / 1000).toFixed(1)} s, ` +
      `${buffers.length} buffers — ` +
      (wrong === 0 ? "all live" : `${wrong} of ${values.length} values wrong (first ${values[0]})`),
  );
  release();
  return wrong === 0;
}

console.log(without
  ? "without reclaim() — this is the mutation, and it is expected to fail"
  : "with reclaim() between the stages");

let ok = true;
for (const [label, bytes] of STAGES) {
  ok = (await stage(device, label, bytes)) && ok;
  if (!without) await device.reclaim();
  if (!ok) break;
}

console.log(ok
  ? "every stage ran on the card the one before it filled"
  : "a stage did not survive the one before it");
// Green means **what was expected happened**, which for `--without` is a
// failure. A mutation that passes is the thing worth exiting nonzero for.
const asExpected = ok === !without;
console.log(asExpected
  ? (without ? "expected: the mutation failed, so this script measures reclaim()" : "expected: reclaim() held")
  : (without ? "UNEXPECTED: it worked without reclaim() — this script is measuring nothing"
    : "UNEXPECTED: reclaim() did not hold"));
device.destroy();
process.exit(asExpected ? 0 : 1);
