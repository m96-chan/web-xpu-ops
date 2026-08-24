/**
 * Sweeps flash-attention shapes at the dimensions Anima actually uses.
 *
 * Issue #177. The shipped kernel is 80.7% of a forward and 3.3% of this
 * device's measured roofline. **Every candidate is checked against
 * `ops/flash_attention`'s reference before it is timed** — a wrong kernel is
 * arbitrarily fast and would win the sweep (rule 8).
 *
 *     npx tsx ops/flash_attention/tools/bench.ts [--self|--cross]
 */
import { createRunner, params, type Dispatch } from "../../../harness/wgsl.js";
import { measureRoofline } from "../../../harness/roofline.js";
import { flashAttention } from "../reference.js";
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { flashStorageBytes, rejectReason, tiledFlash, type FlashShape } from "./tiled.wgsl.js";
import { FLASH_TILE } from "../index.js";
import { describeSweep, sweep } from "../../../harness/sweep.js";

const D = 128;
const SHAPES = [
  { name: "self", L: 3952, S: 3952, heads: 8 },
  { name: "cross", L: 3952, S: 512, heads: 16 },
].filter((s) => process.argv.includes(`--${s.name}`) || (!process.argv.includes("--self") && !process.argv.includes("--cross")));

function candidates(): FlashShape[] {
  const out: FlashShape[] = [];
  for (const bq of [1, 2, 4, 8, 16, 32]) {
    for (const tileS of [8, 16, 32, 64]) {
      for (const threads of [128, 256, 512]) {
        const shape = { bq, tileS, threads };
        if (rejectReason(shape, D) === null) out.push(shape);
      }
    }
  }
  return out;
}

const label = (s: FlashShape): string => `BQ=${s.bq}, TILE_S=${s.tileS}, ${s.threads} threads`;


const runner = await createRunner();
if (!runner) { console.error("bench: no adapter"); process.exit(2); }
const roof = await measureRoofline(runner);
if (roof) console.log(`measured roofline: ${(roof.compute / 1e12).toFixed(1)} TFLOP/s`);

const shipped = readFileSync(fileURLToPath(new URL("../wgsl/kernel.wgsl", import.meta.url)), "utf8");
const uniforms = (H: number, L: number, S: number): ArrayBuffer =>
  params([
    ["u32", H], ["u32", L], ["u32", S], ["u32", D], ["u32", D],
    ["f32", 1 / Math.sqrt(D)], ["u32", 0], ["i32", 0], ["u32", 1], ["u32", 1], ["u32", 1],
  ]);

function dispatchFor(
  code: string, q: Float32Array, k: Float32Array, v: Float32Array, H: number, L: number, S: number, bq = 1,
): Dispatch {
  return {
    code,
    bindings: [
      { kind: "storage", data: q },
      { kind: "storage", data: k },
      { kind: "storage", data: v },
      { kind: "storage", data: new Float32Array(S) },
      { kind: "out", type: "f32", length: H * L * D },
      { kind: "uniform", data: uniforms(H, L, S) },
    ],
    workgroups: [Math.ceil(L / bq), H, 1],
  };
}

// Ragged and small: 13 queries against 37 keys divides by no tile.
const CL = 13, CS = 37, CH = 2;
const fill = (n: number, f: number): Float32Array => Float32Array.from({ length: n }, (_, i) => Math.sin(i * f) * 0.5);
const cq = fill(CH * CL * D, 0.7), ck = fill(CH * CS * D, 0.3), cv = fill(CH * CS * D, 0.11);
const want = flashAttention({
  q: cq, k: ck, v: cv, B: 1, H: CH, L: CL, S: CS, D, Dv: D, causal: false,
}).output;

const all = candidates();
console.log(`\n${all.length} shapes fit this device's limits`);
console.log("checking each against ops/flash_attention's reference on a ragged 13x37 …");
const correct: FlashShape[] = [];
for (const shape of all) {
  const [got] = await runner.run(dispatchFor(tiledFlash(shape, D), cq, ck, cv, CH, CL, CS, shape.bq));
  const out = got as Float32Array;
  let worst = 0;
  for (let i = 0; i < want.length; i += 1) worst = Math.max(worst, Math.abs(out[i]! - want[i]!));
  if (worst < 1e-4) correct.push(shape);
  else console.log(`  WRONG: ${label(shape)} (worst ${worst.toExponential(2)})`);
}
console.log(`  ${correct.length}/${all.length} compute attention\n`);

for (const { name, L, S, heads } of SHAPES) {
  const q = fill(heads * L * D, 0.001), k = fill(heads * S * D, 0.002), v = fill(heads * S * D, 0.003);
  const flops = 2 * 2 * heads * L * S * D;
  console.log(`${name}  L=${L} S=${S} heads=${heads}  (${(flops / 1e9).toFixed(1)} GFLOP)`);

  // `FLASH_TILE.bq`, not 1. The shipped kernel takes BQ queries per workgroup,
  // and dispatching one per query gives it sixteen times the work — which is
  // what made a baseline read 83.93 ms against its real 8.43, and would have
  // made every candidate look like a triumph.
  const reference = dispatchFor(shipped, q, k, v, heads, L, S, FLASH_TILE.bq);
  const entries = correct.map((shape) => ({
    candidate: shape,
    label: label(shape),
    dispatch: dispatchFor(tiledFlash(shape, D), q, k, v, heads, L, S, shape.bq),
  }));

  const report = await sweep(runner, reference, entries);
  if (!report) {
    console.log("  the device declined to time it\n");
    continue;
  }
  for (const line of describeSweep(report, flops, roof?.compute ?? null)) console.log(line);
  console.log();
}

runner.destroy();
