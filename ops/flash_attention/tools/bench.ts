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
import { rejectReason, type FlashShape, type Generation } from "./shape.js";
import { fa2Flash } from "./fa2.wgsl.js";
import { fa3Flash } from "./fa3.wgsl.js";
import { FLASH_GENERATION, flashGrid } from "../index.js";
import { describeSweep, sweep } from "../../../harness/sweep.js";

const arg = (name: string): string | undefined => {
  const at = process.argv.indexOf(name);
  return at >= 0 ? process.argv[at + 1] : undefined;
};

const D = 128;
const SHAPES = [
  { name: "self", L: 3952, S: 3952, heads: 8 },
  { name: "cross", L: 3952, S: 512, heads: 16 },
].filter((s) => process.argv.includes(`--${s.name}`) || (!process.argv.includes("--self") && !process.argv.includes("--cross")));

/** A candidate is a generation and a shape — the two are swept together. */
interface Candidate {
  generation: Generation;
  shape: FlashShape;
}

/** The WGSL a candidate compiles to. Each generation is its own kernel. */
const codeFor = ({ generation, shape }: Candidate): string =>
  generation === "fa3" ? fa3Flash(shape, D) : fa2Flash(shape, D);

/**
 * Every generation crossed with every shape this device can dispatch.
 *
 * `prefetch` only changes fa3 — fa2 does not overlap anything — so pairing it
 * with fa2 would time the same program twice and report the pair's noise as a
 * difference between them.
 */
/**
 * `--quick` restricts the grid to the neighbourhood of the shipped shape.
 *
 * The full grid is 156 pairs and every one is checked against the reference
 * before it is timed, which is the right default and too slow to iterate on.
 * Narrowing is a claim that the answer is nearby — true only just after a full
 * sweep said so, which is why it is a flag and not the default.
 */
const quick = process.argv.includes("--quick");

function candidates(): Candidate[] {
  const out: Candidate[] = [];
  for (const generation of ["fa2", "fa3"] as const) {
    const prefetches = generation === "fa3" ? (["direct", "registers"] as const) : (["direct"] as const);
    for (const bq of quick ? [16, 32] : [1, 2, 4, 8, 16, 32, 64, 128]) {
      for (const tileS of quick ? [8] : [4, 8, 16, 32, 64]) {
        for (const threads of quick ? [128, 256] : [128, 256, 512, 1024]) {
          for (const accumulate of quick ? (["key"] as const) : (["row", "key"] as const)) {
            for (const prefetch of prefetches) {
              for (const scoreReads of ["scalar", "vec4"] as const) {
                for (const padRows of [false, true]) {
                  const shape = { bq, tileS, threads, accumulate, prefetch, scoreReads, padRows };
                  if (rejectReason(shape, D, generation) === null) out.push({ generation, shape });
                }
              }
            }
          }
        }
      }
    }
  }
  return out;
}

const label = ({ generation, shape: s }: Candidate): string =>
  `${generation.toUpperCase()} BQ=${s.bq} TILE_S=${s.tileS} ${s.threads}t ${s.accumulate}-outer` +
  (generation === "fa3" ? ` ${s.prefetch}` : "") + ` ${s.scoreReads}${s.padRows ? "+pad" : ""}`;


const runner = await createRunner();
if (!runner) { console.error("bench: no adapter"); process.exit(2); }
const roof = await measureRoofline(runner);
if (roof) console.log(`measured roofline: ${(roof.compute / 1e12).toFixed(1)} TFLOP/s`);

const shipped = readFileSync(fileURLToPath(new URL(`../wgsl/${FLASH_GENERATION}.wgsl`, import.meta.url)), "utf8");
const uniforms = (H: number, L: number, S: number): ArrayBuffer =>
  params([
    ["u32", H], ["u32", L], ["u32", S], ["u32", D], ["u32", D],
    ["f32", 1 / Math.sqrt(D)], ["u32", 0], ["i32", 0], ["u32", 1], ["u32", 1], ["u32", 1],
  ]);

function dispatchFor(
  code: string, q: Float32Array, k: Float32Array, v: Float32Array, H: number, L: number, S: number,
  grid?: [number, number, number], bq = 1,
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
    workgroups: grid ?? [Math.ceil(L / bq), H, 1],
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
console.log(`\n${all.length} generation/shape pairs fit this device's limits`);
console.log("checking each against ops/flash_attention's reference on a ragged 13x37 …");
const correct: Candidate[] = [];
for (const candidate of all) {
  const [got] = await runner.run(dispatchFor(codeFor(candidate), cq, ck, cv, CH, CL, CS, undefined, candidate.shape.bq));
  const out = got as Float32Array;
  let worst = 0;
  for (let i = 0; i < want.length; i += 1) worst = Math.max(worst, Math.abs(out[i]! - want[i]!));
  if (worst < 1e-4) correct.push(candidate);
  else console.log(`  WRONG: ${label(candidate)} (worst ${worst.toExponential(2)})`);
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
  // The shipped kernel's grid comes from the op, not from a tile restated here.
  const reference = dispatchFor(shipped, q, k, v, heads, L, S, flashGrid(L, heads, 1));
  const entries = correct.map((candidate) => ({
    candidate,
    label: label(candidate),
    dispatch: dispatchFor(codeFor(candidate), q, k, v, heads, L, S, undefined, candidate.shape.bq),
  }));

  const report = await sweep(runner, reference, entries);
  if (!report) {
    console.log("  the device declined to time it\n");
    continue;
  }
  for (const line of describeSweep(report, flops, roof?.compute ?? null, Number(arg("--top") ?? 6))) console.log(line);
  console.log();
}

runner.destroy();
