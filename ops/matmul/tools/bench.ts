/**
 * Sweeps `matmul` tile shapes at the dimensions Anima actually uses.
 *
 * Issue #177 / #134. The shipped kernel says of itself that `TILE = 16 is not a
 * measured optimum — nothing here is tuned yet`, and it reaches 1.5-1.8% of
 * this device's measured roofline. This is the measurement that was missing.
 *
 * **Every candidate is checked against `ops/matmul`'s reference before it is
 * timed.** A wrong kernel is arbitrarily fast and would win the sweep, which is
 * rule 8's whole point: correctness first, and a tuning tool that can report a
 * broken shape as the best one is worse than no tool.
 *
 * The correctness check runs at a small shape and the timing at the real one —
 * a full reference matmul at 3952 x 2048 x 8192 is 132 GFLOP of JavaScript.
 *
 *     npx tsx ops/matmul/tools/bench.ts [--shape qkv|mlp-up|mlp-down|all] [--quick]
 */
import { createRunner, params, type Dispatch } from "../../../harness/wgsl.js";
import { measureRoofline } from "../../../harness/roofline.js";
import { matmul } from "../reference.js";
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { gridFor, rejectReason, storageBytes, tiledMatmul, type TileShape } from "./tiled.wgsl.js";

const arg = (name: string): string | undefined => {
  const at = process.argv.indexOf(name);
  return at >= 0 ? process.argv[at + 1] : undefined;
};
const quick = process.argv.includes("--quick");
const want = arg("--shape") ?? "all";

/**
 * The three matmuls a DiT block does, at 832x1216.
 *
 * `M` is the token count and is 3,952 — not a multiple of any tile, which is
 * why the kernel guards every load and store.
 */
const SHAPES: { name: string; M: number; K: number; N: number; per: number }[] = [
  { name: "qkv/out", M: 3952, K: 2048, N: 2048, per: 8 },
  { name: "mlp-up", M: 3952, K: 2048, N: 8192, per: 1 },
  { name: "mlp-down", M: 3952, K: 8192, N: 2048, per: 1 },
];

/**
 * Candidates, filtered by what this device can dispatch.
 *
 * Deliberately a coarse net rather than a clever one: the question is which
 * order of magnitude of thread tile wins, and a sweep that assumes the answer
 * to narrow itself is the same mistake as assuming the answer outright.
 */
function candidates(): TileShape[] {
  const out: TileShape[] = [];
  for (const wgM of [8, 16, 32]) {
    for (const wgN of [8, 16, 32]) {
      for (const tileM of [1, 2, 4, 8]) {
        for (const tileN of [1, 2, 4, 8]) {
          for (const tileK of [8, 16, 32]) {
            const shape = { wgM, wgN, tileM, tileN, tileK };
            if (rejectReason(shape) === null) out.push(shape);
          }
        }
      }
    }
  }
  return out;
}

const label = (s: TileShape): string =>
  `${s.wgM}x${s.wgN} threads, ${s.tileM}x${s.tileN} each, K${s.tileK}`;

/**
 * Brings the clocks up before anything is timed.
 *
 * An idle RTX 5090 sits at 195 MHz against 2850 under load, and a sweep that
 * starts cold measures the ramp rather than the kernel: an unchanged shipped
 * kernel read 9.56 ms warm and 83.69 ms cold in this very tool, which is a
 * factor of nine attributable to nothing in the code.
 *
 * Untimed on purpose. The point is not to measure this dispatch but to be past
 * the ramp before measuring the next one.
 */
async function warmUp(runner: { run: (d: Dispatch) => Promise<unknown> }, dispatch: Dispatch, rounds = 8): Promise<void> {
  for (let i = 0; i < rounds; i += 1) await runner.run(dispatch);
}

const runner = await createRunner();
if (!runner) {
  console.error("bench: no WebGPU adapter available.");
  process.exit(2);
}
const roof = await measureRoofline(runner);
if (roof) {
  console.log(`measured roofline: ${(roof.compute / 1e12).toFixed(1)} TFLOP/s, ${(roof.bandwidth / 1e12).toFixed(2)} TB/s`);
} else {
  console.log("roofline unavailable on this device — shares cannot be reported (rule 9)");
}

const shipped = readFileSync(fileURLToPath(new URL("../wgsl/kernel.wgsl", import.meta.url)), "utf8");

/** The shipped kernel's own dispatch, for the baseline row. */
function shippedDispatch(a: Float32Array, b: Float32Array, M: number, K: number, N: number): Dispatch {
  const TILE = 16;
  return {
    code: shipped,
    bindings: [
      { kind: "storage", data: a },
      { kind: "storage", data: b },
      { kind: "out", type: "f32", length: M * N },
      { kind: "uniform", data: params([["u32", M], ["u32", N], ["u32", K]]) },
    ],
    workgroups: [Math.ceil(N / TILE), Math.ceil(M / TILE), 1],
  };
}

function tiledDispatch(shape: TileShape, a: Float32Array, b: Float32Array, M: number, K: number, N: number): Dispatch {
  return {
    code: tiledMatmul(shape),
    bindings: [
      { kind: "storage", data: a },
      { kind: "storage", data: b },
      { kind: "out", type: "f32", length: M * N },
      { kind: "uniform", data: params([["u32", M], ["u32", N], ["u32", K]]) },
    ],
    workgroups: gridFor(shape, M, N),
  };
}

/**
 * Does this shape compute a matmul at all?
 *
 * A small, ragged case: 37 x 43 x 29 divides by nothing, so a kernel that only
 * works when the tile fits exactly fails here rather than in the timing.
 */
const CM = 37, CK = 43, CN = 29;
const ca = Float32Array.from({ length: CM * CK }, (_, i) => Math.sin(i * 0.7));
const cb = Float32Array.from({ length: CK * CN }, (_, i) => Math.cos(i * 0.3));
const expected = matmul({ a: ca, b: cb, M: CM, N: CN, K: CK });

async function isCorrect(shape: TileShape): Promise<boolean> {
  const [got] = await runner!.run(tiledDispatch(shape, ca, cb, CM, CK, CN));
  const out = got as Float32Array;
  for (let i = 0; i < expected.length; i += 1) {
    if (Math.abs(out[i]! - expected[i]!) > 1e-3 * Math.max(1, Math.abs(expected[i]!))) return false;
  }
  return true;
}

const all = candidates();
console.log(
  `\n${all.length} shapes fit this device's limits — 49152 B of workgroup storage, ` +
    "1024 invocations, f32 staging (this device has no shader-f16)",
);

console.log("checking each against ops/matmul's reference on a ragged 37x43x29 …");
const correct: TileShape[] = [];
for (const shape of all) {
  if (await isCorrect(shape)) correct.push(shape);
  else console.log(`  WRONG: ${label(shape)}`);
}
console.log(`  ${correct.length}/${all.length} compute a matmul\n`);

const shapes = want === "all" ? SHAPES : SHAPES.filter((s) => s.name === want);
const pool = quick ? correct.filter((s) => s.tileM * s.tileN >= 4) : correct;

for (const { name, M, K, N, per } of shapes) {
  const a = Float32Array.from({ length: M * K }, (_, i) => Math.sin(i * 0.001));
  const b = Float32Array.from({ length: K * N }, (_, i) => Math.cos(i * 0.002));
  const flops = 2 * M * K * N;
  console.log(`${name}  M=${M} K=${K} N=${N}  (${per} per block, ${(flops / 1e9).toFixed(1)} GFLOP each)`);

  const baseline = await runner.time(shippedDispatch(a, b, M, K, N));
  if (baseline !== null) {
    const share = roof ? ` (${((flops / baseline / roof.compute) * 100).toFixed(1)}% of roofline)` : "";
    console.log(`  shipped TILE=16:  ${(baseline * 1000).toFixed(2)} ms, ${(flops / baseline / 1e12).toFixed(2)} TFLOP/s${share}`);
  }

  const scored: { shape: TileShape; seconds: number }[] = [];
  for (const shape of pool) {
    const seconds = await runner.time(tiledDispatch(shape, a, b, M, K, N));
    if (seconds !== null) scored.push({ shape, seconds });
  }
  scored.sort((x, y) => x.seconds - y.seconds);

  console.log("  best 5:");
  for (const { shape, seconds } of scored.slice(0, 5)) {
    const achieved = flops / seconds;
    const share = roof ? `${((achieved / roof.compute) * 100).toFixed(1)}%` : "n/a";
    const speedup = baseline !== null ? `${(baseline / seconds).toFixed(1)}x` : "n/a";
    console.log(
      `    ${label(shape).padEnd(34)} ${(seconds * 1000).toFixed(2).padStart(7)} ms  ` +
        `${(achieved / 1e12).toFixed(2).padStart(6)} TFLOP/s  ${share.padStart(6)}  ${speedup.padStart(5)}  ` +
        `${(storageBytes(shape) / 1024).toFixed(1)} KB shmem`,
    );
  }
  console.log();
}

runner.destroy();
