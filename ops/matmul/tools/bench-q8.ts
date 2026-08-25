/**
 * Sweeps `matmulQ8` tile shapes at the dimensions Anima actually uses.
 *
 * Issue #177. The dense kernel went from 1.4-3.0% of this device's measured
 * roofline to 70-72% by holding a thread tile in registers; `matmulQ8` is at
 * **15.3%** and is 36.8% of an Anima forward. It starts from a better place
 * than the dense one did, so the headroom is smaller and the point of measuring
 * is to find out how much smaller rather than to assume.
 *
 * **Every candidate is checked against `ops/matmul`'s reference before it is
 * timed** — a wrong kernel is arbitrarily fast and would win the sweep (rule 8).
 *
 *     npx tsx ops/matmul/tools/bench-q8.ts [--shape qkv|mlp-up|mlp-down] [--quick]
 */
import { createRunner, params, type Dispatch } from "../../../harness/wgsl.js";
import { measureRoofline } from "../../../harness/roofline.js";
import { matmulQ8 } from "../reference.js";
import { packQ8 } from "../../matvec/index.js";
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { rejectReason, type TileShape } from "./tiled.wgsl.js";
import { q8StorageBytes, tiledMatmulQ8 } from "./tiled-q8.wgsl.js";

const arg = (name: string): string | undefined => {
  const at = process.argv.indexOf(name);
  return at >= 0 ? process.argv[at + 1] : undefined;
};
const quick = process.argv.includes("--quick");
const want = arg("--shape") ?? "all";

/** N is the token count, M the output channels — `q8.wgsl`'s own naming. */
const SHAPES: { name: string; N: number; K: number; M: number }[] = [
  { name: "qkv", N: 3952, K: 2048, M: 2048 },
  { name: "mlp-up", N: 3952, K: 2048, M: 8192 },
  { name: "mlp-down", N: 3952, K: 8192, M: 2048 },
];

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

const label = (s: TileShape): string => `${s.wgM}x${s.wgN} threads, ${s.tileM}x${s.tileN} each, K${s.tileK}`;

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
  console.error("bench-q8: no WebGPU adapter available.");
  process.exit(2);
}
const roof = await measureRoofline(runner);
if (roof) console.log(`measured roofline: ${(roof.compute / 1e12).toFixed(1)} TFLOP/s`);
else console.log("roofline unavailable — shares cannot be reported (rule 9)");

const shipped = readFileSync(fileURLToPath(new URL("../wgsl/q8.wgsl", import.meta.url)), "utf8");

/** Quantize a weight the way `convert_dit.py` does, per row, absmax to [-127, 127]. */
function quantize(dense: Float32Array, M: number, K: number): { weight: Uint32Array; scale: Float32Array } {
  const codes = new Int32Array(M * K);
  const scale = new Float32Array(M);
  for (let r = 0; r < M; r += 1) {
    let absmax = 0;
    for (let c = 0; c < K; c += 1) absmax = Math.max(absmax, Math.abs(dense[r * K + c]!));
    scale[r] = absmax === 0 ? 1 : absmax / 127;
    const inverse = absmax === 0 ? 0 : 127 / absmax;
    for (let c = 0; c < K; c += 1) {
      const scaled = dense[r * K + c]! * inverse;
      const floored = Math.floor(scaled);
      codes[r * K + c] = Math.max(-127, Math.min(127, floored + (scaled - floored >= 0.5 ? 1 : 0)));
    }
  }
  return { weight: packQ8({ codes, N: M, K }), scale };
}

function shippedDispatch(a: Float32Array, w: Uint32Array, s: Float32Array, N: number, K: number, M: number): Dispatch {
  const TILE = 16;
  return {
    code: shipped,
    bindings: [
      { kind: "storage", data: a },
      { kind: "storage", data: w },
      { kind: "storage", data: s },
      { kind: "out", type: "f32", length: N * M },
      { kind: "uniform", data: params([["u32", N], ["u32", M], ["u32", K]]) },
    ],
    workgroups: [Math.ceil(M / TILE), Math.ceil(N / TILE), 1],
  };
}

function tiledDispatch(
  shape: TileShape, a: Float32Array, w: Uint32Array, s: Float32Array, N: number, K: number, M: number,
): Dispatch {
  return {
    code: tiledMatmulQ8(shape),
    bindings: [
      { kind: "storage", data: a },
      { kind: "storage", data: w },
      { kind: "storage", data: s },
      { kind: "out", type: "f32", length: N * M },
      { kind: "uniform", data: params([["u32", N], ["u32", M], ["u32", K]]) },
    ],
    workgroups: [Math.ceil(M / (shape.wgN * shape.tileN)), Math.ceil(N / (shape.wgM * shape.tileM)), 1],
  };
}

// Ragged on purpose: 37 x 43 x 29 divides by no tile, and K = 43 is not a
// multiple of 4 either, so the packing's own tail is exercised too.
const CN = 37, CK = 43, CM = 29;
const ca = Float32Array.from({ length: CN * CK }, (_, i) => Math.sin(i * 0.7));
const cdense = Float32Array.from({ length: CM * CK }, (_, i) => Math.cos(i * 0.3));
const { weight: cw, scale: cs } = quantize(cdense, CM, CK);
const expected = matmulQ8({ a: ca, weight: cw, scale: cs, N: CN, M: CM, K: CK });

async function isCorrect(shape: TileShape): Promise<boolean> {
  const [got] = await runner!.run(tiledDispatch(shape, ca, cw, cs, CN, CK, CM));
  const out = got as Float32Array;
  for (let i = 0; i < expected.length; i += 1) {
    if (Math.abs(out[i]! - expected[i]!) > 1e-3 * Math.max(1, Math.abs(expected[i]!))) return false;
  }
  return true;
}

const all = candidates();
console.log(`\n${all.length} shapes fit this device's limits`);
console.log("checking each against ops/matmul's reference on a ragged 37x43x29 …");
const correct: TileShape[] = [];
for (const shape of all) {
  if (await isCorrect(shape)) correct.push(shape);
  else console.log(`  WRONG: ${label(shape)}`);
}
console.log(`  ${correct.length}/${all.length} compute a matmul\n`);

const pool = quick ? correct.filter((s) => s.tileM * s.tileN >= 4) : correct;
for (const { name, N, K, M } of SHAPES.filter((s) => want === "all" || s.name === want)) {
  const a = Float32Array.from({ length: N * K }, (_, i) => Math.sin(i * 0.001));
  const dense = Float32Array.from({ length: M * K }, (_, i) => Math.cos(i * 0.002));
  const { weight, scale } = quantize(dense, M, K);
  const flops = 2 * N * K * M;
  console.log(`${name}  N=${N} K=${K} M=${M}  (${(flops / 1e9).toFixed(1)} GFLOP)`);

  const baseline = await runner.time(shippedDispatch(a, weight, scale, N, K, M));
  if (baseline !== null) {
    const share = roof ? ` (${((flops / baseline / roof.compute) * 100).toFixed(1)}% of roofline)` : "";
    console.log(`  shipped TILE=16:  ${(baseline * 1000).toFixed(2)} ms, ${(flops / baseline / 1e12).toFixed(2)} TFLOP/s${share}`);
  }

  const scored: { shape: TileShape; seconds: number }[] = [];
  for (const shape of pool) {
    const seconds = await runner.time(tiledDispatch(shape, a, weight, scale, N, K, M));
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
        `${(q8StorageBytes(shape) / 1024).toFixed(1)} KB shmem`,
    );
  }
  console.log();
}
runner.destroy();
