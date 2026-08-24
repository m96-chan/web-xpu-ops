/**
 * `scores` and `context` at a real model's shape, timed on their own.
 *
 * Issue #177. The in-forward profile put `context` at 29% of an Anima forward
 * and `scores` at 0.3%, which would make `scores` run at 190 TFLOP/s on a
 * device whose measured ceiling is 50.7 — impossible, so one of the two numbers
 * is wrong and the forward cannot say which.
 *
 * Timing them alone, at the shape the model actually uses, can. Each is
 * dispatched by itself with a timestamp around it, repeated, and reported
 * against `measureRoofline`'s figure for *this* device rather than a spec
 * sheet — WebGPU exposes no clock and no compute-unit count, so there is
 * nothing to compute a theoretical peak from (rule 9).
 *
 * `ops/flash_attention` is timed beside them, because it computes the same
 * function in one dispatch without ever materialising `[B, H, L, S]` — the
 * comparison this repository already had and that `examples/anima` was not
 * using.
 *
 *     npx tsx ops/attention/tools/bench.ts [--L 3952] [--S 3952] [--D 128] [--heads 16]
 */
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { createRunner, params, type Dispatch } from "../../../harness/wgsl.js";
import { measureRoofline } from "../../../harness/roofline.js";

const arg = (name: string, fallback: number): number => {
  const at = process.argv.indexOf(name);
  return at >= 0 ? Number(process.argv[at + 1]) : fallback;
};
const L = arg("--L", 3952);
const S = arg("--S", 3952);
const D = arg("--D", 128);
const heads = arg("--heads", 16);
const repeats = arg("--repeats", 3);

const read = (name: string): string =>
  readFileSync(fileURLToPath(new URL(`../wgsl/${name}.wgsl`, import.meta.url)), "utf8");
const readFlash = (): string =>
  readFileSync(fileURLToPath(new URL("../../flash_attention/wgsl/kernel.wgsl", import.meta.url)), "utf8");

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
console.log(`L=${L} S=${S} D=${D} heads=${heads}`);
if (roof) {
  console.log(`  measured roofline: ${(roof.compute / 1e12).toFixed(1)} TFLOP/s, ${(roof.bandwidth / 1e12).toFixed(2)} TB/s\n`);
} else {
  console.log("  roofline unavailable on this device — shares cannot be reported (rule 9)\n");
}

// Non-trivial values: a buffer of zeros is served without reaching memory on
// this hardware, which `harness/roofline.ts` learned the hard way.
const fill = (n: number): Float32Array => Float32Array.from({ length: n }, (_, i) => Math.sin(i * 0.7) * 0.5);
const q = fill(heads * L * D);
const k = fill(heads * S * D);
const v = fill(heads * S * D);
const probs = fill(heads * L * S);
const bias = new Float32Array(S);

/** Best of `repeats`, so a scheduling hiccup does not become the number. */
async function best(label: string, once: () => Promise<number | null>): Promise<number | null> {
  let bestSeconds: number | null = null;
  for (let i = 0; i < repeats; i += 1) {
    const seconds = await once();
    if (seconds !== null && (bestSeconds === null || seconds < bestSeconds)) bestSeconds = seconds;
  }
  if (bestSeconds === null) console.log(`  ${label.padEnd(9)} the device declined to time it`);
  return bestSeconds;
}

const scoresDispatch: Dispatch = {
  code: read("scores"),
  bindings: [
    { kind: "storage", data: q },
    { kind: "storage", data: k },
    { kind: "storage", data: bias },
    { kind: "out", type: "f32", length: heads * L * S },
    {
      kind: "uniform",
      data: params([
        ["u32", heads], ["u32", L], ["u32", S], ["u32", D],
        ["f32", 1 / Math.sqrt(D)], ["u32", 0], ["i32", 0], ["u32", 1], ["u32", 1], ["u32", 1],
      ]),
    },
  ],
  workgroups: [L, heads, 1],
};

const contextDispatch: Dispatch = {
  code: read("context"),
  bindings: [
    { kind: "storage", data: probs },
    { kind: "storage", data: v },
    { kind: "out", type: "f32", length: heads * L * D },
    { kind: "uniform", data: params([["u32", heads], ["u32", L], ["u32", S], ["u32", D]]) },
  ],
  workgroups: [L, heads, 1],
};

// One dispatch, no `[B, H, L, S]` anywhere. An additive bias of zeros is what
// "no mask" is, matching `ops/attention`'s own convention.
const flashDispatch: Dispatch = {
  code: readFlash(),
  bindings: [
    { kind: "storage", data: q },
    { kind: "storage", data: k },
    { kind: "storage", data: v },
    { kind: "storage", data: new Float32Array(S) },
    { kind: "out", type: "f32", length: heads * L * D },
    {
      kind: "uniform",
      data: params([
        ["u32", heads], ["u32", L], ["u32", S], ["u32", D], ["u32", D],
        ["f32", 1 / Math.sqrt(D)], ["u32", 0], ["i32", 0], ["u32", 1], ["u32", 1], ["u32", 1],
      ]),
    },
  ],
  workgroups: [L, heads, 1],
};

// `time()` reads a timestamp query written around the compute pass. Wall clock
// here would measure buffer creation and the readback round trip — about a
// millisecond, several times a real dispatch (`Runner.time`'s own doc).
const scoresSeconds = await best("scores", () => runner.time(scoresDispatch));
const contextSeconds = await best("context", () => runner.time(contextDispatch));
const flashSeconds = await best("flash", () => runner.time(flashDispatch));

runner.destroy?.();

// Both are the same batched matmul shape: `heads * L * S * D` multiply-adds.
const flops = 2 * heads * L * S * D;
console.log(`  each is ${(flops / 1e12).toFixed(2)} TFLOP of multiply-add\n`);
console.log("  kernel      seconds     TFLOP/s   share of the measured roofline");
for (const [label, seconds] of [
  ["scores", scoresSeconds], ["context", contextSeconds],
  ["split total", scoresSeconds !== null && contextSeconds !== null ? scoresSeconds + contextSeconds : null],
  ["flash", flashSeconds],
] as const) {
  if (seconds === null) continue;
  const achieved = flops / seconds;
  const share = roof ? `${((achieved / roof.compute) * 100).toFixed(1)}%` : "unavailable";
  console.log(
    `  ${label.padEnd(9)} ${seconds.toFixed(4).padStart(9)} ${(achieved / 1e12).toFixed(2).padStart(11)} ${share.padStart(9)}`,
  );
}
if (scoresSeconds !== null && contextSeconds !== null && flashSeconds !== null) {
  const split = scoresSeconds + contextSeconds;
  console.log(`\n  split (scores + context) / flash = ${(split / flashSeconds).toFixed(1)}x`);
  const probsBytes = heads * L * S * 4;
  console.log(`  the split path also materialises ${(probsBytes / 1e9).toFixed(2)} GB of scores; flash materialises none`);
}
