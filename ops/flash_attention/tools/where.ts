/**
 * Which part of the flash kernel the time is actually in.
 *
 * Issue #177. Three generations of FlashAttention measured within a few percent
 * of each other here, and one shape moving 4x more data than another landed on
 * the same 240 GB/s. Both facts say the limit is not the schedule and not the
 * memory system — it is something inside the innermost loops that every
 * generation shares. This tool says which one, by **deleting** each in turn and
 * timing what is left.
 *
 * **Every variant but `full` and `D fixed at 128` computes a wrong answer.**
 * That is the point: a lower bound on the time is what remains when a stage
 * cannot contribute, and there is no way to get that while keeping the stage.
 * Nothing here is a candidate to ship, which is why it is a separate tool from
 * `bench.ts` — `bench.ts` refuses to time anything that disagrees with the
 * reference, and it should keep refusing.
 *
 * Each edit is applied by string replacement and **checked to have landed**
 * (rule 1): a mutation that silently misses reports the unmodified kernel as
 * though the stage were free, which is the most expensive kind of wrong answer
 * a tool like this can give.
 *
 *     npx tsx ops/flash_attention/tools/where.ts [--self|--cross]
 */
import { createHash } from "node:crypto";
import { createRunner, params, type Dispatch } from "../../../harness/wgsl.js";
import { measureRoofline } from "../../../harness/roofline.js";
import { describeSweep, sweep } from "../../../harness/sweep.js";
import { shippedKernels } from "./generate.js";
import { FLASH_TILE } from "../index.js";

const D = 128;
const SHAPES = [
  { name: "self", L: 3952, S: 3952, heads: 8 },
  { name: "cross", L: 3952, S: 512, heads: 16 },
].filter(
  (s) =>
    process.argv.includes(`--${s.name}`) ||
    (!process.argv.includes("--self") && !process.argv.includes("--cross")),
);

const base = shippedKernels().fa2;
const digest = (s: string): string => createHash("md5").update(s).digest("hex").slice(0, 8);

/** A replacement that must land, or the run stops. */
function edit(code: string, from: string, to: string, why: string): string {
  if (!code.includes(from)) {
    console.error(`the edit for "${why}" did not match — the kernel has changed shape under this tool`);
    process.exit(3);
  }
  return code.replace(from, to);
}

const SCORE_LOOP = `        for (var d = 0u; d < params.D; d = d + 1u) {
          dot_ = fma(sq[r * params.D + d], sk[(0u) * K_STRIDE + slot * params.D + d], dot_);
        }`;
const ACC_LOOP = `        for (var r = 0u; r < BQ; r = r + 1u) {
          acc[r][p] = fma(ss[r * TILE_S + slot], value, acc[r][p]);
        }`;

const VARIANTS: { label: string; code: string; correct: boolean }[] = [
  { label: "full (shipped)", code: base, correct: true },

  // The one honest candidate here. `params.D` is a uniform, so the score loop's
  // bound is a runtime value and no compiler can unroll or vectorise it. The
  // kernel is generated, so the bound it was generated for is known.
  {
    label: "D fixed at 128 (compile-time bound)",
    code: edit(base, "for (var d = 0u; d < params.D; d = d + 1u) {", "for (var d = 0u; d < 128u; d = d + 1u) {", "const D"),
    correct: true,
  },

  // Diagnostics. Each keeps a dependency on the staged tile so the staging
  // cannot be optimised away with the loop.
  {
    label: "WRONG: score dot deleted",
    code: edit(base, SCORE_LOOP, "        dot_ = sq[r * params.D] * sk[slot * params.D];", "no score dot"),
    correct: false,
  },
  {
    label: "WRONG: accumulate over rows deleted",
    code: edit(base, ACC_LOOP, "        acc[0][p] = acc[0][p] + value * ss[slot];", "no accumulate"),
    correct: false,
  },
  {
    label: "WRONG: both inner loops deleted",
    code: edit(
      edit(base, SCORE_LOOP, "        dot_ = sq[r * params.D] * sk[slot * params.D];", "no score dot"),
      ACC_LOOP,
      "        acc[0][p] = acc[0][p] + value * ss[slot];",
      "no accumulate",
    ),
    correct: false,
  },
];

// Every edit must have produced a distinct program. Two variants with the same
// digest means one of them did nothing.
const seen = new Map<string, string>();
for (const v of VARIANTS) {
  const d = digest(v.code);
  const clash = seen.get(d);
  if (clash) {
    console.error(`"${v.label}" and "${clash}" are the same program (${d}) — an edit did not land`);
    process.exit(3);
  }
  seen.set(d, v.label);
}

const runner = await createRunner();
if (!runner) {
  console.error("where: no adapter");
  process.exit(2);
}
const roof = await measureRoofline(runner);

const uniforms = (H: number, L: number, S: number): ArrayBuffer =>
  params([
    ["u32", H], ["u32", L], ["u32", S], ["u32", D], ["u32", D],
    ["f32", 1 / Math.sqrt(D)], ["u32", 0], ["i32", 0], ["u32", 1], ["u32", 1], ["u32", 1],
  ]);

const fill = (n: number, f: number): Float32Array =>
  Float32Array.from({ length: n }, (_, i) => Math.sin(i * f) * 0.5);

function dispatchFor(code: string, q: Float32Array, k: Float32Array, v: Float32Array, H: number, L: number, S: number): Dispatch {
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
    workgroups: [Math.ceil(L / FLASH_TILE.bq), H, 1],
  };
}

console.log(`\n${VARIANTS.length} variants, each a distinct program:`);
for (const v of VARIANTS) console.log(`  ${digest(v.code)}  ${v.label}`);
console.log("\nthe WRONG ones are lower bounds, not candidates — they do not compute attention\n");

for (const { name, L, S, heads } of SHAPES) {
  const q = fill(heads * L * D, 0.001), k = fill(heads * S * D, 0.002), v = fill(heads * S * D, 0.003);
  const flops = 2 * 2 * heads * L * S * D;
  console.log(`${name}  L=${L} S=${S} heads=${heads}`);
  const report = await sweep(
    runner,
    dispatchFor(base, q, k, v, heads, L, S),
    VARIANTS.slice(1).map((v2) => ({
      candidate: v2,
      label: v2.label,
      dispatch: dispatchFor(v2.code, q, k, v, heads, L, S),
    })),
  );
  if (!report) {
    console.log("  the device declined to time it\n");
    continue;
  }
  for (const line of describeSweep(report, flops, roof?.compute ?? null, VARIANTS.length)) console.log(line);
  console.log();
}

runner.destroy();
