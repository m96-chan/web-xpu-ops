import { spawnSync } from "node:child_process";
import { fileURLToPath } from "node:url";
import { describe, expect, it } from "vitest";
import { quantize } from "../ops/quantize/reference.js";

/**
 * Cross-checks `llm/tools/quant_common.py#quantize_per_row` — the
 * quantization `gen_fixture_q8.py` and `convert_weights.py` both use to bake
 * int8 weights (issue #105) — against this repository's own
 * `ops/quantize/reference.ts#quantize`, on the same input rows.
 *
 * The two are two different implementations of "the same" rounding rule by
 * construction, not by inspection: `quant_common.py` uses `np.floor(x + 0.5)`
 * specifically because `Math.round` disagrees with `np.round`'s
 * banker's-rounding at every `.5` boundary (`Math.round(62.5) === 63`,
 * `np.round(62.5) === 62`) — asserted here rather than trusted, per rule 2
 * ("推測でコードを書かない"). Row 0 of the selftest is chosen so several of
 * its values land exactly on that boundary after scaling.
 *
 * Skips rather than fails when `python3` (or `numpy`) is not on `PATH` — the
 * same posture `harness/suite.ts#useGpu` takes for a missing GPU adapter:
 * this checks an external interpreter's behaviour, not this repository's own
 * code, so its absence is an environment fact, not a broken build.
 */

const scriptPath = fileURLToPath(new URL("./tools/quant_common.py", import.meta.url));

interface SelftestRow {
  input: number[];
  codes: number[];
  scale: number;
}

function runSelftest(): SelftestRow[] | null {
  const result = spawnSync("python3", [scriptPath, "--selftest"], { encoding: "utf8" });
  if (result.error || result.status !== 0) return null;
  try {
    const parsed = JSON.parse(result.stdout) as unknown;
    return Array.isArray(parsed) ? (parsed as SelftestRow[]) : null;
  } catch {
    return null;
  }
}

const rows = runSelftest();

describe("quant_common.py / ops/quantize parity", () => {
  it.skipIf(rows === null)(
    "quant_common.py's codes and scale match ops/quantize/reference.ts#quantize row for row, including the .5 rounding boundary",
    () => {
      for (const { input, codes, scale } of rows!) {
        const { output, scales } = quantize({ input: Float32Array.from(input), N: 1, D: input.length });
        expect(Array.from(output)).toEqual(codes);
        expect(scales[0]).toBeCloseTo(scale, 6);
      }
    },
  );
});
