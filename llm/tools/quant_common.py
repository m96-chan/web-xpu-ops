"""Per-row absmax int8 quantization, shared by `gen_fixture_q8.py` (the tiny
int8 fixture) and `convert_weights.py` (the real-model converter) — issue
#105's requirement that both bake the **same** quantization error, so a
single implementation is imported by both rather than reimplemented twice
(rule 7: do not pick a convention twice, silently or otherwise).

## Matching `ops/quantize/reference.ts#quantize` bit-for-bit

- Symmetric range `[-127, 127]`, `scale = absmax/127` (or `1` for an all-zero
  row, `quantize`'s own guard against a division by zero).
- **Rounding**: `quantize()` uses JS's `Math.round`, which rounds ties
  **toward +Infinity** (`Math.round(62.5) === 63`, `Math.round(-62.5) === -62`)
  — not "round half away from zero" and not "round half to even". numpy's
  `np.round` is banker's rounding and disagrees at every `.5` boundary
  (`np.round(62.5)` is `62`). The tempting `np.floor(x + 0.5)` is *also*
  wrong, in a narrower band: for `x` just below a half-integer (e.g.
  `0.49999999999999994`, the largest double below 0.5) the addition `x + 0.5`
  itself rounds up to `1.0` before `floor` sees it, giving `1` where
  `Math.round` gives `0`. So this module computes
  `floor(x) + (x - floor(x) >= 0.5)`, which introduces no second rounding:
  `x - floor(x)` is exact in IEEE double for every finite `x` whose fraction
  can decide the comparison. Verified, not assumed (rule 2) — see
  `quantize-parity.test.ts`, which spawns this module's `--selftest` and
  diffs its output against `ops/quantize/reference.ts#quantize` directly,
  including a row constructed inside that double-rounding band.
- All arithmetic promoted to float64 before rounding, mirroring the fact that
  JS numbers (and therefore every intermediate in `quantize()`) are float64
  regardless of the `Float32Array` the values are read from/written to; only
  the final `scale` narrows to float32 (`Float32Array` storage), matching
  this module's `.astype(np.float32)` on `scale` alone.
"""
from __future__ import annotations

import json
import sys

import numpy as np


def quantize_per_row(w: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """`w`: `[N, K]` float. Returns `(codes int8 [N, K], scale f32 [N])`."""
    w64 = w.astype(np.float64)
    absmax = np.max(np.abs(w64), axis=1)
    scale64 = np.where(absmax == 0, 1.0, absmax / 127.0)
    # errstate suppresses the (harmless) divide-by-zero warning from computing
    # 127.0/absmax at rows np.where discards anyway — the all-zero-row branch
    # is never selected, only evaluated, same as `quantize()`'s own ternary
    # guard against dividing by an absmax of 0.
    with np.errstate(divide="ignore", invalid="ignore"):
        inverse64 = np.where(absmax == 0, 0.0, 127.0 / absmax)
    scaled = w64 * inverse64[:, None]
    # Math.round semantics without the double rounding of floor(x + 0.5):
    # see the module doc's Rounding note.
    floored = np.floor(scaled)
    codes = np.clip(floored + (scaled - floored >= 0.5), -127, 127).astype(np.int8)
    scale = scale64.astype(np.float32)
    return codes, scale


def dequantize_per_row(codes: np.ndarray, scale: np.ndarray) -> np.ndarray:
    """Inverse of `quantize_per_row`'s scale application: `codes * scale[:, None]`, f32."""
    return (codes.astype(np.float64) * scale.astype(np.float64)[:, None]).astype(np.float32)


def _selftest() -> None:
    """Prints quantize_per_row's output for a few hand-picked rows as JSON, for
    `llm/quantize-parity.test.ts` to diff against `ops/quantize/reference.ts#quantize`
    on the identical input. Row 0 is chosen so every value's `col * inverse`
    lands exactly on a `.5` boundary (`absmax=127` makes `scale=inverse=1`,
    so the row's own values sit at the boundary directly) — the case that
    tells `floor(x+0.5)` apart from `np.round`'s banker's rounding.
    """
    rows = [
        [127.0, 62.5, -62.5, 0.0, -0.5, 0.5, 1.5, -1.5],
        [3.0, -1.0, 2.5, -2.5, 0.25, -0.75],
        [0.0, 0.0, 0.0],  # all-zero row: scale must be 1, not a division by zero
        [-5.0, 5.0, 2.0, -2.0, 1.0],
        # Double-rounding band: both values are exactly float32-representable,
        # and 0.005327165126800537 * (127 / 1.3530999422073364) is
        # 0.49999999999999994 — the largest double below 0.5. Math.round gives
        # 0; naive floor(x + 0.5) gives 1, because the addition itself rounds
        # x + 0.5 up to 1.0 before floor sees it. This row is what tells the
        # two apart when every input is representable in the f32 the TS
        # reference reads.
        [1.3530999422073364, 0.005327165126800537, -0.005327165126800537],
    ]
    out = []
    for row in rows:
        w = np.array([row], dtype=np.float64)
        codes, scale = quantize_per_row(w)
        out.append({"input": row, "codes": codes[0].tolist(), "scale": float(scale[0])})
    json.dump(out, sys.stdout)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--selftest":
        _selftest()
    else:
        print("usage: quant_common.py --selftest", file=sys.stderr)
        sys.exit(1)
