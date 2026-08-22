export interface NearestUpsample2dArgs {
  /** `[N, C, H, W]`, row-major. */
  input: Float32Array;
  /** Batch. */
  N: number;
  /** Channels. Untouched: this op resamples the two spatial axes only. */
  C: number;
  /** Input height. */
  H: number;
  /** Input width. */
  W: number;
  /** Output height. Must be `>= H` — see the doc on `nearestUpsample2d`. */
  outH: number;
  /** Output width. Must be `>= W`. */
  outW: number;
}

/**
 * The ratio PyTorch's nearest resampler multiplies the destination index by:
 * `float32(inSize) / outSize`, rounded to f32 exactly once.
 *
 * Exported because the WGSL kernel cannot compute it itself and stay in step.
 * WGSL guarantees f32 `*` is correctly rounded but allows `/` to be off by up
 * to 2.5 ULP, and this ratio's last bit decides whole rows (see the H=14 → 46
 * case in the doc below), so the division happens once on the host and the
 * shader receives the result. Both the reference and the kernel read their
 * scale from here, so there is one definition of it rather than two that agree
 * until they do not. That is measured, not inferred from the spec: dividing in
 * the shader instead moves H=14 -> 46's destination row 23 from source row 6
 * (torch's answer) to source row 7 on an RTX 5090 — see `wgsl/kernel.wgsl`.
 *
 * `Math.fround(a / b)` divides in f64 and then rounds to f32, which is a double
 * rounding and therefore not obviously the same value as C's single-rounded
 * `(float)a / (float)b` that torch computes. Checked rather than assumed: the
 * two agree for every integer pair with `a <= 1024, b <= 4096` (4,194,304
 * pairs, compared against numpy float32 division). Both operands here are small
 * exact integers, which is why.
 */
export function nearestUpsampleScale(inSize: number, outSize: number): number {
  return Math.fround(inSize / outSize);
}

/**
 * Nearest-neighbour 2D upsample over `[N, C, H, W]`, matching
 * `torch.nn.functional.interpolate(input, size=(outH, outW), mode='nearest')`.
 *
 * The definition of correct for this op. Every backend is measured against it,
 * and it is deliberately the slowest, plainest expression of the move — its job
 * is to be obviously right, not fast.
 *
 * No weights and no arithmetic on the values: every output element is a bitwise
 * copy of some input element. That makes this a different op from
 * `convTranspose1d`, which also raises resolution but does it with learned
 * weights; VAE / codec decoders that avoid checkerboard artefacts use "nearest
 * upsample, then conv" *instead of* a transposed conv, so a decoder needs both
 * ops and neither substitutes for the other.
 *
 * ## Conventions (rule 7: follow PyTorch)
 *
 * Measured against torch 2.10.0+cu128 rather than read off the docs. Every
 * number quoted below was produced by running `F.interpolate`.
 *
 * - **Output size, not scale factor.** `F.interpolate` accepts either, and the
 *   two are *not* two spellings of one thing — with the default
 *   `recompute_scale_factor=None` they take different paths and disagree.
 *   Measured, `H = 3`: `scale_factor=1.6` gives 4 rows mapped `0, 0, 1, 1`
 *   (source index `floor(dst / 1.6)`), while `size=(4, ...)` gives the same 4
 *   rows mapped `0, 0, 1, 2` (source index `floor(dst * 3/4)`). This op takes
 *   the output size, so `nearestUpsample2d` is the `size=` path and only that
 *   one:
 *     - the output size is what the kernel needs to allocate and dispatch, so
 *       taking a float scale would move a rounding decision (`floor(H * s)`)
 *       into the op, where a caller cannot see which way it went;
 *     - integers in, integers out — nothing about the *shape* then depends on
 *       f32 rounding;
 *     - a caller who wants a scale factor computes `outH = floor(H * s)`, which
 *       is what torch's `recompute_scale_factor=True` does. That path is
 *       *this* op's answer. `recompute_scale_factor=False` (torch's default
 *       when a scale is passed) is a different answer and is deliberately not
 *       reachable from here; it would need the float scale carried all the way
 *       into the index formula. **They agree whenever `outH` is an exact
 *       integer multiple of `H`** — including the 2x of every decoder this op
 *       exists for — and can disagree otherwise, which is why the disagreement
 *       is written down here instead of being discovered later.
 * - **Source index is `floor(dst * scale)` in f32**, `scale = inSize / outSize`
 *   — torch's `nearest_neighbor_compute_source_index` in
 *   `ATen/native/UpSample.h`, whose own comment reads "Index computation
 *   matching OpenCV INTER_NEAREST which is buggy and kept for BC". The f32 is
 *   load-bearing, not an implementation detail: at `H = 14, outH = 46`,
 *   destination row 23 has exact source `23 * 14 / 46 = 7` exactly, but
 *   `float32(14/46) = 0.30434784` is below `14/46`, so `23 * 0.30434784`
 *   floors to **6**. torch returns row 6 there. Doing the same index in exact
 *   integer arithmetic (`(dst * inSize) / outSize | 0`) returns row 7 — a
 *   whole row different, silently. So this reference multiplies in f32 too,
 *   and the tests pin `[…, 6, 6, 6, 6, 7, …]` at that shape. Every destination
 *   index that lands exactly on a source pixel boundary is a candidate for
 *   this; there are 156 such indices among the pairs `in <= 64,
 *   out <= 4 * in`, so it is not a corner case reachable only by absurd sizes.
 * - **`align_corners` does not exist for nearest, and must not be added.** It
 *   is not "defaulted to false" here — torch *raises* for it:
 *   `F.interpolate(x, scale_factor=2, mode='nearest', align_corners=True)` →
 *   "align_corners option can only be set with the interpolating modes: linear
 *   | bilinear | bicubic | trilinear". It parameterises how a *sample grid* is
 *   aligned before interpolating between neighbours, and nearest interpolates
 *   between nothing. An `alignCorners` flag on this op would therefore be a
 *   flag with no defined meaning to copy from anywhere.
 * - **`mode='nearest-exact'` is a different function**, not a refinement of
 *   this one, and is not implemented. It uses `floor((dst + 0.5) * scale)` —
 *   the Pillow / scikit-image convention. At `H = 3, outH = 7` this op gives
 *   rows `0, 0, 0, 1, 1, 2, 2` and `nearest-exact` gives `0, 0, 1, 1, 1, 2, 2`
 *   (both measured). A model exported against one and run against the other is
 *   off by a row with no error, so the two are named apart rather than hidden
 *   behind a default.
 * - **Non-integer ratios are allowed.** `outH` and `outW` are independent
 *   positive integers, so 3 → 5 and 4 → 5 in one call is legal and matches
 *   torch. Nothing rounds: there is no scale factor to round, which is the
 *   point of taking the size.
 * - **Downsampling throws.** `outH < H` or `outW < W` is refused rather than
 *   quietly evaluated. The same formula *is* what torch uses when shrinking,
 *   but shrinking is out of scope by ISSUE #146 and nothing here is measured
 *   against torch for it; an op that quietly answers a question it was never
 *   checked on is worse than one that says no (#143). `outH === H` is fine and
 *   is a copy — resampling one axis only has to work.
 *
 * ## Why there is no clamp on the source index
 *
 * torch's index helper ends in `std::min(..., input_size - 1)`. This does not,
 * because `outSize >= inSize` makes it unreachable: `scale <= 1`, so the exact
 * product is at most `(outSize - 1) * inSize / outSize = inSize - inSize/outSize`,
 * and pushing that up to `inSize` would take a relative error near
 * `1 / outSize` — some twelve orders of magnitude more than the f32 rounding
 * involved for any size that fits in a buffer. Searched directly as well: the
 * clamp fires for no pair with `inSize <= 299, outSize <= 1200`. A guard that
 * no test can ever turn red is a line that hides rather than protects, so the
 * restriction to upsampling carries the safety instead.
 */
export function nearestUpsample2d({
  input,
  N,
  C,
  H,
  W,
  outH,
  outW,
}: NearestUpsample2dArgs): Float32Array {
  if (!Number.isInteger(outH) || !Number.isInteger(outW) || outH < 1 || outW < 1) {
    // The realistic caller error: `outH: H * 1.5` without the floor. Left to
    // run it would allocate a fractional length — Float32Array truncates — and
    // return an array whose shape nothing downstream can explain.
    throw new Error(`nearestUpsample2d(): outH=${outH} and outW=${outW} must be positive integers`);
  }
  if (outH < H || outW < W) {
    throw new Error(
      `nearestUpsample2d(): ${H}x${W} -> ${outH}x${outW} shrinks an axis; this op only upsamples (ISSUE #146)`,
    );
  }
  if (input.length !== N * C * H * W) {
    throw new Error(`nearestUpsample2d(): expected ${N * C * H * W} input elements, got ${input.length}`);
  }

  const scaleH = nearestUpsampleScale(H, outH);
  const scaleW = nearestUpsampleScale(W, outW);
  const output = new Float32Array(N * C * outH * outW);

  for (let plane = 0; plane < N * C; plane += 1) {
    // n and c never appear apart: the resample is per (n, c) plane, so the two
    // batch axes are one flat plane index here and in the kernel.
    const inPlane = plane * H * W;
    const outPlane = plane * outH * outW;
    for (let oh = 0; oh < outH; oh += 1) {
      // Math.fround around the product, not only around the scale: torch
      // multiplies two f32 and rounds the result to f32 before flooring, and
      // the f64 product of an f32 by a small integer is exact, so this is that
      // multiply rather than an approximation of it.
      const ih = Math.floor(Math.fround(oh * scaleH));
      for (let ow = 0; ow < outW; ow += 1) {
        const iw = Math.floor(Math.fround(ow * scaleW));
        output[outPlane + oh * outW + ow] = input[inPlane + ih * W + iw]!;
      }
    }
  }
  return output;
}
