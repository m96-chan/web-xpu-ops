/**
 * Padding one axis of a tensor, as `torch.nn.functional.pad`.
 *
 * Issue #200. `ops/conv` deliberately has no `padding_mode`: its own doc says a
 * pad is a pad op, not a convolution argument. MiniMax-H3 is the model that
 * makes that separation load-bearing rather than tidy — its visual VAE pads
 * `reflect` on the spatial axes and, on the temporal axis, **causally**:
 * `2 * padding` frames before the data and none after, so the frame at time `t`
 * cannot see `t + 1` (`video_vae/conv.py`, `_apply_temporal_padding`). No
 * symmetric `padding` argument can say that, whatever its mode.
 *
 * ## One axis at a time, and why that is the whole design
 *
 * `F.pad` takes a flat list of pairs and pads several axes at once. This takes
 * **one** axis, and a caller that wants three calls it three times. That is not
 * a simplification of torch — it is the same thing, and it was measured to be:
 *
 *     F.pad(x, (1,1,1,1), mode="reflect")   on [1,1,3,4]
 *
 * gives exactly what padding W by (1,1) and then H by (1,1) gives, element for
 * element. `reference.test.ts` pins that, because it is the property the whole
 * interface rests on and it is not obvious — a reflection reads its neighbours,
 * so padding H *after* W means the corners are reflections of already-reflected
 * values, and it would have been reasonable to expect the two to differ.
 *
 * The axis is named by three numbers rather than by a shape: the tensor is
 * viewed as `[outer, L, inner]`, where `L` is the axis being padded. Every case
 * this library has is one of
 *
 *     [N, C, L]        pad L      outer = N*C,      inner = 1
 *     [N, C, D, H, W]  pad W      outer = N*C*D*H,  inner = 1
 *                      pad H      outer = N*C*D,    inner = W
 *                      pad D      outer = N*C,      inner = H*W
 *
 * so the op never needs to know the rank, and the same kernel serves the audio
 * VAE's 1D `replicate` and the visual VAE's 3D `reflect`.
 */

/**
 * How the padded positions are filled.
 *
 * The names and the behaviour are torch's, measured on
 * `x = [1,2,3,4,5,6]` with `pad=(2,3)` against torch 2.10.0+cu130:
 *
 * | mode | result |
 * | --- | --- |
 * | `constant` (value 0) | `[0,0,1,2,3,4,5,6,0,0,0]` |
 * | `replicate` | `[1,1,1,2,3,4,5,6,6,6,6]` |
 * | `reflect` | `[3,2,1,2,3,4,5,6,5,4,3]` |
 *
 * The row that matters is the last one. **`reflect` does not repeat the edge**:
 * the element before `1` is `2`, not `1`. `replicate` is the one that repeats
 * it. Getting these two the wrong way round produces a tensor of the right
 * shape whose every interior element is correct, and that is the kind of wrong
 * answer that survives to the output image.
 *
 * `circular` is torch's fourth mode and is not here: nothing in H3 uses it, and
 * an unused mode is an untested one.
 */
export type PadMode = "constant" | "reflect" | "replicate";

export interface PadArgs {
  /** Viewed as `[outer, L, inner]`, row-major. */
  input: Float32Array;
  /** Elements before the padded axis — the product of every earlier dimension. */
  outer: number;
  /** Length of the axis being padded. */
  L: number;
  /** Elements after the padded axis — the product of every later dimension. */
  inner: number;
  /** Positions added before the data. May differ from `after`; that is the point. */
  before: number;
  /** Positions added after the data. */
  after: number;
  /** Default `"constant"`, as torch. */
  mode?: PadMode;
  /** `constant` only. Default 0, as torch. */
  value?: number;
}

/**
 * Where an output position reads from, or `-1` for "fill with `value`".
 *
 * Split out because it is the entire semantic content of the op, and because
 * the WGSL kernel has to compute the same thing — one definition rather than
 * two that can drift.
 */
export function padSourceIndex(position: number, L: number, before: number, mode: PadMode): number {
  const source = position - before;
  if (source >= 0 && source < L) return source;
  switch (mode) {
    case "constant":
      return -1;
    case "replicate":
      // Clamp to the nearest real element.
      return source < 0 ? 0 : L - 1;
    case "reflect":
      // Mirror about the edge *element*, which is therefore not repeated:
      // -1 reads 1, and L reads L-2.
      return source < 0 ? -source : 2 * (L - 1) - source;
  }
}

/** Length of the padded axis. */
export function padOutputLength({ L, before, after }: { L: number; before: number; after: number }): number {
  return before + L + after;
}

/**
 * Pads one axis of `[outer, L, inner]`.
 *
 * The definition of correct: the plainest expression of the table above, three
 * nested loops, no attempt at anything else.
 */
export function pad({ input, outer, L, inner, before, after, mode = "constant", value = 0 }: PadArgs): Float32Array {
  if (before < 0 || after < 0) {
    // torch: "Padding size should be greater than or equal to zero" for these
    // modes. Negative padding *crops* in torch's constant mode, which is a
    // different operation with a different output length; refusing it here
    // keeps one meaning per argument.
    throw new Error(`pad(): negative padding is not supported, got before=${before} after=${after}`);
  }
  if (mode === "reflect" && (before >= L || after >= L)) {
    // torch: "Padding size should be less than the corresponding input
    // dimension". Measured: L=6 accepts (5,5) and raises on (6,0). A reflection
    // that reaches past the far edge has nothing to mirror, and the index this
    // would compute walks off the axis rather than wrapping.
    throw new Error(
      `pad(): reflect needs padding smaller than the axis, got before=${before} after=${after} for L=${L}`,
    );
  }
  if (input.length !== outer * L * inner) {
    throw new Error(`pad(): expected ${outer * L * inner} input elements, got ${input.length}`);
  }

  const Lout = padOutputLength({ L, before, after });
  const output = new Float32Array(outer * Lout * inner);

  for (let o = 0; o < outer; o += 1) {
    for (let position = 0; position < Lout; position += 1) {
      const source = padSourceIndex(position, L, before, mode);
      const to = (o * Lout + position) * inner;
      if (source < 0) {
        output.fill(value, to, to + inner);
        continue;
      }
      const from = (o * L + source) * inner;
      for (let i = 0; i < inner; i += 1) output[to + i] = input[from + i]!;
    }
  }
  return output;
}
