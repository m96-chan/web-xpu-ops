/**
 * Qwen3-VL's vision tower — where a reference's pixels become tokens.
 *
 * Issue #212. This is the half that makes R2V R2V, and it is the **small**
 * half: 0.60 GB of the conditioner's 33.36 at int8, against the text stack's
 * 24.87. Worth measuring before assuming the opposite.
 *
 * **No new kernel.** `ops/conv`'s `conv3d` is the patch embedding — a second
 * caller for the op written for #201 — and `layernorm`, `matmul`, `attention`,
 * `activation` and `elementwise` cover the rest. `rope`'s axes entry takes the
 * two-axis rotation the same way `text-encoder.ts` takes M-RoPE.
 *
 * ## Tokens are in merge-block order
 *
 * Everything after the patch embed is indexed `(t, h/m, w/m, m, m)`, not
 * `(t, h, w)`: the position embedding is permuted into it, the rotary
 * coordinates are generated in it, and the mergers consume `m * m` consecutive
 * tokens as one. A raster reading has the right shape and is a picture read in
 * the wrong order.
 *
 * ## Two different GELUs
 *
 * The blocks' MLP is `gelu_pytorch_tanh`; the mergers use `nn.GELU()`, the
 * exact one. `ops/activation` has both, they differ by about 1e-3 at the knee,
 * and nothing about a wrong choice looks wrong.
 *
 * ## Two different norm placements
 *
 * The final merger normalises **before** the `m * m` shuffle, on `hidden`; the
 * deepstack mergers normalise **after**, on `hidden * m * m`. The checkpoint
 * says so in its shapes — `merger.norm.weight` is `[hidden]` and
 * `deepstack_merger_list.0.norm.weight` is `[hidden * 4]` — which is the only
 * place it is visible without reading `use_postshuffle_norm`.
 */
import { ACTIVATION, activation } from "../../../ops/activation/index.js";
import { attention } from "../../../ops/attention/index.js";
import { conv3d } from "../../../ops/conv/index.js";
import { ELEMENTWISE, elementwise } from "../../../ops/elementwise/index.js";
import { layernorm } from "../../../ops/layernorm/index.js";
import { matmul } from "../../../ops/matmul/index.js";
import { ropeAxes } from "../../../ops/rope/index.js";

export interface VisionConfig {
  hiddenSize: number;
  intermediateSize: number;
  numHeads: number;
  depth: number;
  inChannels: number;
  patchSize: number;
  spatialMergeSize: number;
  temporalPatchSize: number;
  outHiddenSize: number;
  /** `sqrt(num_position_embeddings)` — the learned table is square. */
  numGridPerSide: number;
  deepstackVisualIndexes: number[];
  ropeTheta?: number;
}

export interface VisionBlockWeights {
  norm1Weight: Float32Array;
  norm1Bias: Float32Array;
  qkvWeight: Float32Array;
  qkvBias: Float32Array;
  projWeight: Float32Array;
  projBias: Float32Array;
  norm2Weight: Float32Array;
  norm2Bias: Float32Array;
  fc1Weight: Float32Array;
  fc1Bias: Float32Array;
  fc2Weight: Float32Array;
  fc2Bias: Float32Array;
}

export interface MergerWeights {
  normWeight: Float32Array;
  normBias: Float32Array;
  fc1Weight: Float32Array;
  fc1Bias: Float32Array;
  fc2Weight: Float32Array;
  fc2Bias: Float32Array;
}

export interface VisionWeights {
  patchEmbedWeight: Float32Array;
  patchEmbedBias: Float32Array;
  posEmbed: Float32Array;
  blocks: VisionBlockWeights[];
  merger: MergerWeights;
  deepstackMergers: MergerWeights[];
}

/** `[t, h, w]` per image or video, in patches. */
export type Grid = [number, number, number];

const LAYERNORM_EPS = 1e-6;

/**
 * `torch.linspace(0, stop, n)` in f32, element for element.
 *
 * The three details are #211's: the step is rounded to f32 **first**, the
 * second half counts **down from the end**, and each element is one **fused**
 * multiply-add. Naive `i * stop / (n - 1)` disagrees at some points by one ulp,
 * which here moves an `int()` truncation across an integer boundary and picks a
 * different pair of taps for the interpolation.
 */
export function torchLinspace(stop: number, n: number): Float64Array {
  const out = new Float64Array(n);
  if (n === 1) {
    out[0] = 0;
    return out;
  }
  const step = Math.fround(stop / (n - 1));
  const halfway = Math.floor(n / 2);
  for (let i = 0; i < n; i += 1) {
    out[i] = i < halfway ? Math.fround(step * i) : Math.fround(stop - step * (n - i - 1));
  }
  return out;
}

/** `x @ Wᵀ + b`, `nn.Linear`'s layout. */
function linear(
  x: Float32Array, weight: Float32Array, rows: number, inDim: number, outDim: number, bias?: Float32Array,
): Float32Array {
  const wT = new Float32Array(inDim * outDim);
  for (let o = 0; o < outDim; o += 1) {
    for (let i = 0; i < inDim; i += 1) wT[i * outDim + o] = weight[o * inDim + i]!;
  }
  const out = matmul({ a: x, b: wT, M: rows, N: outDim, K: inDim });
  if (bias) {
    for (let r = 0; r < rows; r += 1) {
      for (let o = 0; o < outDim; o += 1) out[r * outDim + o] = out[r * outDim + o]! + bias[o]!;
    }
  }
  return out;
}

/**
 * The learned position table resampled onto one image's grid, bilinearly.
 *
 * Four taps per output, `floor` and `floor + 1` clipped to the table's edge on
 * both axes, weighted by the fractional part. **`int()` truncates**, which is
 * `floor` only because every index here is non-negative — worth saying, because
 * the difference is a different tap.
 */
export function interpolatePositionEmbedding(
  table: Float32Array, numGridPerSide: number, hidden: number, height: number, width: number,
): Float32Array {
  const hIdx = torchLinspace(numGridPerSide - 1, height);
  const wIdx = torchLinspace(numGridPerSide - 1, width);
  const out = new Float32Array(height * width * hidden);
  for (let y = 0; y < height; y += 1) {
    const hFloor = Math.trunc(hIdx[y]!);
    const hCeil = Math.min(hFloor + 1, numGridPerSide - 1);
    const dh = hIdx[y]! - hFloor;
    for (let x = 0; x < width; x += 1) {
      const wFloor = Math.trunc(wIdx[x]!);
      const wCeil = Math.min(wFloor + 1, numGridPerSide - 1);
      const dw = wIdx[x]! - wFloor;
      const taps: [number, number][] = [
        [hFloor * numGridPerSide + wFloor, (1 - dh) * (1 - dw)],
        [hFloor * numGridPerSide + wCeil, (1 - dh) * dw],
        [hCeil * numGridPerSide + wFloor, dh * (1 - dw)],
        [hCeil * numGridPerSide + wCeil, dh * dw],
      ];
      const at = (y * width + x) * hidden;
      for (const [index, weight] of taps) {
        for (let c = 0; c < hidden; c += 1) out[at + c] = out[at + c]! + table[index * hidden + c]! * weight;
      }
    }
  }
  return out;
}

/**
 * Raster `[t * h * w, dim]` into merge-block order `(t, h/m, w/m, m, m)`.
 *
 * The permutation the whole tower is indexed by. Applied to the interpolated
 * position embedding, which arrives in raster order, and generated directly for
 * the rotary coordinates.
 */
export function toMergeBlockOrder(
  x: Float32Array, frames: number, height: number, width: number, merge: number, dim: number,
): Float32Array {
  if (height % merge || width % merge) {
    throw new Error(`toMergeBlockOrder: ${height}x${width} is not a whole number of ${merge}x${merge} blocks`);
  }
  const out = new Float32Array(frames * height * width * dim);
  let at = 0;
  for (let t = 0; t < frames; t += 1) {
    for (let by = 0; by < height / merge; by += 1) {
      for (let bx = 0; bx < width / merge; bx += 1) {
        for (let iy = 0; iy < merge; iy += 1) {
          for (let ix = 0; ix < merge; ix += 1) {
            // The table is one frame's worth and repeats over `t`, which is
            // what `pos_embed.repeat(t, 1)` does before the view.
            const from = ((by * merge + iy) * width + bx * merge + ix) * dim;
            out.set(x.subarray(from, from + dim), at * dim);
            at += 1;
          }
        }
      }
    }
  }
  return out;
}

/** `(row, col)` per token, in merge-block order. */
export function visionCoordinates(frames: number, height: number, width: number, merge: number): [number, number][] {
  const coords: [number, number][] = [];
  for (let by = 0; by < height / merge; by += 1) {
    for (let bx = 0; bx < width / merge; bx += 1) {
      for (let iy = 0; iy < merge; iy += 1) {
        for (let ix = 0; ix < merge; ix += 1) coords.push([by * merge + iy, bx * merge + ix]);
      }
    }
  }
  const out: [number, number][] = [];
  for (let t = 0; t < frames; t += 1) out.push(...coords);
  return out;
}

/**
 * `[tokens, headDim / 2]` positions with the frequency folded in.
 *
 * Two axes — row then column — each owning half of `headDim / 2`, and each
 * channel keeping `theta ** (-2i / (headDim / 2))` where `i` is its index
 * *within its axis*. Unlike the text stack's M-RoPE, this one really is a
 * per-axis sweep, because `rot_pos_emb` looks both coordinates up in the *same*
 * `freq_table` and concatenates.
 */
export function visionPositions(
  coords: [number, number][], headDim: number, theta: number,
): Float32Array {
  const half = headDim / 2;
  const perAxis = half / 2;
  const out = new Float32Array(coords.length * half);
  for (let token = 0; token < coords.length; token += 1) {
    for (let i = 0; i < perAxis; i += 1) {
      const invFreq = Math.pow(theta, (-2 * i) / half);
      out[token * half + i] = coords[token]![0] * invFreq;
      out[token * half + perAxis + i] = coords[token]![1] * invFreq;
    }
  }
  return out;
}

/** `c` against `c + headDim / 2`, as `rotate_half` pairs them. */
export function visionPermutation(headDim: number): number[] {
  const half = headDim / 2;
  const order = new Array<number>(headDim);
  for (let c = 0; c < half; c += 1) {
    order[2 * c] = c;
    order[2 * c + 1] = c + half;
  }
  return order;
}

function splitHeads(x: Float32Array, seq: number, heads: number, dim: number): Float32Array {
  const out = new Float32Array(seq * heads * dim);
  for (let s = 0; s < seq; s += 1) {
    for (let h = 0; h < heads; h += 1) {
      for (let d = 0; d < dim; d += 1) out[(h * seq + s) * dim + d] = x[(s * heads + h) * dim + d]!;
    }
  }
  return out;
}

/**
 * `norm -> fc1 -> exact GELU -> fc2` over `m * m` consecutive tokens.
 *
 * `postShuffle` is where the norm sits: **after** the shuffle for a deepstack
 * merger, **before** it for the tower's own. Visible in the checkpoint as the
 * norm's width, and nowhere else.
 */
export function visionMerger(
  w: MergerWeights, x: Float32Array, tokens: number, hidden: number, merge: number, outHidden: number,
  postShuffle: boolean,
): Float32Array {
  const wide = hidden * merge * merge;
  const rows = tokens / (merge * merge);
  const normed = postShuffle
    ? layernorm({ input: x, weight: w.normWeight, bias: w.normBias, N: rows, D: wide, eps: LAYERNORM_EPS })
    : layernorm({ input: x, weight: w.normWeight, bias: w.normBias, N: tokens, D: hidden, eps: LAYERNORM_EPS });
  const first = linear(normed, w.fc1Weight, rows, wide, wide, w.fc1Bias);
  // **The exact GELU**, not the tanh approximation the blocks' MLP uses.
  const activated = activation({ input: first, kind: ACTIVATION.gelu });
  return linear(activated, w.fc2Weight, rows, wide, outHidden, w.fc2Bias);
}

/** One block: pre-norm attention with a fused QKV, then a pre-norm MLP. */
export function visionBlock(
  cfg: VisionConfig, w: VisionBlockWeights, x: Float32Array, seq: number, positions: Float32Array,
  segments: number[],
): Float32Array {
  const { hiddenSize, numHeads } = cfg;
  const headDim = hiddenSize / numHeads;
  const axisDims = new Array<number>(headDim / 2).fill(2);

  const normed = layernorm({
    input: x, weight: w.norm1Weight, bias: w.norm1Bias, N: seq, D: hiddenSize, eps: LAYERNORM_EPS,
  });
  const qkv = linear(normed, w.qkvWeight, seq, hiddenSize, 3 * hiddenSize, w.qkvBias);
  const take = (part: number): Float32Array => {
    const out = new Float32Array(seq * hiddenSize);
    for (let s = 0; s < seq; s += 1) {
      out.set(qkv.subarray(s * 3 * hiddenSize + part * hiddenSize, s * 3 * hiddenSize + (part + 1) * hiddenSize),
        s * hiddenSize);
    }
    return out;
  };
  let q = take(0);
  let k = take(1);
  const v = take(2);

  q = ropeAxes({ input: q, N: seq, numHeads, axisDims, positions, thetaBase: 1 });
  k = ropeAxes({ input: k, N: seq, numHeads, axisDims, positions, thetaBase: 1 });

  // **One attention document per image**, which is what `cu_seqlens` is: two
  // references in one batch must not see each other, and with a single image
  // the distinction is invisible.
  const merged = new Float32Array(seq * hiddenSize);
  let start = 0;
  for (const length of segments) {
    const slice = (src: Float32Array): Float32Array =>
      src.slice(start * hiddenSize, (start + length) * hiddenSize);
    const attended = attention({
      q: splitHeads(slice(q), length, numHeads, headDim),
      k: splitHeads(slice(k), length, numHeads, headDim),
      v: splitHeads(slice(v), length, numHeads, headDim),
      B: 1, H: numHeads, L: length, S: length, D: headDim, Dv: headDim, causal: false,
    });
    for (let h = 0; h < numHeads; h += 1) {
      for (let s = 0; s < length; s += 1) {
        for (let d = 0; d < headDim; d += 1) {
          merged[((start + s) * numHeads + h) * headDim + d] = attended.output[(h * length + s) * headDim + d]!;
        }
      }
    }
    start += length;
  }

  let hidden = elementwise({
    a: x, b: linear(merged, w.projWeight, seq, hiddenSize, hiddenSize, w.projBias), kind: ELEMENTWISE.add,
  });
  const normed2 = layernorm({
    input: hidden, weight: w.norm2Weight, bias: w.norm2Bias, N: seq, D: hiddenSize, eps: LAYERNORM_EPS,
  });
  const ff = linear(
    // **`gelu_pytorch_tanh` here**, the approximation — the mergers use the
    // exact one.
    activation({
      input: linear(normed2, w.fc1Weight, seq, hiddenSize, cfg.intermediateSize, w.fc1Bias),
      kind: ACTIVATION.gelu_tanh,
    }),
    w.fc2Weight, seq, cfg.intermediateSize, hiddenSize, w.fc2Bias,
  );
  return elementwise({ a: hidden, b: ff, kind: ELEMENTWISE.add });
}

export interface VisionOutput {
  /** `[tokens, hidden]` — the tower's own output, before the merger. */
  lastHiddenState: Float32Array;
  /** `[tokens / merge², outHidden]` — the vision tokens the conditioner reads. */
  pooled: Float32Array;
  /** One per deepstack index, each the same shape as `pooled`. */
  deepstack: Float32Array[];
}

/** The whole tower: patches in, vision tokens out. */
export function visionForward(
  cfg: VisionConfig, w: VisionWeights, pixels: Float32Array, grids: Grid[],
): VisionOutput {
  const { hiddenSize, patchSize, temporalPatchSize, inChannels, spatialMergeSize: merge } = cfg;
  const patchDim = inChannels * temporalPatchSize * patchSize * patchSize;
  const tokens = grids.reduce((sum, [t, h, wd]) => sum + t * h * wd, 0);
  if (pixels.length !== tokens * patchDim) {
    throw new Error(`visionForward: ${pixels.length} values for ${tokens} patches of ${patchDim}`);
  }

  // **`ops/conv`'s conv3d**, with the kernel equal to the stride and to the
  // input — which is what `nn.Conv3d` over one patch per item is. A second
  // caller for the op #201 added.
  const embedded = conv3d({
    input: pixels,
    weight: w.patchEmbedWeight,
    bias: w.patchEmbedBias,
    N: tokens, Cin: inChannels, D: temporalPatchSize, H: patchSize, W: patchSize,
    Cout: hiddenSize, KD: temporalPatchSize, KH: patchSize, KW: patchSize,
    stride: [temporalPatchSize, patchSize, patchSize],
    padding: [0, 0, 0],
    dilation: [1, 1, 1],
    groups: 1,
  });

  // The position embedding, interpolated per image and permuted into
  // merge-block order.
  const positionsEmbedded = new Float32Array(tokens * hiddenSize);
  const coords: [number, number][] = [];
  const segments: number[] = [];
  let at = 0;
  for (const [frames, height, width] of grids) {
    const raster = interpolatePositionEmbedding(w.posEmbed, cfg.numGridPerSide, hiddenSize, height, width);
    const repeated = new Float32Array(frames * height * width * hiddenSize);
    for (let t = 0; t < frames; t += 1) repeated.set(raster, t * height * width * hiddenSize);
    positionsEmbedded.set(
      toMergeBlockOrder(repeated, frames, height, width, merge, hiddenSize), at * hiddenSize);
    coords.push(...visionCoordinates(frames, height, width, merge));
    // `cu_seqlens` repeats `h * w` for each frame: a frame is its own document.
    for (let t = 0; t < frames; t += 1) segments.push(height * width);
    at += frames * height * width;
  }

  let x = elementwise({ a: embedded, b: positionsEmbedded, kind: ELEMENTWISE.add });

  const headDim = hiddenSize / cfg.numHeads;
  const positions = visionPositions(coords, headDim, cfg.ropeTheta ?? 10000);

  const deepstack: Float32Array[] = [];
  for (const [index, block] of w.blocks.entries()) {
    x = visionBlock(cfg, block, x, tokens, positions, segments);
    const tap = cfg.deepstackVisualIndexes.indexOf(index);
    if (tap >= 0) {
      const merger = w.deepstackMergers[tap];
      if (!merger) throw new Error(`visionForward: no merger for deepstack index ${index}`);
      deepstack.push(visionMerger(merger, x, tokens, hiddenSize, merge, cfg.outHiddenSize, true));
    }
  }

  return {
    lastHiddenState: x,
    pooled: visionMerger(w.merger, x, tokens, hiddenSize, merge, cfg.outHiddenSize, false),
    deepstack,
  };
}
