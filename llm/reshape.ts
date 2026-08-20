/**
 * Pure host-side layout changes the engine needs between dispatches.
 *
 * None of these are compute — every one is a relabelling of the same numbers —
 * so they run on the CPU rather than adding a WGSL kernel each. That is a
 * deliberate scope choice for this skeleton (issue #98's "f32 weight path,
 * correctness first" — rule 8), not an oversight: the volumes involved are one
 * forward pass's worth of activations (`N` tokens, not the model's weights),
 * and fusing them into dedicated kernels is exactly the kind of optimisation
 * rule 8 says comes *after* a reference-correct engine exists, not before.
 */

/**
 * `[rows, cols]` row-major → `[cols, rows]` row-major.
 *
 * Every weight in `llm/weights.ts` is stored `[outFeatures, inFeatures]` —
 * `matvec`'s `[M, K]` convention, and how `torch.nn.Linear.weight` itself is
 * laid out, so a checkpoint's weight needs no reshaping to feed `matvec`
 * directly during decode.
 *
 * `matmul`, used during prefill, wants its second operand as `[K, N]` —
 * contracted dimension first — so an `[out, in]` weight has to be transposed
 * into `[in, out]` before it can stand in as `matmul`'s `b`. This is that
 * transpose, applied once per weight at engine construction rather than once
 * per forward pass: the weights do not change between tokens, and computing it
 * per weight rather than per dispatch is the entire saving available before
 * this becomes an op of its own.
 */
export function transposeRowMajor(matrix: Float32Array, rows: number, cols: number): Float32Array {
  const output = new Float32Array(rows * cols);
  for (let r = 0; r < rows; r += 1) {
    for (let c = 0; c < cols; c += 1) {
      output[c * rows + r] = matrix[r * cols + c]!;
    }
  }
  return output;
}

/**
 * `[tokens, heads, dim]` (token-major) → `[heads, tokens, dim]` (head-major).
 *
 * `matvec` / `matmul` project a hidden state `[tokens, hidden]` straight into
 * `[tokens, heads*dim]` — a token's `heads*dim` output channels sit together
 * because they are one contiguous run of one output row — and `rope` reads and
 * writes exactly that same `[N, numHeads, headDim]` layout unchanged. So Q, K
 * and V arrive from projection and rotation token-major.
 *
 * `ops/gqa`'s two kernels want the opposite: `[B, H, L, D]` for Q and
 * `[B, kvHeads, S, D]` for K/V, head-major, because that is the axis the KV
 * cache is addressed by (`ops/gqa/reference.ts`'s `kvHeadOf`) and the axis a
 * dispatch is parallel over (`workgroups: [L, H, B]`). This is the transpose
 * between those two layouts — PyTorch's `.transpose(1, 2)` on
 * `[B, L, H, D] → [B, H, L, D]`, with `B` folded away since this engine keeps
 * batch size 1 throughout (issue #98's scope).
 */
export function splitHeadsMajor(
  tokenMajor: Float32Array,
  tokens: number,
  heads: number,
  dim: number,
): Float32Array {
  const output = new Float32Array(tokens * heads * dim);
  for (let t = 0; t < tokens; t += 1) {
    for (let h = 0; h < heads; h += 1) {
      const from = (t * heads + h) * dim;
      const to = (h * tokens + t) * dim;
      output.set(tokenMajor.subarray(from, from + dim), to);
    }
  }
  return output;
}

/**
 * The inverse of `splitHeadsMajor`: `[heads, tokens, dim]` → `[tokens, heads, dim]`.
 *
 * `ops/gqa`'s context kernel returns `[B, H, L, Dv]` — head-major, one block per
 * query head — and the O-projection that follows wants `[L, H*Dv]`, one row per
 * token with every head's output concatenated, because that row is exactly
 * `matvec`/`matmul`'s input vector for that token. This merges the heads back
 * before that projection, undoing the split `splitHeadsMajor` made going in.
 */
export function mergeHeadsMajor(
  headMajor: Float32Array,
  heads: number,
  tokens: number,
  dim: number,
): Float32Array {
  const output = new Float32Array(tokens * heads * dim);
  for (let h = 0; h < heads; h += 1) {
    for (let t = 0; t < tokens; t += 1) {
      const from = (h * tokens + t) * dim;
      const to = (t * heads + h) * dim;
      output.set(headMajor.subarray(from, from + dim), to);
    }
  }
  return output;
}

/**
 * Stacks `[rows[i], cols]` row-major matrices into one `[sum(rows), cols]`
 * matrix — plain concatenation along the output-feature axis.
 *
 * `LlamaEngine` uses this to fuse Q/K/V (and gate/up) into a single
 * projection weight, so one `matvec`/`matmul` dispatch computes all of them:
 * `Wq @ x`, `Wk @ x` and `Wv @ x` read the same `x` and are independent rows
 * of the same linear map, so `[Wq; Wk; Wv] @ x` computes exactly the three
 * separate results, concatenated — not an approximation, a regrouping of the
 * same sums. `splitConcatRows` below undoes the concatenation on the output
 * side.
 *
 * This exists for GPU dispatch count, not FLOPs: this environment's `webgpu`
 * binding is measurably unstable past roughly a hundred dispatches within one
 * device's lifetime (`llm/engine.wgsl.test.ts` documents the measurement,
 * which is why fusing Q/K/V and gate/up was worth doing here rather than left
 * for a later optimisation pass — rule 8 is about correctness before *speed*,
 * and this is neither: an unfused engine cannot be run through this binding
 * at all, on this device, for long enough to be tested).
 */
export function concatRows(matrices: Float32Array[], rows: number[], cols: number): Float32Array {
  const total = rows.reduce((a, b) => a + b, 0);
  const output = new Float32Array(total * cols);
  let offset = 0;
  matrices.forEach((matrix, i) => {
    output.set(matrix, offset);
    offset += rows[i]! * cols;
  });
  return output;
}

/**
 * `concatRows`, for `Int8Array` rather than `Float32Array`.
 *
 * Issue #105's int8 engine path fuses Q/K/V (and gate/up) the same way the f32
 * engine does (`concatRows` above) — one `matvecQ8`/dequant-then-`matmul`
 * dispatch instead of three — but the source rows are per-row-quantized
 * **codes**, not weights, so concatenating them needs no separate function
 * body, only a separate element type: a code is just an int8 the row's scale
 * has not yet been applied to, and stacking rows does not touch column values,
 * so it commutes with quantization exactly (quantizing three matrices and
 * concatenating the codes gives the same per-row codes as concatenating first
 * and quantizing — the scale stays per-row regardless of order, which is why
 * `LlamaLayerWeightsQ8`'s per-projection `scale` arrays are concatenated
 * separately, not recomputed here).
 */
export function concatRowsInt8(matrices: Int8Array[], rows: number[], cols: number): Int8Array {
  const total = rows.reduce((a, b) => a + b, 0);
  const output = new Int8Array(total * cols);
  let offset = 0;
  matrices.forEach((matrix, i) => {
    output.set(matrix, offset);
    offset += rows[i]! * cols;
  });
  return output;
}

/**
 * The inverse of concatenating projections along the output axis, applied
 * after the fused dispatch instead of before it: `flat` is `[tokens, sum(sizes)]`
 * token-major — one row per token, that row being every fused output
 * concatenated — and this splits each row back into `sizes.length` separate
 * `[tokens, sizes[i]]` token-major arrays.
 */
export function splitConcatRows(flat: Float32Array, tokens: number, sizes: number[]): Float32Array[] {
  const rowWidth = sizes.reduce((a, b) => a + b, 0);
  const outputs = sizes.map((size) => new Float32Array(tokens * size));
  for (let t = 0; t < tokens; t += 1) {
    let colOffset = 0;
    for (let p = 0; p < sizes.length; p += 1) {
      const size = sizes[p]!;
      const from = t * rowWidth + colOffset;
      outputs[p]!.set(flat.subarray(from, from + size), t * size);
      colOffset += size;
    }
  }
  return outputs;
}
