/**
 * Block transpose: `[dim0, dim1, D]` -> `[dim1, dim0, D]`, row-major, `D`
 * contiguous elements moved as one unit per `(dim0, dim1)` pair.
 *
 * No arithmetic — every element of the output is bit-identical to its
 * source, only its position changes, so this is the whole definition of
 * correct (rule 8's usual "naive, obviously right" reference is not a
 * simplification here; it *is* the operation).
 *
 * Exists for issue #117's GPU-resident LLM prefill
 * (`llm/engine-q8-resident.ts#runPrefillResident`), which needs the
 * `[tokens, heads, dim]` token-major <-> `[heads, tokens, dim]` head-major
 * reshape `llm/reshape.ts#splitHeadsMajor`/`mergeHeadsMajor` already define
 * on the CPU, but as a GPU dispatch: reading Q/K/V back to the CPU to
 * reshape and re-uploading would put a round trip exactly where going
 * resident removes one. `splitHeadsMajor(x, tokens, heads, dim)` is this op
 * at `dim0 = tokens, dim1 = heads`; `mergeHeadsMajor(x, heads, tokens, dim)`
 * is `dim0 = heads, dim1 = tokens` — the indexing below does not know or
 * care which axis is "tokens" and which is "heads", so one op serves both
 * directions (`ops/permute/wgsl.test.ts` checks both against those two
 * functions directly, since they are already proven correct by every LLM
 * engine test that exercises prefill — not a re-derivation, rule 2).
 *
 * Lives in `ops/` rather than as `llm/`-only plumbing (an earlier version
 * was `llm/wgsl/permute.wgsl`, with no `reference.ts` of its own) because
 * this repository's browser bundle (`examples/llm-demo/src/browser-runtime.ts`)
 * resolves every kernel `llm/kernels.ts`/`llm/engine-q8-resident.ts` ask for
 * by parsing `.../ops/<name>/index.ts` out of the URL `harness/suite.ts#kernel`
 * is called with (`opNameFromUrl`) — a shape only `ops/` provides, not an
 * arbitrary `llm/` path.
 */
export interface PermuteArgs {
  /** `[dim0, dim1, D]`, row-major. */
  input: Float32Array;
  dim0: number;
  dim1: number;
  D: number;
}

/** Returns a row-major `[dim1, dim0, D]` array. */
export function permute({ input, dim0, dim1, D }: PermuteArgs): Float32Array {
  if (input.length !== dim0 * dim1 * D) {
    throw new Error(`permute(): expected ${dim0 * dim1 * D} input elements, got ${input.length}`);
  }
  const output = new Float32Array(dim0 * dim1 * D);
  for (let i0 = 0; i0 < dim0; i0 += 1) {
    for (let i1 = 0; i1 < dim1; i1 += 1) {
      const from = (i0 * dim1 + i1) * D;
      const to = (i1 * dim0 + i0) * D;
      output.set(input.subarray(from, from + D), to);
    }
  }
  return output;
}
