// Block transpose: `[dim0, dim1, D]` -> `[dim1, dim0, D]`, row-major, `D`
// floats moved as one unit per (dim0, dim1) pair.
//
// Issue #117's GPU-resident prefill needs exactly the reshape
// `llm/reshape.ts#splitHeadsMajor`/`mergeHeadsMajor` already define on the
// CPU (`[tokens, heads, dim]` token-major <-> `[heads, tokens, dim]`
// head-major — see that file's doc for *why* `ops/gqa` needs head-major
// while `matmul`'s projections produce token-major). Prefill cannot pay a
// CPU round trip for it the way the legacy engine does — the whole point of
// going resident is one `queue.submit` for the entire prefill, and reading
// Q/K/V/context back to reshape on the CPU and re-uploading would put a
// round trip exactly where issue #110 removed one for decode. So this is
// that same reshape, as one GPU dispatch.
//
// One kernel serves both directions in `llm/engine-q8-resident.ts`'s
// prefill, because splitting and merging are the same permutation with
// `dim0`/`dim1` swapped: `splitHeadsMajor(x, tokens, heads, dim)` is this
// kernel at `dim0 = tokens, dim1 = heads`, and `mergeHeadsMajor(x, heads,
// tokens, dim)` is this kernel at `dim0 = heads, dim1 = tokens` — the
// indexing below does not know or care which axis is "tokens" and which is
// "heads".
//
// `ops/permute/reference.ts` is the definition of correct — no arithmetic,
// so it and this kernel do the same index bookkeeping, not the same sums.
// `ops/permute/wgsl.test.ts` checks this kernel against that reference
// directly, and separately against `llm/reshape.ts#splitHeadsMajor`/
// `mergeHeadsMajor` (the functions this op exists to replace with one GPU
// dispatch — see this file's doc above), which are the same permutation
// under llm/'s own names and already proven correct by every engine test
// that exercises prefill.

// A dispatch wider than one row of workgroups is folded back into one index
// here rather than by a uniform: `num_workgroups.x` is what the host asked for,
// so **every existing one-dimensional caller keeps working unchanged** — at
// `[n]` the y extent is 1 and `gid.y` is 0. The ceiling is 65,535 workgroups on
// every backend measured (#211), which at 256 threads is 16.7 M elements, and a
// video VAE's first level passes that on a 256x256 reference with five frames.
// `ops/pad` solves the same problem with a `stride_y` uniform; this way needs no
// caller to change.

struct Params {
  dim0: u32,
  dim1: u32,
  D: u32,
}

@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;
@group(0) @binding(2) var<uniform> params: Params;

const WORKGROUP_SIZE: u32 = 256u;

@compute @workgroup_size(256)
fn main(
  @builtin(global_invocation_id) gid: vec3<u32>,
  @builtin(num_workgroups) workgroups: vec3<u32>,
) {
  // Unlike `ops/gqa`'s kernels (dispatched at an exact `[L, H, B]` so every
  // invocation has real work, rule 1), this dispatch is sized by
  // `ceil(total / WORKGROUP_SIZE)` because `total` is rarely a multiple of
  // it — so a guard is reachable here and has to be tested, not just
  // written (`ops/permute/wgsl.test.ts`'s non-multiple-of-256 case).
  let idx = gid.x + gid.y * workgroups.x * WORKGROUP_SIZE;
  let total = params.dim0 * params.dim1 * params.D;
  if (idx >= total) {
    return;
  }
  // `input` is `[dim0, dim1, D]`, so `idx` decomposes directly into its
  // three indices without needing `input`'s own strides again.
  let d = idx % params.D;
  let rest = idx / params.D;
  let i1 = rest % params.dim1;
  let i0 = rest / params.dim1;
  output[(i1 * params.dim0 + i0) * params.D + d] = input[idx];
}
