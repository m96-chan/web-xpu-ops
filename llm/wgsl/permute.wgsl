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
// Not an `ops/` op: it has no arithmetic and nothing to be "correct" about
// beyond "which index goes where" — a permutation, in the same allowed
// category as the KV-cache copies `llm/engine-q8-resident.ts`'s own doc
// already uses for the equivalent reason (issue #110: "必要な並べ替えは小さな
// GPUコピーカーネルで(新規「演算」カーネルは書かない。並べ替え専用は可)"). Its
// correctness gate is `llm/permute.wgsl.test.ts`, which checks it against
// `splitHeadsMajor`/`mergeHeadsMajor` directly rather than against a
// separate reference of its own — those two functions *are* the reference,
// proven correct already by every engine test that exercises prefill.

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
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  // Unlike `ops/gqa`'s kernels (dispatched at an exact `[L, H, B]` so every
  // invocation has real work, rule 1), this dispatch is sized by
  // `ceil(total / WORKGROUP_SIZE)` because `total` is rarely a multiple of
  // it — so a guard is reachable here and has to be tested, not just
  // written (`llm/permute.wgsl.test.ts`'s non-multiple-of-256 case).
  let idx = gid.x;
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
