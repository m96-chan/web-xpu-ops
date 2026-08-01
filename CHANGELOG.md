# Changelog

Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/); versions
follow [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

Entries record **why** a change was needed. What changed is in the diff.

## [Unreleased]

### Changed

- `npm test` runs one test file per vitest process. A single process cannot cross
  a test-file boundary with a GPU device in play — it aborts inside Dawn's thread
  pool or hangs, with no kernel of its own required to trigger it. The runner also
  refuses to report a false pass: a crashed vitest worker can exit 0 having
  skipped most of the suite, so every file is now accounted for individually.

### Added

- `harness` — a WGSL runner over Dawn in Node, an `agree` comparator that passes
  an element on either relative or absolute difference, and a suite helper that
  creates exactly one GPU device per run. The comparator is not equality because
  the references run in f64 and the kernels in f32; the single device is because
  the GPU binding is a native module that does not survive vitest recycling its
  workers.
- `rmsnorm` — workgroup tree reduction, with `eps` guarding an all-zero row.
- `softmax` — max-subtracted, so real logits do not overflow `exp`.
- `activation` — `relu2` and `silu`.
- `elementwise` — `add` and `multiply`.
- `rope` — rotary position embedding, with a KV-cache offset.
- `quantize` — per-row absmax to int8, symmetric over `[-127, 127]`.
- `dequantize` — applies both the weight scale and the activation scale.
- `matvec` (GEMV) — one vector against a `[M, K]` row-major matrix, following
  `torch.mv` rather than BLAS `sgemv`: no `alpha`, no `beta`, no transpose flag.
  It exists as its own op rather than as a path through a future `matmul`
  because it reads every weight exactly once and reuses none of them, so the
  kernel is written to stream — lanes walk a row at the workgroup stride, which
  keeps each pass one contiguous burst — and the tiling that makes GEMM fast has
  nothing to capture here. Autoregressive decoding is this shape at every step.
  Speed is **unmeasured**: the bandwidth roofline it should be reported against
  does not exist yet.
- `matmul` — GEMM (`C = A @ B`) with a shared-memory tiled WGSL kernel. Separate
  from GEMV because the reuse a tile buys is the only reason this shape can be
  compute-bound, and none of it applies to a batch of one. Shapes that do not
  divide by the tile are where tiled kernels are fast and wrong, so the ragged
  tails are tested on their own and together — including against buffers longer
  than the operands, because this device reads past the end of a buffer as zero
  and would otherwise hide a missing tail guard.
- `scatter` — indexed writes where **colliding indices accumulate**, via an f32
  compare-exchange atomic. The rule had to be decided rather than discovered:
  "last write wins" is undefined behaviour on a GPU, since nothing orders the
  threads reaching a slot, and callers would have built on whichever answer
  their own device happened to give. Accumulation is the only rule that is the
  same for every ordering, and it is what gradient accumulation, MoE dispatch
  and bincount need. Matches `scatter_add_` in PyTorch, not `scatter_`.
- `transpose` — 2D, staged through a 16x16 workgroup tile. The tile is there
  because transpose computes nothing: the only thing it can get wrong is where a
  value lands, and the only thing it can be slow at is reaching memory. Turning
  the tile inside the workgroup keeps both the read and the write consecutive,
  which the obvious one-line version does for the read only. Shapes that do not
  divide by 16 are the case that matters — the leftover threads address inside
  the buffer, so an unguarded write replaces a real value instead of faulting.
- `reduce` — `sum` / `max` / `min` / `mean` along one axis, over a tensor viewed
  as `[outer, axis, inner]` so that any axis of any rank fits. rmsnorm and
  softmax each carry their own copy of the workgroup tree reduction, and a third
  copy was about to be written; this is that reduction with the combiner and the
  identity lifted out. The edges follow PyTorch and are stated rather than
  implied: an empty axis sums to `0`, means to `NaN`, and is an error for `max`
  and `min`; `mean` always divides by the axis length. Callers who get those
  wrong get them wrong quietly, which is why they are written down.
- `layernorm` — mean-subtracted normalisation with a bias term, following
  `torch.nn.functional.layer_norm`: the variance is **biased** (`1/D`, not
  `1/(D-1)`) and `eps` sits inside the square root. Both were checked against
  PyTorch in float64 rather than read off the documentation, because the two
  variance conventions differ by less than a percent on a wide row and a caller
  would never notice which one they got. The variance is computed in two passes
  — mean, then the mean of the squared deviations — instead of the one-pass
  `E[x²] - E[x]²`. That identity is not a performance choice with a numerical
  footnote; it is broken where LayerNorm is most often used. Measured on this
  device by running both shaders: on a row of `8192 ± 4`, true variance 6.678162,
  the two-pass kernel returns 6.67816 and the one-pass identity returns a
  *negative* number — the squares land past 2^26 where f32 steps by 8, the
  subtraction cancels every digit the spread had, and `inverseSqrt` of that is
  NaN. The same test at `1024 ± 4` is the more dangerous half: 6.62347 against
  6.678162, 0.8% low, a number nobody would look at twice. Both rows are in the
  suite, and with the one-pass form in place they are the only test of the eight
  that fails — which is the whole reason they had to be written.
- `gather` — row selection for embedding lookup, matching
  `torch.index_select(table, 0, indices)` rather than `torch.gather`, because
  embedding lookup is why the op exists and the two names are close enough to
  pick the wrong one by accident. An index outside `[0, rows)` gathers zeros:
  PyTorch raises there and a kernel cannot, and the alternatives — clamping or
  wrapping — hand back a real embedding for a token that was never in the
  vocabulary, which looks plausible all the way downstream.
- `alibi` and `pope` — two more position encodings, so that position encoding is
  a family in this repository rather than a synonym for `rope`. Models pick
  differently and none of the three can be substituted for another: `rope`
  rotates Q and K, `alibi` biases the attention scores, `pope` builds a table
  that is added to the embeddings. They run on different tensors at different
  points in the layer.

  `alibi` ships as two kernels because its two halves fail differently. The
  per-head slopes are where implementations quietly disagree, and the
  disagreement is invisible at 8 heads: for a power-of-two head count everyone
  produces the same geometric run, and for anything else the paper appends
  every other slope of the *next* run rather than interpolating — so the
  sequence is not monotonic, and an implementation that sorts or truncates is
  wrong only at head counts nobody tests. The slopes therefore have their own
  kernel, their own GPU test, and a reference test that pins the published
  numbers, so a fault in the slopes cannot hide inside a correct bias and the
  convention is checked against the paper rather than against the kernel. The
  bias itself follows the paper's relative form, `m * (j - i)`; BLOOM's
  `m * j` differs by a per-row constant that the following softmax erases, but
  this op returns the tensor and not the softmax, so the two are not
  interchangeable here. Masking is left to the caller — writing `-inf` above
  the diagonal would fold a masking policy into a bias op.

  `pope` records where its paper is silent instead of guessing. The polynomial
  order is the token position and the argument sweeps the feature index across
  `[-1, 1)`, which is the paper's Equation (14); whether positions start at 0
  or 1 is not stated, and it matters, because at order 0 the polynomial is the
  constant 1 and the first token would carry no position at all. That is a
  required `posOffset` argument rather than a default, the same way `rope`
  takes one. It is evaluated by the three-term recurrence, not by Rodrigues'
  formula, and the reason an f32 kernel can walk the recurrence is that
  `|P_n(x)| <= 1` holds on the domain — measured, not assumed: the f32
  recurrence sits 4.1e-6 from f64 at order 70 and 1.6e-5 at order 128, so the
  tolerance is stated for the order range the tests reach and not beyond it.

  Speed is **unmeasured** for both; the roofline to report against does not
  exist yet.

[Unreleased]: https://github.com/m96-chan/web-xpu-ops/compare/main...HEAD
