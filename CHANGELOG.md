# Changelog

Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/); versions
follow [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

Entries record **why** a change was needed. What changed is in the diff.

## [Unreleased]

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
- `reduce` — `sum` / `max` / `min` / `mean` along one axis, over a tensor viewed
  as `[outer, axis, inner]` so that any axis of any rank fits. rmsnorm and
  softmax each carry their own copy of the workgroup tree reduction, and a third
  copy was about to be written; this is that reduction with the combiner and the
  identity lifted out. The edges follow PyTorch and are stated rather than
  implied: an empty axis sums to `0`, means to `NaN`, and is an error for `max`
  and `min`; `mean` always divides by the axis length. Callers who get those
  wrong get them wrong quietly, which is why they are written down.

[Unreleased]: https://github.com/m96-chan/web-xpu-ops/compare/main...HEAD
