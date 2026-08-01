# web-xpu-ops

Web primitive layer implementations for xPU.

One **reference** per op, one implementation per **backend**, and tests that hold
them together.

```
ops/rmsnorm/
  reference.ts        what correct means — plain, slow, obviously right
  wgsl/kernel.wgsl    the WGSL implementation
  wgsl.test.ts        runs it on a real GPU and compares
```

The reference is the point. Backends multiply; correctness does not. A kernel is
only ever measured against the reference, never against another kernel, so
adding a backend cannot quietly redefine what the op does.

## Why

Hand-written kernels are usually tested by whether the model still produces
sensible output. That is a real signal, and it is not enough to change them by:
it cannot tell a correct optimisation from a subtly wrong one, and it says
nothing about the edge cases real inputs never reach.

These ops came out of [0xBitNet](https://github.com/m96-chan/0xBitNet), where
they worked but had never been executed by a test — the suite exercised
TypeScript references that mirrored the shaders, with nothing holding the two
together.

## Ops

| op | backends | notes |
| --- | --- | --- |
| `rmsnorm` | wgsl | workgroup reduction; `eps` guards an all-zero row |
| `softmax` | wgsl | max-subtracted, so real logits do not overflow `exp` |
| `activation` | wgsl | `relu2` and `silu` |
| `elementwise` | wgsl | `add` and `multiply` |
| `rope` | wgsl | rotary position embedding, with KV-cache offset |
| `quantize` | wgsl | per-row absmax to int8, symmetric `[-127, 127]` |
| `dequantize` | wgsl | applies both the weight and the activation scale |

WASM and WebNN are the intended next backends. WebNN will not look like the
others — it builds an `MLGraphBuilder` graph rather than shipping a kernel — so
its entry per op is a mapping, not a file of source.

## Running the tests

```bash
npm install
npm test
```

They need a GPU. Without one they **skip rather than fail**, so a machine with no
adapter still reports a passing suite instead of a wall of red nobody can act on.

## Tolerances

Comparisons are agreement, not equality: the reference runs in f64, the kernels
in f32. An element passes on either relative or absolute difference, because
relative alone is meaningless where a result nears zero through cancellation and
absolute alone is meaningless where the values are large.

Where a tolerance is loosened, the reason is measured and recorded next to it.
`rope` is the standing example — this GPU's `sin` and `cos` carry up to **1.86e-4**
of absolute error, three orders of magnitude worse than f32 epsilon (1.2e-7), and
`rope` calls both per element. `pow` was measured separately at 2.8e-7, so the
transcendentals account for all of it. No shader using `sin`, `cos` or `exp` can
be checked tighter than its hardware allows.

## Notes for anyone extending this

The GPU binding is a native module and does not survive vitest recycling its
workers — it aborts with `std::system_error`. One device is created for the whole
run and destroyed at the end; `vitest.config.ts` pins the pool accordingly. If
you split the suite differently, expect to revisit that.

## License

MIT
