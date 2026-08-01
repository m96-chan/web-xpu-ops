# web-xpu-ops

Web primitive layer implementations for xPU.

One **reference** per op, one implementation per **backend and target**, and
tests that hold them together.

---

## Mission

The browser is becoming a heterogeneous machine. WebGPU is here, WebNN is
arriving, WASM SIMD is everywhere, and the silicon underneath is a CPU, a GPU,
and increasingly an NPU that all want different code for the same operation.

Nothing today lets you write an op once and answer the question *"is this fast
on the thing it is actually running on?"* — and that question is the whole job
at this layer. This is where the answer should live.

Not as fast as hand-tuned CUDA. That is not the bar. The bar is: **on the
hardware in front of the user, how close to that machine's ceiling are we, and
does anyone notice when it regresses?**

---

## What exists today

Eight ops, WGSL only, verified against their references on a real GPU.

| op | notes |
| --- | --- |
| `matvec` | GEMV; `torch.mv` convention, streaming rather than tiled. Speed unmeasured |
| `rmsnorm` | workgroup reduction; `eps` guards an all-zero row |
| `softmax` | max-subtracted, so real logits do not overflow `exp` |
| `activation` | `relu2`, `silu` |
| `elementwise` | `add`, `multiply` |
| `rope` | rotary position embedding, with KV-cache offset |
| `quantize` | per-row absmax to int8, symmetric `[-127, 127]` |
| `dequantize` | applies both the weight and the activation scale |

Everything below this line is design, not code. It is written down so the shape
is decided before there is enough built for the shape to be hard to change.

---

## The reference is the point

```
ops/rmsnorm/
  reference.ts        what correct means — plain, slow, obviously right
  wgsl/kernel.wgsl
  wgsl.test.ts
```

Backends multiply; correctness does not. A kernel is only ever measured against
the reference, never against another kernel. That is what stops a second backend
from quietly redefining an op to match whatever it happens to compute, and it is
why the reference is deliberately the slowest expression of the maths — its job
is to be obviously right.

---

## Targets

The same WGSL is not the same speed everywhere, and sometimes it should not be
the same WGSL. Workgroup size, tiling, whether subgroup operations exist, whether
`f16` exists at all — these differ per target, and a single kernel that is
adequate everywhere is usually good nowhere.

Planned axis:

```
ops/matmul/
  reference.ts
  wgsl/
    kernel.wgsl           portable, correct, unremarkable
    apple.wgsl            unified memory, wide subgroups
    nvidia.wgsl
    amd.wgsl
    soc.wgsl              tight power and bandwidth budgets
  webnn/graph.ts
  wasm/kernel.ts
```

Selection resolves in order, first hit wins:

```
explicit override  →  target + dtype  →  target  →  dtype  →  portable
```

**Overrides are a first-class input, not an escape hatch.** Anyone integrating
this will eventually know something the library cannot — that their sequence
length is always 1, that their weights are static, that they would rather have
lower peak memory than higher throughput. Refusing that knowledge makes the
library something to work around.

Target detection has to be honest about how little is knowable. `adapter.info`
gives a vendor and an architecture string and not much else, so detection is a
hint, and the override exists partly because the hint will sometimes be wrong.

---

## Dtypes

`f32` is the floor and always works. Everything above it is conditional:

| dtype | availability |
| --- | --- |
| `f32` | always |
| `f16` | requires the `shader-f16` feature |
| `i32` / `u32` | always |
| packed int8 / int4 / ternary | manual bit packing; no native type |

The `f16` line is not theoretical. Measured during this project's own work:

- Apple M3 — `shader-f16` present, `q4f16` selected, and **still slower than real
  time** for the workload under test.
- Linux / NVIDIA with Chrome's Vulkan backend disabled — Dawn falls back to an
  ANGLE compatibility adapter advertising exactly one feature. No `shader-f16`
  at all.

So `f16` is not a switch to flip for a free win. It is a per-target, per-op
question with an answer that has to be measured, and the availability check is
load-bearing rather than defensive.

Integer paths matter more here than in a datacentre library: quantized weights
are how a model fits down a browser's throat at all.

---

## GEMM and GEMV are separate ops

Not two paths through one kernel. Two kernels.

GEMV — one vector against a matrix — reads every weight exactly once and reuses
nothing. It is **bandwidth-bound**, and the only thing that matters is how fast
the weights can be pulled through. Tiling for reuse is wasted code, because there
is no reuse to find.

GEMM has reuse proportional to the tile size and is **compute-bound** once the
tiling is right. Everything that makes GEMM fast is irrelevant to GEMV, and the
work of keeping both inside one kernel is spent on branches that make each of
them worse.

Autoregressive decoding is GEMV-shaped: batch of one, every step. Prefill is
GEMM-shaped. A library that only does one of them well is only good at half of
inference.

---

## Performance, measured against the machine's own ceiling

Correctness tests answer *does it work*. At this layer the other half of the job
is *is it fast*, and nothing catches a regression there today.

**Not against a spec sheet.** WebGPU exposes no clock, no compute-unit count, no
memory bus width — there is nothing to compute a theoretical peak *from*, and a
number derived from a marketing figure would be worse than no number, because it
would look authoritative.

So the ceiling is measured on the same device, in the same session:

- **Bandwidth roofline** — a kernel that does nothing but stream memory. The best
  bytes-per-second that device will give anyone.
- **Compute roofline** — a kernel that does nothing but fused multiply-adds out
  of registers. The best FLOPs that device will give anyone.

Every op then reports what fraction of the relevant roofline it reached, and
which of the two it is bound by. `rmsnorm` hitting 78% of measured bandwidth is a
statement about the kernel. "412 GFLOP/s" is a statement about the GPU someone
happened to run it on.

This also makes CI results comparable across wildly different hardware, which is
the only way a browser-targeting library can have meaningful performance CI at
all. A percentage regresses visibly on a laptop and a workstation alike; an
absolute number regresses invisibly on both.

---

## Op roadmap

Three layers, by what they are rather than by what they compute.

### `primitive/` — the algebra

`matmul` (GEMM) · `matvec` (GEMV) ✅ · `conv` · `add` · `mul` · `gather` ·
`scatter` · `transpose` · `reduce`

Small, total, boring. Everything else is built from these, and they are where
target-specific tuning pays off most.

### `kernel/` — one fused, named operation

`rope` ✅ · `rmsnorm` ✅ · `layernorm` · `softmax` ✅ · `activation` ✅ ·
`elementwise` ✅ · `quantize` ✅ · `dequantize` ✅ · `attention` ·
`flash_attention` · `ctc_decode` · `mel` · `stft` / `istft`

Fusion is the reason this layer exists rather than being composed from
`primitive/` at call time. `flash_attention` is not `matmul` + `softmax` +
`matmul`; it is the one that never writes the score matrix to memory, which is
the entire point of it.

`mel`, `stft` and `istft` are here because speech pipelines need them and no ML
kernel library ships them — they are DSP, so everyone assumes someone else has
them. The inverse STFT in particular is the thing ONNX cannot express, being
unable to carry complex tensors.

### `attention/` — the variants that are their own problem

Position: `RoPE` ✅ · `ALiBi` · `PoPE` · `YaRN` · `NTK scaling` · `rotary cache`

Sharing: `GQA` · `MQA`

Routing: `MoE router` · `MoE dispatch` · `MoE gather`

Serving: `paged KV cache` · `speculative decode`

Separated from `kernel/` because these are not variations in arithmetic, they are
variations in *what memory gets touched*. Paged KV cache and speculative decoding
are not faster attention; they are different answers to where the state lives.

### `block/` — deliberately not here yet

Transformer, conformer, U-Net, decoder blocks are **compositions**, not kernels.
They belong to whoever is building a model, and a kernel library that ships them
starts making architectural decisions on its users' behalf.

The exception is a block that only pays off fused — where crossing it as separate
dispatches costs more than the arithmetic does. If one of those turns up, it
belongs in `kernel/` under its own name, not in a `block/` directory that invites
everything else in behind it.

---

## Backends

| backend | status | shape |
| --- | --- | --- |
| WGSL | 7 ops | a kernel per op, per target |
| WASM | planned | a kernel per op, SIMD where available |
| WebNN | planned | **not a kernel** — an `MLGraphBuilder` graph, so the entry is a mapping |

WebNN is worth calling out because it breaks the symmetry. You do not hand it a
kernel; you describe a graph and the browser compiles it for whatever accelerator
it has. So an op's WebNN entry says *which operators this decomposes into*, and
the tuning story is entirely different — there is nothing to tune, only a choice
of decomposition.

[NxPU](https://github.com/m96-chan/NxPU) is the fourth direction: compiling the
WGSL here to CoreML, TFLite and StableHLO. The reference in each op is what would
prove such a transpilation preserved meaning.

---

## Running the tests

```bash
npm install
npm test
```

They need a GPU. Without one they **skip rather than fail**, so a machine with no
adapter reports a passing suite instead of a wall of red nobody can act on.

`npm test` runs **one test file per vitest process** (`scripts/test.mjs`) rather
than calling `vitest run` once. `npm run test:file <path>` is the direct escape
hatch while working on a single op.

## Tolerances

Comparisons are agreement, not equality: the reference runs in f64, the kernels
in f32. An element passes on either relative or absolute difference, because
relative alone is meaningless where a result nears zero through cancellation, and
absolute alone is meaningless where the values are large.

Where a tolerance is loosened, the reason is measured and recorded beside it.
`rope` is the standing example — this GPU's `sin` and `cos` carry up to **1.86e-4**
of absolute error, three orders of magnitude worse than f32 epsilon (1.2e-7), and
`rope` calls both per element. `pow` was measured separately at 2.8e-7, so the
transcendentals account for all of it. No shader using `sin`, `cos` or `exp` can
be checked tighter than its hardware allows.

## Notes for anyone extending this

The GPU binding is a native module and does not survive vitest recycling its
workers — it aborts with `std::system_error`. One device is created per test
file and destroyed when that file's process exits; `vitest.config.ts` pins the
pool accordingly.

**A single vitest process cannot cross a test-file boundary with a GPU device in
play.** It dies with a glibc assertion out of Dawn's thread pool
(`pthread_mutex_lock`, `__pthread_tpp_change_priority`), with `std::system_error`,
or it hangs. It is not about any kernel: copying an existing op directory to a
new name reproduces it with no new WGSL, and two files are enough. What was
measured and did **not** fix it:

- moving teardown from `afterAll` to `process.once("exit")`, so one device
  genuinely lives for the whole run — this made it *worse*, dying at the first
  boundary
- `pool: "forks"` with `isolate: true`, a process per file
- `pool: "forks"` with `maxForks: 1` and `fileParallelism: false`
- `pool: "threads"` with `singleThread: true`

The native module is not the cause either — it is evaluated exactly once, which
was checked by counting evaluations rather than assumed. Sixteen create/destroy
cycles *inside* one file are fine. A full `/tmp` was ruled out too: the failure
reproduces unchanged with a disk-backed `TMPDIR`.

So `npm test` gives each file its own process. That also removes a worse hazard:
a crashed vitest worker prints `Test Files 1 passed (2)` and can still exit 0, so
the files that never ran look like files that passed. The runner accounts for
every file and fails the run if any is unaccounted for. It retries a file **only**
when vitest produced no summary at all — an infrastructure crash, where nothing
was learned — and announces the retry. A file whose tests ran and failed is never
retried, because that would launder a broken kernel into a green suite.

Measured after the change: 10 consecutive full runs green, one announced retry
across all ten. Issue #38 has the bisection.

## License

MIT
