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

Thirteen ops, WGSL only, verified against their references on a real GPU.

| op | notes |
| --- | --- |
| `matvec` | GEMV; `torch.mv` convention, streaming rather than tiled. Speed unmeasured |
| `rmsnorm` | workgroup reduction; `eps` guards an all-zero row |
| `layernorm` | two workgroup reductions; **biased** variance (`1/D`) and `eps` inside the `sqrt`, as `torch.nn.functional.layer_norm`. Speed unmeasured |
| `softmax` | max-subtracted, so real logits do not overflow `exp` |
| `activation` | `relu2`, `silu` |
| `elementwise` | `add`, `multiply` |
| `rope` | rotary position embedding, with KV-cache offset and NTK / YaRN context scaling. Follows `jquesnelle/yarn` and `transformers`, which agree; YaRN's attention temperature is included |
| `alibi` | linear attention-score bias (arXiv:2108.12409); slopes follow the paper's own `get_slopes`, including the **non-monotonic** appended tail for head counts that are not a power of two. Bias is the paper's relative form `m * (j - i)`, not BLOOM's `m * j`; masking is the caller's. Speed unmeasured |
| `pope` | Legendre polynomial position table (arXiv:2405.04585, Eq. 14); order is the position, argument sweeps `[-1, 1)`. `posOffset` is required because the paper does not say whether positions start at 0 or 1. Speed unmeasured |
| `quantize` | per-row absmax to int8, symmetric `[-127, 127]` |
| `dequantize` | applies both the weight and the activation scale |
| `matmul` | GEMM; `torch.mm` convention, shared-memory tiling. Speed unmeasured |
| `transpose` | turned through workgroup memory so both read and write stay consecutive |
| `reduce` | `sum` / `max` / `min` / `mean` along an axis |
| `gather` | row selection, as `torch.index_select(table, 0, indices)` — not `torch.gather`; an out-of-range index gathers zeros |
| `scatter` | indexed writes; **colliding indices accumulate** — see below |
| `stft` / `istft` | `torch.stft` / `torch.istft` conventions: centred, reflect padding, one-sided, unnormalised, periodic Hann; `istft` divides by the `w²` envelope — see below. Speed unmeasured |
| `conv` | 1D only, as `torch.nn.functional.conv1d` — a **cross-correlation**, so the kernel is *not* flipped; `stride` / `padding` / `dilation` / `groups` / optional `bias`. Speed unmeasured |
| `attention` | unfused SDPA in two dispatches; `torch.nn.functional.scaled_dot_product_attention` convention — `scale` is `1/sqrt(D)` from the query's head dim, and `causal` is upper-left aligned (`queryOffset = S - L` gives `causal_lower_right`). Speed unmeasured |
| `ctc_decode` | greedy only. Collapse repeats **then** drop blanks, as `torch.unique_consecutive` + a blank filter does; `blank=0` as in `torch.nn.CTCLoss`. Lengths are written by the kernel, so nothing reads back |
| `flash_attention` | the same function as `attention`, one dispatch, tiled online softmax; the `[B, H, L, S]` score matrix is never allocated, which is tested by counting bound bytes and not only by the answer. `128n + 32` bytes at `L = S = n, D = Dv = 8` against unfused `4n² + 64n + 28`. Speed unmeasured |
| `mel` | filterbank construction and its application, as two kernels. Defaults are `torchaudio.transforms.MelSpectrogram`: **HTK** mel scale, **unnormalised** triangles, **power** spectrum, and `AmplitudeToDB(stype="power")` for the log — base 10, scaled by `20/power`, flooring its *argument* at `1e-10` rather than adding an epsilon. `{ scale: "slaney", norm: "slaney" }` gives **`librosa.filters.mel`**'s defaults instead; on the same audio the two differ by 200x, so neither is a default worth leaving unstated. No `top_db` — it needs a reduction over the whole spectrogram. Speed unmeasured |
| `moe` | MoE routing: router, dispatch, gather. Softmax before top-k with the k gates renormalised or not, as `MixtralSparseMoeBlock` and `norm_topk_prob` (no default: the Switch Transformer must not renormalise at `k = 1`); **top-k ties go to the lower expert index**, which `torch.topk` leaves undefined; capacity overflow drops **by rank, then by token index**, as GShard / Switch / fairseq `top2gating`, not by arrival; the gate is applied in gather and only there. Speed unmeasured |

### `scatter`: colliding indices accumulate

Two slots naming the same target **add**. That is a decision, and it is the one
thing about this op a caller has to know before using it.

The alternative usually offered is "last write wins", and on a GPU that is not a
rule, it is undefined behaviour with a reassuring name: the order threads reach a
slot is unspecified, so *last* means whatever the driver did that day. Callers
would build on whichever answer their first device gave. Accumulation is the only
rule that returns the same answer for every possible ordering, and it is what the
things scatter is actually used for — gradient accumulation, MoE dispatch,
bincount — want anyway. The kernel pays an atomic per write for it.

It matches `torch.zeros(N, D).scatter_add_(1, index, src)`, deliberately **not**
`scatter_`, which PyTorch itself documents as non-deterministic on collision.
Three departures from PyTorch, spelled out in `ops/scatter/reference.ts`: `self`
is implicitly zero, out-of-range indices are dropped rather than raising, and
`index` is i32 because WGSL has no 64-bit integer.

What stays order-dependent is the last bit or two of a collided sum — f32
addition is not associative. Measured on this GPU at 75 collisions per slot: up
to 4.1e-7 relative, about three f32 epsilons, which is the tolerance those tests
are set from.

### `stft` / `istft`: the conventions, and what they match

Window, hop, centring, padding and normalisation each have more than one
reasonable answer. Choosing silently means every caller has a 50% chance of a
subtly wrong waveform, so all of them are named here and every one follows
**`torch.stft` / `torch.istft`**, checked against torch 2.10 numerically rather
than read off the documentation:

| thing | here | matches |
| --- | --- | --- |
| centring | `center = true`: frame `f` centres on sample `f * hop` | `torch.stft(center=True)` |
| padding | reflect, without repeating the edge sample | `pad_mode="reflect"` |
| frames | `1 + floor(L / hop)` centred | torch |
| sidedness | one-sided, `floor(nFft / 2) + 1` bins | `onesided=True` |
| scaling | none forward, `1 / nFft` inverse | `normalized=False` |
| `hannWindow` | periodic | `torch.hann_window`, `scipy.signal.get_window("hann")`, librosa — **not** `np.hanning`, which is symmetric |
| Nyquist bin | counted once, imaginary part dropped | `torch.fft.irfft` |

**`istft` divides by the overlap-added `w²` envelope** rather than assuming the
window is COLA. That inverts any window satisfying NOLA — the same least-squares
inverse torch computes — and it matters for the ordinary case: a periodic Hann at
50% overlap is COLA in `w` but **not** in `w²`, since `sin⁴θ + cos⁴θ` runs
between 0.5 and 1. Skipping the division is wrong by up to 2x and still sounds
like audio. The reference refuses a window whose envelope drops below `1e-11`,
which is torch's own threshold, bracketed by bisection against it.

Two deliberate departures, both explained in `ops/stft/reference.ts`: the layout
is frame-major `[frames, bins]` where torch is `[bins, frames]`, because a
vocoder head emits one row per frame — MioCodec's `istft_head` is a
`Linear[.., 1922]` with `1922 = 2 * (1920 / 2 + 1)` — and because it lets the
kernel write consecutively. And asking for more output samples than the frames
reach raises, where torch pads the tail with zeros and warns; a silent zero tail
is indistinguishable from silence in a vocoder output.

The kernels are a naive DFT per frame, not an FFT. 1920 is `2^7 * 15`, so radix-2
does not apply, and this runs beside a transformer over a few hundred frames.
Speed is **unmeasured**.

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

`matmul` (GEMM) ✅ · `matvec` (GEMV) ✅ · `conv` ✅ (1D) · `add` · `mul` · `gather` ✅ ·
`scatter` ✅ · `transpose` ✅ · `reduce` ✅

Small, total, boring. Everything else is built from these, and they are where
target-specific tuning pays off most.

### `kernel/` — one fused, named operation

`rope` ✅ · `rmsnorm` ✅ · `layernorm` ✅ · `softmax` ✅ · `activation` ✅ ·
`elementwise` ✅ · `quantize` ✅ · `dequantize` ✅ · `attention` ✅ ·
`flash_attention` ✅ · `ctc_decode` ✅ (greedy) · `mel` ✅ · `stft` / `istft` ✅

Fusion is the reason this layer exists rather than being composed from
`primitive/` at call time. `flash_attention` is not `matmul` + `softmax` +
`matmul`; it is the one that never writes the score matrix to memory, which is
the entire point of it.

`mel`, `stft` and `istft` are here because speech pipelines need them and no ML
kernel library ships them — they are DSP, so everyone assumes someone else has
them. The inverse STFT in particular is the thing ONNX cannot express, being
unable to carry complex tensors.

### `attention/` — the variants that are their own problem

Position: `RoPE` ✅ · `ALiBi` ✅ · `PoPE` ✅ · `YaRN` ✅ · `NTK scaling` ✅ · `rotary cache`

Sharing: `GQA` · `MQA`

Routing: `MoE router` ✅ · `MoE dispatch` ✅ · `MoE gather` ✅

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
