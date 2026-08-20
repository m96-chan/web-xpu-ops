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

Twenty-seven ops, WGSL only, verified against their references on a real GPU.

**Speed is unmeasured for every one of them.** The roofline each would be
reported against does not exist yet, and a number without one would be a
statement about this GPU rather than about the kernel — so the column says so
rather than being left blank.

| op | notes |
| --- | --- |
| `matvec` | GEMV; `torch.mv` convention, streaming rather than tiled. Speed unmeasured |
| `matvecQ8` | W8A32 GEMV: `matvec` with the weight held as int8 instead of f32. Weight is `[N, ceil(K/4)]` `u32`, four codes packed per word least-significant byte first — the layout a little-endian host gets for free by viewing an `Int8Array`'s buffer as `Uint32Array`; scale is `[N]`, `quantize`'s per-row absmax convention, applied once per row after the dot product rather than per term. `packQ8` packs `quantize`'s `Int32Array` codes into this layout; the two compose (`packQ8(quantize(w).output, N, K)` + `quantize(w).scales`) rather than this op quantizing on its own. Speed unmeasured |
| `rmsnorm` | workgroup reduction; `eps` guards an all-zero row. An optional per-group `weight` of `[G, D]` for QK-norm — row `n` takes group `n % G`, so the grouped axis has to be the one just left of `D` (`[B, S, H, Dh]`, not `[B, H, S, Dh]`); `G = 1` is the single shared gamma. Reduction stays over `D` alone, matching `F.rms_norm(x, (D,)) * w` rather than `torch.nn.RMSNorm((H, Dh))`, which reduces over both. Speed unmeasured |
| `layernorm` | two workgroup reductions; **biased** variance (`1/D`) and `eps` inside the `sqrt`, as `torch.nn.functional.layer_norm`. Speed unmeasured |
| `group_norm` | `torch.nn.functional.group_norm`: statistics pooled over a **group of channels**, affine applied **per channel**. For `[N, C, L]` each of the `N × G` groups reduces `(C/G) × L` values, so this is not `layernorm` with a longer row — that gives one mean per channel, and the two disagree with no error and no change of shape. `G = 1` normalises the whole sample, `G = C` is InstanceNorm. Biased variance and `eps` inside the `sqrt`, both measured against torch 2.10 rather than inherited from `layernorm` (the unbiased form is out by 1.6e-1, `eps` outside the root by 4.3e-6). `C % G != 0` throws, as torch does. Speed unmeasured |
| `softmax` | max-subtracted, so real logits do not overflow `exp` |
| `activation` | `relu2`, `silu`, `elu`, `tanh`, `gelu`, `gelu_tanh`. `elu`'s `alpha` is a scalar hyperparameter defaulting to 1.0, as `torch.nn.ELU`. GELU is two functions, not one: the default `gelu` is the exact `erf` form (`torch.nn.functional.gelu`'s own default, `approximate="none"`) and `gelu_tanh` is `approximate="tanh"` — they differ by up to **4.73e-4, at x = 2.699**, so neither is a silent stand-in for the other. Speed unmeasured |
| `snake` | Two entry points, because the name covers two functions. `kernel`: `x + sin²(α·x)/α` with a **learned per-channel** α (arXiv:2006.08195), as `Snake1d` in DACVAE. `beta`: `x + sin²(α·x)/β` with **both** learned per channel, as BigVGAN's `SnakeBeta` and MioCodec's decoder — β = α recovers the first. A checkpoint's `alpha` tensor does not say which it belongs to, and the BigVGAN family stores **logarithms** while DACVAE stores values; neither kernel exponentiates, because doing so would be wrong for the other. The epsilon is upstream's, inside the reciprocal and at upstream's value, guarding the **divisor** — α in the first, β in the second. `sin²` is the square of the sine. Its own op rather than an `activation` kind, because α is a buffer and a channel stride rather than a scalar — the reason is written out in `ops/snake/reference.ts`. Speed unmeasured |
| `elementwise` | `add`, `multiply` |
| `rope` | rotary position embedding, with KV-cache offset and NTK / YaRN context scaling. Follows `jquesnelle/yarn` and `transformers`, which agree; YaRN's attention temperature is included. An optional precomputed angle table (`ropeCache`); past its end the angle is recomputed rather than wrapped. `headOffset` / `headCount` rotate a subset of the **heads** and copy the rest through — the axis Irodori-TTS's `_apply_rotary_half` uses (`chunk(2, dim=-2)`), not channel-wise partial rotary (`rotaryDim`), which is the other thing "half-RoPE" is used to mean and is not implemented. Speed unmeasured |
| `alibi` | linear attention-score bias (arXiv:2108.12409); slopes follow the paper's own `get_slopes`, including the **non-monotonic** appended tail for head counts that are not a power of two. Bias is the paper's relative form `m * (j - i)`, not BLOOM's `m * j`; masking is the caller's — and `attention`'s `mask` is an additive bias of exactly this shape (`maskShape: [1, H, L]`), so the two compose by addition. Speed unmeasured |
| `pope` | Legendre polynomial position table (arXiv:2405.04585, Eq. 14); order is the position, argument sweeps `[-1, 1)`. `posOffset` is required because the paper does not say whether positions start at 0 or 1. Speed unmeasured |
| `quantize` | per-row absmax to int8, symmetric `[-127, 127]` |
| `dequantize` | applies both the weight and the activation scale |
| `matmul` | GEMM; `torch.mm` convention, shared-memory tiling. Speed unmeasured |
| `transpose` | turned through workgroup memory so both read and write stay consecutive |
| `reduce` | `sum` / `max` / `min` / `mean` along an axis |
| `gather` | row selection, as `torch.index_select(table, 0, indices)` — not `torch.gather`; an out-of-range index gathers zeros |
| `scatter` | indexed writes; **colliding indices accumulate** — see below |
| `stft` / `istft` | `torch.stft` / `torch.istft` conventions: centred, reflect padding, one-sided, unnormalised, periodic Hann; `istft` divides by the `w²` envelope — see below. `istft` also takes `padding: "same"`, the Vocos / X-Codec-2 / MioCodec vocoder convention that crops `(nFft - hop) / 2` per end so `T` frames give `T * hop` samples — **not a torch mode**, and not composable from one, because the samples `center: false` would return fail NOLA. Speed unmeasured |
| `conv` | 1D only, as `torch.nn.functional.conv1d` — a **cross-correlation**, so the kernel is *not* flipped; `stride` / `padding` / `dilation` / `groups` / optional `bias`. Speed unmeasured |
| `conv_transpose` | 1D only, as `torch.nn.functional.conv_transpose1d` — the decoder half of `conv`, and what a DAC-style codec upsamples with. Weight is **`[Cin, Cout/groups, K]`**, the transpose of `conv`'s layout; `padding` **crops** the output rather than extending the input; `output_padding` lengthens the trailing end only and takes no part in the sum. The kernel is *not* flipped, for the same reason `conv`'s is not. `weight_norm` is an offline conversion, not a flag here. Speed unmeasured |
| `attention` | unfused SDPA in two dispatches; `torch.nn.functional.scaled_dot_product_attention` convention — `scale` is `1/sqrt(D)` from the query's head dim, and `causal` is upper-left aligned (`queryOffset = S - L` gives `causal_lower_right`). `mask` is torch's **float** `attn_mask`, **added** to the scores (`-Infinity` masks), not the boolean one — the additive form is what composes with `alibi`; `keyPaddingBias` converts a boolean mask, whose polarity is torch's `attn_mask` (`true` = attend) and therefore the **reverse** of `nn.MultiheadAttention`'s `key_padding_mask`. Broadcast over batch, heads and query rows via `maskShape`, so `[B,1,1,S]` and `[B,H,L,S]` are both legal. `causal` with `mask` **throws**, as torch does. A fully masked row returns **zeros**, as `aten::_safe_softmax` does and plain `torch.softmax` does not. Speed unmeasured |
| `ctc_decode` | greedy only. Collapse repeats **then** drop blanks, as `torch.unique_consecutive` + a blank filter does; `blank=0` as in `torch.nn.CTCLoss`. Lengths are written by the kernel, so nothing reads back |
| `flash_attention` | the same function as `attention`, one dispatch, tiled online softmax; the `[B, H, L, S]` score matrix is never allocated, which is tested by counting bound bytes and not only by the answer. `132n + 44` bytes at `L = S = n, D = Dv = 8` against unfused `4n² + 68n + 40`. Takes the same additive `mask` as `attention`, on the same terms. Speed unmeasured |
| `mel` | filterbank construction and its application, as two kernels. Defaults are `torchaudio.transforms.MelSpectrogram`: **HTK** mel scale, **unnormalised** triangles, **power** spectrum, and `AmplitudeToDB(stype="power")` for the log — base 10, scaled by `20/power`, flooring its *argument* at `1e-10` rather than adding an epsilon. `{ scale: "slaney", norm: "slaney" }` gives **`librosa.filters.mel`**'s defaults instead; on the same audio the two differ by 200x, so neither is a default worth leaving unstated. No `top_db` — it needs a reduction over the whole spectrogram. Speed unmeasured |
| `moe` | MoE routing: router, dispatch, gather. Softmax before top-k with the k gates renormalised or not, as `MixtralSparseMoeBlock` and `norm_topk_prob` (no default: the Switch Transformer must not renormalise at `k = 1`); **top-k ties go to the lower expert index**, which `torch.topk` leaves undefined; capacity overflow drops **by rank, then by token index**, as GShard / Switch / fairseq `top2gating`, not by arrival; the gate is applied in gather and only there. Speed unmeasured |
| `gqa` | grouped-query **and** multi-query attention: one op, parameterised by `kvHeads` — `kvHeads = 1` is MQA and `kvHeads = H` is `attention` unchanged. **Contiguous** groups (`kvHead = h / (H / kvHeads)`), as `enable_gqa=True` in torch; `H % kvHeads != 0` throws rather than guessing a grouping. What it buys, in bytes: one Llama-3-8B decoder layer (`B=1, S=8192, D=Dv=128`, f32) caches **268,435,456** at `kvHeads=32`, **67,108,864** at `8`, **8,388,608** at `1` — over 32 layers, 32→8 saves **6,442,450,944** bytes. `kvCacheBytes()` computes it. Takes the same additive `mask` as `attention`, with `maskHeads` counting **query** heads. Speed unmeasured |

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

---

## LLM tokenizer

`llm/tokenizer.ts` is not a WGSL op — it never touches a GPU device — but it
lives beside the ops because the LLM engine (#98) that consumes those ops
needs text turned into token ids and back before any of them run. It is a
from-scratch TypeScript reimplementation of **SentencePiece unigram**
encode (Viterbi) / decode: no wasm, no vendored binary, no runtime dependency
on the `sentencepiece` package.

```ts
import { SentencePieceTokenizer } from "web-xpu-ops/llm/tokenizer";

// vocab is whatever llm/tools/export_tokenizer.py produced for your model —
// fetch it, import it as JSON, however your bundler prefers.
const tokenizer = new SentencePieceTokenizer(vocab);
const ids = tokenizer.encode("<|system|>You are Alibi.</s>");
tokenizer.decode(ids); // "<|system|>You are Alibi.</s>"
```

Every normalizer/byte-fallback/special-token behavior is read out of a real
model rather than assumed — this matters because SentencePiece's own common
defaults (NFKC normalization, an implicit leading `▁`) do **not** apply to
every model, including the one this was built against
(Sarashina2.2-1B-Instruct). `llm/tools/export_tokenizer.py` parses
`tokenizer.model`'s protobuf directly and refuses to silently drop a
`normalizer_spec` it does not implement; `llm/tokenizer.ts`'s module doc
records exactly what was verified and how. Ground truth for correctness is
Python `sentencepiece`'s `SentencePieceProcessor`, never `transformers` —
`transformers` 5.3.0 silently converts a UNIGRAM model with no
`tokenizer.json` beside it into an approximate BPE tokenizer that does not
reproduce true unigram segmentation.

`llm/tools/gen_fixtures.py` bakes real-tokenizer encode/decode pairs (80+
cases: Japanese/English/mixed, code, emoji, whitespace patterns, and the
`<|system|>`/`<|user|>`/`</s>`-style chat-template boundaries the engine
emits) that `llm/tokenizer.test.ts` holds this implementation to exactly.

---

## Install

```bash
npm install web-xpu-ops
```

Compiled JavaScript and `.d.ts` ship beside the `.wgsl` kernels, so nothing at
the other end needs a TypeScript toolchain to consume this. ESM only, Node ≥ 20.

## Using it

An op is two things, and they are imported separately because they run in
different places.

The **reference** is plain TypeScript with no dependencies — no GPU, no `node:`
imports — so the same import works in a browser, in Node and in a test runner:

```ts
import { matmul } from "web-xpu-ops/ops/matmul";

const c = matmul({ a, b, M: 64, N: 64, K: 64 }); // Float32Array, [M, N]
```

It is the definition of what the op means, not a fast path. It accumulates in
f64 and is written to be read rather than to be quick, which is the whole reason
it can be trusted as the thing a kernel is checked against. Reach for it to
verify a result, not to produce one at speed.

The **kernel** is the WGSL, published at a path that mirrors the resolution
grammar (`<entry>[.<target>][.<dtype>].wgsl`):

```ts
import code from "web-xpu-ops/ops/matmul/wgsl/kernel.wgsl?raw";
```

Turning a `.wgsl` file into a string is your bundler's job rather than this
package's — `?raw` is Vite's spelling, webpack wants `asset/source`, and fetching
the file at runtime works too. What is promised here is only that the path is
stable and the file is present.

You supply the `GPUDevice`. This library has no opinion about how an application
gets one, and the Node-and-Dawn runner under `harness/` is test infrastructure
rather than a runtime: it imports vitest, so it is deliberately not published.

---

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
    kernel.apple.wgsl     unified memory, wide subgroups
    kernel.nvidia.wgsl
    kernel.amd.wgsl
    kernel.soc.wgsl       tight power and bandwidth budgets
  webnn/graph.ts
  wasm/kernel.ts
```

Selection resolves in order, first hit wins:

```
explicit override  →  target + dtype  →  target  →  dtype  →  portable
```

**The resolution is implemented** (`harness/resolve.ts`, `harness/target.ts`);
the per-target kernels are not — no op ships a variant yet, so every op resolves
to its portable kernel today.

An op's `wgsl/` directory holds one or more **entry points**, and each may have
variants. The filename is the whole grammar — `<entry>[.<target>][.<dtype>].wgsl`:

```
kernel.wgsl              the default entry point, portable
inverse.wgsl             a second entry point, portable         (ops/stft)
scores.wgsl              one of two entry points, portable      (ops/attention)
kernel.nvidia.wgsl       a target variant of kernel
inverse.f16.wgsl         a dtype variant of inverse
scores.apple.f16.wgsl    both
```

Every entry point needs its own portable kernel, and **resolution never leaves
the entry point it was asked about** — `istft` falling back to the forward
transform because the chain ran off the end would be a wrong answer that still
looks like a result.

Entry points are named rather than inferred because an op genuinely may need
several kernels that are not variants of each other: `stft` computes the inverse
transform with different arithmetic, and `attention` is two dispatches with a
buffer between them, deliberately split so `layout: "auto"` cannot drop bindings
an entry point does not reference.

A suffix that is not a known target or dtype is an error, so `kernel.nvidai.wgsl`
is rejected rather than left in the tree looking tuned. A **bare** `nvidia.wgsl`
is rejected too, as ambiguous: it is far likelier to be a mis-written variant
than an entry point that happens to be called "nvidia".

What that cannot catch is a misspelt entry point — `inverze.wgsl` reads as a new
one. The asymmetry is deliberate. A misspelt variant fails silently, because
resolution falls back to portable and everything still runs; a misspelt entry
point fails loudly the moment its test asks for it by name. Only the silent one
needs a guard.

**Overrides are a first-class input, not an escape hatch.** Anyone integrating
this will eventually know something the library cannot — that their sequence
length is always 1, that their weights are static, that they would rather have
lower peak memory than higher throughput. Refusing that knowledge makes the
library something to work around. An override that names a variant which does
not exist raises; falling back to portable would hand the caller a kernel they
did not ask for and never say so.

Target detection has to be honest about how little is knowable. `adapter.info`
gives a vendor and an architecture string and not much else, so detection is a
hint, and the override exists partly because the hint will sometimes be wrong.
It is allowed to answer "I don't know", and that answer is portable rather than
a guess: `intel` covers both an on-die iGPU and a discrete Arc card, and nothing
in `adapter.info` says which, so it resolves to no target at all. A fallback
adapter is treated the same way — the vendor string there names silicon the code
is not running on.

Nothing picks a target-specific kernel silently. `describeAdapter(adapter)`
reports what the device said, which target was detected and why, and `resolve()`
returns the rung that hit along with every candidate it tried. A wrong guess that
is invisible is worse than a portable kernel.

It is a call the caller makes, not something the runner does on the way past.
`createRunner`'s job is to run a kernel and read the result back; which target a
device looks like is a question about the device, and the two do not have to be
answered at the same moment.

Adding a variant cannot skip the reference test. `eachVariant(url, entry, …)`
builds an op's test loop from its `wgsl/` directory rather than from a list
someone has to remember to extend, and `unguardedOps` fails the suite for an
entry point that grows a variant no test iterates. The check is per entry point,
not per op: looping `scores` says nothing about `context`. A target-specific
kernel that is fast and wrong is the failure this axis exists to design against.

All of it lives outside `harness/index.ts` — import `harness/variants.js`,
`harness/resolve.js` and `harness/target.js` directly. `index.ts` is what every
op's test imports, so anything re-exported from it lands on the import graph of
every GPU test; keeping resolution off it leaves `index.ts` and `suite.ts`
byte-identical to what they were before any of this existed.

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

`matmul` (GEMM) ✅ · `matvec` (GEMV) ✅ · `conv` ✅ (1D) · `conv_transpose` ✅ (1D) ·
`add` · `mul` · `gather` ✅ · `scatter` ✅ · `transpose` ✅ · `reduce` ✅

Small, total, boring. Everything else is built from these, and they are where
target-specific tuning pays off most.

### `kernel/` — one fused, named operation

`rope` ✅ · `rmsnorm` ✅ · `layernorm` ✅ · `group_norm` ✅ · `softmax` ✅ ·
`activation` ✅ · `snake` ✅ · `elementwise` ✅ · `quantize` ✅ · `dequantize` ✅ ·
`attention` ✅ · `flash_attention` ✅ · `ctc_decode` ✅ (greedy) · `mel` ✅ ·
`stft` / `istft` ✅

Fusion is the reason this layer exists rather than being composed from
`primitive/` at call time. `flash_attention` is not `matmul` + `softmax` +
`matmul`; it is the one that never writes the score matrix to memory, which is
the entire point of it.

`mel`, `stft` and `istft` are here because speech pipelines need them and no ML
kernel library ships them — they are DSP, so everyone assumes someone else has
them. The inverse STFT in particular is the thing ONNX cannot express, being
unable to carry complex tensors.

### `attention/` — the variants that are their own problem

Position: `RoPE` ✅ · `ALiBi` ✅ · `PoPE` ✅ · `YaRN` ✅ · `NTK scaling` ✅ · `rotary cache` ✅ · `half-RoPE (head range)` ✅

Sharing: `GQA` ✅ · `MQA` ✅ (one op — `gqa`, parameterised by `kvHeads`)

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

## `llm/`: an inference engine built from these ops

`llm/` is a config-driven llama-architecture (decoder-only, GQA, RoPE, SiLU
MLP, RMSNorm) forward pass, composed entirely from the ops above rather than a
new fused kernel — the "`block/`" exception the roadmap notes above rules out
does not apply, because a llama decoder is not a block that only pays off
fused; it is a *sequence* of the primitives and kernels this repository
already ships, and issue [#98](https://github.com/m96-chan/web-xpu-ops/issues/98)
tracks that specific composition, not a new op.

```ts
// llm/ is source-only for now — not part of this package's published
// `exports` (real-model use needs #96's weight converter and tokenizer,
// both later issues), so this is a within-repo import, not a package one.
import { LlamaEngine, TINY_FIXTURE_CONFIG } from "./llm/index.js";

const engine = new LlamaEngine(config, weights, runner.run);
const prefillLogits = await engine.forward(promptTokens); // matmul path, N > 1
const [nextLogits] = await engine.forward([nextToken]);    // matvec path, N === 1
```

**Per layer:** `rmsnorm → QKV projection (fused) → rope(Q, K) → gqa (+ KV
cache) → O projection → residual → rmsnorm → gate/up projection (fused) →
silu(gate) * up → down projection → residual`, then one final
`rmsnorm → lm_head`.

**What "config-driven" means in practice** — `llm/config.ts#LlamaConfig` has
one field per dimension a llama checkpoint's `config.json` carries (layers,
hidden size, query/KV heads, head dim, FFN width, vocab, RoPE base, RMSNorm
`eps`, weight tying). Two example configs live there: `TINY_FIXTURE_CONFIG`
(2 layers, hidden 64, 4 query / 2 KV heads, FFN 128, vocab 256 — what the
fixture below actually runs) and `SARASHINA_2_2_1B_CONFIG` (24 layers, hidden
1792, 16 query / 8 KV heads, FFN 6272, vocab 102400, RoPE base 500000 — issue
#96's real target, documentation only: running the actual checkpoint needs a
weight-conversion tool, a later issue — the tokenizer side is covered by
`llm/tokenizer.ts` above).

**KV cache and dispatch fusion.** The cache is pre-allocated f32,
`[kvHeads, maxSeqLen, headDim]` per layer (`llm/kv-cache.ts`) — no growth
during generation, optimisation left for later per the "correctness first"
rule below. Q/K/V and gate/up are each fused into one projection weight
(`llm/reshape.ts#concatRows` / `#splitConcatRows`) purely to cut GPU dispatch
count: this repository's `webgpu` (Dawn-through-Node) binding is measurably
unable to sustain an unfused prefill-then-decode run's ~195 dispatches inside
one device's lifetime on some machines, and fusing Q/K/V and gate/up brings
that down to ~155 without changing a single number the engine computes — see
`llm/reshape.ts` and `llm/engine.wgsl.test.ts` for the measurement.

**Correctness is a fixture, not an eyeball.** `llm/tools/gen_fixture.py`
builds a tiny, randomly-initialised llama model with `transformers` itself
(rule 7 — a real HF forward pass, `attn_implementation="eager"`, seeded), runs
an 8-token prefill and a 4-step greedy decode, and writes the weights and
logits to `llm/fixtures/tiny.*`. `llm/engine.wgsl.test.ts` runs the same
tokens through `LlamaEngine` on a real GPU and checks logits against the
fixture (measured: worst absolute diff **1.49e-7**, worst relative **3.4e-4**
across prefill and decode — tighter than any individual op's own tolerance,
because a tiny random model's logits do not accumulate error the way a
trained one's sharper distributions might) and greedy-decoded tokens for
exact equality. The fixture (weights + logits + a JSON manifest) is committed
rather than generated at test time — 436 KiB total (420 KiB weights, 12 KiB
logits, 4 KiB manifest), small enough that regenerating it on every `npm test`
would buy nothing; see `llm/tools/gen_fixture.py` for the reproduction steps.

One channel-ordering fact worth knowing if you are reading the weight-loading
code: HF Llama's RoPE (`rotate_half`) pairs channel `i` with channel
`i + headDim/2`, while `ops/rope` pairs **adjacent** channels `2i`/`2i+1` —
the same rotation, numbered differently within a head. `llm/weights.ts#permuteRopeChannels`
relabels a checkpoint's Q/K projection rows accordingly (exact, not
approximate — the derivation and a numeric proof are in
`llm/weights.ts` and `llm/rope-permutation.test.ts`), and
`llm/tools/gen_fixture.py` applies the same permutation before writing
`tiny.weights.bin`.

**Scope.** f32 weights only (int8 lands after #97's quantized `matvec`
connects); the tokenizer and sampler elsewhere in this README are not wired
into the engine yet, and there is no browser wiring (later issues under #96).
Correctness first, per the rule below — nothing here has been tuned for
speed.

---

## `llm/`: sampler and token-level constraints

`llm/sampler.ts` turns a next-token logits vector into a token id — `greedy`
(argmax) or `temperature` + `top-p` (nucleus) — and takes an optional
`Constraint`:

```ts
interface Constraint {
  nextAllowed(prefixTokens: readonly number[]): ReadonlySet<number> | null;
}
```

`null` means unconstrained; an empty set means no legal continuation, which
`sampleNext` treats as an error rather than guessing. The mask is applied to
the raw logits before either sampling mode runs, so a constrained draw can
never land outside the allowed set regardless of temperature or top-p.

There is **no repetition penalty**, on purpose. The usual per-seen-token
logit penalty was tried against Alibi's Japanese model and measured to
degrade output rather than de-loop it — see
[technologies-moe/alibi-ai#3](https://github.com/technologies-moe/alibi-ai/issues/3).
A caller that wants to discourage repetition should express it as a
`Constraint`, this module's own tool for narrowing the next token, rather
than as a penalty baked into the sampler.

`llm/constraints/line-format.ts` is one `Constraint`: a state machine for a
fixed line shape — literal text, an enum choice, more literal text, free
text (a forbidden-character set and a max length), then EOS. It exists for
schemas as small as `policy: <enum>\ntopic: <short text>`, where a full
GBNF/grammar engine is more machinery than the shape needs. It is
**tokenizer-agnostic**: it takes an injected `TokenCodec`
(`encode` / `idToToken` / `vocabSize`) instead of depending on any one
tokenizer —

```ts
const constraint = new LineFormatConstraint(codec, {
  segments: [
    { kind: "literal", text: "policy: " },
    { kind: "enum", choices: ["allow", "deny", "review"] },
    { kind: "literal", text: "\ntopic: " },
    { kind: "freeText", forbiddenChars: ["\n"], maxLength: 80 },
  ],
  eosTokenId,
});

const next = sampleNext(logits, generatedSoFar, { mode: "greedy" }, constraint);
```

Enum choices are matched by tokenizing each candidate once and walking a
token-id trie — the "candidate string, tokenized, forward-matched" approach —
so choices must be **token-prefix-free**: no choice's tokenization may be a
strict prefix of another's, or completion would be ambiguous. Such a spec is
rejected at construction rather than silently misclassified during
generation.

Neither file is wired into the engine above yet — the decode loop that would
call `sampleNext` per step is a separate, later issue.

---

## Backends

| backend | status | shape |
| --- | --- | --- |
| WGSL | 27 ops | a kernel per op, per target; resolution is `<entry>[.<target>][.<dtype>].wgsl` |
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

## Releasing

Publishing is driven by a tag, not by a merge, so that preparing a release and
shipping it stay two decisions:

```bash
# on main, with package.json already at the new version
git tag v0.2.0 && git push origin v0.2.0
```

`.github/workflows/release.yml` then lints, builds, runs the suite, checks that
the tag agrees with `package.json` — a mistyped tag would otherwise publish a
version nobody asked for, and npm versions cannot be replaced — and publishes
with `--provenance`, which links the tarball to the workflow run and commit that
produced it.

It authenticates with an `NPM_TOKEN` repository secret (an npm **automation**
token, since a publish token subject to 2FA cannot be used unattended). npm's
trusted publishing (OIDC) would remove that secret, but it cannot be configured
for a package that does not yet exist on the registry, so it is a change to make
after the first release rather than before it.

## License

MIT
