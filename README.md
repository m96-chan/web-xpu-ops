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

**Scope.** f32 weights (this section) and int8 weights (next section) both
exist now; the tokenizer and sampler are exported from `llm/index.ts` and
driven by the browser demo's generation loop (below), not called by the
engine's own `forward`. Correctness first, per the rule below — nothing here
has been tuned for speed.

---

## `llm/`: weight converter and the int8 (W8A32) engine path

Issue [#105](https://github.com/m96-chan/web-xpu-ops/issues/105) closes the
loop from "an engine that runs a tiny fixture" to "an engine that loads and
generates from a real checkpoint": `llm/tools/convert_weights.py` converts a
HF safetensors checkpoint (bf16) into per-row int8, and `LlamaEngineQ8`
(`llm/engine-q8.ts`) runs it.

Every number in this section was measured on one machine (rule 9): NVIDIA
GeForce RTX 5090, driver 610.57.04, Linux (Arch, kernel 7.1.5-arch1-2),
backend Dawn via `webgpu@0.4.x` under Node v25.6.1, Python 3.14 / NumPy 2.4
for the converter; checkpoint Sarashina2.2-1B-alibi-v1 (bf16, 2.68 GiB).
Timing figures are single observations from that machine, not averages —
comparable only against a rerun under the same conditions.

```ts
// llm/tools/convert_weights.py --model-dir <hf checkpoint> --out-dir <out>
// writes <out>/manifest.json + weights.codes.bin + weights.scales.bin + weights.norms.bin

import { loadConvertedWeightsQ8 } from "./llm/real-model-weights.js";
import { LlamaEngineQ8 } from "./llm/engine-q8.js";

const { config, weights } = loadConvertedWeightsQ8("<out-dir>", /* maxSeqLen */ 4096);
const engine = new LlamaEngineQ8(config, weights, runner.run);
const prefillLogits = await engine.forward(promptTokens);
```

**Converted format.** Every Linear weight (`wq`/`wk`/`wv`/`wo`/`gate`/`up`/`down`/
`lm_head`) and the embedding table become per-row absmax int8 codes (`[N, K]`,
one signed byte per element — `ops/quantize/reference.ts#quantize`'s own
convention) plus an f32 scale per row; `wq`/`wk` get `permuteRopeChannels`
applied before quantizing (permuting rows and quantizing per row commute, so
this is equivalent to permuting the codes afterward — the tiny fixture's
generator does it the other way around, on purpose, for a reason its own
module doc explains). Norm weights stay f32. Manifest entries are
`{ name, kind: "quant" | "norm", shape, codesOffset, scaleOffset }` or
`{ name, kind: "norm", shape, offset }` — the exact shape `llm/weights-q8-io.ts#buildLlamaWeightsQ8`
parses, shared by the tiny fixture's loader (`fixture-q8.ts`) and the real
checkpoint's (`real-model-weights.ts`) so the two formats cannot silently
drift apart. Converting Sarashina2.2-1B-alibi-v1 (2.68 GiB bf16) produced a
1.41 GiB int8 checkpoint (1,407,451,136 bytes codes + 2,711,552 bytes scales +
351,232 bytes norms) in about 8 seconds.

**`LlamaEngineQ8`'s resident memory.** Only the packed `matvecQ8` wire format
(`ops/matvec/reference.ts#packQ8`) is kept per projection after construction —
not also the unpacked codes the loader handed in, since packing does not
change a weight's size (repacking would roughly double memory for nothing).
The embedding table is the one exception kept in its original (unpacked)
form, decoupled from the loader's shared buffer via a copy
(`weights-q8.ts#cloneQuantizedLinear`) — without that copy, the loader's
manifest-parsing convenience (every weight a *view* into one buffer covering
the whole checkpoint) would keep the entire raw codes buffer resident (~1.4
GiB) for the engine's whole lifetime just to reach the ~183 MiB embedding
table. Measured on the real checkpoint: loading is ~220ms, `LlamaEngineQ8`
construction (packing every projection) is ~1.1s, and process RSS falls from
~2.96 GiB (while the loader's own buffers are still referenced) to ~1.56 GiB
once the caller drops that reference and a GC runs.

**Decode vs. prefill.** Decode (`tokens === 1`) dispatches `matvecQ8` directly
against the resident packed weight. Prefill (`tokens > 1`) dequantizes the
needed projection into a transient f32 matrix and runs `matmul` — issue
#105's own stated scope ("プリフィルは当面「行スケールdequantしてf32 matmul」
でもよい"), which happens once per generation (the prompt), not once per
token, since `greedyGenerate` calls `forward` with more than one token exactly
once. A prefill kernel that reads packed int8 directly is explicit follow-up
work, not done here.

**The embedding table's gather is CPU-side, not a GPU dispatch.** `embedTokens`
is quantized like every other weight, but `LlamaEngineQ8` dequantizes only the
rows a `forward` call's tokens actually name, on the CPU
(`weights-q8.ts#gatherDequantRows`), instead of dispatching `runGather`
against a fully-dequantized 733 MiB table for a call that reads at most
`maxSeqLen` of its rows.

**Correctness: an int8-quantization-aware fixture.** `llm/tools/gen_fixture_q8.py`
quantizes the tiny fixture's own weights per row, **substitutes the
dequantized weights back into the `transformers` model**, and re-runs the
same prefill/decode loop `gen_fixture.py` runs — so the reference
(`llm/fixtures/tiny_q8.*`) has the *same* quantization error baked in that
`LlamaEngineQ8` produces, rather than being compared against an f32-exact
answer a genuinely quantized engine could never match. `llm/engine-q8.wgsl.test.ts`
checks `LlamaEngineQ8` against it on a real GPU: worst observed absolute diff
**1.49e-7**, relative **7.22e-4** (well inside the `rel 1e-2, abs 5e-3`
tolerance, and close to `LlamaEngineQ8`'s own f32 counterpart's numbers —
evidence the int8 path and the Python reference agree on what quantization
error to expect, not that either one is loose). `llm/quantize-parity.test.ts`
separately checks that `llm/tools/quant_common.py` (the quantizer
`convert_weights.py` and `gen_fixture_q8.py` both call) rounds ties exactly
the way `ops/quantize/reference.ts#quantize`'s `Math.round` does
(`floor(x) + (frac >= 0.5)`, not `np.round`'s banker's rounding — which
disagrees at every `.5` boundary — and not the tempting `np.floor(x + 0.5)`,
whose addition double-rounds just below half-integers; see
`quant_common.py`'s module doc) by spawning the Python script and diffing its
output directly, including both boundary cases.

**Real-checkpoint status.** Converting, loading (`real-model-weights.ts`,
checked in `real-model-weights.test.ts`), and constructing `LlamaEngineQ8`
from the real Sarashina2.2-1B-alibi-v1 checkpoint all succeed and are
verified. **Live GPU generation from the real checkpoint does not run yet**,
for two separate reasons, neither a numerics problem (the int8 path is
verified above, to a tight tolerance, on real GPU dispatches):

- On the machine this was built on, a Node+Dawn binding fragility is
  triggered by real-model-scale CPU-bound work (loading and packing a
  ~1.4 GiB checkpoint, on the order of a second) immediately preceding a GPU
  dispatch; see [#107](https://github.com/m96-chan/web-xpu-ops/issues/107)
  for the isolated repro (an unrelated large allocation, and separately a
  pure CPU busy-loop with no allocation at all, both reproduce it — GPU
  contention and buffer count/size were ruled out).
- Independently of #107 and of the machine, the real vocabulary size breaks
  the lmHead projection against WebGPU's own limits: decode dispatches one
  workgroup per output row (102,400 > the default 65,535
  `maxComputeWorkgroupsPerDimension`), and prefill dequantizes lmHead to a
  ~700 MiB f32 matrix that exceeds the 512 MiB buffer/binding cap the
  harness requests (and the 128 MiB browser default). Found in review, not
  yet hit at runtime only because #107 aborts earlier; tracked as
  [#112](https://github.com/m96-chan/web-xpu-ops/issues/112), which blocks
  #106's real-model demo as well.

`llm/engine-q8.real-model.test.ts` runs this end to end (prefill + greedy
decode, tok/s per step) when a converted checkpoint and an encoded prompt are
supplied via environment variables, and skips (visibly, not as a silent pass)
otherwise; resolving #107 and #112 is what turns its assertions on. Live
generation, `llama.cpp` comparison, and tok/s are planned for
[#106](https://github.com/m96-chan/web-xpu-ops/issues/106) (browser demo),
where WebGPU runs in a separate GPU process rather than sharing Node's — the
condition #107 depends on does not exist there (but #112 applies to the
browser all the same).

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

`llm/index.ts` re-exports the tokenizer, the sampler and this constraint
(issue #106 — both had landed without ever being re-exported from the
package's own entry point, a gap #106's own text called out). The decode
loop that calls `sampleNext` once per step is still not inside
`LlamaEngine`/`LlamaEngineQ8` itself — a caller drives it, the way
`examples/llm-demo/src/main.ts` does below — since a constraint is a
generation-level policy, not something the engine's `forward` (one call, one
set of logits) has an opinion about.

---

## Browser demo (`examples/llm-demo/`)

Issue [#106](https://github.com/m96-chan/web-xpu-ops/issues/106): every
piece above — tokenizer, `LlamaEngineQ8`, sampler, `LineFormatConstraint` —
running together in a real browser tab over WebGPU, loading a real converted
checkpoint over HTTP. This is the first time this repository's `llm/` code
has generated anything outside Node; PR #108 could not complete a live GPU
dispatch on the machine it was built on (a Node+Dawn binding fragility
triggered by real-model-scale CPU-bound work immediately before a dispatch —
[#107](https://github.com/m96-chan/web-xpu-ops/issues/107)), and moved that
gate here on the strength of one fact: a browser's WebGPU implementation runs
in a separate GPU process, so the "CPU-bound work in the same process right
before a dispatch" condition #107 isolated does not exist there.

### Running it

```bash
npm run demo:build   # esbuild: examples/llm-demo/src/main.ts -> dist/bundle.js
npm run demo:serve   # node examples/llm-demo/server.mjs — Node standard library only
```

Then open `http://localhost:8770/examples/llm-demo/` in a WebGPU-capable
browser. `demo:serve` serves this repository (so `llm/data/*.vocab.json` and
the demo's own `dist/bundle.js` are reachable) and additionally maps
`/weights/` onto a converted-checkpoint directory outside this repository —
`convert_weights.py`'s output (`manifest.json` + `weights.codes.bin` +
`weights.scales.bin` + `weights.norms.bin`), by default
`technologies-moe/alibi-ai`'s `third_party/webgpu-weights/sarashina2.2-1b-alibi-v1-q8/`
(override with `ALIBI_WEIGHTS_DIR`). No `Range` support — `Content-Length` is
always set, since the page's progress bar reads it while streaming the ~1.4
GiB `weights.codes.bin`.

### Persistent weight cache (issue #121)

`loadWeightsQ8FromUrl` (`llm/browser-weights.ts`) caches the checkpoint into
IndexedDB by default — the demo above (and `technologies-moe/alibi-ai`'s own
integration, this issue's parent #96) no longer re-fetches ~1.4 GiB on every
visit. No demo-side code changed to get this: caching is on unless the fifth
argument's `enabled` is `false`, and every one of the demo's existing calls
already passes fewer than five arguments.

- **Versioning**: the cache key includes a SHA-256 hash of `manifest.json`'s
  raw bytes (`llm/weight-cache.ts#sha256Hex`). `manifest.json` itself (tens of
  KB) is fetched on every load — the only way to know whether a cached
  checkpoint is still current — but `weights.codes.bin`/`.scales.bin`/`.norms.bin`
  (1.41 GiB combined) are not, on a cache hit. A re-converted checkpoint under
  the same URL is detected by its changed hash and re-downloaded automatically;
  the previous version's chunks are then swept from IndexedDB.
- **Chunking**: each cached file is split into 96 MiB chunks
  (`DEFAULT_CHUNK_SIZE_BYTES`) before being written — never one ~1.4 GiB
  `IndexedDB` value.
- **Fallback**: IndexedDB unavailable, `indexedDB.open()` failing (Safari
  private mode), a read failing (`get()` rejecting — storage pressure closing
  the connection, a stale database from an older store-name layout), or
  `navigator.storage.estimate()` reporting insufficient free space *for a
  write* all fall back transparently to a plain network load — no feature
  loss, only a persistence difference. The quota check gates only the write
  path, deliberately: a read frees no space and needs none, so a device whose
  reported `usage` includes this very cache (true from the moment one write
  succeeds) never has that alone make a valid cache unreadable.
- **Storage**: behind `llm/chunk-store.ts#ChunkStore` (`get`/`put`/`delete`/`list`),
  injected — `llm/idb-chunk-store.ts#createIndexedDbChunkStore` is the real
  IndexedDB backend (its `put`/`delete` resolve on the owning transaction's
  `oncomplete`, not the individual request's `onsuccess`, so a write whose
  transaction aborts on commit — a large value's `QuotaExceededError` — is
  reported as the failure it is), `InMemoryChunkStore` is what every Node test
  (`chunk-store.test.ts`, `weight-cache.test.ts`, `browser-weights.cache.test.ts`)
  runs the cache logic against instead.
- **Known limitations**: versioning one hash across all three binary files
  means a hypothetical future producer of this checkpoint format that keeps
  `manifest.json` byte-identical while republishing only a binary could — if
  interrupted mid-write — mix old- and new-version chunks under one hash
  undetectably ([#124](https://github.com/m96-chan/web-xpu-ops/issues/124),
  not reachable by this repository's own `convert_weights.py`, which always
  changes `manifest.json` too). Concurrent tabs open across a deploy window
  can each treat the other's freshly-written version as stale and re-download
  ([#125](https://github.com/m96-chan/web-xpu-ops/issues/125)) — wasteful, but
  never data-corrupting, since a load's `manifestHash` and the chunks it reads
  always agree.

**Real-hardware verification (non-headless Chrome, DevTools Network domain
over CDP).** Driven with raw CDP (Node's own `WebSocket`, no puppeteer) against
a real, visible Chrome window — a dedicated `--user-data-dir` and
`--remote-debugging-port`, `examples/llm-demo/server.mjs` on a private port,
both pointed at a scratch copy of the real converted checkpoint (`manifest.json`
copied, the three large binaries symlinked — no 1.4 GiB duplicated, and no
interference with another verification running against the repository's own
default checkpoint directory at the same time). `Network.loadingFinished`'s
`encodedDataLength` (bytes actually received over the wire, including headers)
is what "network transfer" below means — not an assumption from the page's own
progress bar:

| step | `weights.codes.bin` | `.scales.bin` | `.norms.bin` | requests issued | page-reported load time |
| --- | --- | --- | --- | --- | --- |
| 1. First load (cold cache) | 1,407,451,307 B | 2,711,720 B | 351,399 B | 1 each | 2507 ms |
| 2. Reload (cache hit) | **0 B** | **0 B** | **0 B** | **0** (no `Network.requestWillBeSent` at all) | 848 ms |
| 3. `manifest.json` mutated by 1 byte, reload | 1,407,451,307 B | 2,711,720 B | 351,399 B | 1 each | 1995 ms |

Step 2 is the strongest form of this issue's "reload transfers 0 bytes":
not merely 0 bytes counted, but zero `Network.requestWillBeSent` events for
any of the three files — the cache path never calls `fetch` on them at all.
`manifest.json` itself (~37 KB) is fetched in all three steps, per this file's
own "Versioning" note above. Step 3's manifest was mutated by appending one
byte (`printf ' ' >> manifest.json` — valid JSON either side, so the parsed
config/weights are identical; only the file's bytes, and so its SHA-256,
differ) to simulate a re-converted checkpoint under an unchanged URL, and
`IndexedDB`'s key count stayed at 17 (16 chunks + 1 version record) across the
mutation rather than growing to 33 — the old version's chunks were swept, not
accumulated. `navigator.storage.estimate()` reported `usage: 1,409,094,418`
bytes after step 1, matching the checkpoint's real size.

Load-time notes: both server and browser were on the same machine (loopback
HTTP), so step 1's 2507 ms is network-plus-parse over localhost, not a real
internet download — the ~3x speedup to 848 ms on a cache hit is from skipping
`fetch` and streaming-decode entirely in favor of several `IndexedDB` reads
plus a buffer join, and would be a much larger, latency-dominated win on a
real network (this repository's own default deployment, `alibi-ai`, is served
over the internet, not loopback). Chunk size: 96 MiB
(`DEFAULT_CHUNK_SIZE_BYTES`, the midpoint of this issue's 64–128 MiB range) —
14 chunks for `weights.codes.bin`, 1 each for `.scales.bin`/`.norms.bin` at
this checkpoint's size; not swept across multiple chunk sizes for this issue,
since nothing in `weight-cache.ts`'s logic depends on the exact value and nothing
in this run's numbers suggested it was a bottleneck worth tuning before
shipping.

The page: load the weights (progress bar, byte-weighted across the four
fetched files since `weights.codes.bin` dominates), then either mode —

- **スタイラ**: `<|system|>{SYSTEM_PROMPT}</s><|user|>[{policy}] {text}</s><|assistant|>`,
  `SYSTEM_PROMPT` and the prompt format copied verbatim from
  `technologies-moe/alibi-ai`'s `assets/llm.js` (this issue's own
  instruction — that file's wording is what the checkpoint was trained
  against, per that repository's LoRA training pipeline).
- **聞く層**: the same repository's `LISTEN_SYSTEM_PROMPT` / few-shot /
  `<|system|>…</s>(<|user|>…</s><|assistant|>…</s>)*<|user|>{text}</s><|assistant|>`
  prompt, decoded under a `LineFormatConstraint` built from the identical
  spec `assets/llm.js`'s GBNF grammar
  (`root ::= "policy: " ("full_gear" | "engage" | "brush_off") "\ntopic: "
  [^\n]{1,24}`) describes — a toggle switches the constraint on and off, so
  the difference it makes is directly visible.

Generation is greedy, decoding token by token until `</s>` (`vocab.eosId`) or
a step cap, with the page's tok/s split into **prefill** (one
`engine.forward(promptTokens)` call) and **decode** (one
`engine.forward([token])` call per step) — measured at the `forward()` call
boundary with `performance.now()`, not from a GPU timestamp query (this
demo's `browser-runtime.ts` does not implement `Runner.time()`; wall-clock
around each `forward()` await is simpler and sufficient for this issue's
ask).

### Why this needed a bundler, and why `kernels.ts` did not need to change

`llm/kernels.ts` reads its ten kernels' WGSL source via
`node:fs#readFileSync` (through `harness/index.ts#kernel()`) and
`harness/wgsl.ts#createRunner` is built on the `webgpu` npm package (Node's
Dawn binding) — both fine for `npm test`, both unusable in a browser.
`examples/llm-demo/build.mjs` (esbuild, this repository's first bundler —
justified in that file's own module doc) resolves `kernels.ts`'s
`import { kernel, params } from "../harness/index.js"` to
`examples/llm-demo/src/browser-runtime.ts` at bundle time instead — a
`navigator.gpu`-based `Runner` port of `harness/wgsl.ts#createRunner`, plus
`kernel()`/`params()` implementations with the identical signatures, sourcing
their WGSL from ten `.wgsl` files esbuild's `text` loader inlines as JS
strings. The redirect is an `onResolve` hook keyed on the import specifier,
and `llm/kernels.browser-parity.test.ts` (plain text parsing, no `.wgsl`
import, runs under `npm test`) fails if the two kernel tables ever name
different ops or entry points.

`kernels.ts` itself did need one real change, found by this demo rather than
assumed ahead of it: `runMatVec`/`runMatVecQ8` dispatch one workgroup per
output row with no tiling, which is exact and cheap until the row count
exceeds WebGPU's `maxComputeWorkgroupsPerDimension` (`65535`, measured the
same on Dawn/Node and Chrome). Sarashina2.2-1B-alibi-v1's `lm_head`
(`vocabSize=102400`) is past that, and past it a dispatch is not rejected
loudly — the validation error is reported through the device's own
asynchronous error callback, not by throwing where `dispatchWorkgroups` is
called — so every decode step's logits silently came back all zero and
argmax always picked token 0. Both functions now split a too-wide dispatch
into several within the limit and concatenate the results; see the
`CHANGELOG`'s `### Fixed` entry and `llm/kernels.chunking.test.ts` for the
full story, including why that test's proof is a mocked `Runner["run"]`
rather than a real GPU dispatch at 65,535 workgroups (which reproducibly
crashed this repository's own Node/Dawn binding — the `#38`/`#49`/`#107`
family again).

### Real-hardware verification (RTX 5090, Chrome, non-headless)

Three generations, greedy, against the real Sarashina2.2-1B-alibi-v1 int8
checkpoint, compared to `llama-server`'s `/completion` endpoint (same GGUF
family, `temperature: 0`, `top_k: 1` — raw completion, not its chat-template
path, so the exact same prompt string reaches both engines). Quantization
schemes differ (this engine: per-row absmax int8; `llama.cpp`: GGUF `Q8_0`
block quantization), so token-for-token identity is not expected, but on the
`full_gear` case it happened anyway:

| case | this engine (browser) | `llama.cpp` (`Q8_0`) |
| --- | --- | --- |
| `full_gear` | `【talk2】新しいGPU買ったから、ベンチマーク回した。\n【talk】……意外と速くて、地味に驚いた。` | identical |
| `brush_off` | `【base】昨日推しのライブ配信見てたら、朝になってた。` | `【base】……あれ、もう朝じゃん。` (diverges after `【base】`) |
| 聞く層 (constrained) | `policy: brush_off\ntopic: セグフォ` | `policy: full_gear\ntopic: セグフォの直し方` (diverges at the very first enum choice) |

tok/s (unoptimized — no batching or overlap between dispatches, one
`await` per kernel call):

| case | prefill | decode |
| --- | --- | --- |
| `full_gear` | 76 tok, 12.6s (6.0 tok/s) | 25 tok, 33.0s (0.76 tok/s) |
| `brush_off` | 69 tok, 12.1s (5.7 tok/s) | 13 tok, 17.3s (0.75 tok/s) |
| 聞く層 | 363 tok, 13.4s (27.0 tok/s) | 13 tok, 18.3s (0.71 tok/s) |

Weight load (1.41 GiB over loopback HTTP, warm OS page cache): ~600–650ms —
not representative of a real network, only of this fetch-and-parse path.
Engine construction (packing every projection into `matvecQ8`'s wire format)
was ~1.0–1.1s per generation. See the PR for the full prompts, decoded
output, and a screenshot.

### `LlamaEngineQ8Resident`: GPU-resident decode (issue #110)

The tok/s table above is `LlamaEngineQ8`'s own dispatch structure: one
`Runner.run()` — its own buffer allocation, `queue.submit`, and readback —
per kernel call, per layer, roughly 155 GPU↔CPU round trips for one decode
step. `LlamaEngineQ8Resident` (`llm/engine-q8-resident.ts`) is the same
model, the same WGSL, the same `matvecQ8` weights, restructured so one
generated token costs **one `queue.submit`** and **one readback** (the final
logits alone). Prefill is unchanged in scope (issue #110 is decode only) —
the first `forward()` call is delegated whole to a `LlamaEngineQ8` instance
internally; every call after that must be exactly one token.

Same real checkpoint, same machine (RTX 5090, Chrome, non-headless, the
demo's own "GPU常駐デコード" toggle), same two prompts as the table above —
byte-for-byte identical generated text between the two engines on both:

| case | engine | prefill | decode |
| --- | --- | --- | --- |
| `full_gear` | `LlamaEngineQ8` | 76 tok, 12.0s (6.4 tok/s) | 25 tok, 31.2s (**0.80 tok/s**) |
| `full_gear` | `LlamaEngineQ8Resident` | 76 tok, 12.7s (6.0 tok/s) | 25 tok, 0.154s (**161.8 tok/s**) |
| `brush_off` | `LlamaEngineQ8` | 69 tok, 12.0s (5.8 tok/s) | 13 tok, 16.4s (**0.79 tok/s**) |
| `brush_off` | `LlamaEngineQ8Resident` | 69 tok, 12.9s (5.4 tok/s) | 13 tok, 0.095s (**137.6 tok/s**) |

~175–200x on decode. The raw prefill numbers above are **not** a clean
before/after comparison, though (PR #116 review, item 7): `runPrefill`
(`llm/engine-q8-resident.ts`) constructs a whole `LlamaEngineQ8` delegate —
the same "pack every projection into `matvecQ8`'s wire format" step the
"Engine construction: ~1.0–1.1s" line below measures — on the *first*
`forward()` call, and that call is what the demo's `prefillMs` times. For
the non-resident row, that identical construction happens once, before
`generate()` even starts, and is **not** counted in `prefillMs` at all — so
the resident row's prefill number carries an extra ~1.0–1.1s the
non-resident row's does not, not a difference in prefill compute. Subtracting
that (using this table's own "Engine construction" figure for
`LlamaEngineQ8`, since it is the same construction happening in both places):
`full_gear` 12.7s − ~1.05s ≈ 11.65s (**~6.5 tok/s**, against `LlamaEngineQ8`'s
own 6.4 tok/s) and `brush_off` 12.9s − ~1.05s ≈ 11.85s (**~5.8 tok/s**,
against 5.8 tok/s) — once that's accounted for, prefill really is within
run-to-run noise of itself, as expected (issue #110 left it unchanged); the
raw ~6–8% gap the unadjusted numbers above show is this hidden construction
cost, not a regression, and not something to report as "noise" without
saying so (rule 9 — a claim needs its own measurement, not an assumption).
Engine construction: `LlamaEngineQ8` ~1.0–1.1s, `LlamaEngineQ8Resident`
~1.4–1.5s (a few more `matvecQ8` weight/scale buffer pairs — see "buffer
design" below — plus every bind group built up front instead of on first
use); the resident figure is `LlamaEngineQ8Resident.create()`'s own cost
only; the *legacy delegate's* construction is the ~1.0–1.1s folded into its
`prefill` column above instead, per the previous paragraph.

**Roofline decomposition (rule 9 — measured, not estimated).** This
machine's own bandwidth ceiling, from `harness/roofline.ts` at the time of
this measurement: **1.707 TB/s** (94–95% of the RTX 5090's 1.792 TB/s rated
figure — the same calibration `harness/roofline.test.ts` already checks).
Sarashina2.2-1B-alibi-v1-q8's resident weight footprint is 1,410,513,920
bytes (codes + scales + norms, PR #108's own numbers) — every `matvecQ8`
projection's weight is read exactly once per decode step, so that figure is
also the minimum bytes one token must move:

| quantity | value |
| --- | --- |
| measured bandwidth ceiling | 1.707 TB/s |
| resident weight bytes / token | 1,410,513,920 (1.31 GiB) |
| **bandwidth-bound lower bound** | **0.826 ms/token (1210 tok/s)** |
| measured decode (`full_gear`) | 6.16 ms/token (161.8 tok/s) — **13.4% of the lower bound** |
| measured decode (`brush_off`) | 7.31 ms/token (137.6 tok/s) — **11.3% of the lower bound** |

The gap (~5.3–6.5 ms/token) is not unaccounted for — it is two things this
issue's own scope left in place, both documented in `llm/engine-q8-resident.ts`'s
class doc:

1. **Attention scans `maxSeqLen`, not the true KV length.** `ops/gqa`'s
   kernels take `S` as both the softmax loop bound *and* the per-head
   stride into K/V (`k_head = kv_head * S * D`), so a KV cache addressed by
   `maxSeqLen = 4096` has to be *read* with `S = 4096` every step, relying
   on the causal mask to skip the dot product past the real position rather
   than a shorter, growing `S`. At Sarashina2.2-1B's shape (`kvHeads=8`,
   `headDim=112`) that is `8 × 4096 × 112 × 4 bytes ≈ 14.7 MiB` each for K
   and V, per layer, per token — **≈ 672 MiB/token** across 24 layers,
   comparable in order of magnitude to the 1.31 GiB weight traffic above,
   and it does not shrink as the KV cache empties out early in a
   generation. Fixing it needs a stride parameter `ops/gqa` does not have,
   which is a new-kernel change this issue's scope excludes ("カーネル融合
   〈次ISSUE〉" / "no new kernels").
2. **~800 GPU op launches per token.** Because every projection gets its
   own buffer rather than a fused one (below), one decode step is ~411 real
   dispatches (17 per layer × 24 layers, plus the final norm and two
   `lm_head` chunks) and ~384 small `copyBufferToBuffer` KV-cache writes
   (`2 × kvHeads × numLayers`) — all inside one `queue.submit`, so this is
   encoding and pass-boundary overhead, not per-dispatch round-trip
   latency. Not broken out further than that: this device's
   `timestamp-query` support times one dispatch at a time
   (`harness/wgsl.ts#Runner.time`), and timing ~800 of them individually
   inside one resident batch is a measurement this PR does not attempt —
   left here as "unmeasured", per rule 9, rather than guessed.

**Buffer design.** `LlamaEngineQ8`'s CPU-side engine fuses Q/K/V and
gate/up into one packed weight so one `matvecQ8` dispatch computes all of
them — worthwhile there because each dispatch used to cost a full
submit+readback round trip. That reasoning does not carry over to a
resident engine that already pays one submit for an entire token's
dispatch chain, and fusing costs something real here: splitting a fused
output by a byte offset into one buffer only works when that offset is a
multiple of `minStorageBufferOffsetAlignment` (measured 256 bytes on this
device) — the tiny fixture's own `kvDim = 32` floats lands `v`'s slice at
byte 384, not a multiple of 256. `LlamaEngineQ8Resident` gives every
distinct tensor its own buffer instead: a few more bandwidth-bound
`matvecQ8` dispatches per layer than a fused version, always valid
regardless of the model's dimensions (`harness/resident.ts#bindGroup`'s
doc has the validation error that caught this).

**KV-cache writes** are a GPU-to-GPU `copyBufferToBuffer`, not
`queue.writeBuffer`: the new token's K (after RoPE) and V already live in a
GPU buffer, written by dispatches a few ops earlier in the same batch, so
routing them through the CPU to satisfy `writeBuffer`'s signature would
undo "logits-only readback". `KVCache`'s own `[kvHeads, maxSeqLen, headDim]`
layout means one position's write is `kvHeads` small copies, not one — see
`llm/engine-q8-resident.ts`'s class doc.

**One device, not two.** `LlamaEngineQ8Resident.create()` takes a single
`ResidentDevice` and derives its prefill delegate's `Runner["run"]` from it
(`harness/resident.ts#runnerFromResident`) rather than taking a second,
independently-constructed `Runner`. An earlier version did take a second
`Runner`, and constructing two native `webgpu` `GPUDevice`s in one Node
process reproducibly crashed this repository's Node/Dawn binding partway
through prefill — not on every run, and not always at the same call, the
signature of binding instability under load (issue #38/#49/#107's own
family) rather than a logic bug.

**Correctness gate.** `llm/engine-q8-resident.wgsl.test.ts` runs the tiny
int8 fixture's full prefill + every decode step through
`LlamaEngineQ8Resident` and checks its logits against the fixture's own
`prefillLogits`/`decodeLogits` and its greedy tokens against
`fixture.decodeTokens` — the same numbers `engine-q8.wgsl.test.ts` already
proves `LlamaEngineQ8` itself produces, so the two together are a
transitive "matches the pre-optimization engine" proof without building
two independent engines' worth of dispatches on one device at once (see
that test file's own doc for why that mattered in practice on this
machine). On a clean run, both engines' worst prefill/decode diffs against
the Python reference are bit-identical: `abs: 1.49e-7, rel: 7.2e-4`
(prefill) / `7.1e-4` (decode) — the same figures PR #108 recorded for
`LlamaEngineQ8` alone. This machine's Node/Dawn binding is flaky under
concurrent GPU load regardless of this change (confirmed by re-running
`npm test`: pre-existing, unmodified files — `harness/timing.test.ts`,
`llm/engine.wgsl.test.ts`, `llm/engine-q8.wgsl.test.ts` — fail the same way,
independent of anything in this PR); `scripts/test.mjs`'s retry-once
handles the common case, and this test passed cleanly, repeatedly, once
retried.

### Scope

Reaching into `technologies-moe/alibi-ai` (a separate repository) itself —
wiring this engine into that project's actual chat UI, not just reusing its
prompt wording — is out of scope here and tracked on that repository's own
side. Mobile WebGPU and UI polish are likewise out of scope (issue #106's own
text).

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
