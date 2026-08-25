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

Thirty-one ops, WGSL only, verified against their references on a real GPU. That is
the count of directories under `ops/` — the number the backends table states and
the number `harness/distribution.test.ts` asserts against the tree, so it is the
one that cannot go stale silently. Counting the rows of the table below gives
something else and always will: a few ops take a row per entry point
(`matvecQ8`, `matmulQ8`, `ropeAxes`), and `permute` and `dequant_transpose` are
described in the LLM-engine sections that produced them rather than here.

**Speed is unmeasured for all but one of them.** The roofline each would be
reported against does not exist yet, and a number without one would be a
statement about this GPU rather than about the kernel — so the column says so
rather than being left blank. The exception is `axpy`, whose whole reason to
exist is replacing two dispatches with one, so it had to be measured against
the pair it replaces (#152); its row carries both the figure and the conditions
it was taken under.

| op | notes |
| --- | --- |
| `matvec` | GEMV; `torch.mv` convention, streaming rather than tiled. Speed unmeasured |
| `matvecQ8` | W8A32 GEMV: `matvec` with the weight held as int8 instead of f32. Weight is `[N, ceil(K/4)]` `u32`, four codes packed per word least-significant byte first — the layout a little-endian host gets for free by viewing an `Int8Array`'s buffer as `Uint32Array`; scale is `[N]`, `quantize`'s per-row absmax convention, applied once per row after the dot product rather than per term. `packQ8` packs `quantize`'s `Int32Array` codes into this layout; the two compose (`packQ8(quantize(w).output, N, K)` + `quantize(w).scales`) rather than this op quantizing on its own. Speed unmeasured |
| `matvecQ4G128` | W4A32 GEMV: `matvecQ8` at four bits, with one scale per **group of 128 columns** instead of one per row. Weight is `[N, ceil(K/8)]` `u32`, eight two's-complement nibbles per word least-significant nibble first — `packQ8`'s byte order at half the width; scale is `[N, ceil(K/128)]`, `quantizeQ4G128`'s per-group absmax. Range is symmetric **`[-7, 7]`**, `scale = absmax/7`, and the reciprocal is formed as `7/absmax` rather than `1/f32(scale)` — all three stated because the alternatives are live conventions and one of them (Q4_0's `[-8, 7]`) measured *better* on weight error while breaking argmax. **Not "Q4_0 compatible"**: block 128 vs 32, `[-7, 7]` vs `[-8, 7]`, f32 scale vs fp16. See "The q4 format" below. Speed unmeasured |
| `rmsnorm` | workgroup reduction; `eps` guards an all-zero row. An optional per-group `weight` of `[G, D]` for QK-norm — row `n` takes group `n % G`, so the grouped axis has to be the one just left of `D` (`[B, S, H, Dh]`, not `[B, H, S, Dh]`); `G = 1` is the single shared gamma. Reduction stays over `D` alone, matching `F.rms_norm(x, (D,)) * w` rather than `torch.nn.RMSNorm((H, Dh))`, which reduces over both. Speed unmeasured |
| `layernorm` | two workgroup reductions; **biased** variance (`1/D`) and `eps` inside the `sqrt`, as `torch.nn.functional.layer_norm`. Speed unmeasured |
| `group_norm` | `torch.nn.functional.group_norm`: statistics pooled over a **group of channels**, affine applied **per channel**. For `[N, C, L]` each of the `N × G` groups reduces `(C/G) × L` values, so this is not `layernorm` with a longer row — that gives one mean per channel, and the two disagree with no error and no change of shape. `G = 1` normalises the whole sample, `G = C` is InstanceNorm. Biased variance and `eps` inside the `sqrt`, both measured against torch 2.10 rather than inherited from `layernorm` (the unbiased form is out by 1.6e-1, `eps` outside the root by 4.3e-6). `C % G != 0` throws, as torch does. Speed unmeasured |
| `softmax` | max-subtracted, so real logits do not overflow `exp` |
| `activation` | `relu2`, `silu`, `elu`, `tanh`, `gelu`, `gelu_tanh`. `elu`'s `alpha` is a scalar hyperparameter defaulting to 1.0, as `torch.nn.ELU`. GELU is two functions, not one: the default `gelu` is the exact `erf` form (`torch.nn.functional.gelu`'s own default, `approximate="none"`) and `gelu_tanh` is `approximate="tanh"` — they differ by up to **4.73e-4, at x = 2.699**, so neither is a silent stand-in for the other. Speed unmeasured |
| `snake` | Two entry points, because the name covers two functions. `kernel`: `x + sin²(α·x)/α` with a **learned per-channel** α (arXiv:2006.08195), as `Snake1d` in DACVAE. `beta`: `x + sin²(α·x)/β` with **both** learned per channel, as BigVGAN's `SnakeBeta` and MioCodec's decoder — β = α recovers the first. A checkpoint's `alpha` tensor does not say which it belongs to, and the BigVGAN family stores **logarithms** while DACVAE stores values; neither kernel exponentiates, because doing so would be wrong for the other. The epsilon is upstream's, inside the reciprocal and at upstream's value, guarding the **divisor** — α in the first, β in the second. `sin²` is the square of the sine. Its own op rather than an `activation` kind, because α is a buffer and a channel stride rather than a scalar — the reason is written out in `ops/snake/reference.ts`. Speed unmeasured |
| `elementwise` | `add`, `multiply`, over two equally sized arrays. `elementwiseRows` (entry point `rows`) is the same two operations with the right-hand side broadcast **along the last dimension** — `[S, D] ⊕ [D]`, which is a `Linear`'s bias for `add` and AdaLN's per-channel scale for `multiply`. It is a separate function, and takes `S` and `D` explicitly, rather than `elementwise` inferring a broadcast from the lengths: a `[3,3]` and a `[3]` admit two different broadcasts (torch 2.10: `+ c` is `[[1,3,5],…]`, `+ c.unsqueeze(1)` is `[[1,2,3],…]`) and nothing in the lengths says which. Alignment is from the right as in NumPy/PyTorch, so `b.length === a.length` is a mistake and throws — torch refuses `[2,3] + [6]` for the same reason. Speed unmeasured |
| `axpy` | `out[i] = y[i] + a * x[i]` with a **scalar** `a` — `torch.add(input, other, alpha=)`, BLAS's `saxpy` by name. Exists because `ops/elementwise` is same-shape only, so a rectified-flow sampler's `latent += dt * velocity` otherwise costs a full-length buffer of copies of `dt` plus multiply-then-add. Two entry points: `kernel` writes a third buffer, `inplace` updates `y` through a single `read_write` binding (aliasing `y` into `kernel`'s two bindings instead is **not** an error you get told about — the command buffer is invalidated at `finish()` and the readback is all zeros). It also **rounds once**: this device's compiler contracts `y + a*x` into an FMA, which is what `torch.add` does on CPU and CUDA, where multiply-then-add rounds the product first — at `a = f32(0.1), x = 3, y = -f32(0.3)` the two-dispatch path cancels to exactly 0 and this one gives -2^-27. Measured (RTX 5090, driver 610.57.04, Dawn via `webgpu@0.4`, Node v25.6.1, f32, GPU timestamps, median of 5 per session, three sessions, otherwise-idle GPU; ceiling from `harness/roofline.ts` in the same sessions, 1.69-1.72 TB/s). At **N = 262,144** (one 16×128×128 latent): **6.5 µs against 12.8-16.9 µs** for `elementwise(multiply)` + `elementwise(add)` — **2.0-2.6x**, but at only 480-487 GB/s, **28% of the ceiling**, so neither path is bandwidth-bound there and the win is doing half the work rather than doing it faster. At **N = 1,048,576** it is: **7.7-9.0 µs against 17.1-18.4 µs** — **2.0-2.4x**, at 1.40-1.63 TB/s or **83-95% of the ceiling**, which is what a kernel moving 12 bytes per element and nothing else should reach. That larger size was unmeasurable while writing this op — it aborted or hung three times running — and is measurable now because #161 fixed the cause (the harness let Dawn's `GPU` instance be collected out from under the device); nothing in this op changed |
| `rope` | rotary position embedding, with KV-cache offset and NTK / YaRN context scaling. Follows `jquesnelle/yarn` and `transformers`, which agree; YaRN's attention temperature is included. An optional precomputed angle table (`ropeCache`); past its end the angle is recomputed rather than wrapped. `headOffset` / `headCount` rotate a subset of the **heads** and copy the rest through — the axis Irodori-TTS's `_apply_rotary_half` uses (`chunk(2, dim=-2)`), not channel-wise partial rotary (`rotaryDim`), which is the other thing "half-RoPE" is used to mean and is not implemented. Speed unmeasured |
| `ropeAxes` | multi-axis RoPE, in `ops/rope` beside the 1-D one: the head dim split into contiguous per-axis blocks (Z-Image's `[32, 48, 48]`), each rotated by that token's own position on that axis. Follows Z-Image's `RopeEmbedder` / `apply_rotary_emb` (Tongyi-MAI/Z-Image @ `26f23ed`), which is what decides all three of its conventions: positions arrive as an explicit `[N, axes]` `Int32Array` (upstream's `ids`) rather than being derived from a patch grid; one shared `thetaBase` (256 there), with the exponent normalised by **the axis's own channel count**, not by the head dim; and pairing is **adjacent** channels `2i`/`2i+1` — the same convention `rope` uses, `torch.view_as_complex` upstream, and *not* HF Llama's `rotate_half`, so a Z-Image checkpoint needs no `permuteRopeChannels` and a Llama-style one needs the same permutation as for `rope`. Angles are computed rather than tabulated, so upstream's `axes_lens` is not a parameter and a negative position turns backwards instead of wrapping to the end of a table. An odd axis dim throws (upstream cannot express one either). No scaling, no head range, no cache — none of the three is what Z-Image asks for, and `ropeCacheAxes` waits on a measurement. Speed unmeasured |
| `alibi` | linear attention-score bias (arXiv:2108.12409); slopes follow the paper's own `get_slopes`, including the **non-monotonic** appended tail for head counts that are not a power of two. Bias is the paper's relative form `m * (j - i)`, not BLOOM's `m * j`; masking is the caller's — and `attention`'s `mask` is an additive bias of exactly this shape (`maskShape: [1, H, L]`), so the two compose by addition. Speed unmeasured |
| `pope` | Legendre polynomial position table (arXiv:2405.04585, Eq. 14); order is the position, argument sweeps `[-1, 1)`. `posOffset` is required because the paper does not say whether positions start at 0 or 1. Speed unmeasured |
| `quantize` | per-row absmax to int8, symmetric `[-127, 127]` |
| `dequantize` | applies both the weight and the activation scale |
| `matmul` | GEMM; `torch.mm` convention, shared-memory tiling. Speed unmeasured |
| `matmulQ8` | W8A32 GEMM: `matmul` with the right-hand operand held as an int8 weight instead of f32, `matvecQ8`'s own `[M, ceil(K/4)]` `u32` packed wire format read in-kernel — no separate dequant/transpose pass. Scale is `[M]`, `quantize`'s per-row absmax convention, applied once per output element. Speed unmeasured |
| `matmulQ4G128` | W4A32 GEMM: `matmulQ8` at four bits, reading `matvecQ4G128`'s own `[M, ceil(K/8)]` `u32` packed wire format and its `[M, ceil(K/128)]` per-group scales in-kernel — the prefill half of the q4 format, same tiling (`TILE = 16`) and same argument names as `matmulQ8`. **No bias**, deliberately: `matmul` and `matmulQ8` have none either, and a fused one cannot be measured against anything until the plain form agrees with the reference. Speed unmeasured |
| `transpose` | turned through workgroup memory so both read and write stay consecutive |
| `reduce` | `sum` / `max` / `min` / `mean` along an axis |
| `gather` | row selection, as `torch.index_select(table, 0, indices)` — not `torch.gather`; an out-of-range index gathers zeros |
| `scatter` | indexed writes; **colliding indices accumulate** — see below |
| `stft` / `istft` | `torch.stft` / `torch.istft` conventions: centred, reflect padding, one-sided, unnormalised, periodic Hann; `istft` divides by the `w²` envelope — see below. `istft` also takes `padding: "same"`, the Vocos / X-Codec-2 / MioCodec vocoder convention that crops `(nFft - hop) / 2` per end so `T` frames give `T * hop` samples — **not a torch mode**, and not composable from one, because the samples `center: false` would return fail NOLA. Speed unmeasured |
| `conv` | `conv1d` and `conv2d`, as `torch.nn.functional.conv1d` / `conv2d` — a **cross-correlation**, so the kernel is *not* flipped (in 2D, on neither axis); `stride` / `padding` / `dilation` / `groups` / optional `bias`. 2D is NCHW (`[N, Cin, H, W]`, weight `[Cout, Cin/groups, KH, KW]`) and its three spatial arguments take **`number | [H, W]`**, PyTorch's `int | tuple[int, int]` with the pair order measured rather than assumed. `padding` is an integer count on both: `'same'` / `'valid'` are **not** accepted, because `'same'` with an even effective kernel pads asymmetrically in torch and one integer per axis cannot say that — unlike `istft`'s `"same"`, which is a genuinely different result rather than sugar. Speed unmeasured, and 2D is one thread per output element with no tiling |
| `conv_transpose` | 1D only, as `torch.nn.functional.conv_transpose1d` — the decoder half of `conv`, and what a DAC-style codec upsamples with. Weight is **`[Cin, Cout/groups, K]`**, the transpose of `conv`'s layout; `padding` **crops** the output rather than extending the input; `output_padding` lengthens the trailing end only and takes no part in the sum. The kernel is *not* flipped, for the same reason `conv`'s is not. `weight_norm` is an offline conversion, not a flag here. Speed unmeasured |
| `upsample` | `nearestUpsample2d`: nearest-neighbour 2D resample over `[N, C, H, W]`, as `F.interpolate(x, size=(outH, outW), mode='nearest')` — the **`size=` path**, not `scale_factor=`. The two are not two spellings of one thing: at `H = 3`, `scale_factor=1.6` maps the four output rows `0, 0, 1, 1` and `size=(4, ...)` maps them `0, 0, 1, 2` (both measured), and only the size path is decided by integers rather than by how a float scale rounds. Source index is `floor(dst * f32(inSize / outSize))` **computed in f32**, torch's OpenCV `INTER_NEAREST` formula — at `H = 14 -> 46`, destination row 23 copies source row **6**, where exact integer arithmetic says 7. `align_corners` is not a parameter, because torch *raises* for it in this mode rather than defaulting it; `mode='nearest-exact'` is a different function (`floor((dst + 0.5) * scale)`) and is not implemented. Downsampling **throws**. No weights and no arithmetic on the values — this is the `nearest upsample -> conv` half of a decoder that avoids `conv_transpose`'s checkerboard, so a decoder wants both ops and neither substitutes for the other. Speed unmeasured |
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

## The q4 format (W4A32, group-128) — issue #137

The second quantized weight format in this library, beside `quantize`'s per-row
int8. `matvecQ4G128` reads it.

| | value | why not the alternative |
| --- | --- | --- |
| bits | 4, **8 codes per `u32`**, least-significant nibble first | `packQ8`'s byte order at half the width, so a converter that already emits q8 changes constants, not code |
| range | **`[-7, 7]`**, symmetric | `[-8, 7]` (Q4_0) clips one tail only, which turns quantization error into a systematic bias — see below |
| scale | `absmax / 7`, **f32**, one per **128 contiguous columns** of a row, `[N, ceil(K/128)]` | per-row loses too much at 4 bits; `g64` was not distinguishable from `g128` by anything but bpw |
| reciprocal | `7 / absmax`, formed in f64 | `1 / f32(scale)` (llama.cpp) rounds differently at code boundaries |
| rounding | `Math.round` — ties toward `+Infinity` | matches `ops/quantize` and `llm/tools/quant_common.py` exactly |

**This is not Q4_0 and does not claim to be.** It is in the same family
(per-block symmetric absmax, no zero point), and differs in three ways that
change the numbers: block 128 rather than 32, `[-7, 7]` rather than `[-8, 7]`,
f32 scale rather than fp16. The group axis is the one GPTQ/AWQ call
`group_size`, but those carry a zero point and this does not. Q4_K / Q4_1 —
asymmetric, with a per-block minimum — are not implemented.

### What is measured here, and what is not

**Measured in this repository** by `ops/matvec/q4.quality.test.ts` — CPU only,
so it cannot pass vacuously for want of a GPU; `npm run test:file
ops/matvec/q4.quality.test.ts` reprints the table. N=256, K=2560, deterministic
LCG input, f64 reference arithmetic. These are **synthetic matrices, not model
weights**:

| matrix | weight RMS rel: q8-row / q4-row / q4-g128 | GEMV peak rel: q8-row / q4-row / q4-g128 |
| --- | --- | --- |
| iid gaussian | 7.97e-3 / 1.45e-1 / 1.14e-1 | 9.44e-3 / 2.08e-1 / 1.50e-1 |
| + a 20x outlier column every 64 | 4.05e-2 / 3.83e-1 / 2.90e-1 | 3.22e-2 / 2.85e-1 / 2.33e-1 |
| per-128-column magnitude bands (1x/10x/100x) | 1.32e-2 / 1.68e-1 / 1.14e-1 | 1.88e-2 / 2.25e-1 / 1.36e-1 |
| …the same, **restricted to the 1x columns** | 7.13e-1 / **1.00e+0** / 1.15e-1 | — |

The last row is the whole argument for the group axis, and it is the reason the
three rows above it look unimpressive: a global RMS-relative figure is dominated
by whichever columns are largest, so it barely moves. Restricted to the columns
that are two orders of magnitude *smaller* than their row's peak, per-row q4
scores exactly **1.0** — every code in those columns rounds to zero and the band
is annihilated — while `g128` is unaffected at 1.15e-1. Note that q8's per-row
scale loses those columns too (7.13e-1): the axis is doing the work here, not
the bit width. An iid matrix cannot show any of this, because there is nothing
for a per-group scale to adapt to, and neither can uniformly-spread outliers —
both are in the table so that "group-wise is better" is not read as
unconditional. Real weights are neither.

Wire size, measured from the buffers `packQ4` actually produces (same test):
**4.250 bpw** for q4-g128 at any `K` that is a multiple of 128 — 4 bits of code
plus one f32 scale per 128 weights — against 4 + 32/K for per-row q4 and
8 + 32/K for q8. The group axis costs **a quarter of a bit per weight**.
A `[2560, 2560]` weight is 26,214,400 bytes as f32, 6,563,840 as q8, and
3,481,600 as q4-g128.

### Does a 4B model fit?

Arithmetic, from the measured 4.250 bpw: 4e9 parameters at 4.25 bits is
**2,125,000,000 bytes ≈ 1.98 GiB**, against 4.02 GiB at q8 (8.012 bpw) and
16 GiB at f32. That is the number issue #149 exists for, and it is arithmetic
rather than a measurement — this repository has no 4B checkpoint, has converted
none, and therefore does not know what the non-Linear tensors (embeddings,
norms, biases) add on top, nor whether any single tensor exceeds
`maxStorageBufferBindingSize` on a given browser. Issue #112 records that
exceeding a limit is answered with **zeros rather than an error**, so "it fits
in total" is not the same claim as "it loads".

**Not measured here**: what this does to a real model. No logits comparison, no
greedy-trajectory comparison, no audio. Issue #137's decisions rest on
voxshot's measurements on MioTTS-0.6B (per-row q4 at 4.5e-1 peak-relative
logit error against q8's 4.8e-2; `[-8, 7]` flipping the argmax in 4 of 4 cases
despite the best weight RMS error of any configuration tried), and those are
**voxshot's numbers, not this repository's**. The harness that produced them
(`spike/miotts/measure_q4.ts`) does not use these kernels, so running it against
weights quantized by `quantizeQ4G128` would isolate this implementation's own
error — that comparison has not been run.

**Speed is unmeasured**, for the kernel and for the format.

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
`upsample` ✅ (nearest, 2D) · `add` · `mul` · `gather` ✅ · `scatter` ✅ ·
`transpose` ✅ · `reduce` ✅
`matmul` (GEMM) ✅ · `matvec` (GEMV) ✅ · `conv` ✅ (1D, 2D) · `conv_transpose` ✅ (1D) ·
`add` · `mul` · `gather` ✅ · `scatter` ✅ · `transpose` ✅ · `reduce` ✅

Small, total, boring. Everything else is built from these, and they are where
target-specific tuning pays off most.

### `kernel/` — one fused, named operation

`rope` ✅ · `rmsnorm` ✅ · `layernorm` ✅ · `group_norm` ✅ · `softmax` ✅ ·
`activation` ✅ · `snake` ✅ · `elementwise` ✅ · `axpy` ✅ · `quantize` ✅ · `dequantize` ✅ ·
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

Position: `RoPE` ✅ · `multi-axis RoPE (ropeAxes)` ✅ · `ALiBi` ✅ · `PoPE` ✅ · `YaRN` ✅ · `NTK scaling` ✅ · `rotary cache` ✅ · `half-RoPE (head range)` ✅

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
// The engine itself is source-only — `llm/tokenizer`, `llm/sampler`,
// `llm/kv-cache` and `llm/reshape` are published (0.2.0), but `LlamaEngine`
// and everything under it is not, so this is a within-repo import rather
// than a package one.
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
`tiny.weights.bin`. `ropeAxes` pairs adjacent channels too — Z-Image's
`torch.view_as_complex(x.reshape(*, -1, 2))` is the same convention — so the
rule is per checkpoint, not per op: `rotate_half` weights need the permutation
whichever of the two rotates them.

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
logits alone). Prefill was unchanged in scope at issue #110's own landing
(decode only) — the first `forward()` call delegated whole to a
`LlamaEngineQ8` instance internally; every call after that had to be
exactly one token. **Issue #117 (below) made prefill resident too** —
the rest of this section documents #110's own measurement as it stood at
the time, prefill delegation included, since that is the comparison its
own numbers were taken against; see "GPU-resident prefill +
effective-seq-length bound" below for the current prefill path and its
own numbers.

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
gate/up into one *packed weight buffer* so one `matvecQ8` dispatch computes
all of them — worthwhile there because each dispatch used to cost a full
submit+readback round trip. That reasoning does not carry over to a
resident engine that already pays one submit for an entire token's
dispatch chain, and fusing a *buffer* costs something real here: splitting
a fused output by a byte offset into one buffer only works when that
offset is a multiple of `minStorageBufferOffsetAlignment` (measured 256
bytes on this device) — the tiny fixture's own `kvDim = 32` floats lands
`v`'s slice at byte 384, not a multiple of 256. `LlamaEngineQ8Resident`
gives every distinct tensor its own buffer instead — that part is
unchanged by issue #111, and still true for Q/K/V, which stay three
separate buffers *and* three separate dispatches, same as at issue #110's
own landing (`harness/resident.ts#bindGroup`'s doc has the validation
error that caught the alignment problem above).

**What issue #111 changes is *dispatch* count, not this buffer design.**
`q8_ffn`/`q8_residual` (below) still read gate/up, and each residual
projection's weight, from their own separate buffers — no packed-weight
buffer, no byte-offset splitting, the alignment problem above is still
sidestepped exactly the same way — but now cost one dispatch each instead
of the "a few more dispatches than a fused version" this paragraph
described before #111 existed. Buffers separate, dispatches fused: the two
are independent axes, and #111 only moved the second one.

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

### GPU-resident prefill + effective-seq-length bound (issue #117)

Two gaps the #110 section above names explicitly as left open: prefill still
paid `LlamaEngineQ8`'s per-op round trips (12.0–12.7s for a 69–76 token
prompt), and decode's attention always scanned `S = maxSeqLen`, relying on
the causal mask to skip the arithmetic past the real KV length rather than
not scanning it at all.

**`ops/gqa` gains `sEff`.** A uniform-passed parameter that bounds the
softmax scan itself, leaving `S` as `k`/`v`'s address stride unchanged —
default `sEff = S`, so every existing caller (`ops/gqa`'s own reference,
`llm/kernels.ts#runGqa`) is unaffected unless it opts in. `LlamaEngineQ8Resident`'s
decode step now passes `sEff = position + 1` instead of always `maxSeqLen`.
See `ops/gqa/reference.ts`'s own doc for the safety contract (`sEff >=
min(S, L + queryOffset)` whenever `causal` is true; rejected outright
otherwise) and `ops/gqa/wgsl-seff.test.ts`'s `SEFF_CASES`/`seffEquivalence()` for
why `context.wgsl`'s bound is mutation-checkable numerically (`+Infinity`
poison past `sEff`, read unconditionally there, produces `NaN` if the scan
isn't bounded) while `scores.wgsl`'s is proven by agreement with `S` shrunk
outright instead — the causal contract makes its own bound numerically
invisible under any legal input, a fact worth stating rather than a test gap.

**Prefill is resident now.** `runPrefillResident` (`llm/engine-q8-resident.ts`)
encodes every prompt token's pass through every layer into the same flat op
list `runDecodeStep` builds — one `queue.submit` for the whole prompt, one
readback (the **final** position's logits only — `forward()`'s prefill
return shape changed to `[finalPositionLogits]`, matching what every real
caller already read). It kept a `matmul`-shaped path rather than `matvecQ8`
per token — that would multiply prefill's weight traffic by the prompt
length, the opposite of the goal (still true today: see "int8 prefill
matmul (matmulQ8)" below, which changed *which* `matmul`-shaped kernel runs,
not this decision). Two new ops existed for this at the time:

- `ops/permute` — the `[tokens, heads, dim]` token-major <-> `[heads,
  tokens, dim]` head-major reshape `ops/gqa` needs, `LlamaEngineQ8`'s own
  CPU-side `llm/reshape.ts` functions ported to one GPU dispatch (reading
  Q/K/V back to the CPU to reshape, then re-uploading, would put a round
  trip exactly where going resident removes one). Still part of prefill's
  own kernel set today.
- `ops/dequant_transpose` — dequantized a packed int8 weight and transposed
  it to `matmul`'s `[K, N]` operand shape in one GPU dispatch. Measured
  necessary, not merely nicer, at the time: the equivalent three CPU passes
  (`packInt8Rows` → `dequantizePackedQ8` → `transposeRowMajor`, the detour
  `LlamaEngineQ8#project`'s own prefill branch takes from *already-packed*
  resident weight) cost ~100ms/layer on Sarashina2.2-1B's shape — **~2.5s
  for 24 layers, on the critical path of every prefill call** — while
  packing raw codes (a byte copy, no arithmetic) and dequantizing+
  transposing on the GPU instead brought that down to ~170ms total. This
  single change took measured prefill time from 2.96s to 1.24s for a
  76-token prompt (see the table below) — the single largest contributor
  found in this issue's work, ahead of the round-trip elimination it was
  layered on top of (which itself took 76-token prefill from 8.0s to 2.96s
  by removing the CPU pack+dequant+transpose *three-pass* detour's earlier,
  even slower form — see the PR for the intermediate numbers).

  **No longer part of prefill's own kernel set** — issue #128 (below)
  replaced this dispatch and plain `matmul` together with `matmulQ8`, which
  reads the packed int8 weight in-kernel and needs neither. `ops/dequant_transpose`
  itself still exists and is still exercised elsewhere (`llm/kernels.ts#runDequantTranspose`'s
  own Node-side integration test, and `examples/llm-demo`'s WGSL parity
  table), so this bullet is left as issue #117's own historical record
  rather than rewritten — the "int8 prefill matmul (matmulQ8)" section is
  where prefill's *current* weight-read path is described.

KV writes go straight into the persistent, `maxSeqLen`-strided cache
`runDecodeStep` reads: one contiguous `copyBufferToBuffer` per head (each
moving that head's whole `N`-position block at once, since prefill's own
attention runs against *transient* `[kvHeads, N, headDim]` buffers at `S = N`
— already the tight bound, so `sEff` stays at its default there — not the
persistent cache).

**Prefill, three prompt lengths (RTX 5090, Chrome, real Sarashina2.2-1B-alibi-v1
checkpoint, `LlamaEngineQ8Resident`, same session):**

| case | prompt | prefill | vs. `LlamaEngineQ8` (#110 table, same prompts where reused) |
| --- | --- | --- | --- |
| short (`full_gear`) | 76 tok | **1.24s (61.2 tok/s)** | 12.0–12.7s (6.0–6.4 tok/s) |
| mid (`brush_off`) | 70 tok | **1.27s (55.0 tok/s)** | 12.0–12.9s (5.4–5.8 tok/s) |
| 聞く層級 (`listen`, few-shot + grammar) | 365 tok | **1.35s (269.0 tok/s)** | 13.4s (27.0 tok/s, unoptimized table above) |

All three land in the 1-second range (issue #117's own completion bar,
"1秒台以下"), including the longest, few-shot-heavy prompt — prefill tok/s
actually *rises* with prompt length here because the fixed ~1.1–1.2s of
per-generation cost (weight pack+dequant+transpose, `matmul` dispatch
overhead) is amortized over more tokens, while the CPU-dequant portion that
used to dominate is now flat regardless of `N` (`ops/dequant_transpose`'s
own doc). Output text is byte-for-byte identical to the pre-#117 recordings
above at the two prompts kept identical (`full_gear`: `【talk2】新しいGPU
買ったから、ベンチマーク回した。\n【talk】……意外と速くて、地味に驚いた。`;
`listen`: `policy: brush_off\ntopic: セグフォ`) — greedy tokens did not move,
only how fast they arrived. (`brush_off`'s prompt text differs slightly
from the #110 table's, so its output is not expected to match verbatim; it
is style-consistent with the earlier `【base】昨日推しのライブ配信見てたら、
朝になってた。` recording.)

**Decode, re-measured with `sEff` (same session, same three prompts):**
231.1 tok/s (`full_gear`, position ~76), 213.1–226.8 tok/s (`brush_off`,
position ~70), 194.9–200.6 tok/s (`listen`, position ~365) — run-to-run
variance on this box is real (a repeated `full_gear` measurement read
172.8 tok/s once, with a single-step latency spike mid-run; every other
repeat clustered at 220–231 tok/s), reported rather than smoothed over, per
rule 9.

**`sEff`'s own before/after, isolated by mutation** (revert `scores.wgsl`/
`context.wgsl`'s scan bound from `s_eff` back to `S` — the exact mutation
`ops/gqa/wgsl-seff.test.ts` catches — rebuild, remeasure in the same browser
session, then restore):

| position | with `sEff` | with `S` (pre-#117 equivalent) | speedup |
| --- | --- | --- | --- |
| ~76 (`full_gear`) | 231.1 tok/s | 176.1 tok/s | **1.31x** |
| ~365 (`listen`) | 200.6 tok/s | 167.7 tok/s | **1.20x** |

**Position dependence, within one run** (issue #117's own ask —
"シーケンス位置依存があるので生成序盤/後半を分けて"): a 400-step decode run
from `full_gear`'s prompt (`?maxDecodeSteps=400&ignoreEos=1`,
`examples/llm-demo`'s new debug switches — `stepMs` recorded per step,
`window.__lastRun`), position 76 -> 476:

| window | position range | tok/s |
| --- | --- | --- |
| first 50 steps | 76–126 | 230.4 |
| last 50 steps | 426–476 | 190.3 |

A **17.4% slower** decode step at the deeper position, from `sEff` itself
growing every step — a signature the pre-#117 code could not have had,
confirmed by the same long-run shape under the `S`-reverted mutation: 176.1
tok/s at position 76 vs. 167.7 tok/s at position 365 (5% apart, within this
box's ordinary run-to-run noise, i.e. flat). Both readings are still small
next to the ~85–89% of decode time `matvecQ8` weight bandwidth accounts for
(roofline table below) — `sEff`'s attention-scan saving grows with position
but starts from a small base at Sarashina2.2-1B's shape (`kvHeads=8`,
`headDim=112`, well under `maxSeqLen=4096`'s own asymptote).

**Roofline, updated.** Same machine, same calibration (`harness/roofline.ts`:
1.707 TB/s), same model (Sarashina2.2-1B-alibi-v1-q8, 1,410,513,920 resident
weight bytes/token, → **0.826 ms/token, 1210 tok/s bandwidth-bound lower
bound** — unchanged, since neither the hardware nor the weight footprint
changed):

| case | measured decode | ms/token | % of lower bound | #110's own figure |
| --- | --- | --- | --- | --- |
| `full_gear` | 231.1 tok/s | 4.33 | **19.1%** | 13.4% |
| `brush_off` | 213.1 tok/s | 4.69 | **17.6%** | 11.3% |
| `listen` | 200.6 tok/s | 4.99 | **16.6%** | (not measured at #110) |

A real improvement (+4.2–5.7 percentage points, +31–39% relative), entirely
attributable to `sEff` removing the `maxSeqLen` scan — decode's dispatch
count and structure are otherwise unchanged from #110, so the remaining gap
to 100% is still `#110`'s own "~800 GPU op launches per token" term
(unmeasured individually, same reasoning as that section), not a new one
this issue introduces.

**`weightTBuf`, reused per shape instead of allocated per layer** (PR #119
review, recommended item). `runPrefillResident`'s dequant step wrote each
layer's dequantized-and-transposed `matmul` operand (`ops/dequant_transpose`,
above) into a **fresh** `GPUBuffer` every layer, for all 24 layers × 7
projections (`wq`/`wk`/`wv`/`wo`/`wGate`/`wUp`/`wDown`) — none of them ever
`destroy()`ed until the whole engine was, so a single prefill call's peak
resident footprint included all `24 × 7` of these buffers alive at once, even
though only one layer's worth is ever read at a time (the per-layer `for`
loop's own `await`-ordering already guarantees each projection's dequant
write strictly precedes that same layer's `matmul` read, before the next
layer's dequant overwrites it). Since a `GPUBindGroup` binds to a buffer
*object*, not its contents, the fix is to allocate one `weightTBuf` (and one
`matmulGroup` bound to it) per projection **shape** — the 7 shapes recur
identically across all 24 layers — build all 7 once before the layer loop
(`setupMatmulProjectionShape`), and re-run only the dequant dispatch into the
same buffer each layer (`dequantIntoShape`), with every transient buffer
(packed int8 weight, scale, dequant uniform) explicitly `destroy()`ed after
`batch()` runs, and the 7 shared `weightTBuf`s destroyed once at the very end
of the call. Correctness verified unaffected: `llm/engine-q8-resident.wgsl.test.ts`
and `llm/engine-q8-resident.limits.wgsl.test.ts` report the same worst
abs/rel diffs against the fixture before and after this change (prefill
`abs=1.19e-7, rel=2.02e-5`; decode `abs=1.49e-7, rel=7.14e-4`).

Node/Dawn (the `webgpu` npm package this test suite runs against) exposes no
live VRAM query API, so the reduction below is the same config-derived byte
arithmetic this README already treats as measurement elsewhere (the
resident-weight-bytes/token roofline figures above) — computed directly from
`SARASHINA_2_2_1B_CONFIG`'s real dimensions
(`hiddenSize=1792, numHeads=16, numKvHeads=8, headDim=112, ffnHidden=6272,
numLayers=24`), not estimated:

| | before (per layer, never destroyed) | after (shared across all layers) |
| --- | --- | --- |
| `weightTBuf` bytes, all 7 projections | 173,408,256 (165.4 MiB) × 24 layers | 173,408,256 (165.4 MiB), once |
| peak `weightTBuf` residency | **4,161,798,144 (3.88 GiB)** | **173,408,256 (165.4 MiB)** |

A **~3.71 GiB** reduction in this one buffer family's peak residency, on top
of whatever the packed-int8 weight/scale/uniform buffers already freed by
being `destroy()`ed per layer rather than accumulated (43,430,912 bytes/layer
× 24 = 0.97 GiB, if none of those were ever destroyed either — the case
before this fix for the ones `buildMatmulProjection`'s old per-layer
`weightTBuf` allocation sat beside). The reviewer's own "5–7 GiB" estimate
folds in KV-cache and per-token activation buffers this change does not
touch (those were already `destroy()`ed on their own lifetimes, unrelated to
`weightTBuf` sharing); this measurement is scoped to the specific buffer
family the fix changes, not a claim about total prefill VRAM peak.

**Scope not touched, same reasons #110 gave:** kernel fusion (#111),
f16 activations, speculative decode. Prefill's own `S = N` attention (this
issue) is a separate design point from decode's `sEff`, not the same
mechanism reused — see `llm/engine-q8-resident.ts`'s class doc, "Prefill's
own KV is scanned at S = N, not S = maxSeqLen".

### `reset()`: multi-generation reuse without rebuilding (issue #120)

`forward()`'s own contract, up through issue #117, was "prefill exactly
once, then decode only" — every new *independent* prompt needed a whole new
`ResidentDevice` and `LlamaEngineQ8Resident.create()` call, since there was
no other way back to "accepts a new prefill". `technologies-moe/alibi-ai`'s
chat integration (this repository's real consumer) measured what that costs
in practice: 17-33s per independent turn, dominated by `create()`'s own
`device.upload()` calls rebuilding every persistent `matvecQ8` weight buffer
from the ~1.4 GiB checkpoint — for a workload (a chat: many short,
independent generations against a model that never changes between turns)
where none of that needed to happen again. This is a **within-repo (source-only)
API** — the same status the engine layer has had since issue #98: the leaf
modules `llm/tokenizer`, `llm/sampler`, `llm/kv-cache` and `llm/reshape` are
published (0.2.0, issue #138), but the engines and the storage layer around
them are not, so a consumer of `reset()` needs a source/`llm/` import rather
than a package one. Not a blocker for
this repository's own real consumer: `alibi-ai` already imports `llm/`
straight from source and bundles it with esbuild, not through `npm install`,
so this status quo is unchanged by issue #120 and no `exports` work is
required to use `reset()` there. Widening the published surface is tracked
as a separate, later concern if anyone needs it.

`engine.reset()` sets the position counter back to 0 and re-arms `forward()`'s
prefill routing, and touches nothing else — every pipeline, bind group and
persistent buffer `create()` built stays exactly what it was, weights are not
re-uploaded, and the next `forward()` call is accepted as a fresh prompt the
same way the very first one was.

**Contract change: the CPU-side quantized weights are now kept for the
engine's whole lifetime, not dropped after the first prefill.** Before this
issue, `LlamaEngineQ8Resident` held its `LlamaWeightsQ8` argument only until
its first `forward()` call finished, then set the field to `null` so it
could be garbage-collected. `reset()` needs those same CPU-side packed
weights again for every later prefill's dequant-transpose step, so this
class now keeps that reference for as long as the instance itself lives —
in a long-lived tab (exactly `reset()`'s own target workload), that is an
extra ~1.4 GiB kept reachable from this instance for however long it stays
alive. In practice this costs nothing *additional* for this repository's own
demo and for `alibi-ai`: both already keep their own reference to the same
weights object for the whole page session regardless (`examples/llm-demo/src/main.ts`'s
`loaded.weights`), so this instance holding a second reference to the exact
same object is one pointer, not another 1.4 GiB — but a caller that used to
rely on `LlamaEngineQ8Resident` being the *sole* owner of that memory, and
dropped its own reference right after `create()` expecting the engine to
release it after the first prefill, will now see that memory stay live for
as long as the engine instance does. No `releaseWeights()` escape hatch
exists for that caller today — a caller in that situation should not call
`reset()` (a single-generation engine, discarded and rebuilt like before
issue #120, still works exactly as it always did) rather than expect this
memory to free itself mid-lifetime.

**Correctness: old KV cache contents are left in place, not cleared.**
Nothing clears them — the next generation's own prefill only ever *writes*
into the KV cache (never reads it), and decode's attention is bounded by
`sEff = position + 1` (issue #117), which starts from the same `0` a freshly
`create()`d engine would after `reset()`. So the old generation's leftover
bytes, wherever they still physically sit in the buffer, are simply never in
scanning range again — proved rather than assumed:
`llm/engine-q8-resident.reset.wgsl.test.ts` overwrites the **entire** KV
cache with `+Infinity` (`debugPoisonKVCache`, test/debug-only) before
`reset()`, then runs the exact same fixture prompt through the same engine a
second time and checks the result against the fixture's own ground-truth
logits — unchanged, and no `NaN` (`context.wgsl` reads `v` unconditionally
within its `sEff` bound, so a scan that leaked into the poison would have
produced `NaN`, not a plausible wrong number). That poison test carries its
own **positive control** (PR #126 review, item 2 — measured to matter: an
earlier version of `debugPoisonKVCache` that silently wrote nothing still
passed every other assertion in the file): *before* calling `reset()`, one
more `forward()` call is made on the still-poisoned, still-unreset engine —
routed into decode, whose `sEff` bound at that point covers exactly the
region just poisoned — and its own logits are asserted non-finite, proving
the poison actually reached the buffer the real scan reads before the parity
check below is trusted at all. A second test uses a *shorter* second
generation (prefill plus exactly one decode step, checked against
`fixture.decodeLogits[0]`, so `runDecodeStep`'s own `sEff` bound is
exercised too, not only prefill's) — genuine (non-poisoned, but still
`reset()`-irrelevant) fixture-derived values from the first generation's own
later positions sit beyond the second generation's final position and must
equally not be read; the first test alone cannot tell "correctly bounded"
apart from "coincidentally the same length as before". A third test starts
a `forward()` call, calls `reset()` while it is still in flight (before
`await`ing it), and checks that the in-flight call *rejects* rather than
silently writing a stale position after `reset()`'s own write (PR #126
review, item 3 — `reset()`'s writes are synchronous but
`runPrefillResident`/`runDecodeStep` only write `this.tokensSoFar` after
their own GPU work resolves, so a `reset()` racing a still-pending
`forward()` used to be silently undone the moment that pending call finished;
a `generationEpoch` counter, bumped by `reset()` and checked once each
`forward()`-family method's GPU work resolves, makes that a thrown error
instead — see `reset()`'s own doc for why this cannot, and does not need to,
undo that stale call's GPU-side KV writes, only its effect on
`tokensSoFar`). All of `reset()`'s own two-line body and both `assertSameEpoch`
call sites are individually mutation-verified: reverting any one of them
reddens exactly the test(s) built to catch that specific line and nothing
else (confirmed via `md5sum` before/after each edit, per this repository's
rule 1).

**Measured effect (RTX 5090, Chrome, real Sarashina2.2-1B-alibi-v1
checkpoint, real hardware over CDP — not estimated, rule 9).** Three
independent, short casual-Japanese prompts — 76, 68 and 71 tokens under this
demo's own style-prompt template (`buildStylePrompt`) — greedy decode to
`</s>`, driven through the exact same `generate()` the demo's own button
calls: "build once, `reset()` before generations 2 and 3" against "build
fresh for every generation" (`examples/llm-demo/src/main.ts#__resetBenchmark`,
exposed on `window` for a CDP script rather than reachable from the UI).

A first measurement round (three trials, `reset()` strategy always run
first) reported a 35.2% average reduction, but PR #126 review correctly
flagged that as potentially an ordering artifact: this box's own `create()`
calls can cost more later in a page session than earlier, and running
`reset()` first meant only the *rebuild* strategy's three `create()` calls
ever landed in the "later, more expensive" part of the session. Re-measured
with the order counterbalanced — four trials, alternating which strategy
runs first (`order: "reset-first" | "rebuild-first"`, threaded through
`__resetBenchmark` so a driving script controls it explicitly) — and with
`create()`'s own cost timed separately from `generate()`'s in both
strategies (two `performance.now()` pairs, not inferred from `prefillMs`,
which structurally cannot include `create()`'s cost at all — item 6's own
finding about the first round's write-up):

| trial | order | reset() total (3 gens) | rebuild total (3 gens) | reduction |
| --- | --- | --- | --- | --- |
| 1 | reset-first | 6,633.9 ms | 10,131.5 ms | 34.5% |
| 2 | rebuild-first | 6,683.9 ms | 12,459.8 ms | 46.4% |
| 3 | reset-first | 6,201.8 ms | 10,093.0 ms | 38.6% |
| 4 | rebuild-first | 6,100.4 ms | 11,907.8 ms | 48.8% |
| **avg** | — | **6,405.0 ms** | **11,148.0 ms** | **42.5%, ~4.7s saved / 3 gens** |

Grouped by order: reset-first trials averaged **36.5%** reduction,
rebuild-first trials averaged **47.5%** — the *opposite* of what an
order-artifact would predict (if going first favoured a strategy, `reset()`
running first in trials 1/3 should have shown the *larger* gap, not the
smaller one). The counterbalanced numbers land close to the first round's
35.2% and, if anything, a bit higher — so the original headline was not an
artifact of running `reset()` first, and the effect holds up (36-49%
reduction) regardless of which strategy goes first.

**`create()`'s own cost, isolated** (the actual answer to what "session
position" does to it, replacing the first round's `prefillMs`-based guess):
averaged **1,589.0 ms** across the four single `create()` calls the `reset()`
strategy ever pays (n=4, one per trial) against **1,914.5 ms** across the
twelve `create()` calls the rebuild strategy pays (n=12, three per trial,
every position in the session). A real difference (~325 ms, `create()`
costing more when it is not the session's first GPU work) but a modest one
— nothing close to the first round's apparent ~2,500 ms gap, which came from
conflating `create()`'s cost with `generate()`'s inside an un-split
`prefillMs` figure rather than from timing `create()` on its own. Per-`create()`
variance was real and visible in both strategies (individual calls ranged
roughly 1.5-2.8 s) and is attributed to this box being under genuine
concurrent GPU load from other processes during this measurement (see
below), not to `reset()` or `create()` themselves.

**Measurement conditions (rule 9):** this machine was not idle during either
round — other, unrelated GPU/browser workloads from concurrent processes
were running throughout, and the counterbalanced round's per-`create()`
variance above is consistent with that contention rather than with anything
`reset()` or `create()` do internally. A fully quiet re-run was not achieved
(a shared machine with other legitimate concurrent work, not something this
measurement could control) — reported as a real condition of these numbers,
not smoothed over. The **aggregate** conclusion (`reset()` reliably beats
rebuild by roughly a third to a half, for this three-generation workload, on
this hardware) is robust across both orderings and both measurement rounds;
the *exact* percentage is sensitive to concurrent system load and should be
read as "42.5% under this run's conditions", not a hardware constant.

**Scope not touched:** multi-session/concurrent generation and prompt-prefix
caching (shared-prefix KV reuse across *different* prompts) are explicitly
out of scope for this issue — tracked as future work, not attempted here.

### Fused decode kernels (issue #111)

Issue #110's own doc ("**~800 GPU op launches per token**", above) left the
remaining decode-time overhead as "not broken out further than that" —
resident decode already pays exactly one `queue.submit`/readback per token
(that was #110's whole point), so the cost left on the table is *encoding
and pass-boundary* overhead per dispatch, not round-trip latency. Issue #111
fuses two shapes of that pattern that repeat every layer, on `ops/matvec`'s
existing `q8` entry point rather than a new op:

- **`q8_ffn`**: `silu(wGate·x) * (wUp·x)` — reads both gate and up weights
  in one pass over `x`, applies `silu` and the elementwise multiply inline,
  in registers, instead of four separate dispatches (`matvecQ8(gate)`,
  `matvecQ8(up)`, `activation(silu)`, `elementwise(multiply)`).
- **`q8_residual`**: `residual + w·x` — folds the post-projection residual
  add into the same dispatch as the projection, for both `o_proj` (post-
  attention) and `down_proj` (post-FFN), replacing `matvecQ8` + a separate
  `elementwise(add)` each.

Both are decode-only (`llm/engine-q8-resident.ts#runDecodeStep`) — prefill
(`runPrefillResident`) keeps its existing unfused `matmul` +
`activation` + `elementwise` path unchanged, per this issue's own scope
("プリフィル専用最適化はスコープ外"): prefill's projections go through `matmul`
(a tiled GEMM against all prompt positions at once), not `matvecQ8`, so
these two fused entry points do not apply there. Per decode layer this cuts
8 dispatches to 3 (the FFN triad 4→1, each of the two residual-add pairs
2→1), taking one token's real dispatch count from **411 to 291** at
Sarashina2.2-1B's shape (17→12 per layer × 24 layers, plus the unchanged
final norm and two `lm_head` chunks — see issue #110's own count above) —
a 29.2% reduction.

**Correctness gate.** Both fused kernels have their own `reference.ts`
functions (`matvecQ8Ffn`, `matvecQ8Residual`, `ops/matvec/reference.ts`),
composed from `matvecQ8` (already the correctness definition for a
quantized GEMV row) plus `silu`'s formula copied verbatim from
`ops/activation/reference.ts` (rule 7) rather than re-derived, and their own
WGSL-vs-reference test files (`ops/matvec/q8_ffn.wgsl.test.ts`,
`q8_residual.wgsl.test.ts` — split from `q8.wgsl.test.ts` for the same
per-process GPU dispatch ceiling reason, issue #38). Both kernels' mutation
coverage was confirmed by hand (`silu` dropped to identity, the residual
add dropped entirely — each caught by a *different* failure shape: a
numeric mismatch for the first, a dropped-binding validation error for the
second, since `layout: "auto"` removes a binding an unused entry point
variable stops referencing — both reverted after, md5-confirmed). The full
resident-engine fixture gate (`llm/engine-q8-resident.wgsl.test.ts`,
comparing every decode step's logits against `LlamaEngineQ8`'s own
pre-optimization fixture) passes unchanged with the fused kernels wired in
— worst diff `1.19e-7` abs / `1.26e-4` rel across a full generation, inside
the same tolerance issue #110 already used.

**Measured (rule 9 — same machine as the "Real-hardware verification" section
above: NVIDIA GeForce RTX 5090, driver 610.57.04, Linux (Arch, kernel
7.1.5-arch1-2), Chrome 151.0.7922.71 non-headless via CDP, backend Dawn/Vulkan;
real Sarashina2.2-1B-alibi-v1 int8 checkpoint, `LlamaEngineQ8Resident`,
synthetic-but-in-range token ids so prompt length is exact rather than
whatever a real tokenizer happens to produce — see
`examples/llm-demo/src/main.ts#__decodeFixedCostBenchmark`).** One
`create()`, `reset()` between prompt lengths (issue #120), 20 decode steps
per prompt length, first step of each excluded from the decode average
(`llm/engine-q8-resident.wgsl.test.ts`'s own layer-count-independent
first-dispatch warm-up cost, ~13ms both before and after — unrelated to
this issue's fusion, present in both columns equally). Single observations
from one run each (baseline, then fused), not averaged across repeated
runs — comparable only against a rerun under the same conditions, same as
every other timing figure in this README (rule 9):

| | baseline (pre-#111) | fused (#111) |
| --- | --- | --- |
| decode, N=76 prompt | 4.347 ms/tok (230.0 tok/s) | 4.153 ms/tok (240.8 tok/s) |
| decode, N=365 prompt | 4.595 ms/tok (217.6 tok/s) | 4.147 ms/tok (241.1 tok/s) |
| decode, combined (n=38 steps) | 4.471 ms/tok (223.7 tok/s) | 4.150 ms/tok (241.0 tok/s) |
| prefill, N=76 | 1180.6 ms (64.4 tok/s) | 1168.9 ms (65.0 tok/s) |
| prefill, N=365 | 1279.8 ms (285.2 tok/s) | 1267.2 ms (288.0 tok/s) |

**Decode: 7.2% lower latency / 7.7% higher tok/s** (4.471ms → 4.150ms),
consistent with a 29.2% dispatch-count cut buying a smaller-than-proportional
wall-time win on hardware this fast — dispatch encoding/pass-boundary
overhead is real but is not the dominant term in a 4ms decode step here, so
the win is genuine but modest. **Prefill is within measurement noise of
itself** (both columns ~1% apart), exactly as expected: prefill's own code
path was not touched by this issue.

One more thing this table makes visible, and worth stating plainly because
it bears on this issue's own opening motivation: **the "prefill costs about
the same at 76 tokens as at 365" fixed cost this issue's own background
cites is real (1180.6ms vs. 1279.8ms — 4.8x the tokens for 1.08x the time)
but lives entirely in prefill**, which both this measurement and this
issue's own written scope ("スコープ外: プリフィル専用最適化") leave untouched.
Decode's own fixed cost — the thing #111 actually fuses — was never the
~1.2s-per-`forward()` number; it is the much smaller (~4ms) per-token figure
above, and this issue's fusions do measurably shrink it (7-8%), just not the
1.2s figure a reader might expect from the issue's own framing. Fusing
prefill's own dispatch count (its `matmul`+`dequantTranspose`+`permute`
chain, a structurally different kernel set from decode's `matvecQ8`) is
tracked as separate, future work.

### int8 prefill matmul: `matmulQ8` (issue #128)

The "future work" the previous section names directly: issue #117's
`ops/dequant_transpose`+plain `matmul` pair dequantized-and-transposed every
projection's packed int8 weight into a `[K, M]` f32 buffer once per
`forward()` call, purely so `matmul` had an operand to read — a full GPU
write of `inFeatures * outFeatures` f32 values immediately followed by a
full read of the same, on every prefill regardless of prompt length.
`matmulQ8` (`ops/matmul/reference.ts`) removes that pass: the tiled GEMM
kernel reads the packed int8 weight directly, in-kernel, using `matvecQ8`'s
own `[M, ceil(K/4)]` wire format — no transpose, no intermediate buffer.
`runPrefillResident` packed and uploaded each layer's weight straight into
`matmulQ8`'s bind group at this issue's own landing (issue #142, below,
removed that per-call pack/upload too — it turned out to be the actual
dominant cost, not the GPU kernel this section measures).

**Measured, and the result is smaller than this issue's own hypothesis —
reported as measured, not as hoped for (rule 9).** `examples/llm-demo`'s
`__decodeFixedCostBenchmark` (RTX 5090, NVIDIA driver 610.57.04, Chrome
151.0.7922.71, real Sarashina2.2-1B-alibi-v1 checkpoint, synthetic token ids
so the comparison is about prompt *length* only, one `LlamaEngineQ8Resident`
instance with `reset()` between calls — the "聞く層+スタイラ, reset() 運用"
shape this issue asked about), 8 samples
per prompt length across three separate runs, comparing this change against
the immediately preceding commit (`dequant_transpose`+`matmul` still in the
prefill path):

| prompt length | before (`dequant_transpose`+`matmul`) | after (`matmulQ8`) | delta |
| --- | --- | --- | --- |
| 76 tok | 1197.0ms avg (1157.5–1273.3ms, n=8) | 1175.0ms avg (1138.8–1200.7ms, n=8) | ~22ms (~1.8%) |
| 365 tok | 1275.9ms avg (1262.0–1308.7ms, n=8) | 1273.0ms avg (1253.7–1285.4ms, n=8) | ~3ms (~0.2%) |

(The "before" column here reproduces the #111 table above almost exactly —
1180.6/1279.8ms there vs. 1197.0/1275.9ms here — a useful cross-check that
both measurements are reading the same real cost, not an artifact of one
session. **A later session's own measurement of this same N=76 prefill,
against this same documented hardware/software configuration, read ~2000ms
instead — see "Where prefill's ~1.2s fixed cost actually goes" below for
the direct same-page cross-check and why this gap is reported as
unexplained rather than assumed away.**)

Both deltas sit inside the run-to-run spread of either condition (roughly
±30-50ms at these prompt lengths, on an otherwise-idle GPU — `nvidia-smi`
read 0% utilization between runs, so this is not contention from another
process on the machine). `ops/dequant_transpose/reference.ts`'s own doc
measured its *GPU-side* cost at ~170ms total across 24 layers back at issue
#117 — real, but small enough next to prefill's ~1.15-1.28s fixed cost that
removing it outright lands inside this measurement's own noise floor rather
than producing a visible win.

The fixed cost this issue set out to explain — 76-token and 365-token
prompts landing at effectively the same wall-clock time (`alibi-ai`'s own
"プリフィル固定費≈1.2秒がトークン数非依存" observation) — reproduces exactly
in both the before and after numbers above, so it is real. Its source is
**not** primarily `dequant_transpose`, contrary to this issue's own working
hypothesis. The most likely remaining source, unmeasured here and out of
this change's scope: `runPrefillResident`'s per-layer `Promise.all` of nine
`device.bindGroup()` calls (the #117 section above already names the
`pushErrorScope`/`await popErrorScope()` round trip inside each one) still
runs once per layer — 24 sequential awaited rounds per prefill call,
independent of prompt length. A plausible next target, not a confirmed
cause; finding out needs its own measurement, not assumed here.

Correctness: `llm/engine-q8-resident.wgsl.test.ts`'s fixture gate (prefill
and decode logits vs. the pre-optimization engine) passes unchanged — abs
diff ~1.2e-7 for prefill logits, matching the pre-existing float32 rounding
noise this fixture's tolerance was already sized for, not a new source of
error. `ops/dequant_transpose` itself is untouched and still exercised —
`llm/kernels.ts#runDequantTranspose` (its own Node-side integration test
path) and `examples/llm-demo/src/browser-runtime.ts`'s WGSL parity table
both still reference it; only `runPrefillResident`'s one production call
site moved to `matmulQ8`.

### Where prefill's ~1.2s fixed cost actually goes (issue #131)

The previous section's own PR (#130) removed `dequant_transpose` and found
the ~1.2s fixed cost barely moved (22ms/1.8% at 76 tokens, inside noise) —
the working hypothesis it set out to test was **wrong**, and its own
write-up named a next candidate without measuring it: `runPrefillResident`'s
per-layer `matmulQ8IntoShape` packs and re-uploads every projection's weight
**on every `forward()` call** (`packInt8Rows`, CPU, then `queue.writeBuffer`,
~1 GiB total) even though the *same* packed bytes are already resident on
the GPU for decode (`buildProjection`, built once in `create()`). This issue
does not optimize anything — per its own scope, it only measures which of
four candidates (CPU pack, transfer, GPU kernel time, CPU
submit/bindGroup/readback overhead) the ~1.2s actually is, so a fix (if one
lands) has a target instead of a guess.

**Method.** `ForwardProfile` (`llm/engine-q8-resident.ts`), an opt-in
argument threaded through `forward()`/`runPrefillResident()`/`runDecodeStep()`
— `undefined` by every existing caller, so passing nothing changes no
dispatch, no bind group, and no arithmetic (proved directly: a profiled
prefill call's logits are asserted bit-for-bit identical to an unprofiled
one, `llm/engine-q8-resident.profile.wgsl.test.ts`). When given, it records:

- `packInt8Rows`'s own CPU cost, `performance.now()`-timed per layer per
  projection (item 1) — necessarily wall-clock, not `timestamp-query`: this
  work runs entirely on the CPU, before any compute pass exists, so a GPU
  timestamp query cannot see it at all.
- `queue.writeBuffer` bytes and CPU enqueue time (item 2).
- Per-call `device.bindGroup(...)` await time, and — because prefill issues
  nine of those per layer inside one `Promise.all(...)`, so their individual
  await windows overlap — **also** the wall-clock time of that whole block
  (`layerSetupMs`), which is the number that is actually additive against
  the call's own total (item 3; the per-call sum is not — measured directly,
  see below).
- `harness/resident.ts#batch()` extended with an opt-in `BatchProfile`: GPU
  submit-to-completion wait via `queue.onSubmittedWorkDone()`, the readback
  `mapAsync` phase timed separately from that, and — when the device
  negotiated `timestamp-query` — one GPU-side duration per labeled dispatch,
  via a dedicated compute pass per labeled dispatch (WebGPU's
  `GPUComputePassTimestampWrites` only covers a whole pass, not a point
  inside one, so per-dispatch attribution costs a pass boundary; real,
  measurable overhead, confined to this opt-in path — item 4). Ported
  identically into `examples/llm-demo/src/browser-resident-runtime.ts`
  (this repository's established Node/browser duplication for this exact
  module). No fallback branch was exercised on this measurement's own
  hardware — `timestamp-query` was negotiated on every run below
  (`ResidentDevice.timestampsSupported`/`ForwardProfile.gpuTimestampsSupported`
  both `true`); an environment without the feature gets `submitToDoneMs`/
  `readbackMs` (no GPU feature needed) but an empty, explicitly-flagged
  `gpuEntries` rather than a fabricated zero.

**A real bug turned up before this measurement could be trusted.** An
earlier version of `batch()` split every dispatch into its own compute pass
whenever `BatchProfile.labels` was present *at all* — not gated on whether
the caller actually wanted the GPU breakdown (`wantGpuBreakdown` below did
not exist yet). That earlier version's own first measurement round reported
`packInt8Rows` alone (1689.9ms) as *larger* than prefill's entire total from
this same README's own #128 section (1175.0ms) — impossible for a sub-phase
of that same call, and review caught it before merge. `ForwardProfile` now
takes a second, separate opt-in, `wantGpuBreakdown` (default `false`): only
when a caller explicitly sets it does `runPrefillResident`/`runDecodeStep`
build a per-dispatch `labels` array at all, so a caller who only wants the
CPU-side fields below (`packEntries`, `uploadMs`, `layerSetupMs`,
`submitToDoneMs`) never pays the pass-splitting cost, and `batch()`'s own
pass-splitting branch is gated on `wantsGpuTiming` (device support **and**
caller intent), not merely a `labels` array being present.
`__prefillProfileBenchmark` was also extended to run a **genuinely
unprofiled control `forward()` call, in the same session, immediately
before every profiled one** — the fix this bug actually needed: the
original comparison was against a *different* PR's *differently-conditioned*
measurement, not against this run's own unprofiled baseline, so there was
no way to tell instrumentation overhead from ordinary session-to-session
variance. Re-measured below against that control.

**This section's own control (2097.7ms, table below) is ~1.8x the ~1175.0ms
the "int8 prefill matmul" section above reports for the same N=76, under
what both sections describe as identical conditions (same GPU/driver/Chrome/
checkpoint, same `reset()`-between-generations cadence) — flagged in review
as a discrepancy this README must not leave silently standing next to each
other. Checked directly rather than left as two competing numbers: **in one
page load**, `__decodeFixedCostBenchmark` (the exact function the ~1175.0ms
figure above came from) was called, then `__prefillProfileBenchmark`
immediately after, then `__decodeFixedCostBenchmark` again — three calls,
same session, same loaded weights, same device:

| call | N=76 prefill |
| --- | --- |
| `__decodeFixedCostBenchmark` (1st) | 1969.0ms / 2096.0ms (two repeats) |
| `__prefillProfileBenchmark` control | 2041.5ms / 2065.4ms |
| `__decodeFixedCostBenchmark` (2nd, after the other function ran) | 2112.5ms / 2048.2ms |

**The two functions agree with each other (1969-2112ms across both, no
function-attributable gap) and neither is anywhere near 1175.0ms.** This
rules out "which benchmark function measures it" as the explanation — the
*same* function that produced 1175.0ms for #128 now measures ~2000ms in the
same repository, same machine, same documented conditions. What changed is
not identified: this run's own environment (this machine, at measurement
time) is genuinely slower at this workload than whatever state it was in
when #128's own numbers were recorded, for a reason this measurement did
not isolate further — GPU/driver/Chrome version and checkpoint identity were
all checked and match, so the gap sits somewhere those fields do not
capture. Per rule 9, this is reported as **unexplained**, not guessed at:
the #128 section's own 1175.0ms and this section's own numbers below were
both real measurements on this repository, just not comparable to each
other as an absolute value — only this section's own control-vs-profiled
comparison (same session, same run) is being used for #131's own
conclusion.

**Measured (rule 9 — RTX 5090, NVIDIA driver 610.57.04, Linux/Arch kernel
7.1.5-arch1-2, Chrome 151.0.7922.71 non-headless via CDP on a dedicated
`--user-data-dir`/`--remote-debugging-port`, adapter `vendor: nvidia,
architecture: blackwell`, backend Dawn/Vulkan; real Sarashina2.2-1B-alibi-v1
int8 checkpoint, 24 layers, `hiddenSize=1792`, `vocabSize=102400`;
`examples/llm-demo/src/main.ts#__prefillProfileBenchmark`, synthetic-but-
in-range token ids so prompt length is exact; one `LlamaEngineQ8Resident`,
`reset()` between every generation, including between each prompt length's
own control and profiled call). Machine load at measurement time: `load
average 0.59, 0.77, 0.75`, top CPU consumers 4-5% each (confirmed with the
reviewer independently, after an earlier round's contention claim turned
out to be stale idle processes rather than real load — corrected here
rather than left in). 5 samples per prompt length with `wantGpuBreakdown:
false` (the cost figures below), 3 more with `wantGpuBreakdown: true` (the
GPU row below, plus an independent overhead check) — all percentages are
against **this run's own control**, not a number from a different PR:

| | N=76 (mean, range) | N=365 (mean, range) |
| --- | --- | --- |
| **control `forward()`, unprofiled** (n=5) | **2097.7ms** (2071.9–2155.7) | **2178.5ms** (2152.1–2216.7) |
| profiled `totalMs`, `wantGpuBreakdown: false` (n=5) | 2089.2ms — **overhead −0.4%** (range −1.0% to +0.7%) | 2199.6ms — **overhead +1.0%** (range −0.6% to +2.2%) |
| `packInt8Rows` CPU sum (item 1), % **of control** | 1686.7ms (**80.4%**) | 1700.1ms (**78.0%**) |
| `queue.writeBuffer` CPU enqueue (item 2) | ~38ms | ~38ms |
| `queue.writeBuffer` bytes | ~1.04 GiB | ~1.04 GiB |
| submit→GPU-done wait (item 4, CPU-visible half) | 41.6ms | 117.0ms |
| profiled `totalMs`, `wantGpuBreakdown: true` (n=3) | 2084.7ms — overhead −0.8% | 2179.4ms — overhead +0.5% |
| GPU kernel time, timestamp-query (item 4, n=3) | 35.17ms (**1.67% of control**) | 114.19ms (**5.27% of control**) |
| decode control / profiled `totalMs` | 27.1ms / 27.3ms | 13.0ms / 13.9ms |
| decode `packEntries` / `bindGroupCalls` (every sample) | 0 / 0 | 0 / 0 |

**`packInt8Rows` alone accounts for ~78-80% of prefill's own wall time,
measured against this run's own unprofiled control — instrumentation
overhead is noise-level (−1.0% to +2.2%) in both profiling modes, so this
is not an artifact of the earlier pass-splitting bug.** GPU kernel time is
1.7-5.3% and scales with `N` as expected for a GEMM's row count (35ms at
N=76 → 114ms at N=365), while control `totalMs` itself barely moves
(2097.7ms → 2178.5ms, 3.9% for 4.8x the tokens) — the same fixed-cost-
independent-of-prompt-length shape issue #128/#130 already established,
now attributed to a specific phase with a same-session control behind it.
Decode's own profile is the direct contrast: `packEntries`/`bindGroupCalls`
are exactly zero on every single sample (`runDecodeStep` never calls
`matmulQ8IntoShape` — its bind groups were all built once, in `create()`),
and its ~13-27ms total tracks its own control closely.

**This transfer buys nothing, and the fix is already known (tracked as
#142, not attempted here — see Scope below).** `buildProjection`
(`create()`, line ~269, decode's resident weight) and `matmulQ8IntoShape`
(`runPrefillResident`, line ~431, prefill's per-call weight) both call
`packInt8Rows(linear.codes, ...)` on the **same underlying `linear.codes`
array** — identical bytes, identical function, only the argument names
differ. `buildProjection`'s own upload already sits on the GPU, resident,
for the whole engine's lifetime; `matmulQ8IntoShape` re-derives and
re-uploads that same ~1 GiB from scratch on every `forward()` call and then
discards the buffer once `batch()` resolves. The two call sites differ only
in bind-group order (`[weight, scale, vector, out, uniform]` for decode's
`matvecQ8` vs. `[a, weight, scale, out, uniform]` for prefill's `matmulQ8`)
— binding the same resident `GPUBuffer` at the position `matmulQ8` expects
removes the CPU pack and the GPU upload both, with no new kernel and no
precision change.

`bindGroupMs` (the sum of each individual `device.bindGroup()` call's own
await, not wall-clock) is reported in the code but **not** in the table
above deliberately — under the *pre-fix* code it reached ~11.5-12.1s on a
76-token prefill whose own `totalMs` was ~2.1s, because prefill's nine
per-layer calls run concurrently under `Promise.all` and their individual
await windows overlap; summing them is not a wall-clock quantity and would
misstate this section's own conclusion if read as one. `layerSetupMs`
(wall-clock around the whole `Promise.all` block, tracked closely with
`packInt8Rows` above in every sample of this run) is the field to use
instead — see `ForwardProfile`'s own doc for the full reasoning.

**Scope.** No optimization is included here — per this issue's own
completion condition, only the breakdown. The fix named above (keep
decode's resident weight/scale buffers alive and bind them into `matmulQ8`
instead of re-packing per call) is tracked as its own follow-up issue (#142)
so a correctness-first, one-change-at-a-time PR can review it against this
measurement as its own baseline.

### Resident weight buffers in prefill, not a re-pack per call (issue #142)

The previous section's own "most likely remaining source, unmeasured here"
guess (`runPrefillResident`'s per-layer `Promise.all` of nine
`device.bindGroup()` calls) turned out not to be it. Issue #131/#141
actually measured where prefill's ~1.2–2.2s fixed cost went, instead of
guessing further: `packInt8Rows` (CPU, inside `matmulQ8IntoShape`) alone was
**77–79%** of it; the GPU kernel `matmulQ8` itself was only 2–6%.

The decisive fact issue #142 acts on: the bytes `packInt8Rows` produced
fresh on every `forward()` call were **already sitting on the GPU**.
`buildProjection`/`buildFfnProjection`/`buildResidualProjection`
(`llm/engine-q8-resident.ts`) call the exact same `packInt8Rows` on the
exact same `weights.layers[l]` object once in `create()`, to build decode's
own `matvecQ8`/`matvecQ8Ffn`/`matvecQ8Residual` bind groups — the output is
byte-identical by construction (same function, same input), and only the
*position* in the bind group differs between decode's layout
(`[weight, scale, vector, out, uniform]`) and `matmulQ8`'s
(`[a, weight, scale, out, uniform]`), which has nothing to do with the
bytes underneath.

Those three builders now keep the resulting `GPUBuffer` handles
(`ResidentWeight`, one pair per projection per layer) instead of discarding
them once the decode bind group is built. `runPrefillResident`'s
`matmulQ8` bind group (`bindMatmulQ8`, replacing `matmulQ8IntoShape` for
every production projection — `wq`/`wk`/`wv`/`wo`/gate/up/`wDown`) binds
those same resident buffers directly: no `packInt8Rows`, no
`queue.writeBuffer`, no new `GPUBuffer`, on any prefill call, including the
very first one. (`matmulQ8IntoShape` itself is not deleted — `lmHead`'s
own debug-only, full-`N`-position path, `debugAllPositionLogits`, still
uses it: `lmHead`'s only resident buffers are chunked by
`matvecQ8`'s decode-shaped row limit, not by `matmulQ8`'s `[M, K]` tile
shape, and re-deriving that mapping for a path real generation never runs
was not worth it — see `matmulQ8IntoShape`'s own doc.)

**VRAM residency does not change.** These bytes were already resident for
decode; prefill now reads the same allocation instead of paying for a
second, transient ~1 GiB one that lived only until that call's own
`batch()` resolved and was then `destroy()`ed. Nothing is kept alive for
longer than before — this is reuse, not new residency.

**Measured** (RTX 5090, NVIDIA driver 610.57.04, Linux (Arch, kernel
7.1.5-arch1-2), Chrome 151.0.7922.71 non-headless via CDP on a dedicated
`--user-data-dir`/`--remote-debugging-port`, real Sarashina2.2-1B-alibi-v1
int8 checkpoint (24 layers, hiddenSize=1792, vocabSize=102400),
`examples/llm-demo`'s `__decodeFixedCostBenchmark`, one
`LlamaEngineQ8Resident` instance, `reset()` between prompt lengths, 8
samples per prompt length, control and fixed measured in the same session
via `git stash`/rebuild between them — machine shared with other concurrent
sessions throughout, `uptime` read `load average: 0.6–1.3`, not idle):

| prompt length | before (`matmulQ8IntoShape`, per-call repack) | after (`bindMatmulQ8`, resident) | speedup |
| --- | --- | --- | --- |
| 76 tok | 1266.4ms median (1256.9–1277.0ms, n=8) | 42.7ms median (38.3–45.0ms, n=8; one 233.6ms outlier under concurrent GPU load) | ~29.7x |
| 365 tok | 1344.4ms median (1326.9–1378.9ms, n=8) | 120.6ms median (119.0–121.6ms, n=8; one 317.0ms outlier under concurrent GPU load) | ~11.1x |

This is the "prefill's ~1.2s fixed cost" this file's own prefill sections
have been chasing since issue #117 — not shaved at the margin, but removed
down to roughly the same order of magnitude as decode's own per-step cost.
Development on this change started from the commit immediately before #141
merged `ForwardProfile` (same base SHA as the previous section's own
measurement, before it landed on `main`), so this table is
`performance.now()`-timed `prefillMs` — the same metric
`__decodeFixedCostBenchmark`'s own earlier tables in this file already use —
not the `packMs`/`layerSetupMs` breakdown the previous section's own numbers
give; it was not re-derived here. This section's own control (1266.4ms/
1344.4ms median) sits below the previous section's own control (2097.7ms/
2178.5ms mean) for the same N=76/365 — a different session, different
concurrent load (see the previous section's own "unexplained... reported as
measured" note for an earlier instance of this exact same class of
session-to-session gap on this shared machine); both are real numbers for
the code they measured, and the ~29.7x/~11.1x speedup below is against
*this* section's own control, measured in the same session as the "after"
column.

Correctness: prefill and decode logits are **bit-for-bit identical**
whether `matmulQ8`'s weight comes from `bindMatmulQ8` (resident) or the
pre-#142 `matmulQ8IntoShape` (packed) — `debugPrefillWithPackedWeights`
(`llm/engine-q8-resident.ts`, test/debug-only) runs the packed path on
demand so `llm/engine-q8-resident.residentweight.wgsl.test.ts` can diff it
against a plain `forward()` prefill **on the same engine, in the same
session, on the same device**, rather than against a golden capture pinned
to one GPU/driver. `rmsnorm`'s `rsqrt`, `rope`'s `sin`/`cos` and `gqa`'s
softmax `exp` are all vendor/driver-dependent at the ULP level (rule 2), so
a literal-float fixture is reproducible only on the machine that captured
it — an earlier version of this test used exactly that shape and was
caught and replaced in review before merge, since it is not portable the
way an in-session A/B diff is. Exact equality, not within tolerance, per
issue #130's own established criterion ("1ビットでも動いたら丸めではなく設計の
違い"): both paths read the identical packed bytes through the identical
`matmulQ8` kernel, so any difference would mean a design bug, not
floating-point rounding. `resident.stats.buffersCreated`'s own
per-prefill-call delta drops from a measured 75 to a measured 47 for that
fixture (`numLayers: 2`) — 28 fewer buffers, exactly `2 (weight + scale) ×
7 (projections) × 2 (layers)`, the weight/scale pair this change stops
re-allocating — asserted with a margin threshold in the same test file,
mutation-confirmed by temporarily reverting `bindMatmulQ8` back to
`matmulQ8IntoShape` during development and watching the assertion fail at
the measured pre-#142 value (75).

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
| WGSL | 31 ops | a kernel per op, per target; resolution is `<entry>[.<target>][.<dtype>].wgsl` |
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

MIT — **and that covers the code in this repository, not any model it runs.**

Every model this repository can run is somebody else's, under somebody else's
terms, and those terms are not MIT and do not become MIT by being loaded here.
Nothing in `ops/`, `llm/` or `harness/` contains model weights; the examples
tell you where to fetch them and the licence you accept when you do.

Anima, which `examples/anima/` runs, is the sharpest case: it is
**non-commercial**, and this library is not.

| | licence | |
| --- | --- | --- |
| this repository | MIT | code only |
| `circlestone-labs/Anima` and anything derived from it, including `lylogummy/Anima-3.8B` and any conversion of it | [CircleStone Labs Non-Commercial License v1.2](https://huggingface.co/circlestone-labs/Anima/blob/main/LICENSE.md) | **non-commercial and non-production use only** |
| its base, `nvidia/Cosmos-Predict2-2B-Text2Image` | [NVIDIA Open Model License](https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-open-model-license) | commercially usable, but stacked under the above; requires a "Built on NVIDIA Cosmos" attribution |
| the text encoder and VAE, upstream `Qwen/Qwen3-0.6B-Base` and `Qwen/Qwen-Image` | Apache-2.0 | distributed inside the CircleStone repo, whose §9 keeps third-party terms in force |

Two licences apply to the Anima weights at once, and **the more restrictive one
governs**: NVIDIA permits commercial use, CircleStone does not, so the answer is
non-commercial.

A copy of Anima converted to this repository's int8 format is published at
[`m96-chan/Anima-3.8B-q8-web-xpu-ops`](https://huggingface.co/m96-chan/Anima-3.8B-q8-web-xpu-ops),
because the demos need somewhere to fetch from. **Converting it did not change
its licence**, which is why that repository carries the licence texts and the
verbatim attribution its terms require rather than this one's MIT.

Z-Image and the LLM demo carry their own upstreams' terms in the same way; see
each example's README.

Running MIT code against a non-commercially-licensed checkpoint does not make
your use non-commercial-exempt, and putting the two side by side does not merge
them. If you intend to build a product on any of this, read the model's licence,
not this line.
