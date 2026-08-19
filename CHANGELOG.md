# Changelog

Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/); versions
follow [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

Entries record **why** a change was needed. What changed is in the diff.

## [Unreleased]

### Added

- `llm/`: a config-driven llama-architecture inference engine built by
  composing existing ops (`gather` → N × [`rmsnorm` → fused QKV projection →
  `rope` → `gqa` + a pre-allocated f32 KV cache → O projection → residual →
  `rmsnorm` → fused gate/up projection → SiLU-gated MLP → residual] →
  `rmsnorm` → `lm_head`), rather than a new op — issue #98. Prefill uses
  `matmul`, decode uses `matvec`; Q/K/V and gate/up are fused into single
  projections purely to keep this repository's `webgpu` (Dawn) binding within
  the dispatch count it can sustain in one device's lifetime on some
  machines, not for FLOPs. Correctness is defined by a `transformers`-generated
  fixture (`llm/tools/gen_fixture.py`, committed at `llm/fixtures/tiny.*`,
  436 KiB): an 8-token prefill and a 4-step greedy decode on a tiny random
  model, checked on a real GPU to a measured tolerance (worst observed
  absolute diff 1.49e-7, relative 3.4e-4) and exact greedy-token equality.
  `SARASHINA_2_2_1B_CONFIG` documents the real target model's dimensions;
  running it needs a weight converter and tokenizer, both later issues under
  #96. Needed because wllama's WASM CPU path is too slow for in-browser
  inference (technologies-moe/alibi-ai#3: 22.6s few-shot prefill) and WebLLM's
  4-bit-only quantization collapses on the target Japanese model.

## [0.1.0] - 2026-08-04

### Documentation

- The README's op count, backend table and roadmap match what is on disk:
  **twenty-seven** ops, where both count lines still said twenty-four. This is
  the second time these same lines have gone stale, and the reason is structural
  rather than careless — op PRs land in parallel and each is told to leave the
  shared count line alone, since otherwise every one of them conflicts on it. The
  count is therefore correct when it is written and wrong again once the next
  batch merges. `group_norm` had also reached the per-op table without reaching
  the roadmap's `kernel/` layer, which is the same drift wearing a different
  shape. The per-op rows were right throughout; only the totals and that one
  roadmap tick were not. The count line still states outright that speed is
  unmeasured for every op, since rule 9 treats an omission as a claim of speed.

  Nothing here stops it happening a third time. What would is a test asserting
  the count against `ops/` on disk, the way `harness/coverage.ts` already asserts
  that a kernel variant cannot exist without a test looping it — worth doing, and
  deliberately not smuggled into a release-preparation change.

- The README says how to install this and what to import. It had neither: written
  as a design document, it could tell a reader why `scatter` accumulates but not
  how to call it. An op's two halves are now given as two imports, because they
  run in different places — the reference is dependency-free TypeScript that runs
  anywhere, while the kernel is a `.wgsl` file whose loading belongs to the
  caller's bundler and is documented as such rather than guessed at here.

### Added

- `istft` synthesises `"same"` padding, the vocoder convention: `(nFft - hop) / 2`
  cropped from each end, so `T` frames give exactly `T * hop` samples. torch has
  no such mode and it cannot be composed from one — `center: false` returns edge
  samples whose `w²` envelope is zero, so the signal a caller would slice cannot
  be produced in the first place. This is the last stage of a MioCodec-style
  decoder, and the point where latents become audio. `center` keeps working and
  is unchanged; `padding` is the new spelling and giving both raises.

- `group_norm` — `torch.nn.functional.group_norm`, pooling statistics over a
  group of channels while applying the affine transform per channel. Added
  because a codec decoder's residual blocks are `GroupNorm → SiLU → Conv1d` and
  nothing here could express the reduction: `layernorm` over `[N·C, L]` gives one
  mean per channel, where GroupNorm pools `C/G` of them into each. Same input,
  same output shape, no error either way, different numbers — the shape of bug
  this library exists to keep out of a caller's code.

  Both numerical conventions were measured against torch 2.10 rather than
  inherited from `layernorm`, since `group_norm` is a separate kernel upstream
  and its agreement is worth one measurement: the biased variance matched to
  4.4e-16 while the unbiased form was out by 1.6e-1, and moving `eps` outside
  the square root moved the answer by 4.3e-6 — small, but four thousand f32
  ulps and not noise. `C % G != 0` throws, as torch does; distributing a
  remainder would be a different op that happens to run. The layout is `[N, C,
  L]` contiguous, which makes a group one unbroken run rather than a stride.
- `snake_beta`, as a second entry point of `ops/snake` — `x + sin²(α·x)/β`, with
  α and β both learned per channel. Added because "Snake" turns out to name two
  different functions and the library shipped only one of them: DACVAE's has a
  single α, while BigVGAN's `SnakeBeta` — which MioCodec's decoder is built from
  — separates the sine's frequency from the amplitude it is added at. β = α
  recovers the one-parameter form, so this generalises rather than replaces.
  Two entry points rather than one kernel with an optional binding, because an
  unreferenced binding throws since #46 and the α-only path would otherwise have
  to supply a β buffer it never reads.

  The reason this needed saying at all is that the two fail into each other
  silently. A checkpoint holds a tensor called `alpha` and nothing in it says
  which function it belongs to — and the BigVGAN family stores **logarithms**
  where DACVAE stores values, so reading one as the other turns a log-scale
  α of `0.0` into α = 0, which makes `sin(0·x)` vanish and collapses the whole
  activation to the identity. No NaN, no shape mismatch: just a vocoder that
  sounds duller than it should. Neither kernel exponentiates — that belongs to
  whoever reads the checkpoint — and both references now say so where a reader
  will look.
- `snake` — `x + sin²(α·x)/α` with a learned per-channel α, as a **separate op**
  rather than a kind of `activation`. Every neural audio codec worth decoding on
  the web is built from it (DACVAE, BigVGAN), and without it the decoder stops
  at latents. It is separate because `activation` is "one buffer in, one buffer
  out, branch on an integer" and Snake's α is a tensor: folding it in would add
  a storage binding and a channel stride that `relu2` and `silu` would then have
  to supply on every call — the pipeline layout is derived from what the shader
  declares, so an unused binding is not free, it is mandatory. `elu`'s scalar
  `alpha` went the other way for the same reason: it costs a uniform field the
  struct was padding out anyway. The `1e-9` sits inside the reciprocal at
  upstream's value, because `1/α` guarded afterwards returns NaN at α = 0 where
  upstream returns x, and a checkpoint that trained an α to nothing would be the
  one to find that out.
- `activation` gains `elu`, `tanh`, `gelu` and `gelu_tanh`. The first three
  complete the set DACVAE's encoder and decoder stages need; GELU is for the
  other half of the same model, whose text encoder is a GeGLU ModernBERT. GELU
  is **two** functions in torch, and they part company by 4.73e-4 at x = 2.699 —
  measured, in the reference — so both ship and the default is the exact `erf`
  form, matching `torch.nn.functional.gelu`'s own default. A library that picked
  one silently would be right for half its callers and quietly wrong for the
  rest.
- `rmsnorm` takes an optional group count: `weight` may be `[G, D]`, and row `n`
  uses group `n % G`. QK-norm gives every attention head its own gamma, and
  until now this op could not say that — a per-head weight flattened into `[N,
  D]` silently applied head 0's gamma to every head, with no NaN and no change
  of shape to notice it by. A wrong answer that looks well-formed is worse than
  a missing op, which is why it is in the library rather than worked around by
  the caller. `G = 1` and an unpacked group word are both today's behaviour, so
  nothing existing moves. The grouped axis must be the one immediately left of
  `D` (`[B, S, H, Dh]`); `[B, H, S, Dh]` needs a different index and is refused
  rather than served wrongly. Speed unmeasured. `layernorm` has the same gap and
  #81 tracks it, so the two do not end up disagreeing about what a weight is.
- An **additive attention mask** on `attention`, `flash_attention` and `gqa`,
  because without one this library had no answer for an encoder at all. Every
  batched encoder over variable-length sequences needs to say "key `j` is
  padding, never attend to it", and `causal` cannot: it masks by position, not by
  content. Padding positions were contributing to the softmax and every
  conditioned output was quietly wrong — not NaN, not a crash, just slightly
  wrong. The form is PyTorch's **float** `attn_mask`, added to the scores before
  the softmax, and the reason it is the float form rather than the boolean one is
  `ops/alibi`: ALiBi already produces an additive score bias and says masking is
  the caller's, so with one representation the two compose by addition instead of
  needing an op to combine them. `keyPaddingBias(B, S, keep)` converts the
  boolean mask people actually hold, with `keep` naming the polarity because
  torch's own two APIs disagree about it. All three ops take the same mask, so it
  does not decide which kernel a caller may use.
- **A fully masked row now returns zeros** rather than NaN. It used to be
  unreachable — `queryOffset >= 0` guaranteed every row saw key 0, and the
  kernels said so and skipped the guard. A key mask breaks that guarantee, and an
  empty sequence in a padded batch reaches it by accident, so it is guarded in
  all three ops: a NaN row is a wrong answer that looks like a result. Zeros is
  what torch returns, through `aten::_safe_softmax` — measured, and worth
  measuring, because plain `torch.softmax` on the same row returns NaN.
- `rope` takes a head range — `headOffset` / `headCount`, defaulting to every
  head and byte-identical there. The joint-attention blocks of a diffusion TTS
  DiT rotate half their heads and leave the other half position-free, so that a
  block attending over positioned latents and unpositioned conditioning has
  heads that cannot see order at all. The layout is `(token, head, dim)`, so the
  heads to rotate are not contiguous and the honest workaround was a gather, a
  dispatch and a scatter — three passes and a temporary, spent to do less work.
  **The range is over heads, not over the channels within a head**: both are
  called "half-RoPE" in the wild, and channel-wise partial rotary (`rotaryDim`)
  is a different op that this deliberately does not add. Heads outside the range
  are copied rather than left at zero — `rope` returns a fresh array, so an
  unwritten head is silent garbage flowing into attention.
- `ops/conv_transpose` — `convTranspose1d` and `convTranspose1dOutputLength`,
  matching `torch.nn.functional.conv_transpose1d`. `conv` covers a codec's
  encoder; nothing covered the decoder, so the library could run the transformer
  half of a diffusion TTS model and then not turn latents into a waveform. A
  DAC-style codec decodes with transposed convolutions and nothing else — no
  iSTFT head to fall back on — which made this a hard blocker rather than a
  convenience (ISSUE #75, #80). Three conventions are stated rather than chosen
  quietly, because each fails silently: the weight is `[Cin, Cout/groups, K]`
  and not `conv`'s transpose of it, which only diverges when `Cin != Cout` —
  exactly when a codec is upsampling; `padding` *crops* the output instead of
  extending the input, so getting it backwards yields a waveform that is merely
  a little too long; and `output_padding` lengthens the trailing end without
  joining the sum, which matters only at odd `stride`, which is what DACVAE's
  decoder stages use. `weight_norm` is deliberately not a flag — folding
  `weight_g`/`weight_v` into a plain weight is a load-time conversion, not
  per-frame kernel work.
- `roofline(runner)` — this device's measured bandwidth and compute ceilings,
  taken here and now rather than derived from a spec sheet, and cached for the
  session. WebGPU exposes no clock, compute-unit count or bus width, so there is
  nothing to compute a theoretical peak from; a self-calibrating ceiling also
  throttles when the device does, which is what keeps a percentage meaningful
  across machines. Null where the device cannot time a dispatch, rather than an
  estimate.
- `Runner.time(dispatch)` — GPU time for a dispatch, read from a timestamp query
  written around the compute pass, and `null` on devices that cannot report it
  rather than a guess. Host wall-clock cannot stand in: it charges the dispatch
  for buffer creation, submission and the readback round trip, about 1.2 ms here
  against a real dispatch of 0.3 ms. Taking a bandwidth ceiling that way reported
  5.3 TB/s on a 1.79 TB/s card — the authoritative-looking, unrelated figure the
  roofline design exists to avoid.
- A `scratch` binding: storage the kernel uses but nobody uploads or reads back,
  so a measurement of transfer rate does not sit next to a transfer of the same
  size. Reused across dispatches rather than reallocated, because repeated large
  allocations abort this binding — a 64 MiB buffer allocated twice kills the
  worker on the second.
- The device now asks the adapter for the limits it actually offers. A device
  requested with none caps storage bindings at 128 MiB where this adapter allows
  2 GB.

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
- `stft` / `istft` — windowed transform and its inverse, the pair ONNX cannot
  express, since it has no complex tensors and so cannot carry the spectrogram a
  vocoder head emits. Every convention follows `torch.stft` / `torch.istft` and
  is checked against it numerically rather than read off the documentation:
  centred by default, reflect padding without repeating the edge sample,
  one-sided, unnormalised, `hannWindow` periodic like `torch.hann_window` and
  `scipy.signal.get_window` rather than symmetric like `np.hanning`. Named
  because each has more than one defensible answer and picking silently means
  half the callers get a subtly wrong waveform. `istft` divides by the
  overlap-added `w²` envelope instead of assuming the window is COLA — a
  periodic Hann at 50% overlap is COLA in `w` but **not** in `w²`, so a
  reconstruction that skipped the division is wrong by up to 2x and still sounds
  like audio. Two departures from torch, both in `ops/stft/reference.ts`: the
  layout is frame-major `[frames, bins]` because a vocoder head emits one row
  per frame, and asking for more samples than the frames reach raises instead of
  quietly returning a zero tail. Speed is **unmeasured**, and the kernels are a
  naive DFT rather than an FFT: 1920 is not a power of two, this sits beside a
  transformer, and correctness came first.
- Kernel resolution — an op's `wgsl/` directory holds one or more **entry
  points**, each of which may have variants, and which file runs is decided by
  `resolve()`: `explicit override → target + dtype → target → dtype → portable`,
  first hit wins, **within one entry point**. The filename carries it:
  `<entry>[.<target>][.<dtype>].wgsl`, so `kernel.wgsl` and `inverse.wgsl` are
  two entry points and `kernel.nvidia.wgsl` is a variant of the first. Entry
  points are named rather than assumed because ops already need them — `stft`
  computes the inverse transform with different arithmetic, `attention` is two
  dispatches split into two files so `layout: "auto"` cannot drop bindings an
  entry point does not reference — and because resolution must never leave the
  entry point it was asked about: `istft` falling back to the forward transform
  would be a wrong answer that still looks like a result. A suffix that is not a
  known target or dtype is an error, and so is a bare `nvidia.wgsl`, which is far
  likelier to be a mis-written variant than an entry point called "nvidia".
  Target detection reads `adapter.info` and is
  allowed to answer "I don't know", because a vendor string does not say what a
  device is good at; an unknown adapter gets the portable kernel rather than
  someone else's. Intel is the standing example — the same vendor string covers
  an on-die iGPU and a discrete Arc card. Because the hint will sometimes be
  wrong, the override beats detection outright, and one that names a variant that
  does not exist raises instead of quietly falling back. The choice is readable
  through `describeAdapter(adapter)` and the returned `Choice`, which names the
  rung that hit and every candidate tried: a wrong guess nobody can see is worse
  than a portable kernel. Reading the adapter is a call the caller makes rather
  than something `createRunner` does on the way past, so nothing changes for ops
  that only want a kernel run.
  Adding a variant cannot skip the reference test — `eachVariant(url, entry, …)`
  builds an op's test loop from its `wgsl/` directory rather than from a list, and
  `unguardedOps` fails the suite for an entry point that grows a variant no test
  iterates. Per entry point, not per op: looping `scores` says nothing about
  `context`. None of it is re-exported from `harness/index.ts`, which every op's
  test imports, so `index.ts` and `suite.ts` are byte-identical to what they were
  before this landed and no existing op's test loads anything new.
- `ctc_decode` — greedy CTC decoding: argmax per frame, collapse repeats, then
  drop blanks, **in that order**. The order is the whole op. Stripping blanks
  first and collapsing afterwards is the implementation everyone reaches for,
  and it turns `a ␣ a` into a single `a` — losing the doubled letter that the
  blank symbol exists to make expressible. The two orders agree on every other
  input, which is what lets the wrong one survive being tested. Blank defaults
  to `0`, as `torch.nn.CTCLoss` has it, and is a parameter because TensorFlow's
  `ctc_greedy_decoder` puts it last and models trained that way exist. The
  result never leaves the GPU: labels come back as `[B, T]` padded with `-1`,
  and the true lengths in their own `[B]` buffer, both written by the kernel —
  a greedy decode emits at most one label per frame, so the allocation never
  depends on the answer. Reading the labels back to find out how long they are
  would be the per-step readback this op exists to avoid. Beam search is
  deliberately absent: it is a different algorithm and needs its own decision
  about where the beam lives.
- `mel` — the filterbank and its application, in two kernels because the matrix
  depends only on scalars and is built once per configuration while the
  application runs per frame. The other half of the DSP gap next to `stft`: no
  ML kernel library ships it because it is not machine learning, so every voice
  encoder reimplements it in numpy, and that numpy is exactly what has to be
  rewritten to move a pipeline into a browser. Four conventions have more than
  one answer in wide use and all four are named rather than picked silently —
  the HTK mel scale, unnormalised triangles, a power spectrum, and a dB log
  flooring its argument at `1e-10`, which together are
  `torchaudio.transforms.MelSpectrogram` + `AmplitudeToDB(stype="power")`.
  Asking for `{ scale: "slaney", norm: "slaney" }` gives `librosa.filters.mel`'s
  defaults instead, and on the same audio those two differ by a factor of 200,
  which is what silently picking one would have cost a caller. Checked against
  torchaudio 2.10 and librosa 0.11 on a real recorded voice, not on a formula.
  `top_db` is deliberately absent: it clamps against the maximum over the whole
  spectrogram, so it cannot be computed before the last frame exists and it
  makes the answer depend on how the caller chunked their audio — that is
  `reduce` then `elementwise`. Speed is **unmeasured**.
- `gather` — row selection for embedding lookup, matching
  `torch.index_select(table, 0, indices)` rather than `torch.gather`, because
  embedding lookup is why the op exists and the two names are close enough to
  pick the wrong one by accident. An index outside `[0, rows)` gathers zeros:
  PyTorch raises there and a kernel cannot, and the alternatives — clamping or
  wrapping — hand back a real embedding for a token that was never in the
  vocabulary, which looks plausible all the way downstream.
- `conv` — 1D only, matching `torch.nn.functional.conv1d`, with `stride`,
  `padding`, `dilation`, `groups` and an optional `bias`. The convention worth
  stating out loud is that PyTorch's `conv1d` is a **cross-correlation**: it does
  not flip the kernel. `F.conv1d([[[1,2,3,4]]], [[[1,10,100]]])` is `[[[321,
  432]]]`, not `[[[123, 234]]]`. The two definitions agree on every symmetric
  kernel, so a library that quietly picks the mathematical one passes every
  hand-written test and then disagrees with PyTorch on real weights. `groups` is
  in from the start because `groups = Cin = Cout` is a depthwise conv, which is
  what the speech front-ends this op exists for actually run. 2D is deliberately
  absent until something asks for it. Speed is **unmeasured**: the roofline it
  should be reported against does not exist yet.
- `attention` — unfused scaled dot-product attention, in two dispatches:
  `softmax(mask(scale * Q @ K^T))` writes the attention matrix, then `@ V` reads
  it. Slower than a fused kernel by construction, and worth having anyway,
  because it is what makes the fused one verifiable — `flash_attention` has no
  other definition of correct to be measured against.
  Conventions follow `torch.nn.functional.scaled_dot_product_attention`, checked
  against torch 2.10 rather than read off the docs, because the two that matter
  both have a plausible wrong answer: `scale` defaults to `1/sqrt(D)` from the
  **query's** head dim even when V's differs, and `causal` is **upper-left**
  aligned, so with a KV cache longer than the query it keeps keys `0..i` rather
  than the most recent `i+1`. That second one is a trap for the case the op
  exists to serve, so masking is parameterised by `queryOffset` — the absolute
  position of query row 0 in the key sequence — which reaches `is_causal=True`
  at `0` and `causal_lower_right` at `S - L` without a second flag.
  Speed is **unmeasured**: the roofline harness it should be reported against
  does not exist yet.
- `rope` gains NTK and YaRN context scaling, as an optional `scaling` argument
  rather than as new ops. Neither scheme changes the rotation — they change
  which frequency each pair rotates at — so forking `rope` would have left three
  copies of the same rotation to keep in step. There is no PyTorch definition to
  follow, so the convention is `jquesnelle/yarn` and `transformers`, which
  agree; YaRN's attention temperature (`0.1·ln(s) + 1`) **is** included, folded
  into `cos`/`sin` as both of those do, and can be overridden for checkpoints
  fine-tuned with a different one. Omitting `scaling` leaves plain RoPE
  identical bit for bit, which is a property of the arithmetic rather than a
  tolerance: the unscaled path multiplies an exact IEEE zero. Speed is
  **unmeasured** — scaling costs one multiply-add per element over plain RoPE,
  and the roofline it should be reported against does not exist yet.
- `rope` gains an optional precomputed angle table, `ropeCache`. The angles
  depend on position and pair and on nothing else, so decoding recomputes the
  same `pow`, `sin` and `cos` for every head at every step; a table trades
  memory for those. What the table does when decoding runs past its end had to
  be decided rather than discovered, because the tempting answers are silently
  wrong: growing it needs a host that a dispatch does not have, and wrapping
  (`pos % positions`) returns a real angle for the wrong position, which comes
  back as a plausible tensor rather than as an error. So it **falls back** to
  computing the angle — past the end it is exactly the uncached op — and a
  table built for a different `thetaBase` or `scaling` is **refused**, since
  that is the same failure by another route. The transcendental saving is
  counted rather than asserted: 3 calls per (token, head, pair) become 0, and
  the table costs 3 per (position, pair) once, so a decode saves a factor of
  `numHeads`. It does **not** show up in wall time on the GPU measured — RoPE
  there is bandwidth-bound, so the calls were never what it was waiting for;
  see the PR for the numbers and the conditions.
- `flash_attention` — the same function as `attention`, in one dispatch, with the
  `[B, H, L, S]` score matrix never allocated. That is the entire difference and
  the entire reason it is a kernel rather than a composition: at any sequence
  length worth fusing for, the score matrix is the largest thing in the
  computation. Tiled online softmax — a running max and a running sum, so a tile
  of 64 keys folds in without a second pass — and the only score storage anywhere
  is that one tile in workgroup memory, which does not depend on `S`.
  Both halves of the claim are tested, because the first half alone would pass a
  kernel that materialised the matrix and read it back. Agreement is checked
  against **both** references, this op's and the unfused one's. Allocation is
  checked by counting the bytes behind the dispatches that produced those
  answers, at shapes where `L` and `S` grow together — the only sweep where
  `seq x seq` and `seq` can be told apart — and holding the total to a second
  difference of zero. Bound bytes at `B=H=1, L=S=n, D=Dv=8`: `128n + 32` here
  against `4n² + 64n + 28` unfused, which is 32_800 against 278_556 at n = 256
  and doubles its advantage with every doubling of the sequence.
  Conventions are inherited from `attention` unchanged — same `scale` default,
  same `queryOffset` mask — and the arg type is imported rather than restated so
  the two cannot drift. Speed is **unmeasured**: the roofline harness is #3 / #4.
  Memory is measured, because memory is what this op is a claim about.
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

- `moe` — MoE routing: `router` (top-k over expert logits and the gate weights),
  `moeDispatch` (tokens reordered into per-expert contiguous runs) and
  `moeGather` (results back in token order, weighted). One op rather than three
  because none of them is usable without the other two, and because the
  decisions that matter are decisions *between* them. Three of those had to be
  made rather than inherited, and all three change the model's output:
  **ties in top-k go to the lower expert index**, which `torch.topk` explicitly
  leaves open and a GPU cannot — otherwise the same token reaches different
  experts on different runs; **capacity overflow drops by rank first, then by
  token index** (GShard, the Switch Transformer, fairseq `top2gating`), so a
  token's first-choice expert outranks another token's second choice, and not
  by arrival order, which would make the set of dropped tokens depend on the
  scheduler; and the **gate is applied in gather and only there**, since
  weighting on the way in puts it inside the expert FFN and weighting at both
  ends squares it, which stays plausible while being wrong everywhere.
  Renormalising the k gates is a caller's flag with no default, because Mixtral
  renormalises and the Switch Transformer must not — at `k = 1` renormalisation
  makes every gate exactly 1 and deletes the gate entirely. Nothing divides by
  an expert's token count: with 64 experts and a short sequence, most experts
  receive nothing on most steps, and that division is `0/0`. Speed is
  **unmeasured**.
- `gqa` — grouped-query and multi-query attention: `kvHeads` query heads share
  one K and one V. One op rather than two, because MQA is GQA with `kvHeads = 1`
  and MHA is GQA with `kvHeads = H` — splitting them would mean two copies of the
  same index arithmetic, one of them less tested.
  The reason to reach for it is a memory number, so here is the number. One
  Llama-3-8B decoder layer, batch 1, 8192 cached positions, 32 query heads, head
  dim 128, f32: the KV cache is **268,435,456 bytes** at `kvHeads = 32` (MHA),
  **67,108,864 bytes** at `kvHeads = 8` (GQA) and **8,388,608 bytes** at
  `kvHeads = 1` (MQA). Over the model's 32 layers that is 8 GiB against 2 GiB — a
  saving of **6,442,450,944 bytes**, which during decoding is usually the
  difference between the model fitting and not. `kvCacheBytes()` is exported so
  the figure stays a computation rather than a claim in a document.
  The head mapping is **contiguous** groups, `kvHead = h / (H / kvHeads)`,
  matching `enable_gqa=True` on torch 2.10 — measured, because the alternative
  reading (strided, `h % kvHeads`) agrees with it at both `kvHeads = H` and
  `kvHeads = 1` and disagrees everywhere in between, which is how an
  implementation passes the two configurations people test and fails the ones
  they ship. `H % kvHeads != 0` throws rather than picking a grouping for the
  caller, as torch does.
  Speed is **unmeasured**: the roofline harness it should be reported against
  does not exist yet.

### Fixed

- The harness refuses a shader that does not compile instead of reading back
  zeros. Output buffers start zeroed, so a dispatch that never ran produced a
  result — and where an expected value contains zeros, that is a passing test
  over a kernel that executed nothing. Every correctness claim here is "it agrees
  with the reference", so a silent no-op reading as agreement could have hollowed
  out all of them at once. Compiled modules are now cached by source as well, so
  the check costs one await per distinct shader rather than one per dispatch.

- The per-file timeout in `npm test` actually fires. It never had: `npx` starts
  vitest as a grandchild that keeps the stdout pipe open, so killing the child
  left `"close"` unfired and the run waited forever. A hanging file took an outer
  120s kill instead of the 6s limit it was given. It hid because a *GPU* hang
  brings the worker down within seconds on its own, which looked like the timeout
  working — the suite could not tell "wedged" from "slow".

### Changed

- The score kernels of `attention`, `gqa` and `flash_attention` take a **mask
  buffer at every dispatch**, not only when there is a mask, and their uniform
  blocks gained three fields for its broadcast shape. A bias of zeros *is* "no
  mask" — it is what torch's own reference does — and making it unconditional
  removes a per-element branch and a `has_mask` guard whose only failing input
  would be a dummy buffer nobody reads. It costs `B * S` floats against the
  `B * H * L * S` these kernels already move. Callers driving the WGSL directly
  must add the binding; `attention` and `gqa` bind it at index 2, `flash_attention`
  at index 3.
- `npm test` runs one test file per vitest process. A single process cannot cross
  a test-file boundary with a GPU device in play — it aborts inside Dawn's thread
  pool or hangs, with no kernel of its own required to trigger it. The runner also
  refuses to report a false pass: a crashed vitest worker can exit 0 having
  skipped most of the suite, so every file is now accounted for individually.

- The package ships **compiled JavaScript and type declarations** instead of
  TypeScript source. `exports` pointed straight at `.ts` files with no build
  behind them, which requires every consumer to own a TypeScript toolchain
  configured the way this repo's is, and bills a browser bundle for comments that
  belong in the source. `npm run build` emits `dist/`, `prepublishOnly` runs it,
  and CI runs it too — the tests import from the source tree and never touch
  `dist/`, so a broken package is invisible to them by construction.
  `removeComments` applies to the JavaScript only: the declarations keep their
  JSDoc, so the reasoning still reaches an editor's hover.

- `harness` is **no longer exported**. It imports vitest and the Dawn binding,
  both devDependencies, so that subpath resolved to code no consumer's runtime
  could load. Nothing imported it that way regardless — the tests reach it by
  relative path — so this removes a promise that was never kept rather than a
  feature anyone had. It stays in the repo as the test infrastructure it always
  was.

- The `.wgsl` kernels are published beside the JavaScript at
  `web-xpu-ops/ops/<op>/wgsl/<entry>.wgsl`, mirroring the source tree so a
  kernel's path is one string across the repo, the resolution grammar and the
  package. `scripts/assets.mjs` copies them, then fails the build if an op
  compiled to a reference with no kernel beside it: `tsc` emits no assets, so
  without that check a package missing its entire WGSL backend would type-check,
  pack and publish without a word.

- Publishing is a **tag-triggered workflow** rather than a command someone runs
  on a laptop. `.github/workflows/release.yml` fires on `v*`, repeats the lint,
  build and test that CI runs — a tag can point at a commit that never went
  through CI — and publishes with `--provenance`, so the tarball is linked to the
  workflow run and commit that produced it rather than being bytes of unknown
  origin. It refuses to publish when the tag disagrees with `package.json`: the
  tag is typed by hand while `npm publish` reads the version from the file, and
  an npm version, once taken, cannot be replaced.

  The token is an npm automation token in a repository secret. Trusted publishing
  (OIDC) would remove that secret, and is the better end state, but npm cannot
  configure a trusted publisher for a package that does not exist on the registry
  yet — so it is a follow-up to the first release, not part of it.

- `package.json` carries `keywords`, `homepage` and `bugs`. None of it changes
  what is installed; all of it is what npm's own page and search have to work
  from, and a package that cannot be found is not meaningfully published.

- The first released version is `0.1.0`, not `1.0.0`. One backend of the three
  described here exists, every op's speed is unmeasured against the roofline it
  should be reported against, and the resolution grammar has target and dtype
  rungs that no kernel yet uses. A major version would claim those are settled.

[Unreleased]: https://github.com/m96-chan/web-xpu-ops/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/m96-chan/web-xpu-ops/releases/tag/v0.1.0
