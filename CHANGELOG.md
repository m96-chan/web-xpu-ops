# Changelog

Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/); versions
follow [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

Entries record **why** a change was needed. What changed is in the diff.

## [Unreleased]

### Fixed

- **Anima and Z-Image's resident paths drew one flat colour** (regression from
  #206). `ropeAxes`' `positions` binding stopped being `array<i32>` and became
  `array<f32>` when it learned fractional positions; `examples/zimage/src/
  dit-gpu.ts` was updated in that commit and the two **resident** DiTs were not.
- **Anima and Z-Image's resident paths drew one flat colour** (issue #217, a
  regression from #206). `ropeAxes`' `positions` binding stopped being
  `array<i32>` and became `array<f32>` when it learned fractional positions;
  `examples/zimage/src/dit-gpu.ts` was updated in that commit and the two
  **resident** DiTs were not.

  Nothing errors. WebGPU copies the bytes, and a small integer's bit pattern
  read as a float is a denormal — so every rope angle became zero, every token
  got the identity rotation, and the DiT returned a well-formed latent whose
  every channel is constant. Bisected: `597d567` is the first bad commit, and
  the latent's per-channel standard deviation goes 1.02 → 0.038 across it.

  `ropeAxisPositionBuffer` in `ops/rope` is where the type lives now, with the
  slack the kernel expects, and both resident paths build their buffer with it.
  Its test asserts the array type, because that is the thing that was wrong.

- **Changing folder to the wrong one stranded the page** (reported from a
  browser). Pointing Z-Image's folder picker at Anima's folder left it dead with
  `Uncaught (in promise) Error: the folder "anima-3.8B" has no
  "model.safetensors.index.json"`, and reloading landed in the same place.

  Three faults, each of which alone would have been survivable:

  - **A receipt was read without asking what the caller needs.** A folder filled
    for another model carries its own valid receipt, so `readReceipt` said
    "filled", the fill was skipped, and the page got a folder with none of its
    files. It takes the plan now: a receipt that does not name every file the
    caller wants means unfilled, which fills the folder rather than failing
    after a reload.
  - **The folder was remembered before it was known to work.** `bindFolder`
    stored the handle first, so a bad pick became the page's permanent answer.
    It reads every file out of the new folder first and stores it last.
  - **Changing folder forgot the folder that worked.** The old rule was "every
    bind failure forgets", which is right when there is nothing to fall back on
    and wrong when there is. `gate.test.ts` used to assert the two paths were
    the same; it asserts they differ, and why.

  And a failure that gets past all of that now lands somewhere a person can see:
  every page's `void main()` carries a `.catch` that writes the message into the
  status line instead of the console.

- **Three limits a real reference hit, and one of them was silent**
  (issues #212, #211). Running R2V on an actual video and image reference — 8
  frames of 256x448 and a 256x352 still, 1,424 packed rows — found all three:

  - **`qkNorm` dispatched one workgroup per head-row**, so the grid was
    `seq * heads`. At 56 heads that passes 65,535 at **1,171 tokens**, which is
    a 256x256 clip with two references. It is split on head-row boundaries now.
  - **The vision tower never flushed.** Every one of its 27 blocks' buffers
    stayed lent for the whole tower, which at 2,144 patches is gigabytes beside
    25.78 GB of weights.
  - **The per-row norms had the same shape of limit** — one workgroup per row,
    so `seq * heads` rows. The conditioner's QK norms reach 65,535 at **1,024
    tokens** at 64 heads and the DiT's at **1,171** at 56, which is any
    presentation with a video reference in it. Both split on row boundaries
    now, and both are `headDim` floats wide, so the slices were already
    aligned.
  - **`swapLeading` refused instead of tiling.** Its own comment said splitting
    needed a second grid dimension in `ops/permute`, which #214 added; it tiles
    now, in the DiT and in the conditioner's tower.
  - **A dispatch past the grid limit is reported as an invalid *command
    buffer*.** That takes every dispatch recorded beside it with it, so the run
    completed — fifteen sampling steps at 91 ms instead of 1,400, and frames
    written from pool debris. `ResidentDevice.batch` refuses now, naming the op
    and the kernel, and the kernel is named from its own WGSL header because
    every entry point here is `main`.

- **R2V takes more than one reference, and takes video** (issue #212).
  `--reference image:PATH:W:H` and `--reference video:PATH:W:H:FRAMES`,
  repeatable, in packed order. A video is one vision block per merged frame
  group with its own timestamp, its rotary clock advances per block, and its
  latent geometry comes back from the encoder rather than being assumed — the
  causal temporal compression is on the way. The layout is built *after* the
  encoder for that reason.

- **The DiT's latent space is not the decoder's, and nothing was converting
  between them** (issue #212). `AutoencoderKLMiniMaxH3`'s own doc says a
  pipeline "encodes with `(latent - latents_mean) / latents_std` and decodes
  with `latent * latents_std + latents_mean`". Every sampler here handed the
  DiT's output straight to `decode`.

  What came back was a blurred frame with a grid over it, and that was read as
  what int8 costs. It is not. With the transform, the same latent decodes to a
  sharp paper boat with reflections in the water and the printed text on its
  hull legible.

  It surfaced while wiring `ref2va`, where the **encoder** is a caller too and
  the two directions have to agree — a round-trip through the encoder and the
  decoder reproduced the picture only in the raw space, while the sampler's
  output only made sense in the normalised one. Both cannot be true of the same
  buffer.

  `unnormaliseLatent` lives beside `decode` now, the statistics ride in the
  decoder manifest so no caller has to find them, and the three callers
  (`h3-dit`'s sampler, `h3-dit-web`, `h3-ref2v`'s sampler) all use it.

### Added

- **R2V's generator holds the model to its own specification** (issue #212).
  Every number in `MiniMaxAI/MiniMax-H3`'s card is a default or a limit in
  `generate-r2v.ts` now, because every one of them was violated by hand on the
  first real request and each violation read as a property of the model until
  the card was read.

  It defaults to **5 seconds at a short edge of 768** — the official `ref2va`
  request verbatim — derives the canvas from `--aspect`, and refuses, *before*
  uploading 26 GB: a duration outside 4–15 s, a reference clip outside 2–15 s,
  more than 15 s or 3 clips of reference video, more than 9 images or 12 files,
  and a prompt missing any of H3-Context-IR's six rewrite sections.
  `--out-of-spec` proceeds and names the rule.

  The prompt check is the one with a measurement behind it: everything else
  held, replacing a six-word prompt with the six sections moved the seam figure
  from 1.51 to 1.14 — the best of any run — and put the reference's own printed
  shirt on the subject. H3-Context-IR is hosted and not in the open release, so
  this cannot build the rewrite; it can refuse to pretend a sentence is one.

  It also says what it cannot do. Five seconds at 768 on a 9:16 canvas is
  **40,927 packed rows** and the longest this port has been timed at is 4,768,
  at 12.2 s a step — printed before the wait rather than after it.

- **R2V runs end to end on the GPU** (issue #212). `generate-r2v.ts`: a
  reference and a prompt in, frames out, over four models one at a time —
  encoder 0.47 s, conditioner 2.0 s, `transformer_ref` 15 steps at ~1.4 s,
  decoder 0.65 s, with `reclaim()` at each boundary.

  Held to the model at every joint, and the last one needed a new golden:
  `t2va` cannot exercise the reference rows, the third noise level, or a vision
  block tagged video among the text rows, because all three are downstream of
  `num_condition_video_rows` being nonzero. One forward of `transformer_ref` on
  a real `ref2va` sequence is **0.68% of peak** on the video velocity and 1.61%
  on the audio, at four blocks and int8.

  Two things the conversion had to learn. `convert_dit.py --workflow ref2va`
  evaluates modulation tables for **four** noise levels instead of two — a
  `t2va` conversion runs fourteen steps and then hands back a bind group whose
  offset is past the end of its own table. And `maxLevels` comes from the
  manifest now rather than being the constant 2 it was.

  The anchors are checked rather than assumed: after fifteen steps they differ
  from what went in by exactly zero, and the run refuses to write frames if not.

  Two apparent faults were measured until they were not. An "inverted tone" was
  the *prompt* — "warm light" in it, and the correlation with the reference is
  -0.06, so not a negative of anything. A "colour drift across frames" is not
  `ref2va`'s: the per-frame chroma of an R2V run and a `t2va` run are the same
  curve, with troughs exactly at the frames that sit at position 0 of a latent
  frame. What remains is the DiT's int8, which a reference encoded and decoded
  straight back does not have and only DiT-produced latents do.

- **The visual VAE encoder on the GPU** (issue #214). `examples/h3-encoder`'s
  CPU version is a reference and stays one — it is a single-threaded loop over
  `ops/conv`'s scalar `conv3d` and takes **120.5 s on an 8x32x32 clip**, which
  is not something R2V can do once per dropped reference.

  Held to the same golden the CPU one is held to — `EncoderFCN3D` and
  `quant_conv`, the **model's own output**, never the other port:

  | | 8x32x32 | worst against the model |
  | --- | --- | --- |
  | `encoder.ts` (CPU) | 120.5 s | 2.432e-5 |
  | `encoder-gpu.ts` | **0.20 s** | **9.537e-6** |

  At the geometry R2V actually uses, RTX 5090 / Dawn / f32 weights, 236
  dispatches: **0.47 s** for one 256x256 still, 1.47 s for five frames, 5.30 s
  for twenty-one.

  **No new kernel** — `conv3d`, `pad`, `group_norm`, `activation`,
  `elementwise` and `permute` cover it, which was checked against the four the
  CPU version calls before any of it was written.

- **`permute`, `activation` and `elementwise` accept a two-dimensional
  dispatch** (issue #214). One row of workgroups runs out at 65,535, which at
  256 threads is 16.7 M elements — the encoder's first level passes it on a
  256x256 reference at five frames, and the guard that caught it was an
  exception rather than a wrong picture.

  The fold reads `num_workgroups.x` instead of taking a new uniform, so **every
  existing one-dimensional caller keeps working unchanged**: at `[n]` the y
  extent is 1 and `gid.y` is 0. `ops/pad` solves the same problem with a
  `stride_y` uniform and needed its callers to know.

- **`ResidentDevice.reclaim()`** (issue #213). `destroy()` schedules the freeing
  of a buffer's memory; it does not do it. Dawn releases on its next tick and
  **ticks on GPU work, not on a timer** — so a stage that destroys 25 GB and
  immediately allocates 20 GB gets an *invalid* buffer back, which does not
  throw. `examples/h3-dit/src/generate.ts` ran its two phases as separate
  processes because of this, having measured process exit as the only reliable
  release; `examples/h3-ref2v-web` needs three stages in one browser tab, which
  has no process to exit.

  Measured on an RTX 5090 (32 GB, Dawn/Vulkan), 25.78 GB destroyed and 20.66 GB
  asked for: nothing works (3 of 3 fail), `setTimeout` up to 2 s works at no
  delay, one submit-and-readback is *marginal* (1 of 3 failed), two or more
  worked 8 times out of 8. `reclaim()` submits four.

  **Judged by a readback, not by an error flag.** An out-of-memory
  `createBuffer` reports asynchronously, and two earlier versions of that
  measurement read their own staleness instead of the card — one called every
  stage a success, the other reported a 2.08 GB ceiling that was a leftover
  rejection from the round before. `harness/verify-reclaim.ts` runs the four
  stages and has a `--without` half that must fail; it does, at the second
  stage.

  `generate.ts --phase both` calls it now, and the entry says what that is
  worth: at 256x256 over 22 frames the two models are 23.1 GB and fit at once,
  so **removing the call changes nothing there**. It is load-bearing at the
  geometry where they do not fit.

- **R2V's conditioner converts, runs on the GPU, and reproduces the model**
  (issue #212). **25.78 GB** of int8 in 86 s — 0.61 GB of vision tower, 50 text
  layers, no `lm_head` and no layers 50..63. 76 tokens in 2.0 s over 8,182
  dispatches.

  Held to `hidden_states[50]` of the released Qwen3-VL-32B on a real
  presentation, **as a median over rows** rather than as one worst element:
  **1.12%** on the text rows and **3.00%** on the visual ones against the same
  weights round-tripped through this converter's own int8, 2.02% and 4.27%
  against them in bf16.

  Three real bugs, and the third is why the first two were hard to see:

  - **The deepstack add aliased one buffer as read and read-write inside a
    compute pass.** WebGPU refuses that, so the command buffer was invalid and
    the output was whatever the pool held — reported as "100.46% of peak",
    which is not a number.
  - **The fused `qkv` was un-interleaved with a copy per token per block** —
    20,736 for one 256x256 reference. The converter splits it into three now,
    as `examples/h3-video` splits `to_qkv`.
  - **The stack ran one layer too many.** `hidden_states[50]` is the *input* to
    layer 50: `transformers` records hidden states from a forward hook on the
    decoder layer, so state `k` is layer `k - 1`'s output. The conversion kept
    layer 50 and the forward evaluated it — 0.49 GB and a whole layer of
    arithmetic past the answer. `verify-conditioner.ts` refuses to compare a
    conversion whose layer count disagrees with the golden's now.

  **A last number cannot see any of that**, because this stack's last rows are
  massive activations: from layer 43 a few visual tokens grow by a factor of a
  hundred, so worst-over-peak reads ~96% whether the port is right or wrong. It
  moved by 0.01 points when the extra layer was removed; the median row moved
  from 24.9% to 1.12%. `verify-conditioner.ts` reports rows, split by kind.

  **Which tokens become massive is a near-tie, and int8 rounding flips it.**
  `gen_real_conditioner_golden.py --quantised` runs the released weights through
  this converter's own quantisation inside `transformers`, and the reference
  then picks the same row this port picks. Without that second reference the
  flip reads as an 88%-of-peak port bug; with it, it is the model's own
  sensitivity, recorded and not fixed.

  **A vision token does not sit at `t = h = w = index`.** A vision block gets a
  2-D grid and the clock advances by `max(h, w) / merge`, the block's longer
  side, not by its token count. `qwen3vlPositionGrid` builds it and
  `verify-conditioner.ts` refuses to run if it disagrees with `get_rope_index`'s
  own output.

- **R2V's image processor and its browser page** (issue #212).
  `examples/h3-ref2v/src/processor.ts` reproduces
  `Qwen2VLImageProcessor` **exactly** — worst difference **0**, once the two f32
  roundings upstream does were matched rather than collapsed into one.
  `smartResize`'s banker's rounding, its two different clamp rules (`floor`
  above the pixel ceiling, `ceil` below the floor), the temporal repeat of a
  still image, and the merge-block patchify. Ten mutations, all caught.

  **The resize itself is not ported**, and that is stated rather than hidden:
  upstream resamples with PIL's bicubic and a browser has `drawImage`. What that
  costs is unmeasured.

  **The tokenizer was already in this repository.** `llm/tokenizer-bpe.ts` is
  Qwen's byte-level BPE with its vocabulary committed, and it reproduces **all
  fourteen** text segments H3's own tokenizer produced plus all four vision
  token ids — measured in `examples/h3-ref2v/src/tokenizer.test.ts`, not
  assumed.

  `examples/h3-ref2v-web` takes dropped images and video, patchifies them,
  assembles and tokenises the presentation, and builds the packed sequence —
  every step held to the model. **It does not generate**: the conditioner and
  the vision tower have CPU references and no GPU path, and no converter has
  written a `conditioner.q8.bin`. The page says which of the three models is
  missing rather than failing vaguely.

- **Qwen3-VL's vision tower** (issue #212). Held to `transformers`' own
  `Qwen3VLVisionModel` at **1.192e-7** on the tower's output and **7.451e-9** on
  the pooled vision tokens and every deepstack feature. Committed fixture,
  random weights, 399 KB.

  **`ops/conv`'s `conv3d` is the patch embedding** — a `Conv3d` whose kernel
  equals its stride and its input — so #201's op has a second caller. Nothing
  else new either: `layernorm` (with bias), `attention`, `matmul`, `activation`,
  `elementwise`, and `ropeAxes` for the two-axis rotation.

  Tokens are in **merge-block order**, the position embedding is **bilinearly
  interpolated** from a learned 48x48 table with #211's `torch.linspace`
  deciding the taps, the blocks' MLP is `gelu_pytorch_tanh` while the mergers
  use the **exact** GELU, and the final merger normalises *before* the shuffle
  while the deepstack mergers normalise *after*.

  **The sweep caught six of ten, and the tolerance was why.** The bounds were
  `2e-5` against an achieved `1.192e-7` — 170x too loose — and both GELU swaps
  walked through. They are `3e-7` and `3e-8` now, measured values times a small
  factor, with the mutation's own effect recorded beside them: swapping the
  blocks' MLP moves the output to 6.109e-7, which is why the bound is not a
  round 1e-6.

- **Qwen3-VL's text decoder** (issue #212). `examples/h3-ref2v`'s conditioner
  stack, held to `transformers`' own `Qwen3VLTextModel` at **3.576e-7** across
  every hidden state, on a **committed** fixture — random weights at a tiny
  geometry, 122 KB, no model licence.

  **No new kernel.** `ops/gqa` takes the 64-query-over-8-key/value grouping,
  `ops/rmsnorm` takes the layer norms and the per-head QK norms, and
  `ops/rope`'s `ropeAxes` takes M-RoPE — through **64 axes of two channels**,
  where `theta ** 0` is exactly 1 so the frequency can be folded into the
  position (which is what #206's fractional positions were for), plus a channel
  permutation applied to Q, K **and** the per-head norm weights.

  **`mrope_section` does not name three contiguous blocks.**
  `apply_interleaved_mrope` overwrites two *strided* slices of an all-time
  array, so channel `c` is on axis `c % 3` while `c < 3 * section[1]` and on
  time after. The chunked reading is what the field name suggests, produces a
  working model, and is wrong — as is a per-axis frequency sweep, since every
  channel keeps the frequency its **global** index gives it.

  **The last hidden state carries the final norm.** `output_hidden_states=True`
  returns `n + 1` entries whose last is `last_hidden_state`, so a port that
  appends its raw last layer output matches every earlier entry and misses that
  one by 0.53. It does not change what H3 reads — `hidden_states[50]` of 64 is
  an ordinary layer input — which is exactly why it could have gone unnoticed.

  Ten mutations, all caught.

- **R2V's transformer converts with no new code** (issue #212).
  `transformer_ref/` is a second 66.28 GB partition and *is* different weights —
  measured at **2.0–2.2%** mean relative difference across the stack, a
  fine-tune of the same base — but its config is **identical to
  `transformer/`'s, field for field**. So `examples/h3-dit`'s `convert_dit.py`
  and `model-gpu.ts` serve it unchanged: 20.08 GB resident in 96 s, held to the
  model's own output at four blocks at **0.96%** of peak on video against the
  `t2va` partition's 0.84%.

  The fifty-block check is not run yet — it needs 20.66 GB and the card was
  holding 27.1.

- **R2V's presentation** (issue #212). How a `ref2va` request is announced to
  the conditioner: `"<Picture i>: "`, `"<Audio j>: "`, `"<Video k>: "`, numbered
  **per modality**, each followed by a vision block of pad tokens — which is
  what produces the text tags the layout takes, because a vision block's rows
  are tagged **video** while sitting among the text rows. A video that carries
  sound is announced `<Audio>` **before** `<Video>`, and gets one timestamped
  block per merged frame group.

  **The timestamp is rendered with Python's round-half-to-even, and the tie is
  not where it looks.** The mean of a 2 fps pair is exactly `0.25`, which
  renders `"0.2"` where `toFixed(1)` gives `"0.3"`. But detecting the tie by
  multiplying by ten is also wrong: `0.15` is really `0.1499999999999999944…`
  and renders `"0.1"`, while `0.15 * 10` rounds to exactly `1.5` and reads as a
  tie. The tie is detected on `toFixed(20)` — the exact decimal expansion —
  and the bug was found by a fixture case at 30 fps, not by reasoning.

  The fixture carries the token ids **and a map from each text segment to its
  ids**, so the assembly is held to upstream without a BPE implementation in
  the way.

- **R2V's packed layout** (issue #212). `examples/h3-ref2v`: MiniMax-H3's
  `ref2va` sequence, `[text | reference blocks | target audio | target video]`,
  held to `build_ref2va_packed_sequence` exactly on a committed fixture.

  What makes it more than an ordering is that **the references advance a shared
  rotary clock** — where the generated video sits depends on how many
  references came before it and how long each was. An image takes **one** slot,
  not a latent frame's `5/3`; a video's soundtrack is packed **before** its
  video rows, sharing their origin; a video advances the clock by
  `max(audioLatents, videoSpan)`; and that span is summed **sequentially**,
  which is deliberately not how the `t2va` keyframe anchor sums the same
  series — the two differ in the last ulp from 16 latent frames onwards and
  upstream keeps both.

  **The first mutation sweep caught seven of ten, and all three survivors were
  the fixture's fault.** No case had a soundtrack longer than its own video, so
  the `max` never chose the audio; every reference shared a canvas with the
  target, so pinning a soundtrack to the wrong width grid was invisible; and
  the tag assignment order genuinely cannot matter, since the three index sets
  are disjoint. Three cases were added and the unobservable mutation was
  replaced by one that is — which found a real gap: the port was filling in
  `TEXT_TAG` for every text row, and `text_token_tags` is an argument precisely
  because **a reference's vision block sits among the text rows and is tagged
  video**.

- **Flat dispatches are split to fit the device's grid** (issue #211). One
  thread per element and `ceil(n / 256)` workgroups runs out of grid at
  **16,776,960 elements** — 65,535 workgroups, which Dawn Node and Chrome both
  report and neither raises when asked. A 14,336-wide feed-forward reaches it
  at **1,170 rows**.

  A 22-frame 256x256 clip is 538 packed rows and fits. **576x320 is 1,350 and
  does not**, which is the second size anybody picks, and it arrived as
  `Dispatch workgroup count X (72240) exceeds max compute workgroups per
  dimension (65535)` after 24.49 GB had been uploaded.

  Split on **row** boundaries, not element ones: `ops/elementwise`'s rows entry
  recovers its column with `idx % D`, so a chunk starting mid-row would read
  the wrong scalar for every element of it — a well-formed tensor, quietly
  wrong. Chunks are rounded to keep every slice 256-byte aligned.

  Held to the model's own output through a lowered ceiling: **223 dispatches
  unchunked, 231 chunked, the same worst element to four digits.**

  `swapLeading`'s transpose is the one that cannot be split — it writes
  strided, so a slice of the input has no slice of the output to land in. It
  runs out at **2,340 tokens**, past every size the page offers, and refuses
  with that number rather than letting `batch is not valid` arrive from inside
  a command buffer.

  This resolves the ambiguity #211 opened with: a **headed** Chrome allows
  `workgroup_size(512)` and a headless one does not, so `examples/zimage-web`'s
  record of running in a browser and this page's refusal are both true of their
  own adapters.

- **A lost WebGPU device says why now** (issue #211). Once a device is lost,
  **every later call reports the same thing**: `popErrorScope` rejects with
  `OperationError: Instance dropped in popErrorScope`, once per buffer, with a
  stack pointing at whatever allocation came next rather than at the fault.
  `device.lost` — where the backend says what it actually hit — was never read
  by `createBrowserResidentDevice`.

  Every error scope in the browser device now handles its **rejection** path as
  well as its resolution, and reports the loss reason plus how much had been
  allocated across how many buffers. `examples/web-common/src/device-lost.ts`
  is the wording, tested without a GPU: an unknown cause is replaced, an
  unresolved loss says it has not said why yet, and an error that *is* the
  fault — a refused pipeline — survives intact.

  **The page reads the adapter's limits before it uploads anything.** The first
  run spent 21 s on 23 GB and then printed `Instance dropped` five times; the
  cause was one number, `maxComputeWorkgroupSizeX = 256` against
  `ops/matmul`'s 512, readable in the first second. It now refuses in that
  first second and prints all five limits.

- **A browser page that generates video** (issue #210).
  `examples/h3-dit-web`: a prompt, a size, a step count and a seed in; the DiT
  samples and the VAE decodes, both on WebGPU, and the frames play on a canvas.

  **It uploads and it does not yet run.** 23.10 GB in 20.9 s, prompts and step
  counts populated, and then the first generation fails with
  `Entry-point uses workgroup_size(512, 1, 1) that exceeds the maximum allowed
  (256, 256, 64)`. The adapter a **headless** Chrome 151 hands out reports
  `maxComputeWorkgroupSizeX = 256`, and `ops/matmul`'s two entry points declare
  512. Issue #211, with the four limits measured.

  **Whether that is headless Chrome or every Chrome is unresolved, and it
  matters**: `examples/zimage-web` dispatches the same kernel and is recorded as
  running end to end in a browser, which cannot both be true of one device. No
  X display was reachable from where this was measured, so it was not settled
  rather than guessed. `examples/h3-video-web`'s "the in-browser decode is
  unmeasured" is consistent with nobody having run it either.

  `?serve=<base>` reads the weights over HTTP instead of from a picked folder —
  not a convenience: `showDirectoryPicker` needs a user gesture and a native
  dialog, so a headless browser cannot otherwise reach the page's work, which is
  exactly why these pages keep ending "unmeasured".

  The page reports what it **uploaded**, not what the device took. WebGPU
  exposes no way to ask, and `nvidia-smi` never moved while the page had
  uploaded 23.10 GB.

- **The 50-layer DiT runs on the GPU** (issue #210). `examples/h3-dit`'s
  `model-gpu.ts` plus `tools/convert_dit.py`: **20.08 GB** of int8 resident,
  one step over 42 rows in 2,078 ms (RTX 5090, driver 610.57.04, Dawn
  `webgpu@0.4.0`), of which 6 ms is queue time and **1,611 ms is host-side
  recording**. No new kernel.

  **20.08 GB, not 33.12.** `adaln_proj` is 13.01 G of the checkpoint's 33.12 G
  parameters — **39.3%** — and it projects `temb`, a *two-row* tensor whose
  value depends only on the timestep. The converter evaluates the modulation
  tables instead of shipping the weights, which costs 0.58 GB of tables for 16
  steps and means a conversion runs only the step counts it was given.

  **What int8 costs, at one block, as a percentage of the golden's peak**:
  0.55% for bf16 activations against f32 with *no quantisation at all*, **0.88%
  for this port**, and 7.59% for int8 weights inside torch's own bf16 pipeline.
  The port sits just above the floor its bf16 golden can resolve. At fifty
  blocks the port is 13.80% and torch's own int8 round-trip is 47.49% — most of
  the gap is precision rather than the port, but the split is **not measured**,
  because an f32 fifty-block reference is 132 GB.

  The rope call was written from memory rather than from
  `ops/rope/wgsl/axes.wgsl` — bindings swapped, a four-field uniform against a
  five-field struct — and since every binding is `array<f32>`, there was no
  validation error, just NaN fifty blocks later. **The comparison reported
  "worst 0.000e+0" on wholly-NaN output**, because `Math.abs(NaN - x) > worst`
  is false; it counts non-finite values now and refuses before reporting.

- **MiniMax-H3's packed sequence layout** (issue #210). The transformer builds
  none of it: row order, modality tags, per-row noise levels and the `(t, h, w)`
  rotary grid all arrive as arguments, so every one is a free choice and **not
  one of them changes a shape**. Ported from
  `diffusers.modular_pipelines.minimax_h3`, fixture committed.

  Six conventions, none guessable: video rows **frame-major then row-major**; a
  spatial grid that is **aspect-normalised** and scaled by 32; built with
  **`np.linspace(endpoint=False)`**, not `torch.linspace`; latent frames spaced
  **`5/3 * (1, 4, 4, 4, 4)`** because the VAE's first latent covers one pixel
  frame and the rest four; **the media clock starting after the text**, so
  prompt length moves the video; audio rows **channel-major**, with no height,
  pinned to the two extremes of the width grid.

  The fixture carries a **square** canvas as well as a wide and a tall one. On a
  square canvas the aspect normalisation is the identity, so squares alone
  would pass with it deleted.

  `resolveCanvasSize` needs Python's **banker's rounding**, and that is not
  pedantry: the default 720p canvas asks for `round(720 / 32) = round(22.5)`,
  which is 22 in Python and 23 in JavaScript, so half-up generates at **736**
  pixels where the model wants 704 — and nothing downstream would object.

  Eleven mutations, all caught.

- **MiniMax-H3's sampling schedule** (issue #210). Rectified flow with an
  exponential shift, ported from `MiniMaxH3Scheduler`. Four conventions decided
  by upstream rather than by this port, each of which yields a *video* rather
  than an error when it is wrong: **`t = 1 - sigma` and `t = 1` is clean** (the
  reverse of the DDPM convention every other schedule here uses); the terminal
  sigma gets **no model evaluation**, so `n` grid points drive `n - 1` forwards;
  `step` recovers its sigma from the **timestep**, not the grid, because
  `1 - (1 - sigma)` is not an exact f32 round trip below 0.5; and `eta` is **0**
  despite the reference class being named "euler ancestral".

  The sigma grid matches torch **exactly, every element**, which took three
  details that were measured rather than reasoned about: `torch.linspace`
  rounds its step to f32 *first*, counts the second half **down from the end**,
  and computes each element as one **fused** multiply-add. Naive
  `1 - i / (n - 1)` disagrees at 4 of 50 points and rounding the product first
  disagrees at a different 4 — one f32 ulp each time, which no sampler would
  ever look wrong for, and which changes which timesteps the transformer is
  conditioned on.

  Both shipped shifts are covered — 12.0 for video, 3.0 for audio — and the
  fixture is committed, because a schedule is arithmetic and carries no
  weights. `unique_consecutive` is exercised at 1,000,000 grid points, where a
  shift of 12 collapses 42,208 of them; **no realistic step count reaches that
  branch**, which is how a port drops it and nobody notices.

- **The whole DiT forward, with a golden that lives in this repository**
  (issue #210). `examples/h3-dit` held **one block** to diffusers'
  implementation and recorded that "the rest is a loop". It is not: around the
  fifty blocks sit one packed sequence built from three separately-projected
  modalities, a two-block text refiner with **no rotary**, an AdaLN table
  addressed per row, a `norm_out` that modulates per row with **shift first**,
  and two heads that run over every row before the modality's rows are
  selected. Each returns a well-formed tensor when it is wrong.

  Held to `MiniMaxH3Transformer3DModel` at **1.490e-7** on the video velocity
  and 1.192e-7 on the audio, against an f64 golden.

  **The fixture is committed and CI runs it.** It is 155 KB because the golden
  is generated at upstream's own tester geometry — hidden 24, two layers — with
  **random weights**, so it is not MiniMax's checkpoint and carries none of its
  licence. Nothing structural depends on the weights being trained, and ten
  mutations confirm it: the refiner skipped (1.6e-2), the AdaLN table addressed
  without its modality (2.7e-2), `norm_out`'s shift and scale swapped
  (5.99e-1), the timestep embedding not flipped to cos-first (7.4e-2), and six
  more.

  **Two of the ten are only caught because both outputs are compared.**
  Modulating `norm_out` from table row 0 leaves the video output
  bit-identical — every video row is at timestep 0 — and moves the audio by
  2.6e-2; dropping the video head's bias does the reverse. A test that checked
  one head would have passed either.
  Its test asserts the array type, because a test that only checked the values
  passed the whole time.

- **Changing folder to the wrong one left the page with no folder at all**
  (issue #219). Reported from a browser: Z-Image's picker was pointed at Anima's
  folder, and the page died with `the folder "anima-3.8B" has no
  "model.safetensors.index.json"`. Reloading landed in the same place.

  Three faults, and each alone would have been survivable.

  A receipt was read without asking what the caller needs. A folder filled for
  another model carries its own valid receipt, so `readReceipt` answered
  "filled", the fill was skipped, and the page got a folder holding none of its
  files. It takes the plan now, and a receipt that does not name every file the
  caller wants means unfilled — which fills the folder rather than failing after
  a reload, the cheap direction to be wrong in.

  The folder was remembered before it was known to work, so a bad pick became
  the page's permanent answer and no reload could escape it. `bindFolder` reads
  every file out of the new folder first and stores the handle last.

  And changing folder forgot the folder that worked. The old rule was "every
  bind failure forgets the folder", which is right when there is nothing to fall
  back on and wrong when there is. `gate.test.ts` asserted the two paths were
  the same thing; it now asserts they differ, and says why, because that rule
  was the bug.

  Each page's `main()` also catches now, so a start-up failure reaches the
  status line instead of being an unhandled rejection under a page that looks
  like it is still loading.

- **The type a buffer is uploaded with is now checked against the type the
  kernel declared** (issue #221, out of #217). `ropeAxes`' `positions` binding
  changed from `array<i32>` to `array<f32>`; one caller was updated, two were
  not, and nothing anywhere said so for eighteen commits. WebGPU has no opinion
  about this — `queue.writeBuffer` copies bytes, and a small integer's bit
  pattern read as an `f32` is a denormal near zero. Every rotation angle became
  zero and the page drew one flat colour, with no error.

  **The type was written down the whole time**, one line above the code that
  read it. Nothing on the host read that line. `harness/binding-types.ts` reads
  it now, and `harness/wgsl.ts#createRunner`, `harness/resident.ts` and
  `examples/web-common/src/browser-resident.ts` refuse the mismatch — so every
  op test and every resident model, in Node and in the browser, is covered by
  running at all.

  Judged where a buffer, a pipeline and a binding number are in the same room,
  which is the bind group and nowhere else. The first version also judged inside
  `upload`, against whatever binding the buffer had last, and on Anima's real
  forward that named the wrong kernel: buffers come from a pool, and the same
  one had been `RMSNorm`'s input a dispatch earlier. It caught the bug and lied
  about where — and the same staleness would eventually have rejected a pooled
  buffer legitimately used as `array<i32>` in one op and `array<f32>` in
  another. A check that fails on correct code is worse than no check.

  Two things are deliberately unchecked, both found by turning tests red rather
  than by reasoning. `array<atomic<T>>` says how a slot is *updated*, not what
  it holds: `ops/scatter` is an f32 scatter-add through
  `atomicCompareExchangeWeak`, because WGSL has no f32 atomic, and the first
  version turned its eight tests red. And a `Uint8Array` is the host saying
  "these are bytes" — `ops/matvec/wgsl/q8.wgsl` declares `array<u32>` and is fed
  four packed int8 weights per word.

- **The visual VAE round-trips** (issue #200): `examples/h3-encoder` is the whole
  encoder — six levels of two `ResnetBlock3D`s, a strided convolution between
  the first four, and `quant_conv` — held to `EncoderFCN3D`'s own output at
  **2.432e-5** on moments peaking at 8.854. With `examples/h3-video`'s decoder,
  a video goes in and frames come out, both halves checked against the model.

  **`Downsample3D` pads asymmetrically before it strides**: nothing before, one
  column and one row after, then a `k=3, stride=2` convolution with zero spatial
  padding. A symmetric pad gives the same output size and a different alignment
  — a video that comes back shifted, level by level. It moves the moments by
  3.8.

  CPU reference only: there is no GPU path for the encoder and no page for it.
  The whole-encoder comparison is a **script**, not a test — the reference takes
  123 s for 8 frames of 32x32, past `scripts/test.mjs`'s per-file minute, which
  is the same reason `examples/anima`'s forward and the video decoder's are
  scripts. What stays in vitest is the channel plan and the compression factors.

- **`conv3d` and `pad` have a caller** (issue #200): one `ResnetBlock3D` of
  MiniMax-H3's visual VAE **encoder**, checked against the model's own module at
  **3.576e-7**. The VAE's decoder is a ViT and needs neither op; the encoder is a
  3D convolutional stack, and an op nothing calls is a liability.

  Three conventions the block settles, each measured rather than recalled:
  **space reflects and time does not** — a causal convolution prepends
  `2 * padding` zero frames and appends none, which is what `ops/pad` taking
  `before` and `after` separately is for; **two frames, not one**, or the output
  is a frame shorter; and **group norm is per frame**, not over the clip
  (`use_t_isolated_gn`), which is worth 2.4 in the output when got wrong.

- **Z-Image runs from a folder too, and is published** (issue #194). The same
  gate, the same receipt, the same offline-after-the-first-download. 14.4 GB in
  total — and **8.04 GB of it comes straight from
  [`Tongyi-MAI/Z-Image`](https://huggingface.co/Tongyi-MAI/Z-Image)**, because
  the text encoder is Qwen3-4B unmodified and copying it into a second
  repository would redistribute what its own publisher already serves. Only the
  6.4 GB this repository converted is at
  [`m96-chan/Z-Image-q8-web-xpu-ops`](https://huggingface.co/m96-chan/Z-Image-q8-web-xpu-ops),
  Apache-2.0 like its source.

  `byte-source.ts`, `provision.ts`, `bound-folder.ts` and the gate moved to
  `examples/web-common/src` rather than being copied: three copies of "is this
  folder complete?" is three places for the check to drift, and that check
  decides whether a half-downloaded folder produces an error or a wrong picture.

- **Every converter records where the weights came from** (issue #190). Tracing
  Anima's licence took an hour because its manifest named a repository that
  declared none and deferred to an upstream it did not name. The repository,
  the licence and — for Anima — the whole chain now live in the manifest the
  converter writes.

- **The Anima demo runs from a folder you pick, and is published** (issues #180,
  #192). Point it at an empty folder; the weights are downloaded into it once
  from
  [`m96-chan/Anima-3.8B-q8-web-xpu-ops`](https://huggingface.co/m96-chan/Anima-3.8B-q8-web-xpu-ops)
  and read from there every time after. No clone, no `npm install`, no dev
  server, and no re-fetching 5.0 GB on a reload.

  **A folder is required rather than offered.** The alternative was the Cache
  API, which puts 5.0 GB in the origin's storage quota where an eviction the
  page cannot see turns the next run into another download; keeping both paths
  would mean keeping one nobody chooses. `preloadAll`, which existed to fill
  that cache, is gone.

  Filling a folder can stop part-way, and what that leaves behind opens, has a
  plausible size, and produces a **wrong image rather than an error**. So a
  receipt naming every file and its exact length is written last and checked
  against the folder before it is used. What the receipt cannot catch — a file
  of the right length holding the wrong bytes — is recorded rather than papered
  over: only a digest would, and that costs a full read of 5 GB every start.

  The model's licence is **not** this repository's. Anima is non-commercial;
  see the README.

- `scripts/pages.mjs` builds the published site, and refuses to emit a page
  whose cache-buster did not fire. The stamp is the bundle's content hash, so a
  rebuild that changes nothing does not invalidate anyone's cache.

- **Z-Image runs end to end, in a browser** (issue #166). A prompt goes in and a
  1024x1024 image comes out, on WebGPU, composed from this repository's ops:
  byte-level BPE, a Qwen3-4B text encoder, the 30-layer DiT, a rectified-flow
  sampler and the VAE decoder. `examples/zimage-web` is the page;
  `examples/zimage/src/generate.ts` is the same pipeline from a command line.

  Each stage was fixed against Z-Image's own implementation before the next was
  wired to it — one block at the shipped width (8.72e-8), the whole DiT forward
  (4.386e-6 at 70 tokens, 4.889e-6 at 1,039, 5.560e-6 at 4,111), the text
  encoder (7.993e-7), the sampler's schedule (exact), the VAE decoder (1.629e-5
  at 1024). Measured conditions are in the example's README, with the image.

- **`conv3d`** (issue #200). MiniMax-H3's visual VAE compresses time as well as
  space — `temporal_downsample_factors [1,2,2,1,1,1]`, 4x — so every convolution
  in it spans frames and none of them decomposes into 2D plus a loop.
  `ops/conv`'s own doc had listed 3D as *deliberately absent*; this is the model
  that makes it necessary, and every convention 1D settled carries over
  unchanged.

  `[N, Cin, D, H, W]`, weight `[Cout, Cin/groups, KD, KH, KW]`, with
  `stride` / `padding` / `dilation` taking `number | [D, H, W]`. **Still no
  `padding_mode`**, and 3D is where that starts to cost something: H3 pads
  `reflect` on the spatial axes and **causally** in time — `2 * padding` frames
  before the data and none after — which is not a value any symmetric padding
  argument can take. That is a pad op, and it is **`pad`**, below.
  argument can take. That is a pad op, and it is not written yet.

  3D is also the first place the *axis order* is checkable. The tests use an
  input whose elements name their own coordinate (`100d + 10h + w`) against a
  corner-tap kernel, so an output element reads back the window it started on:
  a swapped axis, a flipped kernel or an off-by-one pad shows up in the digits.
  Goldens measured against torch 2.10.0+cu130, integers so they are exact in
  f32.

- **A browser page for it** (issue #200). `examples/h3-audio-web`, on the same
  folder-bound, server-free footing as the other demos: 260 MB into a folder
  you pick, read from it every time after.

  **There is no prompt box, and the page says why.** The transformer that would
  write the latent is 20B and the encoder that would read a prompt is
  Qwen3-VL-32B — 27 GB between them at the smallest published quantisation,
  against 260 MB for the decoder. A latent sampled from the prior is what a VAE
  decoder is built to receive, so that is what it gets.

- **MiniMax-H3's audio decoder runs, and it needed no new kernel** (issue #200).
  H3 generates video with **native stereo audio**, and the audio half is a
  BigVGAN vocoder over a 32-channel latent at 40 Hz — a DAC-lineage codec, which
  is the same shape as the one `conv_transpose1d` (#87), `group_norm` (#91),
  `snake_beta` (#90) and `istft`'s `"same"` (#93) were added for. Those landed
  for VoxShot. Every one of them is what this model's decoder wants.

  `examples/h3-audio` has the decoder twice: a reference that calls this
  library's own references, and a GPU version that is the same file with
  dispatches. Both are held to a waveform produced by **the model's own Python**
  on a fixed latent, not to each other — agreeing with a port only says the two
  share a mistake.

  Measured, RTX 5090 / driver 610.57.04 / Dawn `webgpu@0.4.0` / f32, on a
  6,400-sample golden: the reference is **1.788e-6** worst element against
  torch's output and takes 21 s; the GPU version is **5.007e-6** and takes
  **432 ms**.

  Two things that would otherwise have needed a new kernel are folded instead.
  The `ratio *` after the anti-aliasing upsample rides on the filter's twelve
  taps, since a convolution is linear in its weight. And the slice after it is
  the transposed convolution's `padding` — the two trims are equal for this
  filter, and a symmetric crop is what `padding` means there. The port throws
  if a future filter makes them differ rather than quietly dropping a sample,
  which is a phase error no unit test hears.

- **`pad`** (issue #200). `ops/conv`'s doc has always said `padding_mode` is a
  pad op, not a convolution argument. MiniMax-H3 is the model that makes that
  separation load-bearing: its visual VAE pads `reflect` on the spatial axes
  and, in time, **causally** — `2 * padding` frames before the data and none
  after, so the frame at `t` cannot see `t + 1`. No symmetric `padding`
  argument can say that, whatever its mode.

  `constant` (with a value), `reflect` and `replicate`, with `before` and
  `after` separate. **`reflect` does not repeat the edge element and
  `replicate` does** — measured against torch rather than described, because
  swapping the two gives a tensor of the right shape whose entire interior is
  correct.

  It pads **one** axis, viewed as `[outer, L, inner]`; three axes means three
  calls. That is measured to give what torch's single multi-axis call gives,
  element for element — not obvious, since a reflection reads neighbours that
  are themselves reflections. One kernel then serves the audio VAE's 1D
  `replicate` and the visual VAE's 3D `reflect` without knowing either rank.

- **A transformer block of MiniMax-H3's DiT** (issue #200) — the *generator*
  half of the model, against the visual VAE this repository already decodes
  with. Fifty identical blocks over 5,376 channels; a port is right or wrong at
  the block and the rest is a loop.

  The checkpoint ships no code for the DiT — it names a diffusers class — so the
  golden **imports** `MiniMaxH3TransformerBlock` from diffusers' main branch,
  with only its relative import lines rewritten to absolute.

  Worst element **5.859e-3** against an f32 golden and **1.953e-3** against an
  f64 one, on a block output of order 350. A third of that gap is torch's own
  f32 rounding over 5,376- and 14,336-term dot products; the rest is this port
  storing intermediates in `Float32Array`, which is what an f32 port is.

  **A permutation has to cover everything that reads the channels it moves.**
  The RoPE permutation goes into the Q and K weights, as it does everywhere
  here — but this block's QK-norm has *per-channel weights*, and permuting the
  projection without permuting them scales the wrong channels. The visual VAE
  has `qk_norm_affine: false` and no weights at all, so the same permutation is
  complete there and incomplete here. The symptom was 8% error in attention with
  the rope itself exact to 2.4e-7.

  And **SwiGLU is `hidden * silu(gate)` with `hidden` first** here, the opposite
  of the visual VAE's. Two files in one model with opposite conventions.

  `h3RopePermutation` moved into `ops/rope`: both of H3's models rotate the same
  way at different geometries, and the generated table now checks the function.
- **The video decoder's time accounting, corrected twice** (issue #200). At 8
  frames of 128x128 in int8: **40 ms in the block submits, 81 ms in the final
  submit and its readback, 136 ms of host-side recording** over 1,122
  dispatches — 264 ms in total, 131 ms once the scratch pool is warm.

  The first version timed only `flush`'s batches, so the final submit and its
  `mapAsync` landed in the unexplained remainder. The version before that timed
  each `await` inside the dispatch path and blamed `pipelineFor` for **502 ms**,
  which a tight loop then priced at **0.1 µs** — a `performance.now()` pair
  straddling an `await` charges whatever else the event loop runs to whatever is
  awaited.

  Three things measured rather than assumed: **grouping blocks into fewer
  submits is slower** (628 ms at one per submit against 725 at all thirty-six,
  identical output), **pooling the uniform buffers** takes the first decode from
  293 ms to 264 and leaves the steady state alone, and **the scratch pool** is
  worth about 130 ms on the first decode. What the remaining 136 ms is has not
  been established, and the README says so.

  `blocksPerSubmit` is a field rather than a constant so that the measurement can
  be repeated on other hardware.

- **The video decoder runs in int8, and the page uses it** (issue #200).
  `--quant q8` writes **2.43 GB** where f32 writes 9.69, and `matmulQ8` takes
  `nn.Linear`'s `[out, in]` layout untransposed with one absmax scale per output
  row — the opposite of the f32 path, whose kernel reads `[in, out]`.

  What it costs, measured against the model's own pixels **in the units a viewer
  sees**: **2 levels of 255**, RMS 0.564 levels, against f32's 1 and 0.007. In
  the model's normalised space that is 3.606e-2 — a number that says nothing on
  its own, since the denormalisation multiplies by a per-channel std of ~0.22.

  Two levels of 255 for a quarter of the size, and a faster load: 292 ms against
  636 ms for the first decode.

  Two quantiser mutations are strongly observable (the scale taken as `absmax`
  rather than `absmax / 127` costs 230 levels; reversing the byte order inside
  each u32 costs 196) and two are not: quantising per *tensor* rather than per
  output row costs only 5 levels, and widening the clamp to `-128` changes
  nothing at all — which is structural, since an absmax scale never produces
  -128. All four are written down, including the ones that barely show.

- **A browser page that decodes video** (issue #200). `examples/h3-video-web`,
  on the same folder-bound, server-free footing as the other demos: a latent in,
  frames on a canvas, with a transport to play them at 24 fps.

  **9.69 GB of f32 stays on the GPU**, which is the honest requirement and the
  page says it. Quantising to int8 would be 2.4 GB and is not done: `matmulQ8`
  exists, but what int8 costs this decoder has not been measured.

  **The in-browser decode is unmeasured.** Everything quoted is Node against the
  same `VideoDecoderGpu` class the page instantiates; the page adds the folder
  read, the browser device and the canvas, prints its own decode time, and the
  README says so rather than quoting the Node number as though it were the
  browser's.

- **MiniMax-H3's visual VAE decoder runs, end to end, on the GPU** (issue #200).
  A latent in, video frames out: 36 transformer blocks over 2048 channels, the
  embedding, the projection and the two reshapes that turn tokens back into
  pixels.

  Checked against **the model's own `decode`**: worst element **4.530e-6**, RMS
  9.051e-7, on a signal peaking at 1.64. **9.69 GB of f32 weights upload in 4.0
  s and a 2x3x4 latent decodes to 8 frames of 48x64 in 640 ms** (RTX 5090,
  driver 610.57.04, Dawn `webgpu@0.4.0`).

  **No new kernel.** `matmul`, `rmsnorm`, `layernorm`, `flash_attention`,
  `rope`'s axes entry, `activation`, `elementwise` and `permute` cover it. Three
  things that would each be a strided copy per token happen once at conversion
  instead: `to_qkv` split into three projections, `ff.w1` split into gate and
  up, and every weight stored `[in, out]` because that is what `ops/matmul`
  reads.

  The verification decodes the same latent **twice** and requires the two to
  agree exactly. Scratch buffers are pooled, so the second decode is the first
  to see a *used* one — and the zero cls token, which the model builds as
  `torch.zeros_like(...)`, is right the first time and wrong after if the clear
  is dropped. A single-decode check stayed green through exactly that mutation.

- **A transformer block of MiniMax-H3's visual VAE decoder** (issue #200). The
  decoder is 36 identical blocks over 2048 channels — **9.69 GB of the
  checkpoint's 10.42** — so a port is right or wrong at the block and the rest
  is a loop. Built in the order `examples/zimage` and `examples/anima` were.

  Held to **the model's own output**: `tools/gen_block_golden.py` imports
  `TransformerBlock` and `RotaryEmbeddingND` from the bundle the checkpoint
  ships and runs block 0 on a fixed input. Worst element **1.192e-7**, RMS
  1.773e-9.

  Three things it does that Z-Image's block does not, each of which returns a
  well-formed tensor when got wrong: **Q, K and V come from one `to_qkv` and are
  interleaved per head** (`view(B, L, -1, 3 * dim_head)`, not three separate
  blocks); **QK-norm has no weights** (`qk_norm_affine: false`, passed as ones
  rather than a second path); and the branches carry **LayerScale**, a
  per-channel parameter that multiplies each before the residual.

  No new kernel: `matmul`, `rmsnorm`, `attention`, `rope`'s axes entry,
  `activation` and `elementwise` cover it.

### Changed

- **`ropeAxes` takes fractional positions** (issue #200). Z-Image indexes tokens
  by their grid coordinate, so its positions are whole numbers and the binding
  was `i32`. MiniMax-H3's visual VAE normalises each axis to `(-1, 1)` —
  `2 * (i + 0.5) / n - 1` — and multiplies the angle by `2π`, so its are
  fractional. A rotation by a fractional angle is the same rotation; nothing in
  the arithmetic ever needed a whole number, and the alternative was a second
  kernel differing by one binding type.

  `Float32Array` is the general one and integers still fit exactly. Existing
  callers pass `Int32Array` unchanged — the reference takes either — and the one
  GPU caller (`examples/zimage/src/dit-gpu.ts`) uploads f32 now.

  `ops/rope/h3-axes.test.ts` pins the whole mapping onto H3's rope, not just the
  capability: the **frequencies already agree** (both are
  `θ^-[0, ⅛, …, ⅞]`), the pairing does not — H3 tiles `[t8,h8,w8]` twice and
  rotates halves where `ropeAxes` pairs adjacent channels, so a generated
  permutation converts one to the other **in the weights**, as `permuteForRope`
  does for Anima — and `rope_dim_ratio: 0.75` is covered by a fourth axis pinned
  at position 0, which is the identity. Worst element **2.384e-7**.

- **The browser `ResidentDevice` lives in one place** (issue #200).
  `anima-web` and `zimage-web` each had a copy, and they had already drifted:
  one carried the profiling half and the other carried a comment saying a stub
  that reported nothing would be worse than its absence — which is precisely the
  state the first one had reached and then fixed. It is
  `examples/web-common/src/browser-resident.ts` now. One fewer `requestDevice`
  is one fewer place to forget a limit, which
  `harness/device-limits.test.ts` counts.

- **Flash attention stages `k` and `v` without dividing** (issue #177). The
  staging loops turned a flat index into an address the way the arithmetic reads
  — row `base + e / Dv`, channel `e % Dv` — and **a GPU has no integer divide
  instruction**, so each is a short inline sequence the compiler emits and
  cannot fold, because `Dv` is a uniform. They cancel exactly:
  `(base + e / Dv) * Dv + e % Dv == base * Dv + e`. The bound check goes with
  them.

  **4% off the kernel and 3.9% off attention in a forward**, RTX 5090 / driver
  610.57.04 / Dawn `webgpu@0.4.0`, q8 weights and f32 activations, at Anima's
  own shapes (self L=S=3952 8 heads, cross L=3952 S=512 16 heads, D=128): the
  self shape goes 5.84 → 5.60 ms with the reference moving 0% across three runs,
  and an 832x1216 forward's `flashAttention` goes 748 → 719 ms over three
  alternating pairs. Measured in both orders, because with `linear` always first
  the *control* kernel moved 1.2% too — the figure above is from the order where
  the control moved against it.

  The addressing is a swept field rather than a rewrite, so #198 can re-decide
  it on hardware that is not this one.

- **The Anima demo re-read weights it already had on the GPU, once per forward**
  (issue #186). A 40-step generation at 832x1216 goes from **347.4 s to
  225.3 s**, and a steady-state step from **7.26 s to 4.28 s** — measured on the
  same machine in the same session, before and after, with the generated image
  identical to the byte (`sha256[0:8] 7571bd1f8a34836b` both times).

  `animaForwardResident` keeps weight buffers across forwards and both paths
  that need one check that first, so from the second forward onward the weight
  source is never consulted. The page called `preloadPrefix` anyway, at all four
  announcement sites, pulling each block's tensors off disk for a forward that
  would not read them. It was not even a warm-cache no-op: the heap holds 192
  packed tensors against 898 in the model, so every forward re-read what the
  last one evicted. **1258 ms of a 3582 ms forward, 35%, all of it waste.**

  The comment above the callback already said only the first forward does any
  work there. Nothing checked, and it was false. What made it visible was
  #182's breakdown — `batch()`'s own timers named 55% of a browser forward and
  the rest was outside every dispatch.

- **The Anima demo's cache buster had never fired** (issue #177). The server
  stamps the bundle's mtime onto the `<script src>` because `Cache-Control:
  no-store` was measured not to be enough — a tab that had loaded the page
  before the header existed kept running the bundle it already had. It replaced
  the literal `"./dist/bundle.js"`, and that page's tag says
  `"/dist/bundle.js"`, so the rewrite matched nothing and returned its input.
  Nothing failed: the request was logged, 200 was returned, and the stale tab
  went on running an old bundle while the server reported serving a new one.

  Found while confirming a browser would actually run a newly tuned kernel,
  which is exactly the measurement it would have corrupted. Both demo servers
  now match by pattern and **throw when the rewrite does not fire**, since a
  no-op string replacement is otherwise invisible.

- **Flash attention ships as two kernels, FA2 and FA3, and runs 5.4x faster**
  (issue #177). The generations are separate programs with separate schedules
  rather than one program with a switch, because they are separate algorithms;
  both are held to `ops/flash_attention/reference.ts` by the same tests, and
  `tools/generate.ts` writes both so a generator edited without regenerating
  fails a test instead of shipping the old kernel quietly.

  **Neither generation was the problem, and finding that out is what made this
  work.** FA2 and FA3 measured within a few percent of each other, and two
  shapes moving 4x different amounts of data landed on the same 240 GB/s. So
  `tools/where.ts` deletes each inner loop in turn and times what is left: full
  16.45 ms, minus the accumulate 15.23, minus the score dot 5.28, minus both
  2.76. **The score dot product is 76% of the kernel**, and FA1/FA2/FA3 differ
  only in how the tile loop is scheduled around it.

  Two changes to that loop, which compose better than they multiply:
  `vec4` reads take it from two workgroup loads per multiply-add to two per
  four, and **one element of padding per staged row** stops adjacent threads —
  which differ by key slot, so they read `slot * D` apart, a multiple of the
  bank count at D=128 — from serialising on one memory bank. Alone they measure
  0.81 and 0.78; together 0.35.

  Anima's forward at 832x1216 goes from **4.90 s to 1.78 s**, measured against
  `origin/main` in a worktree in the same session so the clock could not move
  between them, with `matmulQ8` reading 882 ms on both as the control.
  Attention reaches **21% of this device's measured 50.4 TFLOP/s**, from 7.7%.
  That is 39% of what its arithmetic intensity allows — at 16 FLOP per byte of
  k/v against a measured crossover of 29.6, bandwidth caps this kernel at 54% of
  compute peak whatever else is done to it.

  Two things were tried and measured worse, and are recorded rather than
  quietly dropped: FlashAttention-3's ping-pong scheduling (a tie — its other
  two ideas need asynchronous copy and an FP8 dtype that WGSL and this adapter
  do not have), and fixing `D` to a compile-time 128 so the score loop could
  unroll (1.06).

- **The kernels are tuned** (issue #177). `ops/matmul` reaches 70-72% of an RTX
  5090's measured roofline where it reached 1.4-3.0%, `matmulQ8` 42-49% where it
  reached 1.1-2.3%, and `ops/flash_attention` 8.8-9.1% where it reached 3.4%.
  Anima's forward at 832x1216 goes from 8.31 s to 5.07 s.

  All three now hold a thread tile in registers rather than computing one output
  per thread, and flash attention carries 16 query rows per workgroup rather
  than one. Every constant was **measured**: the sweeps generate the shapes the
  device can dispatch, check each against the op's own reference on a ragged
  case, and time the survivors — `ops/matmul/tools/bench.ts`,
  `bench-q8.ts` and `ops/flash_attention/tools/bench.ts`.

  Two device limits were never being requested, and Dawn said so in the same
  words it used for `maxBufferSize` before: `maxComputeWorkgroupStorageSize`
  (default 16384, this adapter 49152) and `maxComputeWorkgroupSizeX` (default
  256, adapter 1024). Without them most of a sweep cannot run, and the best of
  what runs under the defaults reads like the best there is.

- Anima's DiT uses `ops/flash_attention` where it used `scores` + `context`
  (issue #177). The forward goes from 8.31 s to 7.72 s at 832x1216, and the
  `[H, L, S]` score matrix stops existing: self-attention was materialising
  0.50 GB of it per dispatch, with a 512 MB budget cutting the heads into
  batches to make it fit.

  **Fusing is not why anything got faster, and it barely did.** Measured on
  their own at both shapes this model uses, the fused kernel is 1.1x the split
  pair, and both sit at 1.5-1.8% of this device's measured roofline of 50.7
  TFLOP/s. The score matrix was never the constraint. What is, is the inner
  loop both share — one thread per output, no register tiling — which is what
  #177 is now about.

- **Anima-3.8B in a browser** (`examples/anima-web`, issue #170 stage 6). The
  same pipeline as `examples/anima`, with weights over HTTP `Range` into the
  browser's disk cache and the DiT resident between steps.

  `animaForwardResident` gains an `onBeforePrefix` hook: the forward reads
  weights synchronously, so the page is told what is about to be needed and
  hydrates its heap a block at a time from disk. `verify-forward-gpu.ts` now
  asserts that the announced prefixes cover every tensor read — 1,049 against
  57 — because a gap there is a run-time failure in the page, after a 4.7 GB
  download, and nowhere else.

- **Anima-3.8B produces an image** (issue #174). Wan 2.1's VAE decoder matches
  ComfyUI's at rel-RMS 6.917e-7 on the GPU and 4.669e-7 on the CPU, and
  `generate.ts --vae` writes a PNG.

  No `conv3d` was needed, and that is a derivation rather than a shortcut. For a
  single frame `WanVAE.decode` computes `iter_ = 1`, so no frame cache exists;
  `CausalConv3d` takes its `x.shape[2] == 1` fast path, which truncates the
  weight to its last temporal tap; and `Resample`'s `time_conv` lives entirely
  inside the cache branch and never runs. What is left is Conv2d, an RMS norm
  over channels, SiLU, nearest 2x upsampling and one single-head attention —
  every one an op this repository already had.

- Anima-3.8B generates a latent from a prompt on the GPU
  (`examples/anima/src/generate.ts`), and the whole loop is checked against the
  same loop in torch step by step — final rel-RMS 3.274e-3 over 8 steps from the
  same noise and the same conditioning. There is no image yet: Anima decodes
  with Wan 2.1's 3D causal VAE, tracked as issue #174. The page writes the
  latent and the 16-to-3 projection ComfyUI shows as a live preview.

  `convert_dit.py` now records the model's configuration in its manifest, read
  by `detect_unet_config` rather than transcribed. Before this a runtime had to
  be told the shape out of band, and the only place carrying it was a test
  fixture.

- Anima-3.8B's two tokenizers (`examples/anima/src/tokenize.ts`), matching
  `transformers` and `tokenizers` on 23 cases each. `llm/tokenizer.ts` grows one
  path it did not have: a model with `byteFallback` off now emits `unk_id` for a
  codepoint no piece covers, merging adjacent ones as SentencePiece does,
  instead of throwing. Before this it refused any prompt containing one unusual
  character.

  T5's `Precompiled` normalizer is a 316 kB charsmap, and the port does not
  carry it. `tools/gen_tokenizer_data.py` compares it against NFKC at every
  codepoint in the BMP and emits the 51 that differ as a table — 31 of them
  control characters, where the charsmap maps tab and newline to a space and
  NFKC does not. So the port is the engine's own `String.normalize("NFKC")`
  plus a measured exception list, derived on every run rather than transcribed.

- Anima-3.8B's sampler (`examples/anima/src/sampler.ts`) — the `beta` schedule
  and `res_multistep`, matched step by step against ComfyUI's own on a toy
  denoiser. It is not Z-Image's sampler with different numbers: that one takes
  Euler steps down a linear schedule, this one inverts a Beta(0.6, 0.6) CDF onto
  a 1000-entry table and takes second-order exponential multistep steps.

  `multiplier` is 1.0 where `ModelSamplingDiscreteFlow` defaults to 1000, so the
  DiT is handed sigma itself rather than sigma times a thousand. The table
  cannot catch a wrong multiplier — it cancels there — so `timestepOf` is where
  it is pinned.

- Anima-3.8B's conditioning path (`examples/anima/src/text-encoder.ts`) matches
  the model at rel-RMS 9.775e-7 — Qwen3-0.6B, then the `llm_adapter` that ships
  inside the DiT checkpoint. Two tokenizers run over the same prompt: the Qwen
  ids condition the 0.6B whose hidden states become the adapter's keys, and T5
  ids index the adapter's own embedding table to become its queries. The T5
  model itself is never loaded.

  The encoder is **not** quantized, and that is measured rather than stylistic:
  q8 on its 196 layer matrices moves its output by rel-RMS 0.223, against 0.028
  for the adapter and 0.040 for the whole 52-block DiT. A 0.6B has one absmax
  scale per 1024 numbers to spend and its outlier channels do not fit in it;
  0.6 GB saved is not worth conditioning on different words.

- Anima-3.8B's DiT runs resident on the device (`examples/anima/src/dit-resident.ts`),
  matching the model's own forward at 1.182e-5 against its quantized weights —
  against 4.018e-2, measured separately in torch, for what q8 itself costs over
  52 blocks. One forward takes 0.2 s once the 4.94 GB of weights are resident,
  against 814 s for the CPU port of the same forward.

  Anima's three rope axes do not share a base (t=10000, h/w=42870.9, from an
  extrapolation ratio of 4.0) and `ops/rope`'s `axes` entry takes one base per
  dispatch. Rather than gathering each axis's channels out of every head, the
  whole head is dispatched three times with the other two axes' positions set to
  zero: an angle of zero is the identity whatever base is in force, so each pass
  rotates one axis and leaves the rest as it found them.

- The DiT's activations stay on the device (`dit-resident.ts`). The per-dispatch
  path moved 28.57 GB up and 19.85 GB back per forward at 1,039 tokens, of which
  only 6.17 GB was weights; the rest was activations going out to the CPU and
  straight back in. One forward went from 20.2 s to 0.17 s once the weights are
  resident — against 990 s for the CPU reference.

- `harness/wgsl.ts` and `harness/resident.ts` request the adapter's own limits
  rather than a fixed 512 MiB. That constant was why the VAE could not decode at
  1024: Dawn's error said the adapter supports 1 TiB and that it had to be asked
  for.

- Both resident runtimes report an out-of-memory `createBuffer` where it
  happens, with the size and what was already allocated. It does not throw on
  its own — the first thing to notice was `createBindGroup`, several frames
  later, naming a binding index and no size.

- `examples/zimage-web`: **Z-Image runs in a browser.** A prompt goes in a text
  box and a picture appears on a canvas — tokenizer, Qwen3-4B text encoder, DiT
  and VAE decoder, all on WebGPU, all composed from this repository's kernels
  (issue #166). It calls the same functions the Node verifiers hold to the
  model rather than carrying a browser copy of them, because the copy is what
  would drift away from the numbers those verifiers earned.

- `examples/zimage`: the whole pipeline, each part measured against Z-Image's
  own implementation before being wired to the next — one block at the shipped
  width (8.72e-8), the full DiT forward (4.386e-6), the text encoder
  (7.993e-7), the sampler's schedule (exact), and generation end to end.

- The DiT's GPU path reads packed q8 weights through `ops/matmul`'s `q8` entry
  instead of `ops/dequant_transpose` followed by `matmul`. The dequantised
  operand is four times the packed weight and was crossing the bus twice; one
  forward went from 26.9s to 6.2s, 160x the CPU reference, with agreement
  unchanged at 4.386e-6.

  Worth recording alongside it: batching attention across its 30 heads removed
  57% of a forward's dispatches and changed the wall clock by **nothing**. The
  count was never the constraint; the bytes were.

- `npm run lint` now type-checks `examples/zimage-web`. It was the one
  directory nothing checked, and esbuild does not read types — which hid a call
  missing an argument until it threw in a browser, and a second bug that had
  not been reached yet.

- `examples/zimage`: the block from #163 now runs on the **shipped weights at
  the shipped width** (issue #166, stage 1), not on a scaled-down golden with a
  random seed. `tools/convert_dit.py` turns the 12.31 GB checkpoint into 3.34 GB
  of q4-g128, `src/weights.ts` reads it back, and `src/verify-real-block.ts`
  puts `layers.0` through `block.ts` at `dim=3840` / `head_dim=128` / axes
  `[32, 48, 48]`.

  The port matches the model to **8.72e-8 relative RMS** on identical quantized
  weights. Against the full-precision model the gap is 1.778e-2 — which is, to
  four figures, exactly what quantization costs when measured independently in
  torch. Two goldens rather than one, so a porting mistake and the format's cost
  cannot be attributed to each other.

  This also answers what #137 reopened. On `layers.0`, quantizing every Linear
  to q4 costs 0.0519 relative RMS, and `adaLN_modulation` alone accounts for
  0.0478 of it — it produces the scales and gates that multiply the whole
  residual stream, so its error is multiplicative across all 3840 channels. The
  converter therefore keeps adaLN at q8 and the rest at q4: +2.6% bytes per
  layer, and 0.0519 down to 0.0178. `q4-g32` was measured too and is worse on
  both counts (0.0398, and 0.25 more bits everywhere).

- `llm/tokenizer-bpe.ts`: a byte-level BPE tokenizer, which is what Qwen uses
  and therefore what Z-Image's text encoder needs (issue #153). A separate
  module from `llm/tokenizer.ts` rather than an option on it: that one is
  SentencePiece **unigram**, where the vocabulary carries a score per piece and
  encoding is a Viterbi search for the likeliest segmentation. BPE has no
  scores and replays an ordered merge list instead. The two share a signature
  and nothing else.

  Four stages in a fixed order, because each one's output is the next one's
  input: NFC (declared by the checkpoint, and without it text arriving
  decomposed from a macOS filesystem tokenises differently), the checkpoint's
  own pre-tokenisation regex, the GPT-2 byte alphabet (which is why there is no
  unknown token — every byte is representable), then the merges.

  Correctness is 32 cases run through the real `tokenizers` library
  (`llm/tools/gen_bpe_fixtures.py`), covering Japanese, Chinese, Korean,
  Cyrillic, RTL Arabic, ZWJ/skin-tone/flag emoji, code, URLs, whitespace runs,
  the pre-tokeniser's contraction list, and an NFC/NFD pair that must encode
  alike — the same case list the SentencePiece side (#104) found to matter,
  plus the ones specific to BPE. `decode` skips special tokens by default
  because the reference does (measured: `[151644, 872]` gives `"user"` there,
  `"<|im_start|>user"` with the flag off), and takes `{ skipSpecial: false }`
  for the times a caller wants the chat scaffolding visible.

  A normalizer other than NFC is refused rather than ignored, since ignoring
  one silently changes every id and no test written against this module alone
  would notice.

- The GPU tests stop dying for no stated reason. `harness/wgsl.ts` and
  `harness/resident.ts` now keep the `GPU` instance and its adapter reachable
  for as long as the device they produced, because the Dawn Node binding does
  not: nothing in a `GPUDevice`'s object graph refers back to the instance, so
  once the factory returned, the collector was free to take it — and a device
  whose instance has been collected starts failing dispatches by aborting the
  process on a glibc futex assertion, segfaulting, or hanging, differently on
  each run.

  That is one bug, and it is the one behind issues #38, #49, #68 and #107. Each
  of those recorded a different face of it: "a test that takes more than a few
  milliseconds before its first dispatch kills the worker" (#49) and
  "real-model-scale CPU work before a dispatch crashes the binding" (#107) are
  both descriptions of *more time for the collector to run in*; "a file dies
  once it holds too many dispatches" (#68) is the same pressure from allocation
  volume. The four vitest pool configurations tried against #38 could not have
  helped — vitest calls a test body as a function, so an instance created inside
  one is unreachable the moment that body returns, whichever pool is in use.

  Measured A-B-A on `ops/gqa/wgsl.test.ts` (46 GPU cases, RTX 5090, driver
  610.57.04, `webgpu` 0.4.0, Node v25.6.1, otherwise-idle GPU): **0/5 runs
  completed** without the retention, **5/5 green** with it, **0/4** after
  reverting. Nothing but that reference changed.

  Diagnosis and the 35-line minimal repro came from the voxshot session, which
  isolated it to `nested` vs `nested-keep` with no model and no weights. What
  this repository had accumulated instead was three weeks of symptom
  descriptions and workarounds — splitting test files (#68), retrying, and
  treating the whole class as "Dawn flake".

- `ops/axpy` (issue #152): `out[i] = y[i] + a * x[i]` with a **scalar** `a`,
  in two entry points — `kernel` (out-of-place) and `inplace` (`y[i] += a *
  x[i]`, BLAS `saxpy`'s own signature). A rectified-flow sampler's entire
  scheduler step is `latent += dt * velocity`, once per diffusion step, and
  `ops/elementwise` cannot express it: it takes two **equally sized** arrays,
  so the scalar had to be materialised as a full-length buffer of copies of
  `dt` and pushed through multiply-then-add. That is two dispatches, twice the
  memory traffic, and — because `dt` changes every step — a latent-sized
  upload per step to carry one number.

  It is also a **different answer**, which is the part worth knowing before
  swapping one for the other. Rounding the product to f32 before the add is
  not the same as rounding once: at `a = f32(0.1)`, `x = 3`, `y = -f32(0.3)`
  multiply-then-add cancels to exactly **0** while a single rounding gives
  **-2^-27**. Which is correct is not this library's call (rule 7) — measured,
  `torch.add(y, x, alpha=a)` returns the single-rounded value on CPU and on
  CUDA alike, and so does this op on this device, whose WGSL compiler
  contracts `y + a*x` into an FMA. Over 4,096 elements at `a = 0.37` the fused
  kernel matched the reference on all 4,096 where the two-dispatch path
  matched on 3,214, and no element was ever further from it — the test asserts
  that ordering, not just agreement.

  In-place is a separate entry point rather than a way of calling the
  out-of-place one, because the obvious shortcut fails **silently**: binding
  one buffer to `kernel`'s read-only `y` and its read_write `output` passes
  `createBindGroup`, then invalidates the command buffer at `finish()`
  (`usage (Storage(read-write)|Storage(read-only)) includes writable usage and
  another usage in the same synchronization scope`), so the submit is dropped
  and the readback is all zeros with nothing thrown — #46's failure mode
  again. One `read_write` binding read and written by the same invocation is
  well defined and measured working across a four-step resident loop.

  `a` rides in the 16-byte uniform next to `N`, so the per-step cost of a
  changing coefficient is one 16-byte `queue.writeBuffer` — the shape
  `harness/resident.ts` already sanctions for position counters — and nothing
  else is re-uploaded, buffers, pipeline and bind group all built once
  (asserted against `ResidentDevice.stats`). That is deliberately not #144's
  trap of re-sending unchanged bytes every call: here the only bytes that move
  are the ones that changed.

  Measured (rule 9; RTX 5090, driver 610.57.04, Dawn via `webgpu@0.4`, Node
  v25.6.1, f32, GPU timestamps, median of 5 per session, three sessions,
  otherwise-idle GPU; ceiling measured by `harness/roofline.ts` in the same
  sessions, 1.69-1.72 TB/s). At **N = 262,144** (one 16×128×128 latent):
  **6.5 µs against 12.8-16.9 µs** for `elementwise(multiply)` +
  `elementwise(add)`, i.e. **2.0-2.6x** — but at 480-487 GB/s, only **28% of
  the ceiling**, so at that size neither path is bandwidth-bound and the win is
  doing half the work rather than doing it faster. At **N = 1,048,576** it is
  bandwidth-bound: **7.7-9.0 µs against 17.1-18.4 µs**, **2.0-2.4x**, at
  1.40-1.63 TB/s = **83-95% of the ceiling**, which is what a kernel that moves
  12 bytes per element and does one FMA with them should reach.

  The larger size is in this entry only because #161 landed. While this op was
  being written, N = 1,048,576 aborted with `std::system_error` and then hung,
  three times running, and this entry said "unmeasured" — that was the collected
  `GPU` instance described above, not a size limit, and it reproduces no longer.
  Nothing in this op changed between the two measurements.
- `ops/matmul` gains a `matmulQ4G128` entry point (issue #149): the tiled GEMM
  of `matmulQ8`, reading the q4 format `ops/matvec` defines — `[M, ceil(K/8)]`
  packed nibbles plus `[M, ceil(K/128)]` per-group scales — with in-kernel
  dequant and no transpose pass. Prefill is the half of inference that runs
  through GEMM, so a quantization format that only has a GEMV covers only
  decode; this is what lets a 4B-parameter text encoder be read at all, at
  4.25 bits per weight (≈1.98 GiB for 4e9 parameters, arithmetic from the
  measured bits-per-weight — no checkpoint of that size has been converted
  here).

  The format is **not re-decided** here (rule 7 — issue #149 says so in as many
  words): `ops/matmul/q4.reference.test.ts` checks that by running this GEMM
  against `matvecQ4G128` row by row, so the two ops cannot drift into two
  spellings of "the q4 format" while each agrees with its own kernel.

  **No bias**, despite issue #149 asking for one. `matmul` and `matmulQ8` have
  none, `biasAdd` (issue #150) composes by addition, and rule 8 wants the plain
  arithmetic agreeing with the reference before anything is fused into it.
  Speed is unmeasured.

- `ops/matvec` gains the **q4 weight format** and a `matvecQ4G128` entry point
  (issue #137): 4-bit codes packed eight to a `u32` (least-significant nibble
  first — `packQ8`'s byte order at half the width), with one absmax scale per
  **group of 128 contiguous columns** rather than one per row. `quantizeQ4G128`
  produces the codes and scales, `packQ4` puts them on the wire, and
  `matvecQ4G128` (reference + WGSL kernel) reads them without a dequant pass.

  Exists because q8 is the only quantization this library had, and 8 bits is
  the wrong size for the models it is being pointed at next: a 0.6B model fits
  a browser at q8, a 3B or 4B one does not. Four bits without a group axis is
  not a usable substitute — issue #137's measurements (voxshot's, on
  MioTTS-0.6B, not this repository's) put per-row q4 at 4.5e-1 peak-relative
  logit error against q8's 4.8e-2, with greedy agreement collapsing from 6
  tokens to 1; group-128 recovers about 3x of that for 0.24 extra bits per
  weight. This repository's own synthetic measurement shows the mechanism
  directly (README, "The q4 format"): on columns two orders of magnitude
  smaller than their row's peak, per-row q4 has RMS-relative error of exactly
  **1.0** — every code rounds to zero — where group-128 is unaffected.

  Three conventions are stated rather than picked quietly (rule 7), because
  each has a live alternative that produces different numbers: the range is
  symmetric **`[-7, 7]`** and not Q4_0's `[-8, 7]` (which measured *best* of
  every configuration tried on weight RMS error and still flipped the argmax
  in 4 of 4 cases, because clipping one tail biases the error instead of
  randomising it); the scale's reciprocal is formed as `7/absmax` in f64, not
  as `1/f32(scale)` as llama.cpp does; and rounding is `Math.round`'s
  ties-toward-`+Infinity`, matching `ops/quantize` and
  `llm/tools/quant_common.py`. The format is **not** called Q4_0-compatible —
  block size, range and scale dtype all differ.

  Speed is unmeasured, and so is the effect on any real model's output.
- `conv2d` / `conv2dOutputSize` in `ops/conv`, with a WGSL kernel at
  `ops/conv/wgsl/conv2d.wgsl` (issue #145). `ops/conv` shipped 1D only, so
  anything with a spatial input — the image decoders #148 is about — had no
  convolution at all. It lives beside `conv1d` rather than in an op of its own
  because every convention it needs is one `conv1d` already settled against
  torch 2.10.0+cu128 (cross-correlation, zeros on both ends of each axis, bias
  once per output element, `groups` splitting both channel axes, throwing where
  torch raises); two files could drift, and a 2D op that flipped its kernel or
  padded its edges differently from the 1D one beside it would be a wrong
  answer of the right shape. The conventions were re-measured in 2D rather
  than assumed, because two of them can only be checked there — a flip
  reverses *both* axes, and a pad has four edges.

  `stride` / `padding` / `dilation` take `number | [H, W]`, PyTorch's own
  `int | tuple[int, int]`, with the pair order measured rather than read off
  the docs (`stride=(2,3)` and `stride=(3,2)` on a `[1,1,4,6]` input return
  `[1,1,2,2]` and `[1,1,2,3]`, so the first member is H). `padding` stays an
  integer count: `'same'` / `'valid'` are refused for the reason `conv1d`
  gives, now with the measurement behind it — torch splits `'same'`'s odd
  total pad asymmetrically for an even kernel, which one integer per axis
  cannot express, and torch itself rejects `'same'` with stride > 1.
  Deliberately absent: `conv_transpose2d`, `padding_mode`, 3D.

  **Speed unmeasured.** One thread per output element, no tiling, workgroup
  256 along the contiguous (W) axis — the same shape `conv1d` uses, kept so
  that whatever #134 measures lands on one shape rather than two.
- `ropeAxes` (issue #151), exported from `ops/rope` beside the 1-D op, with a
  second WGSL entry point `ops/rope/wgsl/axes.wgsl`. Z-Image's DiT gives every
  token a `(t, y, x)` triple and splits the head dim `[32, 48, 48]` so each
  block turns by its own position; `rope` can express one position per token
  and nothing else, so this was the one op of #148's inventory that no existing
  kernel could stand in for. Not three `rope` calls: the blocks share a head and
  a dot product, so that would be three passes writing disjoint thirds of the
  same tensor.

  Three conventions had to be settled and all three are taken from the
  implementation this exists to run (Tongyi-MAI/Z-Image @ `26f23ed`,
  `src/zimage/transformer.py`), not chosen here — `torch` has no RoPE to defer
  to, so rule 7's fallback is the model's own code. **Positions** arrive as an
  explicit `[N, axes]` `Int32Array`, upstream's `ids` flattened, rather than
  derived from a patch grid the op would then have opinions about. **One shared
  `thetaBase`** (256 there), with the exponent normalised by *the axis's own*
  channel count rather than the head's, which is what lets a 32-channel axis
  and a 48-channel one sweep the same frequency range; dividing by the head dim
  instead is the plausible wrong answer and moves Z-Image's own geometry by 2.6
  in absolute terms. **Adjacent-channel pairing** (`2i`/`2i+1`), the same as
  `rope` and *not* HF Llama's `rotate_half`, so a Z-Image checkpoint needs no
  `permuteRopeChannels` — recorded in the README because getting it wrong
  produces a tensor rather than an error.

  Angles are computed rather than tabulated, which makes upstream's `axes_lens`
  not a parameter and gives a negative position the rotation it means instead of
  Python's wrap to the end of a table — the one deliberate divergence, and the
  reason positions are `i32`. An odd axis dim throws, because upstream cannot
  express one either (`axes_dims=[3, 3]` fails in torch 2.10.0 with a shape
  mismatch, measured), so there is no convention to inherit and only ones to
  invent. No `scaling`, no head range, no angle cache: Z-Image asks for none of
  them, and `ropeCacheAxes` waits for a measurement that says it is needed
  (#148, rule 8) rather than being written on spec.

  `ropeAxes` called with one axis equals `rope` **bit for bit** — asserted, not
  assumed, at three offsets — which is the direct evidence that generalising the
  op did not change the op. Correctness against upstream is pinned by
  `ops/rope/axes-cases.ts`, generated by importing upstream's own
  `RopeEmbedder` and `apply_rotary_emb` and running them
  (`ops/rope/tools/gen_axes_fixture.py`); the two agree to 2.384e-7, one f32 ulp
  at the fixture's magnitude, which is the expected gap given that upstream
  rounds the angle to f32 before `cos`/`sin` and this reference does not. Speed
  unmeasured, like every other op here.
- `ops/elementwise` gains `elementwiseRows` and a second WGSL entry point,
  `rows` (issue #150): `add` and `multiply` with the right-hand side broadcast
  along the last dimension — `out[s, d] = a[s, d] ⊕ b[d]` for `a` of `[S, D]`
  and `b` of `[D]`. `add` is a `Linear`'s bias (**biasAdd**) and `multiply` is
  AdaLN's per-channel scale (**rowwiseAffine**); Z-Image's DiT (#148) needs
  both, and until now `elementwise` threw on any pair of unequal lengths, so
  a bias had to be materialised as a full `[S, D]` copy on the host first.

  It is a **separate function taking `S` and `D` explicitly**, not a length
  check added to `elementwise`. Inferring the broadcast from the lengths was
  rejected because the lengths do not carry the intent: a `[3, 3]` activation
  and a `[3]` vector admit two different broadcasts — across the columns and
  down the rows — and both return well-formed, finite, *different* answers
  (torch 2.10: `torch.arange(9).reshape(3,3) + c` is `[[1,3,5],[4,6,8],[7,9,11]]`,
  `+ c.unsqueeze(1)` is `[[1,2,3],[5,6,7],[9,10,11]]`). An inferring
  `elementwise` would have to pick one silently, which is issue #143's
  "returns a plausible value instead of throwing" class. Stating the shape
  turns a caller's mistake into a contradiction the op can refuse: a `b` that
  is not `D` long, an `a` that is not `S*D` long, or a missing/non-positive
  dimension all throw, in `ops/gqa`'s message format.

  Broadcasting aligns from the right, as NumPy and PyTorch do, and only the
  last dimension is supported — so `b.length === a.length` is a mistake here
  and not a same-shape fallback (torch refuses `[2,3] + [6]` for exactly that
  reason, verified against 2.10 rather than recalled).

  The existing same-shape path is **untouched**: `elementwise` and
  `wgsl/kernel.wgsl` are byte-for-byte what they were, so the residual add and
  SwiGLU multiply in `llm/engine.ts`, `llm/engine-q8.ts` and
  `llm/engine-q8-resident.ts` cannot have changed behaviour. That is also why
  the broadcast is a new `.wgsl` file rather than a third field on the shared
  `Params` struct — every current binder of that kernel uploads a two-field
  struct, and a struct change would have made "unchanged" an argument instead
  of a diff that does not touch it. Speed unmeasured.

- `ops/upsample`: `nearestUpsample2d`, nearest-neighbour 2D resampling over
  `[N, C, H, W]` (issue #146). This library had **no resampling op at all** —
  the nearest thing was `convTranspose1d`, which raises resolution with learned
  weights. Image decoders that avoid checkerboard artefacts do it the other
  way, "nearest upsample then conv", so they need both and neither one
  substitutes for the other.

  Matches `torch.nn.functional.interpolate(x, size=(outH, outW),
  mode='nearest')` measured against torch 2.10.0+cu128, and the two conventions
  that could have been chosen silently are written into the API instead:

  - It takes the **output size**, not a scale factor, because torch's two
    entry paths genuinely disagree. At `H = 3`, `scale_factor=1.6` maps the
    four output rows to sources `0, 0, 1, 1` while `size=(4, ...)` maps them
    to `0, 0, 1, 2`. Taking the size keeps the output shape a matter of
    integers rather than of how a float scale rounds, and the two agree
    wherever the ratio is a whole number — including the 2x every decoder
    actually asks for.
  - The source index is `floor(dst * f32(inSize / outSize))` with the multiply
    **in f32**, which is torch's OpenCV `INTER_NEAREST` formula and not the
    same function as exact integer arithmetic: at `H = 14 -> 46` destination
    row 23 lands exactly on source row 7 in exact arithmetic and on row **6**
    in f32, which is what torch returns. Getting this wrong shifts a whole row
    of an image with no error and no shape change.
  - `align_corners` is **not** a parameter and must not be added: torch raises
    for it in this mode ("align_corners option can only be set with the
    interpolating modes"), because it aligns a sample grid before interpolating
    between neighbours and nearest interpolates between nothing.
  - Downsampling throws rather than quietly evaluating the same formula, which
    is out of scope for the issue and is measured against nothing.

  The f32 ratio is divided **on the host** and passed to the shader, because
  WGSL allows f32 `/` 2.5 ULP of error while requiring `*` to be correctly
  rounded. That is not a precaution taken from the spec: dividing inside the
  shader instead makes the `14 -> 46` case return source row 7 on an RTX 5090
  (Dawn, f32), disagreeing with torch and with the reference.

  Speed unmeasured.

- `LlamaEngineQ8Resident.forward()` (and `harness/resident.ts#ResidentDevice.batch()`)
  gain an opt-in `ForwardProfile`/`BatchProfile` argument (issue #131): a
  per-call breakdown of where one `forward()` call's own wall time goes —
  `packInt8Rows` CPU cost, `queue.writeBuffer` bytes/time, bind-group
  creation (both a raw per-call sum and the wall-clock time of prefill's own
  per-layer `Promise.all` block), GPU submit-to-completion wait, readback,
  and — only when the caller also opts into `ForwardProfile.wantGpuBreakdown`
  and the device negotiated `timestamp-query` — one GPU duration per labeled
  dispatch. `undefined` by default (every existing caller), so this changes
  no dispatch, no bind group and no arithmetic for anyone not asking for it;
  proved directly by a bit-for-bit logits comparison between a profiled and
  an unprofiled prefill call (`llm/engine-q8-resident.profile.wgsl.test.ts`).
  `wantGpuBreakdown` exists as a second, separate opt-in (PR #141 review)
  because the GPU-side breakdown costs a dedicated compute pass per labeled
  dispatch (~500 extra pass boundaries at Sarashina2.2-1B's 24-layer scale)
  — real overhead that must not land on a caller who only wants the cheap
  CPU-side fields, and did in an earlier version of this change before
  review caught it (that version's own first measurement had `packInt8Rows`
  alone reading larger than an unrelated PR's entire unprofiled prefill
  total, which cannot be right for a sub-phase of the same call).

  Exists to answer #131's own question — a reviewer comment on PR #130
  named `matmulQ8IntoShape`'s per-`forward()` re-pack-and-reupload of the
  same ~1 GiB of weight bytes decode already keeps resident as prefill's
  likely fixed cost, and this measures it directly instead of guessing
  again. README's new "Where prefill's ~1.2s fixed cost actually goes"
  section has the measured numbers, checked against a genuinely unprofiled
  control run in the same session: `packInt8Rows` alone is ~78-80% of
  prefill's own control wall time, against 1.7-5.3% for actual GPU kernel
  time — instrumentation overhead itself measured at noise level (roughly
  ±2%) once `wantGpuBreakdown` correctly gates the pass-splitting cost.
  **No optimization is included in this change** — per this issue's own
  scope, only the measurement; the fix itself (bind decode's already-
  resident weight buffers into prefill's `matmulQ8` instead of re-packing)
  is tracked separately as issue #142.

- `ops/matmul` gets a `matmulQ8` entry point (issue #128): the same tiled
  GEMM as plain `matmul`, but the right-hand operand is read as a packed
  int8 weight — `matvecQ8`'s own `[M, ceil(K/4)]` wire format — with
  in-kernel dequant instead of a separate f32 operand. `LlamaEngineQ8Resident.runPrefillResident`
  used this to replace `ops/dequant_transpose`+plain `matmul`: that pair
  dequantized-and-transposed every projection's packed weight into a
  `[K, M]` f32 buffer once per `forward()` call, purely so `matmul` had an
  operand to read — a full GPU write of `inFeatures * outFeatures` f32
  values immediately followed by a full read of the same, on every prefill
  regardless of prompt length. `matmulQ8` removes that pass: no transpose,
  no intermediate buffer, the kernel reads the packed weight directly.
  `ops/dequant_transpose` itself is unchanged and still used —
  `llm/kernels.ts#runDequantTranspose` (that op's own Node-side integration
  test path) and `examples/llm-demo/src/browser-runtime.ts`'s WGSL parity
  table both still reference it.

  **Measured, and the result is smaller than hypothesized — reported
  honestly rather than assumed (rule 9):** `examples/llm-demo`'s
  `__decodeFixedCostBenchmark` (RTX 5090, NVIDIA driver 610.57.04, Chrome
  151.0.7922.71, real Sarashina2.2-1B-alibi-v1 checkpoint, synthetic token
  ids, one `LlamaEngineQ8Resident` instance, `reset()` between calls — the
  exact "聞く層+スタイラ, reset() 運用" shape
  this issue asked about), 8 samples per prompt length across three runs,
  comparing this change against the immediately preceding commit (PR #127,
  `dequant_transpose`+`matmul` still in the prefill path):

  | prompt length | before (dequant_transpose+matmul) | after (matmulQ8) | delta |
  | --- | --- | --- | --- |
  | 76 tok | 1197.0ms avg (1157.5–1273.3ms, n=8) | 1175.0ms avg (1138.8–1200.7ms, n=8) | ~22ms (~1.8%) |
  | 365 tok | 1275.9ms avg (1262.0–1308.7ms, n=8) | 1273.0ms avg (1253.7–1285.4ms, n=8) | ~3ms (~0.2%) |

  Both deltas sit inside the run-to-run spread of either condition (roughly
  ±30–50ms at this prompt length, on an otherwise-idle GPU — `nvidia-smi`
  checked at 0% utilization between runs, so this is not GPU contention from
  another process). `ops/dequant_transpose/reference.ts`'s own doc measured
  its *own* GPU-side cost at ~170ms total across 24 layers for one
  projection shape at issue #117's time — a real number, but small enough
  next to prefill's ~1.15–1.28s fixed cost that removing it outright lands
  inside this measurement's own noise floor rather than producing a visible
  win. The fixed cost this issue set out to explain (76-token and 365-token
  prompts landing at effectively the same wall-clock time, `alibi-ai`'s own
  "プリフィル固定費≈1.2秒がトークン数非依存" observation) reproduces exactly
  in both the before and after numbers above — so it is real, but its source
  is **not** primarily `dequant_transpose`, contrary to this issue's own
  working hypothesis. The most likely remaining source, unmeasured here and
  out of this change's scope: `runPrefillResident`'s per-layer `Promise.all`
  of nine `device.bindGroup()` calls (`PR #119 review, item 7`'s own doc
  already named the `pushErrorScope`/`await popErrorScope()` round trip
  inside each one) still runs once per layer, 24 sequential awaited rounds
  per prefill call, independent of prompt length — a plausible next target,
  not confirmed as the cause.

  Correctness: `llm/engine-q8-resident.wgsl.test.ts`'s fixture gate (prefill
  and decode logits vs. the pre-optimization engine) passes unchanged — abs
  diff ~1.2e-7 for prefill logits, matching the pre-existing float32
  rounding noise this fixture's tolerance was already sized for, not a new
  source of error.

- `LlamaEngineQ8Resident.runPrefillResident` no longer re-packs and
  re-uploads every projection's int8 weight on every `forward()` call —
  issue #142, following up directly on #131/#141's own profiling of the
  ~1.2–2.2s prefill fixed cost the entry above left unexplained.
  `packInt8Rows` (CPU) turned out to be **77–79% of prefill's own fixed
  cost** (#131/#141's own measured breakdown; the GPU kernel itself was only
  2–6%) — and the packed bytes it produced every call were already sitting
  on the GPU: `buildProjection`/`buildFfnProjection`/`buildResidualProjection`
  (`llm/engine-q8-resident.ts`) upload the exact same `packInt8Rows` output,
  for the exact same `weights.layers[l]` object, once in `create()`, to
  build decode's own `matvecQ8`/`matvecQ8Ffn`/`matvecQ8Residual` bind
  groups. Those three builders now keep the resulting `GPUBuffer` handles
  (`ResidentWeight`, stored per layer) instead of discarding them once the
  decode bind group is built, and prefill's `matmulQ8` bind group
  (`bindMatmulQ8`, replacing `matmulQ8IntoShape` for every production
  projection) binds those same resident buffers directly — no
  `packInt8Rows`, no `queue.writeBuffer`, no new `GPUBuffer`, on the second
  and every later prefill call in a generation (`reset()`, issue #120), and
  on the very first one too.

  **VRAM residency is unchanged.** These bytes were already resident for
  decode; prefill now reads the same allocation instead of paying for a
  second, transient ~1 GiB one that lived only until that call's own
  `batch()` resolved. Nothing new is kept alive for longer than before.

  **Measured** (RTX 5090, NVIDIA driver 610.57.04, Linux (Arch, kernel
  7.1.5-arch1-2), Chrome 151.0.7922.71 non-headless via CDP on a dedicated
  `--user-data-dir`/`--remote-debugging-port`, real Sarashina2.2-1B-alibi-v1
  int8 checkpoint (24 layers, hiddenSize=1792, vocabSize=102400),
  `examples/llm-demo`'s `__decodeFixedCostBenchmark`, one
  `LlamaEngineQ8Resident` instance, `reset()` between prompt lengths, 8
  samples per prompt length, same session, machine shared with other
  concurrent sessions — `uptime` read `load average: 0.6–1.3` throughout):

  | prompt length | before (`matmulQ8IntoShape`, per-call repack) | after (`bindMatmulQ8`, resident) | speedup |
  | --- | --- | --- | --- |
  | 76 tok | 1266.4ms median (1256.9–1277.0ms, n=8) | 42.7ms median (38.3–45.0ms, n=8; one 233.6ms outlier under concurrent GPU load) | ~29.7x |
  | 365 tok | 1344.4ms median (1326.9–1378.9ms, n=8) | 120.6ms median (119.0–121.6ms, n=8; one 317.0ms outlier under concurrent GPU load) | ~11.1x |

  Correctness: prefill and decode logits are **bit-for-bit identical**
  whether `matmulQ8`'s weight comes from `bindMatmulQ8` (resident) or the
  pre-#142 `matmulQ8IntoShape` (packed) — `llm/engine-q8-resident.ts` keeps
  the latter as a test/debug-only method, `debugPrefillWithPackedWeights`,
  precisely so `llm/engine-q8-resident.residentweight.wgsl.test.ts` can run
  both paths on the **same engine, same session, same device** and diff
  their output directly, rather than pinning literal floats from one
  GPU/driver into a fixture file — `rmsnorm`'s `rsqrt`, `rope`'s `sin`/`cos`
  and `gqa`'s softmax `exp` are all vendor/driver-dependent at the ULP level
  (rule 2), so only an in-session comparison is portable; a golden-fixture
  version of this test was caught in review for exactly that reason and
  replaced before merge. Exact equality (`toEqual`, no tolerance), per
  #130's own established criterion ("1ビットでも動いたら丸めではなく設計の
  違い"): both paths read the identical packed bytes through the identical
  `matmulQ8` kernel, so any difference would mean a design bug, not
  rounding. `resident.stats.buffersCreated`'s own per-prefill-call delta
  drops from a measured 75 to a measured 47 for the fixture in that same
  test (28 fewer buffers = `2 × 7 projections × 2 layers`, exactly the
  weight+scale pair this change stops re-allocating) — asserted with a
  margin threshold, mutation-confirmed by temporarily reverting
  `bindMatmulQ8` back to `matmulQ8IntoShape` and watching the assertion
  fail at the measured pre-#142 value.

## [0.2.0] - 2026-08-21

### Added

- The published package carries `llm/kv-cache`, `llm/reshape` and
  `llm/sampler`. Only `llm/tokenizer` shipped before, which made every other
  `llm/` module unreachable from an installed copy — a consumer linked with
  `file:` reads deep relative paths and so bypasses `exports` entirely, which
  is why this survived until a downstream project (voxshot, issue #138) tried
  to depend on a published version rather than a linked checkout. The three
  are named individually in `tsconfig.build.json` rather than globbed: an op's
  directory *is* its API, so `ops/*` is a correct statement of intent, while
  `llm/` still holds working code nobody has frozen as API, and a glob would
  make "compile this" and "let consumers import this" the same decision.
  `harness/distribution.test.ts` now fails when `exports` promises a module
  `include` never builds, so the two cannot drift apart again silently.

  This release is also the first to carry `ops/matvec`'s int8 entry points
  (`matvecQ8`, `packQ8`, and the fused `matvecQ8Ffn`/`matvecQ8Residual`) and
  their `.wgsl` kernels. They have been on `main` since #97/#99 but no release
  followed, so 0.1.0 offers `matvec` alone — enough for a codec, not for a
  quantized LM.

- `ops/matvec`: two decode-only fused entry points, `q8_ffn`
  (`silu(wGate·x) * (wUp·x)`, one dispatch reading both gate and up weights)
  and `q8_residual` (`residual + w·x`, folding a post-projection residual
  add into the projection itself) — issue #111. `LlamaEngineQ8Resident`'s
  decode step (`llm/engine-q8-resident.ts`) wires both in, cutting one
  token's real GPU dispatch count from 411 to 291 at Sarashina2.2-1B's shape
  (17→12 per layer × 24 layers): the FFN triad (`matvecQ8(gate)` +
  `matvecQ8(up)` + `activation(silu)` + `elementwise(multiply)`) collapses
  4→1, and each of `o_proj`/`down_proj`'s trailing residual add collapses
  2→1. Prefill (`runPrefillResident`) is unchanged — its projections go
  through `matmul`, not `matvecQ8`, so these two entry points do not apply
  there (out of this issue's own scope: "プリフィル専用最適化はスコープ外").
  Measured (RTX 5090, real Sarashina2.2-1B-alibi-v1 checkpoint): decode
  **7.2% lower latency / 7.7% higher tok/s** (4.471ms → 4.150ms/token,
  combined across 76- and 365-token prompts); prefill within measurement
  noise of itself, as expected since it was not touched. See README's
  "Fused decode kernels (issue #111)" for the full table and for why the
  issue's own opening "~1.2s regardless of prompt length" motivation turned
  out to describe prefill's fixed cost, not decode's — prefill fusion is
  tracked as separate future work.

- `LlamaEngineQ8Resident.reset()` (issue #120): starts a second, independent
  generation on an already-`create()`d engine — position counter back to 0,
  next `forward()` accepted as a new prefill — without rebuilding anything
  `create()` built (pipelines, bind groups, persistent KV/activation
  buffers, the resident `matvecQ8` weight buffers). `technologies-moe/alibi-ai`'s
  chat integration measured what rebuilding cost before this: 17-33s per
  independent turn, dominated by `create()`'s own weight re-upload — a cost
  a chat's "many short, independent generations against an unchanging model"
  workload never needed to pay more than once. A within-repo (source-only)
  API, same status as the rest of `llm/` (issue #98's own precedent) — not a
  blocker for `alibi-ai`, which already imports `llm/` from source rather
  than through this package's published `exports`.

  **Contract change:** the CPU-side quantized weights (`LlamaWeightsQ8`,
  ~1.4 GiB) are now kept for the engine instance's whole lifetime instead of
  being dropped after the first prefill — `reset()` needs them again for
  every later generation's dequant-transpose weight prep. Costs nothing
  *additional* for every real caller in this repository, which already keeps
  its own reference to the same weights object for the whole page session
  regardless (one extra pointer, not another 1.4 GiB) — but a caller that
  used to rely on this class releasing that memory after its first `forward()`
  call will now see it stay live for as long as the instance does; no
  `releaseWeights()` escape hatch exists yet (see README for the full
  reasoning and the fallback: a single-generation engine, discarded and
  rebuilt, is unaffected).

  Old KV cache contents are left in place (not cleared) rather than needing
  a separate wipe step; correctness relies on `sEff = position + 1` (issue
  #117) never scanning past what the current generation itself has written,
  proved by `llm/engine-q8-resident.reset.wgsl.test.ts` poisoning the entire
  old KV cache with `+Infinity` before `reset()` and confirming the next
  generation's output is unaffected — including a positive control proving
  the poison is actually observable before trusting the parity check, and a
  shorter-second-generation variant that exercises `runDecodeStep`'s own
  `sEff` bound, not only prefill's. `reset()` is also guarded against racing
  a still-in-flight `forward()` call on the same instance (a `generationEpoch`
  counter; calling `reset()` before every outstanding `forward()` promise has
  settled is not supported and now throws instead of silently rewinding).

  Measured on real hardware (RTX 5090, Chrome, real checkpoint, three
  independent generations, order counterbalanced across trials per PR #126
  review): "build once + `reset()`" beat "build fresh each time" by 34-49%
  across four trials (avg 42.5%, ~4.7s saved per three generations) —
  reproducible and not an ordering artifact (README's "reset(): multi-generation
  reuse without rebuilding" section has the full numbers, the counterbalancing
  design, and this measurement's own limits, including real concurrent
  system load during the run).

- `ops/gqa`'s `scores`/`context` kernels gain an `sEff` parameter (issue
  #117): the number of key/value positions actually resident, bounding the
  softmax scan without changing `S`, the KV cache's own address stride.
  Uniform-passed, defaults to `S` — every existing caller (including
  `ops/gqa`'s own reference and `llm/kernels.ts#runGqa`) is unaffected
  unless it opts in. `LlamaEngineQ8Resident`'s decode step now passes
  `sEff = position + 1` instead of always scanning `maxSeqLen` — the fix
  for the "gqaの全長走査" roofline gap issue #116 measured (672 MiB/token on
  Sarashina2.2-1B's shape) and explicitly left open, since removing it
  needed a parameter `ops/gqa` did not have yet.

  `context.wgsl`'s `v` read is unconditional (masked columns arrive as a
  plain `0` from `scores.wgsl`'s softmax, not a skip), so an `sEff` that
  is not honored there is numerically observable — a dedicated test poisons
  `v` past `sEff` with `+Infinity` and checks for the resulting `NaN`.
  `scores.wgsl`'s own bound has no such numeric signature under the
  `causal = true` contract `sEff < S` requires (every excluded position is
  already causally masked, by construction), so it is instead proven by
  `seffEquivalence()` (`ops/gqa/wgsl-seff.test.ts`): running with `sEff = n` must equal running with `S`
  itself shrunk to `n`. See `ops/gqa/reference.ts`'s `sEff` doc for the full
  safety contract (`sEff >= min(S, L + queryOffset)` whenever `causal` is
  true; rejected outright otherwise).

- `LlamaEngineQ8Resident`'s prefill is GPU-resident now too (issue #117),
  replacing the delegation to `LlamaEngineQ8` issue #110 left in place
  ("プリフィルは現行方式のまま"). Every prompt token's pass through every
  layer is encoded into one `queue.submit`, on the `matmul` path (not
  `matvecQ8` per token — that would multiply prefill's weight traffic by
  the prompt length). A new `ops/permute/wgsl/kernel.wgsl` kernel does `ops/gqa`'s
  required token-major/head-major reshape on the GPU, replacing the CPU
  round trip `LlamaEngineQ8`'s non-resident reshape (`llm/reshape.ts`)
  would otherwise force back in. KV writes go straight into the persistent,
  `maxSeqLen`-strided cache via one contiguous `copyBufferToBuffer` per
  head, so `runDecodeStep` continues seamlessly with no CPU round trip
  between prefill and the first decode step. Only the final prompt
  position is projected through `lm_head` — **`forward()`'s prefill return
  shape changed**: `[finalPositionLogits]`, one element, not one per prompt
  token, matching what every real caller (`llm/engine.ts#greedyGenerate`,
  `examples/llm-demo`) already read (`prefillLogits[prefillLogits.length - 1]`).
  See the PR for prefill tok/s at three prompt lengths, before/after, and the
  updated roofline table.

- Persistent IndexedDB weight cache for `llm/browser-weights.ts#loadWeightsQ8FromUrl`
  (issue #121, parent #96): `technologies-moe/alibi-ai`'s browser integration was
  re-fetching the full ~1.41 GiB int8 checkpoint over HTTP on every visit — no
  caching layer existed between `fetch` and the page. Caching is on by default and
  falls back transparently (no feature loss, only a persistence difference) when
  IndexedDB is unavailable (Node, a browser with none at all) or `indexedDB.open()`
  itself fails (Safari private mode's own failure shape) or when
  `navigator.storage.estimate()` reports too little free space for the checkpoint —
  pass `{ enabled: false }` as `loadWeightsQ8FromUrl`'s new fifth argument to opt out
  entirely.

  Versioning is a SHA-256 hash of `manifest.json`'s raw bytes (`llm/weight-cache.ts#sha256Hex`,
  via WebCrypto — available identically in Node and every browser this package
  targets), embedded in every cache key: a re-converted checkpoint under the same URL
  is detected and re-downloaded automatically, and the previous version's chunks are
  swept from the store once the new one is fully written (`sweepOrphanedChunks`,
  which doubles as a general orphan-chunk cleanup for a write interrupted mid-way, via
  one `ChunkStore.list()` pass rather than exact bookkeeping of what to delete). The
  manifest itself (tens of KB) is still fetched over the network on every load — it is
  the only way to learn whether a cached checkpoint is still current — but the three
  large binaries it may cache (`weights.codes.bin` / `.scales.bin` / `.norms.bin`,
  1.41 GiB combined) are not, on a cache hit; see the README's real-hardware section
  for the DevTools Network-panel proof.

  Each of the three cached files is split into 96 MiB chunks
  (`weight-cache.ts#DEFAULT_CHUNK_SIZE_BYTES`, the midpoint of this issue's own
  64–128 MiB range) before being written — a single ~1.4 GiB `IndexedDB` value is what
  this issue's own spec asks to avoid (browser blob-size implementation differences,
  and no way to show incremental write progress) — written and read one chunk at a
  time (`llm/weight-cache.ts#iterateChunks`, a generator) rather than materializing
  every chunk of a file up front, so a ~1.4 GiB read or write peaks at roughly the
  file's own size plus one chunk, not roughly double it. Storage is behind a small
  injected interface, `llm/chunk-store.ts#ChunkStore` (`get`/`put`/`delete`/`list`, all
  `ArrayBuffer`-valued), so every piece of cache logic — chunk splitting, manifest-hash
  comparison, stale-version eviction, the quota/fallback decision — is unit-tested in
  Node against `InMemoryChunkStore`, with no browser and no IndexedDB; only the real
  backend, `llm/idb-chunk-store.ts#createIndexedDbChunkStore` (hand-rolled minimal
  IndexedDB types, since neither this repository's `tsconfig.json` — deliberately
  `"DOM"`-lib-free — nor `@types/node` has any), needs the real-Chrome verification
  this issue's PR carries.

  The quota check (`decideCacheStrategy`) gates only the *write* path, never a read —
  a cache read frees no space and needs none, and gating it too would have made a
  device whose reported `usage` (which includes this very cache once one write
  succeeds) leaves too little headroom refetch the entire checkpoint on every visit
  from then on, forever, even though the cache it just refused to read was perfectly
  valid. A read failure (an IndexedDB `get()` rejecting — storage pressure closing the
  connection, a stale database from an older store-name layout) is caught and treated
  as a cache miss, the same fallback every other failure mode here already gets, rather
  than failing the whole load. `IndexedDbChunkStore#put`/`#delete` resolve on their
  *transaction*'s `oncomplete` (reject on `onabort`/`onerror`), not on the individual
  request's own `onsuccess` — a large write's request can report success and still have
  its transaction abort during commit (`QuotaExceededError`'s usual shape), and
  resolving on the request alone let a caller believe a chunk was durably written when
  it was not. The orphan sweep (`sweepOrphanedChunks`) now matches exact
  `namespace`/`manifestHash`/`file`/index chunk keys against a version record's own
  `chunkCount`, rather than a `namespace`/`manifestHash`/`file` *prefix* — re-writing
  the same file under the same hash with a different chunk size (a changed
  `DEFAULT_CHUNK_SIZE_BYTES`, say) used to leave every chunk beyond the new, smaller
  `chunkCount` behind forever, since they shared the kept prefix — and it now also runs
  (best-effort) after a cache *hit*, not only at the end of a successful cache-miss
  write, so chunks orphaned by an interrupted write are reclaimed the next time this
  cache is read rather than sitting unclaimed for as long as every later visit keeps
  hitting.

  Known limitation, tracked as [#124](https://github.com/m96-chan/web-xpu-ops/issues/124):
  versioning one `manifestHash` across all three binary files means a hypothetical
  future producer of this checkpoint format that keeps `manifest.json` byte-identical
  while republishing one of the binaries (this repository's own `convert_weights.py`
  does not — it regenerates `manifest.json`, and so the hash, on every run) could have
  an interrupted overwrite leave old- and new-version chunks mixed under one hash that
  this cache's own length check cannot tell apart from a correctly written file. Also
  tracked, lower priority since it cannot corrupt data either way (only causes
  redundant re-downloads and a non-deterministic "winner"):
  [#125](https://github.com/m96-chan/web-xpu-ops/issues/125), concurrent tabs open
  across a deploy window each detecting the other's freshly-written version as stale.

- `LlamaEngineQ8Resident` (`llm/engine-q8-resident.ts`, issue #110): a
  GPU-resident decode path for `LlamaEngineQ8` — one `queue.submit` and one
  logits-only readback per generated token, instead of the ~155 GPU↔CPU
  round trips (one per kernel dispatch, per layer) `LlamaEngineQ8.forward`
  pays today. Real-model measurement (`examples/llm-demo`, RTX 5090,
  Sarashina2.2-1B-alibi-v1): decode went from **0.75–0.80 tok/s to
  137–162 tok/s** (~175–200x), with byte-for-byte identical generated text
  to `LlamaEngineQ8` on both prompts tested — see the PR for the full
  before/after table, roofline decomposition and screenshots.

  `harness/resident.ts` (`ResidentDevice`) is the new low-level layer this
  runs on: build every buffer, pipeline and bind group once at construction,
  then record many dispatches (plus GPU-to-GPU `copyBufferToBuffer` for the
  KV-cache writes) into one command encoder per token. `harness/wgsl.ts`'s
  existing `Runner`/`createRunner()` is unchanged and still what every op's
  own correctness test uses (rule 8: optimize only after a reference-correct
  baseline exists) — `ResidentDevice` exists solely for this decode path,
  where the WGSL, binding order and uniform layout are already known correct
  from `llm/kernels.ts`. `harness/resident.ts#runnerFromResident` adapts one
  `ResidentDevice` back into a `Runner["run"]` — at the time this landed,
  `LlamaEngineQ8Resident` used it to delegate prefill to `LlamaEngineQ8`
  unchanged (issue #110's scope was decode only) instead of constructing a
  second native device: an earlier version took a genuinely separate
  `Runner`, and running both engines' dispatches on two devices in one
  process reproducibly crashed this repository's Node/Dawn binding (the
  same failure family issue #38/#49/#107 already document). Issue #117
  later made prefill resident too and removed that delegation entirely —
  see this file's #117 entry above; `runnerFromResident` itself remains,
  unused by this class since.

  `examples/llm-demo` gained a "GPU常駐デコード" toggle so the same page can
  run either engine against the same loaded weights for a direct comparison.

  PR #116 review: `LlamaEngineQ8Resident.forward` now throws once a decode
  step would exceed `maxSeqLen` (`LlamaEngineQ8`/`LlamaEngine`'s own decode
  paths already do), instead of a KV-cache GPU-to-GPU copy silently landing
  past the buffer it was meant for and the call resolving with the previous
  step's stale logits.

- `examples/llm-demo/`: a browser demo that loads Sarashina2.2-1B-alibi-v1
  (int8, `convert_weights.py`'s output) over HTTP and runs `LlamaEngineQ8` on
  a real WebGPU device — issue #106, and the first time this repository's
  `llm/` engine has generated anything in a browser rather than under Node.
  It closes the live-generation / `llama.cpp`-comparison / tok/s gate PR
  #108 (issue #105) moved here: #108 could not complete a live GPU dispatch
  on the machine it was built on (a Node+Dawn binding fragility triggered by
  real-model-scale CPU-bound work immediately before a dispatch — issue
  #107), and a browser's separate GPU process does not share that condition.
  `llm/index.ts` also now re-exports `llm/tokenizer.ts` (#101), `llm/sampler.ts`
  and `llm/constraints/line-format.ts` (#102) — a known gap since both landed,
  called out in #106's own text, and caught by the new `llm/index.test.ts`
  before being fixed.

  `llm/browser-weights.ts#loadWeightsQ8FromUrl` is a `fetch`-based sibling of
  `real-model-weights.ts#loadConvertedWeightsQ8`, sharing the same
  manifest-parsing core (`weights-q8-io.ts#buildLlamaWeightsQ8`) per PR #108's
  own note that a browser loader "should be a small adapter over the same
  parsing logic, not a reimplementation" — it lives in `llm/` (not
  `examples/`) and is exported from `llm/index.ts` because `fetch`/`Response`
  are standard Web APIs this package already assumes elsewhere (`llm/tokenizer.ts`'s
  `TextEncoder`/`TextDecoder`), unlike the `node:fs` the disk-based loader
  needs, and because that keeps it testable with a mocked `fetch` and no
  browser.

  The demo's own WebGPU plumbing (`examples/llm-demo/src/browser-runtime.ts`)
  is a `navigator.gpu` port of `harness/wgsl.ts#createRunner` — necessarily a
  separate implementation, not a shared import, since `harness/wgsl.ts`
  imports the `webgpu` package (a Node-native Dawn binding with no browser
  build) at module scope. `llm/kernels.ts` itself needed **zero** changes to
  run in a browser: `examples/llm-demo/build.mjs` (esbuild — the first
  bundler this repository has needed, justified in that file's own module
  doc) redirects `kernels.ts`'s `import { kernel, params } from
  "../harness/index.js"` to `browser-runtime.ts`'s matching-signature
  implementations at bundle time, and inlines all ten `.wgsl` kernel sources
  `kernels.ts` reads via `node:fs` under Node into plain JS strings via
  esbuild's `text` loader — `llm/kernels.browser-parity.test.ts` (text-based,
  no `.wgsl` import, runs under `npm test`) guards the two kernel tables
  against drifting apart. `examples/llm-demo/server.mjs` is a Node-standard-
  library-only static file server (repository root + a `/weights/` mount
  pointing at a converted checkpoint directory outside this repository,
  e.g. `technologies-moe/alibi-ai`'s `third_party/webgpu-weights/`), always
  setting `Content-Length` (no `Range` support needed) since the demo's
  progress bar reads it.

- `llm/tools/convert_weights.py`, `llm/weights-q8.ts`, `llm/engine-q8.ts`
  (`LlamaEngineQ8`): a weight converter and an int8 (W8A32) engine path,
  closing #96's loop from "an engine that runs a tiny fixture" to "an engine
  that loads and generates from a real checkpoint" — issue #105.
  `convert_weights.py` streams a HF safetensors checkpoint (bf16) one tensor
  at a time into per-row absmax int8 codes + f32 scale (`wq`/`wk` permuted
  per `permuteRopeChannels` before quantizing) plus f32 norms; converting
  Sarashina2.2-1B-alibi-v1 (2.68 GiB bf16) produced a 1.41 GiB (1345 MiB)
  int8 checkpoint in ~8s. `LlamaEngineQ8` mirrors `LlamaEngine`'s forward pass
  exactly but keeps only the packed `matvecQ8` wire format resident per
  projection (not also an unpacked or dequantized copy — packing does not
  change a weight's resident size, so keeping both would roughly double it):
  decode dispatches `matvecQ8` directly; prefill dequantizes the needed
  projection to a transient f32 matrix and runs `matmul`, once per generation
  (the prompt), not once per token — issue #105's own stated scope
  ("プリフィルは当面「行スケールdequantしてf32 matmul」でもよい"). The
  embedding table is quantized like every other weight but is gathered and
  dequantized **on the CPU**, one row per requested token, rather than
  dispatching a GPU gather against a fully-dequantized 733 MiB table.
  Verified on a real GPU against an int8-quantization-aware `transformers`
  reference (`llm/tools/gen_fixture_q8.py`, which substitutes dequantized
  weights into the tiny fixture model and re-runs its own forward pass, so
  the quantization error is baked into the reference the same way it is
  baked into `LlamaEngineQ8`'s output — worst observed diff abs 1.49e-7, rel
  7.22e-4). The real Sarashina2.2-1B-alibi-v1 checkpoint converts, loads, and
  constructs a `LlamaEngineQ8` successfully; **live GPU generation from it
  does not run yet** — a Node+Dawn binding fragility triggered by
  real-model-scale CPU-bound work before a dispatch (see #107 for the
  isolated repro), and, found in review, the real vocab's lmHead projection
  breaking WebGPU's default dispatch and buffer limits (#112). Neither is a
  numerics problem. Live generation, `llama.cpp` comparison, and tok/s move
  to #106 (browser demo), where #107's condition does not exist — #112
  applies there too and blocks it.
- `matvecQ8` gets a second call site: `llm/kernels.ts#runMatVecQ8`, the GPU
  dispatch wrapper `LlamaEngineQ8`'s decode path uses — issue #97's kernel
  finally consumed by the engine it was built for.
- `llm/weights-q8.ts`, `llm/weights-q8-io.ts`, `llm/reshape.ts#concatRowsInt8`:
  the int8 counterparts of `weights.ts` (`QuantizedLinear`,
  `LlamaWeightsQ8`, `assertWeightShapesQ8`) and `reshape.ts#concatRows`,
  plus a manifest-parsing core (`buildLlamaWeightsQ8`) shared by
  `fixture-q8.ts` (the tiny int8 fixture) and `real-model-weights.ts` (a
  real converted checkpoint) — the two write the identical manifest shape
  (`convert_weights.py`'s own module doc) on purpose, so one parser serves
  both rather than two that could drift apart. `cloneQuantizedLinear` copies
  the embedding table out of the loader's shared buffer at construction time
  rather than keeping a view into it — measured to matter: a `LlamaEngineQ8`
  built from the real checkpoint fell from ~2.96 GiB to ~1.56 GiB resident
  (RSS) once the caller also drops its own reference to the loaded weights
  and a GC runs, because a `TypedArray` view keeps its *entire* backing
  buffer alive, not just the slice it reads.
- `llm/tools/quant_common.py`: the per-row absmax int8 quantizer
  `convert_weights.py` and `gen_fixture_q8.py` both call, sharing one
  implementation rather than risking two silently-different ones (rule 7).
  Uses `floor(x) + (frac >= 0.5)`, not `np.round` and not `np.floor(x + 0.5)`
  — `Math.round` (what `ops/quantize/reference.ts#quantize` uses) rounds ties
  toward `+Infinity`; `np.round` rounds ties to even and disagrees at every
  exact `.5` boundary (`Math.round(62.5) === 63`, `np.round(62.5) === 62`);
  `np.floor(x + 0.5)` disagrees in the narrow band just below half-integers,
  where the addition itself rounds up before `floor` runs
  (`Math.round(0.49999999999999994) === 0`, `floor(… + 0.5) === 1`). Verified
  rather than assumed by `llm/quantize-parity.test.ts`, which spawns
  `quant_common.py --selftest` and diffs its output against `quantize()`
  directly, including rows at both boundaries.
- `llm/tokenizer.ts`: a SentencePiece **unigram** tokenizer (Viterbi encode,
  matching decode) for the Sarashina2.2-1B-Instruct model the LLM engine
  (#98) targets, plus `llm/tools/export_tokenizer.py` (parses
  `tokenizer.model`'s protobuf into a JSON vocab) and
  `llm/tools/gen_fixtures.py` (bakes real-tokenizer encode/decode fixtures
  from Python `sentencepiece`, never `transformers` — see below). No wasm,
  no vendored binary: this is a from-scratch TS reimplementation of the
  algorithm, CPU-only, and it never touches a GPU device.

  Nothing about this model's tokenizer config was assumed. Read directly out
  of the real `tokenizer.model`/`tokenizer_config.json`:

  - The normalizer is literally `"identity"` — **not** NFKC. Composed and
    decomposed forms of the same glyph (が vs か + combining voiced sound
    mark) tokenize to different ids; a normalizing tokenizer would not do
    that. Only the literal ASCII space gets escaped to `▁`; tabs, newlines,
    NBSP and the full-width space do not.
  - `add_dummy_prefix` is `false`: no implicit leading `▁`.
  - Control/special tokens (`<s>`, `</s>`, `<|system|>`, ...) are **not**
    matched by SentencePiece's own Viterbi search against raw text — verified
    directly (`sp.encode("<|system|>")` decomposes it byte-by-byte). They are
    recognized through a separate literal-substring pre-tokenization pass
    over `tokenizer_config.json`'s `added_tokens_decoder`, which is what
    lets the engine's chat-template fragments (`<|system|>`, `<|user|>`,
    `</s>`, ...) round-trip as single ids.
  - `transformers` 5.3.0, on this environment, silently converts this
    UNIGRAM model into an *approximate BPE* tokenizer when loaded with
    `AutoTokenizer.from_pretrained(..., use_fast=False)` and no
    `tokenizer.json` is present (e.g. it segments `"hello"` as
    `['he','ll','o']`, not the single correct unigram piece `'hello'`).
    Fixtures are therefore generated from Python `sentencepiece`'s
    `SentencePieceProcessor` directly, never from `transformers`.
  - Byte fallback is one Viterbi edge per uncovered *codepoint*, scored
    `min_score() - 10.0` (SentencePiece's own `kUnkPenalty`/`min_score()`
    formula from `unigram_model.cc`) — not per byte, and not an arbitrarily
    large sentinel. An earlier version used `-1e6` per byte, which is locally
    harmless (a byte edge never competes with a real piece at the same
    position) but not globally harmless: accumulated over a few dozen
    uncovered codepoints in one string it pushed the running score to a
    magnitude where float32 could no longer distinguish the much smaller
    (~1-2 point) gaps between competing *real* segmentations later in the
    same string, silently picking the wrong one. Caught by a fixture built
    from a Hiragana-block sweep containing unassigned codepoints; kept as a
    permanent regression case (`byte_fallback_dense_sweep`).
  - The Viterbi DP accumulates scores in float32, matching SentencePiece's
    own C++ `Lattice`, and rounds every candidate to float32 *before*
    comparing it against an already-stored (also float32) score — comparing
    an un-rounded float64 sum against an already-rounded value can flip an
    exact tie either way depending on which side happened to round which
    way. This is not cosmetic: for a run of one repeated character, more
    than one tiling by the same multiset of piece lengths can be an exact
    tie at the level of real numbers, and only float32 rounding — applied
    consistently on both sides of the comparison — reproduces which one the
    real tokenizer picks.

  80+ fixture cases (Japanese/English/mixed, code fragments, emoji including
  ZWJ/flag/skin-tone sequences, whitespace and newline patterns, NFC/NFD
  probes, chat-template-shaped special-token boundaries, and the
  byte-fallback stress case above) all match the real tokenizer exactly on
  both encode and decode. Vocab JSON: 102400 entries, 3.48 MiB raw / 1.33 MiB
  gzipped — not compressed further; see the PR for why.
- `llm/sampler.ts`: token sampling with `greedy` and `temperature` + `top-p`
  modes, plus a token-level `Constraint` interface (`nextAllowed(prefixTokens)
  -> allowed token ids, or null`) applied to the logits before either mode
  runs. Repetition penalty is **deliberately not implemented** — the
  consuming project measured it degrading Japanese output rather than
  de-looping it (technologies-moe/alibi-ai#3); see `sampler.ts`'s module
  comment for the reasoning. A caller that wants repetition control should
  express it as a `Constraint` instead.

- `llm/constraints/line-format.ts`: a `Constraint` implementation for a fixed
  line shape — literal text, an enum choice, more literal text, free text
  (forbidden characters, max length), then EOS. Built for a small,
  hand-written schema like `policy: <enum>\ntopic: <short text>`, where a
  full GBNF/grammar engine would be overkill. Tokenizer-agnostic: it takes an
  injected `TokenCodec` (`encode` / `idToToken` / `vocabSize`) rather than
  depending on any one tokenizer, since issue #101's tokenizer is a separate,
  not-yet-merged branch. Enum choices are matched by tokenizing each
  candidate once and walking a token-id trie, so they must be
  **token-prefix-free** (no choice's tokenization may be a strict prefix of
  another's) — an ambiguous spec is rejected at construction rather than
  silently misclassified at generation time.

  Neither module is wired into `llm/engine.ts` yet — that integration is
  left to a follow-up issue, matching #102's stated scope.
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
- `matvecQ8`, a W8A32 GEMV in `ops/matvec`: the weight held as int8 instead of
  f32, packed four codes per `u32` word. Decode-time GEMV is bandwidth-bound,
  and an int8 weight is a quarter the traffic of f32 for the same values — the
  packed layout halves that again versus one code per lane, since the whole
  point of quantizing a weight nobody re-quantizes at inference is to spend
  the packing cost once rather than never taking it (#97). `packQ8` packs
  `quantize`'s existing per-row absmax codes into the layout `matvecQ8` reads,
  so the two compose instead of `matvecQ8` inventing its own quantization.

### Fixed

- `llm/kernels.ts#runMatVec`/`runMatVecQ8` now split a dispatch whose row
  count `M` exceeds WebGPU's `maxComputeWorkgroupsPerDimension` (`65535` on
  every implementation measured against this repo) into several dispatches
  and concatenate the results, instead of asking for `workgroups: [M]`
  directly. Found running Sarashina2.2-1B-alibi-v1 in a browser for the
  first time (issue #106): `lm_head`'s decode-time projection
  (`vocabSize=102400`) silently produced an all-zero logits vector on every
  decode step — a WebGPU validation error past this limit is reported
  asynchronously through the device's error callback, not by throwing where
  `dispatchWorkgroups` is called, so nothing in the existing code path ever
  saw it fail. The visible symptom was the model emitting one correct
  character from prefill (which projects `lm_head` through the tiled
  `matmul` path, unaffected) and then the same wrong token — id 0 — forever,
  since argmax over an all-zero vector always picks index 0. Verified
  end-to-end by re-running the same generation in the browser after the fix
  and getting real, `llama.cpp`-comparable output (see #106's PR); the
  chunking logic itself is proven against a mocked `Runner["run"]`
  (`llm/kernels.chunking.test.ts`) rather than on real hardware at this
  scale, because a real dispatch at 65,535 workgroups reproducibly crashed
  this repository's own Node/Dawn binding (issue #38/#49/#107's family) —
  exactly the fragility issue #106 exists to route real-model verification
  around in the first place.

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

[Unreleased]: https://github.com/m96-chan/web-xpu-ops/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/m96-chan/web-xpu-ops/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/m96-chan/web-xpu-ops/releases/tag/v0.1.0
