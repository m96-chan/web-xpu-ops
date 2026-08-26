# R2V — MiniMax-H3 conditioned on references

Put images and video in, get video out: MiniMax-H3's `ref2va` workflow. Issue
[#212](https://github.com/m96-chan/web-xpu-ops/issues/212).

`examples/h3-dit` is the text-to-video half and is measured; this is the
reference-conditioned one, and it is **being built**. What is here so far is the
packed layout.

## Why R2V is not `t2va` with extra rows

**A different transformer — but not a different shape.** `transformer_ref/` is
66.28 GB and is not `transformer/`: `MiniMaxH3ModularPipeline.patch_size` reads
whichever of the two a workflow loaded, because a workflow loads only its own.

The two configs were compared field by field and are **identical**. The weights
are not: measured on three tensors spread across the stack, the mean absolute
difference is **2.0–2.2% of the mean magnitude** — a fine-tune of the same base,
not a copy and not a different architecture.

Which means `examples/h3-dit`'s converter and `model-gpu.ts` serve it
**unchanged**. Converted with the same script — 20.08 GB resident, 0.58 GB of
tables, 96 s — and held to the model's own output at four blocks:

| | worst | of peak |
| --- | --- | --- |
| video velocity | 1.132e+0 | **0.96%** |
| audio velocity | 9.867e-1 | 2.72% |

and at **all fifty**, once the card was free:

| | worst | of peak |
| --- | --- | --- |
| video velocity | 7.605e-1 | **12.28%** |
| audio velocity | 4.499e-1 | 17.18% |

against the `t2va` partition's 0.84%/2.44% at four blocks and 13.80%/27.67% at
fifty. The same story, and the same cause: int8 compounding over fifty blocks
on an output that is a small difference of large intermediates. See
`examples/h3-dit`'s README for the three ways that was measured.

One step over 42 rows takes **2,175 ms** — 2,063 dispatches, 6 ms of it in the
queue and **1,642 ms recording on the host**, which is where every model here
spends its time.

**It cannot precompute the conditioner.** `examples/h3-dit-web` ships prompt
embeddings baked offline, because the prompt list is fixed. Here **the reference
is the input**, so Qwen3-VL has to be resident. Measured, at int8: vision tower
**0.60 GB**, text layers 0..50 **24.87 GB**, embedding 0.78 GB — **26.25 GB**,
on top of a 20 GB DiT and a 2.43 GB decoder. That does not fit at once on a
32 GB card, and the answer is the staging `examples/h3-dit/src/generate.ts`
already uses: upload, use, drop, upload the next.

**It has a caller for the VAE encoder.** `examples/h3-encoder` was written for
#209 and nothing called it. References are encoded with it.

## The layout, and the six things it decides

`[text | reference blocks | target audio | target video]`. What makes it more
than an ordering is that **the references advance a shared rotary clock**: where
the generated video sits in rotary time depends on how many references came
before it and how long each was. Held to
`MiniMaxH3Ref2VAPrepareLayoutStep.build_ref2va_packed_sequence`, exactly, on a
committed fixture.

| | |
| --- | --- |
| **An image takes exactly one rotary slot** | not a latent frame's `5/3` |
| **A video's soundtrack is packed *before* its video rows** | sharing their origin |
| A video advances the clock by `max(audioLatents, videoSpan)` | whichever is longer |
| That span is summed **sequentially** | the `t2va` keyframe anchor sums the same series *pairwise*, and they differ in the last ulp from 16 latent frames on |
| A **standalone** soundtrack is pinned to the *target's* width grid | a video's soundtrack to *its own* |
| **Text rows carry the caller's tags** | a reference's vision block sits among them and is tagged **video** |

### Ten mutations, all caught — after the fixture was widened

The first sweep caught **seven of ten**, and the three survivors were all the
fixture's fault rather than the test's:

- Advancing the clock by the video span alone survived because **no case had a
  soundtrack longer than its own video**, so the `max` never chose the audio.
- Pinning a video's soundtrack to the *target's* width grid survived because
  **the reference and the target were on the same canvas** in every case.
- Assigning the video tags before the audio ones survived, and still would:
  the three index sets are disjoint by construction, so **the order cannot
  matter**. Recorded rather than tested.

Three cases were added for the first two, and the third mutation was replaced by
one that can be observed — ignoring the caller's text tags. Which found a real
gap: this port had been filling in `TEXT_TAG` for every text row, and
`text_token_tags` is an argument upstream takes precisely because a reference's
vision block is not text.

## The presentation, and the timestamp that is not a tie

Before a reference's pixels reach anything, the reference is *announced*:
`"<Picture i>: "`, `"<Audio j>: "`, `"<Video k>: "`, numbered **per modality**,
each followed by a vision block of pad tokens. That announcement is what
produces the text tags the layout takes. `src/presentation.ts`, held to
`MiniMaxH3Ref2VATextEncoderStep`.

The fixture records the token ids **and a map from each text segment to its
ids**, so the assembly is held to upstream without a BPE implementation in the
way. The tokenizer is a separate concern with its own golden to come.

**A video that carries sound is labelled `<Audio j>` before `<Video k>`**,
mirroring the order its rows are packed in. A video gets **one timestamped
vision block per merged frame group**, not one block.

### `"{:.1f}"` is round-half-to-even, and the tie is not where it looks

The mean of a 2 fps pair is exactly `0.25`, which Python renders `"0.2"` and
`(0.25).toFixed(1)` renders `"0.3"` — a different token, a different embedding,
no other symptom.

Detecting the tie by multiplying by ten is **also wrong**, and that is the more
interesting half. `0.15` is really `0.1499999999999999944…`, so Python renders
`"0.1"` — but `0.15 * 10` rounds to exactly `1.5`, which reads as a tie and
gives `"0.2"`. This port did that until the fixture grew a case where a
reference read at 30 fps put a `0.15` in front of it. The tie is detected on
`toFixed(20)` now, the exact decimal expansion.

### Ten mutations, all caught — after two fixture gaps and a red baseline

The sweep first reported **7 of 10**, and again the survivors were the fixture:
every image came before every video, so a counter that summed the modalities
agreed everywhere; and every stride was 12, so the frame deduplication never
had a duplicate to remove.

Then it reported **10 of 10 while two tests were already failing** — because a
sweep that only asks "did the mutant fail?" calls every mutation caught when
the baseline fails too. Both sweeps refuse to start on a red baseline now. The
two failures were a wrong assertion and the `0.15` bug above.

## The conditioner, read rather than guessed

The biggest open question in #212 was whether the vision tower needs a new
kernel. It was answered by reading `transformers`' `modeling_qwen3_vl.py` and
the checkpoint's own parameter shapes. **It does not.**

### Qwen3-VL's text stack — 51 of 64 layers

hidden 5120, head_dim 128, **64 query heads against 8 key/value heads**,
intermediate 25600, RMSNorm at 1e-6, `rope_theta` **5,000,000**, no attention
bias.

| | this repository |
| --- | --- |
| `input_layernorm`, `post_attention_layernorm` | `ops/rmsnorm` |
| `q_proj` / `k_proj` / `v_proj` / `o_proj` | `ops/matmul` |
| 64 query heads over 8 KV heads | **`ops/gqa`**, which is what it is for |
| `q_norm` / `k_norm`, `[128]` per head | `ops/rmsnorm` with groups — `examples/h3-dit`'s block already does exactly this |
| `mlp` gate/up/down, silu | `ops/activation` + `ops/elementwise` |
| M-RoPE, `mrope_section [24, 20, 20]` | **`ops/rope`'s `ropeAxes`** — three axes of 48/40/40 channels, which is the signature that op has |

The one thing to read carefully is `mrope_interleaved: true`: the three sections
are interleaved rather than concatenated, which is the same *kind* of thing as
H3's own rope permutation and is solved the same way — fold a channel
permutation into the Q and K weights at conversion. `h3RopePermutation`'s doc is
about precisely that, and #208 records what it costs to permute the projection
and forget the per-channel norm weights.

### The vision tower — 27 blocks, 0.60 GB

hidden 1152, 16 heads, patch 16, temporal patch 2, spatial merge 2, out 5120,
deepstack taps at layers 8, 16 and 24.

| | this repository |
| --- | --- |
| `patch_embed.proj`, a `Conv3d` with kernel and stride `(2, 16, 16)` | **`ops/conv`'s `conv3d`** — written for #201, and this is a second caller |
| `norm1` / `norm2`, **with bias** | `ops/layernorm`, not `rmsnorm` |
| `attn.qkv`, fused `[3456, 1152]` | split at conversion, as `examples/h3-video` splits `to_qkv` |
| `mlp`, `gelu_pytorch_tanh` | `ops/activation`'s `gelu_tanh` |
| `merger`, **exact** GELU | `ops/activation`'s `gelu` — a different entry, and the two are not the same function |
| vision rope, two axes, `rotate_half`, `cat(freqs, freqs)` | `ops/rope`'s `ropeAxes` with `[36, 36]` and the same permutation |

**One piece is genuinely new**, and it is not a kernel:
`fast_pos_embed_interpolate` bilinearly resamples a learned `[2304, 1152]`
table onto each image's grid. `ops/upsample` is nearest-neighbour only — but the
table is 2.6 M values and the resampling is per image on the host, so it does
not need to be one.

## The conditioner's text stack

`src/text-encoder.ts`, held to `transformers`' `Qwen3VLTextModel` at
**3.576e-7** across every hidden state. The fixture is **committed**: random
weights at a tiny geometry — 122 KB, no model licence, runs in CI anywhere.

**No new kernel**, as the inventory above said. `ops/gqa` takes the 64-over-8
grouping, `ops/rmsnorm` takes both the layer norms and the per-head QK norms,
and `ops/rope`'s `ropeAxes` takes M-RoPE — but only via two things that belong
in the weights rather than in a kernel:

- **Sixty-four axes of two channels each.** At `dim = 2` the pair index is 0, so
  `theta ** 0` is exactly 1 and the angle is the position verbatim. The
  frequency is then folded into the position, which is what #206's fractional
  positions were added for.
- **A channel permutation**, because Qwen rotates `c` against `c + headDim / 2`
  and `ropeAxes` rotates adjacent pairs. Applied to Q, K **and** the per-head
  norm weights — #208 records the 8% cost of moving only the first two.

### The interleave, which is the part to read twice

`mrope_section` looks like it names three contiguous blocks. It does not.
`apply_interleaved_mrope` starts from an all-time array and overwrites two
**strided** slices, so channel `c` belongs to axis `c % 3` while
`c < 3 * section[1]`, and to time after that. At the released `[24, 20, 20]`
that is `0,1,2,0,1,2,…` for sixty channels and then four more of time.

The chunked reading — `[0..23] = t, [24..43] = h, [44..63] = w` — is what the
field name suggests, produces a working model, and is wrong. And every channel
keeps the frequency its **global** index gives it whichever axis it reads,
which is why a per-axis frequency sweep is also wrong.

### The last hidden state is not like the others

`output_hidden_states=True` returns `numHiddenLayers + 1` entries, and the last
is `last_hidden_state` — **with the final norm applied**. A port that appends
its raw last layer output matches every earlier entry and disagrees with the
golden by **0.53** on that one, which is what happened here.

It does not change what MiniMax-H3 reads: `hidden_states[50]` of a 64-layer
stack is an ordinary layer input. Which is precisely why the discrepancy could
have sat unnoticed in a port that only ever indexed 50.

Ten mutations, all caught.

## The vision tower

`src/vision.ts`, held to `transformers`' `Qwen3VLVisionModel`: **1.192e-7** on
the tower's own output, **7.451e-9** on the pooled vision tokens and on every
deepstack feature. Committed fixture, random weights, 399 KB.

**`ops/conv`'s `conv3d` is the patch embedding** — a `Conv3d` whose kernel
equals its stride and its input — so the op written for #201 has a second
caller, like the VAE encoder before it. `layernorm` (with bias, unlike
everything else in H3), `attention`, `matmul`, `activation` and `elementwise`
cover the rest, and `ropeAxes` takes the two-axis rotation the same way the text
stack takes M-RoPE.

Four things it decides:

- **Tokens are in merge-block order**, `(t, h/m, w/m, m, m)`. The position
  embedding is permuted into it, the rotary coordinates are generated in it, and
  a merger eats `m * m` consecutive tokens. Raster order is the same shape and a
  picture read wrong.
- **The position embedding is bilinearly interpolated** from a learned
  `48 x 48` table, with `torch.linspace` and an `int()` truncation choosing the
  four taps — #211's linspace again, because one ulp moves a truncation across
  an integer boundary.
- **Two different GELUs.** The blocks' MLP is `gelu_pytorch_tanh`; the mergers
  use the exact one.
- **Two different norm placements.** The final merger normalises *before* the
  `m * m` shuffle, on `hidden`; the deepstack mergers normalise *after*, on
  `hidden * m * m`. The only place that is visible is the checkpoint's own
  shapes.

### The sweep caught six of ten, and the tolerance was why

Not the fixture this time. The bounds were `2e-5` against an achieved error of
`1.192e-7` — **a hundred and seventy times too loose** — and two mutations
walked through it: both GELU swaps. `gelu` and `gelu_tanh` differ by
**6.932e-5** on the values this fixture produces, and a tolerance with no
measurement beside it cannot see that.

They are `3e-7` and `3e-8` now, which are the measured errors times a small
factor, with both the measurement and the mutation's effect written next to
them: swapping the blocks' MLP moves the tower's output to **6.109e-7**, which
is why the bound is not a round `1e-6`.

Two of the survivors were also **mutations that changed nothing** — moving an
index lookup, and an `x = x === before ? x : x`. A no-op that "survives" says
nothing, and both were replaced by ones that bite.

## The image processor

`src/processor.ts` reproduces `Qwen2VLImageProcessor` **exactly** — worst
difference **0**. `smartResize` with Python's banker's rounding and its **two
different clamp rules** (`floor` above the pixel ceiling, `ceil` below the
floor), the temporal repeat that turns one still image into two frames, and the
`(0, 3, 6, 4, 7, 2, 1, 5, 8)` patchify that lands tokens in the merge-block
order `vision.ts` reads.

Exact took matching **two f32 roundings**: upstream is `rescale` writing a
float32 and then `normalize` reading it, not one expression. Doing both in f64
and narrowing once is off by 5.9e-8 — small, and the difference between a check
that can assert equality and one that needs a tolerance nobody can justify.

**The resize is not ported.** Upstream resamples with PIL's bicubic; a browser
has `drawImage`. `patchify` refuses pixels that do not already conform rather
than cropping them quietly, and what a browser's resampler costs is
**unmeasured**.

Ten mutations, all caught — after a fixture gap: the only above-the-ceiling
case was 5000x5000, where `height / beta / factor` is exactly 128.0 and `floor`
and `ceil` agree. 3000x6000 gives 90.5, where they do not.

## The tokenizer was already here

`ref2va`'s presentation needs Qwen2's byte-level BPE, and `llm/tokenizer-bpe.ts`
is exactly that with its vocabulary committed. Whether it agrees with H3's own
tokenizer is measured, not assumed: `src/tokenizer.test.ts` runs **all fourteen**
text segments the presentation can produce and **all four** vision token ids.
They agree.

## The conditioner on the GPU — converted, running, and not yet right

`tools/convert_conditioner.py` writes **26.27 GB** of int8 in 96 s (vision tower
0.61 GB), and `src/conditioner-gpu.ts` runs it: 76 tokens in **2.4 s** over 828
dispatches, of which 15 ms is queue time and 2,140 ms is host recording.

**It does not yet reproduce the model.** Held to `hidden_states[50]` of the
released Qwen3-VL-32B on a real presentation, and bisected by layer:

| | worst, as a percentage of peak |
| --- | --- |
| `hidden_states[0]` — embedding, vision tokens scattered in | **4.07%** |
| `hidden_states[1]` — after one text layer | **19.34%** |
| `hidden_states[2]` | 15.03% |
| `hidden_states[4]` | 4.52% |
| `hidden_states[50]` | 100% |

**So the fault is in the first text layer**, and it washes down before
compounding again — the shape of an error that reads as quantisation noise if
only the last number is looked at. What is already ruled out:

- **The position grid.** It is rebuilt from the token stream and compared
  against `get_rope_index`'s own output on all three axes, exactly, before the
  forward runs. Fixing it changed the RMS by 0.0002.
- The layout of the flash-attention uniform, checked against
  `ops/flash_attention/wgsl/fa2.wgsl`'s own struct.
- The GQA grouping, which is `ops/gqa`'s contiguous `floor(h / 8)`.

Two bugs *were* found and fixed on the way, and both were real:

- **The deepstack add aliased one buffer as read and read-write in a compute
  pass.** WebGPU refuses that, so the whole command buffer was invalid and the
  output was whatever the pool held. It writes into a fresh buffer now.
- **The fused `qkv` was un-interleaved with a copy per token per block** —
  20,736 of them for one 256x256 reference, 28,957 dispatches in total. The
  converter splits it into three, as `examples/h3-video` splits `to_qkv`, and
  the count fell to 828.

### What the position grid turned out to be

Worth recording even though it was not the fault. **A vision token does not sit
at `t = h = w = index`.** A text run takes consecutive positions on all three
axes; a vision block gets a *2-D* grid — width cycling, height held for a row,
time constant — and the clock then advances by **`max(h, w) / merge`**, the
block's longer side, not by its token count. For a 256x256 reference that is 64
tokens advancing the clock by 8.

`qwen3vlPositionGrid` builds it, and `verify-conditioner.ts` refuses to run if
it disagrees with the model's own.

## What is not here yet

The bug above, and the three-stage run in `examples/h3-ref2v-web`. See #212.
