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

against the `t2va` partition's 0.84% and 2.44% on the same geometry. The
fifty-block check is **not run yet**: it needs 20.66 GB and the card had 27.1 GB
in use.

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

## What is not here yet

The transformer conversion, the Qwen3-VL text stack and vision tower, the
`Qwen3VLProcessor`, and the page. See #212.
