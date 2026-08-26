# R2V — MiniMax-H3 conditioned on references

Put images and video in, get video out: MiniMax-H3's `ref2va` workflow. Issue
[#212](https://github.com/m96-chan/web-xpu-ops/issues/212).

`examples/h3-dit` is the text-to-video half and is measured; this is the
reference-conditioned one, and it is **being built**. What is here so far is the
packed layout.

## Why R2V is not `t2va` with extra rows

**A different transformer.** `transformer_ref/` is 66.28 GB and is not
`transformer/` — `MiniMaxH3ModularPipeline.patch_size` reads whichever of the
two a workflow loaded, because a workflow loads only its own. The conversion
`examples/h3-dit` already has cannot serve this.

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

## What is not here yet

The transformer conversion, the Qwen3-VL text stack and vision tower, the
`Qwen3VLProcessor`, and the page. See #212.
