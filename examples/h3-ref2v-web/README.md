# R2V in a browser — references in, video out

MiniMax-H3's `ref2va`: drop images and video, get video. Issue
[#212](https://github.com/m96-chan/web-xpu-ops/issues/212).

## Status: the request is built, the models are not wired

**What works, and is held to the model at every step:** a dropped file becomes
patches, the presentation is assembled and tokenised, and the packed sequence is
built. Drop a reference and press Generate and the page tells you exactly what
it would run — how many vision tokens the reference contributed, how many rows
are references, how long the packed sequence is.

**What does not:** the page runs no model. The conditioner does have a GPU path
now (`examples/h3-ref2v/src/conditioner-gpu.ts`, held to the released
Qwen3-VL-32B at a median row of 1.12%) and so do the DiT and the VAE decoder —
what is missing is the **VAE encoder**, which has only a CPU reference, and the
staging between all of them.

The encoder is not a detail: a reference has to be encoded into latents to
become the packed sequence's reference rows, and the CPU one takes **120.5 s on
an 8x32x32 clip**. A 256x256 still is eight times the voxels. Issue #214.

So it is four models one at a time, not three, and one of the four does not
exist yet. That is the honest line. Nothing here pretends to generate.

## Why R2V cannot do what `examples/h3-dit-web` does

That page ships prompt embeddings baked offline, because its prompt list is
fixed. **Here the reference is the input**, so Qwen3-VL has to run in the tab —
which is the whole reason this is a different page and not a checkbox.

Measured, at int8: the conditioner is **25.78 GB** (vision tower 0.61, text
layers 0..49 24.40, embedding 0.78), the DiT **20.66 GB** (20.08 of weights and
0.58 of modulation tables), the VAE decoder 2.43. **48.9 GB**, which fits on no
card this page will meet. So they run **in sequence**: upload, use, drop, upload
the next.

**Dropping is not free and not instant.** `destroy()` schedules the freeing;
Dawn does it on its next tick, and it ticks on GPU work rather than on a timer.
`ResidentDevice.reclaim()` asks for that work — issue #213 has the measurement,
and `examples/h3-dit/src/generate.ts` ran its two phases as separate *processes*
before there was one. A tab has no process to exit.

## The tokenizer was already here

`ref2va`'s presentation needs Qwen2's byte-level BPE, and `llm/tokenizer-bpe.ts`
is exactly that, with its vocabulary committed at `llm/data/`. Whether it agrees
with H3's own tokenizer is a **measurement**, not an assumption:
`examples/h3-ref2v/src/tokenizer.test.ts` runs every text segment the
presentation can produce — labels, timestamps, prompts — plus the four vision
token ids. **Fourteen of fourteen, and all four ids.**

## The one step that is not held to the model

**The resize.** `Qwen2VLImageProcessor` resamples with PIL's bicubic; a browser
has `drawImage`, at `imageSmoothingQuality: "high"`, which is not it. Everything
downstream — `smartResize`'s target, the patchify, the tower, the presentation,
the layout — is checked against upstream. What the resampler itself costs is
**unmeasured**, and the page says so rather than leaving it to this file.

A reference video is read at the conditioner's 2 fps by seeking, not by decoding
everything: a 15-second reference is 360 frames at H3's 24 fps and the
conditioner looks at 30 of them.

## Running it

```bash
node examples/h3-ref2v-web/build.mjs
node examples/h3-ref2v-web/server.mjs --weights ~/h3-ref2v-web
```

Then `http://localhost:8791/`, or `?serve=/weights` to skip the folder gate —
which exists because `showDirectoryPicker` needs a user gesture and a headless
browser therefore cannot reach the page's actual work.

The page reads the adapter's limits **before** it uploads anything and refuses
with the number if `maxComputeWorkgroupSizeX` is below 512, because
`ops/matmul` declares that. Issue #211, and `examples/h3-dit-web` learned it the
expensive way.

## The licence

**Powered by MiniMax H3.** The model is under the MiniMax H3 Community License
Agreement, not this page's MIT, and nothing here redistributes it — the
agreement permits redistribution only within an *Applicable Territory* that
excludes the European Union, the United Kingdom, the Republic of Korea and the
United States of America. See issue #190.
