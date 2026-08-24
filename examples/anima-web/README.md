# `examples/anima-web` — Anima-3.8B in a browser

The same pipeline as `examples/anima`, as a page. Nothing here is new
arithmetic: every stage is the one already checked against ComfyUI's own model,
and this directory is the wiring plus the two things a page needs that a script
does not — weights over HTTP instead of off a disk, and a device that survives
between steps.

```bash
node examples/anima-web/build.mjs
node examples/anima-web/server.mjs \
  --dit ~/anima-q8 \
  --encoder ~/anima-src/qwen_3_06b_base.safetensors \
  --vae ~/anima-src/qwen_image_vae.safetensors
```

Then `http://127.0.0.1:8789/` in a browser with WebGPU. Nothing leaves the
machine.

## What runs where

| | device | why |
| --- | --- | --- |
| the DiT | resident | its 3.76 GB stay uploaded between steps — 0.2 s a forward against 20 s |
| the encoder | per-dispatch `Runner` | touched once per generation |
| the VAE | per-dispatch `Runner` | touched once per generation |
| the adapter's 6 blocks | CPU | 512 tokens at dim 1024, against a DiT of 3,952 at 2048 |

## Weights, and where they live

**In the browser's disk cache, not on its heap.** `preloadAll` walks every
tensor once into the Cache API and keeps none of them; `preloadPrefix` hydrates
one block's worth just before the forward reads it. Two wrong answers came
first and are worth keeping written down:

* fetching per block *during* the forward put 3.76 GB on the wire in every
  denoising step — the same bytes forty times over;
* holding everything quieted the network and put 3.76 GB on the heap, where the
  next allocation to fail is an activation.

The first visit pays for 3.76 GB of DiT, 1.19 GB of encoder and 0.25 GB of VAE.
Every visit after it does no network at all.

`server.mjs` answers `Range` requests, and `fetch-weights.ts` **refuses a 200**:
a server that ignored ranges would "work" while sending whole files.

## The two invariants a page cannot check for itself

* **The rope permutation.** `packedQ8` returns `null` for anything permuted, so
  the resident path's fast route cannot go around the relabelling. Skipping it
  is measured at 1.068e-1 against torch.
* **The preload prefixes.** The forward reads synchronously, so a tensor whose
  name no announced prefix covers is a run-time failure *in the page*, after a
  4.7 GB download. `verify-forward-gpu.ts` asserts the cover in Node instead —
  1,049 tensors read, 57 prefixes announced, nothing uncovered.

## Measured

Chrome 151, NVIDIA GeForce RTX 5090, driver 610.57.04, over `server.mjs` on
loopback.

| | |
| --- | --- |
| first visit, caching the DiT | 4.68 GB in 2,416 range responses |
| a 256x256 generation, 8 steps, CFG 8 | **64.5 s** end to end |
| a DiT forward at 256x256, weights resident | ~4.1 s a step, two per step under CFG |

The first forward is slower than the rest: it hydrates the heap block by block
from the disk cache and uploads 3.63 GB to the device. Every step after it, and
every generation after the first, does neither.

832x1216 is **not measured here**. The command-line path takes 672 s of
sampling for it, and a page that holds the tab for eleven minutes is a
different question from whether the arithmetic is right.
