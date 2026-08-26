# MiniMax-H3's video decoder, in a browser

The visual half of an omni-modal video model: 36 transformer blocks over a
24-channel latent, 16x compression in space and 4x in time, out to RGB frames.
Issue [#200](https://github.com/m96-chan/web-xpu-ops/issues/200).

**There is no prompt here, and that is not an omission.** This page is the
decoder. The transformer that would write the latent is 20B and the encoder that
would read a prompt is Qwen3-VL-32B — **27 GB between them** at the smallest
quantisation MiniMax has published. A latent sampled from the prior is what a
VAE decoder is built to receive, so that is what the page gives it.

## What is checked, and against what

`examples/h3-video/src/decoder-gpu.ts` is the decoder this page runs. It is held
to pixels produced by **the model's own `decode`** — `AutoencoderKLLegacy`'s
`post_quant_conv` and `ViT3DDecoder`, unmodified.

| | worst element | RMS |
| --- | --- | --- |
| one block (`block.ts`, CPU, against `TransformerBlock`) | 1.192e-7 | 1.773e-9 |
| the whole decoder (GPU, against `decode`) at 8x48x64 | 4.530e-6 | 9.051e-7 |
| the whole decoder (GPU) at 8x128x128 | 5.841e-6 | 7.736e-7 |

Decode times, RTX 5090 / driver 610.57.04 / Dawn `webgpu@0.4.0` / f32, 9.69 GB
resident:

| | |
| --- | --- |
| 8 frames, 128x128 | 608 ms |
| 16 frames, 256x256 | 771 ms |
| 16 frames, 512x512 | 574 ms |
| 32 frames, 512x512 | 568 ms |
| 64 frames, 512x512 | 665 ms |

The times barely move with the token count because **almost none of it is
compute**. At 8 frames of 128x128 in int8, the first decode splits as:

| | |
| --- | --- |
| the block submits | 40 ms |
| the final submit and its readback | 81 ms |
| host-side recording, 1,122 dispatches | 136 ms |
| **total** | **264 ms** |
| **second decode**, scratch pool warm | **131 ms** |

The split is approximate in one direction: every `batch` awaits
`onSubmittedWorkDone`, so what is labelled queue time includes the GPU actually
running.

Four candidates were tried and **all four are ruled out**, which is worth
recording so nobody repeats them:

| tried | result |
| --- | --- |
| grouping blocks into fewer submits | **slower** — 628 ms at one per submit, 725 at all thirty-six, identical output |
| pooling the uniform buffers | first decode 293 → 264 ms, steady state unchanged |
| keying the pipeline cache without a concatenation | 0.1 µs a call either way |
| removing `bindGroup`'s validation error scope | 136 → 135 ms, inside the run-to-run spread |

**Warming the scratch pool** is the one thing that helps, and only on the first
decode: it is worth about 130 ms.

A CPU profile puts the decode's own JavaScript at about 55 ms — bind groups 19,
pipeline lookups 14, `createBuffer` 14, `unpackPatches` 9 — and the rest under
`processImmediate`, which is Node pumping Dawn's event loop. So what is left is
**event-loop turns rather than work**, and reducing it means fewer awaits in the
dispatch path rather than faster JavaScript. That has not been attempted.

That is a number to improve, not a throughput to quote.

One attribution here was wrong once and is worth the warning: timing each
`await` inside the dispatch path blamed `pipelineFor` for 502 ms, and a tight
loop then priced the same call at 0.1 µs. A `performance.now()` pair straddling
an `await` charges whatever else the event loop runs to whatever is awaited.

**The in-browser decode is unmeasured.** Everything above is Node against the
same `VideoDecoderGpu` class the page instantiates; what the page adds is the
folder read, the browser device and the canvas. The page prints its own decode
time, and nothing has been recorded here.

## No new kernel

`matmul`, `rmsnorm`, `layernorm`, `flash_attention`, `rope`'s axes entry,
`activation`, `elementwise` and `permute` cover the whole decoder. One op
changed for it: `ropeAxes` takes **fractional** positions now, because H3
normalises each axis to `(-1, 1)` and multiplies by `2π` rather than indexing a
grid.

Three things that would each be a strided copy per token happen once at
conversion instead — `to_qkv` split into three projections, `ff.w1` split into
gate and up, and every weight stored `[in, out]` — along with the RoPE channel
permutation, which `permuteForRope` does for Anima.

## Running it

## The licence, read rather than assumed

The weights are **not in this repository and are not redistributed by it** — and
that was a safe default until now rather than a finding. The MiniMax H3
Community License Agreement (dated 2 August 2026) was read on 26 August 2026,
and it settles it:

- **Redistribution is permitted**, with a copy of the agreement and a NOTICE
  file, **within the "Applicable Territory"**.
- The Applicable Territory is *worldwide except* **the European Union, the
  United Kingdom, the Republic of Korea and the United States of America**.
- Any product using the model must display **"Powered by MiniMax H3"**, and a
  commercial one must show "MiniMax H3" prominently in its interface.
- Commercial use past **$20 M/year** in revenue needs separate written
  authorisation from MiniMax.
- Derivative works are owned by whoever makes them, but must still comply.

**A public mirror cannot honour the territory clause** — anyone can fetch it, in
any of the four excluded jurisdictions — so Anima and Z-Image's arrangement
(convert once, publish to a model host, let the page download it) is not
available here. Converting your own copy is the only path, and that is why the
page has no default `?weights=` base.

This is a reading, not legal advice; the agreement is at
`https://huggingface.co/MiniMaxAI/MiniMax-H3/raw/main/LICENSE`. See issue #190.

## Running it

Convert your own copy:

```bash
# the video_vae/ directory from MiniMaxAI/MiniMax-H3 (as a package, with an
# __init__.py) and source/model.safetensors from the same repository
python examples/h3-video/tools/convert_decoder.py \
  --bundle ~/h3/video_vae --weights ~/h3/video_vae/model.safetensors \
  --out ~/h3-video-web

node examples/h3-video-web/build.mjs
node examples/h3-video-web/server.mjs --weights ~/h3-video-web
```

Then open `http://localhost:8789/` and pick a folder. **9.69 GB** is downloaded
into it once and read from it every time after; nothing goes into the browser's
own storage, where it could be evicted without warning.

To check the conversion against the model yourself:

```bash
npx tsx examples/h3-video/src/verify-decode.ts --dir ~/h3-video-web
npx tsx examples/h3-video/src/verify-decode.ts --dir ~/h3-video-web --bench 8,32,32
```

The second form times a random latent and has nothing to compare against; it
says so on every run.

## int8, and what it costs

The page runs the **int8** conversion: `--quant q8` writes 2.43 GB where f32
writes 9.69, and `matmulQ8` reads `nn.Linear`'s `[out, in]` layout untransposed
with one absmax scale per output row.

What that costs was measured against the model's own pixels, in the units a
viewer sees:

| | worst | RMS |
| --- | --- | --- |
| f32, as 8-bit levels | 1 of 255 | 0.007 |
| **int8, as 8-bit levels** | **2 of 255** | **0.564** |
| int8, in the model's normalised space | 3.606e-2 | 6.860e-3 |

Two levels of 255 for a quarter of the size. The normalised number alone would
have said nothing: the denormalisation multiplies by a per-channel std of about
0.22, so quoting `3.606e-2` without converting it is how a quantisation gets
called accurate or inaccurate without either being established.

It is also faster to load — 292 ms against 636 ms for the first decode, since
most of that is allocating scratch and uploading.

The f32 conversion still exists and `verify-decode.ts` still checks it.

## What it needs

**A GPU with room for 2.43 GB of int8 weights plus its activations**, or 9.69 GB
for the f32 conversion.

## What the controls do

- **Size** — the latent's `T x H x W`. One latent frame is four video frames and
  one latent cell is 16x16 pixels, so `4,32,32` is 16 frames of 512x512.
- **Latent** — how the 24 channels are drawn. `prior` is independent standard
  normals, which is what the model's KL term trains towards. `smooth` correlates
  along time so the frames move together; `drift` crossfades between two draws.
  The last two exist because a latent of independent draws decodes to a field of
  independent noise, and the DiT this normally receives from does not emit that
  either.
- **Seed** — xorshift128+, the same generator the other demos use. It reproduces
  **this port's** runs, not torch's for the same number.
