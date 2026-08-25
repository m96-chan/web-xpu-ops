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

The times barely move with the token count because at these sizes they are not
compute: 36 submits and their pipeline lookups dominate. That is a number to
improve, not a throughput to quote.

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

The weights are **not in this repository and are not redistributed by it**. The
model is under the MiniMax H3 Community License Agreement, which is not this
code's MIT — see issue #190. Convert your own copy:

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

## What it needs

**A GPU with room for 9.69 GB of f32 weights plus its activations.** Quantising
to int8 would be 2.4 GB and is not done: `matmulQ8` exists, but what int8 costs
this decoder in accuracy has not been measured, and a wrong kernel is worth less
than none.

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
