# MiniMax-H3's audio decoder, in a browser

The audio half of an omni-modal video model: a BigVGAN vocoder over a
32-channel latent at 40 Hz, producing 32 kHz audio. Issue
[#200](https://github.com/m96-chan/web-xpu-ops/issues/200).

**There is no prompt here, and that is not an omission.** This page is the
decoder. The transformer that would write the latent is 20B and the encoder that
would read a prompt is Qwen3-VL-32B — **27 GB between them** at the smallest
quantisation MiniMax has published, against **260 MB** for this. A latent
sampled from the prior is what a VAE decoder is built to receive, so that is
what the page gives it. A prompt box that did nothing would be worse than saying
why there is none.

## What is checked, and against what

`examples/h3-audio/src/decoder-gpu.ts` is the decoder this page runs. It is held
to a waveform produced by **the model's own Python** — `DacAudioVAE.decode`,
unmodified, on a fixed latent — not to the CPU port beside it. Two ports that
agree with each other have only shown that they share a mistake.

| | worst element | time |
| --- | --- | --- |
| reference (`decoder.ts`, this library's own reference ops) | 1.788e-6 | 21 s |
| GPU (`decoder-gpu.ts`) | 5.007e-6 | 432 ms |

RTX 5090, driver 610.57.04, Dawn via `webgpu@0.4.0`, Node v25.6.1, f32, over a
6,400-sample golden whose signal peaks at 0.896. **In-browser timing is
unmeasured** — the page prints its own decode time and a real-time ratio, and
nothing has been recorded here yet.

## No new kernel

`ops/conv`, `ops/conv_transpose`, `ops/pad`, `ops/snake` and `ops/axpy` cover the
whole decoder. Four of those five landed for [VoxShot](https://github.com/m96-chan/voxshot)
in the weeks before this model existed here — `conv_transpose1d` "so a codec
decoder can reach a waveform", `snake_beta` because the name covers two
different functions, `group_norm` for a codec's residual blocks. A DAC-lineage
codec is a DAC-lineage codec.

Two things that would otherwise have needed a kernel are folded instead. The
`ratio *` after the anti-aliasing upsample rides on the filter's twelve taps,
since a convolution is linear in its weight. And the slice after it is the
transposed convolution's `padding` — the two trims are equal for this filter,
and a symmetric crop is what `padding` means there. The port **throws** if a
future filter makes them differ rather than quietly dropping a sample off the
front, which is a phase error no unit test hears.

## Running it

## The licence, read rather than assumed

The MiniMax H3 Community License Agreement (dated 2 August 2026) was read on
26 August 2026. It **permits redistribution** — with a copy of the agreement and
a NOTICE file — but only **within an "Applicable Territory" that excludes the
European Union, the United Kingdom, the Republic of Korea and the United States
of America**. A public mirror cannot honour that, so the arrangement Anima and
Z-Image use is not available here and converting your own copy is the only path.

It also requires any product using the model to display **"Powered by MiniMax
H3"**, which the page does, and separate written authorisation for commercial
use past $20 M/year in revenue.

A reading, not legal advice; the text is at
`https://huggingface.co/MiniMaxAI/MiniMax-H3/raw/main/LICENSE`. See issue #190.

## Running it

Convert your own copy:

```bash
# the audio_vae/ directory from MiniMaxAI/MiniMax-H3 — the .py files,
# config.yaml, metadata.json and model.safetensors
python examples/h3-audio/tools/convert_audio_vae.py \
  --bundle ~/h3/audio_vae --out ~/h3-audio-web

node examples/h3-audio-web/build.mjs
node examples/h3-audio-web/server.mjs --weights ~/h3-audio-web
```

Then open `http://localhost:8788/` and pick a folder. The 260 MB is downloaded
into it once and read from it every time after; nothing goes into the browser's
own storage, where it could be evicted without warning.

The converter folds `weight_norm` — the checkpoint stores `weight_g` and
`weight_v`, and the convolution's actual weight is `g * v / ||v||` per output
channel — and **checks the fold against the module's own `.weight`** across 172
modules (worst element 2.384e-7). The anti-aliasing filter is read from the
checkpoint rather than recomputed from the Kaiser window in the model's source:
one less formula to get wrong.

To compare against the golden yourself:

```bash
H3_AUDIO_DIR=~/h3-audio-web npx vitest run examples/h3-audio/src
```

Without `H3_AUDIO_DIR` those comparisons **skip with a message** rather than
passing. A suite that goes green on a missing model is worse than one that fails.

## What the controls do

- **Latent** — how the 32 channels are drawn. `prior` is independent standard
  normals, which is what the model's KL term trains towards. `smooth` runs a
  one-pole filter along time; `tone` crossfades between two draws. The last two
  exist because 40 Hz of white noise sounds like 40 Hz of white noise, and the
  DiT this decoder normally receives from does not emit that either.
- **Latent scale** — multiplies the draw. The prior is unit variance, so 1 is
  the honest setting and everything else is exploration.
- **Seed** — xorshift128+, the same generator the other demos use. It reproduces
  **this port's** runs and not torch's for the same number.
