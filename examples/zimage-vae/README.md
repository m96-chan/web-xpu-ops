# `examples/zimage-vae` — a latent goes in, a picture comes out

Z-Image's VAE decoder, composed from this repository's ops. 49.5M parameters,
real weights, and the output matches the model's own decode to **1.88e-6** on
pixels in `[-1, 1]`.

This is the first thing here that produces something you can look at.

```bash
# 1. fetch the weights and bake a latent (167 MB VAE download, once)
/home/m96-chan/project/therdparty/musubi-tuner/.venv/bin/python \
    examples/zimage-vae/tools/gen_latent.py --size 64 --out examples/zimage-vae/fixtures-small

# 2. decode it with the ops
npx tsx examples/zimage-vae/src/main.ts --fixtures examples/zimage-vae/fixtures-small
```

```
latent 1x16x8x8 -> image 64x64
decoded in 64.9s on the CPU reference path
worst |ours - model| = 1.878e-6 on pixels in [-1, 1]
wrote .../fixtures-small/decoded.png
```

Open `decoded.png` next to `reference.png` (the model's own decode) and
`input.png` (what was encoded). They are the same picture.

## Why a synthetic chart rather than a photograph

The default input is generated, not downloaded: no licence question, and its
bands are chosen so that a wrong decode looks wrong rather than merely
different — saturated primaries catch channel order and clipping, a grey ramp
catches the shift/scale factors, a checkerboard at the latent stride puts block
artefacts exactly where they would show, and thin diagonals are the high
frequency a VAE is *expected* to lose, so normal loss is not mistaken for a bug.

Pass `--image path.png` to use your own.

## What this establishes

Every op in the decoder path works, at full width, on real weights, in
composition:

| op | used for |
| --- | --- |
| `conv2d` (#145) | `conv_in`, both convs in every resnet, the shortcut 1x1s, each upsampler's 3x3, `conv_out` — 30+ calls |
| `nearestUpsample2d` (#146) | the three 2x upsamples between blocks |
| `group_norm` | every resnet's two norms, the mid-block attention's norm, `conv_norm_out` |
| `activation` (silu) | every resnet, and `conv_act` |
| `elementwise` | every residual add |
| `attention` | the mid-block's pixel-to-pixel attention |
| `matmul` | that attention's Q/K/V/out projections |

Two things were read from the model rather than assumed, because either one
guessed wrong yields a plausible picture rather than an error:

- **There is no `post_quant_conv`.** diffusers adds a 1x1 conv before the
  decoder when the config asks for one. Z-Image's does not, and the generator
  checks that rather than trusting it.
- **The mid-block attention is over pixels, not channels.** `[B, C, H, W]`
  becomes `[B, H*W, C]`; each position attends to every other, single-headed.

## What it does not establish

- **Speed, and it is the honest problem here.** 64.9s for a 64×64 image on the
  CPU reference path, which is six nested loops by design — the reference is
  meant to be the slowest, plainest statement of the maths, not a fast one.
  A 256×256 decode did not finish in ten minutes. **The GPU path is the next
  step**, and this is exactly the measurement that motivates it.
- **Not the encoder.** Python bakes the latent. `conv2d` exists but the
  encoder's strided blocks are not ported.
- **Not generation.** No DiT, no text encoder, no sampler — a latent has to
  come from somewhere, and here it comes from encoding a picture. Issue #148
  tracks the rest.

## Weights

`decoder.bin` is 190 MB and gitignored. `tools/gen_latent.py` regenerates it
from [`Tongyi-MAI/Z-Image`](https://huggingface.co/Tongyi-MAI/Z-Image) (VAE
only — 167 MB, against 12 GB for the DiT). The test skips visibly when it is
absent rather than passing silently, since CI has no checkpoint.

## Why the test uses the 64px fixture

The CPU reference cannot do 256px in test time. Correctness does not depend on
the size: every block, channel count and group is identical between them, only
`H` and `W` shrink. The full size is what the GPU path is for.
