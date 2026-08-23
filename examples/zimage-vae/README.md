# `examples/zimage-vae` — a latent goes in, a picture comes out

Z-Image's VAE decoder, composed from this repository's ops. 49.5M parameters,
real weights, running on the GPU, and the output matches the model's own decode
to **1.09e-5** on pixels in `[-1, 1]` (**1.88e-6** on the CPU reference path).

This is the first thing here that produces something you can look at.

```bash
# 1. fetch the weights and bake a latent (167 MB VAE download, once)
/home/m96-chan/project/therdparty/musubi-tuner/.venv/bin/python \
    examples/zimage-vae/tools/gen_latent.py --out examples/zimage-vae/fixtures

# 2. decode it with the ops, on the GPU
npx tsx examples/zimage-vae/src/main.ts --fixtures examples/zimage-vae/fixtures
```

```
latent 1x16x32x32 -> image 256x256
decoded in 1.38s on the GPU path
worst |ours - model| = 1.085e-5 on pixels in [-1, 1]
wrote .../fixtures/decoded.png
```

Add `--cpu` to run the reference path instead. It is the same structure and the
same answer, and it is **224x slower**: 64.9s for a 64x64 image, where the GPU
does 256x256 — sixteen times the pixels — in 1.38s. A 256x256 CPU decode did
not finish in ten minutes.

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

- **Not a speed claim.** The GPU numbers above are wall clock on one machine
  (RTX 5090, driver 610.57.04, Dawn via `webgpu` 0.4, Node v25.6.1), measured
  once each, with no warm-up and no roofline comparison. They say the path is
  usable, not that it is fast — nothing here has been tuned, and every dispatch
  reads its weights from the host on every call.
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
`H` and `W` shrink. The GPU test uses the same fixture so the two paths are
compared on the same thing.

## The bug 256px found

The GPU path was written, passed at 64x64, and **produced a broken image at
256x256 while reporting success** — worst error 9.0e-1 instead of 1.1e-5.

WebGPU caps a dispatch dimension at 65535 workgroups. The second-to-last up
block is `256 x 256 x 256 = 16,777,216` elements, which at 256 lanes per
workgroup is **65,536 — one over**. Going over does not raise: the command
buffer is invalidated, the submit does nothing, and the output buffer comes
back holding whatever it held before. The process exits cleanly and a picture
is written.

That is exactly the failure issue #112 documents for `runElementwise` and
`runActivation`, now with a concrete threshold: **256x256 reaches it**, which
is a small image. The fix here splits those per-element dispatches into
`65535 * 256`-element chunks — exact rather than approximate, since neither op
carries state between lanes.

It is also why this README leads with a number and not a picture. The broken
run produced an image too.
