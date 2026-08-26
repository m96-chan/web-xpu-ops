# MiniMax-H3's visual VAE decoder

A latent in, video frames out: 36 transformer blocks over a 24-channel latent,
16x compression in space and 4x in time. Issue
[#200](https://github.com/m96-chan/web-xpu-ops/issues/200). The browser page is
`examples/h3-video-web`; this directory is the decoder, the converter and the
verification.

**The decoder is a ViT, not a CNN.** The convolutional half of this VAE is its
*encoder*; `block.ts` and `decoder-gpu.ts` are a transformer stack. `ops/conv`'s
3D entry and `ops/pad` were added for the encoder and are not used here.

## What is checked, and against what

| | worst element | RMS |
| --- | --- | --- |
| one block (`block.ts`, CPU, against `TransformerBlock`) | 1.192e-7 | 1.773e-9 |
| the decoder (GPU, f32, against `decode`) at 8x48x64 | 4.530e-6 | 9.051e-7 |
| the decoder (GPU, f32) at 8x128x128 | 5.841e-6 | 7.736e-7 |
| the decoder (GPU, **int8**) at 8x128x128 | 3.606e-2 | 6.860e-3 |

The goldens come from **the model's own code** — `TransformerBlock`,
`RotaryEmbeddingND` and `ViT3DDecoder` imported from the bundle the checkpoint
ships, not transcribed.

int8's 3.606e-2 says nothing on its own. In the units a viewer sees, after the
ImageNet denormalisation the model applies:

| | worst | RMS |
| --- | --- | --- |
| f32 | 1 level of 255 | 0.007 |
| **int8** | **2 levels of 255** | **0.564** |

Two levels of 255 for a quarter of the size (2.43 GB against 9.69).

## No new kernel

`matmul`, `matmulQ8`, `rmsnorm`, `layernorm`, `flash_attention`, `rope`'s axes
entry, `activation`, `elementwise` and `permute` cover the whole decoder. One op
changed for it: `ropeAxes` takes **fractional** positions, because H3 normalises
each axis to `(-1, 1)` and multiplies by `2π` rather than indexing a grid.

Four things happen once at conversion rather than per token:

- `to_qkv` is split into three projections — the model reads it as
  `view(B, L, -1, 3 * dim_head).chunk(3, -1)`, so taking `q` out of the *output*
  is a gather per token and taking it out of the *weight* is three ordinary
  matmuls;
- `ff.w1` is split into gate and up, for the same reason;
- the RoPE channel permutation goes into the q and k weights, as `permuteForRope`
  does for Anima;
- every weight is stored in whichever layout its kernel reads — `[in, out]` for
  f32, `[out, in]` untransposed for int8, which are **opposite**.

## Running it

The weights are **not in this repository and are not redistributed by it** (the
MiniMax H3 Community License Agreement is not this code's MIT — issue #190).

```bash
# the video_vae/ directory from MiniMaxAI/MiniMax-H3, as a package with an
# __init__.py, and source/model.safetensors from the same repository
python examples/h3-video/tools/convert_decoder.py \
  --bundle ~/h3/video_vae --weights ~/h3/video_vae/model.safetensors \
  --out ~/h3-video-web --quant q8         # or --quant f32 for 9.69 GB

npx tsx examples/h3-video/src/verify-decode.ts --dir ~/h3-video-web
npx tsx examples/h3-video/src/verify-decode.ts --dir ~/h3-video-web --bench 8,32,32
```

`--bench` times a random latent and has nothing to compare against; it says so
on every run. `--skip-weights` regenerates a golden at another size without
rewriting 9.69 GB that would come out byte-identical.

`tools/gen_block_golden.py` produces the single-block fixture the vitest suite
uses; `H3_VIDEO_DIR` points at its output, and without it the comparison skips
with a message rather than passing.

## The verification decodes twice, and that is not a formality

Scratch buffers are pooled, so the **second** decode is the first one to see a
*used* buffer — and anything the decoder relies on being zero rather than
writing is right the first time and wrong after. The cls token is exactly that:
`ViT3DDecoder` builds it as `torch.zeros_like(...)`, and with the explicit clear
deleted a single-decode check stayed green at 4.530e-6 while the second decode
was off by 2.478e-1.

## Speed

RTX 5090, driver 610.57.04, Dawn `webgpu@0.4.0`, int8, resident:

| | |
| --- | --- |
| 8 frames, 128x128 | 264 ms first, **131 ms** after |
| 64 frames, 512x512 | 665 ms (f32; int8 unmeasured at this size) |

Almost none of it is compute — see `examples/h3-video-web/README.md` for the
split and for the four optimisations that were measured and **did not work**.
