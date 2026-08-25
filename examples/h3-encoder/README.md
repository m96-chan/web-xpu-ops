# MiniMax-H3's visual VAE encoder, one resnet block

**This is where `conv3d` and `pad` are used.** The visual VAE's *decoder* is a
ViT and needs neither (`examples/h3-video`); its **encoder** is a 3D
convolutional stack, and an op with no caller is a liability. Issue
[#200](https://github.com/m96-chan/web-xpu-ops/issues/200).

`ResnetBlock3D` is the unit the encoder repeats — two per level over six levels
— so a port is right or wrong here in the same way the decoder's transformer
block was.

## What is checked, and against what

`tools/gen_resnet_golden.py` **imports** `ResnetBlock3D` from the bundle the
checkpoint ships and runs level 0's first block on a fixed input.

| | worst element | RMS |
| --- | --- | --- |
| the block, against `ResnetBlock3D` | **3.576e-7** | 5.060e-8 |

f32 rounding on values of order one.

## The three conventions that matter

**Space reflects; time does not.** `BaseConv3d._apply_padding` pads H and W with
`reflect` and hands the temporal axis to `_apply_temporal_padding`, which for a
causal convolution prepends `2 * padding` **zero** frames and appends none. A
symmetric pad of any mode would let frame `t` see `t + 1`, which is the property
the encoder exists to deny — and it is why `ops/pad` takes `before` and `after`
separately rather than one width.

**`2 * padding`, not `padding`.** A `k=3` convolution with `padding=1` is given
*two* frames in front, so the output keeps the input's length and every output
frame is a function of `t`, `t-1`, `t-2`. Prepending one instead gives an output
one frame shorter, which is the plausible mistake.

**Group norm is per frame.** `use_t_isolated_gn: true` selects
`TemporalIsolatedSpatialParallelGroupNorm`, which merges time into the batch,
normalises and splits back — statistics pooled over `C/32 x H x W` *within one
frame*, not over the clip. Pooling over the clip instead moves values by 2.4.

## Running it

The weights are **not in this repository and are not redistributed by it**; the
MiniMax H3 Community License Agreement permits redistribution only within an
Applicable Territory that excludes the EU, the UK, South Korea and the USA, so a
public mirror cannot honour it. See issue #190 and
`examples/h3-video-web/README.md`.

```bash
python examples/h3-encoder/tools/gen_resnet_golden.py \
  --bundle ~/h3/video_vae --weights ~/h3/video_vae/model.safetensors \
  --out ~/h3-encoder-fixtures

H3_ENCODER_DIR=~/h3-encoder-fixtures npx vitest run examples/h3-encoder
```

One block is about 3.5 MB. Without `H3_ENCODER_DIR` the comparison **skips with
a message** rather than passing.

## What is not here

The rest of the encoder: six levels of two blocks each with `Downsample3D`
between them, a final group norm and a `conv_out` into `2 * z_channels`. Every
piece of it is this block plus a strided convolution, so what is left is
assembly rather than arithmetic — but it has not been written, and the encoder
has not been run end to end.
