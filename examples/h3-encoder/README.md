# MiniMax-H3's visual VAE encoder

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
| one block, against `ResnetBlock3D` | **3.576e-7** | 5.060e-8 |
| the whole encoder, against `EncoderFCN3D` + `quant_conv` | **2.432e-5** | 6.045e-6 |

The block is f32 rounding on values of order one. The whole encoder is about
3e-6 relative on moments peaking at 8.854 — f32 accumulating through six levels
and twelve blocks, which is what an f32 port is.

**With `examples/h3-video`'s decoder, the VAE now round-trips**: a video in, a
latent, and frames out, both halves held to the model's own output.

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
# one block — 3.5 MB, and a vitest comparison
python examples/h3-encoder/tools/gen_resnet_golden.py \
  --bundle ~/h3/video_vae --weights ~/h3/video_vae/model.safetensors \
  --out ~/h3-encoder-fixtures
H3_ENCODER_DIR=~/h3-encoder-fixtures npx vitest run examples/h3-encoder

# the whole encoder — 721 MB, and a script
python examples/h3-encoder/tools/gen_resnet_golden.py \
  --bundle ~/h3/video_vae --weights ~/h3/video_vae/model.safetensors \
  --out ~/h3-encoder-whole --whole --video 8,32,32
npx tsx examples/h3-encoder/src/verify-encode.ts --dir ~/h3-encoder-whole
```

Without `H3_ENCODER_DIR` the block comparison **skips with a message** rather
than passing.

The whole encoder is a **script, not a test**: the reference walks six levels of
twelve blocks on the CPU and takes **123 seconds** for 8 frames of 32x32, well
past `scripts/test.mjs`'s per-file minute. `examples/anima` verifies its forward
the same way and so does `examples/h3-video/src/verify-decode.ts`. What stays in
vitest is the cheap arithmetic that is easy to get wrong — the channel plan and
the compression factors.

## The assembly, and the one thing it has to get right on its own

`encoder.ts` is six levels of two blocks with a strided convolution between the
first four, a final norm and a projection into `2 * z_channels` — mean and
log-variance as channels — then `quant_conv`.

**`Downsample3D` pads asymmetrically before it strides.** With
`space_stride == 2` it prepends nothing and appends one column and one row —
`F.pad(x, (0, 1, 0, 1, 0, 0))` — and then runs a `k=3, stride=2` convolution
whose *spatial* padding is zero. A symmetric pad gives the same output size and
a different alignment: a video that comes back shifted, level by level.
Measured, it moves the moments by **3.8**.

Six mutations on the assembly, all caught: the downsample pad made symmetric
(3.8), the causal frames dropped (wrong frame count), the channel plan read from
the current level rather than the previous (wrong shape at level 1), the final
silu replaced (7.6), `quant_conv` skipped (9.5), a downsample at every level
(wrong shape).

## What is not here

The encoder runs on the **CPU reference** only — there is no GPU path and no
browser page for it. The decoder has both. Encoding a clip at any real size
would need one, and it is what would let a page show real video rather than a
sample from the prior.
