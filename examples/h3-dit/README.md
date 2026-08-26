# MiniMax-H3's DiT, one block

The **generator** half of MiniMax-H3: fifty identical transformer blocks over
5,376 channels, producing the latents `examples/h3-video` decodes. Issue
[#200](https://github.com/m96-chan/web-xpu-ops/issues/200).

**One block, not a forward.** A port of a stack like this is right or wrong at
the block and the rest is a loop — the order `examples/zimage`,
`examples/anima` and `examples/h3-video` were all built in. What a *forward*
would need is at the bottom of this file, and it is not close.

## What is checked, and against what

The checkpoint ships **no code** for the DiT; `transformer/config.json` names a
diffusers class and nothing else. So `tools/gen_block_golden.py` fetches
`MiniMaxH3TransformerBlock` from diffusers' main branch, rewrites **only its
relative import lines** to absolute, and runs block 0 on a fixed input. The
arithmetic is upstream's.

| golden | worst element | RMS |
| --- | --- | --- |
| f32 | 5.859e-3 | 3.065e-4 |
| f64 (`--f64`) | **1.953e-3** | 4.353e-5 |

on a block output of order 350. A third of the gap is torch's own f32 rounding
over 5,376- and 14,336-term dot products; the rest is this port storing
intermediates in `Float32Array` between ops, which is what an f32 port is — the
feed-forward turns inputs of order one into outputs of order a thousand, where
one f32 ulp is already 1e-4. `--f64` exists to separate those two, not to ship.

`--temb-scale` defaults to **0.05** deliberately. `time_embedder` is a trained
module whose output is order one; a standard normal through `adaln_proj` gives
gates of order ten and a block output of order 1e4, where cancellation dominates
any comparison and hides what is being measured.

## The three things that differ from every other block here

Each returns a well-formed tensor of the right shape when it is got wrong.

**The modulation table is indexed per row.** `adaln_proj` projects one timestep
embedding into `6 * hidden * 3` values — three *modalities*: text, video, audio
— and every token picks its row with `timestep * 3 + tag`. A forward that is all
one modality still has to pick the right one of the three.

**SwiGLU is `hidden * silu(gate)` with `hidden` the first half.**
`examples/h3-video`'s decoder is the other way round. Two files in one model
with opposite conventions.

**A permutation has to cover everything that reads the channels it moves.** The
RoPE permutation goes into the Q and K projection weights, as it does for Anima
and for the visual VAE — but this block's QK-norm has *per-channel weights*, and
permuting the projection without permuting those scales the wrong channels. The
visual VAE has `qk_norm_affine: false` and no weights at all, so the identical
permutation is complete there and incomplete here. The symptom was an 8% error
in attention with the rope itself exact to 2.4e-7, which is why the rope was
checked in isolation before anything else was suspected.

## No new kernel

`matmul`, `rmsnorm`, `attention`, `rope`'s axes entry, `activation` and
`elementwise` cover the block. `h3RopePermutation` lives in `ops/rope` because
both of H3's models rotate the same way at different geometries — 48 of 64 in
the visual VAE, 96 of 128 here.

## Running it

The weights are **not in this repository and are not redistributed by it** (the
MiniMax H3 Community License Agreement is not this code's MIT — issue #190).

```bash
# upstream's file, with its relative imports made absolute
curl -o h3dit.py https://raw.githubusercontent.com/huggingface/diffusers/main/\
src/diffusers/models/transformers/transformer_minimax_h3.py
sed -i 's/^from \.\.\./from diffusers./; s/^from \.\./from diffusers.models./' h3dit.py

python examples/h3-dit/tools/gen_block_golden.py \
  --module ./h3dit.py \
  --shard ~/h3/transformer/diffusion_pytorch_model-00001-of-00014.safetensors \
  --out ~/h3-dit-fixtures

H3_DIT_DIR=~/h3-dit-fixtures npx vitest run examples/h3-dit/src
```

**A partial download of the shard is enough.** Block 0 ends at 1.47 GB of the
shard's 5.23, and `safetensors.safe_open` insists on a complete file — so the
generator reads by byte offset and widens bf16 by hand. Refetching 3.75 GB to
satisfy a length check would be bandwidth spent on nothing.

Without `H3_DIT_DIR` the comparison **skips with a message** rather than passing.

## What a forward would need, and why it is not here

| | size | state |
| --- | --- | --- |
| this DiT, 50 layers (20B pruned) | 8.8–21 GB quantised, 40–66 GB bf16 | **one block checked** |
| Qwen3-VL-32B text encoder | 13.1–27 GB | not implemented; `llm/` has no VL |
| visual VAE decoder | 2.43 GB int8 | **works** (`examples/h3-video`) |

Twenty-seven gigabytes between the two that are missing, at the smallest
quantisation MiniMax has published. The block being right is the part that could
be established in a night; the rest is not a night's work.
