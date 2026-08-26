#!/usr/bin/env python3
"""One transformer block of MiniMax-H3's visual VAE decoder, run by H3's own code.

Issue #200. The decoder is 36 identical blocks over 2048 channels — 9.69 GB of
the checkpoint's 10.42 — so a port is right or wrong at the block, and the rest
is a loop. This is the same order `examples/zimage` and `examples/anima` were
built in: block first, against the model's own output, then the forward.

The block's modules are **imported, not transcribed**. `TransformerBlock` and
`RotaryEmbeddingND` come from the bundle the checkpoint ships, so the numbers
this repository is checked against are the publisher's arithmetic and not a
reading of it.

    python examples/h3-video/tools/gen_block_golden.py \
      --bundle ~/h3/video_vae --weights ~/h3/video_vae/model.safetensors \
      --out examples/h3-video/fixtures

`--bundle` is the `video_vae/` directory from `MiniMaxAI/MiniMax-H3` (its `.py`
files, as a package with an `__init__.py`); `--weights` is
`source/model.safetensors` from the same repository.

**The weights are not this repository's and are not redistributed by it.** Only
block 0 is written out — 268 MB — and it is written next to the fixture rather
than committed. See issue #190.
"""

import argparse
import json
import math
import pathlib
import sys

import numpy as np
import torch
from safetensors import safe_open


# The decoder's own configuration, from `source/config.json`'s
# `vit_decoder_kwargs` plus `AutoencoderKLLegacy`'s derived values. Written out
# rather than read, so the fixture states the geometry it was generated at and a
# checkpoint with a different one fails the test rather than the generator.
CONFIG = {
    "heads": 32,
    "dim_head": 64,
    "num_layers": 36,
    "norm_type": "rms_norm",
    "norm_affine": True,
    "qk_norm_type": "rms_norm",
    "qk_norm_affine": False,
    "ffn_activation_fn": "silu",
    "ffn_use_gated": True,
    "rope_theta": 100.0,
    "rope_dim_ratio": 0.75,
    "patch_size": 16,
    "patch_size_t": 4,
    "in_channels": 24,
    "out_channels": 3,
    "num_register_tokens": 4,
    "eps": 1e-5,
}


def token_ids(dims):
    """`create_token_ids(..., "length_normalized")` — the axis normalised to (-1, 1)."""
    axes = []
    for n in dims:
        coords = torch.arange(0.5, n, dtype=torch.float32) / n
        axes.append(2.0 * coords - 1.0)
    return torch.stack(torch.meshgrid(*axes, indexing="ij"), dim=-1).flatten(0, len(dims) - 1)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", required=True, help="the video_vae/ directory from MiniMaxAI/MiniMax-H3")
    parser.add_argument("--weights", required=True, help="source/model.safetensors from the same repository")
    parser.add_argument("--out", required=True)
    parser.add_argument("--block", type=int, default=0)
    parser.add_argument("--dims", default="2,3,4", help="latent T,H,W for the golden")
    args = parser.parse_args()

    bundle = pathlib.Path(args.bundle).expanduser().resolve()
    sys.path.insert(0, str(bundle.parent))
    from video_vae.base_module import RotaryEmbeddingND, TransformerBlock  # type: ignore[import-not-found]

    out = pathlib.Path(args.out).expanduser()
    out.mkdir(parents=True, exist_ok=True)

    dim = CONFIG["heads"] * CONFIG["dim_head"]
    block = TransformerBlock(
        heads=CONFIG["heads"],
        dim_head=CONFIG["dim_head"],
        norm_type=CONFIG["norm_type"],
        norm_affine=CONFIG["norm_affine"],
        qk_norm_type=CONFIG["qk_norm_type"],
        qk_norm_affine=CONFIG["qk_norm_affine"],
        ffn_activation_fn=CONFIG["ffn_activation_fn"],
        ffn_use_gated=CONFIG["ffn_use_gated"],
        eps=CONFIG["eps"],
    ).eval()

    prefix = f"decoder.transformer_blocks.{args.block}."
    state = {}
    with safe_open(args.weights, framework="pt", device="cpu") as f:
        for key in f.keys():
            if key.startswith(prefix):
                state[key[len(prefix) :]] = f.get_tensor(key)
    missing, unexpected = block.load_state_dict(state, strict=False)
    # `strict=False` only so the qk norms -- which have no parameters when
    # `qk_norm_affine` is false -- do not have to be listed. Anything else
    # missing means the block this file builds is not the block the checkpoint
    # holds, and every number below would be for a different model.
    if unexpected:
        raise SystemExit(f"checkpoint has tensors this block does not: {unexpected}")
    if missing:
        raise SystemExit(f"block wants tensors the checkpoint does not have: {missing}")

    dims = [int(v) for v in args.dims.split(",")]
    ids = token_ids(dims)
    n_patches = ids.shape[0]
    n_suffix = 1 + CONFIG["num_register_tokens"]
    # Suffix tokens sit at position zero on every axis, as `ViT3DDecoder.forward`
    # builds them -- so they are rotated by the identity.
    img_ids = torch.cat([ids, torch.zeros(n_suffix, 3)], dim=0).unsqueeze(0)

    rope_apply_dim = int(CONFIG["dim_head"] * CONFIG["rope_dim_ratio"])
    pos_embed = RotaryEmbeddingND(rope_apply_dim, CONFIG["rope_theta"], n_dim=3, use_angle=True)
    rotary = pos_embed(img_ids)

    torch.manual_seed(20260826)
    hidden = torch.randn(1, n_patches + n_suffix, dim) * 0.5
    with torch.no_grad():
        want = block(hidden, rotary)

    print(f"block {args.block}: {n_patches} patches + {n_suffix} suffix, dim {dim}")
    print(f"  in  {tuple(hidden.shape)} -> out {tuple(want.shape)}")
    print(f"  out[0,:4,0] {[round(v, 6) for v in want[0, :4, 0].tolist()]}")

    # The weights, flat and f32, in the order the port reads them.
    order = [
        "norm1.weight", "attn.to_qkv.weight", "attn.to_qkv.bias",
        "attn.to_out.weight", "attn.to_out.bias", "scale1",
        "norm2.weight", "ff.w1.weight", "ff.w1.bias",
        "ff.w2.weight", "ff.w2.bias", "scale2",
    ]
    entries, chunks, offset = [], [], 0
    for name in order:
        tensor = state[name].to(torch.float32).contiguous()
        array = tensor.numpy().ravel()
        entries.append({"name": name, "shape": list(tensor.shape), "offset": offset, "count": int(array.size)})
        chunks.append(array)
        offset += int(array.size)
    with (out / "block.bin").open("wb") as f:
        for chunk in chunks:
            f.write(chunk.tobytes())

    (out / "block.manifest.json").write_text(json.dumps({
        "model": "minimax-h3-video-vae-decoder-block",
        "source": "MiniMaxAI/MiniMax-H3 (FL2VA/video_vae/source)",
        "licence": "MiniMax H3 Community License Agreement — not this repository's, and not redistributed by it",
        "block": args.block,
        "config": CONFIG,
        "dim": dim,
        "ffnHidden": dim * 4,
        "ropeApplyDim": rope_apply_dim,
        "patchDims": dims,
        "numPatches": n_patches,
        "numSuffix": n_suffix,
        "torch": torch.__version__,
        "tensors": entries,
        "elements": offset,
    }, indent=1))

    hidden.numpy().astype(np.float32).tofile(out / "block-input.bin")
    want.numpy().astype(np.float32).tofile(out / "block-want.bin")
    # `[N, 3]` angles, already scaled by 2*pi, as `ropeAxes` wants them.
    (2 * math.pi * img_ids[0]).numpy().astype(np.float32).tofile(out / "block-positions.bin")
    print(f"  wrote {out}/block.bin  {offset * 4 / 1e6:.1f} MB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
