#!/usr/bin/env python3
"""Bakes a golden for one `ZImageTransformerBlock`, from Z-Image's own code.

Correctness for the TypeScript port is defined by this file's output, not by
reading the port and agreeing that it looks right. The block is imported from
`musubi-tuner`'s copy of the Tongyi-MAI implementation rather than
reimplemented here, so the golden cannot drift from the model by way of a
transcription mistake in the generator.

The golden is a scaled-down configuration, not the shipped one — see the
comment on `axes_dims` below for exactly what that trades away.

`n_kv_heads == n_heads` here, matching the shipped config (`n_heads=30,
n_kv_heads=30`) — Z-Image's attention is MHA, not GQA, despite the code
carrying a separate `n_kv_heads`. A GQA case would exercise a path the model
never takes.

Run it with musubi-tuner's own interpreter, which has the deps the model file
imports (`accelerate`, and torch built against this machine's CUDA):

    /home/m96-chan/project/therdparty/musubi-tuner/.venv/bin/python \
        examples/zimage/tools/gen_block_golden.py

A bare `python3` fails on `accelerate` before reaching anything of ours.
"""
from __future__ import annotations

import argparse
import json
import struct
import sys
from pathlib import Path

MUSUBI = Path("/home/m96-chan/project/therdparty/musubi-tuner/src")

import torch  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path(__file__).resolve().parent.parent / "fixtures")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    if not MUSUBI.exists():
        sys.exit(f"musubi-tuner not found at {MUSUBI} — this generator reads Z-Image's own block from it")
    sys.path.insert(0, str(MUSUBI))
    from musubi_tuner.zimage.zimage_config import ADALN_EMBED_DIM, ROPE_AXES_DIMS, ROPE_AXES_LENS, ROPE_THETA
    from musubi_tuner.zimage.zimage_model import RopeEmbedder, ZImageTransformerBlock

    # The shipped model is dim=3840 / 30 heads / axes [32, 48, 48]. Reproducing
    # that head width here would put ~4 MB of random weights in the repository,
    # almost all of it width this test does not read: the block's structure does
    # not depend on how wide an axis is, only on there being three of them,
    # split in order, with more than one head to keep the head split from being
    # the identity. So the axes are scaled down by 4 and the head count to 2.
    #
    # What that gives up is stated rather than left implicit: this golden does
    # NOT prove the port handles head_dim=128. `ops/rope`'s own ropeAxes tests
    # own that, at the real widths. This one owns the composition.
    axes_dims = [d // 4 for d in ROPE_AXES_DIMS]  # [8, 12, 12]
    head_dim = sum(axes_dims)  # 32
    n_heads = 2
    dim = n_heads * head_dim  # 64
    seq = 6
    norm_eps = 1e-5

    torch.manual_seed(args.seed)
    torch.set_grad_enabled(False)

    block = ZImageTransformerBlock(
        layer_id=0, dim=dim, n_heads=n_heads, n_kv_heads=n_heads,
        norm_eps=norm_eps, qk_norm=True, modulation=True,
    ).eval().to(torch.float32)

    # Parameters start at ones (RMSNorm) or a small init (Linear); randomising
    # them is what makes the golden able to fail. A block left at init would
    # pass against a port that dropped a norm weight entirely.
    for p in block.parameters():
        p.copy_(torch.randn_like(p) * 0.05)

    x = torch.randn(1, seq, dim)
    adaln_input = torch.randn(1, min(dim, ADALN_EMBED_DIM))

    # Position ids: one row per token, one column per RoPE axis. Kept inside
    # ROPE_AXES_LENS and deliberately not the identity — a port that ignored a
    # column would still match if every axis carried the same positions.
    ids = torch.tensor([[t, t % 3, (t * 2) % 5] for t in range(seq)], dtype=torch.long)
    freqs_cis = RopeEmbedder(theta=ROPE_THETA, axes_dims=axes_dims, axes_lens=ROPE_AXES_LENS)(ids)

    out = block(x, freqs_cis=freqs_cis.unsqueeze(0), adaln_input=adaln_input)

    args.out.mkdir(parents=True, exist_ok=True)
    tensors: dict[str, torch.Tensor] = {
        "x": x,
        "adalnInput": adaln_input,
        "output": out,
        # freqs_cis is complex; split so the port reads plain f32 pairs.
        "freqsCos": torch.view_as_real(freqs_cis)[..., 0].contiguous(),
        "freqsSin": torch.view_as_real(freqs_cis)[..., 1].contiguous(),
    }
    for name, p in block.named_parameters():
        tensors[name.replace(".", "_")] = p.detach()

    blob = bytearray()
    manifest = []
    for name, t in tensors.items():
        flat = t.detach().to(torch.float32).contiguous().reshape(-1)
        manifest.append({"name": name, "shape": list(t.shape), "offset": len(blob) // 4, "length": flat.numel()})
        blob.extend(struct.pack(f"<{flat.numel()}f", *flat.tolist()))

    (args.out / "block.bin").write_bytes(bytes(blob))
    (args.out / "block.manifest.json").write_text(
        json.dumps(
            {
                "note": "Generated by tools/gen_block_golden.py from musubi-tuner's Z-Image block. Do not hand-edit. Axes are ROPE_AXES_DIMS scaled down 4x and heads reduced to 2 — see the generator for what that gives up.",
                "shippedConfig": {"dim": 3840, "nHeads": 30, "ropeAxesDims": ROPE_AXES_DIMS},
                "torch": torch.__version__,
                "seed": args.seed,
                "config": {
                    "dim": dim, "nHeads": n_heads, "nKvHeads": n_heads, "headDim": head_dim,
                    "seq": seq, "normEps": norm_eps, "adalnEmbedDim": min(dim, ADALN_EMBED_DIM),
                    "ffnHidden": int(dim / 3 * 8), "ropeAxesDims": axes_dims, "ropeTheta": ROPE_THETA,
                },
                "ids": ids.tolist(),
                "tensors": manifest,
            },
            indent=2,
        )
        + "\n"
    )
    print(f"wrote {args.out}/block.bin ({len(blob)} bytes), {len(manifest)} tensors")
    print(f"output[0,0,:4] = {out[0, 0, :4].tolist()}")


if __name__ == "__main__":
    main()
