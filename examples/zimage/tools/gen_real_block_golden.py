#!/usr/bin/env python3
"""Bakes a golden for `layers.0` at the shipped width, with the shipped weights.

`gen_block_golden.py` answers "does the composition match the model's algebra",
using a 64-wide block and random weights so the fixture can live in the
repository. It deliberately does not answer "does it match the model", because
its weights are not the model's and its head is not 128 wide.

This one does. Same block class, `dim=3840`, 30 heads, `head_dim=128`, RoPE
axes `[32, 48, 48]`, and the real `layers.0` tensors off the checkpoint. What
keeps it small enough to commit is the sequence length, not the width: only the
inputs and outputs are stored, and the weights come from the converted blob at
verification time — which is the point, since the loader is what is on trial.

**Two outputs, because two different things can be wrong.** Running the block
once with real weights and comparing against the port would fold a porting
mistake and the cost of 4-bit quantization into a single number, and then
neither could be attributed. So the same inputs are run twice:

  - `outputDense` — the block with the checkpoint's own bf16-to-f32 weights.
  - `outputQ4` — the block with those weights put through q4-g128 and back,
    exactly as `convert_dit.py` writes them and the TypeScript loader reads
    them.

The port is then compared against `outputQ4`, where an honest port should agree
to f32 rounding, and `outputDense` against `outputQ4` measures what the format
costs on a real layer — which is what issue #137 was reopened for.

Run with musubi-tuner's interpreter:

    /home/m96-chan/project/therdparty/musubi-tuner/.venv/bin/python \\
        examples/zimage/tools/gen_real_block_golden.py
"""
from __future__ import annotations

import argparse
import json
import struct
import sys
from pathlib import Path

import numpy as np
import torch

MUSUBI = Path("/home/m96-chan/project/therdparty/musubi-tuner/src")
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent.parent / "zimage-vae" / "tools"))

from convert_dit import is_q8, pack_q4, pack_q8, quantize_q4_g128, quantize_q8, should_quantize  # noqa: E402
from models import add_argument, resolve  # noqa: E402


def dequantize(packed: np.ndarray, scale: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """The TypeScript loader's `dequantizeQ4G128`, in numpy.

    Written from the same rule rather than by calling into it — the point of
    the comparison is that two independent readings of the format agree, and a
    shared implementation could be wrong in both places at once.
    """
    rows, cols = shape
    words = packed.reshape(rows, cols // 8)
    nibbles = np.zeros((rows, cols), dtype=np.int32)
    for i in range(8):
        nibbles[:, i::8] = (words >> (4 * i)) & 0xF
    codes = np.where(nibbles >= 8, nibbles - 16, nibbles)  # two's complement, four bits wide
    return (codes * np.repeat(scale, 128, axis=1)).astype(np.float32)


def dequantize_q8(packed: np.ndarray, scale: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """The loader's `dequantizeQ8`, in numpy. Four codes per word, one scale per row."""
    rows, cols = shape
    words = packed.reshape(rows, -1)
    bytes_ = np.zeros((rows, words.shape[1] * 4), dtype=np.int32)
    for i in range(4):
        bytes_[:, i::4] = (words >> (8 * i)) & 0xFF
    codes = np.where(bytes_ >= 128, bytes_ - 256, bytes_)[:, :cols]
    return (codes * scale.reshape(-1, 1)).astype(np.float32)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=HERE.parent / "fixtures")
    ap.add_argument("--seq", type=int, default=8, help="tokens; the only thing keeping the fixture small")
    ap.add_argument("--seed", type=int, default=0)
    add_argument(ap)
    args = ap.parse_args()

    if not MUSUBI.exists():
        sys.exit(f"musubi-tuner not found at {MUSUBI} — this generator reads Z-Image's own block from it")
    sys.path.insert(0, str(MUSUBI))
    from musubi_tuner.zimage.zimage_config import ADALN_EMBED_DIM, ROPE_AXES_DIMS, ROPE_AXES_LENS, ROPE_THETA
    from musubi_tuner.zimage.zimage_model import RopeEmbedder, ZImageTransformerBlock
    from safetensors import safe_open

    src = Path(resolve("transformer", args.model_dir))
    config = json.loads((src / "config.json").read_text())
    index = json.loads((src / "diffusion_pytorch_model.safetensors.index.json").read_text())

    dim = config["dim"]
    n_heads = config["n_heads"]
    head_dim = dim // n_heads
    if head_dim != sum(ROPE_AXES_DIMS):
        sys.exit(f"head_dim {head_dim} != sum(ROPE_AXES_DIMS) {sum(ROPE_AXES_DIMS)}; the config moved")

    torch.set_grad_enabled(False)
    torch.manual_seed(args.seed)

    block = ZImageTransformerBlock(
        layer_id=0, dim=dim, n_heads=n_heads, n_kv_heads=config["n_kv_heads"],
        norm_eps=config.get("norm_eps", 1e-5), qk_norm=True, modulation=True,
    ).eval().to(torch.float32)

    # The checkpoint's `layers.0.*`, loaded by the names the block itself uses.
    wanted = {n: f for n, f in index["weight_map"].items() if n.startswith("layers.0.")}
    dense: dict[str, torch.Tensor] = {}
    for name in sorted(wanted):
        with safe_open(src / wanted[name], framework="pt") as f:
            dense[name[len("layers.0."):]] = f.get_tensor(name).to(torch.float32)

    missing = {n for n, _ in block.named_parameters()} - set(dense)
    if missing:
        sys.exit(f"the checkpoint has no weight for {sorted(missing)}; the block's parameter names moved")
    block.load_state_dict(dense, strict=True)

    x = torch.randn(1, args.seq, dim)
    adaln_input = torch.randn(1, min(dim, ADALN_EMBED_DIM))
    # Deliberately not the identity in any column: a port that ignored one axis
    # would still match if all three carried the same positions.
    ids = torch.tensor([[t, t % 3, (t * 2) % 5] for t in range(args.seq)], dtype=torch.long)
    freqs = RopeEmbedder(theta=ROPE_THETA, axes_dims=ROPE_AXES_DIMS, axes_lens=ROPE_AXES_LENS)(ids)

    out_dense = block(x, freqs_cis=freqs.unsqueeze(0), adaln_input=adaln_input)

    # The same block, with every weight the converter quantizes put through the
    # format and back. `should_quantize` decides, so the split cannot drift
    # from what the converter actually writes.
    quantized_names: dict[str, str] = {}
    round_tripped = {}
    for name, t in dense.items():
        arr = t.numpy()
        if not should_quantize(f"layers.0.{name}", arr.shape):
            round_tripped[name] = t
        elif is_q8(f"layers.0.{name}"):
            codes, scale = quantize_q8(arr)
            round_tripped[name] = torch.from_numpy(dequantize_q8(pack_q8(codes), scale, arr.shape))
            quantized_names[name] = "q8"
        else:
            codes, scale = quantize_q4_g128(arr)
            round_tripped[name] = torch.from_numpy(dequantize(pack_q4(codes), scale, arr.shape))
            quantized_names[name] = "q4"
    block.load_state_dict(round_tripped, strict=True)
    out_q4 = block(x, freqs_cis=freqs.unsqueeze(0), adaln_input=adaln_input)

    cost = (out_q4 - out_dense).abs().max().item()
    rel = cost / out_dense.abs().max().item()
    # RMS relative to the output's own RMS is the headline number: the abs-max
    # ratio above is a tail statistic and reads three times worse than what the
    # block actually does to a typical channel.
    rel_rms = ((out_q4 - out_dense).pow(2).mean().sqrt() / out_dense.pow(2).mean().sqrt()).item()
    cosine = torch.nn.functional.cosine_similarity(out_q4.flatten(), out_dense.flatten(), dim=0).item()

    args.out.mkdir(parents=True, exist_ok=True)
    tensors = {
        "x": x,
        "adalnInput": adaln_input,
        "outputDense": out_dense,
        "outputQ4": out_q4,
        "freqsCos": torch.view_as_real(freqs)[..., 0].contiguous(),
        "freqsSin": torch.view_as_real(freqs)[..., 1].contiguous(),
    }
    blob = bytearray()
    manifest = []
    for name, t in tensors.items():
        flat = t.detach().to(torch.float32).contiguous().reshape(-1)
        manifest.append({"name": name, "shape": list(t.shape), "offset": len(blob) // 4, "length": flat.numel()})
        blob.extend(struct.pack(f"<{flat.numel()}f", *flat.tolist()))

    (args.out / "real-block.bin").write_bytes(bytes(blob))
    (args.out / "real-block.manifest.json").write_text(json.dumps({
        "note": "Generated by tools/gen_real_block_golden.py from the shipped checkpoint's layers.0. "
                "Do not hand-edit. Weights are NOT stored here — they come from convert_dit.py's blob.",
        "torch": torch.__version__,
        "seed": args.seed,
        "config": {
            "dim": dim, "nHeads": n_heads, "headDim": head_dim, "seq": args.seq,
            "normEps": config.get("norm_eps", 1e-5), "adalnEmbedDim": min(dim, ADALN_EMBED_DIM),
            "ffnHidden": block.feed_forward.w1.weight.shape[0],
            "ropeAxesDims": ROPE_AXES_DIMS, "ropeTheta": ROPE_THETA,
        },
        "ids": ids.tolist(),
        "quantizedWeights": quantized_names,
        "quantizationCost": {"absMax": cost, "relativeAbsMax": rel, "relativeRms": rel_rms,
                             "cosine": cosine},
        "tensors": manifest,
    }, indent=2) + "\n")

    print(f"wrote {args.out}/real-block.bin ({len(blob) / 1e3:.0f} kB)")
    print(f"{len(quantized_names)} of {len(dense)} weights quantized "
          f"({sum(1 for v in quantized_names.values() if v == 'q8')} at q8)")
    print(f"quantized vs dense on a real layer: rel-RMS {rel_rms:.4f}, "
          f"cos {cosine:.6f}, abs max {cost:.4g} (relative {rel:.4g})")
    print(f"outputDense[0,0,:4] = {out_dense[0, 0, :4].tolist()}")


if __name__ == "__main__":
    main()
