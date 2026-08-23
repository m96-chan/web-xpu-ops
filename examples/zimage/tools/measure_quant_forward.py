#!/usr/bin/env python3
"""What each quantization choice costs over the **whole** DiT forward.

`gen_real_block_golden.py` measures one layer, where q4-g128 with adaLN at q8
costs 1.78e-2 relative RMS. That number does not answer the question anyone
actually has, because 34 layers do not add — they compound. Measured here: the
same configuration costs **0.21** over the full forward.

So the choice is a real one and needs numbers on both sides: 3.34 GB at one
error, 6.5 GB at another. This runs the model once per configuration, on
identical inputs, and reports the velocity prediction's relative RMS against
the full-precision run.

Weights are streamed shard by shard and quantized into the model in place. The
obvious version — build a quantized state dict, then load it — needs 24.6 GB
for the model and another 24.6 GB for the dict, and this machine does not have
49 GB to spare.

    /home/m96-chan/project/therdparty/musubi-tuner/.venv/bin/python \\
        examples/zimage/tools/measure_quant_forward.py
"""
from __future__ import annotations

import argparse
import inspect
import json
import sys
from pathlib import Path

import numpy as np
import torch

MUSUBI = Path("/home/m96-chan/project/therdparty/musubi-tuner/src")
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent.parent / "zimage-vae" / "tools"))

from convert_dit import pack_q4, pack_q8, quantize_q4_g128, quantize_q8, should_quantize  # noqa: E402
from gen_real_block_golden import dequantize, dequantize_q8  # noqa: E402
from models import add_argument, resolve  # noqa: E402

# Each configuration says, for one tensor name, which format it takes.
# `None` means left at full precision.
CONFIGS: dict[str, callable] = {
    "dense": lambda name: None,
    "all-q4": lambda name: "q4",
    "adaLN-q8": lambda name: "q8" if "adaLN" in name else "q4",
    "adaLN+qk-q8": lambda name: "q8" if ("adaLN" in name or name.endswith(("to_q.weight", "to_k.weight"))) else "q4",
    "all-q8": lambda name: "q8",
}


def round_trip(arr: np.ndarray, kind: str | None) -> np.ndarray | None:
    if kind is None:
        return None
    if kind == "q8":
        codes, scale = quantize_q8(arr)
        return dequantize_q8(pack_q8(codes), scale, arr.shape)
    codes, scale = quantize_q4_g128(arr)
    return dequantize(pack_q4(codes), scale, arr.shape)


def bits_per_weight(kind: str | None) -> float:
    # q4: 4 bits plus one f32 scale per 128 weights. q8: 8 bits plus one f32
    # scale per row, which is negligible and ignored here.
    return {None: 32.0, "q4": 4.25, "q8": 8.0}[kind]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--latent", type=int, default=16)
    ap.add_argument("--cap-len", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    add_argument(ap)
    args = ap.parse_args()

    sys.path.insert(0, str(MUSUBI))
    from musubi_tuner.zimage import zimage_model as zm
    from safetensors.torch import load_file

    src = Path(resolve("transformer", args.model_dir))
    config = json.loads((src / "config.json").read_text())
    index = json.loads((src / "diffusion_pytorch_model.safetensors.index.json").read_text())

    torch.set_grad_enabled(False)
    torch.manual_seed(args.seed)

    accepted = set(inspect.signature(zm.ZImageTransformer2DModel.__init__).parameters)
    kwargs = {k: v for k, v in config.items() if k in accepted}
    for key in ("all_patch_size", "all_f_patch_size"):
        kwargs[key] = tuple(kwargs[key])
    model = zm.ZImageTransformer2DModel(**kwargs).eval().to(torch.float32)

    x = torch.randn(1, config["in_channels"], 1, args.latent, args.latent)
    t = torch.tensor([0.7])
    cap_feats = torch.randn(1, args.cap_len, config["cap_feat_dim"])
    cap_mask = torch.ones(1, args.cap_len, dtype=torch.bool)
    cap_mask[0, -1] = False
    patch = config["all_patch_size"][0]

    shards = sorted(set(index["weight_map"].values()))
    params = dict(model.named_parameters())
    reference: torch.Tensor | None = None

    for label, choose in CONFIGS.items():
        quantized_bits = 0.0
        total_weights = 0
        for shard in shards:
            for name, tensor in load_file(str(src / shard)).items():
                target = params.get(name)
                if target is None:
                    # Buffers, if the model grows any. Reported rather than
                    # skipped silently.
                    print(f"  note: {name} is not a parameter, skipped")
                    continue
                arr = tensor.to(torch.float32).numpy()
                kind = choose(name) if should_quantize(name, arr.shape) else None
                replaced = round_trip(arr, kind)
                target.data.copy_(torch.from_numpy(replaced if replaced is not None else arr))
                total_weights += arr.size
                quantized_bits += arr.size * bits_per_weight(kind)

        out = model(x, t, cap_feats, cap_mask, patch_size=patch, f_patch_size=1)
        size_gb = quantized_bits / 8 / 1e9
        if reference is None:
            reference = out.clone()
            print(f"{label:14s} {size_gb:5.2f} GB   (reference)")
            continue
        rel = ((out - reference).pow(2).mean().sqrt() / reference.pow(2).mean().sqrt()).item()
        cos = torch.nn.functional.cosine_similarity(out.flatten(), reference.flatten(), dim=0).item()
        print(f"{label:14s} {size_gb:5.2f} GB   rel-RMS {rel:.4f}   cos {cos:.6f}")


if __name__ == "__main__":
    main()
