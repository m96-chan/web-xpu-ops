#!/usr/bin/env python3
"""Bakes a golden for Anima's `net.blocks.0`, with the shipped weights.

Issue #170's first checkpoint. The block is imported from ComfyUI's own
`comfy/ldm/cosmos/predict2.py` rather than reimplemented here, so the golden
cannot drift from the model by way of a transcription mistake in the generator
— the same arrangement `examples/zimage` uses.

**Two goldens, for the reason #166 learned the hard way.** Running the block
once with real weights and comparing against a port folds a porting mistake and
the cost of quantization into one number, and then neither can be attributed.
So the same inputs are run twice: once with the checkpoint's own weights, once
with them put through q8 exactly as `convert_dit.py` writes them.

What differs from Z-Image's block, all of it read from the source rather than
assumed:

  - **LayerNorm, not RMSNorm**, with `elementwise_affine=False` and `eps=1e-6`
    (`predict2.py:438`). Three of them, one per sub-block.
  - **Three adaLN modulations**, each `SiLU -> Linear(dim, 256) -> Linear(256,
    3*dim)` and chunked **`shift, scale, gate`** (`:451-465`, `:487-504`).
  - `norm(x) * (1 + scale) + shift`, then `x += gate * result` (`:520-534`).
    No `tanh` on the gate, and a shift Z-Image has no equivalent of.
  - **Cross-attention** against a separate context of width 1024.
  - A plain **GELU** MLP, no gate.

    /home/m96-chan/project/therdparty/musubi-tuner/.venv/bin/python \\
        examples/anima/tools/gen_block_golden.py --src <checkpoint>
"""
from __future__ import annotations

import argparse
import json
import struct
import sys
from pathlib import Path

import numpy as np
import torch

# ComfyUI itself, sparse-cloned. The block cannot be imported in isolation —
# `predict2.py` reaches into `comfy.patcher_extension`, `comfy.ldm.modules.
# attention` and `comfy.ldm.common_dit` — and copying those files here would
# make this generator a transcription of the model rather than the model, which
# is the one thing it exists not to be.
#
#   git clone --depth 1 --filter=blob:none --sparse \
#       https://github.com/comfyanonymous/ComfyUI /tmp/ComfyUI
#   cd /tmp/ComfyUI && git sparse-checkout set comfy
COMFY = Path("/tmp/ComfyUI")
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from convert_dit import pack_q8, quantize_q8, should_quantize  # noqa: E402


def dequantize_q8(packed: np.ndarray, scale: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """The loader's `dequantizeQ8`, in numpy — written from the format rather
    than by calling the packer, so two readings of it have to agree."""
    rows, cols = shape
    words = packed.reshape(rows, -1)
    raw = np.zeros((rows, words.shape[1] * 4), dtype=np.int32)
    for i in range(4):
        raw[:, i::4] = (words >> (8 * i)) & 0xFF
    codes = np.where(raw >= 128, raw - 256, raw)[:, :cols]
    return (codes * scale.reshape(-1, 1)).astype(np.float32)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, required=True, help="Anima-3.8B.safetensors")
    ap.add_argument("--out", type=Path, default=HERE.parent / "fixtures")
    ap.add_argument("--seq", type=int, default=8, help="image tokens; the only thing keeping this small")
    ap.add_argument("--context", type=int, default=6, help="context tokens for the cross-attention")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    if not COMFY.exists():
        sys.exit(
            f"{COMFY} not found. This generator reads ComfyUI's own block:\n"
            f"  git clone --depth 1 --filter=blob:none --sparse "
            f"https://github.com/comfyanonymous/ComfyUI {COMFY}\n"
            f"  cd {COMFY} && git sparse-checkout set comfy"
        )
    sys.path.insert(0, str(COMFY))
    import comfy.ops
    from comfy.ldm.cosmos import predict2

    from safetensors import safe_open

    torch.set_grad_enabled(False)
    torch.manual_seed(args.seed)

    prefix = "net.blocks.0."
    dense: dict[str, torch.Tensor] = {}
    with safe_open(args.src, framework="pt") as f:
        for name in f.keys():
            if name.startswith(prefix):
                dense[name[len(prefix):]] = f.get_tensor(name).to(torch.float32)

    if not dense:
        sys.exit(f"no {prefix}* tensors in {args.src}")

    dim = dense["self_attn.q_proj.weight"].shape[1]
    head_dim = dense["self_attn.q_norm.weight"].shape[0]
    heads = dim // head_dim
    context_dim = dense["cross_attn.k_proj.weight"].shape[1]
    mlp_hidden = dense["mlp.layer1.weight"].shape[0]
    # `[adaln_lora_dim, x_dim]` — the LoRA's **first** matrix, so the rank is
    # row count, not column count. Reading it the other way round asks for a
    # `[2048, 2048]` where the checkpoint has `[256, 2048]`, which
    # `load_state_dict` catches; a shape mismatch is the good failure.
    adaln_dim = dense["adaln_modulation_self_attn.1.weight"].shape[0]
    print(f"dim {dim}, heads {heads}, head_dim {head_dim}, context {context_dim}, "
          f"mlp {mlp_hidden}, adaln_lora {adaln_dim}")

    block = predict2.Block(
        x_dim=dim,
        context_dim=context_dim,
        num_heads=heads,
        mlp_ratio=mlp_hidden / dim,
        use_adaln_lora=True,
        adaln_lora_dim=adaln_dim,
        device=None,
        dtype=torch.float32,
        # ComfyUI's own ops, not `torch.nn`: `cast_bias_weight` reads a `bias`
        # attribute off every norm, which `torch.nn.RMSNorm` does not have.
        operations=comfy.ops.disable_weight_init,
    ).eval().to(torch.float32)

    missing = {n for n, _ in block.named_parameters()} - set(dense)
    if missing:
        sys.exit(f"the checkpoint has no weight for {sorted(missing)[:6]}; the block's names moved")
    block.load_state_dict(dense, strict=True)

    T, H, W = 1, args.seq, args.seq
    x = torch.randn(1, T, H, W, dim)
    # The conditioning the adaLN reads is `x_dim` wide, not the LoRA's rank —
    # `SiLU -> Linear(x_dim, rank) -> Linear(rank, 3*x_dim)` (`predict2.py:451`).
    # `t_embedder` produces it, and at `dim=2048` this is 2048.
    emb = torch.randn(1, T, dim)
    context = torch.randn(1, args.context, context_dim)
    lora = torch.randn(1, T, 3 * dim)

    def run() -> torch.Tensor:
        return block(x, emb, crossattn_emb=context, rope_emb_L_1_1_D=None, adaln_lora_B_T_3D=lora)

    out_dense = run()

    quantized: list[str] = []
    round_tripped = {}
    for name, t in dense.items():
        arr = t.numpy()
        if should_quantize(f"{prefix}{name}", arr.shape):
            codes, scale = quantize_q8(arr)
            round_tripped[name] = torch.from_numpy(dequantize_q8(pack_q8(codes), scale, arr.shape))
            quantized.append(name)
        else:
            round_tripped[name] = t
    block.load_state_dict(round_tripped, strict=True)
    out_q8 = run()

    rel = ((out_q8 - out_dense).pow(2).mean().sqrt() / out_dense.pow(2).mean().sqrt()).item()
    print(f"{len(quantized)} of {len(dense)} weights quantized")
    print(f"q8 vs dense on one real block: rel-RMS {rel:.4g}")

    args.out.mkdir(parents=True, exist_ok=True)
    tensors = {"x": x, "emb": emb, "context": context, "lora": lora,
               "output": out_q8, "outputDense": out_dense}
    blob = bytearray()
    manifest = []
    for name, t in tensors.items():
        flat = t.detach().to(torch.float32).contiguous().reshape(-1)
        manifest.append({"name": name, "shape": list(t.shape), "offset": len(blob) // 4, "length": flat.numel()})
        blob.extend(struct.pack(f"<{flat.numel()}f", *flat.tolist()))

    (args.out / "block.bin").write_bytes(bytes(blob))
    (args.out / "block.manifest.json").write_text(json.dumps({
        "note": "Generated by examples/anima/tools/gen_block_golden.py from ComfyUI's own block. Do not hand-edit.",
        "torch": torch.__version__,
        "seed": args.seed,
        "config": {"dim": dim, "numHeads": heads, "headDim": head_dim, "contextDim": context_dim,
                   "mlpHidden": mlp_hidden, "adalnLoraDim": adaln_dim,
                   "seq": T * H * W, "contextSeq": args.context, "normEps": 1e-6},
        "quantizedWeights": sorted(quantized),
        "quantizationCost": {"relativeRms": rel},
        "tensors": manifest,
    }, indent=2) + "\n")
    print(f"wrote {args.out}/block.bin ({len(blob) / 1e3:.0f} kB)")


if __name__ == "__main__":
    main()
