#!/usr/bin/env python3
"""Bakes a golden for Anima's whole DiT forward, from ComfyUI's own model.

Issue #170's second stage. `gen_block_golden.py` covers one block; this covers
everything around it, which is where the conventions live: patchify with its
padding-mask channel, the timestep embedding, the three-axis RoPE, the final
layer, and the `llm_adapter` that ships inside the DiT checkpoint.

**The configuration is not written here.** `comfy/model_detection.py:837` reads
it out of the state dict — number of blocks, heads, channels, the adaLN LoRA
rank, and the RoPE extrapolation ratios — and that function is called rather
than its results copied. A config transcribed by hand is a second thing to keep
in sync with the model, and the first place a port silently diverges.

Two outputs, as everywhere in this repository: the model with its own weights,
and the model with those weights put through q8 exactly as `convert_dit.py`
writes them. A port compared against one number cannot tell a porting mistake
from what the format costs.

    PYTHONPATH=/tmp/comfy-venv/lib/python3.12/site-packages \\
        /home/m96-chan/project/therdparty/musubi-tuner/.venv/bin/python \\
        examples/anima/tools/gen_forward_golden.py --src <checkpoint>
"""
from __future__ import annotations

import argparse
import json
import struct
import sys
from pathlib import Path

import numpy as np
import torch

COMFY = Path("/tmp/ComfyUI")
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from convert_dit import pack_q8, quantize_q8, should_quantize  # noqa: E402
from gen_block_golden import dequantize_q8  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=HERE.parent / "fixtures")
    ap.add_argument("--latent", type=int, default=16, help="latent H and W; 16 gives 8x8 = 64 image tokens")
    ap.add_argument("--context", type=int, default=8, help="context tokens from the adapter")
    ap.add_argument("--context-nonzero", type=int, default=None,
                    help="how many of --context rows are real; the rest are zeros, as "
                         "preprocess_text_embeds pads to 512. Defaults to all of them.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--t", type=float, default=0.7, help="the diffusion time; sampling starts at 1.0")
    ap.add_argument("--context-file", type=Path, default=None,
                    help="a raw f32 [N, crossattn_emb_channels] context instead of random values. "
                         "A real prompt's context is a hundredth the magnitude of randn and 3%% dense, "
                         "and the two are not interchangeable for finding where a port diverges.")
    args = ap.parse_args()

    if not COMFY.exists():
        sys.exit(f"{COMFY} not found — see gen_block_golden.py for the clone command")
    sys.path.insert(0, str(COMFY))

    import comfy.ops
    from comfy import model_detection
    from comfy.ldm.anima.model import Anima
    from safetensors import safe_open

    torch.set_grad_enabled(False)
    torch.manual_seed(args.seed)

    with safe_open(args.src, framework="pt") as f:
        keys = sorted(f.keys())
        # Loaded once and reused: `detect_unet_config` reads shapes off some of
        # these, and the model wants all of them. Two passes over a 7.5 GB file
        # to save holding it briefly would be the wrong trade.
        state = {k[len("net."):]: f.get_tensor(k).to(torch.float32) for k in keys}

    # The real tensors, under the prefix the detector expects. Passing `None`
    # for the ones it does not read fails on the ones it does — it measures
    # shapes, not just membership.
    config = model_detection.detect_unet_config({f"net.{k}": v for k, v in state.items()}, "net.")

    config.pop("image_model", None)
    print("config from the checkpoint:", json.dumps({k: v for k, v in sorted(config.items())}, default=str))

    # ComfyUI's own ops, not `torch.nn`: `cast_bias_weight` reads a `bias`
    # attribute off every norm, and `torch.nn.RMSNorm` has none.
    model = (
        Anima(**config, device=None, dtype=torch.float32, operations=comfy.ops.disable_weight_init)
        .eval()
        .to(torch.float32)
    )
    missing, unexpected = model.load_state_dict(state, strict=False)
    # The three `pos_embedder` entries are `register_buffer(..., persistent=
    # False)` values that happen to have been saved — frequency tables the model
    # recomputes from the config. #169 identified them; allowed by name rather
    # than by loosening the check, so a genuinely unexpected tensor still fails.
    BUFFERS = {"pos_embedder.dim_spatial_range", "pos_embedder.dim_temporal_range", "pos_embedder.seq"}
    surprises = [n for n in unexpected if n not in BUFFERS]
    if missing or surprises:
        sys.exit(f"state dict mismatch: missing {missing[:5]}, unexpected {surprises[:5]}")

    latent = args.latent
    in_channels = config["in_channels"]
    x = torch.randn(1, in_channels, 1, latent, latent)
    t = torch.tensor([args.t])
    context = torch.randn(1, args.context, config["crossattn_emb_channels"])
    # `preprocess_text_embeds` pads a short prompt to 512 rows of zeros and the
    # DiT cross-attends to all of them unmasked (`predict2.py:166` passes
    # `attn_mask=None`). A golden whose context is dense never exercises that,
    # and a real prompt is 3% dense — so the dilution the padding causes is
    # untested by every fixture that came before this flag.
    if args.context_nonzero is not None:
        context[:, args.context_nonzero:, :] = 0
    if args.context_file is not None:
        raw = np.fromfile(args.context_file, dtype=np.float32)
        channels = config["crossattn_emb_channels"]
        context = torch.from_numpy(raw.reshape(1, -1, channels).copy())
        print(f"context from {args.context_file}: {tuple(context.shape)}, "
              f"{float((context != 0).float().mean()) * 100:.1f}% non-zero, std {float(context.std()):.4f}")

    captured: dict[str, torch.Tensor] = {}
    handles = [
        model.blocks[0].register_forward_hook(lambda _m, _i, o: captured.__setitem__("afterBlock0", o)),
        model.blocks[-1].register_forward_hook(lambda _m, _i, o: captured.__setitem__("afterBlocks", o)),
        # `t_embedder` is an `nn.Sequential` the model never calls as a whole —
        # `_forward` reaches into `t_embedder[1](t_embedder[0](...))`, so a hook
        # on the container never fires. Hooking the inner module is what
        # captures anything, and `t_embedding_norm` is what the blocks actually
        # read (`predict2.py:836`).
        model.t_embedding_norm.register_forward_hook(lambda _m, _i, o: captured.__setitem__("tEmbed", o)),
    ]
    out_dense = model(x, t, context)
    dense_trace = {f"{k}Dense": v for k, v in captured.items()}

    print("quantizing in place for the second run ...")
    quantized = 0
    for name, param in model.named_parameters():
        arr = param.data.numpy()
        if not should_quantize(f"net.{name}", arr.shape):
            continue
        codes, scale = quantize_q8(arr)
        param.data.copy_(torch.from_numpy(dequantize_q8(pack_q8(codes), scale, arr.shape)))
        quantized += 1
    print(f"  {quantized} tensors")

    captured.clear()
    out_q8 = model(x, t, context)
    for handle in handles:
        handle.remove()

    cost = ((out_q8 - out_dense).pow(2).mean().sqrt() / out_dense.pow(2).mean().sqrt()).item()
    print(f"quantization alone, over the whole forward: rel-RMS {cost:.4g}")

    args.out.mkdir(parents=True, exist_ok=True)
    tensors = {"x": x, "t": t, "context": context, "output": out_q8, "outputDense": out_dense}
    tensors.update(captured)
    tensors.update(dense_trace)

    blob = bytearray()
    manifest = []
    for name, tensor in tensors.items():
        flat = tensor.detach().to(torch.float32).contiguous().reshape(-1)
        manifest.append({"name": name, "shape": list(tensor.shape), "offset": len(blob) // 4, "length": flat.numel()})
        blob.extend(struct.pack(f"<{flat.numel()}f", *flat.tolist()))

    (args.out / "forward.bin").write_bytes(bytes(blob))
    (args.out / "forward.manifest.json").write_text(json.dumps({
        "note": "Generated by examples/anima/tools/gen_forward_golden.py from ComfyUI's own model. Do not hand-edit.",
        "torch": torch.__version__,
        "seed": args.seed,
        "config": {k: v for k, v in sorted(config.items()) if not callable(v)},
        "latent": latent,
        "contextSeq": args.context,
        "quantizationCost": {"relativeRms": cost},
        "tensors": manifest,
    }, indent=2, default=str) + "\n")

    print(f"wrote {args.out}/forward.bin ({len(blob) / 1e6:.2f} MB)")
    for name, tensor in tensors.items():
        print(f"  {name:22s} {tuple(tensor.shape)}")


if __name__ == "__main__":
    main()
