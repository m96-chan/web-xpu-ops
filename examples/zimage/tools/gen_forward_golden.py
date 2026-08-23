#!/usr/bin/env python3
"""Bakes a golden for the **whole** DiT forward, from Z-Image's own model class.

`gen_real_block_golden.py` covers one layer. This covers everything around it,
which is where the conventions live and therefore where a port goes wrong:
patchify's axis order, the timestep embedding's `cos`-before-`sin`, the caption
pad token, the position ids that start after the caption, the final layer's
LayerNorm — none of which a stack of correct blocks would catch.

The model is instantiated from its own config and run at f32, so the golden is
not limited by bf16. That costs 24.6 GB of RAM for the weights, which is why
this is a generator and not a test.

**Small on purpose.** A 16x16 latent gives 8x8 = 64 image tokens; the shipped
resolution would give thousands, and the port under test is CPU reference ops.
The token count is the thing that scales, and none of the conventions above
depend on it. What it does NOT cover is stated in the README rather than
implied.

Intermediates are dumped alongside the output — after the noise refiner, after
the context refiner, before the final layer. A port that is wrong somewhere in
a 32-layer stack and only compares the last tensor gets to find out that it is
wrong, and nothing about where.

    /home/m96-chan/project/therdparty/musubi-tuner/.venv/bin/python \\
        examples/zimage/tools/gen_forward_golden.py
"""
from __future__ import annotations

import argparse
import json
import struct
import sys
from pathlib import Path

import torch

MUSUBI = Path("/home/m96-chan/project/therdparty/musubi-tuner/src")
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent.parent / "zimage-vae" / "tools"))

from convert_dit import is_q8, pack_q4, pack_q8, quantize_q4_g128, quantize_q8, should_quantize  # noqa: E402
from gen_real_block_golden import dequantize, dequantize_q8  # noqa: E402
from models import add_argument, resolve  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=HERE.parent / "fixtures")
    ap.add_argument("--latent", type=int, default=16, help="latent H and W; 16 gives 8x8 = 64 image tokens")
    ap.add_argument("--cap-len", type=int, default=6, help="caption tokens, of which the last is padding")
    ap.add_argument("--seed", type=int, default=0)
    add_argument(ap)
    args = ap.parse_args()

    if not MUSUBI.exists():
        sys.exit(f"musubi-tuner not found at {MUSUBI}")
    sys.path.insert(0, str(MUSUBI))
    from musubi_tuner.zimage import zimage_model as zm
    from safetensors.torch import load_file

    src = Path(resolve("transformer", args.model_dir))
    config = json.loads((src / "config.json").read_text())
    index = json.loads((src / "diffusion_pytorch_model.safetensors.index.json").read_text())

    torch.set_grad_enabled(False)
    torch.manual_seed(args.seed)

    # The config carries keys the constructor does not take (`siglip_feat_dim`),
    # so it is filtered against the signature rather than splatted — an
    # unexpected keyword here would be a TypeError, but a *missing* one would
    # silently build a differently-shaped model.
    import inspect

    accepted = set(inspect.signature(zm.ZImageTransformer2DModel.__init__).parameters)
    kwargs = {k: v for k, v in config.items() if k in accepted}
    for key in ("all_patch_size", "all_f_patch_size"):
        kwargs[key] = tuple(kwargs[key])
    model = zm.ZImageTransformer2DModel(**kwargs).eval().to(torch.float32)

    print(f"loading {len(set(index['weight_map'].values()))} shards as f32 ...")
    state: dict[str, torch.Tensor] = {}
    for shard in sorted(set(index["weight_map"].values())):
        for name, tensor in load_file(str(src / shard)).items():
            state[name] = tensor.to(torch.float32)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        sys.exit(f"state dict mismatch: missing {missing[:5]}, unexpected {unexpected[:5]}")
    del state

    latent = args.latent
    x = torch.randn(1, config["in_channels"], 1, latent, latent)
    t = torch.tensor([0.7])
    cap_feats = torch.randn(1, args.cap_len, config["cap_feat_dim"])
    # The last caption token is padding, so the `cap_pad_token` path is exercised.
    # An all-true mask would let a port that ignored it pass.
    cap_mask = torch.ones(1, args.cap_len, dtype=torch.bool)
    cap_mask[0, -1] = False

    captured: dict[str, torch.Tensor] = {}

    # Hooks rather than a reimplemented forward: the intermediates have to come
    # from the model's own execution, or they are a second thing to keep in
    # sync with it.
    handles = [
        model.noise_refiner[-1].register_forward_hook(lambda _m, _i, o: captured.__setitem__("afterNoiseRefiner", o)),
        model.context_refiner[-1].register_forward_hook(lambda _m, _i, o: captured.__setitem__("afterContextRefiner", o)),
        model.layers[0].register_forward_hook(lambda _m, _i, o: captured.__setitem__("afterLayer0", o)),
        model.layers[-1].register_forward_hook(lambda _m, _i, o: captured.__setitem__("afterLayers", o)),
        model.t_embedder.register_forward_hook(lambda _m, _i, o: captured.__setitem__("adalnInput", o)),
    ]
    patch = config["all_patch_size"][0]
    out = model(x, t, cap_feats, cap_mask, patch_size=patch, f_patch_size=1)
    dense_trace = {f"{k}Dense": v for k, v in captured.items()}

    # The same forward with every weight the converter quantizes put through the
    # format and back. Without this the port could only be compared against the
    # full-precision model, and a porting mistake would be indistinguishable
    # from what 4-bit costs — which on one layer is 1.8e-2, far above any bar
    # worth setting. Stage 1 made the same split for the same reason.
    print("quantizing in place for the second run ...")
    quantized = 0
    for name, param in model.named_parameters():
        arr = param.data.numpy()
        if not should_quantize(name, arr.shape):
            continue
        if is_q8(name):
            codes, scale = quantize_q8(arr)
            param.data.copy_(torch.from_numpy(dequantize_q8(pack_q8(codes), scale, arr.shape)))
        else:
            codes, scale = quantize_q4_g128(arr)
            param.data.copy_(torch.from_numpy(dequantize(pack_q4(codes), scale, arr.shape)))
        quantized += 1
    print(f"  {quantized} tensors")

    captured.clear()
    out_q = model(x, t, cap_feats, cap_mask, patch_size=patch, f_patch_size=1)
    for handle in handles:
        handle.remove()

    cost = ((out_q - out).pow(2).mean().sqrt() / out.pow(2).mean().sqrt()).item()
    print(f"quantization alone, over the whole forward: rel-RMS {cost:.4g}")

    args.out.mkdir(parents=True, exist_ok=True)
    tensors = {
        "x": x, "t": t, "capFeats": cap_feats, "capMask": cap_mask.to(torch.float32),
        "output": out_q, "outputDense": out,
    }
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
        "note": "Generated by tools/gen_forward_golden.py from the shipped checkpoint, at f32. Do not hand-edit.",
        "torch": torch.__version__,
        "seed": args.seed,
        "config": {
            **{k: config[k] for k in sorted(config) if not k.startswith("_")},
            "patchSize": config["all_patch_size"][0],
            "latent": latent,
            "capLen": args.cap_len,
            "tScale": model.t_scale,
            "frequencyEmbeddingSize": zm.FREQUENCY_EMBEDDING_SIZE,
            "maxPeriod": zm.MAX_PERIOD,
            "adalnEmbedDim": zm.ADALN_EMBED_DIM,
            "ropeAxesDims": zm.ROPE_AXES_DIMS,
            "ropeAxesLens": zm.ROPE_AXES_LENS,
            "ropeTheta": zm.ROPE_THETA,
        },
        "quantizationCost": {"relativeRms": cost},
        "tensors": manifest,
    }, indent=2) + "\n")

    print(f"wrote {args.out}/forward.bin ({len(blob) / 1e6:.2f} MB)")
    for name, tensor in tensors.items():
        print(f"  {name:20s} {tuple(tensor.shape)}")
    print(f"output[0,0,0,0,:4] = {out[0, 0, 0, 0, :4].tolist()}")


if __name__ == "__main__":
    main()
