#!/usr/bin/env python3
"""Bakes a golden for Wan 2.1's VAE decoder, from ComfyUI's own.

Issue #174. Anima decodes with `comfy/ldm/wan/vae.py` — a 3D causal VAE that
shares nothing with `examples/zimage-vae`'s 2D one. That sounds like it needs
`conv3d`, a frame cache and temporal upsampling, and for video it does.

**For one frame it needs none of them, and this is a derivation from ComfyUI's
own control flow rather than an approximation.** With `T = 1`:

  * `WanVAE.decode` computes `iter_ = 1 + z.shape[2] // 2` = 1, so `feat_map`
    stays `None` and no cache exists.
  * `CausalConv3d.forward` takes its fast path when `x.shape[2] == 1`, calling
    `super().forward(x, autopad="causal_zero")`, and `comfy/ops.py:613` is
    `weight = weight[:, :, -input.shape[2]:, :, :]` — the **last temporal tap
    alone**. Temporal padding was already removed in `__init__`. So each one is
    exactly a `Conv2d` over `weight[:, :, -1]`.
  * `Resample`'s `upsample3d` branch runs entirely inside `if feat_cache is not
    None`, so `time_conv` is never called. Its 768x384x3x1x1 weight is dead
    weight for a still image.

What is left is Conv2d, an RMS norm over channels, SiLU, nearest 2x upsampling
and one single-head attention. **`nearest-exact` and `nearest` are asserted to
agree here rather than assumed**: they differ in general (5 to 8 differs) and
agree at every integer scale factor, which is all this decoder uses.

Checkpoints are recorded stage by stage. A decoder reported only at its output
says that something is wrong and nothing about where, and this one is 108
tensors deep.

    PYTHONPATH=/tmp/comfy-venv/lib/python3.12/site-packages \\
        /tmp/comfy-venv/bin/python examples/anima/tools/gen_vae_golden.py \\
        --vae /home/m96-chan/anima-src/qwen_image_vae.safetensors
"""
from __future__ import annotations

import argparse
import json
import struct
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

COMFY = Path("/tmp/ComfyUI")
HERE = Path(__file__).resolve().parent


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--vae", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=HERE.parent / "fixtures")
    ap.add_argument("--latent", type=int, default=8, help="latent H and W; 8 gives a 64x64 image")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    if not COMFY.exists():
        sys.exit(f"{COMFY} not found — see gen_block_golden.py for the clone command")
    sys.path.insert(0, str(COMFY))

    from comfy.ldm.wan.vae import WanVAE
    from safetensors import safe_open

    torch.set_grad_enabled(False)
    torch.manual_seed(args.seed)

    with safe_open(args.vae, framework="pt") as f:
        state = {k: f.get_tensor(k).to(torch.float32) for k in f.keys()}

    # The shape, read off the tensors rather than transcribed. `dim` is the
    # head's output width, `z_dim` the latent's, and `dim_mult` follows from
    # the widths the upsample stages step through.
    dim = state["decoder.head.2.weight"].shape[1]
    z_dim = state["conv2.weight"].shape[0]
    first = state["decoder.conv1.weight"].shape[0]
    # Which stages are temporal is read off the checkpoint, not assumed from
    # `WanVAE`'s defaults: a `Resample` is `upsample3d` exactly when it carries
    # a `time_conv`. The default `temperal_downsample=[True, True, False]`
    # reverses to `[False, True, True]` and puts the temporal stages in the
    # wrong places for this checkpoint, which `load_state_dict` catches — but
    # only because it is asked to be strict about names.
    num_res_blocks = 2
    resample_at = [i for i in range(64) if f"decoder.upsamples.{i}.resample.1.weight" in state]
    temperal_upsample = [f"decoder.upsamples.{i}.time_conv.weight" in state for i in resample_at]
    # `WanVAE` takes the *downsample* flags and reverses them.
    config = {
        "dim": dim, "z_dim": z_dim, "dim_mult": [1, 2, 4, 4],
        "num_res_blocks": num_res_blocks, "attn_scales": [],
        "temperal_downsample": temperal_upsample[::-1],
    }
    print(f"  resample stages at {resample_at}, temporal {temperal_upsample}")
    if dim * config["dim_mult"][-1] != first:
        sys.exit(f"decoder.conv1 gives {first} channels, but dim {dim} x dim_mult[-1] is "
                 f"{dim * config['dim_mult'][-1]} — the assumed dim_mult is wrong for this checkpoint")
    print(f"VAE: dim {dim}, z_dim {z_dim}, dim_mult {config['dim_mult']}")

    vae = WanVAE(**config, image_channels=3, conv_out_channels=3).eval().to(torch.float32)
    missing, unexpected = vae.load_state_dict(state, strict=False)
    surprises = [n for n in unexpected if not n.startswith("encoder.") and n not in ("conv1.weight", "conv1.bias")]
    if [n for n in missing if not n.startswith("encoder.")] or surprises:
        sys.exit(f"state dict mismatch: missing {missing[:5]}, unexpected {surprises[:5]}")

    # `nearest-exact` against `nearest`, at the scales this decoder actually
    # uses. They are not the same resampler — `5 -> 8` differs — but every
    # `Resample` here is an exact doubling, and at an integer factor they agree.
    # Checked rather than assumed, because the port uses `ops/upsample`'s
    # `nearest` and would be silently blurry-in-the-wrong-way otherwise.
    size = args.latent
    for _ in range(3):
        probe = torch.randn(1, 2, size, size)
        exact = F.interpolate(probe, scale_factor=(2.0, 2.0), mode="nearest-exact")
        plain = F.interpolate(probe, size=(size * 2, size * 2), mode="nearest")
        if not torch.equal(exact, plain):
            sys.exit(f"nearest-exact and nearest disagree at {size} -> {size * 2}; the port cannot use ops/upsample")
        size *= 2
    print(f"nearest-exact == nearest at {args.latent} -> {size}, every doubling")

    z = torch.randn(1, z_dim, 1, args.latent, args.latent)

    captured: dict[str, torch.Tensor] = {}
    handles = [
        vae.decoder.conv1.register_forward_hook(lambda _m, _i, o: captured.__setitem__("afterConv1", o)),
        # `middle[2]`, not `middle`. `Decoder3d.forward` iterates
        # `for layer in self.middle:` and never calls the `nn.Sequential` as a
        # whole, so a hook on the container never fires — the same trap the
        # DiT's `t_embedder` set, and it cost a silently absent checkpoint there
        # before it was noticed here.
        vae.decoder.middle[1].register_forward_hook(lambda _m, _i, o: captured.__setitem__("afterAttention", o)),
        vae.decoder.middle[2].register_forward_hook(lambda _m, _i, o: captured.__setitem__("afterMiddle", o)),
        # The output of each `Resample`, which is where the spatial size changes
        # and where an off-by-one in the padding first shows.
        *[
            layer.register_forward_hook(
                lambda _m, _i, o, name=f"afterUpsample{i}": captured.__setitem__(name, o)
            )
            for i, layer in enumerate(vae.decoder.upsamples)
            if type(layer).__name__ == "Resample"
        ],
    ]
    image = vae.decode(z)
    for handle in handles:
        handle.remove()

    print(f"decoded {tuple(z.shape)} to {tuple(image.shape)}")
    print(f"  image min {float(image.min()):+.3f} max {float(image.max()):+.3f} "
          f"mean {float(image.mean()):+.3f} std {float(image.std()):.3f}")

    args.out.mkdir(parents=True, exist_ok=True)
    tensors: dict[str, torch.Tensor] = {"z": z, "image": image}
    tensors.update({k: v for k, v in sorted(captured.items())})

    blob = bytearray()
    entries = []
    for name, tensor in tensors.items():
        arr = tensor.detach().cpu().numpy().astype(np.float32)
        entries.append({"name": name, "shape": list(arr.shape), "offset": len(blob)})
        blob.extend(arr.tobytes())

    header = json.dumps({
        "vae": args.vae.name, "latent": args.latent, "seed": args.seed,
        "dim": dim, "zDim": z_dim, "dimMult": config["dim_mult"],
        "numResBlocks": config["num_res_blocks"], "tensors": entries,
    }).encode()
    out = args.out / "vae-golden.bin"
    with out.open("wb") as f:
        f.write(struct.pack("<Q", len(header)))
        f.write(header)
        f.write(blob)
    print(f"wrote {out} ({out.stat().st_size / 1e6:.1f} MB)")
    if "afterMiddle" not in captured:
        sys.exit("no afterMiddle checkpoint — a hook did not fire, and a golden with a "
                 "missing stage passes a port that never computes it")
    for e in entries:
        print(f"  {e['name']:20} {tuple(e['shape'])}")


if __name__ == "__main__":
    main()
