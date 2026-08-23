#!/usr/bin/env python3
"""Encodes an image to a Z-Image latent, and decodes it back, with the model.

Produces three things the browser demo needs:

  - `latent.bin`      the encoder's output, what the demo decodes
  - `reference.png`   the model's own decode of that latent
  - `weights.bin`     the decoder's parameters, in the demo's own layout

The reference decode is the point. Without it the demo shows a picture and
invites you to decide whether it looks right, which is not a check — a decoder
with a transposed convolution or a wrong group count produces something that
still looks like an image. With it, the browser can report a number.

The encoder stays in Python on purpose: `ops/conv` has `conv2d` but nothing
this repository ships does strided downsampling with the encoder's block
structure yet, and the demo is about decoding. That asymmetry is deliberate and
stated rather than hidden.

Run with musubi-tuner's interpreter (it has `diffusers`, `torch` for this CUDA,
and `accelerate`):

    /home/m96-chan/project/therdparty/musubi-tuner/.venv/bin/python \
        examples/zimage-vae/tools/gen_latent.py --image path/to.png
"""
from __future__ import annotations

import argparse
import json
import struct
from pathlib import Path

import torch
from diffusers import AutoencoderKL
from PIL import Image

REPO = "Tongyi-MAI/Z-Image"



def synthetic_chart(size: int) -> Image.Image:
    """A chart whose failure modes are visible rather than a photograph.

    Four bands, chosen so that a decode that is subtly wrong looks wrong rather
    than merely different: saturated primaries (channel order and clipping), a
    grey ramp (gamma and the shift/scale factors), a checkerboard at the latent
    stride (block artefacts land exactly on it), and thin diagonals (the high
    frequency a VAE is expected to lose, so the demo does not read normal loss
    as a bug).
    """
    import numpy as np

    a = np.zeros((size, size, 3), dtype=np.float32)
    q = size // 4
    bars = [(1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 1, 0), (1, 0, 1), (0, 1, 1), (1, 1, 1), (0, 0, 0)]
    for i, c in enumerate(bars):
        x0, x1 = i * size // len(bars), (i + 1) * size // len(bars)
        a[0:q, x0:x1] = c
    ramp = np.linspace(0.0, 1.0, size, dtype=np.float32)[None, :, None]
    a[q:2 * q] = ramp
    yy, xx = np.mgrid[0:q, 0:size]
    a[2 * q:3 * q] = (((yy // 8) + (xx // 8)) % 2).astype(np.float32)[..., None]
    yy, xx = np.mgrid[0:size - 3 * q, 0:size]
    a[3 * q:] = (((xx + yy) % 6) < 2).astype(np.float32)[..., None] * np.array([1.0, 0.6, 0.2], dtype=np.float32)
    return Image.fromarray((a * 255).round().astype("uint8"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", type=Path, default=None,
                    help="source image; omitted uses a synthetic chart (no licence question, and its "
                         "hard edges and fine texture are where a VAE visibly loses information)")
    ap.add_argument("--size", type=int, default=256, help="square crop fed to the encoder")
    ap.add_argument("--out", type=Path, default=Path(__file__).resolve().parent.parent / "fixtures")
    args = ap.parse_args()

    torch.set_grad_enabled(False)
    vae = AutoencoderKL.from_pretrained(REPO, subfolder="vae", torch_dtype=torch.float32).eval()

    if args.image is not None:
        img = Image.open(args.image).convert("RGB")
        side = min(img.size)
        img = img.crop(((img.width - side) // 2, (img.height - side) // 2,
                        (img.width + side) // 2, (img.height + side) // 2)).resize((args.size, args.size), Image.LANCZOS)
    else:
        img = synthetic_chart(args.size)

    x = torch.from_numpy(__import__("numpy").asarray(img)).float().permute(2, 0, 1)[None] / 127.5 - 1.0

    posterior = vae.encode(x).latent_dist
    latent = posterior.mode()
    # The demo decodes the *scaled* latent, which is what a sampler would hand
    # it, so the scaling lives here rather than being an extra step over there.
    scaled = (latent - vae.config.shift_factor) * vae.config.scaling_factor
    unscaled = scaled / vae.config.scaling_factor + vae.config.shift_factor
    decoded = vae.decode(unscaled).sample

    args.out.mkdir(parents=True, exist_ok=True)

    def png(t: torch.Tensor, path: Path) -> None:
        arr = ((t[0].permute(1, 2, 0).clamp(-1, 1) + 1) * 127.5).round().to(torch.uint8).numpy()
        Image.fromarray(arr).save(path)

    png(x, args.out / "input.png")
    png(decoded, args.out / "reference.png")

    blob = bytearray()
    manifest = []

    def put(name: str, t: torch.Tensor) -> None:
        flat = t.detach().to(torch.float32).contiguous().reshape(-1)
        manifest.append({"name": name, "shape": list(t.shape), "offset": len(blob) // 4, "length": flat.numel()})
        blob.extend(struct.pack(f"<{flat.numel()}f", *flat.tolist()))

    put("latent", scaled)
    put("reference", decoded)
    (args.out / "latent.bin").write_bytes(bytes(blob))

    wblob = bytearray()
    wmanifest = []
    for name, p in vae.decoder.named_parameters():
        flat = p.detach().to(torch.float32).contiguous().reshape(-1)
        wmanifest.append({"name": name, "shape": list(p.shape), "offset": len(wblob) // 4, "length": flat.numel()})
        wblob.extend(struct.pack(f"<{flat.numel()}f", *flat.tolist()))
    # diffusers puts a 1x1 `post_quant_conv` before the decoder when the config
    # asks for one. Z-Image's does not (`use_quant_conv` is absent, so it is
    # None here), which is checked rather than assumed — a port that silently
    # skipped a conv the model actually had would be wrong in a way the decode
    # still looks plausible through.
    if vae.post_quant_conv is not None:
        for name, p in vae.post_quant_conv.named_parameters():
            flat = p.detach().to(torch.float32).contiguous().reshape(-1)
            wmanifest.append({"name": f"post_quant_conv.{name}", "shape": list(p.shape), "offset": len(wblob) // 4, "length": flat.numel()})
            wblob.extend(struct.pack(f"<{flat.numel()}f", *flat.tolist()))
    (args.out / "decoder.bin").write_bytes(bytes(wblob))

    (args.out / "manifest.json").write_text(json.dumps({
        "note": "Generated by tools/gen_latent.py. Do not hand-edit.",
        "source": REPO,
        "torch": torch.__version__,
        "image": {"size": args.size},
        "config": {k: vae.config[k] for k in
                   ["block_out_channels", "layers_per_block", "norm_num_groups", "latent_channels",
                    "out_channels", "scaling_factor", "shift_factor", "mid_block_add_attention"]},
        "hasPostQuantConv": False,
        "latent": manifest,
        "decoder": wmanifest,
    }, indent=2) + "\n")

    err = (decoded - x).abs().max().item()
    print(f"latent {tuple(scaled.shape)}  decoder params {len(wmanifest)} tensors, {len(wblob)/1e6:.1f} MB")
    print(f"round-trip worst |decoded - input| = {err:.4f} (VAE loss, not a port error)")


if __name__ == "__main__":
    main()
