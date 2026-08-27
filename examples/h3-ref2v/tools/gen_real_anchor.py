"""`encode_vae_condition` on a real reference, whole, as the model runs it.

Issue #216. Every golden this example has feeds the transformer a `randn`
anchor, and the loop matches upstream exactly when it does — measured at 0.0%
rms over all fifty blocks. The flicker only appears with *real* references, so
the anchor's **content** is what has never been in a golden.

This writes that content. The chain is `encoders.py#encode_vae_condition`, and
every step of it matters:

    pixels = (u8 / 255 - pixel_mean) / pixel_std      # ImageNet
    posterior = vae.encode(pixels)                    # encode_temporal for a stack
    latents = posterior.sample(generator=seed 42)     # sampled, not the mode
    latents = latents.to(float16).float()             # ~11 bits, explicitly
    return (latents - latents_mean) / latents_std

`vae.encode` is `encode_base`, which sends a single image through the spatial
encoder alone and a frame stack through `encode_temporal`'s independent
17-frame chunks — the thing `59ac5b1` fixed on the port's side.

    python examples/h3-ref2v/tools/gen_real_anchor.py \
      --bundle ~/h3-work/src/video_vae --weights ~/h3-work/video-vae-source.safetensors \
      --config ~/h3-work/vae-config/Ref2VA/video_vae/config.json \
      --pixels ~/h3-work/h3-cond-real/pixels.bin --size 256,256 --frames 1 \
      --out ~/h3-work/h3-anchor-real
"""

from __future__ import annotations

import argparse
import json
import pathlib
import struct
import sys

import numpy as np
import torch

# ImageNet, which is what `pixel_norm_type` names in the released config.
PIXEL_MEAN = (0.485, 0.456, 0.406)
PIXEL_STD = (0.229, 0.224, 0.225)
ENCODE_SEED = 42


def read_prefix(path: str, prefix: str) -> dict[str, torch.Tensor]:
    with open(path, "rb") as f:
        (header_len,) = struct.unpack("<Q", f.read(8))
        header = json.loads(f.read(header_len))
        base = 8 + header_len
        out: dict[str, torch.Tensor] = {}
        for key, entry in header.items():
            if key == "__metadata__" or not key.startswith(prefix):
                continue
            start, end = entry["data_offsets"]
            f.seek(base + start)
            raw = f.read(end - start)
            dtype = {"F32": np.float32, "BF16": np.uint16}[entry["dtype"]]
            array = np.frombuffer(raw, dtype=dtype)
            if entry["dtype"] == "BF16":
                array = (array.astype(np.uint32) << 16).view(np.float32)
            out[key[len(prefix):]] = torch.from_numpy(array.reshape(entry["shape"]).copy())
    if not out:
        raise SystemExit(f"no tensors under {prefix!r}")
    return out


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--bundle", required=True, help="the released `video_vae` package")
    p.add_argument("--weights", required=True)
    p.add_argument("--config", required=True, help="Ref2VA/video_vae/config.json, for the chunking and the scale")
    p.add_argument("--source-config", help="video_vae/source/config.json (defaults to `source/` beside --config)")
    p.add_argument("--pixels", required=True, help="u8 `[frame][row][col][channel]` RGB")
    p.add_argument("--size", required=True, help="W,H")
    p.add_argument("--frames", type=int, default=1)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    wrapper = json.loads(pathlib.Path(args.config).expanduser().read_text())
    source_path = (
        pathlib.Path(args.source_config).expanduser() if args.source_config
        else pathlib.Path(args.config).expanduser().parent / "source" / "config.json"
    )
    cfg = json.loads(source_path.read_text())
    clip_length = wrapper["vae_clip_length"]
    token_drop = wrapper["vae_token_drop"]

    bundle = pathlib.Path(args.bundle).expanduser().resolve()
    sys.path.insert(0, str(bundle.parent))
    from video_vae.vae_cnn import EncoderFCN3D  # type: ignore[import-not-found]

    encoder = EncoderFCN3D(
        ch=cfg["ch"], ch_mult=cfg["ch_mult"], space_down=cfg["space_down"],
        time_down=cfg["time_down"], num_res_blocks=cfg["num_res_blocks"],
        in_channels=cfg["in_channels"], z_channels=cfg["z_channels"], double_z=True,
        padding_mode=cfg["padding_mode"], causal=cfg["causal_encoder"],
        use_t_isolated_gn=cfg["use_t_isolated_gn"],
    ).eval()
    missing, unexpected = encoder.load_state_dict(read_prefix(args.weights, "encoder."), strict=False)
    if missing or unexpected:
        raise SystemExit(f"encoder tensors disagree: missing={missing} unexpected={unexpected}")
    quant = read_prefix(args.weights, "quant_conv.")
    qw, qb = quant["weight"], quant["bias"]

    def plain(x: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.conv3d(encoder(x), qw, qb)

    def encode(x: torch.Tensor) -> torch.Tensor:
        """`encode_base`: the spatial encoder for one image, `encode_temporal` for a stack."""
        if x.shape[2] == 1:
            return plain(x)
        if x.shape[2] % clip_length != 0:
            pad = x[:, :, -1:].repeat(1, 1, (-x.shape[2]) % clip_length, 1, 1)
            x = torch.cat([x, pad], dim=2)
        z = torch.cat(
            [plain(x[:, :, i * clip_length:(i + 1) * clip_length])
             for i in range(x.shape[2] // clip_length)],
            dim=2,
        )
        return z[:, :, :-token_drop] if token_drop > 0 else z

    W, H = (int(n) for n in args.size.split(","))
    raw = np.fromfile(pathlib.Path(args.pixels).expanduser(), dtype=np.uint8)
    want = args.frames * H * W * 3
    if raw.size != want:
        raise SystemExit(f"{args.pixels}: {raw.size} bytes for {args.frames}x{H}x{W}x3 = {want}")
    # `[frame][row][col][channel]` -> `(1, 3, T, H, W)`.
    pixels = torch.from_numpy(raw.reshape(args.frames, H, W, 3).copy()).permute(3, 0, 1, 2)[None]

    mean = torch.tensor(PIXEL_MEAN).view(1, -1, 1, 1, 1)
    std = torch.tensor(PIXEL_STD).view(1, -1, 1, 1, 1)
    x = (pixels.to(torch.float32).div(255.0) - mean) / std

    with torch.no_grad():
        moments = encode(x)
    z_channels = cfg["z_channels"]
    mu, logvar = moments[:, :z_channels], moments[:, z_channels:]
    # `DiagonalGaussianDistribution.sample`: clamped log-variance, and a
    # generator seeded independently of the request.
    logvar = logvar.clamp(-30.0, 20.0)
    noise = torch.empty_like(mu)
    noise.normal_(generator=torch.Generator().manual_seed(ENCODE_SEED))
    latents = mu + torch.exp(0.5 * logvar) * noise
    # **Rounded to float16 and back**, which upstream does explicitly. About
    # eleven bits of every conditioning latent; leaving it out is a conditioning
    # the model was not given.
    latents = latents.to(torch.float16).float()

    latents_mean = torch.tensor(wrapper["latents_mean"]).view(1, -1, 1, 1, 1)
    latents_std = torch.tensor(wrapper["latents_std"]).view(1, -1, 1, 1, 1)
    anchor = (latents - latents_mean) / latents_std

    out = pathlib.Path(args.out).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    anchor[0].numpy().astype("<f4").tofile(out / "anchor.bin")
    (out / "anchor.json").write_text(json.dumps({
        "source": "MiniMaxAI/MiniMax-H3 video_vae, `encode_vae_condition`",
        "note": "Activations only. No weights are redistributed by this file.",
        "pixels": [args.frames, H, W],
        "shape": list(anchor.shape[1:]),
        "encodeSeed": ENCODE_SEED,
        "clipLength": clip_length, "tokenDrop": token_drop,
        "rms": float(anchor.pow(2).mean().sqrt()),
        "torch": torch.__version__,
    }, indent=1))
    print(
        f"{args.frames}x{H}x{W} -> anchor {tuple(anchor.shape[1:])}  "
        f"rms {float(anchor.pow(2).mean().sqrt()):.4f}  -> {out}/anchor.bin"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
