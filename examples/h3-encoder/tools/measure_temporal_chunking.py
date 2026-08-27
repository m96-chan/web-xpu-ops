"""What the released VAE's temporal chunking does to a conditioning latent.

Issue #216. `examples/h3-encoder` is held to `EncoderFCN3D` + `quant_conv`, which
is `AutoencoderKLLegacy.encode` -- and `gen_resnet_golden.py`'s own docstring
says the chunking path is deliberately not used, calling `clip_length` and
`token_drop` "an inference optimisation for long clips".

**That is the thing being measured here.** `encode_base` does not call `encode`
for a multi-frame input; it calls `encode_temporal`, which pads the clip up to a
multiple of `clip_length`, encodes each 17-frame chunk *independently*, and drops
`token_drop` latent frames off the end. The released checkpoint's own config ships
`vae_clip_length 17`, `vae_token_drop 3`, `vae_encoder_tiling 1`.

For an 8-frame reference the two paths return the same *shape* -- 2 latent frames,
by two different routes -- so nothing downstream complains. Whether they return the
same *numbers* is what this script prints.

    python examples/h3-encoder/tools/measure_temporal_chunking.py \
      --bundle ~/h3-work/src/video_vae --weights ~/h3-work/video-vae-source.safetensors \
      --frames 8,22,17,1
"""

from __future__ import annotations

import argparse
import json
import pathlib
import struct
import sys

import numpy as np
import torch

CONFIG = json.load(
    open(pathlib.Path("~/h3-work/vae-config/Ref2VA/video_vae/source/config.json").expanduser())
)
CLIP_LENGTH = 17
TOKEN_DROP = 3


def read_prefix(path: str, prefix: str) -> dict[str, torch.Tensor]:
    """The tensors under `prefix`, from a safetensors file, without loading the rest."""
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


def build(bundle: pathlib.Path, weights: str):
    sys.path.insert(0, str(bundle.parent))
    from video_vae.vae_cnn import EncoderFCN3D  # type: ignore[import-not-found]

    encoder = EncoderFCN3D(
        ch=CONFIG["ch"], ch_mult=CONFIG["ch_mult"], space_down=CONFIG["space_down"],
        time_down=CONFIG["time_down"], num_res_blocks=CONFIG["num_res_blocks"],
        in_channels=CONFIG["in_channels"], z_channels=CONFIG["z_channels"], double_z=True,
        padding_mode=CONFIG["padding_mode"], causal=CONFIG["causal_encoder"],
        use_t_isolated_gn=CONFIG["use_t_isolated_gn"],
    ).eval()
    missing, unexpected = encoder.load_state_dict(read_prefix(weights, "encoder."), strict=False)
    if unexpected or missing:
        raise SystemExit(f"encoder tensors disagree: missing={missing} unexpected={unexpected}")
    quant = read_prefix(weights, "quant_conv.")
    w, b = quant["weight"], quant["bias"]

    def plain(x: torch.Tensor) -> torch.Tensor:
        """`AutoencoderKLLegacy.encode`: `quant_conv(encoder(x))`. What the port reproduces."""
        return torch.nn.functional.conv3d(encoder(x), w, b)

    def temporal(x: torch.Tensor) -> torch.Tensor:
        """`encode_temporal`, transcribed from `video_vae/klvae.py` lines 461-493.

        `isolated_first_frame`, `isolated_key_frame` and `isolated_last_frame` are
        all absent from the released config and default to False, so the branches
        that depend on them are dropped rather than guessed at.
        """
        if x.shape[2] % CLIP_LENGTH != 0:
            pad = x[:, :, -1:].repeat(1, 1, (-x.shape[2]) % CLIP_LENGTH, 1, 1)
            x = torch.cat([x, pad], dim=2)
        chunks = [
            plain(x[:, :, i * CLIP_LENGTH:(i + 1) * CLIP_LENGTH])
            for i in range(x.shape[2] // CLIP_LENGTH)
        ]
        z = torch.cat(chunks, dim=2)
        return z[:, :, :-TOKEN_DROP] if TOKEN_DROP > 0 else z

    return plain, temporal


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--bundle", required=True)
    p.add_argument("--weights", required=True)
    p.add_argument("--frames", default="1,8,22", help="frame counts to compare")
    p.add_argument("--size", type=int, default=32)
    p.add_argument("--dump", help="write video.bin and conditioning.bin for the LAST frame count, "
                                "so the port can be held to `encode_temporal` itself")
    args = p.parse_args()

    plain, temporal = build(pathlib.Path(args.bundle).expanduser().resolve(), args.weights)
    torch.manual_seed(0)

    print(f"{'frames':>7}  {'plain':>14}  {'temporal':>14}  {'worst':>10}  {'rms diff':>10}  {'rel':>8}")
    for count in [int(n) for n in args.frames.split(",")]:
        x = torch.randn(1, 3, count, args.size, args.size)
        with torch.no_grad():
            a, b = plain(x), temporal(x)
        shape_a = "x".join(str(n) for n in a.shape[1:])
        shape_b = "x".join(str(n) for n in b.shape[1:])
        if a.shape != b.shape:
            print(f"{count:>7}  {shape_a:>14}  {shape_b:>14}  {'different shape':>10}")
            continue
        worst = (a - b).abs().max().item()
        diff = (a - b).pow(2).mean().sqrt().item()
        scale = a.pow(2).mean().sqrt().item()
        print(f"{count:>7}  {shape_a:>14}  {shape_b:>14}  {worst:>10.4f}  {diff:>10.4f}  "
              f"{diff / scale * 100:>7.1f}%")

    if args.dump:
        # The **temporal** side only. The port is held to what the model does,
        # not to the path the port happened to take before.
        out = pathlib.Path(args.dump).expanduser()
        out.mkdir(parents=True, exist_ok=True)
        count = int(args.frames.split(",")[-1])
        x = torch.randn(1, 3, count, args.size, args.size)
        with torch.no_grad():
            z = temporal(x)
        x[0].numpy().astype(np.float32).tofile(out / "video.bin")
        z[0].numpy().astype(np.float32).tofile(out / "conditioning.bin")
        (out / "conditioning.json").write_text(json.dumps({
            "source": "MiniMaxAI/MiniMax-H3 video_vae, `encode_temporal`",
            "note": "Activations only. No weights are redistributed by this file.",
            "video": [count, args.size, args.size],
            "latent": list(z.shape[1:]),
            "clip_length": CLIP_LENGTH, "token_drop": TOKEN_DROP,
            "torch": torch.__version__,
        }, indent=1))
        print(f"wrote {out}/conditioning.bin  {tuple(z.shape[1:])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
