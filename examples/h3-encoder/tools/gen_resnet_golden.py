#!/usr/bin/env python3
"""One `ResnetBlock3D` of MiniMax-H3's visual VAE **encoder**, run by its own code.

Issue #200. The decoder is a ViT and needs no convolutions; the encoder is where
`ops/conv`'s 3D entry and `ops/pad` are actually used. An op with no caller is a
liability, and this is the caller.

The block is the unit the encoder repeats — two per level, six levels — so a
port is right or wrong here in the same way the decoder's transformer block was.

`ResnetBlock3D` is **imported** from the bundle the checkpoint ships.

    python examples/h3-encoder/tools/gen_resnet_golden.py \
      --bundle ~/h3/video_vae --weights ~/h3/video_vae/model.safetensors \
      --out ~/h3-encoder-fixtures

**The weights are not this repository's and are not redistributed by it.** One
block is about 3.6 MB. See issue #190.
"""

import argparse
import json
import pathlib
import struct
import sys

import numpy as np
import torch

# `source/config.json`, stated rather than read.
CONFIG = {
    "ch": 128,
    "ch_mult": [1, 2, 2, 4, 4, 8],
    "num_res_blocks": 2,
    "padding_mode": "reflect",
    "causal_encoder": True,
    "use_t_isolated_gn": True,
    "groups": 32,
    "norm_eps": 1e-6,
}


def read_prefix(path: str, prefix: str) -> dict[str, torch.Tensor]:
    """The tensors under `prefix`, by byte offset — see `gen_block_golden.py`."""
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
            out[key[len(prefix) :]] = torch.from_numpy(array.reshape(entry["shape"]).copy())
    if not out:
        raise SystemExit(f"no tensors under {prefix!r}")
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", required=True)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--level", type=int, default=0)
    parser.add_argument("--block", type=int, default=0)
    parser.add_argument("--dims", default="3,8,8", help="input T,H,W for the golden")
    args = parser.parse_args()

    bundle = pathlib.Path(args.bundle).expanduser().resolve()
    sys.path.insert(0, str(bundle.parent))
    from video_vae.vae_cnn import ResnetBlock3D  # type: ignore[import-not-found]

    out = pathlib.Path(args.out).expanduser()
    out.mkdir(parents=True, exist_ok=True)

    prefix = f"encoder.down.{args.level}.block.{args.block}."
    state = read_prefix(args.weights, prefix)
    in_channels = state["conv1.weight"].shape[1]
    out_channels = state["conv1.weight"].shape[0]

    block = ResnetBlock3D(
        in_channels=in_channels,
        out_channels=out_channels,
        padding_mode=CONFIG["padding_mode"],
        causal=CONFIG["causal_encoder"],
        use_t_isolated_gn=CONFIG["use_t_isolated_gn"],
    ).eval()
    missing, unexpected = block.load_state_dict(state, strict=False)
    if unexpected:
        raise SystemExit(f"the checkpoint has tensors this block does not: {unexpected}")
    if missing:
        raise SystemExit(f"the block wants tensors the checkpoint does not have: {missing}")

    T, H, W = (int(v) for v in args.dims.split(","))
    torch.manual_seed(20260826)
    x = torch.randn(1, in_channels, T, H, W) * 0.5
    with torch.no_grad():
        want = block(x)

    print(f"encoder level {args.level} block {args.block}: {in_channels} -> {out_channels}")
    print(f"  in {tuple(x.shape)} -> out {tuple(want.shape)}")
    print(f"  out[0,0,0,0,:4] {[round(v, 6) for v in want[0, 0, 0, 0, :4].tolist()]}")

    order = ["norm1.weight", "norm1.bias", "conv1.weight", "conv1.bias",
             "norm2.weight", "norm2.bias", "conv2.weight", "conv2.bias"]
    if in_channels != out_channels:
        order += ["nin_shortcut.weight", "nin_shortcut.bias"]
    entries, offset = [], 0
    with (out / "resnet.bin").open("wb") as sink:
        for name in order:
            tensor = state[name].to(torch.float32).contiguous()
            array = tensor.numpy().ravel()
            entries.append({"name": name, "shape": list(tensor.shape), "offset": offset, "count": int(array.size)})
            sink.write(array.tobytes())
            offset += int(array.size)

    (out / "resnet.manifest.json").write_text(json.dumps({
        "model": "minimax-h3-video-vae-encoder-resnet",
        "source": "MiniMaxAI/MiniMax-H3 (FL2VA/video_vae/source)",
        "licence": "MiniMax H3 Community License Agreement — not this repository's, and not redistributed by it",
        "level": args.level, "block": args.block,
        "inChannels": in_channels, "outChannels": out_channels,
        "dims": [T, H, W], "config": CONFIG,
        "torch": torch.__version__, "tensors": entries, "elements": offset,
    }, indent=1))
    x.numpy().astype(np.float32).tofile(out / "resnet-input.bin")
    want.numpy().astype(np.float32).tofile(out / "resnet-want.bin")
    print(f"  wrote {out}/resnet.bin  {offset * 4 / 1e6:.1f} MB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
