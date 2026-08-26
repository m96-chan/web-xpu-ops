#!/usr/bin/env python3
"""One transformer block of MiniMax-H3's DiT, run by diffusers' own implementation.

Issue #200. The DiT is 50 identical blocks over 5,376 channels — the generator
half of the model, against the visual VAE this repository already decodes with.
A port is right or wrong at the block and the rest is a loop, which is the order
`examples/zimage`, `examples/anima` and `examples/h3-video` were all built in.

`MiniMaxH3TransformerBlock` is **imported, not transcribed**. It lives in
diffusers' main branch (`models/transformers/transformer_minimax_h3.py`) and the
checkpoint ships no code of its own for the DiT, so the file is fetched and its
**relative import lines rewritten to absolute** — nothing else. The arithmetic is
upstream's.

    python examples/h3-dit/tools/gen_block_golden.py \
      --module ~/h3/h3dit_absolute.py --shard ~/h3-work/dit/shard-01.safetensors \
      --out ~/h3-dit-fixtures

**The weights are not this repository's and are not redistributed by it.** Block
0 alone is ~866 MB at bf16; only it is written out. See issue #190.
"""

import argparse
import importlib.util
import json
import pathlib
import sys

import numpy as np
import torch

# `transformer/config.json`, stated rather than read so a checkpoint with a
# different geometry fails the comparison instead of the generator.
CONFIG = {
    "hidden_size": 5376,
    "num_attention_heads": 56,
    "attention_head_dim": 128,
    "ffn_dim": 14336,
    "time_embed_dim": 2688,
    "norm_eps": 1e-5,
    "qk_norm_eps": 1e-5,
    "rope_freq_dim": 16,
    "rope_theta": 10000.0,
    "patch_size": [1, 2, 2],
}


def read_block(path: str, prefix: str) -> dict[str, torch.Tensor]:
    """Reads one block's tensors straight out of the shard, by byte offset.

    `safetensors.safe_open` insists on a complete file. The shard is 5.23 GB and
    **block 0 ends at 1.47 GB of it**, so a download that stalls past that point
    still holds everything this needs -- which is what happened, and refetching
    3.75 GB to satisfy a length check would be bandwidth spent on nothing.

    The header is the format's own: eight little-endian bytes of length, then
    JSON naming each tensor's dtype, shape and `[start, end)` within the data
    that follows. bf16 is widened by hand: the top sixteen bits of an f32 *are*
    a bf16, so the conversion is a shift.
    """
    import struct

    with open(path, "rb") as f:
        (header_len,) = struct.unpack("<Q", f.read(8))
        header = json.loads(f.read(header_len))
        base = 8 + header_len

        out: dict[str, torch.Tensor] = {}
        have = pathlib.Path(path).stat().st_size
        for key, entry in header.items():
            if key == "__metadata__" or not key.startswith(prefix):
                continue
            start, end = entry["data_offsets"]
            if base + end > have:
                raise SystemExit(
                    f"{key} ends at {(base + end) / 1e9:.3f} GB and the file holds {have / 1e9:.3f} GB — "
                    "the download stopped before this block"
                )
            f.seek(base + start)
            raw = f.read(end - start)
            if entry["dtype"] == "BF16":
                wide = np.frombuffer(raw, dtype=np.uint16).astype(np.uint32) << 16
                array = wide.view(np.float32)
            elif entry["dtype"] == "F32":
                array = np.frombuffer(raw, dtype=np.float32)
            else:
                raise SystemExit(f"{key}: unhandled dtype {entry['dtype']}")
            out[key[len(prefix) :]] = torch.from_numpy(array.reshape(entry["shape"]).copy())
    if not out:
        raise SystemExit(f"no tensors under {prefix!r} in {path}")
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--module", required=True, help="upstream's transformer_minimax_h3.py, imports made absolute")
    parser.add_argument("--shard", required=True, help="the safetensors shard holding transformer_blocks.0")
    parser.add_argument("--out", required=True)
    parser.add_argument("--block", type=int, default=0)
    parser.add_argument("--dims", default="1,4,4", help="patch grid T,H,W for the golden")
    parser.add_argument("--temb-scale", default="0.05", help="scale of the random timestep embedding")
    parser.add_argument("--f64", action="store_true", help="run the reference in double precision")
    args = parser.parse_args()

    spec = importlib.util.spec_from_file_location("h3dit", args.module)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    out = pathlib.Path(args.out).expanduser()
    out.mkdir(parents=True, exist_ok=True)

    block = module.MiniMaxH3TransformerBlock(
        hidden_size=CONFIG["hidden_size"],
        num_attention_heads=CONFIG["num_attention_heads"],
        attention_head_dim=CONFIG["attention_head_dim"],
        ffn_dim=CONFIG["ffn_dim"],
        time_embed_dim=CONFIG["time_embed_dim"],
        norm_eps=CONFIG["norm_eps"],
        qk_norm_eps=CONFIG["qk_norm_eps"],
    ).eval()

    prefix = f"transformer_blocks.{args.block}."
    state = read_block(args.shard, prefix)
    missing, unexpected = block.load_state_dict(state, strict=False)
    if unexpected:
        raise SystemExit(f"the shard has tensors this block does not: {unexpected}")
    if missing:
        raise SystemExit(f"the block wants tensors the shard does not have: {missing}")
    block = block.to(torch.float32)

    T, H, W = (int(v) for v in args.dims.split(","))
    seq = T * H * W
    # `(t, h, w)` for every token, which is what the packed sequence carries.
    ids = torch.stack(
        torch.meshgrid(torch.arange(T), torch.arange(H), torch.arange(W), indexing="ij"), dim=-1
    ).reshape(-1, 3).to(torch.float32)
    rope = module.MiniMaxH3RotaryPosEmbed(CONFIG["rope_freq_dim"], CONFIG["rope_theta"])
    cos, sin = rope(ids)

    torch.manual_seed(20260826)
    hidden = torch.randn(1, seq, CONFIG["hidden_size"]) * 0.5
    # One timestep, and every token on the video modality -- which is the case a
    # video-only forward is. The table has `MINIMAX_H3_MODALITY_NUM` rows per
    # timestep and the index picks one.
    # Scaled down deliberately. `time_embedder` is a trained module whose output
    # is order one; a standard normal through `adaln_proj` gives modulation gates
    # of order ten and a block output of order 1e4, where f32 cancellation
    # dominates any comparison and hides the thing being measured.
    temb = torch.randn(1, CONFIG["time_embed_dim"]) * float(args.temb_scale)
    modality = int(getattr(module, "MINIMAX_H3_MODALITY_NUM", 3))
    adaln_indices = torch.full((seq,), 1, dtype=torch.long)

    # `--f64` runs the same block in double precision.
    #
    # Not a shipping mode: it exists because a port that accumulates in
    # JavaScript numbers accumulates in **f64**, and a torch reference in f32
    # differs from it by torch's own rounding rather than by a mistake. A
    # 5,376-term dot product in f32 carries about `sqrt(K) * 2^-24` of relative
    # error, and this block's feed-forward turns inputs of order one into
    # outputs of order a thousand. Comparing against an f64 golden separates the
    # two: if the port matches this and not the f32 one, the port is the
    # accurate half.
    if args.f64:
        block = block.to(torch.float64)
        hidden, temb, cos, sin = (t.to(torch.float64) for t in (hidden, temb, cos, sin))
    with torch.no_grad():
        want = block(hidden, temb, adaln_indices, (cos, sin))
    want = want.to(torch.float32)
    hidden = hidden.to(torch.float32)
    temb = temb.to(torch.float32)

    print(f"block {args.block}: seq {seq}, hidden {CONFIG['hidden_size']}, modalities {modality}")
    print(f"  in {tuple(hidden.shape)} -> out {tuple(want.shape)}")
    print(f"  out[0,:4,0] {[round(v, 6) for v in want[0, :4, 0].tolist()]}")

    order = [
        "norm1.weight", "attn.to_q.weight", "attn.to_k.weight", "attn.to_v.weight",
        "attn.norm_q.weight", "attn.norm_k.weight", "attn.to_out.0.weight",
        "norm2.weight", "ff.net.0.proj.weight", "ff.net.2.weight",
        "adaln_proj.linear.weight", "adaln_proj.linear.bias",
    ]
    entries, offset = [], 0
    with (out / "block.bin").open("wb") as sink:
        for name in order:
            tensor = state[name].to(torch.float32).contiguous()
            array = tensor.numpy().ravel()
            entries.append({"name": name, "shape": list(tensor.shape), "offset": offset, "count": int(array.size)})
            sink.write(array.tobytes())
            offset += int(array.size)

    (out / "block.manifest.json").write_text(json.dumps({
        "model": "minimax-h3-dit-block",
        "source": "MiniMaxAI/MiniMax-H3 (transformer), diffusers MiniMaxH3TransformerBlock",
        "licence": "MiniMax H3 Community License Agreement — not this repository's, and not redistributed by it",
        "block": args.block, "config": CONFIG, "patchDims": [T, H, W], "seq": seq,
        "modalityNum": modality, "adalnIndex": 1,
        "torch": torch.__version__, "tensors": entries, "elements": offset,
    }, indent=1))

    hidden.numpy().astype(np.float32).tofile(out / "hidden.bin")
    temb.numpy().astype(np.float32).tofile(out / "temb.bin")
    want.numpy().astype(np.float32).tofile(out / "want.bin")
    ids.numpy().astype(np.float32).tofile(out / "positions.bin")
    print(f"  wrote {out}/block.bin  {offset * 4 / 1e6:.1f} MB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
