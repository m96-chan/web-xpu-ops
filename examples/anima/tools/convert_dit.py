#!/usr/bin/env python3
"""Converts Anima-3.8B's DiT to the q8 format this library reads.

The same job `examples/zimage/tools/convert_dit.py` does, and a separate file
because almost nothing about *which* tensors it applies to carries over. Anima
names its weights `net.blocks.N.*` and has three adaLN modulations per block
where Z-Image has one; sharing the converter would mean a suffix list with two
models' conventions in it and a reader who cannot tell which line is for which.

The format is shared, though — `ops/matvec`'s q8: per-row absmax, `[-127, 127]`,
four codes per `u32` little-endian, one f32 scale per row. `examples/zimage`
measured what it costs on a real model (2.8% relative RMS over 34 layers, and
0.21 for q4, which is why q4 is not offered here) but **that number does not
transfer**: Anima has 52 blocks and a different block, and #166's own lesson was
that per-layer error compounds in a way one layer does not predict. What q8
costs Anima is unmeasured.

    /home/m96-chan/project/therdparty/musubi-tuner/.venv/bin/python \\
        examples/anima/tools/convert_dit.py --out ~/anima-q8
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open

REPO = "lylogummy/Anima-3.8B"
DIT = "difussion_models/Anima-3.8B.safetensors"

# Quantized: every 2D projection wide enough for a per-row scale to average over.
# Not a suffix list, because Anima's names are regular in a way Z-Image's are
# not — every Linear in a block ends `.weight` and is 2D, and everything else
# (norms, the pos_embedder's buffers, the pad tokens) is 1D.
MIN_DIM = 128


def should_quantize(name: str, shape: tuple[int, ...]) -> bool:
    """2D and wide on both axes. Everything else stays f32.

    A `[1024]` norm weight quantized per row would carry one scale for one row
    of nearly identical numbers — no compression and a rounding error for
    nothing. The threshold is what makes the rule readable without listing
    every tensor.
    """
    return len(shape) == 2 and min(shape) >= MIN_DIM


def quantize_q8(w: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-row absmax int8 — `matvecQ8`'s convention, scale = absmax/127.

    Rounding is `floor(x) + (frac >= 0.5)` and the reciprocal is formed in f64,
    both matching `quantizeQ4G128`'s decisions (#137) so that the two formats in
    this repository round the same way.
    """
    absmax = np.abs(w.astype(np.float64)).max(axis=1, keepdims=True)
    scale = np.where(absmax == 0, 1.0, absmax / 127.0)
    inverse = np.where(absmax == 0, 0.0, 127.0 / absmax)
    scaled = w.astype(np.float64) * inverse
    floored = np.floor(scaled)
    codes = np.clip(floored + (scaled - floored >= 0.5), -127, 127).astype(np.int8)
    return codes, scale.reshape(-1).astype(np.float32)


def pack_q8(codes: np.ndarray) -> np.ndarray:
    """4 codes per u32, least-significant byte first — `packQ8`'s layout."""
    rows, cols = codes.shape
    words = (cols + 3) // 4
    padded = np.zeros((rows, words * 4), dtype=np.uint32)
    padded[:, :cols] = codes.astype(np.uint32) & 0xFF
    lanes = padded.reshape(rows, words, 4)
    packed = np.zeros((rows, words), dtype=np.uint32)
    for i in range(4):
        packed |= lanes[:, :, i] << (8 * i)
    return packed


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--limit-blocks", type=int, default=None,
                    help="convert only the first N of the 52 blocks, for a fixture that fits")
    ap.add_argument("--src", type=Path, default=None, help="a local checkpoint instead of the hub")
    args = ap.parse_args()

    if args.src:
        path = args.src
    else:
        from huggingface_hub import hf_hub_download
        path = Path(hf_hub_download(REPO, DIT))

    # The model's own shape, read out of the state dict the way ComfyUI reads
    # it. Requires the clone; see `gen_block_golden.py` for the command.
    comfy = Path("/tmp/ComfyUI")
    if not comfy.exists():
        sys.exit(f"{comfy} not found — needed for model_detection; see gen_block_golden.py")
    sys.path.insert(0, str(comfy))
    from comfy import model_detection

    class Shaped:
        """`detect_unet_config` reads `.shape` and `.ndim`; a slice has neither.

        Wrapping is what keeps this from loading 7.5 GB to measure shapes it
        could have read from the header.
        """

        def __init__(self, entry) -> None:
            self.shape = tuple(entry.get_shape())
            self.ndim = len(self.shape)

    with safe_open(path, framework="pt") as f:
        config = model_detection.detect_unet_config({k: Shaped(f.get_slice(k)) for k in f.keys()}, "net.")
    config.pop("image_model", None)
    config = {k: (v.item() if hasattr(v, "item") else v) for k, v in sorted(config.items())}
    print("config from the checkpoint:", json.dumps(config, default=str))

    args.out.mkdir(parents=True, exist_ok=True)
    q8_out = open(args.out / "dit.q8.bin", "wb")
    scales_out = open(args.out / "dit.q8scales.bin", "wb")
    f32_out = open(args.out / "dit.f32.bin", "wb")
    q8_at = scales_at = f32_at = 0
    manifest: list[dict] = []
    stored = original = 0

    with safe_open(path, framework="pt") as f:
        names = sorted(f.keys())
        if args.limit_blocks is not None:
            def keep(name: str) -> bool:
                if not name.startswith("net.blocks."):
                    return True
                return int(name.split(".")[2]) < args.limit_blocks
            names = [n for n in names if keep(n)]

        for name in names:
            arr = f.get_tensor(name).to(torch.float32).numpy()
            original += arr.size * 2  # bf16 on disk

            if should_quantize(name, arr.shape):
                codes, scale = quantize_q8(arr)
                packed = pack_q8(codes)
                q8_out.write(packed.tobytes())
                scales_out.write(scale.tobytes())
                manifest.append({
                    "name": name, "kind": "q8", "shape": list(arr.shape),
                    "codesOffset": q8_at, "scaleOffset": scales_at,
                })
                q8_at += packed.size
                scales_at += scale.size
                stored += packed.nbytes + scale.nbytes
            else:
                flat = arr.reshape(-1).astype(np.float32)
                f32_out.write(flat.tobytes())
                manifest.append({"name": name, "kind": "f32", "shape": list(arr.shape), "offset": f32_at})
                f32_at += flat.size
                stored += flat.nbytes

    for handle in (q8_out, scales_out, f32_out):
        handle.close()

    blocks = max(
        (int(t["name"].split(".")[2]) for t in manifest if t["name"].startswith("net.blocks.")),
        default=-1,
    ) + 1
    (args.out / "dit.manifest.json").write_text(json.dumps({
        "note": "Generated by examples/anima/tools/convert_dit.py. Do not hand-edit.",
        "repo": REPO,
        "format": {"quant": "q8-per-row", "range": [-127, 127], "codesPerWord": 4,
                   "reciprocal": "127/absmax in f64", "rounding": "floor(x) + (frac >= 0.5)"},
        "blocks": blocks,
        # The model's own shape, so the manifest describes what it holds.
        # Without it a runtime has to be told the configuration out of band —
        # `verify-forward-gpu.ts` read it from a *golden*, which is fine for a
        # checker and wrong for anything that generates: the fixture and the
        # weights are then two things that must agree and nothing checks it.
        # Read by `model_detection.detect_unet_config`, never transcribed.
        "config": config,
        "limitBlocks": args.limit_blocks,
        "tensors": manifest,
    }, indent=1) + "\n")

    q = sum(1 for m in manifest if m["kind"] == "q8")
    print(f"{len(manifest)} tensors ({q} q8, {len(manifest) - q} f32), {blocks} blocks")
    print(f"{original / 1e9:.2f} GB bf16 -> {stored / 1e9:.2f} GB ({stored / original * 100:.1f}%)")


if __name__ == "__main__":
    main()
