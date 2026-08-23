#!/usr/bin/env python3
"""Converts Z-Image's DiT to the q4-g128 format this library reads.

12.31 GB of bf16 does not fit anywhere useful. In the format issue #137 settled
on — group size 128, symmetric [-7, 7], 4 bits per weight plus one f32 scale per
group — this run measured **26.7%** of the bf16 size, so the DiT lands around
3.3 GB. Not 4/16 = 25%, because the per-group scales cost a further 0.25 bits
per weight and the tensors listed below stay f32 entirely.

What gets quantized and what does not, and why:

  - **Linear weights** (`to_q/k/v`, `to_out`, `feed_forward.w1/w2/w3`,
    `cap_embedder`, the final `linear`) go to q4. They are the 12 GB.
  - **`adaLN_modulation` goes to q8 instead**, and this is measured, not a
    hunch. Quantizing one weight of `layers.0` at a time and running the real
    block (`tools/gen_real_block_golden.py`) puts adaLN at 4.78% relative RMS
    on the block's output, against 1.2% for the next worst and 0.4% for the
    best — it alone accounts for nearly all of the 5.19% that quantizing
    everything costs. It is the weight that produces the scales and gates
    multiplying the whole residual stream, so its error is not additive but
    multiplicative across all 3840 channels. Moving just it to q8 costs 2.5 MB
    per layer (96.1 -> 98.6, +2.6%) and takes the block from 5.19% to 1.78%.
    Small groups do not fix it: q4-g32 costs 0.25 more bits everywhere for
    5.19% -> 3.98%, which is worse on both counts.
  - **Norm weights, biases, embedders, pad tokens** stay f32. Together they are
    a rounding error in size, and they are the tensors where a per-group scale
    has nothing to average over — a `[3840]` norm weight quantized in groups of
    128 would carry 30 scales for 30 groups of nearly identical numbers.
  - `x_embedder` is a `Conv2d`-shaped patch embedder and stays f32 for the same
    reason: it is one tensor, and `matmulQ4G128` is not what reads it.

The split is written into the manifest rather than inferred at load time, so a
loader never has to guess which of the two a tensor is.

Run with musubi-tuner's interpreter (it has torch and safetensors):

    /home/m96-chan/project/therdparty/musubi-tuner/.venv/bin/python \\
        examples/zimage/tools/convert_dit.py --out <dir>
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "zimage-vae" / "tools"))
from models import add_argument, resolve  # noqa: E402

GROUP = 128

# Everything else stays f32. Matching on the suffix rather than listing every
# layer keeps this correct when the layer count changes.
QUANTIZED_SUFFIXES = (
    "attention.to_q.weight", "attention.to_k.weight", "attention.to_v.weight",
    "attention.to_out.0.weight",
    "feed_forward.w1.weight", "feed_forward.w2.weight", "feed_forward.w3.weight",
    "adaLN_modulation.0.weight", "adaLN_modulation.1.weight",
    "cap_embedder.1.weight",
    "linear.weight",
)

# The measured exception; see the module docstring for the numbers.
Q8_SUFFIXES = ("adaLN_modulation.0.weight", "adaLN_modulation.1.weight")


def should_quantize(name: str, shape: tuple[int, ...]) -> bool:
    """True if this tensor is quantized at all — q4 or q8. Shared with the golden generator."""
    if len(shape) != 2:
        return False
    if shape[1] % GROUP != 0:
        # A row that does not divide into whole groups would need a partial
        # group, which the format does not have. Left f32 and reported, rather
        # than padded silently into a different set of numbers.
        return False
    return any(name.endswith(suffix) for suffix in QUANTIZED_SUFFIXES)


def is_q8(name: str) -> bool:
    return any(name.endswith(suffix) for suffix in Q8_SUFFIXES)


def quantize_q8(w: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-row absmax int8 — `matvecQ8`'s convention, scale = absmax/127."""
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


def quantize_q4_g128(w: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-group absmax int4, symmetric [-7, 7]. Mirrors ops/matvec's reference.

    The reciprocal is formed as `7 / absmax` in f64 and the rounding is
    round-half-toward-+inf, both to match `quantizeQ4G128` exactly — issue
    #137's decisions, not re-derived here.
    """
    rows, cols = w.shape
    groups = cols // GROUP
    w64 = w.astype(np.float64).reshape(rows, groups, GROUP)
    absmax = np.abs(w64).max(axis=2)
    # An all-zero group stores scale 1, not 0 — `quantizeQ4G128` does the same,
    # so that dequantising it gives 0 rather than 0 * 0 = 0 by a different route
    # and, more to the point, so a consumer dividing by the scale cannot hit a
    # zero. The codes are 0 either way; only this constant differs, which is
    # why a tensor without an all-zero group agrees under both spellings.
    scale = np.where(absmax == 0, 1.0, absmax / 7.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        inverse = np.where(absmax == 0, 0.0, 7.0 / absmax)
    scaled = w64 * inverse[:, :, None]
    floored = np.floor(scaled)
    codes = np.clip(floored + (scaled - floored >= 0.5), -7, 7).astype(np.int8)
    return codes.reshape(rows, cols), scale.astype(np.float32)


def pack_q4(codes: np.ndarray) -> np.ndarray:
    """8 codes per u32, least-significant nibble first — packQ8's byte order at half width."""
    rows, cols = codes.shape
    words = cols // 8
    nibbles = (codes.astype(np.uint32) & 0xF).reshape(rows, words, 8)
    packed = np.zeros((rows, words), dtype=np.uint32)
    for i in range(8):
        packed |= nibbles[:, :, i] << (4 * i)
    return packed


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--limit-layers", type=int, default=None,
                    help="convert only the first N layers, for a fixture small enough to commit")
    add_argument(ap)
    args = ap.parse_args()

    src = Path(resolve("transformer", args.model_dir))
    index = json.loads((src / "diffusion_pytorch_model.safetensors.index.json").read_text())
    config = json.loads((src / "config.json").read_text())

    by_file: dict[str, list[str]] = {}
    for name, file in index["weight_map"].items():
        if args.limit_layers is not None and name.startswith("layers."):
            layer = int(name.split(".")[1])
            if layer >= args.limit_layers:
                continue
        by_file.setdefault(file, []).append(name)

    args.out.mkdir(parents=True, exist_ok=True)
    codes_out = open(args.out / "dit.codes.bin", "wb")
    scales_out = open(args.out / "dit.scales.bin", "wb")
    q8_out = open(args.out / "dit.q8.bin", "wb")
    q8_scales_out = open(args.out / "dit.q8scales.bin", "wb")
    f32_out = open(args.out / "dit.f32.bin", "wb")
    codes_at = scales_at = q8_at = q8_scales_at = f32_at = 0
    manifest: list[dict] = []
    quantized_bytes = original_bytes = 0

    for file, names in sorted(by_file.items()):
        with safe_open(src / file, framework="pt") as f:
            for name in sorted(names):
                t = f.get_tensor(name).to(torch.float32)
                arr = t.numpy()
                original_bytes += arr.size * 2  # bf16 on disk

                if should_quantize(name, arr.shape) and is_q8(name):
                    codes, scale = quantize_q8(arr)
                    packed = pack_q8(codes)
                    q8_out.write(packed.tobytes())
                    q8_scales_out.write(scale.tobytes())
                    manifest.append({
                        "name": name, "kind": "q8", "shape": list(arr.shape),
                        "codesOffset": q8_at, "scaleOffset": q8_scales_at,
                    })
                    q8_at += packed.size
                    q8_scales_at += scale.size
                    quantized_bytes += packed.nbytes + scale.nbytes
                elif should_quantize(name, arr.shape):
                    codes, scale = quantize_q4_g128(arr)
                    packed = pack_q4(codes)
                    codes_out.write(packed.tobytes())
                    scales_out.write(scale.tobytes())
                    manifest.append({
                        "name": name, "kind": "q4", "shape": list(arr.shape),
                        "codesOffset": codes_at, "scaleOffset": scales_at,
                    })
                    codes_at += packed.size
                    scales_at += scale.size
                    quantized_bytes += packed.nbytes + scale.nbytes
                else:
                    flat = arr.reshape(-1).astype(np.float32)
                    f32_out.write(flat.tobytes())
                    manifest.append({
                        "name": name, "kind": "f32", "shape": list(arr.shape), "offset": f32_at,
                    })
                    f32_at += flat.size
                    quantized_bytes += flat.nbytes

    for handle in (codes_out, scales_out, q8_out, q8_scales_out, f32_out):
        handle.close()

    (args.out / "dit.manifest.json").write_text(json.dumps({
        "note": "Generated by examples/zimage/tools/convert_dit.py. Do not hand-edit.",
        "format": {"quant": "q4-g128", "groupSize": GROUP, "range": [-7, 7], "codesPerWord": 8,
                   "reciprocal": "7/absmax in f64", "rounding": "floor(x) + (frac >= 0.5)",
                   "q8": {"quant": "q8-per-row", "range": [-127, 127], "codesPerWord": 4,
                          "appliesTo": list(Q8_SUFFIXES),
                          "why": "measured: adaLN at q4 costs 4.78% relative RMS on layers.0's "
                                 "output, against 1.2% for the next worst weight"}},
        "config": config,
        "limitLayers": args.limit_layers,
        "tensors": manifest,
    }, indent=1) + "\n")

    q = sum(1 for m in manifest if m["kind"] == "q4")
    q8n = sum(1 for m in manifest if m["kind"] == "q8")
    print(f"{len(manifest)} tensors ({q} q4, {q8n} q8, {len(manifest) - q - q8n} f32)")
    print(f"{original_bytes / 1e9:.2f} GB bf16 -> {quantized_bytes / 1e9:.2f} GB "
          f"({quantized_bytes / original_bytes * 100:.1f}%)")


if __name__ == "__main__":
    main()
