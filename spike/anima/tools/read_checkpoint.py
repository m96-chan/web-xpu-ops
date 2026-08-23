#!/usr/bin/env python3
"""Reads Anima-3.8B's architecture out of its checkpoints, without downloading them.

The repository ships no `config.json` — just safetensors and a ComfyUI
workflow — so the architecture is whatever the tensors say it is. A
safetensors header sits at the front of the file and names every tensor's
shape, so an HTTP `Range` request for the first few hundred kilobytes answers
the question that a 7.5 GB download would otherwise be needed for.

This is the spike's evidence. Everything in `README.md` that is stated as fact
about shapes comes from running this, not from reading the model card.

    python3 spike/anima/tools/read_checkpoint.py            # summary
    python3 spike/anima/tools/read_checkpoint.py --tensors  # every distinct shape
"""
from __future__ import annotations

import argparse
import collections
import json
import re
import struct
import urllib.request

REPO = "lylogummy/Anima-3.8B"
FILES = {
    "dit": "difussion_models/Anima-3.8B.safetensors",
    "adapter": "text_encoders/Anima-3.8B-expanded_adapter.safetensors",
    "text_encoder": "text_encoders/qwen35_4b.safetensors",
}

# bytes per element, for the size arithmetic below.
WIDTH = {"BF16": 2, "F16": 2, "F32": 4, "F8_E4M3": 1, "I8": 1}


def header(repo: str, path: str) -> dict:
    """The safetensors header, over two Range requests."""
    url = f"https://huggingface.co/{repo}/resolve/main/{path}"
    request = urllib.request.Request(url, headers={"Range": "bytes=0-7"})
    length = struct.unpack("<Q", urllib.request.urlopen(request).read())[0]
    request = urllib.request.Request(url, headers={"Range": f"bytes=8-{7 + length}"})
    parsed = json.loads(urllib.request.urlopen(request).read())
    parsed.pop("__metadata__", None)
    return parsed


def numel(shape: list[int]) -> int:
    total = 1
    for dim in shape:
        total *= dim
    return total


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tensors", action="store_true", help="print every distinct shape")
    ap.add_argument("--only", choices=sorted(FILES), default=None)
    args = ap.parse_args()

    for name, path in FILES.items():
        if args.only and name != args.only:
            continue
        try:
            entries = header(REPO, path)
        except Exception as error:  # noqa: BLE001 - the point is to report, not to raise
            print(f"\n=== {name}: {type(error).__name__}: {error}")
            continue

        params = sum(numel(v["shape"]) for v in entries.values())
        stored = sum(numel(v["shape"]) * WIDTH.get(v["dtype"], 0) for v in entries.values())
        dtypes = sorted({v["dtype"] for v in entries.values()})
        print(f"\n=== {name}  ({path})")
        print(f"{len(entries)} tensors, {params / 1e9:.2f}B params, dtypes {dtypes}")
        print(f"stored {stored / 1e9:.2f} GB")
        # q8 is one byte per weight plus one f32 scale per row — `ops/matvec`'s
        # convention, and what `examples/zimage` converts to. The f32 tail
        # (norms, biases, embedders) is not quantized there and is not counted
        # as quantized here either.
        quantizable = sum(numel(v["shape"]) for v in entries.values() if len(v["shape"]) == 2 and min(v["shape"]) >= 128)
        rest = params - quantizable
        print(f"q8 estimate {(quantizable + rest * 4) / 1e9:.2f} GB "
              f"({quantizable / 1e9:.2f}B quantizable, {rest / 1e9:.3f}B left f32)")

        shapes: dict[str, tuple[list[int], int]] = {}
        for tensor_name, value in entries.items():
            key = re.sub(r"\.\d+\.", ".N.", tensor_name)
            shape, count = shapes.get(key, (value["shape"], 0))
            shapes[key] = (shape, count + 1)
        if args.tensors:
            for key, (shape, count) in shapes.items():
                print(f"  {count:>4}x  {key:<62} {shape}")
        else:
            groups: collections.Counter[str] = collections.Counter()
            for key, (_, count) in shapes.items():
                groups[key.split(".")[0] if not key.startswith("net.") else ".".join(key.split(".")[:2])] += count
            for key, count in sorted(groups.items()):
                print(f"  {key}: {count}")


if __name__ == "__main__":
    main()
