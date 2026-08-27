"""A copy of a transformer checkpoint with the port's int8 already baked in.

Issue #216. The port runs int8 weights and there is no float conversion of fifty
blocks that fits on a 32 GB card, so the only way to ask "how much of this
disagreement is the quantisation?" is to put the same quantisation on
*upstream's* side.

Doing that in memory does not work: `from_pretrained` hands back mmap-backed
tensors, and writing to them copy-on-writes 62 GB of page cache into anonymous
memory. The process gets killed, and a killed process prints nothing.

So it is done on disk, one shard at a time — a few gigabytes resident, and the
result loads exactly like the original.

**Which tensors, from the port's own manifest.** An earlier version rounded
every `nn.Linear` it could reach and came out *further* from the port than the
untouched reference did, because `convert_dit.py` folds the modulation matrices
into precomputed level tables and never quantises them at all. A reference that
approximates differently from the thing it is a reference for is not a
reference. `ff.hidden`/`ff.gate` are `ff.net.0.proj` split along rows, and the
quantisation is per row, so they collapse back to one name.

    python examples/h3-dit/tools/quantise_checkpoint.py \
      --model ~/h3-work/transformer-ref-dl/transformer_ref \
      --manifest ~/h3-work/h3-ref-gpu/dit.manifest.json \
      --out ~/h3-work/transformer-ref-q8
"""

from __future__ import annotations

import argparse
import json
import pathlib
import shutil
import time

import torch
from safetensors import safe_open
from safetensors.torch import save_file


def quantised_names(manifest_path: pathlib.Path) -> set[str]:
    entries = json.loads(manifest_path.read_text())["tensors"]
    names = set()
    for entry in entries:
        if entry.get("kind") != "q8":
            continue
        name = entry["name"]
        if name.endswith(".ff.hidden.weight") or name.endswith(".ff.gate.weight"):
            name = name.rsplit(".ff.", 1)[0] + ".ff.net.0.proj.weight"
        names.add(name)
    return names


def roundtrip(weight: torch.Tensor, chunk: int = 4096) -> torch.Tensor:
    """`convert_dit.quantize_rows`, dequantised straight back.

    Per-row absmax over `[out, in]`, the same rounding and the same clamp, minus
    the packing — the packing is a layout, not an approximation.
    """
    out = weight.clone()
    for start in range(0, out.shape[0], chunk):
        block = out[start:start + chunk]
        values = block.to(torch.float32)
        absmax = values.abs().amax(dim=1)
        scale = torch.where(absmax == 0, torch.ones_like(absmax), absmax / 127.0)
        inverse = torch.where(absmax == 0, torch.zeros_like(absmax), 127.0 / absmax)
        codes = torch.clamp(torch.round(values * inverse.unsqueeze(1)), -127, 127)
        block.copy_((codes * scale.unsqueeze(1)).to(out.dtype))
    return out


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, help="the transformer/ or transformer_ref/ directory")
    p.add_argument("--manifest", required=True, help="a dit.manifest.json, for which tensors are q8")
    p.add_argument("--out", required=True)
    args = p.parse_args()

    model = pathlib.Path(args.model).expanduser()
    out = pathlib.Path(args.out).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    want = quantised_names(pathlib.Path(args.manifest).expanduser())
    print(f"{len(want)} tensors marked q8 by {args.manifest}")

    # Everything that is not a shard travels unchanged: the config is what makes
    # the copy loadable, and rewriting it would be a second thing to keep true.
    for path in sorted(model.iterdir()):
        if path.is_file() and path.suffix != ".safetensors":
            shutil.copy2(path, out / path.name)

    shards = sorted(model.glob("*.safetensors"))
    if not shards:
        raise SystemExit(f"no .safetensors under {model}")
    started = time.time()
    done = 0
    for shard in shards:
        tensors: dict[str, torch.Tensor] = {}
        with safe_open(shard, framework="pt") as f:
            for name in f.keys():
                tensor = f.get_tensor(name)
                if name in want:
                    tensor = roundtrip(tensor)
                    done += 1
                tensors[name] = tensor
        save_file(tensors, out / shard.name, metadata={"format": "pt"})
        print(f"  {shard.name}: {len(tensors)} tensors, {done} rounded so far", flush=True)
        del tensors
    if done != len(want):
        raise SystemExit(f"rounded {done} tensors and the manifest marks {len(want)}")
    print(f"wrote {out} in {time.time() - started:.0f} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
