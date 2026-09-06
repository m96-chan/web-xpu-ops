#!/usr/bin/env python3
"""Qwen3-VL's vision tower, at a size small enough to check in.

Issue #212. This is the half that makes R2V R2V: a reference image or video's
pixels become the vision tokens the conditioner reads. It is **0.60 GB of the
conditioner's 33.36** — the small half, which was worth measuring before
assuming the opposite.

**The weights are random**, at a tiny geometry, so the fixture is ~250 KB, runs
in CI anywhere and carries no model licence. What it establishes is structure.

Four things it decides, and each yields a well-formed tensor when wrong:

- **Tokens are in merge-block order, not raster order.** After the patch embed
  everything — the interpolated position embedding, the rotary coordinates, the
  mergers — is indexed `(t, h/m, w/m, m, m)`, so a raster reading is a picture
  read in the wrong order with the right shape.
- **The position embedding is bilinearly interpolated** from a learned
  `num_grid_per_side ** 2` table onto the image's own grid, with
  `torch.linspace` and an `int()` truncation deciding the four taps.
- **Two different GELUs.** The blocks' MLP is `gelu_pytorch_tanh`; the mergers
  use `nn.GELU()`, the exact one. They are not the same function.
- **Deepstack taps.** The hidden state at three chosen layers is run through its
  own merger, and those outputs are additional conditioning — not the tower's
  output.

    python examples/h3-ref2v/tools/gen_vision_golden.py --out examples/h3-ref2v/fixtures
"""

import argparse
import json
import pathlib

import torch

from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLVisionConfig
from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLVisionModel

CONFIG = {
    "hidden_size": 32,
    "intermediate_size": 64,
    "num_heads": 2,
    "depth": 4,
    "in_channels": 3,
    "patch_size": 4,
    "spatial_merge_size": 2,
    "temporal_patch_size": 2,
    "out_hidden_size": 24,
    "num_position_embeddings": 64,
    "hidden_act": "gelu_pytorch_tanh",
    "deepstack_visual_indexes": [1, 2],
    "initializer_range": 0.02,
}

# One image, 4x4 patches — two merge blocks each way, so the merge-block
# ordering differs from raster order and a port that used the latter is visible.
GRID = [[1, 4, 4]]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    config = Qwen3VLVisionConfig(**CONFIG)
    model = Qwen3VLVisionModel(config).to(torch.float64).eval()

    grid = torch.tensor(GRID, dtype=torch.long)
    tokens = int((grid[:, 0] * grid[:, 1] * grid[:, 2]).sum())
    patch_dim = CONFIG["in_channels"] * CONFIG["temporal_patch_size"] * CONFIG["patch_size"] ** 2
    generator = torch.Generator("cpu").manual_seed(args.seed)
    pixels = torch.randn(tokens, patch_dim, generator=generator, dtype=torch.float64)

    with torch.no_grad():
        out = model(pixels, grid_thw=grid)
        # The intermediate pieces too, so a divergence names its stage.
        embedded = model.patch_embed(pixels)
        pos = model.fast_pos_embed_interpolate(grid)
        rot = model.rot_pos_emb(grid)

    blob = bytearray()
    tensors = []

    def put(name: str, x: torch.Tensor) -> None:
        a = x.detach().to(torch.float32).contiguous().numpy().astype("<f4")
        tensors.append({"name": name, "shape": list(x.shape), "offset": len(blob), "count": a.size})
        blob.extend(a.tobytes())

    for name, param in model.state_dict().items():
        put(name, param)
    put("input.pixels", pixels)
    put("stage.patch_embed", embedded)
    put("stage.pos_embed", pos)
    put("stage.rot_pos_emb", rot)
    put("output.last_hidden_state", out.last_hidden_state)
    put("output.pooler_output", out.pooler_output)
    for index, feature in enumerate(out.deepstack_features):
        put(f"output.deepstack.{index}", feature)

    out_dir = pathlib.Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "vision.bin").write_bytes(bytes(blob))
    (out_dir / "vision.json").write_text(json.dumps({
        "source": "transformers Qwen3VLVisionModel, random weights",
        "note": "Not Qwen's checkpoint. Random weights at a tiny geometry; no model licence applies.",
        "seed": args.seed,
        "config": CONFIG,
        "numGridPerSide": int(model.num_grid_per_side),
        "grid": GRID,
        "tokens": tokens,
        "patchDim": patch_dim,
        "numDeepstack": len(out.deepstack_features),
        "tensors": tensors,
    }, indent=1) + "\n")

    print(f"{len(tensors)} tensors, {len(blob)/1024:.1f} KB, {tokens} tokens, "
          f"grid per side {int(model.num_grid_per_side)}")
    print(f"pooler {tuple(out.pooler_output.shape)}  deepstack {len(out.deepstack_features)}")
    print(f"wrote {out_dir}/vision.json and vision.bin")


if __name__ == "__main__":
    main()
