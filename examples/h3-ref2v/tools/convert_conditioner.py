#!/usr/bin/env python3
"""Qwen3-VL's conditioner, converted for `examples/h3-ref2v/src/conditioner-gpu.ts`.

Issue #212. R2V cannot precompute its conditioning -- the reference *is* the
input -- so this has to be resident. What that costs, measured on the released
checkpoint at int8:

| | |
| --- | --- |
| vision tower (27 blocks) | **0.60 GB** |
| text layers 0..50 | **24.87 GB** |
| embedding | 0.78 GB |
| **total** | **26.25 GB** |

Three things are dropped and the numbers above are why:

- **Text layers 51..63** (6.34 G params). MiniMax-H3 reads `hidden_states[50]`,
  so thirteen layers of a 64-layer stack are never evaluated. The final norm
  goes with them: `hidden_states[50]` is a layer *input*, not the stack's
  output.
- **`lm_head`** (0.78 G). A vocabulary-wide projection nothing reads.
- **`embed_tokens`** is kept, because the presentation is token ids.

Two channel permutations are folded in here rather than done per token, and
**each has a second half that is easy to miss**:

- The text stack's **M-RoPE** permutation goes into `q_proj` and `k_proj` --
  *and* into `q_norm` / `k_norm`, which index the channels the projection
  produced. #208 records the 8% cost of moving only the projections.
- The vision tower's rotation goes into the **Q and K thirds** of the fused
  `qkv` -- and into the matching thirds of its **bias**. V is never rotated.

**The weights are not redistributed by this repository.** See issue #190.

    python examples/h3-ref2v/tools/convert_conditioner.py \
      --model ~/h3-work/text-encoder-dl/text_encoder --out ~/h3-ref2v-gpu --quant q8
"""

import argparse
import json
import pathlib
import time

import numpy as np
import torch
from safetensors import safe_open

TEXT_ENCODER_LAYER = 50


def mrope_permutation(head_dim: int) -> list[int]:
    """`c` against `c + head_dim / 2`, which is how `rotate_half` pairs them."""
    half = head_dim // 2
    order = [0] * head_dim
    for c in range(half):
        order[2 * c] = c
        order[2 * c + 1] = c + half
    return order


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="the text_encoder/ directory")
    parser.add_argument("--out", required=True)
    parser.add_argument("--quant", choices=["f32", "q8"], default="q8")
    parser.add_argument("--layers", type=int, default=0,
                        help="convert only the first N text layers (0 = through the conditioning layer)")
    args = parser.parse_args()

    model = pathlib.Path(args.model)
    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    config = json.loads((model / "config.json").read_text())
    text = config["text_config"]
    vision = config["vision_config"]

    layers = args.layers or (TEXT_ENCODER_LAYER + 1)
    if layers > text["num_hidden_layers"]:
        raise SystemExit(f"the stack has {text['num_hidden_layers']} layers, not {layers}")

    head_dim = text["head_dim"]
    order = mrope_permutation(head_dim)
    vision_head_dim = vision["hidden_size"] // vision["num_heads"]
    vision_order = mrope_permutation(vision_head_dim)

    index = json.loads((model / "model.safetensors.index.json").read_text())["weight_map"]
    handles: dict[str, object] = {}

    def tensor(name: str) -> torch.Tensor:
        shard = index[name]
        if shard not in handles:
            handles[shard] = safe_open(str(model / shard), framework="pt")
        return handles[shard].get_tensor(name).to(torch.float32)

    entries: list[dict] = []
    offset = 0
    blob = out / ("conditioner.bin" if args.quant == "f32" else "conditioner.q8.bin")
    sink = blob.open("wb")

    def quantize_rows(values: torch.Tensor):
        """Per-row absmax int8 -- `ops/quantize`'s convention, as `examples/h3-video` uses."""
        absmax = values.abs().amax(dim=1)
        scale = torch.where(absmax == 0, torch.ones_like(absmax), absmax / 127.0)
        inverse = torch.where(absmax == 0, torch.zeros_like(absmax), 127.0 / absmax)
        codes = torch.clamp(torch.round(values * inverse.unsqueeze(1)), -127, 127).to(torch.int8)
        rows, k = codes.shape
        words = (k + 3) // 4
        padded = torch.zeros(rows, words * 4, dtype=torch.uint8)
        padded[:, :k] = codes.view(torch.uint8)
        packed = padded.view(rows, words, 4).to(torch.int32)
        word = packed[:, :, 0] | (packed[:, :, 1] << 8) | (packed[:, :, 2] << 16) | (packed[:, :, 3] << 24)
        return word.to(torch.int32).numpy().astype(np.uint32), scale.numpy().astype(np.float32)

    def add(name: str, values: torch.Tensor, transpose: bool = False) -> None:
        nonlocal offset
        if transpose:
            values = values.t()
        array = values.contiguous().numpy().astype("<f4").ravel()
        entries.append({"name": name, "shape": list(values.shape), "offset": offset, "count": int(array.size)})
        sink.write(array.tobytes())
        offset += int(array.size)

    def add_matrix(name: str, values: torch.Tensor) -> None:
        nonlocal offset
        if args.quant == "f32":
            add(name, values, transpose=True)
            return
        words, scale = quantize_rows(values)
        entries.append({"name": name, "shape": list(values.shape), "offset": offset,
                        "count": int(words.size), "kind": "q8"})
        sink.write(words.tobytes())
        offset += int(words.size)
        entries.append({"name": f"{name}.scale", "shape": [values.shape[0]], "offset": offset,
                        "count": int(scale.size), "kind": "f32"})
        sink.write(scale.tobytes())
        offset += int(scale.size)

    def permute_rows(values: torch.Tensor, heads: int, dim: int, channel_order: list[int]) -> torch.Tensor:
        view = values.reshape(heads, dim, -1)
        return view[:, channel_order, :].reshape(values.shape)

    started = time.time()

    # 1. The vision tower. Small, and it goes first so a failure here costs
    # seconds rather than the text stack's minutes.
    # A `Conv3d` whose kernel equals its stride and its input is a linear over
    # the flattened patch, so it is stored and dispatched as one.
    add_matrix("visual.patch_embed.proj.weight",
               tensor("model.visual.patch_embed.proj.weight").reshape(vision["hidden_size"], -1))
    add("visual.patch_embed.proj.bias", tensor("model.visual.patch_embed.proj.bias"))
    add("visual.pos_embed.weight", tensor("model.visual.pos_embed.weight"))
    for i in range(vision["depth"]):
        p = f"model.visual.blocks.{i}"
        q = f"visual.blocks.{i}"
        for suffix in ("norm1.weight", "norm1.bias", "norm2.weight", "norm2.bias",
                       "attn.proj.bias", "mlp.linear_fc1.bias", "mlp.linear_fc2.bias"):
            add(f"{q}.{suffix}", tensor(f"{p}.{suffix}"))
        # **The Q and K thirds of the fused qkv, and their bias thirds.** V is
        # never rotated, and permuting it would reorder channels the output
        # projection reads in the original order.
        qkv = tensor(f"{p}.attn.qkv.weight").clone()
        qkv_bias = tensor(f"{p}.attn.qkv.bias").clone()
        width = vision["hidden_size"]
        for part in range(2):
            block = qkv[part * width:(part + 1) * width]
            qkv[part * width:(part + 1) * width] = permute_rows(
                block, vision["num_heads"], vision_head_dim, vision_order)
            bias_block = qkv_bias[part * width:(part + 1) * width].reshape(vision["num_heads"], vision_head_dim)
            qkv_bias[part * width:(part + 1) * width] = bias_block[:, vision_order].reshape(-1)
        # **Split into three**, as `examples/h3-video` splits `to_qkv` and for
        # the same reason: a fused projection has to be un-interleaved per
        # token, which is a copy per token per block -- 20,736 of them for one
        # 256x256 reference. Three matrices whose outputs land contiguous cost
        # nothing extra and remove all of it.
        for part, name in enumerate(("q", "k", "v")):
            add_matrix(f"{q}.attn.{name}.weight", qkv[part * width:(part + 1) * width])
            add(f"{q}.attn.{name}.bias", qkv_bias[part * width:(part + 1) * width])
        add_matrix(f"{q}.attn.proj.weight", tensor(f"{p}.attn.proj.weight"))
        add_matrix(f"{q}.mlp.linear_fc1.weight", tensor(f"{p}.mlp.linear_fc1.weight"))
        add_matrix(f"{q}.mlp.linear_fc2.weight", tensor(f"{p}.mlp.linear_fc2.weight"))
    for name in ("merger", *[f"deepstack_merger_list.{i}" for i in range(len(vision["deepstack_visual_indexes"]))]):
        for suffix in ("norm.weight", "norm.bias", "linear_fc1.bias", "linear_fc2.bias"):
            add(f"visual.{name}.{suffix}", tensor(f"model.visual.{name}.{suffix}"))
        add_matrix(f"visual.{name}.linear_fc1.weight", tensor(f"model.visual.{name}.linear_fc1.weight"))
        add_matrix(f"visual.{name}.linear_fc2.weight", tensor(f"model.visual.{name}.linear_fc2.weight"))
    vision_bytes = offset * 4
    print(f"  vision tower {vision_bytes / 1e9:.2f} GB, {time.time() - started:.0f} s", flush=True)

    # 2. The embedding, then the text layers up to and including the one whose
    # *input* is read.
    add_matrix("embed_tokens.weight", tensor("model.language_model.embed_tokens.weight"))
    for i in range(layers):
        p = f"model.language_model.layers.{i}"
        q = f"layers.{i}"
        add(f"{q}.input_layernorm.weight", tensor(f"{p}.input_layernorm.weight"))
        add(f"{q}.post_attention_layernorm.weight", tensor(f"{p}.post_attention_layernorm.weight"))
        add_matrix(f"{q}.self_attn.q_proj.weight", permute_rows(
            tensor(f"{p}.self_attn.q_proj.weight"), text["num_attention_heads"], head_dim, order))
        add_matrix(f"{q}.self_attn.k_proj.weight", permute_rows(
            tensor(f"{p}.self_attn.k_proj.weight"), text["num_key_value_heads"], head_dim, order))
        add_matrix(f"{q}.self_attn.v_proj.weight", tensor(f"{p}.self_attn.v_proj.weight"))
        add_matrix(f"{q}.self_attn.o_proj.weight", tensor(f"{p}.self_attn.o_proj.weight"))
        # **The other half of the permutation.** These are per-channel and index
        # the channels the projection produced.
        add(f"{q}.self_attn.q_norm.weight", tensor(f"{p}.self_attn.q_norm.weight")[order])
        add(f"{q}.self_attn.k_norm.weight", tensor(f"{p}.self_attn.k_norm.weight")[order])
        add_matrix(f"{q}.mlp.gate_proj.weight", tensor(f"{p}.mlp.gate_proj.weight"))
        add_matrix(f"{q}.mlp.up_proj.weight", tensor(f"{p}.mlp.up_proj.weight"))
        add_matrix(f"{q}.mlp.down_proj.weight", tensor(f"{p}.mlp.down_proj.weight"))
        if (i + 1) % 10 == 0:
            print(f"  {i + 1}/{layers} text layers, {offset * 4 / 1e9:.2f} GB, "
                  f"{time.time() - started:.0f} s", flush=True)
    sink.close()

    manifest = {
        "model": "qwen3-vl-conditioner",
        "source": "MiniMaxAI/MiniMax-H3 (text_encoder)",
        "licence": "MiniMax H3 Community License Agreement — not this repository's, and not redistributed by it",
        "textConfig": text,
        "visionConfig": vision,
        "textEncoderLayer": TEXT_ENCODER_LAYER,
        "layers": layers,
        "dtype": args.quant,
        "weightLayout": "[in, out]" if args.quant == "f32" else "[out, in], int8 packed four per u32",
        "mropePermutation": order,
        "visionPermutation": vision_order,
        # What the page needs to build a request, so it reads one manifest.
        "processor": {
            "patchSize": vision["patch_size"], "mergeSize": vision["spatial_merge_size"],
            "temporalPatchSize": vision["temporal_patch_size"],
            "minPixels": 65536, "maxPixels": 16777216,
            "imageMean": [0.5, 0.5, 0.5], "imageStd": [0.5, 0.5, 0.5],
        },
        "videoSampleFps": 2.0,
        "tensors": entries,
        "visionBytes": vision_bytes,
        "residentBytes": offset * 4,
    }
    (out / "conditioner.manifest.json").write_text(json.dumps(manifest, indent=1) + "\n")
    print(f"resident {offset * 4 / 1e9:.2f} GB ({vision_bytes / 1e9:.2f} vision), "
          f"{len(entries)} tensors, {time.time() - started:.0f} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
