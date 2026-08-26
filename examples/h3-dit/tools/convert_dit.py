#!/usr/bin/env python3
"""MiniMax-H3's 50-layer DiT, converted for `examples/h3-dit/src/model-gpu.ts`.

Issue #210. The checkpoint is **33.12 G parameters** -- 66.2 GB of bf16 -- and
where they sit decides what this converter has to be:

| | params | share |
| --- | --- | --- |
| `adaln_proj.linear.weight`, 50 x `[96768, 2688]` | **13.01 G** | **39.3%** |
| `ff.net.0.proj.weight`, 50 x `[28672, 5376]` | 7.71 G | 23.3% |
| `ff.net.2.weight` | 3.85 G | 11.6% |
| attention, four matrices per block | 7.71 G | 23.3% |
| everything else | 0.84 G | 2.5% |

**Two-fifths of the model exists to project a two-row tensor.** `adaln_proj`
reads `temb`, which has one row per *distinct noise level* in the sequence --
two, for a text-to-video-and-audio request -- and produces the
`(timestep, modality)` modulation table. Its output does not depend on the
video at all, only on the timestep, and the timesteps of a whole sampling run
are known the moment `num_inference_steps` is chosen.

So this converter **evaluates the modulation tables here** and never ships
`adaln_proj`. `norm_out.linear` is the same shape of thing and goes the same
way. That takes the resident weights from 33.1 GB of int8 to **20.1 GB**, which
is the difference between fitting on a 32 GB card beside the VAE decoder and
not.

The cost is that a converted model runs **only the step counts named here**.
That is a real limitation and the page says so; the alternative is 13 GB of
weights resident to do 520 MFLOP a step.

**The weights are not redistributed by this repository.** See issue #190: the
MiniMax H3 Community License Agreement permits redistribution only within an
Applicable Territory that excludes the EU, the UK, South Korea and the USA.

    python examples/h3-dit/tools/convert_dit.py \
      --model ~/h3-work/transformer-dl/transformer --out ~/h3-dit-gpu \
      --steps 16,32 --quant q8
"""

import argparse
import json
import math
import pathlib
import time

import numpy as np
import torch
from safetensors import safe_open

MODALITY_NUM = 3
VIDEO_SHIFT = 12.0
AUDIO_SHIFT = 3.0


def set_timesteps(num_inference_steps: int, shift: float) -> np.ndarray:
    """`MiniMaxH3Scheduler.set_timesteps`, in numpy so torch need not be loaded twice."""
    base = torch.linspace(1.0, 0.0, num_inference_steps, dtype=torch.float32)
    sigmas = shift * base / (1 + (shift - 1) * base)
    sigmas = torch.unique_consecutive(sigmas)
    return (1.0 - sigmas[:-1]).numpy()


def timestep_embedding(timesteps: np.ndarray, dim: int, max_period: int = 10000) -> torch.Tensor:
    """diffusers' `get_timestep_embedding` with H3's flags: cos first, no frequency shift."""
    half = dim // 2
    exponent = -math.log(max_period) * torch.arange(0, half, dtype=torch.float32) / half
    emb = torch.from_numpy(timesteps.astype("float32"))[:, None] * torch.exp(exponent)[None, :]
    return torch.cat([torch.cos(emb), torch.sin(emb)], dim=-1)


def rope_permutation(head_dim: int, rot_dim: int) -> list[int]:
    """The channel order `ropeAxes` reads, from the order `rotate_half` writes.

    H3 rotates channel `c` against `c + rot_dim / 2`; `ropeAxes` rotates
    adjacent pairs within each axis. Same rotation, different storage, so the
    projection's **rows** are reordered once here rather than per token.
    """
    per_axis = rot_dim // 3
    if per_axis % 2 or per_axis * 3 != rot_dim:
        raise ValueError(f"{rot_dim} does not split into three even halves")
    half = per_axis // 2
    order = []
    for axis in range(3):
        for pair in range(half):
            order.append(axis * half + pair)
            order.append(rot_dim // 2 + axis * half + pair)
    order.extend(range(rot_dim, head_dim))
    return order


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="the transformer/ directory")
    parser.add_argument("--out", required=True)
    parser.add_argument("--quant", choices=["f32", "q8"], default="q8")
    parser.add_argument("--steps", default="16,32", help="comma-separated step counts to precompute tables for")
    parser.add_argument("--layers", type=int, default=0, help="convert only the first N blocks (0 = all)")
    args = parser.parse_args()

    model = pathlib.Path(args.model)
    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    config = json.loads((model / "config.json").read_text())
    num_layers = args.layers or config["num_layers"]
    head_dim = config["attention_head_dim"]
    heads = config["num_attention_heads"]
    hidden = config["hidden_size"]
    rot_dim = 2 * 3 * config["rope_freq_dim"]
    permutation = rope_permutation(head_dim, rot_dim)

    index = json.loads((model / "diffusion_pytorch_model.safetensors.index.json").read_text())["weight_map"]
    handles: dict[str, object] = {}

    def tensor(name: str) -> torch.Tensor:
        shard = index[name]
        if shard not in handles:
            handles[shard] = safe_open(str(model / shard), framework="pt")
        return handles[shard].get_tensor(name).to(torch.float32)

    entries: list[dict] = []
    offset = 0
    blob = out / ("dit.bin" if args.quant == "f32" else "dit.q8.bin")
    sink = blob.open("wb")

    def quantize_rows(values: torch.Tensor):
        """Per-row absmax int8, `ops/quantize`'s convention -- see `examples/h3-video`."""
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
        """`nn.Linear`'s `[out, in]`: transposed for the f32 kernel, untransposed for q8."""
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

    def add_swiglu(prefix: str, values: torch.Tensor) -> None:
        """`ff.net.0.proj` split into its two halves, as `examples/h3-video` does.

        diffusers' `SwiGLU` computes `hidden, gate = proj(x).chunk(2, -1)` and
        returns `hidden * silu(gate)`, so the **first** half is the value and
        the second is the gate -- the opposite of the visual VAE, in the same
        model. Splitting the weight turns one matmul plus a per-row slice into
        two matmuls whose outputs land contiguous; leaving it whole would need
        a dispatch per row, which at 1,024 rows and 50 blocks is 100,000 of
        them a step.
        """
        half = values.shape[0] // 2
        add_matrix(f"{prefix}.ff.hidden.weight", values[:half])
        add_matrix(f"{prefix}.ff.gate.weight", values[half:])

    def permute_rows(values: torch.Tensor) -> torch.Tensor:
        """Reorder each head's `head_dim` rows of a `[heads * head_dim, hidden]` projection."""
        view = values.reshape(heads, head_dim, -1)
        return view[:, permutation, :].reshape(values.shape)

    started = time.time()

    # 1. The parts that are not the block stack.
    add_matrix("proj_in.weight", tensor("proj_in.weight"))
    add("proj_in.bias", tensor("proj_in.bias"))
    add_matrix("audio_proj_in.weight", tensor("audio_proj_in.weight"))
    add("audio_proj_in.bias", tensor("audio_proj_in.bias"))
    add_matrix("context_embedder.weight", tensor("context_embedder.weight"))
    add("context_embedder.bias", tensor("context_embedder.bias"))
    add_matrix("proj_out.weight", tensor("proj_out.weight"))
    add("proj_out.bias", tensor("proj_out.bias"))
    add_matrix("audio_proj_out.weight", tensor("audio_proj_out.weight"))
    add("audio_proj_out.bias", tensor("audio_proj_out.bias"))
    add("norm_out.norm.weight", tensor("norm_out.norm.weight"))

    # 2. The text refiner. **No rope**, so no permutation -- the same parameter
    # names as a DiT block and the opposite preparation.
    for i in range(config["num_refiner_layers"]):
        p = f"token_refiner.refiner_blocks.{i}"
        add(f"{p}.norm1.weight", tensor(f"{p}.norm1.weight"))
        for which in ("to_q", "to_k", "to_v"):
            add_matrix(f"{p}.attn.{which}.weight", tensor(f"{p}.attn.{which}.weight"))
        add(f"{p}.attn.norm_q.weight", tensor(f"{p}.attn.norm_q.weight"))
        add(f"{p}.attn.norm_k.weight", tensor(f"{p}.attn.norm_k.weight"))
        add_matrix(f"{p}.attn.to_out.0.weight", tensor(f"{p}.attn.to_out.0.weight"))
        add(f"{p}.norm2.weight", tensor(f"{p}.norm2.weight"))
        add_swiglu(p, tensor(f"{p}.ff.net.0.proj.weight"))
        add_matrix(f"{p}.ff.net.2.weight", tensor(f"{p}.ff.net.2.weight"))
    add("token_refiner.final_norm.weight", tensor("token_refiner.final_norm.weight"))

    # 3. The block stack, without `adaln_proj`.
    for i in range(num_layers):
        p = f"transformer_blocks.{i}"
        add(f"{p}.norm1.weight", tensor(f"{p}.norm1.weight"))
        add_matrix(f"{p}.attn.to_q.weight", permute_rows(tensor(f"{p}.attn.to_q.weight")))
        add_matrix(f"{p}.attn.to_k.weight", permute_rows(tensor(f"{p}.attn.to_k.weight")))
        add_matrix(f"{p}.attn.to_v.weight", tensor(f"{p}.attn.to_v.weight"))
        # Per-channel and indexed by the channel the projection produced, so
        # they move with it. Leaving them alone scales the wrong channels and
        # costs 8% -- measured, in #208.
        add(f"{p}.attn.norm_q.weight", tensor(f"{p}.attn.norm_q.weight")[permutation])
        add(f"{p}.attn.norm_k.weight", tensor(f"{p}.attn.norm_k.weight")[permutation])
        add_matrix(f"{p}.attn.to_out.0.weight", tensor(f"{p}.attn.to_out.0.weight"))
        add(f"{p}.norm2.weight", tensor(f"{p}.norm2.weight"))
        add_swiglu(p, tensor(f"{p}.ff.net.0.proj.weight"))
        add_matrix(f"{p}.ff.net.2.weight", tensor(f"{p}.ff.net.2.weight"))
        if (i + 1) % 10 == 0:
            print(f"  {i + 1}/{num_layers} blocks, {offset * 4 / 1e9:.2f} GB, {time.time() - started:.0f} s", flush=True)
    sink.close()
    resident = offset * 4

    # 4. The modulation tables, evaluated here rather than shipped as weights.
    step_counts = [int(s) for s in args.steps.split(",")]
    time_w1, time_b1 = tensor("time_embedder.linear_1.weight"), tensor("time_embedder.linear_1.bias")
    time_w2, time_b2 = tensor("time_embedder.linear_2.weight"), tensor("time_embedder.linear_2.bias")
    norm_out_w, norm_out_b = tensor("norm_out.linear.weight"), tensor("norm_out.linear.bias")

    tables_entries: list[dict] = []
    tables_offset = 0
    tables_sink = (out / "adaln.bin").open("wb")

    def add_table(name: str, values: torch.Tensor) -> None:
        nonlocal tables_offset
        array = values.contiguous().numpy().astype("<f4").ravel()
        tables_entries.append({"name": name, "shape": list(values.shape),
                               "offset": tables_offset, "count": int(array.size)})
        tables_sink.write(array.tobytes())
        tables_offset += int(array.size)

    schedules = {}
    temb_cache: dict[int, torch.Tensor] = {}
    for steps in step_counts:
        video_t = set_timesteps(steps, VIDEO_SHIFT)
        audio_t = set_timesteps(steps, AUDIO_SHIFT)
        schedules[steps] = {"video": video_t.tolist(), "audio": audio_t.tolist()}
        # `torch.unique(sorted=True)` per step: the distinct noise levels the
        # sequence carries. Usually two -- the audio schedule runs at shift 3
        # and the video's at 12 -- but **at step 0 there is exactly one**, since
        # both schedules start at sigma 1 and `t = 1 - sigma` is 0 for both.
        #
        # A ragged table would need a per-step offset everywhere it is read, so
        # the single level is stored twice instead. Nothing reads the duplicate:
        # with one distinct level every row's timestep index is 0, so only rows
        # 0..2 of the table are addressed.
        levels_per_step = []
        rows = []
        for v, a in zip(video_t, audio_t):
            distinct = np.unique(np.array([v, a], dtype=np.float32))
            levels_per_step.append(int(distinct.size))
            rows.append(distinct if distinct.size == 2 else np.array([distinct[0], distinct[0]], dtype=np.float32))
        levels = np.stack(rows)
        schedules[steps]["levelsPerStep"] = levels_per_step
        proj = timestep_embedding(levels.reshape(-1), config["freq_dim"])
        temb = torch.nn.functional.silu(proj @ time_w1.t() + time_b1) @ time_w2.t() + time_b2
        temb_cache[steps] = temb
        add_table(f"temb.{steps}", temb.reshape(len(video_t), levels.shape[1], -1))
        activated = torch.nn.functional.silu(temb)
        # **`1 + scale`, not `scale`.** The modulation is `x * (1 + scale) +
        # shift`, and no kernel here fuses that; storing the sum makes it a
        # multiply and an add, both of which `ops/elementwise` already has.
        norm_out_table = (activated @ norm_out_w.t() + norm_out_b).reshape(len(video_t), levels.shape[1], 2, hidden)
        norm_out_table[:, :, 1] += 1.0
        add_table(f"normOut.{steps}", norm_out_table.reshape(len(video_t), levels.shape[1], -1))

    # One pass over `adaln_proj`: 13 GB of weights read, ~12 MB of table per
    # block per 16 steps written, and nothing kept.
    for i in range(num_layers):
        w = tensor(f"transformer_blocks.{i}.adaln_proj.linear.weight")
        bias = tensor(f"transformer_blocks.{i}.adaln_proj.linear.bias")
        for steps in step_counts:
            temb = temb_cache[steps]
            activated = torch.nn.functional.silu(temb)
            table = activated @ w.t() + bias
            # `[steps, levels, 6 * hidden * modalities]` viewed the way the model
            # views it: `[-1, 6 * hidden]`, so row `timestep * 3 + tag`.
            num_steps = table.shape[0] // levels.shape[1]
            view = table.reshape(num_steps, levels.shape[1] * MODALITY_NUM, 6, hidden)
            # Chunks 1 and 4 are `scale_msa` and `scale_mlp`; see `normOut`.
            view[:, :, 1] += 1.0
            view[:, :, 4] += 1.0
            add_table(f"adaln.{steps}.{i}", view.reshape(num_steps, levels.shape[1] * MODALITY_NUM, 6 * hidden))
        del w, bias
        if (i + 1) % 10 == 0:
            print(f"  tables {i + 1}/{num_layers}, {tables_offset * 4 / 1e6:.0f} MB, "
                  f"{time.time() - started:.0f} s", flush=True)
    tables_sink.close()

    manifest = {
        "model": "minimax-h3-dit",
        "source": "MiniMaxAI/MiniMax-H3 (transformer)",
        "licence": "MiniMax H3 Community License Agreement — not this repository's, and not redistributed by it",
        "config": config,
        "layers": num_layers,
        "dtype": args.quant,
        "scaleStoredAsOnePlusScale": True, "weightLayout": "[in, out]" if args.quant == "f32" else "[out, in], int8 packed four per u32",
        "ropePermutation": permutation,
        "stepCounts": step_counts,
        "schedules": schedules,
        "tensors": entries,
        "tables": tables_entries,
        "residentBytes": resident,
    }
    (out / "dit.manifest.json").write_text(json.dumps(manifest, indent=1) + "\n")
    print(f"resident {resident / 1e9:.2f} GB, tables {tables_offset * 4 / 1e9:.2f} GB, "
          f"{time.time() - started:.0f} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
