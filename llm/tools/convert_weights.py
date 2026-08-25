#!/usr/bin/env python3
"""Converts a HF safetensors checkpoint (bf16) into issue #105's per-row int8
weight format for `LlamaEngineQ8` (`llm/engine-q8.ts`) / `llm/weights-q8.ts`.

Every Linear weight (`wq`/`wk`/`wv`/`wo`/`gate`/`up`/`down`/`lm_head`) and the
embedding table are written as per-row absmax int8 codes + f32 scale
(`quant_common.py#quantize_per_row`, the same rounding rule
`llm/weights-q8.ts` documents and `quantize-parity.test.ts` checks against
`ops/quantize/reference.ts`); norm weights are written f32, unquantized.
`wq`/`wk` get `permute_rope_channels` applied — the same permutation
`gen_fixture.py` applies for the tiny fixture, imported from there rather
than re-derived a third time (rule 7) — **before** quantization, so the
converted file's row order already matches what `LlamaEngineQ8` expects; see
`llm/weights.ts#permuteRopeChannels`'s module doc for why permuting rows and
quantizing per row commute (this script does not depend on that commuting —
it just always permutes first — `gen_fixture_q8.py`, which has to quantize
the *unpermuted* rows first for a different reason, is the one that relies on
the commuting property).

## Memory: streamed one tensor at a time

`safetensors.safe_open` memory-maps the checkpoint; `get_tensor(key)`
materializes exactly one tensor. Each tensor is converted (bf16 -> f32 ->
int8 codes + f32 scale) and written to the three open output files
immediately, and every local reference to that tensor's data is dropped
before the next `get_tensor` call, so peak memory is bounded by the single
largest tensor in play (`lm_head.weight` / `embed_tokens.weight`,
102,400 x 1,792 for Sarashina2.2-1B: ~367 MiB bf16 + ~734 MiB f32 + ~183 MiB
int8 codes, transiently, not the whole ~2.7 GiB checkpoint at once) — not
streamed *within* a tensor, because at this model's size a single tensor
comfortably fits, and only whole-checkpoint residency was the risk issue
#105 flags for Node's loader side; nothing here claims the same headroom
holds for a much larger model.

## Output layout

    <out-dir>/manifest.json        config + per-weight {name, kind, shape,
                                    offsets}, "quant" or "norm" per entry
    <out-dir>/weights.codes.bin    int8, one byte per quantized code,
                                    concatenated in manifest order
    <out-dir>/weights.scales.bin   f32 little-endian, one segment per
                                    quantized entry (length = that entry's
                                    row count)
    <out-dir>/weights.norms.bin    f32 little-endian, concatenated norm
                                    weights

Same manifest-entry shape (`{name, kind, shape, offsets}`) `llm/tools/gen_fixture_q8.py`
writes for the tiny fixture, just under different filenames (`weights.*.bin`
here vs. `tiny_q8.*.bin` there) — both are parsed by the same core,
`llm/weights-q8-io.ts#buildLlamaWeightsQ8`, so a manifest field renamed in one
converter is a compile error in the other's loader rather than a silent
mismatch. `llm/real-model-weights.ts#loadConvertedWeightsQ8` is this format's
own loader.

## Usage

    source third_party/venv/bin/activate   # from technologies.moe/alibi-ai
    python llm/tools/convert_weights.py \\
        --model-dir /path/to/sarashina2.2-1b-alibi-v1 \\
        --out-dir   /path/to/webgpu-weights/sarashina2.2-1b-alibi-v1-q8
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from gen_fixture import permute_rope_channels  # noqa: E402  (already-validated copy — see module doc)
from quant_common import quantize_per_row  # noqa: E402


def load_hf_config(model_dir: Path) -> dict:
    raw = json.loads((model_dir / "config.json").read_text())
    rope = raw.get("rope_parameters") or {}
    return {
        "numLayers": raw["num_hidden_layers"],
        "hiddenSize": raw["hidden_size"],
        "numHeads": raw["num_attention_heads"],
        "numKvHeads": raw["num_key_value_heads"],
        "headDim": raw.get("head_dim", raw["hidden_size"] // raw["num_attention_heads"]),
        "ffnHidden": raw["intermediate_size"],
        "vocabSize": raw["vocab_size"],
        "ropeTheta": rope.get("rope_theta", raw.get("rope_theta", 10000.0)),
        "rmsNormEps": raw["rms_norm_eps"],
        "tieEmbeddings": bool(raw.get("tie_word_embeddings", False)),
    }


def convert(model_dir: Path, out_dir: Path) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    config = load_hf_config(model_dir)
    num_layers = config["numLayers"]
    num_heads = config["numHeads"]
    num_kv_heads = config["numKvHeads"]
    head_dim = config["headDim"]

    manifest_weights: list[dict] = []
    codes_offset = 0
    scale_offset = 0
    norm_offset = 0

    st_path = model_dir / "model.safetensors"
    codes_path = out_dir / "weights.codes.bin"
    scales_path = out_dir / "weights.scales.bin"
    norms_path = out_dir / "weights.norms.bin"

    with safe_open(st_path, framework="pt") as f, \
            open(codes_path, "wb") as codes_f, \
            open(scales_path, "wb") as scales_f, \
            open(norms_path, "wb") as norms_f:

        def write_quant(name: str, key: str, permute: tuple[int, int] | None = None) -> None:
            nonlocal codes_offset, scale_offset
            tensor = f.get_tensor(key)
            w = tensor.to(torch.float32).numpy()
            del tensor
            codes, scale = quantize_per_row(w)
            del w
            if permute is not None:
                heads, hd = permute
                codes = permute_rope_channels(codes, heads, hd)
                scale = permute_rope_channels(scale.reshape(-1, 1), heads, hd).reshape(-1)
            codes_f.write(np.ascontiguousarray(codes, dtype="<i1").tobytes())
            scales_f.write(np.ascontiguousarray(scale, dtype="<f4").tobytes())
            manifest_weights.append({
                "name": name,
                "kind": "quant",
                "shape": list(codes.shape),
                "codesOffset": codes_offset,
                "scaleOffset": scale_offset,
            })
            codes_offset += int(codes.size)
            scale_offset += int(scale.size)

        def write_norm(name: str, key: str) -> None:
            nonlocal norm_offset
            tensor = f.get_tensor(key)
            w = tensor.to(torch.float32).numpy()
            del tensor
            norms_f.write(np.ascontiguousarray(w, dtype="<f4").tobytes())
            manifest_weights.append({
                "name": name,
                "kind": "norm",
                "shape": list(w.shape),
                "offset": norm_offset,
            })
            norm_offset += int(w.size)

        write_quant("embedTokens", "model.embed_tokens.weight")
        for i in range(num_layers):
            p = f"model.layers.{i}.self_attn"
            m = f"model.layers.{i}.mlp"
            write_norm(f"layers.{i}.attnNorm", f"model.layers.{i}.input_layernorm.weight")
            write_quant(f"layers.{i}.wq", f"{p}.q_proj.weight", permute=(num_heads, head_dim))
            write_quant(f"layers.{i}.wk", f"{p}.k_proj.weight", permute=(num_kv_heads, head_dim))
            write_quant(f"layers.{i}.wv", f"{p}.v_proj.weight")
            write_quant(f"layers.{i}.wo", f"{p}.o_proj.weight")
            write_norm(f"layers.{i}.ffnNorm", f"model.layers.{i}.post_attention_layernorm.weight")
            write_quant(f"layers.{i}.wGate", f"{m}.gate_proj.weight")
            write_quant(f"layers.{i}.wUp", f"{m}.up_proj.weight")
            write_quant(f"layers.{i}.wDown", f"{m}.down_proj.weight")
            print(f"[convert_weights] layer {i + 1}/{num_layers} done", flush=True)
        write_norm("finalNorm", "model.norm.weight")
        write_quant("lmHead", "lm_head.weight")

    manifest = {
        # Null when the operator did not say. Recorded either way, so "we do not
        # know" is a fact the artifact carries rather than something a reader
        # has to infer from an absent key.
        "sourceRepo": args.source_repo,
        "license": args.license,
        "generatedBy": "llm/tools/convert_weights.py",
        "sourceModelDir": str(model_dir),
        "config": config,
        "weights": manifest_weights,
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    sizes = {
        "codes": codes_path.stat().st_size,
        "scales": scales_path.stat().st_size,
        "norms": norms_path.stat().st_size,
    }
    sizes["total"] = sum(sizes.values())
    return sizes


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model-dir", required=True, type=Path, help="HF checkpoint directory (config.json + model.safetensors)")
    ap.add_argument("--out-dir", required=True, type=Path, help="output directory for manifest.json + weights.*.bin")
    # Provenance, carried by the artifact rather than by whoever remembers.
    # This converter takes a directory, so it cannot know the upstream on its
    # own -- but the operator does, and an artifact that cannot name its
    # source is one nobody can decide anything about later (issue #190).
    ap.add_argument("--source-repo", default=None,
                    help="Where the checkpoint came from, e.g. sbintuitions/sarashina2.2-1b-instruct-v0.1")
    ap.add_argument("--license", default=None,
                    help="Its licence, e.g. mit. Written into the manifest verbatim.")
    args = ap.parse_args()

    sizes = convert(args.model_dir, args.out_dir)

    print(f"wrote {args.out_dir / 'manifest.json'}")
    print(f"codes:  {sizes['codes']:,} bytes")
    print(f"scales: {sizes['scales']:,} bytes")
    print(f"norms:  {sizes['norms']:,} bytes")
    print(f"total:  {sizes['total']:,} bytes ({sizes['total'] / 1024 / 1024:.1f} MiB)")


if __name__ == "__main__":
    main()
