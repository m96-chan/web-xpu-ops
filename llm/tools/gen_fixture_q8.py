#!/usr/bin/env python3
"""Generates llm/fixtures/tiny_q8.* — issue #105's int8 tiny fixture.

Builds the *same* tiny random model `gen_fixture.py` does (same seeds, same
config — imported from it rather than duplicated), quantizes every Linear
weight and the embedding table per-row absmax int8 (`quant_common.py`, the
same rounding rule `llm/weights-q8.ts` documents for the TS side), substitutes
the **dequantized** weights back into the model, and runs the identical
8-token-prefill / 4-step-greedy-decode loop `gen_fixture.py` runs.

The resulting `prefill_logits` / `decode_tokens` / `decode_logits` therefore
have the quantization error baked in — this is deliberately not the same
reference `gen_fixture.py` already committed (`tiny.logits.bin`), because
`LlamaEngineQ8` computes with genuinely quantized weights and comparing it
against f32-exact logits would be checking the wrong thing. Issue #105's own
wording: "Python側でもper-row absmax int8→dequantした重みでforwardした正解を焼く".

## Substituting dequantized weights without disturbing RoPE

`gen_fixture.py#collect_weights` returns `wq`/`wk` **already permuted**
(`permute_rope_channels`) for the TS engine's benefit — but HF's own forward
pass expects `q_proj.weight`/`k_proj.weight` in its *own* (unpermuted)
channel order (its `rotate_half` disagrees with `ops/rope`'s pairing; see
`llm/weights.ts#permuteRopeChannels`'s module doc for why). So this script
quantizes state_dict's **raw, unpermuted** `q_proj`/`k_proj` weights for the
model-substitution step (dequantizing back into HF's own layout, safe to
`load_state_dict`), and separately quantizes+permutes the row-order the
*fixture file* stores for `LlamaEngineQ8` (`llm/weights.ts#permuteRopeChannels`
applied to the int8 **codes** — permuting rows and quantizing per row commute,
since both act independently per output row: `quantize(permute(w))` and
`permute(quantize(w))` are the identical array. Quantizing raw weights once
and permuting the int8 codes therefore needs no second, redundant
quantization pass.)

## Reproducing

    source third_party/venv/bin/activate   # from technologies.moe/alibi-ai
    python llm/tools/gen_fixture_q8.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import gen_fixture as base  # noqa: E402
from quant_common import dequantize_per_row, quantize_per_row  # noqa: E402

FIXTURES_DIR = HERE.parent / "fixtures"

# Names quantized as `[N, K]` Linear weights (permuted for wq/wk, per-row
# int8+scale for every other entry) vs. left f32 (norms). `embedTokens` and
# `lmHead` are quantized too — the same convention as every other Linear —
# even though the *tiny* model has no meaningful embedding/lm_head size
# concern; issue #105's real converter treats them identically, and using one
# code path for both here is what makes this fixture actually exercise that
# converter's shape, not a simplified stand-in for it.
NORM_SUFFIXES = ("attnNorm", "ffnNorm")


def is_norm(name: str) -> bool:
    return name == "finalNorm" or any(name.endswith(f".{suf}") for suf in NORM_SUFFIXES)


def quantize_and_substitute(config, model) -> tuple[list[tuple[str, np.ndarray, np.ndarray]], list[tuple[str, np.ndarray]]]:
    """Returns (quantized entries for the fixture file, norm entries), and
    mutates nothing on `model` directly — the caller loads a new state_dict
    built from this function's dequantized copies.
    """
    sd = model.state_dict()
    heads = config.num_attention_heads
    kv_heads = config.num_key_value_heads
    head_dim = config.head_dim

    raw_entries: list[tuple[str, str, np.ndarray]] = [  # (fixtureName, stateDictKey, rawWeight)
        ("embedTokens", "model.embed_tokens.weight", sd["model.embed_tokens.weight"].numpy()),
    ]
    for i in range(config.num_hidden_layers):
        p = f"model.layers.{i}.self_attn"
        m = f"model.layers.{i}.mlp"
        raw_entries += [
            (f"layers.{i}.wq", f"{p}.q_proj.weight", sd[f"{p}.q_proj.weight"].numpy()),
            (f"layers.{i}.wk", f"{p}.k_proj.weight", sd[f"{p}.k_proj.weight"].numpy()),
            (f"layers.{i}.wv", f"{p}.v_proj.weight", sd[f"{p}.v_proj.weight"].numpy()),
            (f"layers.{i}.wo", f"{p}.o_proj.weight", sd[f"{p}.o_proj.weight"].numpy()),
            (f"layers.{i}.wGate", f"{m}.gate_proj.weight", sd[f"{m}.gate_proj.weight"].numpy()),
            (f"layers.{i}.wUp", f"{m}.up_proj.weight", sd[f"{m}.up_proj.weight"].numpy()),
            (f"layers.{i}.wDown", f"{m}.down_proj.weight", sd[f"{m}.down_proj.weight"].numpy()),
        ]
    raw_entries.append(("lmHead", "lm_head.weight", sd["lm_head.weight"].numpy()))

    quantized_for_fixture: list[tuple[str, np.ndarray, np.ndarray]] = []
    new_sd = {k: v.clone() for k, v in sd.items()}

    for fixture_name, sd_key, raw in raw_entries:
        codes, scale = quantize_per_row(raw)
        new_sd[sd_key] = torch.from_numpy(dequantize_per_row(codes, scale))

        if fixture_name.endswith(".wq"):
            codes = base.permute_rope_channels(codes, heads, head_dim)
            scale = base.permute_rope_channels(scale.reshape(-1, 1), heads, head_dim).reshape(-1)
        elif fixture_name.endswith(".wk"):
            codes = base.permute_rope_channels(codes, kv_heads, head_dim)
            scale = base.permute_rope_channels(scale.reshape(-1, 1), kv_heads, head_dim).reshape(-1)

        quantized_for_fixture.append((fixture_name, codes, scale))

    norm_entries: list[tuple[str, np.ndarray]] = []
    for i in range(config.num_hidden_layers):
        norm_entries.append((f"layers.{i}.attnNorm", sd[f"model.layers.{i}.input_layernorm.weight"].numpy()))
        norm_entries.append((f"layers.{i}.ffnNorm", sd[f"model.layers.{i}.post_attention_layernorm.weight"].numpy()))
    norm_entries.append(("finalNorm", sd["model.norm.weight"].numpy()))

    model.load_state_dict(new_sd)
    return quantized_for_fixture, norm_entries


def write_binary_bytes(path: Path, chunks: list[bytes]) -> None:
    with open(path, "wb") as f:
        for chunk in chunks:
            f.write(chunk)


def write_binary_f32(path: Path, arrays: list[np.ndarray]) -> None:
    with open(path, "wb") as f:
        for arr in arrays:
            f.write(np.ascontiguousarray(arr, dtype="<f4").tobytes())


def main() -> None:
    FIXTURES_DIR.mkdir(parents=True, exist_ok=True)
    config, model = base.build_model()

    torch.manual_seed(base.PROMPT_SEED)
    prompt = torch.randint(0, config.vocab_size, (1, base.PROMPT_LEN))

    quantized_entries, norm_entries = quantize_and_substitute(config, model)
    model.eval()

    prefill_logits, decode_tokens, decode_logits = base.greedy_run(model, prompt)

    codes_offset = 0
    scale_offset = 0
    weight_manifest = []
    codes_chunks: list[bytes] = []
    scale_chunks: list[bytes] = []
    for name, codes, scale in quantized_entries:
        n_codes = int(codes.size)
        n_scale = int(scale.size)
        weight_manifest.append({
            "name": name,
            "kind": "quant",
            "shape": list(codes.shape),
            "codesOffset": codes_offset,
            "scaleOffset": scale_offset,
        })
        codes_chunks.append(np.ascontiguousarray(codes, dtype="<i1").tobytes())
        scale_chunks.append(np.ascontiguousarray(scale, dtype="<f4").tobytes())
        codes_offset += n_codes
        scale_offset += n_scale
    write_binary_bytes(FIXTURES_DIR / "tiny_q8.codes.bin", codes_chunks)
    write_binary_bytes(FIXTURES_DIR / "tiny_q8.scales.bin", scale_chunks)

    norm_offset = 0
    for name, arr in norm_entries:
        weight_manifest.append({
            "name": name,
            "kind": "norm",
            "shape": list(arr.shape),
            "offset": norm_offset,
        })
        norm_offset += int(arr.size)
    write_binary_f32(FIXTURES_DIR / "tiny_q8.norms.bin", [arr for _, arr in norm_entries])

    logits_manifest = {
        "prefill": {"shape": list(prefill_logits.shape), "offset": 0},
        "decode": {"shape": list(decode_logits.shape), "offset": int(prefill_logits.size)},
    }
    write_binary_f32(FIXTURES_DIR / "tiny_q8.logits.bin", [prefill_logits, decode_logits])

    manifest = {
        "generatedBy": "llm/tools/gen_fixture_q8.py",
        "transformersVersion": __import__("transformers").__version__,
        "torchVersion": torch.__version__,
        "weightSeed": base.WEIGHT_SEED,
        "promptSeed": base.PROMPT_SEED,
        "config": {
            "numLayers": config.num_hidden_layers,
            "hiddenSize": config.hidden_size,
            "numHeads": config.num_attention_heads,
            "numKvHeads": config.num_key_value_heads,
            "headDim": config.head_dim,
            "ffnHidden": config.intermediate_size,
            "vocabSize": config.vocab_size,
            "ropeTheta": config.rope_parameters["rope_theta"],
            "rmsNormEps": config.rms_norm_eps,
            "tieEmbeddings": config.tie_word_embeddings,
        },
        "promptTokens": prompt[0].tolist(),
        "decodeTokens": decode_tokens,
        "weights": weight_manifest,
        "logits": logits_manifest,
    }
    (FIXTURES_DIR / "tiny_q8.manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    print(f"wrote {FIXTURES_DIR / 'tiny_q8.manifest.json'}")
    print(f"wrote {FIXTURES_DIR / 'tiny_q8.codes.bin'} ({codes_offset} bytes)")
    print(f"wrote {FIXTURES_DIR / 'tiny_q8.scales.bin'} ({scale_offset * 4} bytes)")
    print(f"wrote {FIXTURES_DIR / 'tiny_q8.norms.bin'} ({norm_offset * 4} bytes)")
    print(f"prompt tokens: {prompt[0].tolist()}")
    print(f"decode tokens: {decode_tokens}")


if __name__ == "__main__":
    main()
