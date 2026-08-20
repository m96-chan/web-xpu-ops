#!/usr/bin/env python3
"""Generates llm/fixtures/tiny.* — the fixture issue #98's engine is checked against.

Builds a tiny, randomly-initialised llama-architecture model with `transformers`
(the real HF implementation, not a hand-rolled one — rule 7: "テストケースに
困ったら PyTorch に従う"), runs an 8-token prefill and a 4-step greedy decode,
and writes:

  llm/fixtures/tiny.manifest.json  config, seeds, prompt tokens, decode tokens,
                                    and a byte layout for the two binary files
  llm/fixtures/tiny.weights.bin    every weight tensor, f32 little-endian,
                                    concatenated in manifest["weights"] order
  llm/fixtures/tiny.logits.bin     prefill logits [8, 256] then decode logits
                                    [4, 256], f32 little-endian

The shape (2 layers, hidden 64, 4 query / 2 KV heads, FFN 128, vocab 256,
rope_theta 10000) is issue #98's own — small enough to run in milliseconds,
large enough that GQA is a real 2-groups-of-2 split and RoPE rotates more than
one pair per head.

## Reproducing

    source third_party/venv/bin/activate   # from technologies.moe/alibi-ai,
                                            # per this issue's instructions —
                                            # transformers 5.3.0 / torch 2.10
    python llm/tools/gen_fixture.py

Deterministic: `WEIGHT_SEED` fixes `torch.manual_seed` before the model is
constructed (weight init reads the RNG), `PROMPT_SEED` fixes it again before
the prompt tokens are drawn, and `attn_implementation="eager"` pins the
attention math to the plain, unfused kernel rather than whatever SDPA backend
happens to be fastest on the machine that ran this.

## The RoPE channel permutation

`q_proj.weight` and `k_proj.weight` are **not** written out as HF stores them.
HF Llama's RoPE (`rotate_half`) pairs channel `i` with channel `i + headDim/2`;
`ops/rope` (this repo's kernel) pairs adjacent channels `2i` / `2i+1`. Both are
the same rotation at the same angle, just numbered differently within a head —
`llm/weights.ts#permuteRopeChannels` derives why relabelling the *rows* that
produce those channels (this function, `permute_rope_channels` below, is the
same relabelling in numpy) reproduces identical attention scores, and
`llm/rope-permutation.test.ts` checks that derivation numerically. Every other
weight is written out unchanged.
"""
from __future__ import annotations

import json
import struct
from pathlib import Path

import numpy as np
import torch
from transformers import LlamaConfig, LlamaForCausalLM

WEIGHT_SEED = 0
PROMPT_SEED = 123
DECODE_STEPS = 4
PROMPT_LEN = 8

FIXTURE_CONFIG = dict(
    vocab_size=256,
    hidden_size=64,
    intermediate_size=128,
    num_hidden_layers=2,
    num_attention_heads=4,
    num_key_value_heads=2,
    tie_word_embeddings=False,
    attention_bias=False,
    mlp_bias=False,
    rms_norm_eps=1e-5,
    rope_parameters={"rope_theta": 10000.0, "rope_type": "default"},
    attn_implementation="eager",
)

HERE = Path(__file__).resolve().parent
FIXTURES_DIR = HERE.parent / "fixtures"


def permute_rope_channels(w: np.ndarray, heads: int, head_dim: int) -> np.ndarray:
    """`[heads*headDim, inFeatures]` -> same shape, rows reordered per head.

    New row `2i` is old row `i`; new row `2i+1` is old row `i + headDim/2`, for
    `i` in `[0, headDim/2)`. Transcribed from `llm/weights.ts#permuteRopeChannels`,
    which documents the derivation this is a numpy restatement of.
    """
    in_features = w.shape[1]
    half = head_dim // 2
    grouped = w.reshape(heads, head_dim, in_features)
    out = np.empty_like(grouped)
    out[:, 0::2, :] = grouped[:, :half, :]
    out[:, 1::2, :] = grouped[:, half:, :]
    return out.reshape(heads * head_dim, in_features)


def build_model() -> tuple[LlamaConfig, LlamaForCausalLM]:
    config = LlamaConfig(**FIXTURE_CONFIG)
    torch.manual_seed(WEIGHT_SEED)
    model = LlamaForCausalLM(config)
    model.eval()
    return config, model


def collect_weights(config: LlamaConfig, model: LlamaForCausalLM) -> list[tuple[str, np.ndarray]]:
    """Every weight `llm/weights.ts#LlamaWeights` needs, in the order the manifest records it."""
    sd = model.state_dict()
    heads = config.num_attention_heads
    kv_heads = config.num_key_value_heads
    head_dim = config.head_dim

    entries: list[tuple[str, np.ndarray]] = [
        ("embedTokens", sd["model.embed_tokens.weight"].numpy()),
    ]
    for i in range(config.num_hidden_layers):
        p = f"model.layers.{i}.self_attn"
        m = f"model.layers.{i}.mlp"
        wq = sd[f"{p}.q_proj.weight"].numpy()
        wk = sd[f"{p}.k_proj.weight"].numpy()
        entries += [
            (f"layers.{i}.attnNorm", sd[f"model.layers.{i}.input_layernorm.weight"].numpy()),
            (f"layers.{i}.wq", permute_rope_channels(wq, heads, head_dim)),
            (f"layers.{i}.wk", permute_rope_channels(wk, kv_heads, head_dim)),
            (f"layers.{i}.wv", sd[f"{p}.v_proj.weight"].numpy()),
            (f"layers.{i}.wo", sd[f"{p}.o_proj.weight"].numpy()),
            (f"layers.{i}.ffnNorm", sd[f"model.layers.{i}.post_attention_layernorm.weight"].numpy()),
            (f"layers.{i}.wGate", sd[f"{m}.gate_proj.weight"].numpy()),
            (f"layers.{i}.wUp", sd[f"{m}.up_proj.weight"].numpy()),
            (f"layers.{i}.wDown", sd[f"{m}.down_proj.weight"].numpy()),
        ]
    entries += [
        ("finalNorm", sd["model.norm.weight"].numpy()),
        ("lmHead", sd["lm_head.weight"].numpy()),
    ]
    return entries


def greedy_run(model: LlamaForCausalLM, prompt: torch.Tensor):
    with torch.no_grad():
        out = model(input_ids=prompt, use_cache=True)
    prefill_logits = out.logits[0].numpy().astype(np.float32)  # [PROMPT_LEN, vocab]
    past = out.past_key_values

    next_token = int(out.logits[0, -1].argmax())
    decode_tokens: list[int] = []
    decode_logits_rows: list[np.ndarray] = []
    for _ in range(DECODE_STEPS):
        with torch.no_grad():
            step = model(input_ids=torch.tensor([[next_token]]), past_key_values=past, use_cache=True)
        past = step.past_key_values
        row = step.logits[0, -1].numpy().astype(np.float32)
        next_token = int(row.argmax())
        decode_tokens.append(next_token)
        decode_logits_rows.append(row)
    decode_logits = np.stack(decode_logits_rows, axis=0)
    return prefill_logits, decode_tokens, decode_logits


def write_binary(path: Path, arrays: list[np.ndarray]) -> None:
    with open(path, "wb") as f:
        for arr in arrays:
            f.write(np.ascontiguousarray(arr, dtype="<f4").tobytes())


def main() -> None:
    FIXTURES_DIR.mkdir(parents=True, exist_ok=True)
    config, model = build_model()

    torch.manual_seed(PROMPT_SEED)
    prompt = torch.randint(0, config.vocab_size, (1, PROMPT_LEN))

    weight_entries = collect_weights(config, model)
    prefill_logits, decode_tokens, decode_logits = greedy_run(model, prompt)

    weight_manifest = []
    offset = 0
    for name, arr in weight_entries:
        n = int(arr.size)
        weight_manifest.append({"name": name, "shape": list(arr.shape), "offset": offset})
        offset += n
    write_binary(FIXTURES_DIR / "tiny.weights.bin", [arr for _, arr in weight_entries])

    logits_manifest = {
        "prefill": {"shape": list(prefill_logits.shape), "offset": 0},
        "decode": {"shape": list(decode_logits.shape), "offset": int(prefill_logits.size)},
    }
    write_binary(FIXTURES_DIR / "tiny.logits.bin", [prefill_logits, decode_logits])

    manifest = {
        "generatedBy": "llm/tools/gen_fixture.py",
        "transformersVersion": __import__("transformers").__version__,
        "torchVersion": torch.__version__,
        "weightSeed": WEIGHT_SEED,
        "promptSeed": PROMPT_SEED,
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
    (FIXTURES_DIR / "tiny.manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    print(f"wrote {FIXTURES_DIR / 'tiny.manifest.json'}")
    print(f"wrote {FIXTURES_DIR / 'tiny.weights.bin'} ({offset * 4} bytes)")
    print(f"wrote {FIXTURES_DIR / 'tiny.logits.bin'} ({(prefill_logits.size + decode_logits.size) * 4} bytes)")
    print(f"prompt tokens: {prompt[0].tolist()}")
    print(f"decode tokens: {decode_tokens}")


if __name__ == "__main__":
    main()
