#!/usr/bin/env python3
"""Export a SentencePiece unigram `tokenizer.model` (plus its HF
`tokenizer_config.json`/`special_tokens_map.json` siblings) into the single
JSON vocabulary file `llm/tokenizer.ts` reads at runtime.

Why a Python tool at all: `tokenizer.model` is a protobuf
(`sentencepiece_model_pb2.ModelProto`) and this repository ships no protobuf
runtime for TypeScript. Parsing it once, offline, into plain JSON is cheaper
and more auditable than carrying a JS protobuf decoder into a browser bundle
that only ever needs to read the result.

This tool is generic — it works on any SentencePiece **unigram** model with an
HF-style config directory beside it. It does not hardcode any particular
model's path; the model directory is a required CLI argument. The vocabulary
JSON committed under `llm/data/` in this repository happens to have been
generated from the Sarashina2.2-1B-Instruct tokenizer (see issue #101 and
`llm/tokenizer.ts` for how the normalizer/byte-fallback settings recorded here
were confirmed against that specific model rather than assumed).

## What was read from the real `tokenizer.model`, not assumed

Ground truth for every field below is `m.trainer_spec` / `m.normalizer_spec`
in the parsed `ModelProto`, dumped and inspected with the `sentencepiece`
Python package before any TypeScript was written:

  - `trainer_spec.model_type`      = UNIGRAM
  - `trainer_spec.byte_fallback`   = true   (256 `<0xXX>` BYTE pieces exist)
  - `trainer_spec.unk_id/bos_id/eos_id` = 0 / 1 / 2
  - `normalizer_spec.name`                 = "identity"  — NOT NFKC. No
    Unicode normalization is applied at all: composed and decomposed forms of
    the same visible glyph (e.g. U+304C "が" vs U+304B U+3099 "か" + combining
    voiced sound mark) tokenize to *different* ids, verified directly against
    `SentencePieceProcessor.encode`.
  - `normalizer_spec.add_dummy_prefix`     = false — no implicit leading `▁`.
  - `normalizer_spec.remove_extra_whitespaces` = false — runs of spaces are
    preserved exactly, never collapsed.
  - `normalizer_spec.escape_whitespaces`   = true — only the literal ASCII
    space (U+0020) is replaced with `▁` (U+2581) before segmentation. Other
    whitespace (tab, newline, NBSP U+00A0, full-width space U+3000, ...) is
    left as-is and, having no single-character NORMAL piece of its own, is
    represented via byte fallback.
  - `normalizer_spec.precompiled_charsmap` = empty, `normalization_rule_tsv`
    = empty — nothing to replay; "identity" really means identity.

## Piece types kept vs. dropped from the search vocabulary

`sentencepiece_model_pb2.ModelProto.SentencePiece.Type`: NORMAL=1, UNKNOWN=2,
CONTROL=3, USER_DEFINED=4, UNUSED=5, BYTE=6. This model has NORMAL, UNKNOWN,
CONTROL and BYTE pieces (no USER_DEFINED/UNUSED).

CONTROL pieces — `<s>`, `</s>`, `<pad>`, and the chat-role markers
(`<|system|>`, `<|user|>`, ...) added via `trainer_spec.control_symbols` —
are **not** matched against raw input text by SentencePiece's own Viterbi
search. This was verified directly: `sp.encode("<|system|>")` does not return
the control token's id, it falls through to byte-by-byte segmentation of the
literal characters `<`, `|`, `s`, ... A real front end (HF `tokenizers`,
llama.cpp, ...) recognizes these strings via a separate literal-substring
pre-tokenization pass, matched against the *added tokens* table
(`tokenizer_config.json`'s `added_tokens_decoder`), before anything is handed
to the unigram model. `llm/tokenizer.ts` reproduces that pass using the
`addedTokens` list this script exports below; the CONTROL/UNKNOWN pieces
themselves are exported for completeness (id -> piece text lookups, decode)
but excluded from the Viterbi trie.

Usage:

    python3 llm/tools/export_tokenizer.py \
        --model-dir /path/to/hf/model/dir \
        --out llm/data/<name>.vocab.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from sentencepiece import sentencepiece_model_pb2 as pb2

# Mirrors sentencepiece_model_pb2.ModelProto.SentencePiece.Type. Not imported
# by name because the generated protobuf module exposes it as a nested enum
# whose Python binding varies across sentencepiece versions; the integer
# values are the actual wire values and are stable across the model file
# format itself.
PIECE_TYPE_NORMAL = 1
PIECE_TYPE_UNKNOWN = 2
PIECE_TYPE_CONTROL = 3
PIECE_TYPE_USER_DEFINED = 4
PIECE_TYPE_UNUSED = 5
PIECE_TYPE_BYTE = 6


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-dir",
        required=True,
        type=Path,
        help="Directory containing tokenizer.model, tokenizer_config.json, special_tokens_map.json",
    )
    parser.add_argument("--out", required=True, type=Path, help="Output JSON path")
    return parser.parse_args()


def load_model_proto(model_path: Path) -> pb2.ModelProto:
    m = pb2.ModelProto()
    m.ParseFromString(model_path.read_bytes())
    return m


def build_added_tokens(tokenizer_config: dict) -> list[dict]:
    decoder = tokenizer_config.get("added_tokens_decoder", {})
    added = []
    for id_str, entry in decoder.items():
        added.append(
            {
                "id": int(id_str),
                "content": entry["content"],
                "special": bool(entry.get("special", False)),
            }
        )
    added.sort(key=lambda e: e["id"])
    return added


def main() -> int:
    args = parse_args()
    model_dir: Path = args.model_dir

    model_path = model_dir / "tokenizer.model"
    config_path = model_dir / "tokenizer_config.json"
    special_map_path = model_dir / "special_tokens_map.json"

    m = load_model_proto(model_path)
    tokenizer_config = json.loads(config_path.read_text(encoding="utf-8"))
    special_tokens_map = (
        json.loads(special_map_path.read_text(encoding="utf-8")) if special_map_path.exists() else {}
    )

    ts = m.trainer_spec
    ns = m.normalizer_spec

    if ts.model_type != pb2.TrainerSpec.ModelType.UNIGRAM:
        print(
            f"error: {model_path} is not a UNIGRAM model (model_type={ts.model_type}); "
            "this exporter and llm/tokenizer.ts only implement unigram Viterbi",
            file=sys.stderr,
        )
        return 1

    if ns.precompiled_charsmap or ns.normalization_rule_tsv:
        print(
            "error: normalizer_spec carries a precompiled_charsmap/normalization_rule_tsv "
            "this exporter does not replay (only the empty/'identity' normalizer is "
            "implemented — see the module docstring). Refusing to silently drop rules.",
            file=sys.stderr,
        )
        return 1

    pieces = []
    for p in m.pieces:
        pieces.append([p.piece, p.score, int(p.type)])

    type_counts: dict[int, int] = {}
    for _piece, _score, t in pieces:
        type_counts[t] = type_counts.get(t, 0) + 1

    out = {
        "vocabSize": len(pieces),
        "byteFallback": bool(ts.byte_fallback),
        "unkId": ts.unk_id,
        "bosId": ts.bos_id,
        "eosId": ts.eos_id,
        "unkSurface": ts.unk_surface or " ⁇ ",
        "normalizer": {
            "name": ns.name,
            "addDummyPrefix": bool(ns.add_dummy_prefix),
            "removeExtraWhitespaces": bool(ns.remove_extra_whitespaces),
            "escapeWhitespaces": bool(ns.escape_whitespaces),
        },
        "addedTokens": build_added_tokens(tokenizer_config),
        "specialTokens": {
            key: value.get("content") if isinstance(value, dict) else value
            for key, value in special_tokens_map.items()
        },
        # [piece, score, type] tuples, ordered by id (index in this array IS
        # the token id) — an array-of-tuples rather than array-of-objects to
        # avoid repeating "piece"/"score"/"type" keys 100k+ times over.
        "pieces": pieces,
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")

    size = args.out.stat().st_size
    print(f"wrote {args.out} ({size:,} bytes, {size / 1024 / 1024:.2f} MiB)")
    print(f"vocabSize={out['vocabSize']} piece type counts={type_counts}")
    print(f"addedTokens={len(out['addedTokens'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
