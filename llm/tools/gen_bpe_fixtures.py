#!/usr/bin/env python3
"""Bakes encode/decode fixtures from the real Qwen tokenizer.

Correctness for `llm/tokenizer-bpe.ts` is whatever `tokenizers` produces, not
whatever seems reasonable. So the cases are run through the actual tokenizer
here and committed, the same arrangement `gen_fixtures.py` uses for the
SentencePiece side (#104) — which is where the byte-fallback and NFC/NFD cases
below come from, because that is where they were found to matter.

The vocabulary is exported alongside, because a fixture the port cannot load is
not a fixture. It is emitted in the two pieces a BPE needs and nothing else:
the token strings and the ordered merge list.

Usage:
    /home/m96-chan/project/therdparty/musubi-tuner/.venv/bin/python \\
        llm/tools/gen_bpe_fixtures.py --tokenizer <dir with tokenizer.json>
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

DEFAULT_TOKENIZER = "Qwen/Qwen3-4B"

# Chosen so that a port passing all of them cannot be doing something simple and
# wrong. Each line says what it is for; a case with no reason to be here is a
# case nobody will maintain.
CASES: list[tuple[str, str]] = [
    ("empty", ""),
    ("ascii-word", "hello"),
    ("ascii-sentence", "The quick brown fox jumps over the lazy dog."),
    # The pre-tokenizer's contraction rules are a literal list; each one is a
    # separate branch of that regex.
    ("contractions", "it's I'm they're we've I'll he'd"),
    ("leading-space", " hello"),
    ("double-space", "hello  world"),
    ("tabs-newlines", "a\tb\nc\r\nd"),
    ("trailing-newlines", "end\n\n\n"),
    ("only-spaces", "     "),
    ("digits", "0123456789 42 3.14 1,000"),
    # \p{N} is per-character, so long runs split differently from words.
    ("long-number", "123456789012345678901234567890"),
    ("japanese", "こんにちは世界"),
    ("japanese-mixed", "AIによる画像生成をブラウザで動かす"),
    ("japanese-punct", "「これは、テストです。」"),
    ("chinese", "通义千问是一个大语言模型"),
    ("korean", "안녕하세요 세계"),
    ("cyrillic", "Привет, мир!"),
    ("arabic-rtl", "مرحبا بالعالم"),
    # Byte-level's whole reason to exist: characters no vocabulary entry covers.
    ("emoji", "🎨🖼️✨"),
    ("emoji-zwj", "👨‍👩‍👧‍👦"),
    ("emoji-skin-tone", "👋🏽"),
    ("emoji-flag", "🇯🇵🇺🇸"),
    # NFC is the declared normalizer, so these two must not encode alike.
    ("nfc-composed", "é"),
    ("nfd-decomposed", "é"),
    ("code", "const x = arr.map((v) => v * 2);"),
    ("json", '{"key": [1, 2, {"nested": true}]}'),
    ("markdown", "# Title\n\n- item\n- **bold**\n"),
    ("url", "https://example.com/path?a=1&b=2#frag"),
    ("special-token-text", "<|im_start|>user\nhi<|im_end|>"),
    ("mixed-scripts", "日本語 and English と 123 と 🎌"),
    ("repeated", "ababababababab"),
    ("long-repeat-space", "a" + " " * 20 + "b"),
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokenizer", default=DEFAULT_TOKENIZER, help="directory or hub id")
    ap.add_argument("--name", default=None,
                    help="file stem; defaults to the hub id. A snapshot directory's name is a "
                         "content hash, which says nothing about which model it is")
    ap.add_argument("--out", type=Path, default=Path(__file__).resolve().parent.parent / "data")
    args = ap.parse_args()

    from tokenizers import Tokenizer

    src = args.tokenizer
    path = Path(src) / "tokenizer.json"
    tok = Tokenizer.from_file(str(path)) if path.exists() else Tokenizer.from_pretrained(src)
    spec = json.loads(tok.to_str())

    if spec["model"]["type"] != "BPE":
        raise SystemExit(f"expected a BPE tokenizer, got {spec['model']['type']}")

    cases = []
    for name, text in CASES:
        enc = tok.encode(text, add_special_tokens=False)
        cases.append({
            "name": name,
            "text": text,
            "ids": enc.ids,
            # Round-tripped rather than assumed equal to `text`: byte-level BPE
            # is not required to be lossless through an arbitrary string, and a
            # port should reproduce what the tokenizer actually gives back.
            "decoded": tok.decode(enc.ids),
        })

    args.out.mkdir(parents=True, exist_ok=True)
    name = (args.name or (src if "/" in str(src) else Path(src).name)).replace("/", "-").lower()

    vocab = spec["model"]["vocab"]
    merges = spec["model"]["merges"]
    added = [
        {"id": t["id"], "content": t["content"], "special": t["special"]}
        for t in spec.get("added_tokens", [])
    ]

    (args.out / f"{name}.bpe-vocab.json").write_text(json.dumps({
        "note": "Exported by llm/tools/gen_bpe_fixtures.py. Do not hand-edit.",
        "source": src,
        "normalizer": spec.get("normalizer", {}).get("type"),
        # The pre-tokenizer's split pattern is part of the contract: the port
        # has to apply the same one before any merge happens.
        "splitPattern": next(
            (p["pattern"]["Regex"] for p in spec["pre_tokenizer"]["pretokenizers"] if p["type"] == "Split"),
            None,
        ),
        "addedTokens": added,
        "vocab": vocab,
        "merges": merges,
    }, ensure_ascii=False), encoding="utf-8")

    (args.out / f"{name}.bpe-fixtures.json").write_text(json.dumps({
        "note": "Generated by llm/tools/gen_bpe_fixtures.py from the real tokenizer. Do not hand-edit.",
        "source": src,
        "cases": cases,
    }, ensure_ascii=False, indent=1), encoding="utf-8")

    lossless = sum(1 for c in cases if c["decoded"] == c["text"])
    print(f"{len(cases)} cases, {lossless} round-trip exactly")
    print(f"vocab {len(vocab)}, merges {len(merges)}, added {len(added)}, normalizer {spec.get('normalizer')}")


if __name__ == "__main__":
    main()
