#!/usr/bin/env python3
"""Exports Anima's two tokenizers, and the fixtures that pin them.

Issue #170 stage 5. `AnimaTokenizer` runs two tokenizers over the same prompt
(`comfy/text_encoders/anima.py:21`): a Qwen2 byte-level BPE whose ids condition
the 0.6B, and a T5 SentencePiece unigram whose ids index the adapter's own
embedding table. Both have to be right; conditioning on nearly the right words
is not a failure that announces itself.

The BPE side reuses `llm/tokenizer-bpe.ts` and its `BpeVocab` format unchanged.

The unigram side needs one thing `llm/tokenizer.ts` refuses: T5's normalizer is
a `Precompiled` charsmap, and that class throws on anything but `identity`
rather than silently mis-normalizing. Rather than carry a 316 kB charsmap into a
browser bundle, this tool **measures** what the charsmap actually does:

    for every codepoint in the BMP:
        precompiled(c) vs unicodedata.normalize("NFKC", c)

Across 0x00..0xFFFF exactly **51 codepoints differ**, and they are emitted below
as an exception table. So the port is `NFKC` — which every JavaScript engine has
built in — plus twenty entries, rather than an approximation of a blob. The
table is derived here on every run, never transcribed, so a different checkpoint
with a different charsmap produces a different table instead of a wrong one.

The exceptions are what an older Unicode's NFKC did, plus NMT's treatment of
controls and exotic spaces: U+32FF, U+A7F2..4 and U+AB69 postdate the charsmap
and are left alone by it; the zero-widths, U+FEFF, U+2028/9 and U+FFFD become a
space; U+7F/8F/9F vanish; U+FF5E is not folded to ASCII tilde.

    PYTHONPATH=/tmp/comfy-venv/lib/python3.12/site-packages \\
        /tmp/comfy-venv/bin/python examples/anima/tools/gen_tokenizer_data.py
"""
from __future__ import annotations

import argparse
import json
import sys
import unicodedata
from pathlib import Path

COMFY = Path("/tmp/ComfyUI")
HERE = Path(__file__).resolve().parent

# Chosen so a port passing all of them cannot be doing something simple and
# wrong. Every line says what it is for, following `llm/tools/gen_bpe_fixtures.py`.
CASES: list[tuple[str, str]] = [
    ("empty", ""),
    ("plain", "a cat"),
    ("the golden's prompt", "1girl, silver hair, red eyes, looking at viewer, detailed background"),
    ("danbooru tags", "masterpiece, best quality, 1girl, solo, looking at viewer"),
    ("leading space", " leading"),
    ("trailing space", "trailing "),
    # `Strip` is right-only and `Replace` turns runs of 2+ spaces into U+2581.
    # A port that collapses runs to one space, or strips both ends, differs here
    # and nowhere else.
    ("double space", "a  b"),
    ("triple space", "a   b"),
    ("both ends", "  both  "),
    ("tab and newline", "a\tb\nc"),
    ("japanese", "女の子、銀髪、赤い目"),
    ("halfwidth katakana", "ｱｲｳｴｵ"),      # NFKC folds these
    ("fullwidth tilde", "a～b"),           # an exception: NOT folded to ASCII
    ("circled digits", "①②③"),            # NFKC folds these
    ("ligature", "ﬁne"),                    # NFKC folds this
    ("reiwa", "㋿"),                        # an exception: NOT folded to 令和
    ("zero width", "a​b"),            # an exception: becomes a space
    ("bom", "﻿hello"),
    ("emoji", "a 🐱 b"),                    # outside the BMP, and byte-fallback
    ("accent composed", "café"),
    ("accent decomposed", "café"),   # NFKC must compose it to match
    ("unknown bytes", "\x00\x01"),
    ("long", "highly detailed illustration of a silver haired girl, " * 4),
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=HERE.parent / "fixtures")
    args = ap.parse_args()

    if not COMFY.exists():
        sys.exit(f"{COMFY} not found — see gen_block_golden.py for the clone command")

    from tokenizers import Tokenizer
    from transformers import AutoTokenizer, Qwen2Tokenizer

    qwen_dir = COMFY / "comfy/text_encoders/qwen25_tokenizer"
    t5_dir = COMFY / "comfy/text_encoders/t5_tokenizer"

    # --- the Qwen side, in `llm/tokenizer-bpe.ts`'s own format ---
    qwen_hf = Qwen2Tokenizer.from_pretrained(qwen_dir)
    backend = json.loads(qwen_hf.backend_tokenizer.to_str())
    qwen_vocab = {
        "vocab": backend["model"]["vocab"],
        "merges": backend["model"]["merges"],
        "addedTokens": [
            {"id": t["id"], "content": t["content"], "special": t["special"]}
            for t in backend.get("added_tokens", [])
        ],
        "normalizer": (backend.get("normalizer") or {}).get("type"),
        "splitPattern": None,
    }
    # The pre-tokenizer's split pattern, taken from the tokenizer rather than
    # from the model card. Qwen's is a `Sequence` whose first entry is the Split.
    pre = backend.get("pre_tokenizer") or {}
    for entry in ([pre] if pre.get("type") != "Sequence" else pre.get("pretokenizers", [])):
        if entry.get("type") == "Split":
            qwen_vocab["splitPattern"] = entry["pattern"].get("Regex")
    print(f"qwen: {len(qwen_vocab['vocab'])} pieces, {len(qwen_vocab['merges'])} merges, "
          f"normalizer {qwen_vocab['normalizer']}, split {'yes' if qwen_vocab['splitPattern'] else 'NO'}")

    # --- the T5 side ---
    t5_backend_tok = Tokenizer.from_file(str(t5_dir / "tokenizer.json"))
    t5_backend = json.loads(t5_backend_tok.to_str())
    model = t5_backend["model"]
    if model["type"] != "Unigram":
        sys.exit(f"expected a Unigram model, got {model['type']}")

    # The measurement described in the docstring. `t.normalizer[0]` is the
    # `Precompiled` step alone — the `Strip` and `Replace` after it are already
    # explicit in the JSON and are ported directly.
    precompiled = t5_backend_tok.normalizer[0]
    exceptions: dict[str, str] = {}
    # From 0, not from 0x20. The first pass started at the space and so never
    # looked at the controls, where the charsmap does its most consequential
    # work: it maps tab and newline to a space, which NFKC leaves alone. The
    # `tab and newline` fixture is what caught it.
    for cp in range(0x00, 0x10000):
        if 0xD800 <= cp <= 0xDFFF:
            continue
        c = chr(cp)
        got = precompiled.normalize_str(c)
        if got != unicodedata.normalize("NFKC", c):
            exceptions[f"{cp:04x}"] = got
    print(f"t5: {len(model['vocab'])} pieces, byte_fallback {model['byte_fallback']}, "
          f"{len(exceptions)} codepoints where the charsmap differs from NFKC")
    print(f"     of which below 0x20: {sum(1 for k in exceptions if int(k, 16) < 0x20)}")

    normalizer_steps = t5_backend["normalizer"]["normalizers"]
    if [s["type"] for s in normalizer_steps] != ["Precompiled", "Strip", "Replace"]:
        sys.exit(f"unexpected normalizer sequence: {[s['type'] for s in normalizer_steps]}")
    strip, replace = normalizer_steps[1], normalizer_steps[2]

    t5_vocab = {
        "pieces": model["vocab"],
        "unkId": model["unk_id"],
        "byteFallback": model["byte_fallback"],
        "addedTokens": [
            {"id": t["id"], "content": t["content"], "special": t["special"]}
            for t in t5_backend.get("added_tokens", [])
        ],
        "normalizer": {
            # NFKC plus the measured exceptions, in place of the charsmap.
            "nfkcExceptions": exceptions,
            "stripLeft": strip["strip_left"],
            "stripRight": strip["strip_right"],
            "collapsePattern": replace["pattern"]["Regex"],
            "collapseTo": replace["content"],
        },
        "metaspace": t5_backend["pre_tokenizer"],
        # `TemplateProcessing` appends `</s>`; recorded as the id so the port
        # does not have to parse a template it will only ever see one shape of.
        "eosId": t5_backend_tok.token_to_id("</s>"),
    }
    if t5_vocab["metaspace"]["type"] != "Metaspace":
        sys.exit(f"expected a Metaspace pre-tokenizer, got {t5_vocab['metaspace']['type']}")

    # --- fixtures: both tokenizers over the same cases ---
    fixtures = []
    for label, text in CASES:
        fixtures.append({
            "label": label,
            "text": text,
            # `has_start_token=False, has_end_token=False` on the Qwen side.
            "qwen": qwen_hf(text, add_special_tokens=False)["input_ids"],
            # The T5 side keeps its `</s>` — appended by the backend's own
            # `TemplateProcessing`, so `encode` already carries it. Adding one
            # here as well is the mistake this comment exists to prevent; it
            # was made once and caught by the fixture disagreeing with the
            # encoder golden, which tokenizes through `transformers` instead.
            "t5": t5_backend_tok.encode(text).ids,
        })
    longest = max(fixtures, key=lambda f: len(f["t5"]))
    print(f"{len(fixtures)} cases, longest {longest['label']}: "
          f"{len(longest['qwen'])} qwen / {len(longest['t5'])} t5 ids")

    # The Qwen vocabulary is **not** written. `llm/data/qwen-qwen3-4b.bpe-vocab
    # .json` already holds it: ComfyUI's `qwen25_tokenizer` and Qwen3-4B's are
    # the same 151,643 pieces, the same 151,387 merges, the same split pattern
    # and the same 26 added tokens — compared field by field here rather than
    # assumed from the family name. Committing a second 6 MB copy of a file this
    # repository already carries would be two things to keep in step.
    existing = Path(__file__).resolve().parents[3] / "llm/data/qwen-qwen3-4b.bpe-vocab.json"
    if existing.exists():
        other = json.loads(existing.read_text())
        same = all(qwen_vocab[k] == other.get(k) for k in ("vocab", "merges", "splitPattern"))
        same = same and {(t["id"], t["content"]) for t in qwen_vocab["addedTokens"]} == {
            (t["id"], t["content"]) for t in other.get("addedTokens", [])
        }
        print(f"qwen vocab identical to {existing.name}: {same}")
        if not same:
            sys.exit(
                f"{existing} differs from ComfyUI's qwen25_tokenizer. The port reads that file; "
                "either it is the wrong vocabulary or Anima needs its own copy written here."
            )
    else:
        sys.exit(f"{existing} not found — llm/tools/gen_bpe_fixtures.py writes it")

    args.out.mkdir(parents=True, exist_ok=True)
    for name, blob in (
        ("t5.unigram-vocab.json", t5_vocab),
        ("tokenizer-fixtures.json", {"cases": fixtures}),
    ):
        out = args.out / name
        out.write_text(json.dumps(blob, ensure_ascii=False))
        print(f"wrote {out} ({out.stat().st_size / 1e6:.2f} MB)")


if __name__ == "__main__":
    main()
