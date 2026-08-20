#!/usr/bin/env python3
"""Generate encode/decode fixtures against the *real* SentencePiece unigram
tokenizer, for `llm/tokenizer.test.ts` to hold `llm/tokenizer.ts` to.

Ground truth is the `sentencepiece` Python package's `SentencePieceProcessor`
(`sp.encode` / `sp.decode_ids`) run directly against `tokenizer.model` — never
`transformers`. That distinction matters and is not cosmetic: on this
environment (`transformers` 5.3.0), `AutoTokenizer.from_pretrained(model_dir,
use_fast=False)` silently converts this UNIGRAM model into an *approximate
BPE* tokenizer (`tokenizers.models.BPE` with derived merges) because no
`tokenizer.json` ships beside `tokenizer.model`. That conversion does not
reproduce true unigram Viterbi segmentation — e.g. it segments "hello" as
`['he', 'll', 'o']` where the real unigram model (and this exporter's
`sentencepiece`-backed ground truth) produces the single piece `'hello'`.
Fixtures built from the `transformers` path would bake that divergence in and
make `llm/tokenizer.ts` wrong in a way the tests would call correct. Verified
by direct comparison; see the PR description for the full transcript.

## Special-token literal splitting

`sp.encode("<|system|>")` on its own does *not* return the control token's
id — SentencePiece's own Viterbi search never matches CONTROL-type pieces
against raw text (verified: it falls through to per-character/byte
segmentation of the literal string). A real front end recognizes these
strings via a separate pre-tokenization pass over the *added tokens* table
(`tokenizer_config.json`'s `added_tokens_decoder`) before the remaining text
spans are handed to the unigram model. This script performs that same
literal, longest-match, left-to-right split before delegating each
in-between span to `sp.encode`/`sp.decode_ids` — this is the oracle
`llm/tokenizer.ts`'s own added-token splitting is checked against.

Usage:

    python3 llm/tools/gen_fixtures.py \
        --model-dir /path/to/hf/model/dir \
        --out llm/data/<name>.fixtures.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import sentencepiece as spm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    return parser.parse_args()


def load_added_tokens(model_dir: Path) -> list[tuple[str, int]]:
    config = json.loads((model_dir / "tokenizer_config.json").read_text(encoding="utf-8"))
    decoder = config.get("added_tokens_decoder", {})
    added = [(entry["content"], int(id_str)) for id_str, entry in decoder.items()]
    # Longest content first: guards against one added token's content being a
    # prefix of another's (not the case in this vocabulary today, but the
    # split must not depend on that happening to be true).
    added.sort(key=lambda pair: len(pair[0]), reverse=True)
    return added


def split_added_tokens(text: str, added_sorted: list[tuple[str, int]]) -> list[tuple[str, str | int]]:
    """[('text', str) | ('special', id), ...], left to right, longest match."""
    out: list[tuple[str, str | int]] = []
    buf: list[str] = []
    i = 0
    n = len(text)
    while i < n:
        matched = None
        for content, tok_id in added_sorted:
            if text.startswith(content, i):
                matched = (content, tok_id)
                break
        if matched is not None:
            if buf:
                out.append(("text", "".join(buf)))
                buf = []
            out.append(("special", matched[1]))
            i += len(matched[0])
        else:
            buf.append(text[i])
            i += 1
    if buf:
        out.append(("text", "".join(buf)))
    return out


def encode_with_specials(sp: spm.SentencePieceProcessor, added_sorted, text: str) -> list[int]:
    ids: list[int] = []
    for kind, value in split_added_tokens(text, added_sorted):
        if kind == "special":
            ids.append(value)  # type: ignore[arg-type]
        else:
            ids.extend(sp.encode(value, out_type=int))
    return ids


def decode_with_specials(sp: spm.SentencePieceProcessor, added_by_id: dict[int, str], ids: list[int]) -> str:
    out: list[str] = []
    buf: list[int] = []
    for tok_id in ids:
        content = added_by_id.get(tok_id)
        if content is not None:
            if buf:
                out.append(sp.decode_ids(buf))
                buf = []
            out.append(content)
        else:
            buf.append(tok_id)
    if buf:
        out.append(sp.decode_ids(buf))
    return "".join(out)


def build_cases() -> list[tuple[str, str]]:
    """(name, text) pairs. >=50 required by issue #101; grouped by category."""
    cases: list[tuple[str, str]] = []

    def add(name: str, text: str) -> None:
        cases.append((name, text))

    # -- Japanese --------------------------------------------------------
    add("ja_greeting", "こんにちは、世界。")
    add("ja_sentence", "今日は良い天気ですね。散歩に行きましょう。")
    add("ja_katakana", "コンピュータサイエンスとプログラミング")
    add("ja_hiragana_only", "ひらがなだけのぶんしょうです")
    add("ja_kanji_dense", "人工知能技術研究開発機構")
    add("ja_mixed_number", "価格は1,980円です。")
    add("ja_particle_heavy", "私はこれをあなたに渡したいと思っています。")
    add("ja_question", "これはテストですか?")
    add("ja_long", "吾輩は猫である。名前はまだ無い。どこで生れたかとんと見当がつかぬ。")

    # -- English -----------------------------------------------------------
    add("en_simple", "hello world")
    add("en_capitalized", "Hello World")
    add("en_sentence", "The quick brown fox jumps over the lazy dog.")
    add("en_contraction", "I can't believe it's already working.")
    add("en_numbers", "The answer is 42, not 43.")
    add("en_punctuation_heavy", "Wait... what?! Really?! (No way.)")
    add("en_all_caps", "THIS IS SHOUTING")
    add("en_single_word", "hello")
    add("en_single_char", "a")

    # -- Mixed JP/EN -------------------------------------------------------
    add("mixed_ja_en", "これはtokenizerのtestです。")
    add("mixed_en_ja_brand", "AlibiちゃんはWebGPUで動きます。")
    add("mixed_code_comment", "# これは日本語のコメントです\nprint('hello')")

    # -- Code fragments ------------------------------------------------------
    add("code_python_fn", "def add(a: int, b: int) -> int:\n    return a + b\n")
    add("code_js_arrow", "const add = (a, b) => a + b;")
    add("code_json", '{"key": "value", "num": 42, "nested": {"a": [1, 2, 3]}}')
    add("code_indent_tabs", "if (x) {\n\tdoSomething();\n\treturn true;\n}")
    add("code_shell", "cd /tmp && ls -la | grep foo")
    add("code_ts_generic", "function identity<T>(x: T): T { return x; }")
    add("code_regex", r"const re = /^[a-z0-9_]+$/i;")
    add("code_sql", "SELECT id, name FROM users WHERE age > 18 ORDER BY name;")

    # -- Emoji / astral plane -----------------------------------------------
    add("emoji_single", "😀")
    add("emoji_with_text", "hello 😀 world")
    add("emoji_sequence", "😀😃😄😁")
    add("emoji_family_zwj", "👨‍👩‍👧‍👦")  # zero-width-joiner sequence
    add("emoji_flag", "🇯🇵🇺🇸")  # regional indicator pairs
    add("emoji_skin_tone", "👍🏽")  # base + Fitzpatrick modifier

    # -- Whitespace patterns --------------------------------------------------
    add("space_leading", " leading space")
    add("space_trailing", "trailing space ")
    add("space_both", " both sides ")
    add("space_double_internal", "double  internal  space")
    add("space_many", "a     b")
    add("space_only", "   ")
    add("space_single", " ")
    add("newline_single", "line one\nline two")
    add("newline_double", "paragraph one\n\nparagraph two")
    add("newline_crlf", "windows\r\nline\r\nendings")
    add("tab_single", "a\tb\tc")
    add("tab_leading", "\tindented")
    add("fullwidth_space", "　全角スペース")
    add("nbsp", " non-breaking")

    # -- Unicode normalization probes (identity normalizer, not NFKC) --------
    add("nfc_ga", "が")  # が precomposed
    add("nfd_ga", "が")  # か + combining voiced sound mark
    add("fullwidth_latin", "ＡＢＣ")  # ＡＢＣ
    add("halfwidth_katakana", "ｶﾝ")  # ｶﾝ

    # -- Zero-width / control-ish -------------------------------------------
    add("zwsp", "zero​width​space")
    add("empty", "")

    # -- Special tokens (chat-template boundary, per engine #98 format) ------
    add("special_bos_eos", "<s></s>")
    add("special_system_only", "<|system|>")
    add("special_system_prompt", "<|system|>You are a helpful assistant.</s>")
    add("special_user_turn", "<|user|>こんにちは</s>")
    add("special_assistant_turn", "<|assistant|>はい、こんにちは。</s>")
    add(
        "special_full_exchange",
        "<|system|>You are Alibi.</s><|user|>今日の天気は?</s><|assistant|>",
    )
    add("special_adjacent_no_space", "prefix<|system|>suffix")
    add("special_with_leading_space", " <|system|> hello")
    add("special_consecutive", "<s><|system|></s>")
    add("special_tool_tokens", "<|available_tools|>[{'name': 'search'}]<|tool_calls|>")
    add("special_unk_literal", "<unk>")
    add("special_pad_sep_mask_cls", "<pad><sep><mask><cls>")
    add("special_fim_tokens", "<|prefix|>def f(<|suffix|>):\n    pass<|middle|>")

    # -- URLs / punctuation-dense --------------------------------------------
    add("url", "https://technologies.moe/alibi?query=1&lang=ja")
    add("email", "contact: y_harada@devenus.com")
    add("path", "/home/user/project/src/main.ts")
    add("markdown", "# Title\n\n- item one\n- item two\n\n**bold** and _italic_")

    # -- Long / repeated ------------------------------------------------------
    # These stress the Viterbi tie-break: a run of one repeated character can
    # be tiled by the same multiset of piece-lengths in more than one order,
    # and (because piece scores are float32 and SentencePiece accumulates at
    # float32 precision) the *order* the real tokenizer picks is not
    # predictable from the exact-real-number sums alone — see
    # llm/tokenizer.ts's `bestScore` comment. Several run lengths are covered
    # since not every length lands on a tie.
    add("repeated_char", "あ" * 30)
    add("repeated_char_short", "あ" * 7)
    add("repeated_char_off_by_one", "あ" * 9)
    add("repeated_char_long", "あ" * 47)
    add("repeated_digit", "0" * 20)
    add("repeated_ascii_word", "ab" * 15)
    add("repeated_word", "test " * 20)

    # This exact case (a sweep through U+3040..U+3099, which includes several
    # unassigned/rare codepoints with no NORMAL-piece coverage) is what
    # surfaced a real bug during development: an early byte-fallback score
    # formula (`-1e6` per *byte*, rather than SentencePiece's real
    # `min_score() - 10.0` per *codepoint*) accumulated across the ~13
    # uncovered codepoints in one 90-character cycle until float32 precision
    # could no longer distinguish the ~1-2 point score gaps between competing
    # *real* segmentations later in the string — corrupting an unrelated,
    # fully-covered stretch of text. Kept as a permanent regression fixture.
    add("byte_fallback_dense_sweep", "".join(chr(0x3040 + (i % 90)) for i in range(300)))
    add("long_mixed", ("Alibiちゃんの部屋へようこそ。今日はどんな話をしましょうか? " * 5).strip())

    return cases


def main() -> int:
    args = parse_args()
    model_dir: Path = args.model_dir
    sp = spm.SentencePieceProcessor(model_file=str(model_dir / "tokenizer.model"))
    added_sorted = load_added_tokens(model_dir)
    added_by_id = {tok_id: content for content, tok_id in added_sorted}

    cases = build_cases()
    if len(cases) < 50:
        raise SystemExit(f"only {len(cases)} fixture cases, issue #101 requires >= 50")

    fixtures = []
    mismatches = []
    for name, text in cases:
        ids = encode_with_specials(sp, added_sorted, text)
        decoded = decode_with_specials(sp, added_by_id, ids)
        if decoded != text:
            mismatches.append((name, text, decoded))
        fixtures.append({"name": name, "text": text, "ids": ids, "decoded": decoded})

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(fixtures, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"wrote {args.out} ({len(fixtures)} cases)")
    if mismatches:
        print(f"NOTE: {len(mismatches)} case(s) where decode(encode(text)) != text (recorded as-is):")
        for name, text, decoded in mismatches:
            print(f"  {name}: {text!r} -> decoded {decoded!r}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
