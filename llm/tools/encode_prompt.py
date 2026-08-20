#!/usr/bin/env python3
"""Encodes a chat-formatted prompt into token IDs for issue #105's real-model
validation, using the target checkpoint's own HF tokenizer.

Tokenizer wiring into the TS engine itself is out of this issue's scope
(#101, not yet merged) — this script exists so the *validation* step can
still run end to end: it renders the prompt exactly as `train_lora.py` /
`gen_val_gguf.py` do (`tokenizer.apply_chat_template`, `add_generation_prompt=True`),
tokenizes it, and prints the resulting token IDs as JSON, which the
real-model validation test reads directly rather than tokenizing anything
itself.

## Usage

    source third_party/venv/bin/activate   # from technologies.moe/alibi-ai
    python llm/tools/encode_prompt.py \\
        --model-dir /path/to/sarashina2.2-1b-alibi-v1 \\
        --system "..." --user "..."
"""
from __future__ import annotations

import argparse
import json

from transformers import AutoTokenizer


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--system", required=True)
    ap.add_argument("--user", required=True)
    args = ap.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    messages = [
        {"role": "system", "content": args.system},
        {"role": "user", "content": args.user},
    ]
    prompt_str = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    token_ids = tokenizer(prompt_str, add_special_tokens=False)["input_ids"]

    print(json.dumps({"promptText": prompt_str, "tokenIds": token_ids}))


if __name__ == "__main__":
    main()
