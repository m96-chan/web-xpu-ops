#!/usr/bin/env python3
"""One prompt through MiniMax-H3's Qwen3-VL conditioner, written out as f32.

Issue #210. **This is the step that does not go in a browser.** The conditioner
is Qwen3-VL-32B — 66.7 GB — against the DiT's 20.08 GB of int8 and the VAE
decoder's 2.43 GB, and `llm/` has no VL model. So a page offers prompts that
have already been through this script, which is a limitation the page has to
state rather than hide.

Three things upstream decides and this does not (rule 7), read out of
`diffusers.modular_pipelines.minimax_h3.encoders`:

- **The prompt goes in verbatim.** `add_special_tokens=False`, no chat
  template, no system prompt. A chat template would be the natural guess and
  produces a well-formed embedding of the wrong thing.
- **`hidden_states[50]`, not the last one.** The final layer is post-norm and
  is not what the released weights were trained against. Layer 50 of 64.
- **`text_encoder.model`, not `text_encoder`.** The language-model head is a
  vocabulary-wide projection over every token and nothing reads it.

The checkpoint is **guidance-distilled**: there is no negative prompt and no
unconditional branch, so one embedding is the whole conditioning.

**Nothing here is redistributed.** The file it writes is an activation for one
prompt, not weights. See issue #190.

    python examples/h3-dit/tools/encode_prompt.py \
      --model ~/h3-work/text-encoder-dl/text_encoder \
      --tokenizer ~/h3-work/text-encoder-dl/tokenizer \
      --out ~/h3-prompts --prompt "a paper boat on a puddle, rain falling"
"""

import argparse
import json
import pathlib
import re
import time

import torch
from transformers import AutoProcessor, AutoTokenizer, Qwen3VLForConditionalGeneration

TEXT_ENCODER_LAYER = 50


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="the text_encoder/ directory")
    parser.add_argument("--tokenizer", required=True, help="the tokenizer/ directory")
    parser.add_argument("--out", required=True)
    parser.add_argument("--prompt", required=True, action="append",
                        help="repeatable; one file is written per prompt")
    args = parser.parse_args()

    started = time.time()
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    try:
        processor = AutoProcessor.from_pretrained(args.model)
    except Exception as error:  # noqa: BLE001 - the processor is optional for a text-only prompt
        print(f"no processor ({error}); mm_token_type_ids will be zeros, which is what a text-only prompt has")
        processor = None

    # bf16 and memory-mapped. 66.7 GB does not fit in this machine's RAM as
    # anonymous pages, but file-backed read-only pages are evictable, so the
    # kernel pages the stack in and out as the forward walks it.
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.model, dtype=torch.bfloat16, low_cpu_mem_usage=True)
    model.eval()
    layers = model.config.text_config.num_hidden_layers
    print(f"loaded {layers} decoder layers in {time.time() - started:.1f} s", flush=True)
    if layers <= TEXT_ENCODER_LAYER:
        raise SystemExit(f"conditioning on hidden_states[{TEXT_ENCODER_LAYER}] needs more than {layers} layers")

    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    index = []

    for prompt in args.prompt:
        # Verbatim: no chat template, no special tokens.
        token_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
        input_ids = torch.tensor([token_ids], dtype=torch.long)
        if processor is not None and hasattr(processor, "create_mm_token_type_ids"):
            mm = torch.tensor(processor.create_mm_token_type_ids([token_ids]), dtype=torch.long)
        else:
            mm = torch.zeros_like(input_ids)

        at = time.time()
        with torch.no_grad():
            outputs = model.model(
                input_ids=input_ids,
                attention_mask=torch.ones_like(input_ids),
                mm_token_type_ids=mm,
                use_cache=False,
                output_hidden_states=True,
            )
        embeds = outputs.hidden_states[TEXT_ENCODER_LAYER][0].to(torch.float32)

        slug = re.sub(r"[^a-z0-9]+", "-", prompt.lower()).strip("-")[:48] or "prompt"
        (out / f"{slug}.bin").write_bytes(embeds.contiguous().numpy().astype("<f4").tobytes())
        index.append({
            "prompt": prompt,
            "file": f"{slug}.bin",
            "tokens": len(token_ids),
            "dim": int(embeds.shape[-1]),
        })
        print(f"{len(token_ids):4d} tokens -> {tuple(embeds.shape)} in {time.time() - at:.1f} s   {prompt!r}", flush=True)

    (out / "prompts.json").write_text(json.dumps({
        "source": f"MiniMaxAI/MiniMax-H3 text_encoder (Qwen3-VL), hidden_states[{TEXT_ENCODER_LAYER}]",
        "note": "Activations for these prompts. No weights are redistributed by this file.",
        "layer": TEXT_ENCODER_LAYER,
        "prompts": index,
    }, indent=1) + "\n")
    print(f"wrote {out}/prompts.json")


if __name__ == "__main__":
    main()
