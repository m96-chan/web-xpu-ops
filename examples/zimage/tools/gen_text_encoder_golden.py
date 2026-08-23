#!/usr/bin/env python3
"""Bakes a golden for Z-Image's text encoder — Qwen3-4B, `hidden_states[-2]`.

The DiT takes `cap_feats` of width 2560, which is Qwen3-4B's `hidden_size`, and
Z-Image takes them from `hidden_states[-2]` (`zimage_utils.py:214`). That index
is the whole question this file settles, and it settles it by **measurement**:
the loop in this version of `transformers` does not collect the hidden states
itself, so which layer `[-2]` names cannot be read off the source. Hooks are
placed on every decoder layer and the tuple entry is matched against them.

Getting it wrong by one layer produces embeddings of the right shape and the
wrong content, which the DiT would happily consume.

Two other conventions come from upstream rather than from a guess:

  - **The prompt goes through the chat template**, with `add_generation_prompt`
    and `enable_thinking` both on. A bare prompt tokenises to different ids.
  - **Padding is to a fixed 512** (`DEFAULT_MAX_SEQUENCE_LENGTH`), and the mask
    that comes back is what the DiT's caption mask is.

    /home/m96-chan/project/therdparty/musubi-tuner/.venv/bin/python \\
        examples/zimage/tools/gen_text_encoder_golden.py
"""
from __future__ import annotations

import argparse
import json
import struct
import sys
from pathlib import Path

import torch

MUSUBI = Path("/home/m96-chan/project/therdparty/musubi-tuner/src")
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent / "zimage-vae" / "tools"))

from models import add_argument, resolve  # noqa: E402

PROMPT = "a red apple on a wooden table"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=HERE.parent / "fixtures")
    ap.add_argument("--prompt", default=PROMPT)
    ap.add_argument("--max-length", type=int, default=None, help="default: the model's own 512")
    ap.add_argument("--keep-tokens", type=int, default=24,
                    help="how many leading token embeddings to store; the full 512x2560 would be 5 MB")
    add_argument(ap)
    args = ap.parse_args()

    sys.path.insert(0, str(MUSUBI))
    from musubi_tuner.zimage import zimage_config
    from transformers import AutoModelForCausalLM, AutoTokenizer

    src = Path(resolve("text_encoder", args.model_dir))
    tok_dir = src.parent / "tokenizer"
    max_length = args.max_length or zimage_config.DEFAULT_MAX_SEQUENCE_LENGTH

    torch.set_grad_enabled(False)
    tokenizer = AutoTokenizer.from_pretrained(str(tok_dir))
    model = AutoModelForCausalLM.from_pretrained(str(src), torch_dtype=torch.float32).eval()
    config = model.config

    formatted = tokenizer.apply_chat_template(
        [{"role": "user", "content": args.prompt}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=True,
    )
    encoded = tokenizer([formatted], padding="max_length", max_length=max_length,
                        truncation=True, return_tensors="pt")
    ids = encoded.input_ids
    mask = encoded.attention_mask.bool()
    valid = int(mask.sum().item())
    print(f"prompt -> {valid} real tokens of {max_length}")

    # One hook per decoder layer, so `[-2]` can be identified rather than assumed.
    per_layer: dict[int, torch.Tensor] = {}
    handles = [
        layer.register_forward_hook(
            lambda _m, _i, o, idx=i: per_layer.__setitem__(idx, o[0] if isinstance(o, tuple) else o)
        )
        for i, layer in enumerate(model.model.layers)
    ]
    out = model(input_ids=ids, attention_mask=mask, output_hidden_states=True)
    for handle in handles:
        handle.remove()

    states = out.hidden_states
    target = states[-2]
    print(f"hidden_states has {len(states)} entries for {config.num_hidden_layers} layers")
    matches = [i for i, h in per_layer.items() if h.shape == target.shape and torch.equal(h, target)]
    if len(matches) != 1:
        raise SystemExit(
            f"hidden_states[-2] matched {matches} decoder layers exactly; expected one. "
            f"Identify it before writing a golden that cannot say which layer it is."
        )
    layer_index = matches[0]
    print(f"hidden_states[-2] is the output of layer {layer_index} "
          f"(of 0..{config.num_hidden_layers - 1}), before model.norm")

    keep = min(args.keep_tokens, max_length)
    tensors = {
        "inputIds": ids[0, :keep].to(torch.float32),
        "mask": mask[0, :keep].to(torch.float32),
        "capFeats": target[0, :keep].contiguous(),
        # The last real token's row, wherever it is, so a port that is right at
        # the start and drifts is caught too.
        "capFeatsLast": target[0, valid - 1].contiguous(),
    }

    blob = bytearray()
    manifest = []
    for name, tensor in tensors.items():
        flat = tensor.detach().to(torch.float32).contiguous().reshape(-1)
        manifest.append({"name": name, "shape": list(tensor.shape), "offset": len(blob) // 4, "length": flat.numel()})
        blob.extend(struct.pack(f"<{flat.numel()}f", *flat.tolist()))

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "text-encoder.bin").write_bytes(bytes(blob))
    (args.out / "text-encoder.manifest.json").write_text(json.dumps({
        "note": "Generated by tools/gen_text_encoder_golden.py from Z-Image's own text encoder. Do not hand-edit.",
        "torch": torch.__version__,
        "prompt": args.prompt,
        "formattedPrompt": formatted,
        "maxLength": max_length,
        "validTokens": valid,
        "keptTokens": keep,
        "hiddenStatesIndex": {
            "asWritten": -2,
            "resolvedLayer": layer_index,
            "totalEntries": len(states),
            "note": "measured by hooking every decoder layer and matching, not read off the source",
        },
        "config": {
            "hiddenSize": config.hidden_size,
            "numLayers": config.num_hidden_layers,
            "numHeads": config.num_attention_heads,
            "numKvHeads": config.num_key_value_heads,
            "headDim": config.head_dim,
            "ffnHidden": config.intermediate_size,
            "rmsNormEps": config.rms_norm_eps,
            "ropeTheta": config.rope_theta,
            "vocabSize": config.vocab_size,
            "tieWordEmbeddings": config.tie_word_embeddings,
        },
        "tensors": manifest,
    }, indent=2, ensure_ascii=False) + "\n")

    print(f"wrote {args.out}/text-encoder.bin ({len(blob) / 1e3:.0f} kB)")
    print(f"capFeats[0,:4] = {target[0, 0, :4].tolist()}")


if __name__ == "__main__":
    main()
