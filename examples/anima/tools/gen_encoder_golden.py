#!/usr/bin/env python3
"""Bakes a golden for Anima's conditioning path, from ComfyUI's own model.

Issue #170's fourth stage. The path a prompt takes before the DiT sees it:

    text --Qwen2 BPE--> qwen ids --Qwen3-0.6B--> source [1, Lq, 1024]
    text --T5 unigram--> t5 ids  --.
                                  '--LLMAdapter--> context [1, Lt, 1024]
                                     * t5 weights, zero-padded to 512

**This is the `native` path, and it is a complete one.** The released workflow
also offers `expanded`, which runs a Qwen3.5-4B through a separately shipped
adapter; `prompt.py:219` shows what that actually is:

    native_context   = native_adapter(source, target_ids)
    expanded_context = native_context + strength * (expanded - native_context)

— a residual on top of this, whose own node describes strength 0.0 as "native
Anima". So `native` is not a degraded mode to be replaced later; it is the base
the other path interpolates away from. Issue #173's selective state-space scan
is what `expanded` needs, and nothing here needs it.

**The adapter ships inside the DiT checkpoint** (`net.llm_adapter.*`), which is
why `--dit` is required alongside `--encoder`. It is quantized by
`convert_dit.py` like everything else, so the golden carries both the dense and
the q8 answer: a port compared against one number cannot tell a porting mistake
from what the format costs.

**Token ids are baked in, not just the tensors.** The prompt is recorded beside
them, so the TypeScript tokenizers are pinned by the same fixture that pins the
model — a port that tokenizes differently fails here rather than silently
conditioning on different words.

    PYTHONPATH=/tmp/comfy-venv/lib/python3.12/site-packages \\
        /tmp/comfy-venv/bin/python examples/anima/tools/gen_encoder_golden.py \\
        --encoder /home/m96-chan/anima-src/qwen_3_06b_base.safetensors \\
        --dit /home/m96-chan/anima-src/Anima-3.8B.safetensors
"""
from __future__ import annotations

import argparse
import json
import struct
import sys
from pathlib import Path

import numpy as np
import torch

COMFY = Path("/tmp/ComfyUI")
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from convert_dit import pack_q8, quantize_q8, should_quantize  # noqa: E402
from gen_block_golden import dequantize_q8  # noqa: E402

# Qwen3-0.6B, read off `Qwen3_06BConfig` (`comfy/text_encoders/llama.py:129`)
# rather than transcribed from a model card. `Qwen3_06B` builds it from a dict,
# so passing an empty one gets exactly the shipped defaults.
PROMPT = "1girl, silver hair, red eyes, looking at viewer, detailed background"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoder", type=Path, required=True)
    ap.add_argument("--dit", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=HERE.parent / "fixtures")
    ap.add_argument("--prompt", default=PROMPT)
    args = ap.parse_args()

    if not COMFY.exists():
        sys.exit(f"{COMFY} not found — see gen_block_golden.py for the clone command")
    sys.path.insert(0, str(COMFY))

    import comfy.ops
    from comfy.ldm.anima.model import LLMAdapter
    from comfy.text_encoders.llama import Qwen3_06B
    from safetensors import safe_open
    from transformers import AutoTokenizer, Qwen2Tokenizer

    torch.set_grad_enabled(False)

    # --- tokenize, with the two tokenizers `anima.py` names ---
    #
    # `AnimaTokenizer` runs both over the same text: the Qwen ids condition the
    # 0.6B, and the T5 ids index the adapter's own `Embedding(32128, 1024)`.
    # The T5 model itself is never loaded — only its tokenizer's ids are used.
    qwen_tok = Qwen2Tokenizer.from_pretrained(COMFY / "comfy/text_encoders/qwen25_tokenizer")
    t5_tok = AutoTokenizer.from_pretrained(COMFY / "comfy/text_encoders/t5_tokenizer")

    # `has_start_token=False, has_end_token=False` (`anima.py:10`) — the Qwen
    # side gets the bare ids, no chat template and no sentinels.
    qwen_ids = qwen_tok(args.prompt, add_special_tokens=False)["input_ids"]
    # The T5 side keeps its `</s>`: `T5XXLTokenizer` leaves `has_end_token` at
    # its default, and only turns the start token off.
    t5_ids = t5_tok(args.prompt, add_special_tokens=True)["input_ids"]
    print(f"prompt: {args.prompt!r}")
    print(f"  qwen ids: {len(qwen_ids)} {qwen_ids[:12]}{' ...' if len(qwen_ids) > 12 else ''}")
    print(f"  t5 ids:   {len(t5_ids)} {t5_ids[:12]}{' ...' if len(t5_ids) > 12 else ''}")

    # --- the 0.6B ---
    with safe_open(args.encoder, framework="pt") as f:
        enc_state = {k: f.get_tensor(k).to(torch.float32) for k in f.keys()}

    encoder = Qwen3_06B({}, dtype=torch.float32, device=None, operations=comfy.ops.disable_weight_init)
    encoder = encoder.eval().to(torch.float32)
    missing, unexpected = encoder.load_state_dict(enc_state, strict=False)
    if missing or unexpected:
        sys.exit(f"encoder state dict mismatch: missing {missing[:5]}, unexpected {unexpected[:5]}")

    ids = torch.tensor([qwen_ids], dtype=torch.long)
    # `Qwen3_06BModel` sets `attention_mask=True`, and one unpadded prompt is
    # all-ones — recorded so the port is compared on the case it actually runs
    # rather than on a mask it never sees.
    mask = torch.ones_like(ids)
    embeds = encoder.model.embed_tokens(ids, out_dtype=torch.float32)

    captured: dict[str, torch.Tensor] = {}
    encoder.model.layers[0].register_forward_hook(
        lambda _m, _i, o: captured.__setitem__("encLayer0", o[0] if isinstance(o, tuple) else o)
    )
    # `layer="last"` takes `outputs[0]` (`sd1_clip.py:281`), which is after the
    # final RMSNorm because `final_norm` is True. Z-Image's encoder needed
    # `hidden_states[-2]`; this one does not, and the difference is worth
    # stating because getting it wrong is invisible until the image is wrong.
    source = encoder(None, attention_mask=mask, embeds=embeds, num_tokens=ids.shape[1], dtype=torch.float32)[0]
    source = source.float()
    print(f"  source: {tuple(source.shape)}")

    # --- the adapter, out of the DiT checkpoint ---
    with safe_open(args.dit, framework="pt") as f:
        prefix = "net.llm_adapter."
        ad_state = {k[len(prefix):]: f.get_tensor(k).to(torch.float32) for k in f.keys() if k.startswith(prefix)}
    if not ad_state:
        sys.exit(f"{args.dit} carries no {prefix}* — is this the Anima DiT?")

    adapter = LLMAdapter(device=None, dtype=torch.float32, operations=comfy.ops.disable_weight_init)
    adapter = adapter.eval().to(torch.float32)
    missing, unexpected = adapter.load_state_dict(ad_state, strict=False)
    # `in_proj` is `nn.Identity` when `model_dim == target_dim`, which is this
    # checkpoint's case — it has no weights to be missing.
    if missing or unexpected:
        sys.exit(f"adapter state dict mismatch: missing {missing[:5]}, unexpected {unexpected[:5]}")

    target = torch.tensor([t5_ids], dtype=torch.long)
    adapter.blocks[0].register_forward_hook(lambda _m, _i, o: captured.__setitem__("adapterBlock0", o))
    context_dense = adapter(source, target)
    dense_trace = {f"{k}Dense": v for k, v in captured.items()}
    print(f"  context: {tuple(context_dense.shape)}")

    # --- the same again, with q8 where the shipped files actually put it ---
    #
    # **Only the adapter.** The encoder is read straight out of its own bf16
    # safetensors, as `examples/zimage-web` reads Qwen3-4B, because q8 costs far
    # too much here: quantizing the 0.6B's 196 layer matrices moves its output
    # by rel-RMS 0.223, against 0.0019 for `embed_tokens` alone and 0.040 for
    # the whole 52-block DiT. A 0.6B is not a 4B; per-row absmax has one scale
    # per 1024 numbers to spend and its outlier channels do not fit.
    #
    # 0.6 GB saved is not worth conditioning on different words, so the encoder
    # ships dense and this golden compares against the pair actually shipped.
    print("quantizing the adapter in place for the second run ...")
    quantized = 0
    for name, param in adapter.named_parameters():
        arr = param.data.numpy()
        if not should_quantize(f"net.llm_adapter.{name}", arr.shape):
            continue
        codes, scale = quantize_q8(arr)
        param.data.copy_(torch.from_numpy(dequantize_q8(pack_q8(codes), scale, arr.shape)))
        quantized += 1
    print(f"  {quantized} tensors")

    captured.clear()
    context_q8 = adapter(source, target)

    def rel(a: torch.Tensor, b: torch.Tensor) -> float:
        return ((a - b).pow(2).mean().sqrt() / b.pow(2).mean().sqrt()).item()

    print(f"q8 on the adapter alone: context rel-RMS {rel(context_q8, context_dense):.4g}")

    args.out.mkdir(parents=True, exist_ok=True)
    tensors: dict[str, torch.Tensor] = {
        "qwenIds": torch.tensor(qwen_ids, dtype=torch.int32),
        "t5Ids": torch.tensor(t5_ids, dtype=torch.int32),
        # One tensor, not two: the encoder is not quantized, so there is no
        # second answer to record. `sourceDense` is kept as an alias so the
        # port's checkpoint table reads the same as the DiT's.
        "source": source,
        "sourceDense": source,
        "context": context_q8,
        "contextDense": context_dense,
    }
    tensors.update(captured)
    tensors.update(dense_trace)

    blob = bytearray()
    entries = []
    for name, tensor in tensors.items():
        arr = tensor.detach().cpu().numpy()
        arr = arr.astype(np.int32 if arr.dtype == np.int32 else np.float32)
        entries.append({
            "name": name,
            "shape": list(arr.shape),
            "dtype": "i32" if arr.dtype == np.int32 else "f32",
            "offset": len(blob),
        })
        blob.extend(arr.tobytes())

    meta = {
        "prompt": args.prompt,
        "encoder": args.encoder.name,
        "dit": args.dit.name,
        "tensors": entries,
    }
    header = json.dumps(meta).encode()
    out = args.out / "encoder-golden.bin"
    with out.open("wb") as f:
        f.write(struct.pack("<Q", len(header)))
        f.write(header)
        f.write(blob)
    print(f"wrote {out} ({out.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
