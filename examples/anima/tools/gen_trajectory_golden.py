#!/usr/bin/env python3
"""Runs Anima's whole sampling loop in torch, and records the trajectory.

Issue #170 stage 5. The pieces are each checked against ComfyUI already — the
tokenizers to exact ids, the conditioning to 9.775e-7, the DiT to 8.3e-6 with a
512-token padded context, the stepper to 2e-6 on a toy denoiser. A pipeline can
still be wired wrongly with every part correct, and this is what catches that:
the same prompt, the same noise, the same schedule, one implementation in torch
and one in TypeScript, compared step by step.

The measurement that motivated it: the port's latent loses its spatial variation
as it denoises. Per-channel spatial standard deviation falls monotonically to
0.007 by the last step, which is a flat picture. Whether that is the model doing
what it should or the port driving it wrongly cannot be settled by looking at
the port.

**Per-channel spatial standard deviation, not the tensor's overall one.** A
latent that is constant across every pixel but offset from channel to channel
has a healthy-looking overall spread and no picture in it. That distinction is
what made the failure visible at all.

    PYTHONPATH=/tmp/comfy-venv/lib/python3.12/site-packages \\
        /tmp/comfy-venv/bin/python examples/anima/tools/gen_trajectory_golden.py \\
        --src <dit.safetensors> --encoder <qwen_3_06b_base.safetensors>
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

PROMPT = "1girl, silver hair, red eyes, looking at viewer, detailed background"


def spread(latent: torch.Tensor) -> float:
    """Median per-channel spatial standard deviation of `[1, C, T, H, W]`."""
    flat = latent.reshape(latent.shape[1], -1)
    return float(flat.std(dim=1).median())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, required=True)
    ap.add_argument("--encoder", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=HERE.parent / "fixtures")
    ap.add_argument("--prompt", default=PROMPT)
    ap.add_argument("--negative", default="")
    ap.add_argument("--steps", type=int, default=8)
    ap.add_argument("--cfg", type=float, default=8.0)
    ap.add_argument("--latent", type=int, default=32, help="latent H and W; 32 is a 256x256 image")
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()

    if not COMFY.exists():
        sys.exit(f"{COMFY} not found — see gen_block_golden.py for the clone command")
    sys.path.insert(0, str(COMFY))

    import comfy.ops
    from comfy import model_detection
    from comfy.ldm.anima.model import Anima, LLMAdapter
    from comfy.model_base import ModelType, model_sampling
    from comfy.samplers import beta_scheduler
    from comfy.text_encoders.anima import AnimaTokenizer
    from comfy.text_encoders.llama import Qwen3_06B
    from safetensors import safe_open

    torch.set_grad_enabled(False)
    torch.manual_seed(args.seed)

    # --- the DiT, quantized exactly as the port reads it ---
    with safe_open(args.src, framework="pt") as f:
        state = {k[len("net."):]: f.get_tensor(k).to(torch.float32) for k in sorted(f.keys())}
    config = model_detection.detect_unet_config({f"net.{k}": v for k, v in state.items()}, "net.")
    config.pop("image_model", None)

    model = (
        Anima(**config, device=None, dtype=torch.float32, operations=comfy.ops.disable_weight_init)
        .eval().to(torch.float32)
    )
    BUFFERS = {"pos_embedder.dim_spatial_range", "pos_embedder.dim_temporal_range", "pos_embedder.seq"}
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or [n for n in unexpected if n not in BUFFERS]:
        sys.exit(f"state dict mismatch: missing {missing[:5]}")

    quantized = 0
    for name, param in model.named_parameters():
        arr = param.data.numpy()
        if not should_quantize(f"net.{name}", arr.shape):
            continue
        codes, scale = quantize_q8(arr)
        param.data.copy_(torch.from_numpy(dequantize_q8(pack_q8(codes), scale, arr.shape)))
        quantized += 1
    print(f"DiT: {config['num_blocks']} blocks, {quantized} tensors quantized")

    # --- the encoder, dense ---
    with safe_open(args.encoder, framework="pt") as f:
        enc_state = {k: f.get_tensor(k).to(torch.float32) for k in f.keys()}
    encoder = Qwen3_06B({}, dtype=torch.float32, device=None, operations=comfy.ops.disable_weight_init)
    encoder = encoder.eval().to(torch.float32)
    encoder.load_state_dict(enc_state, strict=False)

    tok = AnimaTokenizer()

    def conditioning(text: str) -> torch.Tensor:
        pairs = tok.tokenize_with_weights(text)
        qwen_ids = torch.tensor([[k[0] for k in pairs["qwen3_06b"][0]]], dtype=torch.long)
        t5_ids = torch.tensor([[k[0] for k in pairs["t5xxl"][0]]], dtype=torch.int)
        weights = torch.tensor([[k[1] for k in pairs["t5xxl"][0]]])
        embeds = encoder.model.embed_tokens(qwen_ids, out_dtype=torch.float32)
        source = encoder(
            None, attention_mask=torch.ones_like(qwen_ids), embeds=embeds,
            num_tokens=qwen_ids.shape[1], dtype=torch.float32,
        )[0].float()
        # `preprocess_text_embeds` is the model's own — adapter, weights, pad.
        return model.preprocess_text_embeds(source, t5_ids, t5xxl_weights=weights.unsqueeze(-1))

    positive = conditioning(args.prompt)
    negative = conditioning(args.negative) if args.cfg > 1.0 else None
    # How many rows the adapter actually produced, before the pad to 512.
    positive_rows = len(tok.tokenize_with_weights(args.prompt)["t5xxl"][0])
    negative_rows = len(tok.tokenize_with_weights(args.negative)["t5xxl"][0]) if negative is not None else 0
    print(f"context: {tuple(positive.shape)}, {float((positive != 0).float().mean()) * 100:.1f}% non-zero, "
          f"std {float(positive.std()):.4f}")

    # --- the schedule and the loop ---
    class Settings:
        sampling_settings = {"shift": 3.0, "multiplier": 1.0}

    sampling = model_sampling(Settings(), ModelType.FLOW)
    sigmas = beta_scheduler(sampling, args.steps)
    channels = config["in_channels"]
    noise = torch.randn(1, channels, 1, args.latent, args.latent)
    x = sampling.noise_scaling(sigmas[0], noise, torch.zeros_like(noise), max_denoise=True)

    print(f"\n{len(sigmas) - 1} steps, {args.latent}x{args.latent} latent, CFG {args.cfg}")
    trajectory = [x.clone()]
    predictions = []
    old_denoised = None
    old_sigma_down = None

    def sigma_fn(t: torch.Tensor) -> torch.Tensor:
        return t.neg().exp()

    def t_fn(sigma: torch.Tensor) -> torch.Tensor:
        return sigma.log().neg()

    def phi1(t: torch.Tensor) -> torch.Tensor:
        return torch.expm1(t) / t

    def phi2(t: torch.Tensor) -> torch.Tensor:
        return (phi1(t) - 1.0) / t

    for i in range(len(sigmas) - 1):
        sigma = sigmas[i]
        t = sampling.timestep(sigma).reshape(1)
        raw_cond = model(x, t, positive)
        cond = sampling.calculate_denoised(sigma, raw_cond, x)
        if negative is not None:
            raw_uncond = model(x, t, negative)
            uncond = sampling.calculate_denoised(sigma, raw_uncond, x)
            # `cfg_function` (`samplers.py:598`), on the denoised predictions.
            denoised = uncond + (cond - uncond) * args.cfg
        else:
            raw_uncond = None
            denoised = cond
        predictions.append(denoised.clone())
        if i == 0:
            # The two halves of the first step, recorded separately. A port
            # whose CFG'd prediction is wrong could have either a wrong model
            # call or a wrong combination, and one number cannot say which.
            first_step = {"rawCond0": raw_cond.clone()}
            if raw_uncond is not None:
                first_step["rawUncond0"] = raw_uncond.clone()
            print(f"    raw cond spread {spread(raw_cond):.3f}, "
                  f"uncond {spread(raw_uncond):.3f}" if raw_uncond is not None else "")

        # `res_multistep` with `eta=0`: `sigma_down` is the next sigma and no
        # noise is ever added.
        sigma_down = sigmas[i + 1]
        if sigma_down == 0 or old_denoised is None:
            d = (x - denoised) / sigma
            x = x + d * (sigma_down - sigma)
        else:
            tt, t_old, t_next, t_prev = t_fn(sigma), t_fn(old_sigma_down), t_fn(sigma_down), t_fn(sigmas[i - 1])
            h = t_next - tt
            c2 = (t_prev - t_old) / h
            b1 = torch.nan_to_num(phi1(-h) - phi2(-h) / c2, nan=0.0)
            b2 = torch.nan_to_num(phi2(-h) / c2, nan=0.0)
            x = sigma_fn(h) * x + h * (b1 * denoised + b2 * old_denoised)
        old_denoised, old_sigma_down = denoised, sigma_down
        trajectory.append(x.clone())
        print(f"  step {i + 1}/{len(sigmas) - 1}  sigma {float(sigma):.4f}  "
              f"spread {spread(x):.3f}  (denoised {spread(denoised):.3f})")

    print(f"\nfinal spread {spread(x):.4f}")

    args.out.mkdir(parents=True, exist_ok=True)
    tensors = {
        "sigmas": torch.tensor([float(s) for s in sigmas]),
        "noise": noise,
        "x0": trajectory[0],
        # Only the rows that are not zero. `preprocess_text_embeds` pads to 512
        # and 3% of that is real, so storing the padding would be four megabytes
        # of zeros in a committed fixture. The verifier reconstructs it, which
        # is exact: the padding is `F.pad`'s zeros by construction.
        "positive": positive[:, :positive_rows, :],
        "negative": negative[:, :negative_rows, :] if negative is not None else torch.zeros(0),
        "final": x,
    }
    for i, step in enumerate(trajectory[1:]):
        tensors[f"x{i + 1}"] = step
    for i, pred in enumerate(predictions):
        tensors[f"denoised{i}"] = pred
    tensors.update(first_step)

    blob = bytearray()
    entries = []
    for name, tensor in tensors.items():
        arr = tensor.detach().cpu().numpy().astype(np.float32)
        entries.append({"name": name, "shape": list(arr.shape), "offset": len(blob)})
        blob.extend(arr.tobytes())

    header = json.dumps({
        "prompt": args.prompt, "negative": args.negative, "steps": args.steps, "cfg": args.cfg,
        "latent": args.latent, "seed": args.seed, "channels": channels,
        "contextLength": int(positive.shape[1]),
        "positiveRows": positive_rows, "negativeRows": negative_rows,
        "finalSpread": spread(x), "tensors": entries,
    }).encode()
    out = args.out / "trajectory-golden.bin"
    with out.open("wb") as f:
        f.write(struct.pack("<Q", len(header)))
        f.write(header)
        f.write(blob)
    print(f"wrote {out} ({out.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
