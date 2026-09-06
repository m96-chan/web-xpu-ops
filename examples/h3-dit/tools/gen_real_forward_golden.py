#!/usr/bin/env python3
"""The released 50-layer DiT run once, by diffusers' own code, on a tiny sequence.

Issue #210. `fixtures/forward.json` establishes the forward's *structure* at
upstream's tester geometry. This establishes that the same code path runs the
**released weights** -- 66 GB of bf16, `hidden_size` 5376, 50 layers, a two-block
refiner -- and writes what came out, so a GPU port has something at full width
to be wrong against.

**Nothing here is redistributed.** The golden it writes holds activations for a
handful of rows, not weights; the checkpoint stays where it was downloaded. See
issue #190.

The sequence is deliberately tiny -- a couple of latent frames at 8x8 -- because
the point is the *width* of the model, not the length of the sequence. Attention
over 5,376 channels and 56 heads is what a port gets wrong; 30 rows exercise it
as well as 30,000 and fit in RAM.

    python examples/h3-dit/tools/gen_real_forward_golden.py \
      --model ~/h3-work/transformer-dl/transformer --out ~/h3-dit-real
"""

import argparse
import json
import pathlib
import time

import numpy as np
import torch

from diffusers import MiniMaxH3Transformer3DModel
from diffusers.modular_pipelines.minimax_h3.before_denoise import (
    MiniMaxH3PrepareLayoutStep,
    MiniMaxH3SetTimestepsStep,
)
from diffusers.modular_pipelines.minimax_h3.modular_pipeline import (
    MINIMAX_H3_AUDIO_CHANNELS,
    MINIMAX_H3_AUDIO_TAG,
    MINIMAX_H3_TEXT_TAG,
    MINIMAX_H3_VIDEO_TAG,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="the transformer/ directory of MiniMaxAI/MiniMax-H3")
    parser.add_argument("--out", required=True)
    parser.add_argument("--text-tokens", type=int, default=4)
    parser.add_argument("--latent-frames", type=int, default=2)
    parser.add_argument("--latent-size", type=int, default=8, help="latent height and width")
    parser.add_argument("--audio-latents", type=int, default=3)
    parser.add_argument("--layers", type=int, default=0, help="run only the first N blocks (0 = all)")
    parser.add_argument("--seed", type=int, default=0)
    # The GPU port reads **precomputed** modulation tables, which exist only
    # for the schedule step counts the converter was given -- so the golden has
    # to sit on a real schedule entry rather than at an arbitrary noise level.
    parser.add_argument("--steps", type=int, default=16)
    parser.add_argument("--step-index", type=int, default=5)
    args = parser.parse_args()

    started = time.time()
    # `torch_dtype` rather than a `.to()`: the checkpoint is **mixed
    # precision** -- the patch projections, the timestep MLP and the output
    # heads are float32 while the block stack is bfloat16 -- and
    # `_keep_in_fp32_modules` only holds if `from_pretrained` does the casting.
    # `low_cpu_mem_usage=True` is required rather than preferred: diffusers
    # refuses `keep_in_fp32_modules` without it, and it is what keeps the
    # 66 GB memory-mapped instead of copied into 66 GB of anonymous pages --
    # which this machine, with 57 GB free, does not have.
    model = MiniMaxH3Transformer3DModel.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True)
    model.eval()
    print(f"loaded in {time.time() - started:.1f} s", flush=True)

    if args.layers:
        model.transformer_blocks = model.transformer_blocks[: args.layers]
        model.config.num_layers = args.layers
        print(f"running only the first {args.layers} blocks", flush=True)

    c = model.config
    tags = torch.full((args.text_tokens,), MINIMAX_H3_TEXT_TAG, dtype=torch.long)
    (
        position_ids, token_tags, video_indices, audio_indices, text_indices, cond_video, cond_audio,
    ) = MiniMaxH3PrepareLayoutStep.build_packed_sequence(
        tags, args.latent_frames, args.latent_size, args.latent_size, args.audio_latents,
        tuple(c.patch_size), MINIMAX_H3_AUDIO_CHANNELS, MINIMAX_H3_AUDIO_TAG, MINIMAX_H3_VIDEO_TAG, (),
    )
    from diffusers import MiniMaxH3Scheduler
    video_schedule = MiniMaxH3Scheduler(shift=12.0)
    audio_schedule = MiniMaxH3Scheduler(shift=3.0)
    video_schedule.set_timesteps(args.steps)
    audio_schedule.set_timesteps(args.steps)
    video_t = float(video_schedule.timesteps[args.step_index])
    audio_t = float(audio_schedule.timesteps[args.step_index])
    print(f"step {args.step_index}/{args.steps}: video t={video_t:.6f} audio t={audio_t:.6f}", flush=True)
    unique, inverse = MiniMaxH3SetTimestepsStep.build_row_timesteps(
        video_indices, audio_indices, cond_video, cond_audio, int(text_indices.numel()),
        video_t, audio_t, max(video_t, 0.999), 1.0,
    )

    generator = torch.Generator("cpu").manual_seed(args.seed)
    patch_dim = c.in_channels * c.patch_size[0] * c.patch_size[1] * c.patch_size[2]
    video = torch.randn(1, int(video_indices.numel()), patch_dim, generator=generator, dtype=torch.float32)
    audio = torch.randn(1, int(audio_indices.numel()), c.audio_in_channels, generator=generator, dtype=torch.float32)
    text = torch.randn(1, args.text_tokens, c.text_dim, generator=generator, dtype=torch.float32)

    started = time.time()
    with torch.no_grad():
        out_video, out_audio = model(
            hidden_states=video, audio_hidden_states=audio, encoder_hidden_states=text.to(torch.bfloat16),
            timestep=unique, timestep_indices=inverse, token_tags=token_tags, position_ids=position_ids,
            video_indices=video_indices, audio_indices=audio_indices, text_indices=text_indices,
            return_dict=False,
        )
    print(f"forward in {time.time() - started:.1f} s", flush=True)

    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    def write(name: str, t: torch.Tensor) -> None:
        (out / name).write_bytes(t.detach().to(torch.float32).contiguous().numpy().astype("<f4").tobytes())

    write("input.video.bin", video)
    write("input.audio.bin", audio)
    write("input.text.bin", text)
    write("input.timestep.bin", unique)
    write("input.position_ids.bin", position_ids)
    write("output.video.bin", out_video)
    write("output.audio.bin", out_audio)

    (out / "golden.json").write_text(json.dumps({
        "source": "MiniMaxAI/MiniMax-H3 transformer, run by diffusers MiniMaxH3Transformer3DModel",
        "sourcePath": str(args.model),
        "note": "Activations only. No weights are redistributed by this file.",
        "dtype": "bfloat16 block stack, float32 projections and heads",
        "seed": args.seed,
        "steps": args.steps,
        "stepIndex": args.step_index,
        "layers": int(len(model.transformer_blocks)),
        "config": {k: (list(v) if isinstance(v, (tuple, list)) else v)
                   for k, v in dict(model.config).items() if not k.startswith("_")},
        "layout": {
            "numTextTokens": args.text_tokens,
            "numLatentFrames": args.latent_frames,
            "latentHeight": args.latent_size,
            "latentWidth": args.latent_size,
            "numAudioLatents": args.audio_latents,
            "seq": int(token_tags.numel()),
            "tokenTags": token_tags.tolist(),
            "timestepIndices": inverse.tolist(),
            "videoIndices": video_indices.tolist(),
            "audioIndices": audio_indices.tolist(),
            "textIndices": text_indices.tolist(),
        },
    }, indent=1) + "\n")

    v = out_video.float().numpy()
    a = out_audio.float().numpy()
    print(f"video {v.shape} |v| max {np.abs(v).max():.4f}  rms {np.sqrt((v ** 2).mean()):.4f}")
    print(f"audio {a.shape} |a| max {np.abs(a).max():.4f}")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
