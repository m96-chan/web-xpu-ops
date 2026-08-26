#!/usr/bin/env python3
"""One forward of the released `transformer_ref` on a `ref2va` packed sequence.

Issue #212. `examples/h3-dit/tools/gen_real_forward_golden.py` does this for
`t2va` and `examples/h3-dit/src/verify-forward.ts` holds the GPU port to it.
This is the same thing for the workflow that has **reference rows**, which is
what `t2va` cannot exercise:

- The layout is `MiniMaxH3Ref2VAPrepareLayoutStep.build_ref2va_packed_sequence`,
  so `[text | reference blocks | target audio | target video]` rather than
  `[text | audio | video]`.
- `num_condition_video_rows` is not zero, so `build_row_timesteps` produces the
  **anchor level** as well as the two schedules — three or four distinct
  levels, where `t2va` has one or two.
- The conversion has to hold modulation tables for all of them. A `t2va`
  conversion indexes past the end of its own table on the first step whose plan
  needs a third, which is what happened before this existed.

**Nothing is redistributed.** What it writes is one activation.

    python examples/h3-ref2v/tools/gen_real_ref2va_forward_golden.py \
      --model ~/h3-work/transformer-ref-dl/transformer_ref \
      --out ~/h3-work/h3-ref2va-forward --layers 4
"""

import argparse
import json
import pathlib
import time

import numpy as np
import torch

from diffusers import MiniMaxH3Transformer3DModel
from diffusers.modular_pipelines.minimax_h3.before_denoise import (
    MiniMaxH3Ref2VAPrepareLayoutStep,
    MiniMaxH3SetTimestepsStep,
)
from diffusers.modular_pipelines.minimax_h3.modular_pipeline import (
    MINIMAX_H3_AUDIO_CHANNELS,
    MINIMAX_H3_AUDIO_TAG,
    MINIMAX_H3_TEXT_TAG,
    MINIMAX_H3_VIDEO_TAG,
)
from diffusers.modular_pipelines.minimax_h3.references import MiniMaxH3ImageReference

KEYFRAME_NOISE_AUG = 0.999
CONDITION_AUDIO_TIMESTEP = 1.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="the transformer_ref/ directory")
    parser.add_argument("--out", required=True)
    parser.add_argument("--text-tokens", type=int, default=6)
    parser.add_argument("--vision-tokens", type=int, default=4,
                        help="how many of the text rows are a reference's vision block, tagged video")
    parser.add_argument("--reference-frames", type=int, default=1)
    parser.add_argument("--reference-size", type=int, default=4, help="the reference's latent height and width")
    parser.add_argument("--latent-frames", type=int, default=2)
    parser.add_argument("--latent-size", type=int, default=8)
    parser.add_argument("--audio-latents", type=int, default=3)
    parser.add_argument("--layers", type=int, default=0, help="run only the first N blocks (0 = all)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=16)
    parser.add_argument("--step-index", type=int, default=5)
    args = parser.parse_args()

    started = time.time()
    # Mixed precision, memory-mapped: the same reasons the `t2va` generator
    # gives, and the same 66 GB.
    model = MiniMaxH3Transformer3DModel.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True)
    model.eval()
    print(f"loaded in {time.time() - started:.1f} s", flush=True)

    if args.layers:
        model.transformer_blocks = model.transformer_blocks[: args.layers]
        model.config.num_layers = args.layers
        print(f"running only the first {args.layers} blocks", flush=True)

    c = model.config

    # **A reference's vision block is tagged video, not text.** That is why the
    # layout takes the caller's tags rather than filling them in, and a golden
    # whose text rows were all text would never notice.
    tags = torch.full((args.text_tokens,), MINIMAX_H3_TEXT_TAG, dtype=torch.long)
    tags[1 : 1 + args.vision_tokens] = MINIMAX_H3_VIDEO_TAG

    # One image reference, at its own latent geometry -- deliberately *not* the
    # target's, so a port that reused the target's grid disagrees here.
    condition = torch.zeros(
        1, c.in_channels, args.reference_frames, args.reference_size, args.reference_size)
    generator = torch.Generator("cpu").manual_seed(args.seed)
    condition.normal_(generator=generator)

    (
        position_ids, token_tags, video_indices, audio_indices, text_indices, cond_video, cond_audio,
    ) = MiniMaxH3Ref2VAPrepareLayoutStep.build_ref2va_packed_sequence(
        tags,
        [MiniMaxH3ImageReference(image=np.zeros((8, 8, 3), dtype=np.uint8))],
        [condition],
        [],
        args.latent_frames, args.latent_size, args.latent_size, args.audio_latents,
        tuple(c.patch_size), MINIMAX_H3_AUDIO_CHANNELS, MINIMAX_H3_AUDIO_TAG, MINIMAX_H3_VIDEO_TAG,
    )
    print(
        f"seq {int(token_tags.numel())}, {cond_video} conditioning video rows, "
        f"{cond_audio} conditioning audio rows", flush=True,
    )

    from diffusers import MiniMaxH3Scheduler
    video_schedule = MiniMaxH3Scheduler(shift=12.0)
    audio_schedule = MiniMaxH3Scheduler(shift=3.0)
    video_schedule.set_timesteps(args.steps)
    audio_schedule.set_timesteps(args.steps)
    video_t = float(video_schedule.timesteps[args.step_index])
    audio_t = float(audio_schedule.timesteps[args.step_index])
    unique, inverse = MiniMaxH3SetTimestepsStep.build_row_timesteps(
        video_indices, audio_indices, cond_video, cond_audio, int(text_indices.numel()),
        video_t, audio_t, max(video_t, KEYFRAME_NOISE_AUG), CONDITION_AUDIO_TIMESTEP,
    )
    print(
        f"step {args.step_index}/{args.steps}: video t={video_t:.6f} audio t={audio_t:.6f}, "
        f"{int(unique.numel())} distinct levels {[round(float(x), 6) for x in unique]}", flush=True,
    )

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
        "source": "MiniMaxAI/MiniMax-H3 transformer_ref, run by diffusers MiniMaxH3Transformer3DModel",
        "note": "Activations only. No weights are redistributed by this file.",
        "workflow": "ref2va",
        "dtype": "bfloat16 block stack, float32 projections and heads",
        "seed": args.seed,
        "steps": args.steps,
        "stepIndex": args.step_index,
        "layers": int(len(model.transformer_blocks)),
        "config": {k: (list(v) if isinstance(v, (tuple, list)) else v)
                   for k, v in dict(model.config).items() if not k.startswith("_")},
        "layout": {
            "numTextTokens": args.text_tokens,
            "textTokenTags": tags.tolist(),
            "referenceFrames": args.reference_frames,
            "referenceSize": args.reference_size,
            "numLatentFrames": args.latent_frames,
            "latentHeight": args.latent_size,
            "latentWidth": args.latent_size,
            "numAudioLatents": args.audio_latents,
            "numConditionVideoRows": int(cond_video),
            "numConditionAudioRows": int(cond_audio),
            "seq": int(token_tags.numel()),
            "tokenTags": token_tags.tolist(),
            "timestep": [float(x) for x in unique],
            "timestepIndices": inverse.tolist(),
            "videoIndices": video_indices.tolist(),
            "audioIndices": audio_indices.tolist(),
            "textIndices": text_indices.tolist(),
        },
    }, indent=1) + "\n")

    v = out_video.float().numpy()
    print(f"video {v.shape} |v| max {np.abs(v).max():.4f}  rms {np.sqrt((v ** 2).mean()):.4f}")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
