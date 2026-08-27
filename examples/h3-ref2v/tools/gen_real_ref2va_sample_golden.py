#!/usr/bin/env python3
"""A whole `ref2va` sampling loop, run by upstream, at a size a CPU can finish.

Issue #216. `gen_real_ref2va_forward_golden.py` holds **one forward** to the
model and the port matches it to 0.68% of peak. That is not the same claim as
fifteen of them matching: the loop adds the scheduler, the row-timestep plan
changing per step, and the rule that the conditioning rows are re-imposed by
never being written. R2V's output flickers at 2.7x `t2va`'s and its latent
comes out 31% hot, and everything a single forward can see has been ruled out.

So this runs the loop. Same geometry as the forward golden — 48 rows, which one
forward of the released `transformer_ref` does in 36 s on a CPU, so fifteen is
ten minutes rather than a week — and writes the **initial state as well as the
final latents**, because a port that starts somewhere else is not being
compared.

**Nothing is redistributed.** What it writes is activations.

    python examples/h3-ref2v/tools/gen_real_ref2va_sample_golden.py \
      --model ~/h3-work/transformer-ref-dl/transformer_ref \
      --out ~/h3-work/h3-ref2va-sample --layers 4
"""

import argparse
import json
import pathlib
import time

import numpy as np
import torch

from diffusers import MiniMaxH3Scheduler, MiniMaxH3Transformer3DModel
from diffusers.modular_pipelines.minimax_h3.before_denoise import (
    MiniMaxH3PrepareLayoutStep,
    MiniMaxH3Ref2VAPrepareLayoutStep,
    MiniMaxH3SetTimestepsStep,
    patchify_video_latents,
)
from diffusers.modular_pipelines.minimax_h3.modular_pipeline import (
    MINIMAX_H3_AUDIO_CHANNELS,
    MINIMAX_H3_TEXT_TAG,
    MINIMAX_H3_AUDIO_TAG,
    MINIMAX_H3_VIDEO_TAG,
)
from diffusers.modular_pipelines.minimax_h3.references import MiniMaxH3ImageReference

KEYFRAME_NOISE_AUG = 0.999
CONDITION_AUDIO_TIMESTEP = 1.0
VIDEO_SHIFT = 12.0
AUDIO_SHIFT = 3.0



def quantise_roundtrip_(weight: torch.Tensor, chunk: int = 4096) -> None:
    """`convert_dit.quantize_rows`, dequantised straight back, in place.

    Issue #216. One forward over real content disagrees with upstream by 7.38%
    of peak at fifty blocks and fifteen steps of it by 21.71% -- so something
    accumulates that random content does not make accumulate. The port runs int8
    weights and there is no float conversion of fifty blocks that fits on a
    32 GB card, so the only way to ask "is this int8?" is to put int8 on
    *upstream's* side and see whether the number comes back.

    Per-row absmax, the same rounding and the same clamp `convert_dit.py` does,
    minus the packing -- the packing is a layout, not an approximation.
    Row-chunked so a float32 copy of a wide matrix does not sit beside a model
    that already fills the machine.
    """
    for start in range(0, weight.shape[0], chunk):
        block = weight[start:start + chunk]
        values = block.to(torch.float32)
        absmax = values.abs().amax(dim=1)
        scale = torch.where(absmax == 0, torch.ones_like(absmax), absmax / 127.0)
        inverse = torch.where(absmax == 0, torch.zeros_like(absmax), 127.0 / absmax)
        codes = torch.clamp(torch.round(values * inverse.unsqueeze(1)), -127, 127)
        block.copy_((codes * scale.unsqueeze(1)).to(weight.dtype))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="the transformer_ref/ directory")
    parser.add_argument("--out", required=True)
    parser.add_argument("--text-tokens", type=int, default=6)
    parser.add_argument("--vision-tokens", type=int, default=4)
    parser.add_argument("--reference-frames", type=int, default=1)
    parser.add_argument("--reference-size", type=int, default=4)
    parser.add_argument("--latent-frames", type=int, default=2)
    parser.add_argument("--latent-size", type=int, default=8)
    parser.add_argument("--audio-latents", type=int, default=3)
    parser.add_argument("--layers", type=int, default=0, help="run only the first N blocks (0 = all)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--quantised", action="store_true",
                        help="round every matrix through convert_dit.py's int8 first, so the port's "
                             "quantisation is on this side of the comparison too (issue #216)")
    parser.add_argument("--steps", type=int, default=16)
    parser.add_argument("--conditioning", help="a directory `gen_real_conditioner_golden.py` wrote: "
                        "its `hidden.<layer>.bin` and real token tags replace the random text rows (issue #216)")
    parser.add_argument("--conditioning-layer", type=int, default=50)
    parser.add_argument("--reference-latent", help="a directory `gen_real_anchor.py` wrote: its `anchor.bin` "
                        "replaces the random conditioning latent (issue #216)")
    parser.add_argument("--no-reference", action="store_true",
                        help="build the t2va layout instead, so the two can be compared with nothing else changed")
    args = parser.parse_args()

    started = time.time()
    model = MiniMaxH3Transformer3DModel.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True)
    model.eval()
    print(f"loaded in {time.time() - started:.1f} s", flush=True)
    if args.layers:
        model.transformer_blocks = model.transformer_blocks[: args.layers]
        model.config.num_layers = args.layers
        print(f"running only the first {args.layers} blocks", flush=True)

    if args.quantised:
        started = time.time()
        matrices = 0
        with torch.no_grad():
            for _, module in model.named_modules():
                if isinstance(module, torch.nn.Linear):
                    quantise_roundtrip_(module.weight)
                    matrices += 1
        print(f"round-tripped {matrices} matrices through int8 in {time.time() - started:.0f} s", flush=True)

    c = model.config

    # **Real conditioning and a real anchor, when they are given.** Issue #216:
    # every golden here has fed the transformer `randn` for both, and the loop
    # matches upstream exactly when it does -- 0.0% rms over all fifty blocks.
    # The flicker only appears with real references, so the *content* of these
    # two tensors is the thing that has never been in a golden. Nothing else in
    # this script changes, so a run with them and a run without differ only in
    # what the sequence is carrying.
    real_text = None
    if args.conditioning:
        cond_dir = pathlib.Path(args.conditioning).expanduser()
        meta = json.loads((cond_dir / "golden.json").read_text())
        seq, dim = meta["hidden"]
        if dim != c.text_dim:
            raise SystemExit(f"conditioning is {dim} wide and the transformer wants {c.text_dim}")
        real_text = torch.from_numpy(
            np.fromfile(cond_dir / f"hidden.{args.conditioning_layer}.bin", dtype="<f4").reshape(1, seq, dim).copy()
        )
        # The real tags too. A reference's vision block is tagged *video* in the
        # presentation, not text, and which rows carry which tag is part of the
        # layout rather than a detail of the text.
        tags = torch.tensor(meta["tokenTags"], dtype=torch.long)
        args.text_tokens = seq
        print(f"real conditioning: {seq} rows, layer {args.conditioning_layer}, "
              f"{int((tags == MINIMAX_H3_VIDEO_TAG).sum())} of them visual", flush=True)
    else:
        tags = torch.full((args.text_tokens,), MINIMAX_H3_TEXT_TAG, dtype=torch.long)
        tags[1 : 1 + args.vision_tokens] = MINIMAX_H3_VIDEO_TAG

    generator = torch.Generator("cpu").manual_seed(args.seed)
    if args.reference_latent:
        anchor_dir = pathlib.Path(args.reference_latent).expanduser()
        shape = json.loads((anchor_dir / "anchor.json").read_text())["shape"]
        condition = torch.from_numpy(
            np.fromfile(anchor_dir / "anchor.bin", dtype="<f4").reshape(1, *shape).copy()
        )
        args.reference_frames, args.reference_size = shape[1], shape[2]
        print(f"real anchor: {tuple(shape)}  rms {float(condition.pow(2).mean().sqrt()):.4f}", flush=True)
    else:
        condition = torch.empty(
            1, c.in_channels, args.reference_frames, args.reference_size, args.reference_size)
        condition.normal_(generator=generator)

    if args.no_reference:
        # The `t2va` layout: the same target, no conditioning rows, and the text
        # rows all text. Everything else in this script is unchanged, so the two
        # runs differ only in whether the sequence has anchors in it.
        tags = torch.full((args.text_tokens,), MINIMAX_H3_TEXT_TAG, dtype=torch.long)
        (
            position_ids, token_tags, video_indices, audio_indices, text_indices, cond_video, cond_audio,
        ) = MiniMaxH3PrepareLayoutStep.build_packed_sequence(
            tags, args.latent_frames, args.latent_size, args.latent_size, args.audio_latents,
            tuple(c.patch_size), MINIMAX_H3_AUDIO_CHANNELS, MINIMAX_H3_AUDIO_TAG, MINIMAX_H3_VIDEO_TAG, (),
        )
    else:
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

    # **The conditioning noise is drawn first**, one draw per reference, before
    # the target's — upstream's order, and its own doc says that order is part
    # of what a generator reproduces.
    condition_noise = torch.empty_like(condition)
    condition_noise.normal_(generator=generator)
    video_scheduler = MiniMaxH3Scheduler(shift=VIDEO_SHIFT)
    audio_scheduler = MiniMaxH3Scheduler(shift=AUDIO_SHIFT)
    video_scheduler.set_timesteps(args.steps)
    audio_scheduler.set_timesteps(args.steps)
    noised_condition = video_scheduler.scale_noise(condition, KEYFRAME_NOISE_AUG, condition_noise)
    condition_rows = patchify_video_latents(noised_condition, tuple(c.patch_size))[:cond_video]
    assert condition_rows.shape[0] == cond_video, (condition_rows.shape, cond_video)

    patch_dim = c.in_channels * c.patch_size[0] * c.patch_size[1] * c.patch_size[2]
    target_rows = torch.empty(int(video_indices.numel()) - cond_video, patch_dim)
    target_rows.normal_(generator=generator)
    video = torch.cat([condition_rows, target_rows])[None]
    audio = torch.empty(1, int(audio_indices.numel()), c.audio_in_channels)
    audio.normal_(generator=generator)
    if real_text is not None:
        text = real_text
    else:
        text = torch.empty(1, args.text_tokens, c.text_dim)
        text.normal_(generator=generator)

    out_dir = pathlib.Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    def write(name: str, t: torch.Tensor) -> None:
        (out_dir / name).write_bytes(t.detach().to(torch.float32).contiguous().numpy().astype("<f4").tobytes())

    # The initial state, so a port starts where this started.
    write("input.video.bin", video)
    write("input.audio.bin", audio)
    write("input.text.bin", text)

    video_t = video_scheduler.timesteps
    audio_t = audio_scheduler.timesteps
    print(f"{len(video_t)} steps over {int(token_tags.numel())} rows, {cond_video} of them conditioning", flush=True)

    latents, audio_latents = video[0], audio[0]
    started = time.time()
    for i, t in enumerate(video_t):
        unique, inverse = MiniMaxH3SetTimestepsStep.build_row_timesteps(
            video_indices, audio_indices, cond_video, cond_audio, int(text_indices.numel()),
            float(t), float(audio_t[i]), max(float(t), KEYFRAME_NOISE_AUG), CONDITION_AUDIO_TIMESTEP,
        )
        with torch.no_grad():
            out_video, out_audio = model(
                hidden_states=latents[None], audio_hidden_states=audio_latents[None],
                encoder_hidden_states=text.to(torch.bfloat16),
                timestep=unique, timestep_indices=inverse, token_tags=token_tags,
                position_ids=position_ids, video_indices=video_indices,
                audio_indices=audio_indices, text_indices=text_indices, return_dict=False,
            )
        # **Only the generated rows are written.** The anchors are re-imposed by
        # construction; there is no mask and no recomposition.
        latents[cond_video:] = video_scheduler.step(
            out_video[0, cond_video:].float(), t, latents[cond_video:], return_dict=False)[0]
        audio_latents[cond_audio:] = audio_scheduler.step(
            out_audio[0, cond_audio:].float(), audio_t[i], audio_latents[cond_audio:], return_dict=False)[0]
        print(
            f"  step {i + 1}/{len(video_t)}  t={float(t):.4f}  "
            f"velocity rms {float(out_video.float().pow(2).mean().sqrt()):.4f}, "
            f"rows {float(latents.float().pow(2).mean().sqrt()):.4f}",
            flush=True,
        )
    print(f"sampled in {time.time() - started:.1f} s", flush=True)

    write("output.video.bin", latents)
    write("output.audio.bin", audio_latents)
    (out_dir / "golden.json").write_text(json.dumps({
        "source": "MiniMaxAI/MiniMax-H3 transformer_ref, sampled by diffusers MiniMaxH3Scheduler",
        "sourcePath": str(args.model),
        "note": "Activations only. No weights are redistributed by this file.",
        "workflow": "t2va" if args.no_reference else "ref2va",
        "seed": args.seed,
        "steps": args.steps,
        "quantised": bool(args.quantised),
        "conditioning": args.conditioning or None,
        "referenceLatent": args.reference_latent or None,
        "layers": int(len(model.transformer_blocks)),
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
            "videoIndices": video_indices.tolist(),
            "audioIndices": audio_indices.tolist(),
            "textIndices": text_indices.tolist(),
        },
    }, indent=1) + "\n")
    v = latents.float().numpy()
    print(f"final latents {v.shape} rms {np.sqrt((v ** 2).mean()):.4f}  |v| max {np.abs(v).max():.4f}")
    print(f"wrote {out_dir}")


if __name__ == "__main__":
    main()
