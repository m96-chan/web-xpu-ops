#!/usr/bin/env python3
"""MiniMax-H3's packed sequence layout, from diffusers' own pipeline blocks.

Issue #210. The transformer builds none of this: the caller orders the rows,
tags each with its modality and its noise level, and hands over the `(t, h, w)`
rotary grid. **Every part of it is a plausible-looking free choice that is not
free**, and none of it changes any shape:

- video rows are frame-major then row-major within a frame;
- the spatial grid is *aspect-normalised* and scaled by 32, built with
  `np.linspace(..., endpoint=False)` -- which is not `torch.linspace`;
- latent frames are spaced `5/3 * (1, 4, 4, 4, 4)` in rotary time, non-uniform,
  because the VAE's first latent covers one pixel frame and the rest cover four;
- the media clock **starts after the text**, so prompt length moves the video;
- audio rows are channel-major, carry no height, and are pinned to the two
  extremes of the width grid.

No weights and no licence: this is arithmetic on shapes, so the fixture is
committed.

    python examples/h3-dit/tools/gen_layout_golden.py --out examples/h3-dit/fixtures
"""

import argparse
import json
import pathlib

import torch

from diffusers.modular_pipelines.minimax_h3.before_denoise import (
    MiniMaxH3PrepareLayoutStep,
    MiniMaxH3SetTimestepsStep,
    patchify_video_latents,
)
from diffusers.modular_pipelines.minimax_h3.modular_pipeline import (
    MINIMAX_H3_AUDIO_CHANNELS,
    MINIMAX_H3_AUDIO_TAG,
    MINIMAX_H3_TEXT_TAG,
    MINIMAX_H3_VIDEO_TAG,
    align_num_frames,
    audio_latent_num_frames,
    resolve_canvas_size,
    video_latent_num_frames,
)

PATCH = (1, 2, 2)
# Three shapes rather than one: a square canvas (where the aspect normalisation
# is the identity and hides a bug), a wide one, and a tall one.
CASES = [
    {"numTextTokens": 3, "numLatentFrames": 2, "latentHeight": 4, "latentWidth": 4, "numAudioLatents": 5},
    {"numTextTokens": 7, "numLatentFrames": 7, "latentHeight": 4, "latentWidth": 8, "numAudioLatents": 3},
    {"numTextTokens": 1, "numLatentFrames": 3, "latentHeight": 10, "latentWidth": 6, "numAudioLatents": 2},
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    cases = []
    for case in CASES:
        tags = torch.full((case["numTextTokens"],), MINIMAX_H3_TEXT_TAG, dtype=torch.long)
        (
            position_ids,
            token_tags,
            video_indices,
            audio_indices,
            text_indices,
            num_condition_video_rows,
            num_condition_audio_rows,
        ) = MiniMaxH3PrepareLayoutStep.build_packed_sequence(
            tags,
            case["numLatentFrames"],
            case["latentHeight"],
            case["latentWidth"],
            case["numAudioLatents"],
            PATCH,
            MINIMAX_H3_AUDIO_CHANNELS,
            MINIMAX_H3_AUDIO_TAG,
            MINIMAX_H3_VIDEO_TAG,
            (),
        )

        # One `(timestep, timestep_indices)` pair, at a step where the video and
        # audio schedules differ -- which is the only reason the pair exists.
        unique, inverse = MiniMaxH3SetTimestepsStep.build_row_timesteps(
            video_indices, audio_indices, num_condition_video_rows, num_condition_audio_rows,
            int(text_indices.numel()), 0.25, 0.5, max(0.25, 0.999), 1.0,
        )

        # `patchify_video_latents` on an input whose elements name their own
        # coordinate, so a swapped axis shows up in the digits rather than in a
        # tolerance.
        c, f, h, w = 24, case["numLatentFrames"], case["latentHeight"], case["latentWidth"]
        latents = torch.arange(c * f * h * w, dtype=torch.float32).reshape(1, c, f, h, w)
        rows = patchify_video_latents(latents, PATCH)

        cases.append({
            **case,
            "positionIds": position_ids.flatten().tolist(),
            "tokenTags": token_tags.tolist(),
            "videoIndices": video_indices.tolist(),
            "audioIndices": audio_indices.tolist(),
            "textIndices": text_indices.tolist(),
            "uniqueTimesteps": unique.tolist(),
            "timestepIndices": inverse.tolist(),
            "patchify": {"channels": c, "rows": rows.shape[0], "cols": rows.shape[1], "values": rows.flatten().tolist()},
        })

    frames = [{"requested": n,
               "aligned": align_num_frames(n, 17, 5),
               "latentFrames": video_latent_num_frames(align_num_frames(n, 17, 5), 17, 5),
               "audioLatents": audio_latent_num_frames(align_num_frames(n, 17, 5))}
              for n in (1, 5, 22, 120, 121, 122, 346, 360)]

    canvases = [{"aspectWidth": aw, "aspectHeight": ah, "shortEdge": se, "maxPixels": mp, "multiple": 32,
                 "size": list(resolve_canvas_size(aw, ah, 32, se, mp))}
                for aw, ah, se, mp in ((16, 9, 480, 640 * 480), (9, 16, 480, 640 * 480), (1, 1, 480, 640 * 480),
                                       (16, 9, 720, 1280 * 720))]

    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "layout.json").write_text(
        json.dumps({
            "source": "diffusers modular_pipelines.minimax_h3.before_denoise",
            "note": "Arithmetic on shapes -- no weights, no model licence.",
            "patchSize": list(PATCH),
            "cases": cases,
            "frames": frames,
            "canvases": canvases,
        }, indent=1) + "\n"
    )
    for case in cases:
        print(f"{case['numLatentFrames']}x{case['latentHeight']}x{case['latentWidth']} "
              f"-> seq {len(case['tokenTags'])}, {len(case['uniqueTimesteps'])} distinct timesteps")
    print(f"wrote {out}/layout.json")


if __name__ == "__main__":
    main()
