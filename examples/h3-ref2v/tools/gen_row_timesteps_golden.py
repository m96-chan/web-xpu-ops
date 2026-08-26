#!/usr/bin/env python3
"""`ref2va`'s per-row noise levels, from upstream's own static method.

Issue #212. One forward serves every row of the packed sequence, and they are
**not all at the same noise level**: the generated video and audio rows step
down their own schedules while the *conditioning* rows stay pinned near clean.
`examples/h3-dit`'s `buildRowTimesteps` has the two-schedule half; `ref2va` adds
the third and fourth levels, and the assignment order matters.

`MiniMaxH3SetTimestepsStep.build_row_timesteps` is a `@staticmethod` with no
components, so this calls it directly rather than reproducing it — which is the
point: a golden that transcribed the rule would agree with a port that
transcribed it the same wrong way.

**No weights are involved.** The fixture is indices and timesteps.

    python examples/h3-ref2v/tools/gen_row_timesteps_golden.py --out examples/h3-ref2v/fixtures
"""

import argparse
import json
import pathlib

import torch

from diffusers.modular_pipelines.minimax_h3.before_denoise import MiniMaxH3SetTimestepsStep

# `MiniMaxH3ModularPipeline.keyframe_noise_aug` — 0.999, "just short of clean",
# because the released model was trained with its anchors very slightly noised.
KEYFRAME_NOISE_AUG = 0.999
# The audio reference rows' own level, which upstream passes as a literal 1.0.
CONDITION_AUDIO_TIMESTEP = 1.0


def case(name, num_text, video_rows, audio_rows, condition_video, condition_audio, video_t, audio_t):
    """One layout at one step, in the index form the transformer is given."""
    # Text first, then video, then audio — the *index* arrays are what the rule
    # reads, so their order is part of the case rather than an assumption.
    video_indices = torch.arange(num_text, num_text + video_rows)
    audio_indices = torch.arange(num_text + video_rows, num_text + video_rows + audio_rows)
    timestep, indices = MiniMaxH3SetTimestepsStep.build_row_timesteps(
        video_indices,
        audio_indices,
        condition_video,
        condition_audio,
        num_text,
        video_t,
        audio_t,
        max(video_t, KEYFRAME_NOISE_AUG),
        CONDITION_AUDIO_TIMESTEP,
    )
    return {
        "name": name,
        "numTextTokens": num_text,
        "videoIndices": video_indices.tolist(),
        "audioIndices": audio_indices.tolist(),
        "numConditionVideoRows": condition_video,
        "numConditionAudioRows": condition_audio,
        "videoTimestep": video_t,
        "audioTimestep": audio_t,
        "timestep": timestep.tolist(),
        "timestepIndices": indices.tolist(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    cases = [
        # The shape a one-image `ref2va` request has: reference video rows, no
        # reference audio, both schedules mid-run and disagreeing.
        case("one image reference", 12, 20, 6, 8, 0, 0.25, 0.4),
        # **Late in the schedule**, where `max(video_t, 0.999)` stops being
        # 0.999 and the conditioning rows collapse onto the generated ones. A
        # port that hardcoded 0.999 passes every other case and fails this.
        case("video timestep above the anchor", 12, 20, 6, 8, 0, 0.9995, 0.4),
        # A video reference with its own soundtrack: reference audio rows exist
        # and are pinned at 1.0, which is written *after* the generated audio
        # rows and overwrites them where they overlap.
        case("audio reference rows", 6, 16, 10, 4, 4, 0.5, 0.5),
        # **Video and audio at the same timestep**, which collapses two of the
        # four levels — `torch.unique` returns three entries, not four, and the
        # indices have to follow.
        case("schedules agreeing", 6, 16, 10, 4, 0, 0.5, 0.5),
        # No conditioning at all: this must reduce to `t2va`'s two levels.
        case("no conditioning rows", 8, 12, 4, 0, 0, 0.3, 0.7),
        # Every row conditioning, which leaves the generated slices empty.
        case("all rows conditioning", 4, 8, 4, 8, 4, 0.3, 0.7),
    ]

    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "row-timesteps.json").write_text(json.dumps({
        "source": "diffusers MiniMaxH3SetTimestepsStep.build_row_timesteps",
        "note": "Indices and noise levels. No weights.",
        "keyframeNoiseAug": KEYFRAME_NOISE_AUG,
        "conditionAudioTimestep": CONDITION_AUDIO_TIMESTEP,
        "cases": cases,
    }, indent=1) + "\n")
    for c in cases:
        print(f"  {c['name']}: {len(c['timestep'])} distinct levels {[round(t, 4) for t in c['timestep']]}")
    print(f"wrote {out}/row-timesteps.json")


if __name__ == "__main__":
    main()
