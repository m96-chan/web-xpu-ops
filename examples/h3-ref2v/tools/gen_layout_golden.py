#!/usr/bin/env python3
"""MiniMax-H3's `ref2va` packed layout, from diffusers' own pipeline block.

Issue #212. R2V conditions on *references* — images, videos, soundtracks — and
the layout is `[text | reference blocks | target audio | target video]`. What
makes it more than the `t2va` layout with extra rows is that **the references
advance a shared rotary clock**: where the generated video sits in rotary time
depends on how many references came before it and how long each one was. That
is part of the layout, not a detail of the presentation.

Six things it decides, none of which changes a shape:

- **An image takes exactly one rotary slot**, not a latent frame's `5/3`.
- **A video reference's soundtrack rows are packed immediately *before* its
  video rows** and share their origin, so the two are aligned the way the
  generated audio and video are.
- A video reference advances the clock by
  `max(its audio latents, its video span)` — whichever of the two is longer.
- That video span is summed **sequentially**, which is *not* how the `t2va`
  layout's "last frame" anchor sums the same series: that one reproduces
  numpy's pairwise sum, and the two orders differ in the last ulp from 16
  latent frames onwards. Upstream keeps both, one per call site.
- A standalone audio reference is pinned to the **target's** width grid; a
  video reference's soundtrack is pinned to **its own**.
- Reference rows are tagged **video (0)**, including the vision-block rows that
  sit inside the text range.

No weights and no licence: arithmetic on shapes, so the fixture is committed.

    python examples/h3-ref2v/tools/gen_layout_golden.py --out examples/h3-ref2v/fixtures
"""

import argparse
import json
import pathlib

import torch

from diffusers.modular_pipelines.minimax_h3.before_denoise import MiniMaxH3Ref2VAPrepareLayoutStep
from diffusers.modular_pipelines.minimax_h3.modular_pipeline import (
    MINIMAX_H3_AUDIO_CHANNELS,
    MINIMAX_H3_AUDIO_TAG,
    MINIMAX_H3_TEXT_TAG,
    MINIMAX_H3_VIDEO_TAG,
)

PATCH = (1, 2, 2)
Z = 24
AUDIO_Z = 32


class Ref:
    """Only `kind` and `has_audio` are read by the layout; the geometry comes from the latents."""

    def __init__(self, kind: str, has_audio: bool = False):
        self.kind = kind
        self.has_audio = has_audio


# Each case names its references and the shapes their encoders produced.
CASES = [
    {
        "name": "one image",
        "refs": [("image", False)],
        "visual": [(1, 4, 4)],
        "audio": [],
        "text": 5, "frames": 2, "height": 4, "width": 4, "audioLatents": 3,
    },
    {
        "name": "an image and a silent video",
        "refs": [("image", False), ("video", False)],
        "visual": [(1, 4, 6), (3, 4, 4)],
        "audio": [],
        "text": 4, "frames": 2, "height": 4, "width": 6, "audioLatents": 2,
    },
    {
        "name": "a video with sound, then an image",
        "refs": [("video", True), ("image", False)],
        "visual": [(2, 6, 4), (1, 4, 4)],
        "audio": [(4 * MINIMAX_H3_AUDIO_CHANNELS, AUDIO_Z)],
        "text": 3, "frames": 3, "height": 6, "width": 4, "audioLatents": 5,
    },
    {
        "name": "a standalone soundtrack between two images",
        "refs": [("image", False), ("audio", False), ("image", False)],
        "visual": [(1, 4, 4), (1, 6, 4)],
        "audio": [(3 * MINIMAX_H3_AUDIO_CHANNELS, AUDIO_Z)],
        "text": 6, "frames": 2, "height": 4, "width": 8, "audioLatents": 2,
    },
    {
        # Past the sixteen latent frames where the two summation orders diverge.
        "name": "a long video reference",
        "refs": [("video", True)],
        "visual": [(18, 4, 4)],
        "audio": [(20 * MINIMAX_H3_AUDIO_CHANNELS, AUDIO_Z)],
        "text": 2, "frames": 2, "height": 4, "width": 4, "audioLatents": 3,
    },
    {
        # **A soundtrack that outlasts its own video.** One latent frame spans
        # 5/3 of a rotary slot and twelve audio latents span twelve, so the
        # `max` picks the audio -- the only arrangement where it does. Without
        # this case, dropping the `max` and taking the video span alone is
        # invisible.
        "name": "a video whose soundtrack outlasts it",
        "refs": [("video", True)],
        "visual": [(1, 4, 4)],
        "audio": [(12 * MINIMAX_H3_AUDIO_CHANNELS, AUDIO_Z)],
        "text": 3, "frames": 2, "height": 4, "width": 4, "audioLatents": 2,
    },
    {
        # **A video reference on a different canvas from the target.** Its
        # soundtrack is pinned to *its own* width grid; pinning it to the
        # target's is invisible while the two canvases match, which they did in
        # every case above.
        "name": "a video reference on its own canvas",
        "refs": [("video", True)],
        "visual": [(2, 4, 12)],
        "audio": [(3 * MINIMAX_H3_AUDIO_CHANNELS, AUDIO_Z)],
        "text": 2, "frames": 2, "height": 8, "width": 4, "audioLatents": 2,
    },
    {
        # **Text rows that are not all text.** MiniMax-H3 tags the rows of a
        # reference's vision block **video (0)**, and they sit inside the text
        # range -- so `text_token_tags` is an argument, not a constant.
        "name": "a vision block inside the text rows",
        "refs": [("image", False)],
        "visual": [(1, 4, 4)],
        "audio": [],
        "text": 7, "textTags": [1, 1, 0, 0, 0, 0, 1],
        "frames": 2, "height": 4, "width": 4, "audioLatents": 2,
    },
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    cases = []
    for case in CASES:
        references = [Ref(kind, has_audio) for kind, has_audio in case["refs"]]
        condition_latents = [torch.zeros(1, Z, *shape) for shape in case["visual"]]
        audio_condition_latents = [torch.zeros(*shape) for shape in case["audio"]]
        # Text rows carry their own tags: a reference's vision block is tagged
        # **video**, not text, even though it sits in the text range.
        tags = (torch.tensor(case["textTags"], dtype=torch.long) if "textTags" in case
                else torch.full((case["text"],), MINIMAX_H3_TEXT_TAG, dtype=torch.long))

        (
            position_ids, token_tags, video_indices, audio_indices, text_indices,
            num_reference_video_rows, num_reference_audio_rows,
        ) = MiniMaxH3Ref2VAPrepareLayoutStep.build_ref2va_packed_sequence(
            tags, references, condition_latents, audio_condition_latents,
            case["frames"], case["height"], case["width"], case["audioLatents"],
            PATCH, MINIMAX_H3_AUDIO_CHANNELS, MINIMAX_H3_AUDIO_TAG, MINIMAX_H3_VIDEO_TAG,
        )

        cases.append({
            "name": case["name"],
            "references": [{"kind": k, "hasAudio": h} for k, h in case["refs"]],
            "visual": [list(s) for s in case["visual"]],
            "audio": [list(s) for s in case["audio"]],
            "numTextTokens": case["text"],
            "textTokenTags": tags.tolist(),
            "numLatentFrames": case["frames"],
            "latentHeight": case["height"],
            "latentWidth": case["width"],
            "numAudioLatents": case["audioLatents"],
            "seq": int(token_tags.numel()),
            "positionIds": position_ids.flatten().tolist(),
            "tokenTags": token_tags.tolist(),
            "videoIndices": video_indices.tolist(),
            "audioIndices": audio_indices.tolist(),
            "textIndices": text_indices.tolist(),
            "numReferenceVideoRows": int(num_reference_video_rows),
            "numReferenceAudioRows": int(num_reference_audio_rows),
        })
        print(f"{case['name']:42} seq {int(token_tags.numel()):5}  "
              f"ref rows {int(num_reference_video_rows)}v/{int(num_reference_audio_rows)}a")

    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "layout.json").write_text(json.dumps({
        "source": "diffusers MiniMaxH3Ref2VAPrepareLayoutStep.build_ref2va_packed_sequence",
        "note": "Arithmetic on shapes -- no weights, no model licence.",
        "patchSize": list(PATCH),
        "latentChannels": Z,
        "audioLatentChannels": AUDIO_Z,
        "cases": cases,
    }, indent=1) + "\n")
    print(f"wrote {out}/layout.json")


if __name__ == "__main__":
    main()
