#!/usr/bin/env python3
"""MiniMax-H3's `ref2va` presentation, from diffusers' own block.

Issue #212. Before a reference reaches the transformer it is *announced* to the
conditioner: a label per reference, numbered per modality, and a vision block of
pad tokens standing in for its pixels. That announcement is what produces
`text_token_tags` — the argument `examples/h3-ref2v/src/layout.ts` takes because
**a vision block's rows are tagged video (0) while sitting among the text rows**.

Five things it decides, and every one of them yields a well-formed token list
when it is wrong:

- `"<Picture i>: "` for an image, `"<Audio j>: "` for a soundtrack,
  `"<Video k>: "` for a video, each numbered **per modality**, not per reference.
- A video that carries sound is labelled `"<Audio j>: "` **before**
  `"<Video k>: "` -- mirroring the order its rows are packed in.
- A video gets **one timestamped vision block per merged frame group**, not one
  block.
- The timestamp is rendered with `"{:.1f}"`, which is Python's **round half to
  even**: the mean of a 2 fps pair is `0.25` and renders `"<0.2 seconds>"`, not
  `"<0.3 seconds>"`. JavaScript's `toFixed` gives `"0.3"`.
- The prompt follows **verbatim** -- no chat template, no special tokens.

The fixture records the token ids **and a map from each text segment to its
ids**, so the port can be held to the *assembly* without reimplementing
Qwen2's BPE. What is being tested here is the order, the counts, the tags and
the timestamps; the tokenizer is a separate concern with its own golden.

    python examples/h3-ref2v/tools/gen_presentation_golden.py \
      --tokenizer ~/h3-work/text-encoder-dl/tokenizer --out examples/h3-ref2v/fixtures
"""

import argparse
import json
import pathlib

import numpy as np
from transformers import AutoTokenizer

from diffusers.modular_pipelines.minimax_h3.encoders import MiniMaxH3Ref2VATextEncoderStep

FPS = 24.0
SAMPLE_FPS = 2.0
TEMPORAL_PATCH = 2


class Ref:
    def __init__(self, kind, has_audio=False):
        self.kind = kind
        self.has_audio = has_audio


CASES = [
    {
        "name": "one image",
        "refs": [("image", False)],
        "imageTokens": [4], "videoTokens": [], "videoStamps": [],
        "prompt": "a paper boat on a puddle",
    },
    {
        "name": "two images, numbered per modality",
        "refs": [("image", False), ("image", False)],
        "imageTokens": [4, 6], "videoTokens": [], "videoStamps": [],
        "prompt": "in the rain",
    },
    {
        "name": "a silent video, three blocks",
        "refs": [("video", False)],
        "imageTokens": [], "videoTokens": [9], "videoStamps": [[0.25, 1.25, 2.25]],
        "prompt": "keep the motion",
    },
    {
        "name": "a video with sound is labelled Audio first",
        "refs": [("video", True)],
        "imageTokens": [], "videoTokens": [4], "videoStamps": [[0.25, 1.25]],
        "prompt": "and the sound",
    },
    {
        "name": "an image, a soundtrack and a video",
        "refs": [("image", False), ("audio", True), ("video", True)],
        "imageTokens": [4], "videoTokens": [6], "videoStamps": [[0.25]],
        "prompt": "put them together",
    },
    {
        # **A video *before* an image.** Numbering is per modality, so this is
        # still `<Picture 1>` -- and with every image first, a counter that
        # summed the modalities would have agreed everywhere above.
        "name": "a video then an image, still Picture 1",
        "refs": [("video", False), ("image", False)],
        "imageTokens": [4], "videoTokens": [6], "videoStamps": [[0.25, 1.25]],
        "prompt": "the second one is an image",
    },
]

# Frame-sampler cases, which are arithmetic on counts and need no pixels.
SAMPLER = [
    {"numFrames": 24, "fps": 24.0, "sampleFps": 2.0},
    # **A stride below one**, which is the only arrangement where the
    # deduplication does anything. `video_sample_fps` is a public argument, so
    # this is reachable even though the shipped 2 fps never gets near it.
    {"numFrames": 12, "fps": 24.0, "sampleFps": 30.0},
    {"numFrames": 8, "fps": 24.0, "sampleFps": 48.0},
    {"numFrames": 25, "fps": 24.0, "sampleFps": 2.0},
    {"numFrames": 49, "fps": 24.0, "sampleFps": 2.0},
    {"numFrames": 13, "fps": 24.0, "sampleFps": 2.0},
    {"numFrames": 30, "fps": 30.0, "sampleFps": 2.0},
    {"numFrames": 100, "fps": 24.0, "sampleFps": 4.0},
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    step = MiniMaxH3Ref2VATextEncoderStep(video_sample_fps=SAMPLE_FPS)

    segments = {}

    def record(text):
        if text not in segments:
            segments[text] = tokenizer(text, add_special_tokens=False)["input_ids"]

    cases = []
    for case in CASES:
        references = [Ref(kind, has_audio) for kind, has_audio in case["refs"]]
        token_ids, token_tags = step._build_presentation(
            tokenizer, case["prompt"], references,
            case["imageTokens"], case["videoTokens"], case["videoStamps"],
        )
        # Every text segment the presentation could have produced, so the port
        # can be held to the assembly without a BPE implementation.
        counts = {"image": 0, "video": 0, "audio": 0}
        for reference in references:
            if reference.has_audio:
                counts["audio"] += 1
                record(f"<Audio {counts['audio']}>: ")
            if reference.kind == "image":
                counts["image"] += 1
                record(f"<Picture {counts['image']}>: ")
            elif reference.kind == "video":
                counts["video"] += 1
                record(f"<Video {counts['video']}>: ")
                for stamp in case["videoStamps"][counts["video"] - 1]:
                    record(f"<{stamp:.1f} seconds>")
        record(case["prompt"])

        cases.append({
            "name": case["name"],
            "references": [{"kind": k, "hasAudio": h} for k, h in case["refs"]],
            "imageTokenCounts": case["imageTokens"],
            "videoBlockTokenCounts": case["videoTokens"],
            "videoBlockTimestamps": case["videoStamps"],
            "prompt": case["prompt"],
            "tokenIds": token_ids,
            "tokenTags": token_tags,
        })
        print(f"{case['name']:44} {len(token_ids):4} tokens, "
              f"{sum(1 for t in token_tags if t == 0):3} tagged video")

    sampler = []
    for case in SAMPLER:
        frames = np.zeros((case["numFrames"], 2, 2, 3), dtype="uint8")
        sampled, stamps = step._sample_video_condition_frames(
            frames, case["fps"], case["sampleFps"], TEMPORAL_PATCH)
        sampler.append({**case, "numSampled": len(sampled), "blockTimestamps": stamps,
                        "rendered": [f"{s:.1f}" for s in stamps]})
        print(f"  {case['numFrames']:4} frames at {case['fps']:g} -> {len(sampled)} sampled, "
              f"{len(stamps)} blocks {[f'{s:.1f}' for s in stamps][:4]}")

    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "presentation.json").write_text(json.dumps({
        "source": "diffusers MiniMaxH3Ref2VATextEncoderStep",
        "note": "Token ids and segment boundaries. No weights, no model licence.",
        "visionStart": tokenizer.convert_tokens_to_ids("<|vision_start|>"),
        "visionEnd": tokenizer.convert_tokens_to_ids("<|vision_end|>"),
        "imagePad": tokenizer.convert_tokens_to_ids("<|image_pad|>"),
        "videoPad": tokenizer.convert_tokens_to_ids("<|video_pad|>"),
        "temporalPatch": TEMPORAL_PATCH,
        "segments": segments,
        "cases": cases,
        "sampler": sampler,
    }, indent=1) + "\n")
    print(f"wrote {out}/presentation.json")


if __name__ == "__main__":
    main()
