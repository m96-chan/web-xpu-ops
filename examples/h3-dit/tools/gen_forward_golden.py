#!/usr/bin/env python3
"""The whole MiniMax-H3 DiT forward, at a size small enough to check in.

Issue #210. `examples/h3-dit/src/block.ts` holds **one** block to the model
(#208). The rest of the forward is not a loop: it projects three modalities
into one packed sequence, refines the text stream through two plain blocks,
builds a `(timestep, modality)` modulation table and addresses it **per row**,
and ends in a shared norm plus two heads whose rows are selected afterwards.
Every one of those returns a well-formed tensor when it is wrong.

**The weights here are random, and that is the point.** The arithmetic is
upstream's -- `MiniMaxH3Transformer3DModel` is instantiated, not transcribed --
but at the tester's own tiny geometry (hidden 24, 2 layers, heads 2x16), so the
fixture is ~200 KB and can be committed. It is not MiniMax's checkpoint and
carries none of its licence. What it establishes is *structure*: row order,
`index_copy` targets, the AdaLN table's addressing, the refiner, the partial
rotary. The real weights establish nothing structural that this does not.

The geometry is upstream's tester config
(`tests/models/transformers/test_models_transformer_minimax_h3.py`), including
its two deliberate awkwardnesses: `num_attention_heads * attention_head_dim`
(32) differs from `hidden_size` (24), as in the released checkpoint, and
`2 * 3 * rope_freq_dim` (12) is smaller than `attention_head_dim` (16), so the
partial-rotary path is exercised rather than aliased away.

    python examples/h3-dit/tools/gen_forward_golden.py --out examples/h3-dit/fixtures
"""

import argparse
import json
import pathlib

import numpy as np
import torch

from diffusers import MiniMaxH3Transformer3DModel

# The packed layout, from upstream's tester: a text block, then the audio rows,
# then the target video rows -- the order `MiniMaxH3Blocks` builds.
NUM_TEXT_TOKENS = 4
NUM_AUDIO_TOKENS = 6
NUM_VIDEO_TOKENS = 8

INIT = {
    "num_attention_heads": 2,
    "attention_head_dim": 16,
    "hidden_size": 24,
    "num_layers": 2,
    "num_refiner_layers": 2,
    "ffn_dim": 32,
    "in_channels": 4,
    "audio_in_channels": 6,
    "patch_size": (1, 2, 2),
    "text_dim": 8,
    "freq_dim": 8,
    "time_embed_hidden_dim": 24,
    "time_embed_dim": 16,
    "rope_freq_dim": 2,
}


def packed_layout(device):
    """The structural arguments of one packed sequence, as upstream's tester builds them.

    Two distinct timesteps, so the `(timestep, modality)` table is addressed on
    more than one row -- a layout with one timestep and one modality would let
    any indexing at all pass.
    """
    length = NUM_TEXT_TOKENS + NUM_AUDIO_TOKENS + NUM_VIDEO_TOKENS
    text_indices = torch.arange(NUM_TEXT_TOKENS, device=device)
    audio_indices = torch.arange(NUM_TEXT_TOKENS, NUM_TEXT_TOKENS + NUM_AUDIO_TOKENS, device=device)
    video_indices = torch.arange(NUM_TEXT_TOKENS + NUM_AUDIO_TOKENS, length, device=device)

    # 0 = video, 1 = text, 2 = audio.
    token_tags = torch.empty(length, dtype=torch.long, device=device)
    token_tags[text_indices] = 1
    token_tags[audio_indices] = 2
    token_tags[video_indices] = 0

    timestep_indices = torch.zeros(length, dtype=torch.long, device=device)
    timestep_indices[audio_indices] = 1

    position_ids = torch.zeros(length, 3, dtype=torch.float32, device=device)
    position_ids[:, 0] = torch.arange(length, dtype=torch.float32, device=device)
    position_ids[video_indices, 1] = torch.arange(NUM_VIDEO_TOKENS, dtype=torch.float32, device=device) % 4
    position_ids[video_indices, 2] = torch.arange(NUM_VIDEO_TOKENS, dtype=torch.float32, device=device) % 2

    return {
        "timestep": torch.tensor([0.7, 0.3], device=device),
        "timestep_indices": timestep_indices,
        "token_tags": token_tags,
        "position_ids": position_ids,
        "video_indices": video_indices,
        "audio_indices": audio_indices,
        "text_indices": text_indices,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cpu")

    # `float64` throughout. This port stores intermediates in `Float32Array`, so
    # a f32 golden measures torch's rounding as much as the port's; #208 needed
    # the same separation and its README records the two numbers side by side.
    model = MiniMaxH3Transformer3DModel(**INIT).to(device=device, dtype=torch.float64).eval()

    layout = packed_layout(device)
    patch = INIT["patch_size"]
    video_patch_dim = INIT["in_channels"] * patch[0] * patch[1] * patch[2]

    generator = torch.Generator("cpu").manual_seed(args.seed)
    inputs = {
        "hidden_states": torch.randn(1, NUM_VIDEO_TOKENS, video_patch_dim, generator=generator, dtype=torch.float64),
        "audio_hidden_states": torch.randn(
            1, NUM_AUDIO_TOKENS, INIT["audio_in_channels"], generator=generator, dtype=torch.float64
        ),
        "encoder_hidden_states": torch.randn(
            1, NUM_TEXT_TOKENS, INIT["text_dim"], generator=generator, dtype=torch.float64
        ),
    }

    with torch.no_grad():
        video, audio = model(**inputs, **layout, return_dict=False)

    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    # One flat f32 file plus a manifest of offsets, the layout every other
    # example here uses.
    blob = bytearray()
    tensors = []

    def put(name: str, t: torch.Tensor) -> None:
        a = t.detach().to(torch.float32).contiguous().numpy().astype("<f4")
        tensors.append({"name": name, "shape": list(t.shape), "offset": len(blob), "count": a.size})
        blob.extend(a.tobytes())

    for name, param in model.state_dict().items():
        put(name, param)
    for name, t in inputs.items():
        put(f"input.{name}", t)
    put("input.timestep", layout["timestep"])
    put("input.position_ids", layout["position_ids"])
    put("output.video", video)
    put("output.audio", audio)

    (out / "forward.bin").write_bytes(bytes(blob))
    (out / "forward.json").write_text(
        json.dumps(
            {
                "source": "diffusers MiniMaxH3Transformer3DModel, random weights, upstream's tester geometry",
                "note": "Not MiniMax's checkpoint. Random weights at a tiny geometry; no model licence applies.",
                "seed": args.seed,
                "config": {k: (list(v) if isinstance(v, tuple) else v) for k, v in INIT.items()},
                "layout": {
                    "numTextTokens": NUM_TEXT_TOKENS,
                    "numAudioTokens": NUM_AUDIO_TOKENS,
                    "numVideoTokens": NUM_VIDEO_TOKENS,
                    "tokenTags": layout["token_tags"].tolist(),
                    "timestepIndices": layout["timestep_indices"].tolist(),
                    "videoIndices": layout["video_indices"].tolist(),
                    "audioIndices": layout["audio_indices"].tolist(),
                    "textIndices": layout["text_indices"].tolist(),
                },
                "tensors": tensors,
            },
            indent=1,
        )
        + "\n"
    )
    print(f"{len(tensors)} tensors, {len(blob) / 1024:.1f} KB")
    print(f"video {tuple(video.shape)} |v| max {video.abs().max():.4f}")
    print(f"audio {tuple(audio.shape)} |a| max {audio.abs().max():.4f}")
    print(f"wrote {out}/forward.json and forward.bin")


if __name__ == "__main__":
    main()
