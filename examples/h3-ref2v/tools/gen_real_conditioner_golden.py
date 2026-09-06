#!/usr/bin/env python3
"""The released conditioner run once, on a real presentation with a real image.

Issue #212. `text-encoder.ts` and `vision.ts` are held to `transformers`' models
at a tiny geometry with random weights; that establishes *structure*. This
establishes that the same code path runs the **released** Qwen3-VL-32B — 64
text layers of 5,120, a 27-block tower — on a presentation `ref2v` actually
builds, and writes `hidden_states[50]`, which is what MiniMax-H3 reads.

**Nothing here is redistributed.** What it writes is one activation and one
image's patches. See issue #190.

    python examples/h3-ref2v/tools/gen_real_conditioner_golden.py \
      --model ~/h3-work/text-encoder-dl/text_encoder \
      --tokenizer ~/h3-work/text-encoder-dl/tokenizer --out ~/h3-cond-real

## Two references, because one of them cannot answer the question

`--quantised` round-trips every matrix through `convert_conditioner.py`'s own
per-row absmax int8 before the forward. That is the reference a port has to be
held to when a divergence is large: this model has **massive activations** — at
layer 43 a handful of visual tokens grow by a factor of a hundred — and *which*
tokens is a near-tie that int8 rounding flips. Against the bf16 reference that
reads as a 88%-of-peak failure; against the int8 one it is 3%. Neither number
is wrong, and only the pair of them says which is the port's fault.

`--stages all` writes every state 0..50 rather than the first five, which is
what makes "at which layer" a question with an answer.
"""

import argparse
import json
import pathlib
import time

import numpy as np
import torch
from PIL import Image

from transformers import AutoProcessor, AutoTokenizer, Qwen3VLForConditionalGeneration
from diffusers.modular_pipelines.minimax_h3.encoders import MiniMaxH3Ref2VATextEncoderStep

TEXT_ENCODER_LAYER = 50


class Ref:
    kind = "image"
    has_audio = False

    def __init__(self, image):
        self.image = image


def quantise_roundtrip_(weight: torch.Tensor, chunk: int = 4096) -> None:
    """`convert_conditioner.quantize_rows`, dequantised back, in place.

    Row-chunked because the token table is 151,936 rows and a float32 copy of
    the whole of it is 3 GB beside a model that already fills the machine.
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
    parser.add_argument("--model", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--prompt", default="the reference, moving")
    parser.add_argument("--size", type=int, default=256, help="the reference image's edge, already a multiple of 32")
    parser.add_argument("--stages", default="0,1,2,3,4",
                        help='which hidden states to write beside the conditioning one; "all" is 0..50')
    parser.add_argument("--quantised", action="store_true",
                        help="round-trip every matrix through the converter's int8 first")
    args = parser.parse_args()

    stages = (list(range(TEXT_ENCODER_LAYER + 1)) if args.stages == "all"
              else sorted({int(s) for s in args.stages.split(",") if s != ""}))
    if stages and (stages[0] < 0 or stages[-1] > TEXT_ENCODER_LAYER):
        raise SystemExit(f"--stages must lie in 0..{TEXT_ENCODER_LAYER}")

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    processor = AutoProcessor.from_pretrained(args.model)

    # Every pixel names its own coordinate, so a transposed axis anywhere in the
    # chain shows up in the digits.
    n = args.size
    pixels = np.zeros((n, n, 3), dtype=np.uint8)
    for y in range(n):
        for x in range(n):
            pixels[y, x] = [(y * 7 + x) % 256, (x * 3 + 1) % 256, (y * 5 + x * 2 + 2) % 256]
    image = Image.fromarray(pixels)

    step = MiniMaxH3Ref2VATextEncoderStep()
    references = [Ref(image)]
    vision_inputs, image_tokens, video_tokens, video_stamps = step._gather_vision_features(
        processor, references, 24.0)
    token_ids, token_tags = step._build_presentation(
        tokenizer, args.prompt, references, image_tokens, video_tokens, video_stamps)
    print(f"presentation: {len(token_ids)} tokens, {sum(1 for t in token_tags if t == 0)} tagged video", flush=True)

    started = time.time()
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.model, dtype=torch.bfloat16, low_cpu_mem_usage=True).eval()
    print(f"loaded in {time.time() - started:.1f} s", flush=True)

    if args.quantised:
        # Layers 51..63 and `lm_head` are never evaluated for
        # `hidden_states[50]`, and the converter drops them for that reason.
        # Dropping them here is also what makes the round-trip *fit*:
        # materialising all 66.71 GB gets the process killed on a 121 GB
        # machine, and a killed process prints nothing at all.
        text = model.model.language_model
        text.layers = torch.nn.ModuleList(list(text.layers)[:TEXT_ENCODER_LAYER + 1])
        model.lm_head = None
        started = time.time()
        matrices = 0
        with torch.no_grad():
            for _, module in model.model.named_modules():
                if isinstance(module, torch.nn.Linear):
                    quantise_roundtrip_(module.weight)
                    matrices += 1
            quantise_roundtrip_(text.embed_tokens.weight)
            matrices += 1
        print(f"round-tripped {matrices} matrices through int8 in {time.time() - started:.0f} s", flush=True)

    input_ids = torch.tensor([token_ids], dtype=torch.long)
    # `Qwen3VLProcessor.create_mm_token_type_ids` is newer than this
    # `transformers`, so the rule is written out: **1 at an image pad, 2 at a
    # video pad, 0 everywhere else** -- including at the `<|vision_start|>` and
    # `<|vision_end|>` markers, which is what makes the visual mask the block's
    # *interior* rather than the block.
    mm = torch.zeros_like(input_ids)
    mm[input_ids == processor.image_token_id] = 1
    mm[input_ids == processor.video_token_id] = 2
    kwargs = {name: (value.to(model.dtype) if name.startswith("pixel_") else value)
              for name, value in vision_inputs.items()}

    # **The position grid the model built for itself**, dumped so a port is held
    # to it directly rather than to its effect forty layers later. Vision tokens
    # do not sit at `t = h = w = index`: `get_vision_position_ids` gives them a
    # 2-D grid, and the clock advances by `max(h, w) / merge` for the whole
    # block rather than by its token count.
    grid_thw = vision_inputs.get("image_grid_thw")
    position_ids, _ = model.model.get_rope_index(
        input_ids=input_ids, image_grid_thw=grid_thw,
        video_grid_thw=vision_inputs.get("video_grid_thw"),
        attention_mask=torch.ones_like(input_ids), mm_token_type_ids=mm)
    print(f"position grid t[:10] {position_ids[0, 0, :10].tolist()}", flush=True)

    started = time.time()
    with torch.no_grad():
        out = model.model(
            input_ids=input_ids, attention_mask=torch.ones_like(input_ids),
            mm_token_type_ids=mm, use_cache=False, output_hidden_states=True, **kwargs)
    # **Every requested state as well as the one that gets read**, so a port can
    # bisect a divergence to a layer instead of learning only that layer 50 is
    # wrong. State 0 is the embedding with the vision tokens already scattered
    # in and no deepstack yet, which separates three stages at once.
    #
    # These come from a **forward hook on the decoder layer**, so
    # `hidden_states[k]` is layer `k - 1`'s output *before* the deepstack add
    # the model applies between the layers. `conditioner-gpu.ts` adds deepstack
    # at the top of the layer that reads it for exactly this reason.
    saved = {index: out.hidden_states[index][0].to(torch.float32) for index in stages}
    hidden = out.hidden_states[TEXT_ENCODER_LAYER][0].to(torch.float32)
    print(f"forward in {time.time() - started:.1f} s -> {tuple(hidden.shape)}, "
          f"|h| max {hidden.abs().max():.4f}", flush=True)

    # Which rows carry the massive activation, and from which layer. This is
    # not decoration: it is a near-tie, int8 rounding flips it, and a port held
    # to the wrong one of the two references is chasing a bug that is not there.
    norms = hidden.norm(dim=1)
    top = norms.argsort(descending=True)[:4].tolist()
    print("massive rows at layer 50: " + "  ".join(f"{r}:{norms[r]:.0f}" for r in top), flush=True)

    out_dir = pathlib.Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "pixels.bin").write_bytes(pixels.ravel().tobytes())
    (out_dir / "hidden.bin").write_bytes(hidden.contiguous().numpy().astype("<f4").tobytes())
    for index, state in saved.items():
        (out_dir / f"hidden.{index}.bin").write_bytes(state.contiguous().numpy().astype("<f4").tobytes())
    grid = np.asarray(vision_inputs["image_grid_thw"])[0].tolist()
    # The rows a vision block occupies, which is what the port has to scatter
    # the tower's output into.
    runs, start = [], None
    for i, tag in enumerate(token_tags):
        if tag == 0 and start is None:
            start = i
        elif tag != 0 and start is not None:
            runs.append([start, i - start])
            start = None
    if start is not None:
        runs.append([start, len(token_tags) - start])
    # The markers are tagged video too, and the tower produces no token for
    # them: the pad run is the block's interior.
    interior = [[s + 1, length - 2] for s, length in runs]

    (out_dir / "golden.json").write_text(json.dumps({
        "source": "MiniMaxAI/MiniMax-H3 text_encoder (Qwen3-VL), hidden_states[50]",
        "note": "One activation and one image's patches. No weights are redistributed by this file.",
        "prompt": args.prompt,
        "imageSize": n,
        "tokenIds": token_ids,
        "tokenTags": token_tags,
        "visualRuns": interior,
        "grid": grid,
        "seq": len(token_ids),
        "hidden": list(hidden.shape),
        "layer": TEXT_ENCODER_LAYER,
        "stages": stages,
        # **Which reference this is.** A port compared against the wrong one of
        # the two reads the model's own int8 instability as its own bug.
        "quantised": bool(args.quantised),
        "positionIds": {
            "t": position_ids[0, 0].tolist(),
            "h": position_ids[1, 0].tolist(),
            "w": position_ids[2, 0].tolist(),
        },
    }, indent=1) + "\n")
    print(f"wrote {out_dir}, {'int8 round-trip' if args.quantised else 'bf16'}, "
          f"{len(saved)} staged states, visual runs {interior}, grid {grid}")


if __name__ == "__main__":
    main()
