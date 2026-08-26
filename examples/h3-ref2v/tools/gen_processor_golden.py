#!/usr/bin/env python3
"""Qwen3-VL's image processor, from `transformers`' own implementation.

Issue #212, step 7. Between a reference's pixels and the vision tower sits a
resize, a normalisation and a patchify, and each one is a convention rather
than a choice this port gets to make.

**The resize is deliberately excluded from this golden**, and that is the
finding rather than a shortcut. `Qwen2VLImageProcessor` resizes with **PIL's
bicubic**, which is a real resampler with its own filter and its own
support-scaling rule for downsampling. A browser has `drawImage`, which is not
it. So the golden is generated with `do_resize=False` on an already-conforming
image, the port takes pixels that are already the right size, and what the
browser's own resampler costs is **unmeasured** and said so.

Everything else is here and is exact:

- **`smart_resize` decides the target size**, and its `round` is Python's
  **banker's** rounding at a factor of `patch * merge = 32`. It also clamps to
  `[65536, 16777216]` pixels by two different rules — `floor` above the ceiling,
  `ceil` below the floor.
- **A still image is repeated to fill the temporal patch.** `temporal_patch_size`
  is 2, so one image becomes two frames; a port that emitted one produces half
  a patch and the right-looking shape.
- **The patchify permutation is `(0, 3, 6, 4, 7, 2, 1, 5, 8)`**, which lands
  tokens in **merge-block order** with each row laid out `C, t, py, px` — the
  order `examples/h3-ref2v/src/vision.ts` reads.

    python examples/h3-ref2v/tools/gen_processor_golden.py --out examples/h3-ref2v/fixtures
"""

import argparse
import json
import pathlib

import numpy as np
from PIL import Image

from transformers.models.qwen2_vl.image_processing_qwen2_vl import Qwen2VLImageProcessor, smart_resize

PATCH = 16
MERGE = 2
TEMPORAL = 2
FACTOR = PATCH * MERGE
MIN_PIXELS = 65536
MAX_PIXELS = 16777216

# Sizes chosen so each branch of `smart_resize` is the only thing separating two
# of them: already conforming, above the ceiling, below the floor, and one where
# the rounding lands on an exact half.
SIZES = [(480, 640), (1080, 1920), (64, 64), (100, 50), (37, 4000), (48, 48), (5000, 5000),
         # Past the 200:1 ceiling upstream refuses at.
         (10, 4000),
         # **Above the pixel ceiling and not evenly divisible.** 5000x5000
         # lands on `height / beta / factor == 128.0` exactly, where `floor` and
         # `ceil` agree -- so it cannot tell the two apart. 3000x6000 gives
         # 90.5, where they differ.
         (3000, 6000)]

# The image the patchify is checked on: small, already a whole number of merge
# blocks, and with every pixel naming its own coordinate so a transposed axis
# shows up in the digits rather than in a tolerance.
IMAGE_H, IMAGE_W = 64, 96


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    sizes = []
    for height, width in SIZES:
        try:
            out = smart_resize(height, width, factor=FACTOR, min_pixels=MIN_PIXELS, max_pixels=MAX_PIXELS)
            sizes.append({"height": height, "width": width, "resized": list(out)})
        except ValueError as error:
            sizes.append({"height": height, "width": width, "error": str(error)[:60]})
        print(f"  {height:5} x {width:<5} -> {sizes[-1].get('resized', sizes[-1].get('error'))}")

    # Every pixel names its own coordinate. uint8 wraps past 255, which is fine:
    # what matters is that neighbours differ and that the pattern is not
    # symmetric in the axes.
    pixels = np.zeros((IMAGE_H, IMAGE_W, 3), dtype=np.uint8)
    for y in range(IMAGE_H):
        for x in range(IMAGE_W):
            pixels[y, x] = [(y * 7 + x) % 256, (x * 3 + 1) % 256, (y * 5 + x * 2 + 2) % 256]

    processor = Qwen2VLImageProcessor(
        patch_size=PATCH, temporal_patch_size=TEMPORAL, merge_size=MERGE,
        image_mean=[0.5, 0.5, 0.5], image_std=[0.5, 0.5, 0.5],
    )
    # **`do_resize=False`** — see this file's own note. The image already
    # conforms, so the resampler is not in the way of what is being checked.
    features = processor(images=Image.fromarray(pixels), do_resize=False, return_tensors="np")
    values = np.asarray(features["pixel_values"])
    grid = np.asarray(features["image_grid_thw"])[0].tolist()
    print(f"  patchify: {pixels.shape} -> {values.shape}, grid {grid}")

    out_dir = pathlib.Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    blob = bytearray()
    blob.extend(pixels.ravel().tobytes())
    pixel_bytes = len(blob)
    blob.extend(values.astype("<f4").ravel().tobytes())
    (out_dir / "processor.bin").write_bytes(bytes(blob))
    (out_dir / "processor.json").write_text(json.dumps({
        "source": "transformers Qwen2VLImageProcessor (the class Qwen3VLProcessor uses)",
        "note": "Pixels and their patches. No weights, no model licence. `do_resize=False`.",
        "patchSize": PATCH, "mergeSize": MERGE, "temporalPatchSize": TEMPORAL,
        "factor": FACTOR, "minPixels": MIN_PIXELS, "maxPixels": MAX_PIXELS,
        "imageMean": [0.5, 0.5, 0.5], "imageStd": [0.5, 0.5, 0.5],
        "smartResize": sizes,
        "image": {"height": IMAGE_H, "width": IMAGE_W, "channels": 3, "bytes": pixel_bytes},
        "patches": {"rows": int(values.shape[0]), "cols": int(values.shape[1]), "offset": pixel_bytes},
        "grid": grid,
    }, indent=1) + "\n")
    print(f"wrote {out_dir}/processor.json and processor.bin")


if __name__ == "__main__":
    main()
