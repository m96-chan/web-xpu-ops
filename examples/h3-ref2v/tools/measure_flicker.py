"""How much a generated video moves between frames, and how that compares.

Issue #216. This is the number that decided R2V was wrong, and until now it was
measured by hand each time — so the definition lived in a shell history and the
comparisons could not be reproduced. It is the most load-bearing metric this
example has; rule 9 says a number without its conditions cannot be compared.

**Mean absolute difference between consecutive frames, in 8-bit levels.** Not a
perceptual metric and not trying to be: it is the quantity that separates a
video from a flickering slideshow, it needs no reference to compute, and the
same definition applied to the *input* reference gives a floor to read the
output against.

    python examples/h3-ref2v/tools/measure_flicker.py ~/h3-work/h3-out-spec ~/h3-work/h3-out

Every argument is a directory holding `frames.rgb` and `frames.json` as
`generate-r2v.ts` writes them, or a bare `.rgb` with `--shape T,H,W`.
"""

from __future__ import annotations

import argparse
import json
import pathlib

import numpy as np


def load(path: pathlib.Path, shape: tuple[int, int, int] | None) -> tuple[np.ndarray, dict]:
    """`[frame][channel][row][col]` u8, as `generate-r2v.ts` writes it."""
    if path.is_dir():
        meta = json.loads((path / "frames.json").read_text())
        raw = np.fromfile(path / "frames.rgb", dtype=np.uint8)
        T, H, W = meta["frames"], meta["height"], meta["width"]
        C = meta.get("channels", 3)
        return raw.reshape(T, C, H, W).astype(np.float32), meta
    if shape is None:
        raise SystemExit(f"{path} is not a directory, so --shape T,H,W is required")
    T, H, W = shape
    raw = np.fromfile(path, dtype=np.uint8)
    return raw.reshape(T, 3, H, W).astype(np.float32), {"frames": T, "height": H, "width": W}


def flicker(frames: np.ndarray) -> tuple[float, list[float]]:
    """The mean absolute frame-to-frame difference, and the per-step series."""
    steps = [float(np.abs(frames[t] - frames[t - 1]).mean()) for t in range(1, len(frames))]
    return (float(np.mean(steps)) if steps else 0.0), steps


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("inputs", nargs="+")
    p.add_argument("--shape", help="T,H,W for a bare .rgb")
    p.add_argument("--colour", action="store_true",
                   help="also report how far each frame is from monochrome — the colour fringe R2V shows "
                        "is periodic, and the period is the thing to compare against upstream")
    p.add_argument("--period", type=int, default=0,
                   help="also report the mean per position in a cycle of this length — "
                        "4 tests whether the VAE's temporal upsample is the source")
    args = p.parse_args()
    shape = tuple(int(n) for n in args.shape.split(",")) if args.shape else None

    print(f"{'input':<28}  {'frames':>6}  {'size':>9}  {'flicker':>8}  {'worst step':>10}")
    for name in args.inputs:
        path = pathlib.Path(name).expanduser()
        frames, meta = load(path, shape)  # type: ignore[arg-type]
        mean, steps = flicker(frames)
        size = f"{meta['width']}x{meta['height']}"
        worst = max(steps) if steps else 0.0
        print(f"{path.name:<28}  {len(frames):>6}  {size:>9}  {mean:>8.2f}  {worst:>10.2f}")
        if args.colour:
            # Mean distance from grey, per frame. The fringe is a *colour*
            # artefact and the flicker number cannot see colour at all: a frame
            # that is uniformly brighter and a frame whose channels have come
            # apart give the same mean absolute difference.
            spread = np.abs(frames - frames.mean(1, keepdims=True)).mean((1, 2, 3))
            print(f"{'':<28}  colour spread per frame: "
                  + " ".join(f"{v:.1f}" for v in spread[:24])
                  + (" …" if len(spread) > 24 else ""))
        if args.period > 1 and steps:
            # A latent frame becomes `temporal_compression_ratio` pixel frames.
            # If the upsample were the source, the difference would land on one
            # position in the cycle rather than spreading across it.
            by_phase = [
                float(np.mean([s for i, s in enumerate(steps) if i % args.period == phase]))
                for phase in range(args.period)
            ]
            print(f"{'':<28}  per position in a cycle of {args.period}: "
                  + "  ".join(f"{v:.2f}" for v in by_phase))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
