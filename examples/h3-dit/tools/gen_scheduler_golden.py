#!/usr/bin/env python3
"""MiniMax-H3's sampling schedule, from diffusers' own scheduler.

Issue #210. No weights and no licence: the schedule is arithmetic on
`num_inference_steps` and `shift`, so the fixture is a few kilobytes of JSON
and lives in the repository.

Both shipped shifts are covered -- `scheduler/` is 12.0 and `audio_scheduler/`
is 3.0 -- because the shift is what compresses the grid near sigma = 1, where
`torch.unique_consecutive` collapses f32 collisions. A port that skipped the
collapse would agree at 3.0 and diverge at 12.0.

    python examples/h3-dit/tools/gen_scheduler_golden.py --out examples/h3-dit/fixtures
"""

import argparse
import json
import pathlib

import torch

from diffusers import MiniMaxH3Scheduler


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    cases = []
    for shift in (12.0, 3.0):
        for steps in (2, 8, 32, 50, 1000):
            scheduler = MiniMaxH3Scheduler(shift=shift)
            scheduler.set_timesteps(steps)

            # One step of the sampler on a fixed sample and velocity, at the
            # first, middle and last schedule entries -- the last is where
            # `sigma_next` is 0 and the blend collapses to the denoised sample.
            generator = torch.Generator("cpu").manual_seed(steps)
            sample = torch.randn(7, generator=generator, dtype=torch.float32)
            velocity = torch.randn(7, generator=generator, dtype=torch.float32)
            steps_taken = []
            n = int(scheduler.timesteps.numel())
            for index in sorted({0, n // 2, n - 1}):
                scheduler._step_index = index
                t = scheduler.timesteps[index]
                out = scheduler.step(velocity, t, sample, return_dict=False)[0]
                steps_taken.append(
                    {"index": index, "timestep": float(t), "prevSample": out.tolist()}
                )

            noised = MiniMaxH3Scheduler(shift=shift).scale_noise(sample, 0.3, velocity)

            cases.append(
                {
                    "shift": shift,
                    "numInferenceSteps": steps,
                    "sigmas": scheduler.sigmas.tolist(),
                    "timesteps": scheduler.timesteps.tolist(),
                    "sample": sample.tolist(),
                    "velocity": velocity.tolist(),
                    "steps": steps_taken,
                    "scaleNoiseAt0_3": noised.tolist(),
                }
            )

    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "scheduler.json").write_text(
        json.dumps(
            {
                "source": "diffusers MiniMaxH3Scheduler",
                "note": "Arithmetic only -- no weights, no model licence.",
                "cases": cases,
            },
            indent=1,
        )
        + "\n"
    )
    for case in cases:
        print(
            f"shift {case['shift']:5} steps {case['numInferenceSteps']:4} -> "
            f"{len(case['sigmas'])} sigmas, {len(case['timesteps'])} evaluations"
        )
    print(f"wrote {out}/scheduler.json")


if __name__ == "__main__":
    main()
