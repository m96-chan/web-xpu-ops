#!/usr/bin/env python3
"""Bakes a golden for Anima's sampler, from ComfyUI's own code.

Issue #170 stage 5. Anima does not use the Euler-on-a-linear-schedule that
`examples/zimage` needed. Its released workflow asks for `res_multistep` on a
`beta` schedule, and the settings come from three separate places, none of them
a default:

| | where |
| --- | --- |
| `shift` 3.0, `multiplier` 1.0 | `supported_models.py:1136` |
| `beta` schedule, 40 steps, CFG 8 | `workflows/workflow_2_pass.json` |
| `res_multistep` with `eta=0` | `sample_res_multistep`, `sampling.py:1459` |

`multiplier` 1.0 is the one worth saying out loud: `ModelSamplingDiscreteFlow`
defaults it to 1000, and Z-Image's port multiplies the timestep by 1000 for
exactly that reason. Anima's model reads sigma **as** the timestep. A port that
carries Z-Image's habit across gets a plausible image conditioned on the wrong
point of the trajectory, which is the failure this whole file exists to catch.

Two goldens, because the sampler has two independently wrong-able halves:

  * `sigmas` — the schedule. `beta_scheduler` inverts a Beta(0.6, 0.6) CDF and
    rounds onto a 1000-entry table, so it is a special function and a rounding
    convention, not a formula.
  * `trajectory` — `res_multistep` driven by a **toy denoiser**, so the sampler
    is pinned without the DiT. The toy is nonlinear on purpose: a linear one
    cannot tell a second-order multistep step from a first-order one.

    PYTHONPATH=/tmp/comfy-venv/lib/python3.12/site-packages \\
        /tmp/comfy-venv/bin/python examples/anima/tools/gen_sampler_golden.py
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

COMFY = Path("/tmp/ComfyUI")
HERE = Path(__file__).resolve().parent

# `supported_models.Anima.sampling_settings`, not a default.
SHIFT = 3.0
MULTIPLIER = 1.0
# `workflow_2_pass.json`: node 8 is `['beta', 40, 1]`, node 173 `['beta', 14, 0.5]`.
#
# 200 and 1000 are not settings anyone would use; they are here because they are
# where `beta_scheduler`'s duplicate-dropping becomes observable. Below 200 every
# rounded index is distinct, so a port that never drops duplicates returns the
# same schedule and no test at the shipped step counts can tell. Measured: 200
# steps collapse to 199 sigmas, 1000 steps to 847.
STEP_COUNTS = [40, 14, 8, 1, 200, 1000]


class ToyDenoiser:
    """Stands in for the DiT, and for the plumbing `res_multistep` reaches into.

    `res_multistep` reads `model.inner_model.model_patcher.get_model_object(
    'model_sampling').noise_scale` before its loop (`sampling.py:1398`). With
    `eta=0` the value is multiplied into a `sigma_up` that is always zero, so it
    changes nothing — but the attribute chain is walked whether or not the
    result matters, and a stub that omits it fails before the first step.
    """

    def __init__(self, sampling):
        self.inner_model = self
        self.model_patcher = self
        self._sampling = sampling

    def get_model_object(self, _name):
        return self._sampling

    def __call__(self, x, sigma, **_kwargs):
        # Nonlinear in both arguments. A denoiser linear in `x` would make the
        # second-order branch agree with Euler to within rounding, and the
        # golden would pass on a port that never implemented it.
        s = sigma.reshape(-1, *([1] * (x.ndim - 1)))
        return x * 0.85 - torch.sin(x) * s + 0.1 * s * s


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=HERE.parent / "fixtures")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    if not COMFY.exists():
        sys.exit(f"{COMFY} not found — see gen_block_golden.py for the clone command")
    sys.path.insert(0, str(COMFY))

    from comfy.k_diffusion.sampling import sample_res_multistep
    from comfy.model_base import ModelType, model_sampling
    from comfy.samplers import beta_scheduler

    torch.set_grad_enabled(False)
    torch.manual_seed(args.seed)

    class Settings:
        sampling_settings = {"shift": SHIFT, "multiplier": MULTIPLIER}

    # `model_sampling(config, FLOW)` composes `CONST` onto
    # `ModelSamplingDiscreteFlow` (`model_base.py:120`). The table alone is not
    # enough: `noise_scaling` and `calculate_denoised` — how the first latent is
    # built and how a model output becomes a denoised one — live on `CONST`, and
    # picking a different mixin is exactly the silent mistake to rule out.
    sampling = model_sampling(Settings(), ModelType.FLOW)
    print(f"model sigmas: {len(sampling.sigmas)} entries, "
          f"[{float(sampling.sigmas[0]):.6g} .. {float(sampling.sigmas[-1]):.6g}]")

    schedules = {}
    for steps in STEP_COUNTS:
        sigmas = beta_scheduler(sampling, steps)
        schedules[str(steps)] = [float(s) for s in sigmas]
        print(f"  beta {steps:3d} steps -> {len(sigmas)} sigmas, "
              f"first {float(sigmas[0]):.6g}, last-but-one {float(sigmas[-2]):.6g}")

    # A small latent, in the shape Anima actually samples: `[B, 16, T, H, W]`.
    # 16 channels is `latent_formats.Wan21.latent_channels`; the port's own
    # `T=1` still has to be a dimension because the DiT patchifies over it.
    x = torch.randn(1, 16, 1, 4, 4)
    sigmas = beta_scheduler(sampling, 8)
    model = ToyDenoiser(sampling)

    trajectory = []
    sample = sample_res_multistep(
        model, x.clone(), sigmas,
        callback=lambda info: trajectory.append(info["x"].clone()),
        disable=True,
    )
    print(f"trajectory: {len(trajectory)} callbacks, final {tuple(sample.shape)}")

    # `noise_scaling` / `inverse_noise_scaling` — how the first latent is built
    # and how the last is read back. Pinned here because `CONST` is not the only
    # convention in ComfyUI and picking the wrong one is silent.
    # Two cases, because the one that matters is not the one that runs.
    # Text-to-image starts at sigma 1.0 with a zero image, where
    # `sigma * noise + (1 - sigma) * image` is just `noise` and every wrong
    # formula that keeps the noise term agrees. The second case has a sigma
    # strictly inside the schedule and a non-zero image, which separates them.
    noise = torch.randn_like(x)
    latent_image = torch.zeros_like(x)
    mid_sigma = float(sigmas[len(sigmas) // 2])
    mid_image = torch.randn_like(x)
    noise_cases = [
        {
            "sigma": float(sigmas[0]),
            "noise": noise.flatten().tolist(),
            "image": latent_image.flatten().tolist(),
            "scaled": sampling.noise_scaling(sigmas[0], noise, latent_image, max_denoise=True).flatten().tolist(),
        },
        {
            "sigma": mid_sigma,
            "noise": noise.flatten().tolist(),
            "image": mid_image.flatten().tolist(),
            "scaled": sampling.noise_scaling(
                torch.tensor(mid_sigma), noise, mid_image, max_denoise=False
            ).flatten().tolist(),
        },
    ]

    args.out.mkdir(parents=True, exist_ok=True)
    out = args.out / "sampler-golden.json"
    out.write_text(json.dumps({
        "shift": SHIFT,
        "multiplier": MULTIPLIER,
        "modelSigmas": {
            "count": len(sampling.sigmas),
            "first": float(sampling.sigmas[0]),
            "last": float(sampling.sigmas[-1]),
            # Every hundredth, so a port whose `time_snr_shift` is wrong in the
            # middle of the table fails even though both ends agree.
            "every100": [float(sampling.sigmas[i]) for i in range(0, len(sampling.sigmas), 100)],
        },
        "betaSchedules": schedules,
        "trajectory": {
            "sigmas": [float(s) for s in sigmas],
            "x0": x.flatten().tolist(),
            "steps": [t.flatten().tolist() for t in trajectory],
            "final": sample.flatten().tolist(),
        },
        "noiseScaling": noise_cases,
    }, indent=1))
    print(f"wrote {out} ({out.stat().st_size / 1e3:.0f} kB)")


if __name__ == "__main__":
    main()
