#!/usr/bin/env python3
"""Qwen3-VL's text decoder, at a size small enough to check in.

Issue #212. MiniMax-H3 conditions `ref2va` on `hidden_states[50]` of this stack,
so R2V cannot precompute its conditioning the way `examples/h3-dit-web` does —
the reference *is* the input. What that costs is 24.87 GB of int8 for layers
0..50 alone, and what it needs is this forward.

**The weights are random, and that is the point.** The arithmetic is
`transformers`' `Qwen3VLTextModel` — instantiated, not transcribed — at a
geometry small enough to commit, so the fixture runs in CI on any machine and
carries no model licence. What it establishes is structure, and none of that
depends on the weights being trained.

Four things this stack does that the DiT does not:

- **64 query heads over 8 key/value heads.** Grouped-query attention, which
  `ops/gqa` exists for; the tiny config keeps the ratio at 2:1 so a port that
  ignored the grouping would read the wrong keys.
- **Interleaved M-RoPE.** `mrope_section` says how many of the `head_dim / 2`
  frequencies each of the three axes owns, but they are **not contiguous**:
  channel `c` belongs to axis `c % 3` while `c < 3 * section[1]`, and to the
  time axis after that. Upstream calls this "preserving frequency continuity" —
  every channel keeps the frequency its *global* index gives it, whichever axis
  it reads its position from.
- **QK-norm per head**, with weights, before the rotation.
- Causal attention with a **shared** position grid of three rows, one per axis.

    python examples/h3-ref2v/tools/gen_text_encoder_golden.py --out examples/h3-ref2v/fixtures
"""

import argparse
import json
import pathlib

import torch

from transformers import Qwen3VLTextConfig
from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLTextModel

# `head_dim // 2` is 8, and `mrope_section` sums to it. The interleave then puts
# H at channels 1 and 4, W at 2 and 5, and leaves 0, 3, 6, 7 to T -- four, which
# is `mrope_section[0]`. A section list that summed differently would be a
# different model.
CONFIG = {
    "vocab_size": 64,
    "hidden_size": 32,
    "intermediate_size": 64,
    "num_hidden_layers": 3,
    "num_attention_heads": 2,
    "num_key_value_heads": 1,
    "head_dim": 16,
    "rms_norm_eps": 1e-6,
    "attention_bias": False,
    "hidden_act": "silu",
    "max_position_embeddings": 128,
    "rope_theta": 5000000.0,
    "rope_scaling": {"rope_type": "default", "mrope_section": [4, 2, 2], "mrope_interleaved": True},
}

SEQ = 7


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    config = Qwen3VLTextConfig(**CONFIG)
    # f64 throughout: this port stores intermediates in `Float32Array`, and an
    # f32 golden would measure torch's rounding as much as the port's.
    model = Qwen3VLTextModel(config).to(torch.float64).eval()

    generator = torch.Generator("cpu").manual_seed(args.seed)
    embeds = torch.randn(1, SEQ, CONFIG["hidden_size"], generator=generator, dtype=torch.float64)

    # `(3, batch, seq)` — t, h, w. Text tokens sit at the same value on all
    # three; a vision block's tokens do not, which is the whole reason the grid
    # has three rows. Rows 2..5 here are a 2x2 vision block.
    t = [0, 1, 2, 2, 2, 2, 3]
    h = [0, 1, 0, 0, 1, 1, 2]
    w = [0, 1, 0, 1, 0, 1, 2]
    position_ids = torch.tensor([[t], [h], [w]], dtype=torch.long)

    with torch.no_grad():
        out = model(inputs_embeds=embeds, position_ids=position_ids, use_cache=False,
                    output_hidden_states=True, return_dict=True)

    blob = bytearray()
    tensors = []

    def put(name: str, x: torch.Tensor) -> None:
        a = x.detach().to(torch.float32).contiguous().numpy().astype("<f4")
        tensors.append({"name": name, "shape": list(x.shape), "offset": len(blob), "count": a.size})
        blob.extend(a.tobytes())

    for name, param in model.state_dict().items():
        put(name, param)
    put("input.embeds", embeds)
    # Every hidden state, so a port can bisect a divergence to a layer instead
    # of only learning that the last one is wrong.
    for index, state in enumerate(out.hidden_states):
        put(f"output.hidden.{index}", state)

    out_dir = pathlib.Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "text-encoder.bin").write_bytes(bytes(blob))
    (out_dir / "text-encoder.json").write_text(json.dumps({
        "source": "transformers Qwen3VLTextModel, random weights",
        "note": "Not Qwen's checkpoint. Random weights at a tiny geometry; no model licence applies.",
        "seed": args.seed,
        "config": CONFIG,
        "seq": SEQ,
        "positionIds": {"t": t, "h": h, "w": w},
        "numHiddenStates": len(out.hidden_states),
        "tensors": tensors,
    }, indent=1) + "\n")

    print(f"{len(tensors)} tensors, {len(blob) / 1024:.1f} KB, {len(out.hidden_states)} hidden states")
    last = out.hidden_states[-1]
    print(f"final {tuple(last.shape)}  |h| max {last.abs().max():.4f}")
    print(f"wrote {out_dir}/text-encoder.json and text-encoder.bin")


if __name__ == "__main__":
    main()
