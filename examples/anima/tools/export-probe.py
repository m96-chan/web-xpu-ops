"""What `torch.export` produces for Anima, and how much of it is shape.

Issue #185. The question behind (b) — "run any model" — is whether a forward can
be *data* rather than the 1,035 lines of TypeScript in `dit-resident.ts`. This
answers two parts of it against the **real architecture**, not a model of it:

1.  Does `torch.export` accept `MiniTrainDIT` at all, with dynamic H/W?
2.  Of the nodes it emits, how many need a kernel and how many are shape?

The second number decides the design. A graph that is mostly shape needs a layer
that resolves views *without dispatching*, and how big that layer has to be
depends on how many results come out non-contiguous — which is measured here
from `meta["val"]`, not reasoned about.

This is a probe, not part of the build: it needs a PyTorch and a ComfyUI checkout
that this repository does not depend on, so it takes both as arguments and is
never imported by anything.

    python examples/anima/tools/export-probe.py \
      --comfyui ~/path/to/ComfyUI --manifest ~/anima-q8/dit.manifest.json

Measured on 2026-08-25, torch 2.11, ComfyUI at diffusion-pipe's submodule:
static 38 ops / 408 nodes; dynamic H/W 42 ops / 421 nodes; of the shape-ish
results, 68 free, 26 free-with-offset, 69 strided — of which 12 are `view`
relabelling the `permute` immediately before it, leaving **14 permutes** as the
only results that need a copy. That is the same 14 `dit-resident.ts` issues by
hand.
"""

import argparse
import importlib.abc
import importlib.machinery
import json
import sys
import types
from collections import Counter

parser = argparse.ArgumentParser()
parser.add_argument("--comfyui", required=True, help="a ComfyUI checkout containing comfy/ldm/cosmos/predict2.py")
parser.add_argument("--manifest", required=True, help="dit.manifest.json from a converted Anima checkpoint")
parser.add_argument("--blocks", type=int, default=2)
parser.add_argument("--channels", type=int, default=512)
parser.add_argument("--heads", type=int, default=4)
parser.add_argument("--latent", type=int, default=16, help="H and W of the latent to trace with")
args = parser.parse_args()

sys.path.insert(0, args.comfyui)


# ComfyUI's import chain reaches `comfy_aimdo`, an optional native package.
# Stubbed rather than installed: nothing in the model uses it, and the question
# under test is whether the *model* exports.
class _Stub(types.ModuleType):
    def __getattr__(self, name):
        if name.startswith("__"):
            raise AttributeError(name)
        return _Stub(f"{self.__name__}.{name}")


class _Finder(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] != "comfy_aimdo":
            return None
        return importlib.machinery.ModuleSpec(fullname, self, is_package=True)

    def create_module(self, spec):
        return _Stub(spec.name)

    def exec_module(self, module):
        return None


sys.meta_path.insert(0, _Finder())

import torch  # noqa: E402
from torch.export import Dim, export  # noqa: E402

from comfy.ldm.cosmos.predict2 import MiniTrainDIT  # noqa: E402
import comfy.ops  # noqa: E402
import comfy.quant_ops  # noqa: E402


class _CK:
    """A pure-torch stand-in for `comfy.quant_ops.ck.apply_rope_split_half`.

    **Substituted, and that has to be stated wherever this run is quoted.** The
    checkout calls a native kernel that is not installed here. `rope_emb` is
    `[L, D/2, 2, 2]` rotation matrices built from `[cos, -sin, sin, cos]`
    (`position_embedding.py:150-163`), and the halves are split rather than
    adjacent — which is what `permuteForRope` in this repository converts.

    It uses only mul / add / cat, so the op inventory this script prints is not
    distorted by the substitution. It is still not upstream's code, and the
    fact that upstream reaches for a native kernel here is itself a finding for
    (b): `torch.export` emits those as opaque custom ops.
    """

    @staticmethod
    def apply_rope_split_half(q, k, rope_emb):
        def rot(x):
            d = x.shape[-1]
            x1, x2 = x[..., : d // 2], x[..., d // 2 :]
            m = rope_emb.to(x.dtype)
            c, ms, sn, c2 = m[..., 0, 0], m[..., 0, 1], m[..., 1, 0], m[..., 1, 1]
            while c.dim() < x1.dim():
                c, ms, sn, c2 = c.unsqueeze(-2), ms.unsqueeze(-2), sn.unsqueeze(-2), c2.unsqueeze(-2)
            return torch.cat([x1 * c + x2 * ms, x1 * sn + x2 * c2], dim=-1)

        return rot(q), rot(k)


comfy.quant_ops.ck = _CK()

cfg = json.load(open(args.manifest))["config"]

# Two blocks rather than 52 by default: the question is whether the *shape* of
# this model exports, and 52 copies of a block that exports is 52 copies of one
# answer. `--blocks 52` is there for when that stops being good enough.
kw = dict(
    max_img_h=cfg["max_img_h"],
    max_img_w=cfg["max_img_w"],
    max_frames=cfg["max_frames"],
    in_channels=cfg["in_channels"],
    out_channels=cfg["out_channels"],
    patch_spatial=cfg["patch_spatial"],
    patch_temporal=cfg["patch_temporal"],
    concat_padding_mask=cfg["concat_padding_mask"],
    model_channels=args.channels,
    num_blocks=args.blocks,
    num_heads=args.heads,
    crossattn_emb_channels=cfg["crossattn_emb_channels"],
    pos_emb_cls=cfg["pos_emb_cls"],
    pos_emb_interpolation=cfg["pos_emb_interpolation"],
    pos_emb_learnable=cfg["pos_emb_learnable"],
    min_fps=cfg["min_fps"],
    max_fps=cfg["max_fps"],
    use_adaln_lora=cfg["use_adaln_lora"],
    adaln_lora_dim=cfg["adaln_lora_dim"],
    rope_h_extrapolation_ratio=cfg["rope_h_extrapolation_ratio"],
    rope_w_extrapolation_ratio=cfg["rope_w_extrapolation_ratio"],
    rope_t_extrapolation_ratio=cfg["rope_t_extrapolation_ratio"],
    extra_h_extrapolation_ratio=cfg["extra_h_extrapolation_ratio"],
    extra_w_extrapolation_ratio=cfg["extra_w_extrapolation_ratio"],
    extra_t_extrapolation_ratio=cfg["extra_t_extrapolation_ratio"],
    rope_enable_fps_modulation=cfg["rope_enable_fps_modulation"],
    operations=comfy.ops.disable_weight_init,
    dtype=torch.float32,
)
model = MiniTrainDIT(**kw).eval()
print(
    f"built MiniTrainDIT: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M params, "
    f"{args.blocks} blocks at {args.channels}"
)

hw = args.latent
example = (
    torch.randn(1, cfg["in_channels"], 1, hw, hw),
    torch.rand(1, 1),
    torch.randn(1, 8, cfg["crossattn_emb_channels"]),
)
with torch.no_grad():
    print("eager forward OK:", tuple(model(*example).shape))

# `Dim("H", min=8, max=64)` is rejected: `patch_spatial = 2` makes H and W even,
# and export asks for that as a guard rather than discovering it at run time --
#     (-1)*((2 + (-1)*(W % 2)) % 2) >= 0
# `2 * Dim(...)` states it. `dit-resident.ts` relies on the same fact and says
# so nowhere, which is the argument for the IR carrying constraints at all.
DYNAMIC = {
    "x": {3: 2 * Dim("Hh", min=4, max=hw), 4: 2 * Dim("Wh", min=4, max=hw)},
    "timesteps": None,
    "context": None,
}


def opname(target):
    """Base name, for matching against `VIEWISH`."""
    schema = getattr(target, "_schema", None)
    return schema.name.split("::")[-1] if schema else str(target)


def qualified(target):
    """Overload-qualified, for the inventory.

    `to.dtype` and `to.dtype_layout` are different work — one is a cast, the
    other can change layout — so the inventory must not collapse them the way
    `opname` does.
    """
    schema = getattr(target, "_schema", None)
    if not schema:
        return str(target)
    overload = schema.overload_name or "default"
    return f"{schema.name.split('::')[-1]}.{overload}"


def kind_of(val):
    """`tensor` / `tuple` / `none` / `sym` — SymInt nodes are not work."""
    if isinstance(val, torch.Tensor):
        return "tensor"
    if isinstance(val, (list, tuple)):
        return "tuple"
    if val is None:
        return "none"
    return "sym"


for label, dynamic in [("static", None), ("dynamic H/W", DYNAMIC)]:
    try:
        ep = export(model, example, dynamic_shapes=dynamic)
    except Exception as exc:  # noqa: BLE001 — the failure text is the finding
        print(f"\n=== {label}: FAILED ===\n  {type(exc).__name__} {str(exc)[:600]}")
        continue

    ops = Counter()
    for node in ep.graph.nodes:
        if node.op == "call_function":
            ops[(qualified(node.target), kind_of(node.meta.get("val")))] += 1
    print(f"\n=== {label}: EXPORTED — {len({o for o, _ in ops})} distinct ops, {sum(ops.values())} nodes ===")
    for (op, kind), count in sorted(ops.items(), key=lambda kv: -kv[1]):
        print(f"  {count:5}  {op:38} {kind}")

# How much of the graph is shape, measured rather than assumed. `free` needs
# only metadata; `free+offset` needs a base address too; `strided` is the set
# that either has to be absorbed into the consuming kernel's indexing or copied.
VIEWISH = {
    "reshape", "select", "slice", "permute", "view", "unsqueeze",
    "expand", "transpose", "flatten", "chunk", "squeeze", "alias", "detach",
}

ep = export(model, example, dynamic_shapes=DYNAMIC)
classes, strided = Counter(), []
for node in ep.graph.nodes:
    if node.op != "call_function" or opname(node.target) not in VIEWISH:
        continue
    vals = node.meta.get("val")
    for val in vals if isinstance(vals, (list, tuple)) else [vals]:
        if not isinstance(val, torch.Tensor):
            continue
        name = opname(node.target)
        if not val.is_contiguous():
            classes[(name, "strided")] += 1
            strided.append((name, tuple(val.shape), tuple(val.stride()), val.storage_offset()))
        elif val.storage_offset() == 0:
            classes[(name, "free")] += 1
        else:
            classes[(name, "free+offset")] += 1

print("\n=== shape-ish results, by what resolving them costs ===")
totals = Counter()
for (name, cls), count in sorted(classes.items()):
    print(f"  {name:12} {cls:12} {count}")
    totals[cls] += count
print("  " + "  ".join(f"{cls}={count}" for cls, count in totals.most_common()))

# Printed in full rather than counted: 12 of these turned out to be a `view`
# repeating the shape *and stride* of the `permute` before it, which is a
# relabel and not work. A count alone would have hidden that.
print("\n=== every non-contiguous result ===")
for name, shape, stride, offset in strided:
    print(f"  {name:10} shape={shape} stride={stride} offset={offset}")
