#!/usr/bin/env python3
"""MiniMax-H3's visual VAE decoder, converted, plus a golden it has to reproduce.

Issue #200. Two jobs in one pass because the checkpoint is 10.42 GB and reading
it twice is a minute of disk for nothing:

  1. the decoder's 440 tensors -- 9.69 GB of f32 -- flattened into one buffer
     with a manifest, the way every other example here reads weights;
  2. a latent in and pixels out, produced by **the model's own `decode`**, for
     the port to be checked against.

The RoPE permutation is applied to `to_qkv` here rather than at run time. H3
rotates channel `c` with `c + 24` and `ops/rope`'s axes entry rotates adjacent
pairs inside per-axis blocks; permuting the q and k rows of the weight makes
them agree at no cost per forward. `permuteForRope` does the same for Anima.
**Q and K only** -- V is never rotated, and permuting it would reorder channels
the output projection reads in the original order.

    python examples/h3-video/tools/convert_decoder.py \
      --bundle ~/h3/video_vae --weights ~/h3/video_vae/model.safetensors \
      --out ~/h3-video-web --dims 2,3,4

**The weights are not this repository's and are not redistributed by it.** The
model is under the MiniMax H3 Community License Agreement; this reads a copy the
user obtained themselves. See issue #190.
"""

import argparse
import json
import math
import pathlib
import sys

import numpy as np
import torch
from safetensors import safe_open

# `source/config.json`'s `vit_decoder_kwargs`, plus the values
# `AutoencoderKLLegacy` derives. Stated rather than read so a checkpoint with a
# different geometry fails the comparison instead of the converter.
CONFIG = {
    "heads": 32, "dim_head": 64, "num_layers": 36,
    "norm_type": "rms_norm", "norm_affine": True,
    "qk_norm_type": "rms_norm", "qk_norm_affine": False,
    "ffn_activation_fn": "silu", "ffn_use_gated": True,
    "rope_theta": 100.0, "rope_dim_ratio": 0.75,
    "patch_size": 16, "patch_size_t": 4,
    "in_channels": 24, "out_channels": 3,
    "num_register_tokens": 4, "eps": 1e-5,
}
# `pixel_norm_type: "imagenet"`. The decoder's output is normalised, and
# `get_denormalize_transform` undoes it as `x * std + mean`.
PIXEL_MEAN = (0.485, 0.456, 0.406)
PIXEL_STD = (0.229, 0.224, 0.225)


def rope_permutation(dim_head: int, rot_dim: int) -> list[int]:
    """H3 channel order -> `ropeAxes` channel order. Same table as `ops/rope`'s."""
    perm = [0] * dim_head
    per_axis = rot_dim // 2 // 3
    for c in range(rot_dim):
        half, rest = divmod(c, rot_dim // 2)
        axis, freq = divmod(rest, per_axis)
        perm[axis * (2 * per_axis) + 2 * freq + half] = c
    for c in range(rot_dim, dim_head):
        perm[c] = c
    return perm


def split_qkv(weight: torch.Tensor, bias: torch.Tensor, heads: int, dim_head: int, perm: list[int]):
    """`to_qkv` into three separate projections, with q and k permuted for RoPE.

    Two things happen here that would otherwise happen every forward.

    **The split.** The model stores one `[3 * heads * dim_head, dim]` matrix and
    reads it as `view(B, L, -1, 3 * dim_head).chunk(3, dim=-1)`, so head `h` owns
    the contiguous rows `[q(64), k(64), v(64)]`. Taking `q` out of the *output*
    of one matmul means a strided copy per token; taking it out of the *weight*
    means three ordinary matmuls, which is what Z-Image's `to_q`/`to_k`/`to_v`
    already are. Same arithmetic, one fewer kernel.

    **The RoPE permutation**, on q and k only. H3 rotates channel `c` with
    `c + 24`; `ops/rope`'s axes entry rotates adjacent pairs inside per-axis
    blocks. Permuting the weight rows makes them agree at no run-time cost --
    `permuteForRope` does the same for Anima. V is never rotated, and permuting
    it would reorder channels the output projection reads in the original order.
    """
    width = heads * dim_head
    out = []
    for part in range(3):
        w = torch.empty(width, weight.shape[1], dtype=weight.dtype)
        b = torch.empty(width, dtype=bias.dtype)
        for head in range(heads):
            base = head * 3 * dim_head + part * dim_head
            for c in range(dim_head):
                src = base + (perm[c] if part < 2 else c)
                w[head * dim_head + c] = weight[src]
                b[head * dim_head + c] = bias[src]
        out.append((w, b))
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", required=True)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--dims", default="2,3,4", help="latent T,H,W for the golden")
    parser.add_argument("--skip-golden", action="store_true")
    parser.add_argument("--quant", choices=["f32", "q8"], default="f32",
                        help="how the big matrices are stored; q8 is 2.4 GB against 9.69")
    # For a second golden at a different size without rewriting 9.69 GB that
    # would come out byte-identical.
    parser.add_argument("--skip-weights", action="store_true")
    args = parser.parse_args()

    bundle = pathlib.Path(args.bundle).expanduser().resolve()
    out = pathlib.Path(args.out).expanduser()
    out.mkdir(parents=True, exist_ok=True)

    heads, dim_head = CONFIG["heads"], CONFIG["dim_head"]
    dim = heads * dim_head
    rot_dim = int(dim_head * CONFIG["rope_dim_ratio"])
    perm = rope_permutation(dim_head, rot_dim)

    state = {}
    with safe_open(args.weights, framework="pt", device="cpu") as f:
        for key in f.keys():
            if key.startswith("decoder.") or key.startswith("post_quant_conv."):
                state[key] = f.get_tensor(key)
    print(f"read {len(state)} tensors")

    entries, offset = [], 0
    written = out / ("decoder.bin" if args.quant == "f32" else "decoder.q8.bin")
    with (written.open("wb") if not args.skip_weights else open("/dev/null", "wb")) as sink:
        def quantize_rows(tensor: torch.Tensor):
            """Per-row absmax int8, `ops/quantize`'s convention exactly.

            `[-127, 127]`, not `[-128, 127]`: symmetry matters more than the one
            extra level, since an asymmetric range biases the dequantised result.
            An all-zero row keeps a scale of 1 and codes of 0 rather than
            dividing by zero.

            **Widening the clamp to -128 changes nothing here, and that is
            structural rather than lucky**: the scale is `absmax / 127`, so the
            most negative value in a row maps to exactly -127 and nothing ever
            reaches -128. Measured -- the decoded pixels are identical, two
            levels of 255 either way. The choice would matter under a scale
            convention that is not absmax-derived.

            Two mutations that *are* observable, measured the same way: taking
            the scale as `absmax` rather than `absmax / 127` costs **230 levels
            of 255**, and reversing the byte order inside each u32 costs 196.
            One is only weakly observable: quantising per *tensor* instead of
            per output row costs **5 levels** against the correct 2 -- this
            checkpoint's rows are similar enough in scale that the finer
            granularity buys less than it usually does, which is worth knowing
            before anyone spends effort on a finer one still.

            The rows are **output channels**, which is what `matmulQ8` wants: it
            reads the weight as `[M, ceil(K/4)]` with one scale per `M`, so the
            q8 path takes `nn.Linear`'s `[out, in]` **untransposed** — the
            opposite of the f32 path, whose kernel reads `[in, out]`.
            """
            values = tensor.to(torch.float32)
            absmax = values.abs().amax(dim=1)
            scale = torch.where(absmax == 0, torch.ones_like(absmax), absmax / 127.0)
            inverse = torch.where(absmax == 0, torch.zeros_like(absmax), 127.0 / absmax)
            codes = torch.clamp(torch.round(values * inverse.unsqueeze(1)), -127, 127).to(torch.int8)
            # Four codes per u32, least-significant byte first -- `matvecQ8`'s
            # wire format. Rows are padded to a whole word.
            rows, k = codes.shape
            words = (k + 3) // 4
            padded = torch.zeros(rows, words * 4, dtype=torch.uint8)
            padded[:, :k] = codes.view(torch.uint8)
            packed = padded.view(rows, words, 4).to(torch.int32)
            word = (packed[:, :, 0] | (packed[:, :, 1] << 8) | (packed[:, :, 2] << 16) | (packed[:, :, 3] << 24))
            return word.to(torch.int32).numpy().astype(np.uint32), scale.numpy().astype(np.float32)

        def add_matrix(name, tensor):
            """A weight matrix, in whichever form `--quant` asked for."""
            if args.quant == "f32":
                add(name, tensor, transpose=True)
                return
            words, scale = quantize_rows(tensor)
            nonlocal offset
            entries.append({"name": name, "shape": list(tensor.shape), "offset": offset,
                            "count": int(words.size), "kind": "q8"})
            sink.write(words.tobytes())
            offset += int(words.size)
            entries.append({"name": f"{name}.scale", "shape": [tensor.shape[0]], "offset": offset,
                            "count": int(scale.size), "kind": "f32"})
            sink.write(scale.tobytes())
            offset += int(scale.size)

        def add(name, tensor, transpose=False):
            """Appends one tensor, optionally as `[in, out]` rather than `[out, in]`.

            `ops/matmul`'s kernel reads `b` as `[K, N]` and `nn.Linear` stores
            `[out, in]`, so every weight matrix is transposed **here**, once.
            Doing it per forward would move 9.69 GB through a transpose kernel
            to compute the same numbers.
            """
            nonlocal offset
            if transpose:
                tensor = tensor.t()
            array = tensor.detach().to(torch.float32).contiguous().numpy().ravel()
            entries.append({"name": name, "shape": list(tensor.shape), "offset": offset, "count": int(array.size)})
            sink.write(array.tobytes())
            offset += int(array.size)

        # `post_quant_conv` is a 1x1x1 Conv3d -- a per-token linear. Stored with
        # its trailing singleton dims squeezed so the port reads a matrix.
        add_matrix("post_quant.weight", state["post_quant_conv.weight"].reshape(CONFIG["in_channels"], -1))
        add("post_quant.bias", state["post_quant_conv.bias"])

        add_matrix("x_embedder.weight", state["decoder.x_embedder.weight"])
        add("x_embedder.bias", state["decoder.x_embedder.bias"])
        add("register_tokens", state["decoder.register_tokens"])

        for i in range(CONFIG["num_layers"]):
            p = f"decoder.transformer_blocks.{i}."
            parts = split_qkv(state[p + "attn.to_qkv.weight"], state[p + "attn.to_qkv.bias"], heads, dim_head, perm)
            add(f"blocks.{i}.norm1.weight", state[p + "norm1.weight"])
            for name, (w, b) in zip(("q", "k", "v"), parts):
                add_matrix(f"blocks.{i}.{name}.weight", w)
                add(f"blocks.{i}.{name}.bias", b)
            add_matrix(f"blocks.{i}.out.weight", state[p + "attn.to_out.weight"])
            add(f"blocks.{i}.out.bias", state[p + "attn.to_out.bias"])
            add(f"blocks.{i}.scale1", state[p + "scale1"])
            add(f"blocks.{i}.norm2.weight", state[p + "norm2.weight"])
            # `chunk(2, dim=-1)` on the *output* is a strided copy per token;
            # splitting the weight is two matmuls that land contiguous. The
            # **gate is the first half** -- swapping them is a different
            # function whose output has the same shape and a plausible range.
            inner = state[p + "ff.w1.weight"].shape[0] // 2
            add_matrix(f"blocks.{i}.gate.weight", state[p + "ff.w1.weight"][:inner])
            add(f"blocks.{i}.gate.bias", state[p + "ff.w1.bias"][:inner])
            add_matrix(f"blocks.{i}.up.weight", state[p + "ff.w1.weight"][inner:])
            add(f"blocks.{i}.up.bias", state[p + "ff.w1.bias"][inner:])
            add_matrix(f"blocks.{i}.w2.weight", state[p + "ff.w2.weight"])
            add(f"blocks.{i}.w2.bias", state[p + "ff.w2.bias"])
            add(f"blocks.{i}.scale2", state[p + "scale2"])

        add("norm_out.weight", state["decoder.norm_out.weight"])
        add("norm_out.bias", state["decoder.norm_out.bias"])
        add_matrix("proj_out.weight", state["decoder.proj_out.weight"])
        add("proj_out.bias", state["decoder.proj_out.bias"])

    manifest = {
        "model": "minimax-h3-video-vae-decoder",
        "source": "MiniMaxAI/MiniMax-H3 (FL2VA/video_vae/source)",
        "licence": "MiniMax H3 Community License Agreement — not this repository's, and not redistributed by it",
        "config": CONFIG, "dim": dim, "ffnHidden": dim * 4, "ropeApplyDim": rot_dim,
        "ropeAxisDims": [2 * (rot_dim // 2 // 3)] * 3 + [dim_head - rot_dim],
        "pixelMean": list(PIXEL_MEAN), "pixelStd": list(PIXEL_STD),
        "ropePermutation": perm,
        "dtype": args.quant,
        "weightLayout": "[in, out]" if args.quant == "f32" else "[out, in], int8 packed four per u32", "tensors": entries, "elements": offset,
    }
    if not args.skip_weights:
        (out / ("decoder.manifest.json" if args.quant == "f32" else "decoder.q8.manifest.json")).write_text(json.dumps(manifest, indent=1))
        print(f"wrote {written}  {offset * 4 / 1e9:.2f} GB, {len(entries)} tensors")
    else:
        print(f"skipped the weights; {len(entries)} tensors would have been {offset * 4 / 1e9:.2f} GB")

    if args.skip_golden:
        return 0

    sys.path.insert(0, str(bundle.parent))
    from video_vae.base_module import RotaryEmbeddingND, TransformerBlock  # noqa: F401
    from video_vae.vae_vit import ViT3DDecoder  # type: ignore[import-not-found]

    decoder = ViT3DDecoder(
        patch_size=CONFIG["patch_size"], patch_size_t=CONFIG["patch_size_t"], t_causal=False,
        in_channels=CONFIG["in_channels"], out_channels=CONFIG["out_channels"],
        num_layers=CONFIG["num_layers"], heads=heads, dim_head=dim_head,
        norm_type=CONFIG["norm_type"], norm_affine=CONFIG["norm_affine"],
        qk_norm_type=CONFIG["qk_norm_type"], qk_norm_affine=CONFIG["qk_norm_affine"],
        ffn_activation_fn=CONFIG["ffn_activation_fn"], ffn_use_gated=CONFIG["ffn_use_gated"],
        rope_theta=CONFIG["rope_theta"], rope_dim_ratio=CONFIG["rope_dim_ratio"],
        eps=CONFIG["eps"], num_register_tokens=CONFIG["num_register_tokens"],
    ).eval()
    missing, unexpected = decoder.load_state_dict(
        {k[len("decoder.") :]: v for k, v in state.items() if k.startswith("decoder.")}, strict=False
    )
    if unexpected:
        raise SystemExit(f"checkpoint has decoder tensors this build does not: {unexpected}")
    if [m for m in missing if "mask_token" not in m]:
        raise SystemExit(f"decoder wants tensors the checkpoint does not have: {missing}")

    post_w = state["post_quant_conv.weight"].reshape(CONFIG["in_channels"], -1)
    post_b = state["post_quant_conv.bias"]

    T, H, W = (int(v) for v in args.dims.split(","))
    torch.manual_seed(20260826)
    latent = torch.randn(1, CONFIG["in_channels"], T, H, W) * 0.5
    with torch.no_grad():
        # `AutoencoderKLLegacy.decode`: post_quant_conv, then the ViT.
        z = torch.einsum("oc,bcthw->bothw", post_w, latent) + post_b.view(1, -1, 1, 1, 1)
        pixels = decoder(z)
    print(f"latent {tuple(latent.shape)} -> pixels {tuple(pixels.shape)}")
    print(f"  range [{pixels.min():.4f}, {pixels.max():.4f}]  mean {pixels.mean():.4f}")

    fixtures = out / f"golden-{T}x{H}x{W}" if args.skip_weights else out
    fixtures.mkdir(parents=True, exist_ok=True)
    latent.numpy().astype(np.float32).tofile(fixtures / "latent.bin")
    pixels.numpy().astype(np.float32).tofile(fixtures / "pixels.bin")
    (fixtures / "golden.json").write_text(json.dumps({
        "dims": [T, H, W], "latentChannels": CONFIG["in_channels"],
        "frames": T * CONFIG["patch_size_t"], "height": H * CONFIG["patch_size"], "width": W * CONFIG["patch_size"],
        "torch": torch.__version__, "seed": 20260826,
    }, indent=1))
    print(f"  wrote {fixtures}/pixels.bin  {pixels.numel()} values")
    return 0


if __name__ == "__main__":
    sys.exit(main())
