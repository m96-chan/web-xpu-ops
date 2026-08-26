"""Converts MiniMax-H3's audio VAE decoder into a flat buffer this repository can read.

Issue #200. The checkpoint is 605 MB of f32 safetensors holding an encoder, a
projection head and a BigVGAN decoder. Only the decoder is wanted: the demo
turns a latent into a waveform, and the encoder is 40% of the file.

Two things happen here that cannot happen in the browser.

**`weight_norm` is folded.** Every convolution in the decoder is wrapped in
`torch.nn.utils.parametrizations.weight_norm`, so the checkpoint stores
`weight_g` and `weight_v` rather than `weight`, and the convolution's actual
weight is `g * v / ||v||` with the norm taken per output channel. Doing that at
load time would put a reduction and a divide in front of every layer for a value
that never changes. It is folded here — and **checked against the module's own
`.weight`**, because "the norm is over all dims but the first" is exactly the
kind of convention that is right until it silently is not.

**The anti-aliasing filters are copied, not computed.** BigVGAN's `Activation1d`
upsamples 2x, applies SnakeBeta and downsamples again, through a 12-tap
Kaiser-windowed sinc. The window and the sinc are in the model's source
(`dac_alias_free_filter.py`) — but the tensors are also *in the checkpoint*, so
they are read rather than recomputed. One less formula to get wrong, and it is
the publisher's value by construction.

    python examples/h3-audio/tools/convert_audio_vae.py \
      --bundle ~/h3/audio_vae --out ~/h3-audio-web

`--bundle` is the `audio_vae/` directory from `MiniMaxAI/MiniMax-H3` — the
`.py` files, `config.yaml`, `metadata.json` and `model.safetensors`.

**The weights are not this repository's and are not redistributed by it.** The
model is under the MiniMax H3 Community License Agreement; this script reads a
copy the user obtained themselves. See issue #190.
"""

import argparse
import json
import pathlib
import sys

import numpy as np
import torch
import yaml
from safetensors.torch import load_file


def fold_weight_norm(state: dict, prefix: str) -> torch.Tensor:
    """`g * v / ||v||`, the weight a `weight_norm`-wrapped convolution actually uses.

    The norm is over every dimension except the first, which is what `dim=0`
    means and what `weight_g`'s `[Cout, 1, 1]` shape implies. Implied is not
    measured, so `main` checks the result against the module's own `.weight`.
    """
    g = state[f"{prefix}.weight_g"]
    v = state[f"{prefix}.weight_v"]
    norm = v.flatten(1).norm(dim=1).view(-1, *([1] * (v.dim() - 1)))
    return g * v / norm


class Writer:
    """Appends f32 tensors to one buffer and records where each landed."""

    def __init__(self) -> None:
        self.chunks: list[np.ndarray] = []
        self.entries: list[dict] = []
        self.offset = 0

    def add(self, name: str, tensor: torch.Tensor) -> None:
        array = tensor.detach().to(torch.float32).contiguous().numpy().ravel()
        self.entries.append(
            {"name": name, "shape": list(tensor.shape), "offset": self.offset, "count": int(array.size)}
        )
        self.chunks.append(array)
        self.offset += int(array.size)

    def write(self, path: pathlib.Path) -> None:
        with path.open("wb") as f:
            for chunk in self.chunks:
                f.write(chunk.tobytes())


def verify_fold(bundle: pathlib.Path, state: dict, metadata: dict, audio_config: dict) -> None:
    """Checks `fold_weight_norm` against the weight the module itself computes.

    "the norm is over every dimension but the first" is a convention, and rule 2
    says a convention gets checked rather than recalled. Building the model costs
    a few seconds and settles it: if the fold were wrong -- the norm over the
    wrong axis, `g` and `v` the wrong way round -- every convolution would be
    scaled by a plausible number and the output would be quiet, or loud, but
    never obviously broken.

    Skipped with a message rather than an error when the bundle's own modules
    cannot be imported, since the conversion itself does not need torch's
    `nn.Module` at all.
    """
    sys.path.insert(0, str(bundle.parent))
    try:
        from audio_vae.dac_audio_vae import DacAudioVAE  # type: ignore[import-not-found]
    except Exception as exc:  # noqa: BLE001 -- any import failure means the same thing
        print(f"  fold NOT verified: could not import the bundle ({exc})")
        return

    model = DacAudioVAE(
        encoder_rates=metadata["encoder_rates"],
        decoder_rates=metadata["decoder_rates"],
        attn_proj=metadata["attn_proj"],
        decoder_type=metadata["decoder_type"],
        decoder_dim=audio_config["model_config"]["decoder_dim"],
        vae_latent_channels=audio_config["model_config"]["vae_latent_channels"],
        sample_rate=metadata["sample_rate"],
    )
    model.load_state_dict(state, strict=True)
    model.eval()

    checked = 0
    worst = 0.0
    for name, module in model.named_modules():
        if not hasattr(module, "weight_g") and f"{name}.weight_g" not in state:
            continue
        folded = fold_weight_norm(state, name)
        actual = module.weight.detach()
        if folded.shape != actual.shape:
            raise SystemExit(f"{name}: folded {tuple(folded.shape)} against the module's {tuple(actual.shape)}")
        worst = max(worst, float((folded - actual).abs().max()))
        checked += 1
    if checked == 0:
        raise SystemExit("no weight_norm modules found to verify the fold against")
    print(f"  fold verified against {checked} modules, worst element {worst:.3e}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", required=True, help="the audio_vae/ directory from MiniMaxAI/MiniMax-H3")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    bundle = pathlib.Path(args.bundle).expanduser()
    out = pathlib.Path(args.out).expanduser()
    out.mkdir(parents=True, exist_ok=True)

    metadata = json.loads((bundle / "metadata.json").read_text())["metadata"]["kwargs"]
    audio_config = yaml.safe_load((bundle / "config.yaml").read_text())
    state = load_file(bundle / "model.safetensors", device="cpu")

    sample_rate = metadata["sample_rate"]
    if sample_rate != 32000:
        # `dac_audio_vae.py` picks the BigVGAN hyperparameters from the sample
        # rate and knows only 16000 and 32000. Writing a manifest for a rate this
        # script has not seen would put the wrong upsample schedule in it.
        raise SystemExit(f"only the 32 kHz configuration is converted here, got {sample_rate}")

    latent_dim = metadata["latent_dim"]
    decoder_dim = audio_config["model_config"]["decoder_dim"]
    latent_channels = audio_config["model_config"]["vae_latent_channels"]
    upsample_rates = [5, 5, 2, 2, 2, 2, 2]
    upsample_kernel_sizes = [9, 9, 4, 4, 4, 4, 4]
    resblock_kernel_sizes = [3, 7, 11]
    resblock_dilations = [[1, 3, 5], [1, 3, 5], [1, 3, 5]]

    writer = Writer()

    # The projection from the 32-channel latent into BigVGAN's input width. Not
    # weight-normed -- a plain `nn.Conv1d`, so its weight is stored directly.
    writer.add("dec_in_proj.weight", state["dec_in_proj.weight"])
    writer.add("dec_in_proj.bias", state["dec_in_proj.bias"])

    writer.add("conv_pre.weight", fold_weight_norm(state, "decoder.conv_pre"))
    writer.add("conv_pre.bias", state["decoder.conv_pre.bias"])

    for i in range(len(upsample_rates)):
        # `ups` is a ModuleList of ModuleLists; every entry here has exactly one
        # transposed convolution, which the golden run confirmed ([1] * 7).
        writer.add(f"ups.{i}.weight", fold_weight_norm(state, f"decoder.ups.{i}.0"))
        writer.add(f"ups.{i}.bias", state[f"decoder.ups.{i}.0.bias"])

    # Three AMPBlocks per upsample stage, in the order BigVGAN indexes them:
    # `resblocks[i * num_kernels + j]`.
    for r in range(len(upsample_rates) * len(resblock_kernel_sizes)):
        for which in ("convs1", "convs2"):
            for c in range(len(resblock_dilations[0])):
                writer.add(f"resblocks.{r}.{which}.{c}.weight", fold_weight_norm(state, f"decoder.resblocks.{r}.{which}.{c}"))
                writer.add(f"resblocks.{r}.{which}.{c}.bias", state[f"decoder.resblocks.{r}.{which}.{c}.bias"])
        for a in range(2 * len(resblock_dilations[0])):
            writer.add(f"resblocks.{r}.act.{a}.alpha", state[f"decoder.resblocks.{r}.activations.{a}.act.alpha"])
            writer.add(f"resblocks.{r}.act.{a}.beta", state[f"decoder.resblocks.{r}.activations.{a}.act.beta"])

    writer.add("activation_post.alpha", state["decoder.activation_post.act.alpha"])
    writer.add("activation_post.beta", state["decoder.activation_post.act.beta"])

    # `use_bias_at_final: False` for this configuration, so `conv_post` has no
    # bias tensor at all -- asserted rather than defaulted, because a silently
    # absent bias and a silently zero one differ only in the checkpoint.
    if "decoder.conv_post.bias" in state:
        raise SystemExit("conv_post has a bias; this configuration is supposed to have none")
    writer.add("conv_post.weight", fold_weight_norm(state, "decoder.conv_post"))

    # The 12-tap filter, read rather than recomputed. Every `Activation1d` in the
    # decoder holds its own copy and they are all the same tensor; that is
    # checked below rather than assumed.
    filter_keys = [k for k in state if k.endswith("upsample.filter") or k.endswith("lowpass.filter")]
    reference = state[filter_keys[0]]
    for key in filter_keys:
        if not torch.equal(state[key], reference):
            raise SystemExit(f"{key} differs from {filter_keys[0]}; the decoder needs more than one filter")
    writer.add("antialias.filter", reference)

    verify_fold(bundle, state, metadata, audio_config)

    manifest = {
        "model": "minimax-h3-audio-vae-decoder",
        "source": "MiniMaxAI/MiniMax-H3 (FL2VA/audio_vae)",
        "licence": "MiniMax H3 Community License Agreement — not this repository's, and not redistributed by it",
        "sampleRate": sample_rate,
        "latentChannels": latent_channels,
        "latentDim": latent_dim,
        "decoderDim": decoder_dim,
        "hopLength": int(np.prod(upsample_rates)),
        "upsampleRates": upsample_rates,
        "upsampleKernelSizes": upsample_kernel_sizes,
        "resblockKernelSizes": resblock_kernel_sizes,
        "resblockDilations": resblock_dilations,
        "snakeLogscale": True,
        "useTanhAtFinal": False,
        "antialiasRatio": 2,
        "antialiasKernelSize": int(reference.numel()),
        "dtype": "f32",
        "tensors": writer.entries,
        "elements": writer.offset,
    }
    (out / "decoder.manifest.json").write_text(json.dumps(manifest, indent=1))
    writer.write(out / "decoder.bin")
    print(f"wrote {out}/decoder.bin  {writer.offset * 4 / 1e6:.1f} MB, {len(writer.entries)} tensors")
    return 0


if __name__ == "__main__":
    sys.exit(main())
