"""Where Z-Image's weights live, resolved once so nothing downloads twice.

The DiT is 12 GB and the text encoder 8 GB. Fetching either more than once is
the kind of waste that is invisible until it has happened several times, so
every generator in this repository resolves its paths through here.

Resolution order, first hit wins:

1. `--model-dir` / `ZIMAGE_MODEL_DIR` — a directory laid out like the
   repository (`transformer/`, `vae/`, `text_encoder/`). Nothing is downloaded:
   if a file is missing the caller is told which one, rather than silently
   pulling gigabytes.
2. The Hugging Face cache (`HF_HOME`, else `~/.cache/huggingface`). Already
   holds the VAE. `snapshot_download` re-uses whatever is there and fetches
   only what is not, so pointing at the cache twice costs nothing the second
   time.

The distinction matters because the two answer different questions. Option 1 is
"I have the weights, use them"; option 2 is "get them if needed". Silently
downloading 12 GB because a path was typed wrong is the failure worth designing
against.
"""
from __future__ import annotations

import os
from pathlib import Path

REPO = "Tongyi-MAI/Z-Image"

# What each component needs, as `snapshot_download` patterns.
COMPONENTS = {
    "vae": ["vae/*.safetensors", "vae/*.json"],
    "transformer": ["transformer/*.safetensors", "transformer/*.json"],
    "text_encoder": ["text_encoder/*.safetensors", "text_encoder/*.json", "tokenizer/*"],
}


def model_dir(explicit: str | Path | None = None) -> Path | None:
    """The local directory to read from, or None to use the HF cache."""
    chosen = explicit or os.environ.get("ZIMAGE_MODEL_DIR")
    if not chosen:
        return None
    path = Path(chosen).expanduser()
    if not path.is_dir():
        raise SystemExit(
            f"ZIMAGE_MODEL_DIR / --model-dir points at {path}, which is not a directory.\n"
            f"Leave it unset to use the Hugging Face cache instead."
        )
    return path


def resolve(component: str, explicit: str | Path | None = None) -> str:
    """A path `from_pretrained` can take for one component.

    With a local directory, the component's subdirectory is returned and its
    absence is an error naming the component — not a download.
    """
    if component not in COMPONENTS:
        raise ValueError(f"unknown component {component!r}; expected one of {sorted(COMPONENTS)}")

    local = model_dir(explicit)
    if local is not None:
        sub = local / component
        if not sub.is_dir():
            raise SystemExit(
                f"{sub} does not exist.\n"
                f"The directory given for --model-dir should be laid out like {REPO}:\n"
                f"  <dir>/vae/  <dir>/transformer/  <dir>/text_encoder/\n"
                f"Nothing was downloaded, because a wrong path should not cost 12 GB."
            )
        return str(sub)

    from huggingface_hub import snapshot_download

    root = snapshot_download(REPO, allow_patterns=COMPONENTS[component], max_workers=4)
    return str(Path(root) / component)


def add_argument(parser) -> None:  # noqa: ANN001 - argparse.ArgumentParser
    """The flag every generator shares, so the spelling cannot drift."""
    parser.add_argument(
        "--model-dir",
        default=None,
        help="directory holding vae/ transformer/ text_encoder/ (else ZIMAGE_MODEL_DIR, "
             "else the Hugging Face cache, downloading only what is missing)",
    )
