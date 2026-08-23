# `examples/zimage-web` — Z-Image, in a browser

A prompt goes in a text box and a picture appears on a canvas. Tokenizer,
Qwen3-4B text encoder, the DiT and the VAE decoder, all on WebGPU, all composed
from this repository's kernels. Nothing leaves the machine.

## Running it

```bash
# once: convert the DiT (12.31 GB bf16 -> 6.17 GB q8)
/home/m96-chan/project/therdparty/musubi-tuner/.venv/bin/python \
    examples/zimage/tools/convert_dit.py --out ~/zimage-q8

npm run zimage:build
npm run zimage:serve -- --dit ~/zimage-q8
# open http://127.0.0.1:8788
```

The server mounts three directories under `/weights`: the converted DiT, the
text encoder from the Hugging Face cache, and the VAE fixture from this
repository. It answers `Range` requests, which is not optional — every weight
read is a byte range, and the client treats a `200` as an error rather than as
a slow success.

## What it is not

**Not a copy of the pipeline.** `main.ts` calls the same `qwen3EncodeGpu`,
`ditForwardGpu`, `decodeGpu` and `flowSchedule` that `examples/zimage`'s Node
verifiers hold to the model. Two things are swapped: kernels arrive as strings
from esbuild's `text` loader instead of `readFileSync`, and weights arrive over
`fetch` instead of file descriptors. A second copy of the forward pass would be
the thing that drifts away from the numbers those verifiers earned:

| piece | against the model |
| --- | --- |
| tokenizer | exact ids |
| text encoder | 7.993e-7 |
| DiT | 4.386e-6 |
| sampler | exact schedule |
| VAE decoder | 1.085e-5 |

**Not fast, and the reasons are measured.** See below.

## Where the time goes

Three things were wrong at first, and only the first one was the one that
looked obvious.

**The dispatch count was not the problem.** A forward made 3,610 dispatches at
7.1 ms each, with 95% of wall time inside `run()`. Batching attention across
its 30 heads — 57% of all dispatches — brought that to 1,638 and changed the
total by nothing: 7.1 ms per dispatch became 14.1 ms. That is what a
bandwidth-bound loop looks like when you take away half its calls.

**The bytes were.** `linearPacked` dequantised and transposed on the GPU
(`ops/dequant_transpose`) and then ran `matmul`. The dequantised operand is
four times the packed weight, and in this harness it came back to the CPU
between the two dispatches and went up again. `ops/matmul`'s `q8` entry reads
the packed codes directly:

| | dispatches | one forward, 70 tokens |
| --- | --- | --- |
| CPU reference (`dit.ts`) | — | 990.3s |
| GPU, per-head attention, dequant+matmul | 3,610 | 26.9s |
| GPU, batched attention, dequant+matmul | 1,638 | 24.2s |
| **GPU, batched attention, `matmulQ8`** | **957** | **6.2s** |

160x the CPU reference, with the model agreement unchanged at 4.386e-6 —
checked before the speed was, not after.

**The cache bound was wrong for HTTP.** The browser loader mirrored the Node
one and held two dozen tensors. That is right when a file and the OS page cache
are behind it and badly wrong over the wire: a forward touches every tensor
once, so a bound smaller than the model re-fetches everything every step —
6.17 GB per step, 49 GB for an eight-step generation, all the same bytes. The
DiT is now fetched once at start-up, with a progress bar, and held. After that
a generation does no network at all.

That costs 6.17 GB of JavaScript memory, which is real and is why
`maxCachedTensors` still exists for a machine that would rather pay the other
way.

## Still to do

Weights are uploaded to the GPU **per dispatch**. The network is quiet now, but
the PCIe bus is not: `harness/resident.ts` exists for weights that stay on the
device, and 6.17 GB of packed q8 fits in a 24 GB card. That is the next thing,
and it is named here rather than left for "WebGPU" to imply "fast".

The text encoder still transposes its weights on the CPU and uploads them
dense — it is bf16 in the checkpoint rather than q8 in a converted blob, so it
does not yet take the `matmulQ8` path the DiT does.
