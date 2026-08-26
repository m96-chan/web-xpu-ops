# MiniMax-H3 generating video, in a browser

A prompt in, frames on a canvas: fifty transformer blocks over 5,376 channels
write a latent and the visual VAE turns it into video. Issue
[#210](https://github.com/m96-chan/web-xpu-ops/issues/210).

Nothing here is new arithmetic. `examples/h3-dit/src/model-gpu.ts` is the DiT
and `examples/h3-video/src/decoder-gpu.ts` is the decoder; this is the wiring,
plus the two things a page needs that a script does not — weights from a folder
the visitor picked, and a device that survives between generations.

## What it does not do, and why

**The prompt list is fixed.** MiniMax-H3 conditions on Qwen3-VL-32B — **66.7 GB**
against the DiT's 20.08 and the decoder's 2.43 — so the text encoder runs
offline in `examples/h3-dit/tools/encode_prompt.py` and this page reads the
embeddings it wrote. A prompt nobody has encoded cannot be generated here. The
page says so rather than showing a disabled text field.

**Only the step counts the conversion was given.** `adaln_proj` is 39.3% of the
checkpoint and exists to project a two-row tensor, so `convert_dit.py` evaluates
the modulation tables instead of shipping the weights — and a step count with no
table has no modulation at all. The `<select>` is built from
`manifest.stepCounts`, so the page cannot offer a button that throws.

## Status: it uploads, and it does not yet run

Measured on this machine — RTX 5090, driver 610.57.04, **headless** Chrome
151.0.7922.71 over Vulkan, another browser already holding 11.4 GB of the card:

| | |
| --- | --- |
| DiT + tables + decoder uploaded | **23.10 GB in 20.9 s** |
| prompts and step counts populated | yes |
| first generation | **fails** |

The failure is specific:

```
pipeline is not valid: Entry-point uses workgroup_size(512, 1, 1)
that exceeds the maximum allowed (256, 256, 64)
```

`ops/matmul`'s `kernel` and `q8` entry points both declare
`@compute @workgroup_size(512)`, and **the adapter this headless Chrome hands
out reports `maxComputeWorkgroupSizeX = 256`** — measured, along with
`maxComputeInvocationsPerWorkgroup` 256, `maxComputeWorkgroupStorageSize`
32,768 and `maxBufferSize` 1 GB. `createBrowserResidentDevice` already asks for
the adapter's own ceiling on all four; the ceiling *is* 256 there. The same
error, escalating to a lost device, is what killed the GPU process on the first
attempt before the smaller conversion made it legible.

**Whether that is headless Chrome or every Chrome is not resolved here, and it
matters.** `examples/zimage-web` dispatches the same 512-thread kernel and the
CHANGELOG records it running end to end in a browser with numbers, which cannot
both be true of the same device — so either the limit is headless-specific, or
that record is about a run nobody repeated. A headed Chrome would settle it in
a minute; there is no X display reachable from where this was measured, so it
has not been settled.

`examples/h3-video-web`'s README ends with "the in-browser decode is
unmeasured", and it dispatches the same kernel. That is consistent with nobody
having run it.

## Running it

```bash
node examples/h3-dit-web/build.mjs
node examples/h3-dit-web/server.mjs --weights ~/h3-dit-web
```

Then open `http://localhost:8790/` and pick a folder. The folder needs, in one
place:

| from | files |
| --- | --- |
| `examples/h3-dit/tools/convert_dit.py` | `dit.manifest.json`, `dit.q8.bin`, `adaln.bin` |
| `examples/h3-video/tools/convert_decoder.py --quant q8` | `decoder.q8.manifest.json`, `decoder.q8.bin` |
| `examples/h3-dit/tools/encode_prompt.py` | `prompts.json` and one `.bin` per prompt |

**`?serve=/weights` skips the folder** and reads over HTTP from the same server.
Not a convenience: `showDirectoryPicker` needs a user gesture and a native
dialog, so a headless browser cannot otherwise reach the page's actual work —
which is how "the in-browser run is unmeasured" keeps being the honest ending.
Nothing is cached on that path, so every reload re-reads 23 GB; the gate stays
the default.

## What it needs

**About 23 GB of GPU memory** — 20.08 GB of int8 DiT weights, 0.58 GB of
precomputed modulation tables, 2.43 GB of decoder — plus scratch.

The page reports what it *uploaded*, not what the device took: WebGPU exposes no
way to ask, and on this machine `nvidia-smi` never moved while the page had
uploaded 23.10 GB. Reading that number as device memory would be reading a claim
as a measurement.

## The licence

**Powered by MiniMax H3.** The model is under the MiniMax H3 Community License
Agreement, not this page's MIT, and nothing here redistributes it: the agreement
permits redistribution only within an *Applicable Territory* that excludes the
European Union, the United Kingdom, the Republic of Korea and the United States
of America, and a public mirror cannot honour that. Converting your own copy is
the only path, which is why there is no default `?weights=` base. See issue
#190.
