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

## Status: it runs in a headed browser

**22 frames at 256x256 generates.** Measured in a headed Chrome on an RTX 5090:
23–27 GB uploaded, the sampling loop runs, the frames land on the canvas.

**Headless Chrome cannot run it, and that turned out to be headless-specific.**
The adapter a headless Chrome 151 hands out reports
`maxComputeWorkgroupSizeX = 256` — measured, with
`maxComputeInvocationsPerWorkgroup` 256,
`maxComputeWorkgroupStorageSize` 32,768 and `maxBufferSize` 1 GB — while
`ops/matmul` declares `@compute @workgroup_size(512)`. A headed Chrome on the
same machine allows 512 and gets past it. The page reads the limits and refuses
in the first second rather than uploading 23 GB to find out; issue #211.

That resolves an ambiguity worth recording: `examples/zimage-web` dispatches the
same 512-thread kernel and is recorded as running in a browser, which is
consistent now — the two adapters really are different.

### The size that failed, and why

**576x320 hit a second ceiling**, after 24.49 GB had been uploaded:

```
batch is not valid: Dispatch workgroup count X (72240) exceeds
max compute workgroups per dimension (65535)
```

One thread per element and `ceil(n / 256)` workgroups runs out of grid at
**16,776,960 elements** — and 65,535 is what Dawn Node *and* Chrome both report,
neither raising it when asked. A 14,336-wide feed-forward reaches it at 1,170
rows. A 22-frame 256x256 clip is 538 packed rows and fits; **576x320 is 1,350
and does not**, which is the second size anybody picks.

Every flat dispatch is split on **row** boundaries now — rows, not elements,
because `ops/elementwise`'s rows entry recovers its column with `idx % D` and a
chunk starting mid-row reads the wrong scalar for all of it. Held to the model's
own output at a lowered ceiling: 223 dispatches unchunked and 231 chunked, and
**the same worst element to four digits**.

One dispatch cannot be split: `swapLeading`'s transpose writes strided, so a
slice of the input has no slice of the output to land in. At 56 heads of 128 it
runs out of grid at **2,340 tokens**, past every size this page offers, and it
refuses with that number rather than letting `batch is not valid` arrive from
inside a command buffer. Splitting it needs a second grid dimension in
`ops/permute`.

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
