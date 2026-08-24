# `examples/anima` — Anima-3.8B, composed from this repository's ops

A prompt goes in and a **latent** comes out, on the GPU, in 675 seconds at
832x1216. Not an image yet: Anima decodes with Wan 2.1's 3D causal VAE, which
this repository does not have. Issue #174 tracks it.

![the latent's RGB projection](fixtures/preview-832x1216.png)

That is not the model's picture. It is `latent_rgb_factors` — the linear 16-to-3
map ComfyUI shows as a live preview while sampling — at one eighth the
resolution, against a decoder with 194 tensors. It shows composition and colour
and nothing finer, and it is here because the honest alternative was a
hexdump.

## What is checked, and against what

Every number below comes from running ComfyUI's own model and comparing, never
from this code's own reasoning.

| stage | what it establishes | against the model |
| --- | --- | --- |
| one block (#169) | the algebra, at `dim=2048` | **1.63e-7** |
| the whole DiT (#170) | patchify, timesteps, three-axis RoPE, 52 blocks | **1.18e-5** |
| the DiT, resident on the GPU | the same, with weights that stay uploaded | **1.18e-5** |
| tokenizers | Qwen2 BPE and T5 unigram | **exact ids**, 24 cases each |
| conditioning | Qwen3-0.6B, then the `llm_adapter` | **9.78e-7** |
| the schedule and the stepper | `beta`, `res_multistep` | **2e-6** on a trajectory |
| **the whole sampling loop** | all of the above, joined | **3.27e-3** over 8 steps |

The last row is the one that mattered. Every row above it passed while the
pipeline produced a latent with no picture in it — see *The bug the parts could
not find*.

## The conventions that are not this repository's to choose

Anima is not Z-Image with different numbers, and the places it differs are all
silent when got wrong.

| | Z-Image | Anima |
| --- | --- | --- |
| encoder | Qwen3-4B, `hidden_states[-2]` | Qwen3-0.6B, `layer="last"` — **after** `model.norm` |
| conditioning | the encoder's output | encoder → a 6-block `llm_adapter` that ships **inside the DiT** |
| tokenizer | one | **two**: Qwen ids condition the encoder, T5 ids index the adapter's own embedding |
| timestep | `sigma * 1000` | `sigma * 1.0` — `multiplier` is 1.0, not the class default |
| schedule | linear | `beta`, which inverts a Beta(0.6, 0.6) CDF onto a 1000-entry table |
| stepper | Euler | `res_multistep`, second-order, with `eta = 0` |
| CFG | `cond + s*(cond - uncond)` | `uncond + s*(cond - uncond)` — one unit of scale apart |
| latent | one scale and shift | Wan 2.1's per-channel mean and standard deviation |
| RoPE | three axes, one base | three axes, **three bases** (t 10000, h/w 42870.9) |

The rope bases are the checkpoint's own extrapolation ratios — 4.0 for h and w
against 1.0 for t. `ops/rope`'s `axes` entry takes one base per dispatch, so the
head is dispatched three times with the other two axes' positions zeroed: an
angle of zero is the identity whatever base is in force.

## The bug the parts could not find

The pipeline ran end to end and the latent's **per-channel spatial standard
deviation** fell monotonically to 0.007 — a picture of nothing. The tensor's
overall standard deviation was 1.02 the whole way, which is why the failure was
invisible until the right quantity was measured: the latent was flat in space
and merely offset from channel to channel.

`verify-forward-gpu.ts` wrapped its weight source to put the self-attention q/k
projections and their QK-Norms into `ops/rope`'s channel order. `generate.ts`
passed the source straight through, so the DiT rotated channels that had not
been relabelled: **1.068e-1** against torch on identical inputs.

Finding it needed a golden that did not exist. `tools/gen_trajectory_golden.py`
runs the *whole* loop in torch — tokenizer, encoder, adapter, 52 blocks, CFG,
`res_multistep` — from the same noise, and records every intermediate.

The rule now lives in `withRopePermutation`, because a correctness rule that
lives in one caller is a rule the next caller does not have. It returns `null`
from `packedQ8` for anything it permutes: the resident path prefers packed codes
and would otherwise take the fast path straight past the permutation. Delegating
that one method instead of refusing it puts the bug back, measured at 3.831e-1.

Two more the diagnosis turned up, and one it nearly hid:

* CFG was one unit of scale out, so the workflow's 8 was silently a 9.
* The resident path could not dispatch the shipped resolution at all — 3,952
  tokens through an 8,192-wide activation is 126,464 workgroups against a limit
  of 65,535 (issue #112).
* The preview above was **black** after the latent was already correct: a
  generated array literal had a doubled comma, an elision in an array literal is
  legal TypeScript, and `Float32Array.from` turns the hole into NaN. `tsc` said
  nothing. A test that the 48 factors are 48 finite numbers says it now.

## Quantization

The DiT is q8 per-row absmax, 7.50 GB to 3.76 GB, costing rel-RMS **4.018e-2**
over 52 blocks measured independently in torch. The `llm_adapter` inside it
costs **2.845e-2**.

**The encoder is not quantized**, and that is measured rather than stylistic:
q8 on Qwen3-0.6B's 196 layer matrices moves its output by rel-RMS **0.223**,
against 0.0019 for its embedding table alone. A 0.6B has one absmax scale per
1024 numbers to spend and its outlier channels do not fit in it. Saving 0.6 GB
is not worth conditioning on different words, so it is read dense from its own
bf16 file.

## Speed

Measured, not estimated. Conditions first, because a number without them cannot
be reproduced or compared.

| | |
| --- | --- |
| device | NVIDIA GeForce RTX 5090, driver 610.57.04 |
| backend | Dawn via `webgpu@0.4.0`, Node v25.6.1 |
| dtype | q8 weights, f32 activations |
| image | 832x1216 — a 104x152 latent, 3,952 tokens |
| settings | 40 steps, CFG 8, the released workflow's own |

| | |
| --- | --- |
| conditioning (both prompts, CPU) | 36 s |
| sampling, 80 model calls | **675 s** — 8.44 s each |
| one forward at 256x256 (256 tokens) | 0.75 s |
| first forward, uploading 4.94 GB of weights | 6.6 s |
| the same forward, weights resident | 0.2 s |
| the CPU reference, one forward at 64 tokens | 814 s |

Roofline: **not measured.** The DiT is 3,290 dispatches over 55 submits, and
what fraction of the device's achievable bandwidth that reaches has not been
established.

## Running it

```bash
npx tsx examples/anima/src/generate.ts \
  --encoder ~/anima-src/qwen_3_06b_base.safetensors --dit ~/anima-q8 \
  --prompt "1girl, silver hair, red eyes" --steps 40 --cfg 8 \
  --width 832 --height 1216
```

Weights:

| file | from |
| --- | --- |
| `Anima-3.8B.safetensors` | `lylogummy/Anima-3.8B`, then `tools/convert_dit.py` |
| `qwen_3_06b_base.safetensors` | `circlestone-labs/Anima`, `split_files/text_encoders` |
| `qwen_image_vae.safetensors` | the same repo — **not used yet**, see #174 |

Verifying, each against its own golden:

```bash
npx tsx examples/anima/src/verify-forward.ts     --dit ~/anima-q8   # the DiT, on the CPU
npx tsx examples/anima/src/verify-forward-gpu.ts --dit ~/anima-q8   # the DiT, resident
npx tsx examples/anima/src/verify-encoder.ts     --encoder ... --dit ~/anima-q8
npx tsx examples/anima/src/verify-trajectory.ts  --dit ~/anima-q8   # the whole loop
npm test -- examples/anima                                          # tokenizers and sampler
```

## What `expanded` is, and why #173 is not on this path

The released workflow offers two conditionings. `prompt.py:219` builds the
second from the first:

```python
native_context   = native_adapter(source, target_ids)
expanded_context = native_context + strength * (expanded - native_context)
```

and its own tooltip calls strength 0.0 "native Anima". Everything above is the
`native` path — not a degraded mode standing in until something else exists, but
the base the other interpolates away from. `expanded` runs a Qwen3.5-4B, whose
selective state-space scan is issue #173. Nothing here needs it.

## Still to come

| | |
| --- | --- |
| #174 | the Wan 2.1 VAE decoder — the latent above becomes an image |
| #170 stage 6 | a browser page, as `examples/zimage-web` is for Z-Image |
| #173 | the `expanded` conditioning |
