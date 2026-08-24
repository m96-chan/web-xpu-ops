# `examples/zimage` — Z-Image, composed from this repository's ops

**Nothing here generates an image yet, and nothing here runs on a GPU.** What
it establishes is narrower and comes first: that the ops compose into Z-Image's
actual computation, checked against the model's own implementation at every
step. Issue #166 tracks the rest.

| stage | what it establishes | against the model |
| --- | --- | --- |
| one block, 64 wide, random weights (#163) | the algebra | **5.96e-8** |
| one block, shipped width and weights (#166) | the composition, at `dim=3840` | **8.72e-8** |
| the whole DiT forward (#166) | patchify, timesteps, refiners, 30 layers, final layer | **1.93e-6** |
| the text encoder (#166) | Qwen3-4B to `hidden_states[-2]` | **6.20e-7** |

Still to come: the sampler and wiring (stage 4) and a browser demo (stage 5).

Every row above is measured against a golden produced by running the model
itself, never against this code's own reasoning. Where a number below has not
been measured, it says so.

## One block, against the model's algebra

`src/block.ts` is a composition — every line is an existing op, and nothing new
is introduced. It matches Z-Image's own block to within one f32 ulp:

| | |
| --- | --- |
| worst absolute difference | **5.96e-8** |
| worst relative difference | **1.05e-6** |
| outputs compared | 384 |

Measured with the tolerance forced to zero, so it is the real number rather
than the bound. It comes from summation order in f32, not from different
arithmetic.

That covers, in one check: RMSNorm's eps placement, QK-Norm's axis, three-axis
RoPE's channel split and its adjacent-pair convention, MHA's scale, SwiGLU's
operand order, the `1.0 +` on both scales, `tanh` on both gates, and both
residuals. A mistake in any of them moves the output far more than the bound —
dropping a single `1.0 +` is measured at 3.43e-2 absolute, six orders of
magnitude above it.

Two facts about the model were read from its source rather than assumed
(rule 2), because getting either wrong produces a plausible wrong answer:

- **RoPE pairs adjacent channels** (`view_as_complex` on a `(..., -1, 2)`
  reshape), the same convention `ops/rope` already uses. HF Llama's
  `rotate_half` pairs `i` with `i + D/2` and would have needed the permutation
  `llm/weights.ts` carries for the LLM path. Z-Image needs none.
- **Attention is MHA, not GQA.** The shipped config is `n_heads=30,
  n_kv_heads=30`. The code carries a separate `n_kv_heads` but never uses a
  smaller one.

## What it does not establish

- **Not the shipped width.** The golden uses axes `[8, 12, 12]` — `ROPE_AXES_DIMS`
  scaled down 4× — and 2 heads, against the model's `[32, 48, 48]` and 30. That
  keeps the file at 262 KB instead of ~4 MB, almost all of which would have been
  width this check does not read. `ops/rope`'s own `ropeAxes` tests own the real
  widths; this owns the composition.
- **Not the whole model.** One block. Z-Image stacks the same block, so the rest
  is weights and repetition — but the text encoder (Qwen3, blocked on the BPE
  tokenizer, #153) and the VAE decoder are separate work.
- **Not a GPU run, and not speed.** Both are #148's job. The composition here is
  CPU reference ops throughout, which is what makes it a statement about
  correctness only.
- **Not the real weights** — *no longer true, see below.* The 64-wide golden
  above still uses a fixed random seed; the shipped weights are covered by a
  second check.

## The same block, at the shipped width, with the shipped weights

Issue #166's first stage. `src/weights.ts` reads what `tools/convert_dit.py`
writes, and `src/verify-real-block.ts` runs `layers.0` through `block.ts` at
`dim=3840`, `head_dim=128`, axes `[32, 48, 48]` — the real thing, not a
scaled-down stand-in — against a golden that stores only inputs and outputs so
it stays at 374 KB.

Measured, on this machine:

| | |
| --- | --- |
| port vs the model, same quantized weights | **8.72e-8** relative RMS, cos 1.000000000 |
| port vs the model at full precision | 1.778e-2 relative RMS, cos 0.999842 |
| of which quantization alone, measured in torch | 1.778e-2 relative RMS, cos 0.999842 |

The second and third lines agreeing to four figures is the point: the entire
gap to the full-precision model is the format, and the port contributes 8.72e-8
on top of it. Comparing against one number would have left those two
indistinguishable.

### What 4-bit costs, and where

The DiT is 12.31 GB of bf16. In q4-g128 (#137's format) it is 3.34 GB, and the
question #137 was reopened to answer is what that costs. On `layers.0`, with
each weight quantized alone and the rest left dense:

| weight | relative RMS on the block's output |
| --- | --- |
| `adaLN_modulation.0.weight` | **0.0478** |
| `attention.to_q.weight` | 0.0121 |
| `attention.to_k.weight` | 0.0090 |
| `feed_forward.w2.weight` | 0.0062 |
| `feed_forward.w1/w3.weight` | 0.0054 |
| `attention.to_v.weight` | 0.0042 |
| `attention.to_out.0.weight` | 0.0037 |

adaLN dominates, and not by a little: quantizing everything costs 0.0519, of
which it is 0.0478. It is the weight that produces the scales and gates
multiplying the whole residual stream, so its error is multiplicative across
all 3840 channels rather than additive.

So the converter keeps **adaLN at q8 and everything else at q4**. That costs
2.5 MB per layer — 96.1 to 98.6, +2.6% — and takes the block from 0.0519 to
0.0178. Smaller groups do not substitute: `q4-g32` costs 0.25 more bits on
every weight and only reaches 0.0398. Whole-model q8 would reach 0.0034 at
203.5 MB per layer, which is 6.5 GB of browser.

None of that is a claim about image quality — it is one layer, on random
activations, and 30 layers compound. What it rules out is choosing the format
by feel.

### Running it

```bash
/home/m96-chan/project/therdparty/musubi-tuner/.venv/bin/python \
    examples/zimage/tools/convert_dit.py --out ~/zimage-q4
npx tsx examples/zimage/src/verify-real-block.ts --dit ~/zimage-q4
```

Not part of `npm test`: it needs the 3.34 GB blob, which CI does not have and
should not fetch. `block.test.ts` is the part that runs everywhere.

## The whole DiT forward

`src/dit.ts` is everything around the blocks, and that is where a port goes
wrong — a stack of correct blocks says nothing about any of it.
`src/verify-forward.ts` runs it against the model's own forward, hooked at six
points and compared **in order**, stopping at the first disagreement: a
34-layer stack that reports only its last tensor tells you that something is
wrong and nothing about where.

| checkpoint | vs the model, same quantized weights | vs the model at full precision |
| --- | --- | --- |
| `adalnInput` | 1.436e-6 | 1.436e-6 |
| `afterNoiseRefiner` | 2.228e-6 | 3.796e-3 |
| `afterContextRefiner` | 2.552e-7 | 3.574e-2 |
| `afterLayer0` | 1.777e-6 | 2.031e-2 |
| `afterLayers` (30 layers) | 1.852e-6 | 8.203e-2 |
| **`output`** | **1.931e-6** | 2.113e-1 |

The right-hand column is the format's cost compounding, not the port drifting;
torch measures the same 2.113e-1 for the quantization alone. (Those two columns
are the q4-with-adaLN-at-q8 layout, which was the default when they were taken.
The q8 default's own end-to-end figure is being remeasured.)

### Six conventions that were read, not guessed

Each of these produces a plausible wrong answer, and none is visible in the
shape of the output.

- **The timestep embedding puts `cos` first.** `cat([cos, sin])`, the opposite
  of what diffusers and most DiT code do. Getting it backwards denoises for the
  wrong amount of time.
- **`t` is scaled by 1000** before it is embedded.
- **Image positions start after the caption**, `(cap_seq_len + 1 + f, h, w)`,
  while caption positions are `(i + 1, 0, 0)`.
- **The final layer is a LayerNorm**, `eps=1e-6` and no affine, scaled by
  `1 + adaLN(SiLU(c))` — not the RMSNorm used everywhere else.
- **`context_refiner`'s blocks have no modulation**, and genuinely lack the
  adaLN tensors: 13 per layer against the main stack's 15.
- **The caption's padding is handled twice.** The row is zeroed and
  `cap_pad_token` written into it, *and* it is dropped from attention.

That last one cost two attempts and is worth the detail. Ignoring the padding
entirely measured **1.21** relative RMS — the learned pad vector is exactly
what the softmax spends weight on if nothing stops it. Adding an additive
`-Infinity` key mask measured **6.6e-1**, which is right about the keys and
wrong about the query rows: `modules/attention.py` does not mask at all when
every sequence in the batch is the same length, which for one image is always.
It **slices** `q`, `k` and `v` to the valid length, discards the mask, and pads
the output back with **zeros**. A padded row's attention output is therefore
exactly zero and the residual leaves that row untouched. Only trying both
separated the two mistakes.

## The text encoder

`src/text-encoder.ts` is Qwen3-4B, run to where Z-Image stops it.

`hidden_states[-2]` is **the output of layer 34** of 0..35, taken before
`model.norm` — measured by hooking every decoder layer and matching, because
this version of `transformers` does not collect the hidden states in the loop
where it could be read. Layer 35, the final norm and the LM head are never
reached. Off by one gives embeddings of the right shape and the wrong content.

Verified at **6.197e-7** relative RMS on the shipped weights, at full precision:
no quantization is in the picture yet, deliberately, so that a porting mistake
cannot be confused with a format's cost. Quantizing the encoder is a separate
step with its own number.

**QK-Norm's weight is permuted with the projection it follows.** HF pairs RoPE
channels half a head apart where `ops/rope` pairs them adjacently, and
`llm/weights.ts`'s `permuteRopeChannels` already handles `q_proj`/`k_proj`. The
part that is easy to miss: `q_norm`/`k_norm` are `[headDim]` and multiply
channel by channel *inside* a head, exactly where the permutation reorders
things. RMSNorm's normalisation is permutation-invariant; its scale is not.
Dropping that permutation measures **1.002e-1** against 6.197e-7.

**The 512-token padding is not reproduced.** Attention is causal and the
padding is on the right, so no real token attends to one. That is an argument
rather than a measurement, so the golden is generated *padded* and compared
against an *unpadded* run — the rows agree, which is what makes skipping it
safe rather than merely plausible.

### Why not `llm/`'s engines

Qwen3's decoder layer is Llama's plus QK-Norm, and **none of the three engines
has it** (`q_norm`/`k_norm` appear nowhere in `llm/`). Closing that gap has to
happen in the CPU engine and both GPU ones together, which is its own piece of
work; stage 3 needs a correct encoder before it needs a fast one. This golden
is what that work can be checked against.

## A prompt goes in and a picture comes out

`src/generate.ts` for the command line, `examples/zimage-web` for a page you can
type into. Both run the same verified pieces.

![1024x1024, Z-Image Base, 30 steps, CFG 4.0](fixtures/generated-1024-base.png)

```
A clean anime illustration of a young woman with long silver hair and violet
eyes, wearing a white sailor uniform, standing in a sunlit classroom by the
window. Crisp black lineart, flat cel shading, bright saturated colors, clear
outlines.
```

Measured on this machine — an RTX 5090, driver 610.57.04, `webgpu@0.4.0` (Dawn),
node v25.6.1, Linux 7.1.5-arch1-2 — with the DiT converted to q8:

| | |
| --- | --- |
| resolution | 1024 x 1024 (4,111 tokens) |
| variant | Z-Image **Base**, 30 steps, CFG 4.0 |
| tokenize | 0.1 s |
| text encoder | 30.5 s (GPU) |
| denoise | 710.3 s — **23.7 s a step**, two forwards each because of CFG |
| VAE decode | 18.5 s |
| **total** | **759.5 s** |
| VRAM | 6.17 GB weights + 5.25 GB activations |

At 512 x 512 (1,039 tokens) the same settings cost 1.9 s a step and 1.32 GB of
activations. The DiT forward itself is **0.17 s** once the weights are resident
(70 tokens); the rest is CFG doubling the passes and attention growing with the
square of the sequence.

### The two checkpoints are not interchangeable

`Tongyi-MAI/Z-Image` is **Base**: undistilled, 28-50 steps, CFG 3.0-5.0.
`Z-Image-Turbo` is distilled for 8 steps and does not use CFG.
`zimage_config.py`'s `DEFAULT_INFERENCE_STEPS = 8` and `DEFAULT_GUIDANCE_SCALE
= 0.0` are **Turbo's** numbers, and this example ran Base on them for a while.
Same prompt, same seed, same everything else:

| settings | result |
| --- | --- |
| 8 steps, no CFG | shattered-glass artefacts; every style word in the prompt ignored |
| 25 steps, no CFG | much better; artefacts left in the background |
| **30 steps, CFG 4.0** | the image above |

What made that hard to diagnose is that every part measured correct while the
picture was wrong — DiT 5.560e-6, VAE 1.629e-5, the schedule exact. It was
settled by decoding our own latent with **torch's** VAE and getting the same
artefacts, which cleared the VAE and pointed at the settings rather than the
port. `--variant base|turbo` now carries each checkpoint's own numbers, and the
browser demo fills them in when you pick a model.

## Regenerating the golden

The generator imports `ZImageTransformerBlock` from musubi-tuner's copy of the
Tongyi-MAI implementation and runs it, rather than reimplementing it — a
transcription mistake in a hand-written generator would produce a golden that
agrees with a wrong port, which is the failure this arrangement rules out.

```bash
/home/m96-chan/project/therdparty/musubi-tuner/.venv/bin/python \
    examples/zimage/tools/gen_block_golden.py
```

That interpreter is required: the model file imports `accelerate`, which a bare
`python3` on this machine does not have.

## Op coverage

Every op the block needed, and where:

| op | used for |
| --- | --- |
| `rmsnorm` | `attention_norm1/2`, `ffn_norm1/2`, and QK-Norm (same op, rows counted per head) |
| `matmul` | every `nn.Linear` — QKV, output projection, FFN, adaLN |
| `elementwiseRows` | `* scale_msa`, `* gate_msa`, `* scale_mlp`, `* gate_mlp` (#150) |
| `elementwise` | both residual adds, and SwiGLU's `silu(w1) * w3` |
| `activation` | `silu` in the FFN, `tanh` on both gates |
| `ropeAxes` | three-axis RoPE on Q and K (#151) |
| `attention` | the attention itself, non-causal |

Nothing was missing. Three things are still done on the CPU in `block.ts` and
are marked there rather than hidden, because a CPU step here is a candidate op
there: the weight transpose for `nn.Linear`'s `[out, in]` layout, the head
split/merge around `ops/attention`, and adaLN's bias add and 4-way chunk.
