# MiniMax-H3's DiT

The **generator** half of MiniMax-H3: fifty identical transformer blocks over
5,376 channels, producing the latents `examples/h3-video` decodes. Issues
[#200](https://github.com/m96-chan/web-xpu-ops/issues/200) and
[#210](https://github.com/m96-chan/web-xpu-ops/issues/210).

**One block was where this started, and "the rest is a loop" was wrong.** Around
the fifty blocks sit five things that each return a well-formed tensor when they
are wrong, and `src/model.ts` is the forward that gets them right.

## The forward, and a fixture that is in this repository

| | worst element | RMS |
| --- | --- | --- |
| video velocity | **1.490e-7** | 4.341e-8 |
| audio velocity | 1.192e-7 | — |

on outputs of order 1.34, against an **f64** golden.

`tools/gen_forward_golden.py` instantiates diffusers'
`MiniMaxH3Transformer3DModel` at the geometry **upstream's own tester** uses —
hidden 24, two layers, heads 2x16 — with **random weights**. So `fixtures/` is
155 KB, is committed, runs in CI everywhere, and carries **none of MiniMax's
licence**: it is not their checkpoint.

Random weights lose nothing. What a forward gets wrong is *structural*, and
none of it depends on the weights being trained:

- **One packed sequence, not cross-attention.** The three modalities are
  projected separately and scattered into one buffer at caller-chosen rows.
  There is no cross-attention anywhere in H3.
- **The text stream is refined first**, by two *plain* pre-norm blocks — no
  AdaLN, no rotary — and a final norm. **Their Q and K must not be permuted for
  RoPE**, while the DiT blocks' must: two attention modules with identical
  parameter names and opposite weight preparation.
- **The AdaLN table is addressed per row.** `timestep_indices * 3 + token_tags`,
  so one forward serves rows at different noise levels.
- **`norm_out` modulates per row too, with `shift` first** — the reverse of the
  block's six-way chunk.
- **Both heads run over every row**; the modality's rows are selected after.

Two of the tester's geometry choices are load-bearing and deliberate:
`numHeads * headDim` (32) differs from `hiddenSize` (24), as in the released
checkpoint, and `2 * 3 * ropeFreqDim` (12) is smaller than `headDim` (16), so
the partial-rotary path is exercised rather than aliased away.

### Ten mutations, all caught

refiner skipped (1.6e-2), its final norm skipped (2.4e-4), the AdaLN table
addressed without the modality (2.7e-2) and without the timestep (1.2e-3),
`norm_out`'s shift and scale swapped (5.99e-1), `norm_out` modulated from row 0
for every token, the timestep embedding not flipped to cos-first (7.4e-2), the
timestep MLP without its activation (1.0e-1), the refiner's residual dropped
(8.3e-2), the video head's bias dropped (1.7e-1).

**Two of those are only caught because both outputs are compared.** Modulating
`norm_out` from row 0 leaves the *video* output bit-identical — every video row
is at timestep 0 — and moves the audio by 2.6e-2. Dropping the video head's
bias does the reverse. A test that checked one head would have passed either.

## On the GPU, at full size

`src/model-gpu.ts` runs the released 50-layer checkpoint. RTX 5090 / driver
610.57.04 / Dawn `webgpu@0.4.0`, int8 weights and f32 activations, 42 rows:

| | |
| --- | --- |
| resident weights | **20.08 GB** int8 |
| modulation tables (16 steps) | 0.58 GB |
| upload | 12.0 s |
| one step | 2078 ms — 2,063 dispatches, 6 ms in the queue, 460 ms in the final submit and readback, **1,611 ms recording** |

As with `examples/h3-video`, almost none of that is the GPU. The host-side
recording is the cost, and it has not been attacked.

### 20.08 GB, not 33.12

The checkpoint is **33.12 G parameters**, and `adaln_proj` is **13.01 G of them
— 39.3%**. It exists to project `temb`, a *two-row* tensor whose value depends
only on the timestep, into the `(timestep, modality)` modulation table. Two
fifths of a video model, for 520 MFLOP a step.

`tools/convert_dit.py` therefore **evaluates the tables at conversion time** and
never ships those weights; `norm_out.linear` goes the same way. The cost is that
a conversion runs only the step counts it was given, which the page has to say.

### What int8 costs, measured three ways

A port's accuracy cannot be read off a single number here, because the golden is
**bf16** and the output is a small difference of large intermediates — peak 276
after one block, peak 5.02 at the output, so relative error at the output is
amplified about fiftyfold.

At **one** block, video velocity, as a percentage of the golden's peak:

| | |
| --- | --- |
| bf16 activations against f32, **no quantisation at all** | **0.55%** |
| **this port** (int8 weights, f32 activations) against the bf16 golden | **0.88%** |
| int8 weights *in torch* (bf16 activations) against the same golden | 7.59% |

So the port sits just above the floor its golden can resolve, and torch's own
int8 round-trip is eight times further away. Quoting the port's number without
the floor beside it would claim a precision the comparison does not have.

At **fifty** blocks the port is **13.80%** of peak on video and 27.67% on audio;
torch's own int8 round-trip is **47.49%** and 42.72%. Most of the gap is
quantisation and precision rather than the port — but **the split is not
measured**, because an f32 fifty-block reference is 132 GB and the kernel kills
it. What int8 does to a *generated video* is unmeasured too, and needs the text
encoder.

### One bug, and how it presented

The rope call was written from memory instead of from
`ops/rope/wgsl/axes.wgsl`: bindings 1 and 2 swapped, and a four-field uniform
where the struct declares five. Every binding is `array<f32>` and a short
uniform reads whatever follows it, so there was **no validation error** — just
NaN, fifty blocks later.

It was found by running with `--layers 0`, which was finite, so the fault was in
a block; the refiner block shares everything with a DiT block *except* the
modulation and the rope, and it was finite too.

**The comparison said "worst 0.000e+0" on wholly-NaN output.**
`Math.abs(NaN - x) > worst` is false, so the worst-difference loop never
updates and prints a perfect score next to an RMS of NaN. `verify-forward.ts`
counts non-finite values now and refuses before it reports anything.

## The packed sequence, and the schedule

The transformer builds **neither**. `forward` takes the row order, the modality
tags, every row's noise level and the `(t, h, w)` rotary grid as *arguments*, so
each is a free choice — and **not one of them changes a shape**. A wrong layout
generates a video; just not the one the weights were trained for.

`src/layout.ts` and `src/scheduler.ts` are ported from
`diffusers.modular_pipelines.minimax_h3` and `MiniMaxH3Scheduler`, and their
fixtures are committed too — arithmetic on shapes and step counts carries no
weights.

**Six layout conventions**, none of them guessable:

- video rows are **frame-major, then row-major** within a frame;
- the spatial grid is **aspect-normalised** and scaled by 32 — on a *square*
  canvas the normalisation is the identity, which is exactly where a bug hides,
  so the fixture has a wide and a tall case as well;
- it is built with **`np.linspace(..., endpoint=False)`**, not `torch.linspace`;
- latent frames are spaced **`5/3 * (1, 4, 4, 4, 4)`** in rotary time, because
  the VAE's first latent covers one pixel frame and the rest cover four;
- **the media clock starts after the text**, so prompt length moves the video;
- audio rows are **channel-major**, carry **no height**, and are pinned to the
  two extremes of the width grid.

Eleven mutations, all caught.

**Two rounding details that are not pedantry.** `resolveCanvasSize` needs
Python's **banker's** rounding: the default 720p canvas asks for
`round(720 / 32) = round(22.5)`, which is 22 in Python and 23 in JavaScript, so
half-up silently generates at **736** pixels instead of 704. And the sigma grid
needed `torch.linspace` reproduced element for element — f32 step, second half
counted **down from the end**, each element one **fused** multiply-add. Naive
`1 - i / (n - 1)` disagrees at 4 of 50 points, one ulp each, which changes which
timesteps the transformer is conditioned on and would never look wrong.

**Four schedule conventions**: `t = 1 - sigma` with **`t = 1` clean**, the
terminal sigma getting **no** model evaluation, `step` recovering its sigma from
the *timestep* rather than the grid, and `eta = 0` despite the reference class
being named "euler ancestral". Both shipped shifts are covered — **12.0** video,
**3.0** audio.

There is **no classifier-free guidance**: one forward per step.

## One block, against the real checkpoint

## What is checked, and against what

The checkpoint ships **no code** for the DiT; `transformer/config.json` names a
diffusers class and nothing else. So `tools/gen_block_golden.py` fetches
`MiniMaxH3TransformerBlock` from diffusers' main branch, rewrites **only its
relative import lines** to absolute, and runs block 0 on a fixed input. The
arithmetic is upstream's.

| golden | worst element | RMS |
| --- | --- | --- |
| f32 | 5.859e-3 | 3.065e-4 |
| f64 (`--f64`) | **1.953e-3** | 4.353e-5 |

on a block output of order 350. A third of the gap is torch's own f32 rounding
over 5,376- and 14,336-term dot products; the rest is this port storing
intermediates in `Float32Array` between ops, which is what an f32 port is — the
feed-forward turns inputs of order one into outputs of order a thousand, where
one f32 ulp is already 1e-4. `--f64` exists to separate those two, not to ship.

`--temb-scale` defaults to **0.05** deliberately. `time_embedder` is a trained
module whose output is order one; a standard normal through `adaln_proj` gives
gates of order ten and a block output of order 1e4, where cancellation dominates
any comparison and hides what is being measured.

## The three things that differ from every other block here

Each returns a well-formed tensor of the right shape when it is got wrong.

**The modulation table is indexed per row.** `adaln_proj` projects one timestep
embedding into `6 * hidden * 3` values — three *modalities*: text, video, audio
— and every token picks its row with `timestep * 3 + tag`. A forward that is all
one modality still has to pick the right one of the three.

**SwiGLU is `hidden * silu(gate)` with `hidden` the first half.**
`examples/h3-video`'s decoder is the other way round. Two files in one model
with opposite conventions.

**A permutation has to cover everything that reads the channels it moves.** The
RoPE permutation goes into the Q and K projection weights, as it does for Anima
and for the visual VAE — but this block's QK-norm has *per-channel weights*, and
permuting the projection without permuting those scales the wrong channels. The
visual VAE has `qk_norm_affine: false` and no weights at all, so the identical
permutation is complete there and incomplete here. The symptom was an 8% error
in attention with the rope itself exact to 2.4e-7, which is why the rope was
checked in isolation before anything else was suspected.

## No new kernel

`matmul`, `rmsnorm`, `attention`, `rope`'s axes entry, `activation` and
`elementwise` cover the block. `h3RopePermutation` lives in `ops/rope` because
both of H3's models rotate the same way at different geometries — 48 of 64 in
the visual VAE, 96 of 128 here.

## Running it

The weights are **not in this repository and are not redistributed by it** (the
MiniMax H3 Community License Agreement is not this code's MIT — issue #190).

```bash
# upstream's file, with its relative imports made absolute
curl -o h3dit.py https://raw.githubusercontent.com/huggingface/diffusers/main/\
src/diffusers/models/transformers/transformer_minimax_h3.py
sed -i 's/^from \.\.\./from diffusers./; s/^from \.\./from diffusers.models./' h3dit.py

python examples/h3-dit/tools/gen_block_golden.py \
  --module ./h3dit.py \
  --shard ~/h3/transformer/diffusion_pytorch_model-00001-of-00014.safetensors \
  --out ~/h3-dit-fixtures

H3_DIT_DIR=~/h3-dit-fixtures npx vitest run examples/h3-dit/src
```

**A partial download of the shard is enough.** Block 0 ends at 1.47 GB of the
shard's 5.23, and `safetensors.safe_open` insists on a complete file — so the
generator reads by byte offset and widens bf16 by hand. Refetching 3.75 GB to
satisfy a length check would be bandwidth spent on nothing.

Without `H3_DIT_DIR` the comparison **skips with a message** rather than passing.

## Running the forward comparison

Nothing to download — the fixture is in the repository:

```bash
npx vitest run examples/h3-dit/src/model.test.ts
```

To regenerate it (needs `diffusers` from main, which is where the H3 classes
live):

```bash
python examples/h3-dit/tools/gen_forward_golden.py --out examples/h3-dit/fixtures
```

## What a forward at full size still needs

| | size | state |
| --- | --- | --- |
| the forward's arithmetic | — | **checked**, at the tester's geometry |
| this DiT's weights, 50 layers | 66 GB bf16, 8.8–21 GB quantised | **one block checked** |
| Qwen3-VL-32B text encoder | 13.1–27 GB | not implemented; `llm/` has no VL |
| visual VAE decoder | 2.43 GB int8 | **works** (`examples/h3-video`) |
| visual VAE encoder | — | **works**, CPU (`examples/h3-encoder`) |

The *shape* of the forward is settled. What is not: the 50-layer weights on a
GPU, and a text encoder. **Qwen3-VL-32B is not going into a browser**, so the
plan for #210 is to precompute text embeddings offline and ship them — which
means a page can only offer prompts somebody has already run. That is a
limitation, not a detail, and it will be written on the page.
