# `spike/anima` — what Anima-3.8B needs that this library does not have

Issue #169. **This is a spike, not a port** — the port is #170. What follows is
what the checkpoints and the reference implementation say, with line numbers, so
that #170 starts from facts rather than from a reading of the model card.

Nothing here has been run against the model. Every claim is either a shape read
out of a checkpoint header (reproduce with `tools/read_checkpoint.py`) or a
quotation from the reference implementation. Where neither exists, it says
**unmeasured**.

## What #170 measured, and what it overturned

Three of the claims below turned out to be wrong, and they are corrected here
rather than edited away: a spike is a record of what was believed from reading,
and the value of keeping it is being able to see where reading was not enough.

| claim below | what running it says |
| --- | --- |
| RoPE splits `head_dim` **44 / 44 / 40** | **44 / 42 / 42**. The formula is right; `dim_h = dim // 6 * 2` is 42 for head_dim 128, not 44. Arithmetic done by eye. |
| whether `ops/rope` can express the DiT's 2x2 matrices is **unresolved**, and "the one place a new kernel looks plausible" | **It can, and no new kernel was written.** The matrices satisfy `a == d`, `b == -c`, determinant 1 — the same rotation `ops/rope` applies. What the spike missed is that the three axes do **not share a base**: h and w carry an extrapolation ratio of 4.0 against t's 1.0, so their bases are 42870.9 against 10000. `ops/rope`'s `axes` entry takes one base per dispatch, so the head is dispatched three times with the other two axes' positions zeroed. |
| `text-encoder.ts` **does not transfer**; Qwen3.5's hybrid `SSMBlock` needs a new kernel (#173) | It transfers. The released workflow's own node builds its second conditioning as `native + strength * (expanded - native)`, and calls strength 0.0 "native Anima" — so `native` is **Qwen3-0.6B**, which `text-encoder.ts` runs with a config change and one optional final norm. #173's scan is what the `expanded` residual needs, and nothing on the path to an image does. |

A fourth is narrower: **F8_E4M3 reading is not needed** either. The encoder the
native path loads is `qwen_3_06b_base.safetensors`, which is BF16 throughout.

The port is `examples/anima`, and its README carries the numbers.

## Reproducing the shapes

```bash
python3 spike/anima/tools/read_checkpoint.py --tensors
```

The repository ships no `config.json`. The tool reads each safetensors header
over an HTTP `Range` request — a few hundred kilobytes — so the architecture can
be established without downloading 12 GB.

```
dit           1168 tensors, 3.75B params, BF16,     stored 7.50 GB
adapter         49 tensors, 0.04B params, BF16,     stored 0.09 GB
text_encoder   426 tensors, 4.14B params, BF16+F8,  stored 4.78 GB
```

**The text encoder is already partly `F8_E4M3`.** `examples/zimage`'s
`SafetensorsFile` implements F32 and BF16 and **refuses anything else**, which
is the correct behaviour and also means it cannot read this file as it stands.
That is a concrete, small gap: either add F8 to the reader, or convert.

### Size, if it followed `examples/zimage`'s q8 route

| | stored now | q8 estimate |
| --- | --- | --- |
| DiT | 7.50 GB | **3.75 GB** |
| adapter | 0.09 GB | 0.04 GB |
| text encoder | 4.78 GB | **4.15 GB** |
| total | 12.4 GB | **~7.9 GB** |

Against Z-Image's 6.17 GB DiT this is smaller, and the text encoder is
comparable. Browser-feasible on the same terms `examples/zimage-web` already
runs on — one disk-cached download, then no network.

**Unmeasured**: what q8 costs Anima's output. Z-Image's number (2.8% relative
RMS over 34 layers, and 21% for q4) does not transfer — Anima has 52 blocks and
a different block, and #166's own lesson was that per-layer error compounds in a
way a single layer's measurement does not predict.

## Architecture, from the reference implementation

Two files define it. Neither is in the model's repository:

- `comfy/ldm/anima/model.py` — Anima's own additions (214 lines)
- `comfy/ldm/cosmos/predict2.py` — `MiniTrainDIT`, which Anima subclasses
- `comfy/ldm/cosmos/position_embedding.py` — the 3-axis RoPE

The ComfyUI node pack and the Forge port both treat the DiT as already
supported upstream, which is why neither contains it.

### The DiT block is not Z-Image's

| | Z-Image (#166) | Anima |
| --- | --- | --- |
| text conditioning | image and caption in one sequence, self-attention | **cross-attention**, separate context |
| blocks | 30 + 2 refiners, dim 3840 | **52**, dim 2048, head_dim 128 |
| MLP | SwiGLU (`w1`, `w2`, `w3`) | **`Linear → GELU → Linear`, no gate** |
| adaLN | one per block, 4 chunks | **three per block** (self / cross / mlp), 3 chunks each |
| adaLN chunks | `scale, gate, scale, gate` | **`shift, scale, gate`** |
| gate | `tanh(gate)` | **no tanh** |
| shift term | none | **present** |

`predict2.py:487-504` for the chunk order, `:520-521` for the arithmetic:

```python
shift_self_attn, scale_self_attn, gate_self_attn = self.adaln_modulation_self_attn(emb).chunk(3, dim=-1)
...
def _fn(_x, _norm_layer, _scale, _shift):
    return _norm_layer(_x) * (1 + _scale) + _shift
```

then `x = torch.addcmul(x, gate, result)` (`:534`). So `1 + scale` survives from
Z-Image, `tanh` does not, and there is a shift Z-Image has no equivalent of.

### adaLN is a LoRA

`predict2.py:451-465`: `SiLU → Linear(2048, 256) → Linear(256, 6144)`. That is
why the header shows `[6144, 256]` rather than `[6144, 2048]`, and why there are
104 `adaln_modulation_*` tensors for 52 blocks — two matrices each.

### `x_embedder.proj` is `[2048, 68]`, and 68 is not 16 × 2 × 2

`predict2.py:703`:

```python
in_channels = in_channels + 1 if concat_padding_mask else in_channels
```

`concat_padding_mask` defaults to `True` (`:638`). So 16 latent channels **plus
one padding-mask channel** is 17, and 17 × 2 × 2 = 68. A port has to build that
mask channel and concatenate it before patchifying (`:806`).

Worth flagging for #170: getting the channel *order* wrong produces a tensor of
the right shape and different contents — the same failure mode that
`examples/zimage`'s patchify comment already warns about.

### RoPE: three axes, but not Z-Image's three axes

`position_embedding.py:83-93` splits `head_dim` as `dim_h = dim // 6 * 2`,
`dim_w = dim_h`, `dim_t = dim - 2 * dim_h` — for head_dim 128 that is
~~**44 / 44 / 40**~~ **44 / 42 / 42** (`128 // 6 * 2` is 42, not 44; corrected
above), against Z-Image's `[32, 48, 48]`. Both are three-axis;
`ops/rope`'s `axes` entry takes the split as a parameter, so this is
configuration rather than a missing kernel.

Two things are genuinely different and need care in #170:

- **The pairing.** `model.py:7-11`'s `rotate_half` pairs a channel with the one
  half a head away — HF's convention, not `ops/rope`'s adjacent pairs. This is
  the case `llm/weights.ts`'s `permuteRopeChannels` already handles, and #166
  used it for Qwen3. Note that it is used by the **LLM adapter**; the DiT's own
  blocks take `rope_emb` from `VideoRopePosition3DEmb`, which builds explicit
  2×2 rotation matrices (`position_embedding.py:151-163`) rather than cos/sin
  pairs. ~~**Whether `ops/rope` can express that form is unresolved** — it is the
  one place in this spike where a new kernel looks plausible.~~ **It can** — see
  the corrections above; the matrices are the same rotation, and no new kernel
  was written. The thing to carry into #170 is the *per-axis base*, which this
  spike did not notice.
- **The three buffers in the checkpoint** — `dim_spatial_range [21]`,
  `dim_temporal_range [22]`, `seq [256]` — are not learned parameters. They are
  `register_buffer(..., persistent=False)` values (`:85-93`) that happen to have
  been saved. A port computes them; it does not need to read them.

### The LLM adapter lives inside the DiT checkpoint

`model.py:143-191`. Six blocks at width 1024, head_dim 64, each with
self-attention, cross-attention and an MLP — and **no adaLN**. Its
`embed` is `Embedding(32128, 1024)`.

32128 is T5's vocabulary, and `Anima.forward` takes `t5xxl_ids` /
`t5xxl_weights` (`model.py:210-214`). So the adapter's *queries* come from T5
token ids and its *context* is the Qwen3.5 features. The output is padded to 512
tokens (`:206`).

**Resolved.** The prompt is tokenized **twice**. `text_encoder/tokenizer.py:82`
holds a `T5XXLTokenizer` alongside the Qwen3.5 one, and `:93` tokenizes the same
text with both; `text_encoder/clip.py:79-86` puts the T5 ids and their per-token
weights into the conditioning dict. So a port needs **two tokenizers**, and the
T5 one is not the byte-level BPE `llm/tokenizer-bpe.ts` implements — T5 is
SentencePiece **unigram**, which `llm/tokenizer.ts` is. Both already exist here;
what is new is that one prompt feeds both.

`t5xxl_weights` multiplies the adapter's output (`model.py:200-202`) and carries
ComfyUI's per-token prompt weighting.

### The separate adapter file

49 tensors: `semantic_attentions` × 36, `query_norms` × 6, `source_norms` × 6,
and one `layer_mix_logits`.

`progressive_cross_adapter.py:58-63` shapes the last as
`[len(adapter.blocks), len(layer_indices)]` = `[6, 4]`, and `:162` uses it:

```python
mix = self.layer_mix_logits.float().softmax(dim=-1).to(dtype=x.dtype)
```

So **each of the six adapter blocks gets its own softmax-weighted mixture of the
four Qwen3.5 layers**, rather than one mixture shared by all. `ops/softmax`
covers it; the shape is 6x4 and the cost is nothing.

## Op gap

| what Anima needs | in this library | notes |
| --- | --- | --- |
| RMSNorm | `ops/rmsnorm` | `eps=1e-6` throughout (`model.py:53`) |
| LayerNorm | `ops/layernorm` | the `layer_norm=True` branch; Anima's blocks use RMSNorm |
| Linear / GEMM | `ops/matmul`, `matmulQ8` | no biases in the DiT's attention (`model.py:52-61`) |
| self-attention | `ops/attention` | |
| **cross-attention** | `ops/attention` | takes `L != S` and a separate K/V source already; **untested at this shape** |
| GELU | `ops/activation` | has both `gelu` (exact) and `gelu_tanh`. `nn.GELU()` with no argument is **exact** (`model.py:121`) |
| SiLU | `ops/activation` | inside every adaLN |
| adaLN modulation | `ops/elementwise` `rows` | three per block instead of one; arithmetic differs (shift, and no tanh) |
| 3-axis RoPE | `ops/rope` `axes` | **split differs (44/44/40); the DiT's form is 2×2 matrices, not cos/sin pairs — unresolved** |
| HF-style RoPE | `llm/weights.ts` `permuteRopeChannels` | for the LLM adapter |
| embedding lookup | `ops/gather` | the adapter's `Embedding(32128, 1024)` |
| patchify / unpatchify | `examples/zimage/src/dit.ts` | different patch dim (68) and a mask channel |
| **F8_E4M3 reading** | **absent** | the text encoder ships partly in it; `SafetensorsFile` refuses it |
| softmax over 4 layers | `ops/softmax` | the adapter's `layer_mix_logits` |
| T5 tokenizer (unigram) | `llm/tokenizer.ts` | the prompt is tokenized twice, Qwen3.5 **and** T5 |
| **selective state-space scan** | **absent** | Qwen3.5's `SSMBlock`; see below |
| **softplus** | **absent** | `SSMBlock`'s `dt_bias` |

**The DiT needs no new kernel** — the one caveat is its RoPE form, which builds
2x2 rotation matrices rather than cos/sin pairs and may or may not fit
`ops/rope`'s `axes` entry. Everything else there is an op this library ships or
a composition of them, the same answer #163 reached for Z-Image.

**The text encoder is a different story**, and it is the finding that matters
most in this spike. See below.

## What is still unread

Named rather than left out, because a spike that reports only what it managed to
look at reads as a completeness claim:

- `AnimaQwen35UnifiedPrompt` — where the T5 ids come from
- `progressive_cross_adapter.py` — the `layer_mix_logits` forward
- ~~Qwen3.5 versus Qwen3~~ — **answered, and it is the biggest finding here.
  See below.**
- The sampler. The card says `res_multistep + Beta`, 28–50 steps, CFG 7–8; none
  of that has been read from an implementation
- Anything about accuracy or speed. **Nothing here has been run.**

## Qwen3.5 is not Qwen3, and this is the part that needs new kernels

> **Corrected by #170.** Everything in this section is true *of Qwen3.5*, and
> Qwen3.5 is not on the path to an image. The released workflow's own node
> (`prompt.py:219`) builds its second conditioning as `native + strength *
> (expanded - native)` and describes strength 0.0 as "native Anima": `native` is
> **Qwen3-0.6B**, and `text-encoder.ts` runs it with a config change and one
> optional final norm. What follows is what #173 needs for the `expanded`
> residual, not what a port needs to generate.

`examples/zimage/src/text-encoder.ts` **does not transfer** *to Qwen3.5*. It is
a **hybrid state-space model**, not a stack of attention blocks.
`text_encoder/layers.py:248-280`'s `HybridBlock` chooses per layer:

```python
if use_ssm:
    self.linear_attn = SSMBlock(...)
else:
    self.self_attn = GatedSelfAttention(...)
```

`SSMBlock` (`:43-128`) needs, per the constructor:

| piece | in this library |
| --- | --- |
| `in_proj_qkv` / `in_proj_z` / `in_proj_a` / `in_proj_b`, `out_proj` | `ops/matmul` |
| **`Conv1d` over the sequence** (`:62`) | `ops/conv` has conv1d — **shape and grouping unverified** |
| `A_log`, `dt_bias` parameters with `F.softplus` | **`softplus` is absent from `ops/activation`** |
| **the recurrent scan itself** | **absent.** This is the kernel Anima needs and this library does not have |
| `RMSNorm` over `head_v_dim`, `torch.sigmoid`, `F.silu` | present |

`GatedSelfAttention` (`:130-180`) is closer to familiar ground — `q_norm`/`k_norm`
over `head_dim`, RoPE, SDPA — but it is *gated*, which the existing engine is
not.

**This is the first model in this repository that needs a genuinely new
kernel.** Z-Image needed none; every op it wanted already existed. A selective
state-space scan is not a composition of what is here.

Two ways out for #170, and picking between them needs a measurement nobody has
taken:

1. **Implement the scan.** It is its own issue, and a sequential recurrence is
   an awkward shape for a GPU — the usual answer is a chunked parallel scan.
2. **Precompute the text features.** The DiT only ever sees the adapter's
   output. A demo with a fixed set of prompts could ship those, and the
   browser would run only the DiT and the VAE. That is not "Anima in a
   browser", and saying so is the point.

## What this implies for #170

CFG at 7–8 means **two forwards per step**, and 28–50 steps against Z-Image's 8.
That is roughly seven times the work per image before any difference in the
model itself. `examples/zimage-web` currently uploads weights to the GPU per
dispatch — measured at 28.57 GB up and 19.85 GB back per forward at 1,039
tokens — so #166's remaining residency work is a prerequisite for this being
usable rather than an optimisation to do afterwards.
