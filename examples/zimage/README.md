# `examples/zimage` — does the op set reach Z-Image?

One `ZImageTransformerBlock`, composed from this repository's ops and checked
against the model's own implementation.

**This is not a demo.** Nothing here generates an image, and nothing here runs
on a GPU. It answers a narrower question, asked by issue #163: after #145,
#146 and #149–#152 landed, do those ops actually compose into a real model's
block, or do they only satisfy each op's own reference? Those are different
claims, and only the second one had been shown.

## What it establishes

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
- **Not the real weights.** Random, fixed seed. What it proves is that the
  arithmetic lines up, not that any particular checkpoint loads.

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
