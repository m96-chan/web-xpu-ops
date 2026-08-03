/**
 * Snake: `x + (1/α_c) · sin²(α_c · x)`, with a **learned per-channel** α.
 *
 * The periodic activation from arXiv:2006.08195, as neural audio codecs use it:
 * BigVGAN introduced it to vocoders and DACVAE (`dacvae/nn/layers.py`) builds
 * every encoder and decoder stage from it. Its inductive bias is periodicity —
 * the thing a waveform has and ReLU cannot express.
 *
 * ## Why this is its own op rather than a kind of `activation`
 *
 * This is the decision issue #76 exists to make, so it is written down rather
 * than left to be inferred from the directory layout.
 *
 * `ops/activation` is "one buffer in, one buffer out, branch on an integer".
 * Every member of it is a pure function of a single element: no layout, no
 * second input, `N` is the only thing the kernel is told. Snake breaks all
 * three at once — it needs a second storage buffer, and it needs to know the
 * input's channel stride to index it.
 *
 * Folding it in would not be a new arm of the switch, it would be a new
 * *binding*. The harness builds pipelines with `layout: "auto"`, so a bind
 * group is derived from what the shader declares and every binding it declares
 * must be supplied — and since #46 an unsupplied or unreferenced one throws
 * rather than quietly reading zeros. So `relu2` and `silu` would each have to
 * bind a dummy α buffer and pass a channel count they do not have, on every
 * call, to serve the one member that wants them. The common case would pay for
 * the rare one, in the caller's code as well as in the shader.
 *
 * The converse decision is `elu`'s: its `alpha` *is* a scalar hyperparameter,
 * so it rides in the uniform block that already exists and costs the other
 * kinds four bytes they were padding out to anyway. That one stays in
 * `activation`. The dividing line is "does it need a buffer", not "does it have
 * a parameter".
 *
 * ## The two numerical details, both upstream's rather than choices
 *
 * 1. **The epsilon is inside the reciprocal, and it is `1e-9`.** Upstream is
 *    literally `x + (alpha + 1e-9).reciprocal() * torch.sin(alpha * x).pow(2)`,
 *    so the scale is `1/(α + 1e-9)` — *not* `1/α` guarded afterwards, and *not*
 *    `α/(α² + ε)`. Those three agree to seven digits for a trained α and
 *    disagree completely near zero, which is where a checkpoint that trained an
 *    α to nearly nothing would find out. Reproduced exactly, sign included: at
 *    α = -1e-9 upstream divides by zero and so does this.
 * 2. **`sin²(αx)` is the square of the sine**, not the sine of a square. The
 *    usual way of writing this op, `x + sin²(αx)/α`, is ambiguous enough to be
 *    worth stating; upstream's `torch.sin(alpha * x).pow(2)` is not.
 */
export interface SnakeArgs {
  /** `[N, C, L]`, contiguous — the shape a 1-D codec stage carries. */
  input: Float32Array;
  /** `[C]`: one trained value per channel, shared across batch and length. */
  alpha: Float32Array;
  /** Batch. */
  N: number;
  /** Channels; `alpha` is indexed by this axis. */
  C: number;
  /** Length; upstream flattens every trailing axis into this one. */
  L: number;
}

/**
 * Upstream's epsilon, at upstream's value and upstream's place.
 *
 * `Snake1d` in `dacvae/nn/layers.py` (and BigVGAN's `activations.py` before
 * it). Not a tunable and not a guard this library chose.
 */
export const SNAKE_EPS = 1e-9;

export function snake({ input, alpha, N, C, L }: SnakeArgs): Float32Array {
  const output = new Float32Array(N * C * L);
  for (let n = 0; n < N; n += 1) {
    for (let c = 0; c < C; c += 1) {
      const a = alpha[c]!;
      const scale = 1 / (a + SNAKE_EPS);
      for (let l = 0; l < L; l += 1) {
        const index = (n * C + c) * L + l;
        const x = input[index]!;
        const s = Math.sin(a * x);
        output[index] = x + scale * (s * s);
      }
    }
  }
  return output;
}
