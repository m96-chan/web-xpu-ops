/**
 * Snake: `x + (1/α_c) · sin²(α_c · x)`, with a **learned per-channel** α.
 *
 * The periodic activation from arXiv:2006.08195, as neural audio codecs use it:
 * BigVGAN introduced it to vocoders and DACVAE (`dacvae/nn/layers.py`) builds
 * every encoder and decoder stage from it. Its inductive bias is periodicity —
 * the thing a waveform has and ReLU cannot express.
 *
 * **This is the one-parameter form.** {@link snakeBeta} is the other one, and a
 * checkpoint's `snake.alpha` tensor does not say which of the two it belongs
 * to. Reading that table before assuming this one is cheaper than finding out
 * by ear — the two differ in how many parameters they carry, in whether those
 * parameters are stored logarithmically, and in which of them the epsilon
 * guards.
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

/**
 * SnakeBeta: `x + (1/β_c) · sin²(α_c · x)`, with **two** learned per-channel
 * parameters.
 *
 * The form BigVGAN calls `SnakeBeta` and MioCodec's decoder is built from
 * (`m96-chan/candle-miotts`, `src/models/miocodec/snake.rs`). It generalises
 * {@link snake}, which is the case `β = α`: there, one number sets both how
 * fast the sine runs and how much of it is added, and here they are separate.
 *
 * ## Two Snakes, one name, and a failure that makes no noise
 *
 * A checkpoint holding a `snake.alpha` tensor may mean either function, and
 * nothing in the file says which. That is why both live here, side by side,
 * rather than one of them being "the" Snake:
 *
 * | | DACVAE `Snake1d` | BigVGAN / MioCodec `SnakeBeta` |
 * | --- | --- | --- |
 * | parameters | `alpha` | `alpha`, `beta` |
 * | stored as | the value itself | **its logarithm** |
 * | divisor | `alpha` | `beta` |
 *
 * ## This function does not exponentiate, deliberately
 *
 * MioCodec stores `log α` and `log β` and its loader applies `exp()` before
 * calling. Folding that in here would be wrong for every DACVAE checkpoint,
 * whose α is already a value — so the log scale is a **checkpoint convention**,
 * belonging to whoever reads the file, and this function takes actual α and β.
 *
 * It is worth knowing which way that error falls, because it does not announce
 * itself. A log-scale α of `0.0` means `α = 1`; passed through unconverted it
 * is `α = 0`, so `sin(0 · x)` is 0 and the activation collapses to `y = x`. No
 * NaN, no shape mismatch, nothing to catch — just a vocoder that sounds duller
 * than it should. Convert on load.
 *
 * ## Where the epsilon goes, and why it moved
 *
 * {@link SNAKE_EPS} guards the **divisor**, which in {@link snake} is `α` and
 * here is `β` — `candle-miotts` writes `(beta + 1e-9).recip()`, matching
 * upstream's placement rather than its parameter. The two are not symmetric,
 * and the asymmetry is real: at `α = 0` the sine vanishes and the answer is `x`
 * whatever β is, while at `β = 0` the scale is 1e9 multiplying a sine that is
 * *not* zero, so the output is enormous — but finite, where `1/β` would give an
 * infinity that no tolerance survives.
 *
 * Through a checkpoint that regime is unreachable: `exp` is never zero, so β is
 * strictly positive. The guard is for callers of this API, not for weights.
 */
export interface SnakeBetaArgs extends SnakeArgs {
  /**
   * `[C]`: the trained amplitude scale, one per channel, and the divisor.
   *
   * An **actual value**, not the `log β` a BigVGAN-family checkpoint stores.
   */
  beta: Float32Array;
}

export function snakeBeta({ input, alpha, beta, N, C, L }: SnakeBetaArgs): Float32Array {
  const output = new Float32Array(N * C * L);
  for (let n = 0; n < N; n += 1) {
    for (let c = 0; c < C; c += 1) {
      const a = alpha[c]!;
      const scale = 1 / (beta[c]! + SNAKE_EPS);
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
