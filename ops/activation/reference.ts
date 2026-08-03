/** Which non-linearity to apply. The numbers are the shader's `activation_type`. */
export const ACTIVATION = {
  relu2: 0,
  silu: 1,
  elu: 2,
  tanh: 3,
  gelu: 4,
  gelu_tanh: 5,
} as const;
export type ActivationKind = (typeof ACTIVATION)[keyof typeof ACTIVATION];

/**
 * `erf`, to f64 accuracy, because JavaScript does not have one and the exact
 * GELU is defined by it.
 *
 * The series is the confluent form,
 *
 *   erf(x) = 2x·e^(-x²)/√π · Σ_{n≥0} (2x²)ⁿ / (2n+1)!!
 *
 * whose terms are all positive. The textbook Maclaurin series alternates and
 * has lost four digits to cancellation by |x| = 3, which is squarely inside
 * the range GELU is evaluated over. Past |x| = 6 the answer is ±1: erfc(6) is
 * 2.2e-17, below the ulp of 1, so there is nothing left to add.
 */
function erf(x: number): number {
  if (x < 0) return -erf(-x);
  if (x >= 6) return 1;
  const twoXSquared = 2 * x * x;
  let term = 1;
  let sum = 1;
  for (let n = 1; n < 400; n += 1) {
    term *= twoXSquared / (2 * n + 1);
    sum += term;
    if (term < sum * 1e-18) break;
  }
  return ((2 * x) / Math.sqrt(Math.PI)) * Math.exp(-x * x) * sum;
}

export interface ActivationArgs {
  input: Float32Array;
  kind: ActivationKind;
  /**
   * `elu`'s negative-branch scale, and nothing else's.
   *
   * A **scalar hyperparameter**, not a learned tensor: `torch.nn.ELU` takes it
   * in the constructor and defaults it to 1.0, which is also what DACVAE passes
   * explicitly. Snake's `alpha` looks similar and is not — it is a trained
   * per-channel tensor, which is why it is `ops/snake` and not a kind here.
   */
  alpha?: number;
}

/**
 * Elementwise activation.
 *
 * Every member is a pure function of one element, needs no layout, and reads no
 * second buffer. That is the shape of this op, and the reason Snake is not in
 * it.
 *
 * | kind        | definition                          | matches                                    |
 * | ----------- | ----------------------------------- | ------------------------------------------ |
 * | `relu2`     | `max(0, x)²`                        | BitNet's official 2B-4T FFN                |
 * | `silu`      | `x · sigmoid(x)`                    | `torch.nn.SiLU`                            |
 * | `elu`       | `x if x > 0 else α·(eˣ - 1)`        | `torch.nn.ELU`, `alpha` default 1.0        |
 * | `tanh`      | `tanh(x)`                           | `torch.tanh`                               |
 * | `gelu`      | `x·Φ(x)` = `0.5x(1 + erf(x/√2))`    | `torch.nn.functional.gelu` **default**     |
 * | `gelu_tanh` | `0.5x(1 + tanh(√(2/π)(x + 0.044715x³)))` | `gelu(approximate="tanh")`            |
 *
 * On GELU (rule 7): torch ships **two** GELUs and they are not the same
 * function. Measured over x ∈ [-8, 8] in this reference, they differ by at most
 * **4.73e-4, at x = 2.699** — around the knee, which is where activations
 * actually live, and four orders of magnitude above f32 noise. Neither is "the"
 * GELU, so both are here and the default is `gelu`, the exact one, because that
 * is `approximate="none"` and torch's default. BERT-family checkpoints and
 * ModernBERT's GeGLU are trained against whichever their config names; a
 * library that silently picked one would be right for half its callers.
 */
export function activation({ input, kind, alpha = 1 }: ActivationArgs): Float32Array {
  const output = new Float32Array(input.length);
  for (let i = 0; i < input.length; i += 1) {
    const x = input[i]!;
    output[i] = apply(x, kind, alpha);
  }
  return output;
}

function apply(x: number, kind: ActivationKind, alpha: number): number {
  switch (kind) {
    case ACTIVATION.relu2:
      return Math.max(0, x) ** 2;
    case ACTIVATION.silu:
      return x / (1 + Math.exp(-x));
    case ACTIVATION.elu:
      // `x > 0`, as torch's kernel writes it. At x = 0 both branches are 0, so
      // the boundary is not observable — but the strict comparison is.
      return x > 0 ? x : alpha * (Math.exp(x) - 1);
    case ACTIVATION.tanh:
      return Math.tanh(x);
    case ACTIVATION.gelu:
      return 0.5 * x * (1 + erf(x / Math.SQRT2));
    default:
      return 0.5 * x * (1 + Math.tanh(Math.sqrt(2 / Math.PI) * (x + 0.044715 * x ** 3)));
  }
}
