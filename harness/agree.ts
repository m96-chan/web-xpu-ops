export interface Tolerance {
  /** Relative difference allowed. */
  rel?: number;
  /** Absolute difference below which an element passes regardless. */
  abs?: number;
}

export interface Disagreement {
  index: number;
  got: number;
  want: number;
  rel: number;
  abs: number;
}

/**
 * Compares a backend's output against an op's reference.
 *
 * Agreement, not equality: the reference runs in f64 and the kernels in f32.
 * An element passes on either measure, because relative alone is useless where
 * a result nears zero through cancellation, and absolute alone is useless where
 * the values are large.
 *
 * Returns the worst offender rather than a boolean, so a failing test can say
 * which element and by how much.
 */
export function agree(
  got: ArrayLike<number>,
  want: ArrayLike<number>,
  { rel = 1e-5, abs = 1e-6 }: Tolerance = {},
): Disagreement | null {
  if (got.length !== want.length) {
    throw new Error(`length mismatch: got ${got.length}, want ${want.length}`);
  }
  let worst: Disagreement | null = null;
  for (let index = 0; index < want.length; index += 1) {
    const absolute = Math.abs(got[index]! - want[index]!);
    if (absolute <= abs) continue;
    const relative = absolute / Math.max(Number.MIN_VALUE, Math.abs(want[index]!));
    if (relative <= rel) continue;
    if (!worst || relative > worst.rel) {
      worst = { index, got: got[index]!, want: want[index]!, rel: relative, abs: absolute };
    }
  }
  return worst;
}
