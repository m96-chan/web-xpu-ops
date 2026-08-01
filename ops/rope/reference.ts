/**
 * Rotary position embedding.
 *
 * Each adjacent pair within a head is rotated by an angle that depends on the
 * token's position and on how far into the head dimension the pair sits.
 *
 * `posOffset` exists for KV-cache continuation: during decoding the tensor
 * holds one token but its true position is however many came before.
 *
 * Note for anyone comparing a backend against this: GPU `sin` and `cos` are
 * markedly less accurate than the f64 ones here — measured at up to 1.86e-4 of
 * absolute error against 1.2e-7 for f32 epsilon — and this op calls both per
 * element. Tolerances have to reflect the hardware, not the arithmetic.
 */
export interface RoPEArgs {
  input: Float32Array;
  /** Tokens. */
  N: number;
  numHeads: number;
  headDim: number;
  /** Position of the first token, for KV-cache continuation. */
  posOffset: number;
  /** 10000 conventionally; some models use 500000. */
  thetaBase: number;
}

export function rope({ input, N, numHeads, headDim, posOffset, thetaBase }: RoPEArgs): Float32Array {
  const output = new Float32Array(input.length);
  const halfDim = headDim / 2;

  for (let token = 0; token < N; token += 1) {
    for (let head = 0; head < numHeads; head += 1) {
      for (let pair = 0; pair < halfDim; pair += 1) {
        const theta = (token + posOffset) * Math.pow(thetaBase, (-2 * pair) / headDim);
        const cos = Math.cos(theta);
        const sin = Math.sin(theta);

        const base = (token * numHeads + head) * headDim + pair * 2;
        const x0 = input[base]!;
        const x1 = input[base + 1]!;
        output[base] = x0 * cos - x1 * sin;
        output[base + 1] = x0 * sin + x1 * cos;
      }
    }
  }
  return output;
}
