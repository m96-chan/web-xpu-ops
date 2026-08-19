import { describe, expect, it } from "vitest";
import { concatRows, mergeHeadsMajor, splitConcatRows, splitHeadsMajor, transposeRowMajor } from "./reshape.js";

describe("transposeRowMajor", () => {
  it("transposes a non-square matrix", () => {
    // [2, 3]: [[1,2,3],[4,5,6]] -> [3, 2]: [[1,4],[2,5],[3,6]]
    const input = Float32Array.from([1, 2, 3, 4, 5, 6]);
    expect(Array.from(transposeRowMajor(input, 2, 3))).toEqual([1, 4, 2, 5, 3, 6]);
  });

  it("is its own inverse", () => {
    const input = Float32Array.from({ length: 5 * 7 }, (_, i) => Math.sin(i));
    const once = transposeRowMajor(input, 5, 7);
    const twice = transposeRowMajor(once, 7, 5);
    expect(Array.from(twice)).toEqual(Array.from(input));
  });
});

describe("splitHeadsMajor / mergeHeadsMajor", () => {
  // [tokens=2, heads=3, dim=2] token-major, values encode (token, head) so a
  // swapped axis shows up as the wrong pair rather than a coincidentally equal
  // number.
  const tokens = 2;
  const heads = 3;
  const dim = 2;
  const tokenMajor = Float32Array.from({ length: tokens * heads * dim }, (_, i) => {
    const t = Math.floor(i / (heads * dim));
    const h = Math.floor((i % (heads * dim)) / dim);
    const d = i % dim;
    return t * 100 + h * 10 + d;
  });

  it("moves the head axis in front of the token axis", () => {
    const headMajor = splitHeadsMajor(tokenMajor, tokens, heads, dim);
    // head-major layout: [h, t, d] flattened
    for (let h = 0; h < heads; h += 1) {
      for (let t = 0; t < tokens; t += 1) {
        for (let d = 0; d < dim; d += 1) {
          expect(headMajor[(h * tokens + t) * dim + d]).toBe(t * 100 + h * 10 + d);
        }
      }
    }
  });

  it("mergeHeadsMajor is the exact inverse of splitHeadsMajor", () => {
    const headMajor = splitHeadsMajor(tokenMajor, tokens, heads, dim);
    const roundTrip = mergeHeadsMajor(headMajor, heads, tokens, dim);
    expect(Array.from(roundTrip)).toEqual(Array.from(tokenMajor));
  });
});

describe("concatRows / splitConcatRows", () => {
  it("concatRows stacks matrices along the row axis", () => {
    const a = Float32Array.from([1, 2, 3, 4]); // [2, 2]
    const b = Float32Array.from([5, 6]); // [1, 2]
    expect(Array.from(concatRows([a, b], [2, 1], 2))).toEqual([1, 2, 3, 4, 5, 6]);
  });

  it("splitConcatRows undoes a fused projection's output back into per-weight token-major arrays", () => {
    // Emulates 2 tokens projected through a fused [Wq;Wk] (sizes 3 and 2 per token).
    const tokens = 2;
    const sizes = [3, 2];
    // token 0: q=[10,11,12] k=[20,21]; token 1: q=[13,14,15] k=[22,23]
    const flat = Float32Array.from([10, 11, 12, 20, 21, 13, 14, 15, 22, 23]);
    const [q, k] = splitConcatRows(flat, tokens, sizes);
    expect(Array.from(q!)).toEqual([10, 11, 12, 13, 14, 15]);
    expect(Array.from(k!)).toEqual([20, 21, 22, 23]);
  });

  it("splitConcatRows is the exact inverse of a fused matvec/matmul dispatch built from concatRows", () => {
    // Build a fused weight from two real weights, project by hand (no GPU —
    // this checks the reshape only), and confirm splitting the fused output
    // recovers each unfused projection's output exactly.
    const hidden = 3;
    const wq = Float32Array.from([1, 0, 0, 0, 1, 0]); // [2, 3]
    const wk = Float32Array.from([0, 0, 1]); // [1, 3]
    const fused = concatRows([wq, wk], [2, 1], hidden);

    const project = (w: Float32Array, rows: number, x: Float32Array) => {
      const out = new Float32Array(rows);
      for (let r = 0; r < rows; r += 1) {
        let sum = 0;
        for (let c = 0; c < hidden; c += 1) sum += w[r * hidden + c]! * x[c]!;
        out[r] = sum;
      }
      return out;
    };

    const x = Float32Array.from([7, 8, 9]);
    const fusedOut = project(fused, 3, x); // [2+1, 3] @ [3] -> [3]
    const [q, k] = splitConcatRows(fusedOut, 1, [2, 1]);
    expect(Array.from(q!)).toEqual(Array.from(project(wq, 2, x)));
    expect(Array.from(k!)).toEqual(Array.from(project(wk, 1, x)));
  });
});
