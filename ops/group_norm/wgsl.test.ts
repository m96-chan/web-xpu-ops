import { describe } from "vitest";
import { expectAgrees, gpuTest, kernel, params, useGpu } from "../../harness/index.js";
import { groupNorm } from "./reference.js";

const code = kernel(import.meta.url);

const wave = (n: number, k = 0.37) => Float32Array.from({ length: n }, (_, i) => Math.sin(i * k) * 3);

/**
 * Non-uniform **within** a group, which is the only way this op's weight
 * indexing is observable.
 *
 * Channels in one group share a mean and a variance and do not share their
 * affine transform. A weight that were constant inside each group would pass
 * for a kernel that indexed it by group — and every case here has more than one
 * channel per group, so that mistake has somewhere to show up.
 */
const weights = (c: number) => Float32Array.from({ length: c }, (_, i) => 0.5 + Math.cos(i * 0.7) * 0.4);
const biases = (c: number) => Float32Array.from({ length: c }, (_, i) => Math.sin(i * 1.3) * 0.3);
const EPS = 1e-5;

describe("group_norm / wgsl", () => {
  useGpu();

  // The reduced run is `(C/G) × L`, and it is that number — not `L` — which
  // decides how the strided loop and the two tree reductions behave. The cases
  // put it below 256, exactly on it, not a multiple of it, and well past it.
  // G is never 1 and never C in any of them: both extremes collapse the
  // channel pooling that is the whole op, and a kernel could get either right
  // while pooling nothing.
  for (const [N, C, L, G] of [
    [2, 6, 4, 3], //   8 — two channels per group, below one workgroup
    [1, 4, 128, 2], // 256 — exactly one workgroup
    [3, 8, 75, 4], //  150 — not a multiple of 256, and N > 1 with G > 1
    [2, 8, 300, 2], // 1200 — the strided loop runs several times
    [1, 64, 40, 8], // 320 — eight channels per group, as a codec stage carries
  ] as const) {
    gpuTest(`agrees with the reference at N=${N} C=${C} L=${L} G=${G}`, async (run) => {
      const input = wave(N * C * L);
      const weight = weights(C);
      const bias = biases(C);
      await expectAgrees(
        run,
        {
          code,
          bindings: [
            { kind: "storage", data: input },
            { kind: "storage", data: weight },
            { kind: "storage", data: bias },
            { kind: "out", type: "f32", length: N * C * L },
            {
              kind: "uniform",
              data: params([
                ["u32", N],
                ["u32", C],
                ["u32", L],
                ["u32", G],
                ["f32", EPS],
              ]),
            },
          ],
          workgroups: [N * G],
        },
        [groupNorm({ input, weight, bias, N, C, L, G, eps: EPS })],
      );
    });
  }

  gpuTest("pools across channels rather than along one", async (run) => {
    // The mistake this op exists to prevent, made observable: give each channel
    // in a group a different offset, so pooling them produces a different mean
    // from normalising each on its own.
    //
    // Channel 0 sits around +50 and channel 1 around -50 inside the same group.
    // Pooled, the group mean is ~0 and both channels come out far from zero.
    // Per channel, each is centred on its own mean and comes out around zero —
    // a well-formed answer, a different one, and no error either way.
    const [N, C, L, G] = [1, 4, 16, 2];
    const input = Float32Array.from({ length: N * C * L }, (_, i) => {
      const channel = Math.floor(i / L) % C;
      return (channel % 2 === 0 ? 50 : -50) + Math.sin(i * 0.9);
    });
    const weight = weights(C);
    const bias = biases(C);
    await expectAgrees(
      run,
      {
        code,
        bindings: [
          { kind: "storage", data: input },
          { kind: "storage", data: weight },
          { kind: "storage", data: bias },
          { kind: "out", type: "f32", length: N * C * L },
          {
            kind: "uniform",
            data: params([
              ["u32", N],
              ["u32", C],
              ["u32", L],
              ["u32", G],
              ["f32", EPS],
            ]),
          },
        ],
        workgroups: [N * G],
      },
      [groupNorm({ input, weight, bias, N, C, L, G, eps: EPS })],
    );
  });

  gpuTest("holds on groups whose mean dwarfs their spread", async (run) => {
    // The case that decides between two passes and the one-pass identity
    // `var = E[x²] - E[x]²`, carried over from `layernorm` because this op
    // makes it worse rather than better: the run being reduced is `(C/G) × L`
    // rather than `L`, so there are more squares to accumulate past 2^26.
    //
    // Integers by construction, so everything up to the variance is exact in
    // f32 and none of this is a tolerance argument.
    const [N, C, L, G] = [1, 4, 128, 2];
    const deviation = (i: number) => ((i * 37) % 9) - 4;
    const input = Float32Array.from({ length: N * C * L }, (_, i) =>
      (i < (C / G) * L ? 8192 : 1024) + deviation(i),
    );
    const weight = weights(C);
    const bias = biases(C);
    await expectAgrees(
      run,
      {
        code,
        bindings: [
          { kind: "storage", data: input },
          { kind: "storage", data: weight },
          { kind: "storage", data: bias },
          { kind: "out", type: "f32", length: N * C * L },
          {
            kind: "uniform",
            data: params([
              ["u32", N],
              ["u32", C],
              ["u32", L],
              ["u32", G],
              ["f32", EPS],
            ]),
          },
        ],
        workgroups: [N * G],
      },
      [groupNorm({ input, weight, bias, N, C, L, G, eps: EPS })],
    );
  });

  gpuTest("still answers when dispatched more workgroups than groups", async (run) => {
    // Callers round the dispatch up, so this establishes that doing so is safe.
    //
    // It does **not** establish that the kernel's `slot >= N * G` guard is what
    // makes it safe: removing that guard leaves this green, because a surplus
    // workgroup addresses past the end of the buffer and WebGPU discards the
    // out-of-range writes. Said here as well as in the kernel so nobody reads
    // this test as covering more than it does.
    const [N, C, L, G] = [2, 6, 4, 3];
    const input = wave(N * C * L);
    const weight = weights(C);
    const bias = biases(C);
    await expectAgrees(
      run,
      {
        code,
        bindings: [
          { kind: "storage", data: input },
          { kind: "storage", data: weight },
          { kind: "storage", data: bias },
          { kind: "out", type: "f32", length: N * C * L },
          {
            kind: "uniform",
            data: params([
              ["u32", N],
              ["u32", C],
              ["u32", L],
              ["u32", G],
              ["f32", EPS],
            ]),
          },
        ],
        workgroups: [N * G + 5],
      },
      [groupNorm({ input, weight, bias, N, C, L, G, eps: EPS })],
    );
  });
});
