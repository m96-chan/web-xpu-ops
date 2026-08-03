import { describe } from "vitest";
import { gpuTest, useGpu } from "../../harness/index.js";
import { check, prepare, type Case } from "./cases.js";

// Lout spans the 256-wide workgroup deliberately: well below it, one under it,
// exactly on it, and one that is neither. One thread per output element means
// the ragged-tail guard only has anything to guard when Lout is not a multiple
// of 256.
//
// The channel-structure and edge cases live in `wgsl-shapes.test.ts`; see
// `cases.ts` for why they are not all in one file.
//
//   Lout = (L-1)*stride - 2*padding + dilation*(K-1) + output_padding + 1
const CASES: Case[] = [
  {
    name: "plain, Lout=12",
    N: 1, Cin: 1, Cout: 1, L: 10, K: 3, stride: 1, padding: 0, outputPadding: 0, dilation: 1, groups: 1,
  },
  {
    name: "strided + cropped, Lout=255",
    N: 3, Cin: 2, Cout: 2, L: 128, K: 7, stride: 2, padding: 3, outputPadding: 0, dilation: 1, groups: 1,
  },
  {
    name: "cropped, Lout=256",
    N: 2, Cin: 3, Cout: 4, L: 256, K: 5, stride: 1, padding: 2, outputPadding: 0, dilation: 1, groups: 1,
  },
  {
    name: "strided + dilated + grouped + output_padding, Lout=300",
    N: 2, Cin: 4, Cout: 6, L: 150, K: 3, stride: 2, padding: 2, outputPadding: 1, dilation: 2, groups: 2,
  },
];

// Built before the device exists; the tests below only dispatch and compare.
// See `prepare` in cases.ts for why that ordering is load-bearing.
const PREPARED = CASES.map(prepare);

describe("convTranspose1d / wgsl", () => {
  useGpu();

  for (const prepared of PREPARED) {
    gpuTest(`agrees with the reference: ${prepared.name}`, async (run) => {
      await check(run, prepared);
    });
  }
});
