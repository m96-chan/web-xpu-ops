import { describe } from "vitest";
import { gpuTest, useGpu } from "../../harness/index.js";
import { check, prepare, type Case } from "./cases.js";

// Channel structure and the two edges the guards exist for. The workgroup-width
// sweep is in `wgsl.test.ts`; see `cases.ts` for why they are separate files.
const CASES: Case[] = [
  {
    name: "depthwise, Lout=777",
    N: 1, Cin: 3, Cout: 3, L: 776, K: 4, stride: 1, padding: 1, outputPadding: 0, dilation: 1, groups: 3,
  },
  // The two shapes DACVAE's decoder actually builds:
  // kernel_size = 2*stride, padding = (stride+1)//2,
  // output_padding = 1 if stride % 2 else 0. Odd stride is the only case where
  // output_padding is non-zero. Cin != Cout in both, because a codec decoder
  // changes the channel count as it upsamples — which is the only situation in
  // which reading weight as conv1d's [Cout, Cin/groups, K] would be caught at
  // all, rather than being an indexing accident that happens to agree.
  {
    name: "DACVAE stage, odd stride=3 (output_padding=1), Lout=771",
    N: 1, Cin: 4, Cout: 2, L: 257, K: 6, stride: 3, padding: 2, outputPadding: 1, dilation: 1, groups: 1,
  },
  {
    name: "DACVAE stage, even stride=4 (output_padding=0), Lout=260",
    N: 1, Cin: 2, Cout: 3, L: 65, K: 8, stride: 4, padding: 2, outputPadding: 0, dilation: 1, groups: 1,
  },
  // padding=2 with K=6 crops four of the nine uncropped positions away, so
  // nearly every output element depends on a tap being rejected rather than on
  // the data. Cin=2 and N=2 matter — with more than one channel row, an index
  // that runs off the left of its own row wraps to the *previous* row's tail,
  // which is real data rather than the zero an out-of-range read returns. With
  // a single row the left-edge guard has nothing to disagree with.
  {
    name: "crop dominates the window, both ends",
    N: 2, Cin: 2, Cout: 3, L: 4, K: 6, stride: 1, padding: 2, outputPadding: 0, dilation: 1, groups: 1,
  },
  // dilation=4 with K=4 spreads the taps twelve apart, so the last outputs are
  // reached only from samples well before them and the first outputs only from
  // samples well after. Past the final channel that is the sentinel slack, not
  // zero, so a missing right-edge guard shows up as a wrong number rather than
  // as the right one.
  {
    name: "dilated taps reach past both ends of the row",
    N: 1, Cin: 2, Cout: 2, L: 20, K: 4, stride: 1, padding: 0, outputPadding: 0, dilation: 4, groups: 1,
  },
];

// Built before the device exists; see `prepare` in cases.ts.
const PREPARED = CASES.map(prepare);

describe("convTranspose1d / wgsl shapes", () => {
  useGpu();

  for (const prepared of PREPARED) {
    gpuTest(`agrees with the reference: ${prepared.name}`, async (run) => {
      await check(run, prepared);
    });
  }
});
