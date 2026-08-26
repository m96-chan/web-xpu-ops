/**
 * The encoder's channel plan.
 *
 * The model-scale comparison is `verify-encode.ts`, a script rather than a
 * test: the reference walks six levels of twelve blocks on the CPU and takes
 * well over `scripts/test.mjs`'s per-file minute — the same reason
 * `examples/anima` verifies its forward with a script, and the same reason
 * `examples/h3-video/src/verify-decode.ts` is one.
 *
 * What is left here is the arithmetic that is cheap and easy to get wrong.
 */
import { describe, expect, it } from "vitest";
import { channelPlan, type EncoderConfig } from "./encoder.js";

describe("h3 video vae / encoder", () => {
  it("derives the channel plan the model derives", () => {
    // `block_mid = ch * ch_mult`, `block_in = [block_mid[0], *block_mid[:-1]]`.
    // Written out because the first block of each level takes the *previous*
    // level's width, and reading it as the current one is a shape error only at
    // level 1 — everything before it would still run.
    const cfg = { ch: 128, chMult: [1, 2, 2, 4, 4, 8] } as EncoderConfig;
    const { blockIn, blockMid } = channelPlan(cfg);
    expect(blockMid).toEqual([128, 256, 256, 512, 512, 1024]);
    expect(blockIn).toEqual([128, 128, 256, 256, 512, 512]);
  });

  it("compresses by the factors the config states", () => {
    // 2*2*2*2*1*1 in space and 1*2*2*1*1*1 in time — 16x and 4x, which is what
    // `vae_ratio` and `vae_ratio_t` are derived from and what the decoder's
    // `patch_size` and `patch_size_t` have to match.
    const space = [2, 2, 2, 2, 1, 1].reduce((a, b) => a * b, 1);
    const time = [1, 2, 2, 1, 1, 1].reduce((a, b) => a * b, 1);
    expect(space).toBe(16);
    expect(time).toBe(4);
  });
});
