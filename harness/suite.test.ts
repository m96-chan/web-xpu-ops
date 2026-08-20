import { describe, expect, it, vi } from "vitest";
import { skipUnlessPresent } from "./suite.js";

/**
 * PR #116 review, item 2: `residentTest`'s old body was
 * `it(name, async () => { if (!currentResident) return; ... })` — on a
 * machine with no WebGPU adapter, that `return` reports the test as a green
 * PASS with zero assertions, not a skip. This repository has already hit
 * that exact shape twice (`gpuTest`'s own identical pattern, and #108's
 * real-model test before it switched to `describe.skipIf`) — see
 * `llm/engine-q8.real-model.test.ts`'s comment on why `skipIf`/`ctx.skip()`
 * beats an early `return` here.
 *
 * `residentTest` cannot know at collection time whether an adapter exists
 * (that is only known after `useResidentGpu`'s `beforeAll` resolves), so
 * `describe.skipIf` — evaluated at collection time — is not an option here;
 * `skipUnlessPresent` calls the per-test `ctx.skip()` instead, at run time.
 * Tested here as a pure function (a fake `ctx.skip`) rather than through a
 * real vitest run without an adapter, which this repository's own machine
 * cannot exercise (the dev machine always has one) and which vitest's own
 * test suite already covers for `ctx.skip()` itself — see this file's
 * neighbor `harness/resident.test.ts` for the real-adapter path.
 */
describe("harness/suite skipUnlessPresent", () => {
  it("calls ctx.skip() and reports absent when the resource is null", () => {
    const skip = vi.fn();
    const result = skipUnlessPresent({ skip } as any, null);
    expect(skip).toHaveBeenCalledOnce();
    expect(result).toBe(false);
  });

  it("does not call ctx.skip() and reports present when the resource exists", () => {
    const skip = vi.fn();
    const resource = { stats: { buffersCreated: 0 } };
    const result = skipUnlessPresent({ skip } as any, resource);
    expect(skip).not.toHaveBeenCalled();
    expect(result).toBe(true);
  });
});
