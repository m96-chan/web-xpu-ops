import { describe, expect, it } from "vitest";
import type { Dispatch } from "../harness/wgsl.js";
import { LlamaEngineQ8 } from "./engine-q8.js";
import { loadTinyFixtureQ8 } from "./fixture-q8.js";

/**
 * `LlamaEngineQ8#cacheState` (issue #110's prefill handoff — see that
 * getter's own doc) against a **fake** `Runner["run"]`, no GPU: this getter
 * has no GPU-dependent behaviour of its own, it just exposes two fields
 * `forward` already maintains, so the useful thing to prove here is the
 * bookkeeping — `position` tracks tokens consumed, and the cache holds
 * exactly that many positions per layer, per `KVCache.read`'s own contract —
 * not the arithmetic inside a dispatch (`engine-q8.wgsl.test.ts` already
 * covers that on real hardware).
 */
describe("LlamaEngineQ8#cacheState (mocked Runner, no GPU)", () => {
  const fixture = loadTinyFixtureQ8();

  const echoRun = async (dispatch: Dispatch) =>
    dispatch.bindings
      .filter((b): b is Extract<Dispatch["bindings"][number], { kind: "out" }> => b.kind === "out")
      .map((b) => (b.type === "i32" ? new Int32Array(b.length) : b.type === "u32" ? new Uint32Array(b.length) : new Float32Array(b.length)));

  it("position is 0 before any forward call", () => {
    const engine = new LlamaEngineQ8(fixture.config, fixture.weights, echoRun);
    expect(engine.cacheState.position).toBe(0);
    expect(engine.cacheState.cache.numLayers).toBe(fixture.config.numLayers);
  });

  it("position advances by the prompt length after prefill, and the cache holds that many positions", async () => {
    const engine = new LlamaEngineQ8(fixture.config, fixture.weights, echoRun);
    await engine.forward(fixture.promptTokens);

    const { cache, position } = engine.cacheState;
    expect(position).toBe(fixture.promptTokens.length);
    // `KVCache.read` throws if asked for more positions than `maxSeqLen`
    // holds, so a successful read at exactly `position` is itself the
    // "cache learned this many positions" assertion — reading one more
    // would still succeed (maxSeqLen headroom) without checking anything.
    const { k, v } = cache.read(0, position);
    expect(k.length).toBe(cache.kvHeads * position * cache.headDim);
    expect(v.length).toBe(cache.kvHeads * position * cache.headDim);
  });

  it("position keeps advancing across decode-shaped (single-token) forward calls", async () => {
    const engine = new LlamaEngineQ8(fixture.config, fixture.weights, echoRun);
    await engine.forward(fixture.promptTokens);
    const afterPrefill = engine.cacheState.position;

    await engine.forward([fixture.promptTokens[0]!]);
    await engine.forward([fixture.promptTokens[0]!]);

    expect(engine.cacheState.position).toBe(afterPrefill + 2);
  });
});
