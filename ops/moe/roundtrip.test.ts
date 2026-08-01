import { describe, expect } from "vitest";
import { expectAgrees, gpuTest, kernel, useGpu } from "../../harness/index.js";
import { moeDispatch, moeGather, router } from "./reference.js";
import { maxLoad, moeParams, routerParams, tokenThreads, wave } from "./testing.js";

const routerCode = kernel(import.meta.url, "router");
const dispatchCode = kernel(import.meta.url, "dispatch");
const gatherCode = kernel(import.meta.url, "gather");

/**
 * All three kernels chained on the GPU, which is the only place the op is
 * actually itself: each of them can be right on its own and the three still not
 * compose.
 *
 * Fixtures at module scope, and shapes kept small — see the note in
 * `testing.ts`. Inside these tests the only work between the three dispatches is
 * the chaining itself.
 */

/**
 * Room for everything, so nothing is dropped and the answer is x itself.
 *
 * Capacity comes from the reference rather than from `T*k`, which would be the
 * obvious way to say "enough": `T*k` seats per expert is an `E*T*k*D` buffer,
 * eight times larger than anything else in the suite and past what this harness
 * survives. The largest load any expert actually receives is exactly as safe
 * and two orders of magnitude smaller.
 */
const WHOLE = (() => {
  const [T, k, E, D] = [300, 2, 8, 8];
  const logits = wave(T * E, 0.23);
  const C = maxLoad(router({ logits, T, E, k, normalize: true }).expert, E);
  return { shape: { T, k, E, C, D }, logits, x: wave(T * D, 0.11) };
})();

/** Capacity biting, checked against the reference chain rather than against x. */
const CROWDED = (() => {
  const shape = { T: 128, k: 2, E: 4, C: 20, D: 16 };
  const logits = wave(shape.T * shape.E, 0.23);
  const x = wave(shape.T * shape.D, 0.11);
  const { expert, gate } = router({ logits, ...shape, normalize: true });
  const { buffer, pos } = moeDispatch({ x, expert, ...shape });
  const dropped = [...Array(shape.T).keys()].filter((t) =>
    [...Array(shape.k).keys()].every((r) => pos[t * shape.k + r]! < 0),
  );
  return { shape, logits, x, pos, dropped, expected: moeGather({ y: buffer, expert, pos, gate, ...shape }) };
})();

/** Expert 0 wins every token: the other three are addressed by nothing. */
const EMPTY_EXPERTS = (() => {
  const shape = { T: 32, k: 1, E: 4, C: 40, D: 8 };
  const logits = new Float32Array(shape.T * shape.E);
  for (let t = 0; t < shape.T; t += 1) logits[t * shape.E] = 10;
  return { shape, logits, x: wave(shape.T * shape.D, 0.11) };
})();

describe("moe round trip / wgsl", () => {
  useGpu();

  gpuTest("restores token order and applies the gate once", async (run) => {
    // The experts are left as the identity, so the answer is known without the
    // reference: normalised gates sum to 1 over the k, so gather has to hand
    // back x itself, in token order.
    //
    // Two failures this catches that nothing else does. A permutation that
    // dispatch and gather agree on but that is not the identity — say both
    // reverse the rows within an expert — cancels out in every per-kernel test
    // and shows up here as a shuffled batch. And a gate applied at both ends
    // squares it: sum(g^2) < 1 for any split, so the output shrinks.
    const { shape, logits, x } = WHOLE;
    const { T, k, E, C, D } = shape;

    const [expert, gate] = (await run({
      code: routerCode,
      bindings: [
        { kind: "storage", data: logits },
        { kind: "out", type: "i32", length: T * k },
        { kind: "out", type: "f32", length: T * k },
        { kind: "uniform", data: routerParams(T, E, k, true) },
      ],
      workgroups: tokenThreads(T),
    })) as [Int32Array, Float32Array];

    const [buffer, pos] = (await run({
      code: dispatchCode,
      bindings: [
        { kind: "storage", data: x },
        { kind: "storage", data: expert },
        { kind: "out", type: "f32", length: E * C * D },
        { kind: "out", type: "i32", length: T * k },
        { kind: "uniform", data: moeParams(T, k, E, C, D) },
      ],
      workgroups: [T * k],
    })) as [Float32Array, Int32Array];

    await expectAgrees(
      run,
      {
        code: gatherCode,
        bindings: [
          { kind: "storage", data: buffer },
          { kind: "storage", data: expert },
          { kind: "storage", data: pos },
          { kind: "storage", data: gate },
          { kind: "out", type: "f32", length: T * D },
          { kind: "uniform", data: moeParams(T, k, E, C, D) },
        ],
        workgroups: [T],
      },
      [x],
      // The harness default, and it is not luck. The kernel sums the k gates in
      // f32 and divides by that sum, so `sum(g) * x` returns x to within a
      // rounding step rather than exactly. Measured by running this test with
      // the tolerance set to zero: worst relative difference 1.76e-7 (absolute
      // 4.77e-7) at element 818, about one and a half f32 epsilons. The default
      // `rel` of 1e-5 is 57x that, and still five orders of magnitude below the
      // ~0.2 relative that a squared gate costs here.
    );

    // Nothing was dropped, so every slot found a seat — otherwise the identity
    // above would hold for a reason other than the one being tested.
    expect(Array.from(pos).every((p) => p >= 0)).toBe(true);
  });

  gpuTest("drops tokens over capacity and gives them zero rows", async (run) => {
    const { shape, logits, x, dropped, expected } = CROWDED;
    const { T, k, E, C, D } = shape;
    // A token that lost every seat comes back as zeros, and there is at least
    // one of them here — otherwise this test would be the previous one.
    expect(dropped.length).toBeGreaterThan(0);

    const [expert, gate] = (await run({
      code: routerCode,
      bindings: [
        { kind: "storage", data: logits },
        { kind: "out", type: "i32", length: T * k },
        { kind: "out", type: "f32", length: T * k },
        { kind: "uniform", data: routerParams(T, E, k, true) },
      ],
      workgroups: tokenThreads(T),
    })) as [Int32Array, Float32Array];

    const [buffer, pos] = (await run({
      code: dispatchCode,
      bindings: [
        { kind: "storage", data: x },
        { kind: "storage", data: expert },
        { kind: "out", type: "f32", length: E * C * D },
        { kind: "out", type: "i32", length: T * k },
        { kind: "uniform", data: moeParams(T, k, E, C, D) },
      ],
      workgroups: [T * k],
    })) as [Float32Array, Int32Array];

    await expectAgrees(
      run,
      {
        code: gatherCode,
        bindings: [
          { kind: "storage", data: buffer },
          { kind: "storage", data: expert },
          { kind: "storage", data: pos },
          { kind: "storage", data: gate },
          { kind: "out", type: "f32", length: T * D },
          { kind: "uniform", data: moeParams(T, k, E, C, D) },
        ],
        workgroups: [T],
      },
      [expected],
    );
  });

  gpuTest("an expert that receives no tokens produces no NaN", async (run) => {
    // Not a corner case: with 64 experts and a short sequence most of them get
    // nothing, every step. A kernel that averaged over an expert's token count
    // would return 0/0 here for three of the four experts, and NaN survives
    // every downstream op silently.
    const { shape, logits, x } = EMPTY_EXPERTS;
    const { T, k, E, C, D } = shape;

    const [expert, gate] = (await run({
      code: routerCode,
      bindings: [
        { kind: "storage", data: logits },
        { kind: "out", type: "i32", length: T * k },
        { kind: "out", type: "f32", length: T * k },
        { kind: "uniform", data: routerParams(T, E, k, true) },
      ],
      workgroups: tokenThreads(T),
    })) as [Int32Array, Float32Array];

    const [buffer, pos] = (await run({
      code: dispatchCode,
      bindings: [
        { kind: "storage", data: x },
        { kind: "storage", data: expert },
        { kind: "out", type: "f32", length: E * C * D },
        { kind: "out", type: "i32", length: T * k },
        { kind: "uniform", data: moeParams(T, k, E, C, D) },
      ],
      workgroups: [T * k],
    })) as [Float32Array, Int32Array];

    // k = 1 normalised makes the gate exactly 1, so the identity expert gives x
    // back: the three empty experts changed nothing rather than poisoning it.
    await expectAgrees(
      run,
      {
        code: gatherCode,
        bindings: [
          { kind: "storage", data: buffer },
          { kind: "storage", data: expert },
          { kind: "storage", data: pos },
          { kind: "storage", data: gate },
          { kind: "out", type: "f32", length: T * D },
          { kind: "uniform", data: moeParams(T, k, E, C, D) },
        ],
        workgroups: [T],
      },
      [x],
    );

    expect(Array.from(expert).every((e) => e === 0)).toBe(true);
    expect(Array.from(gate).every(Number.isFinite)).toBe(true);
    expect(Array.from(buffer).every(Number.isFinite)).toBe(true);
  });
});
