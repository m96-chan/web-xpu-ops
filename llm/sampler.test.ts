import { describe, expect, it } from "vitest";
import { sampleNext, type Constraint, type SamplerOptions } from "./sampler.js";

/** A constraint that always allows exactly the given ids, ignoring the prefix. */
function fixedConstraint(...allowed: number[]): Constraint {
  const set = new Set(allowed);
  return { nextAllowed: () => set };
}

/** A deterministic "random" source: returns the next value from a fixed queue. */
function queueRng(...values: number[]): () => number {
  let i = 0;
  return () => {
    const v = values[i]!;
    i = Math.min(i + 1, values.length - 1);
    return v;
  };
}

describe("sampleNext / greedy", () => {
  it("picks the argmax when unconstrained", () => {
    const logits = [0.1, 5.0, -3.0, 2.0];
    const options: SamplerOptions = { mode: "greedy" };
    expect(sampleNext(logits, [], options)).toBe(1);
  });

  it("picks the argmax among only the allowed ids when constrained", () => {
    // Index 1 is the global argmax, but it is masked out — the winner among
    // {0, 2, 3} is index 3 (2.0).
    const logits = [0.1, 5.0, -3.0, 2.0];
    const options: SamplerOptions = { mode: "greedy" };
    const constraint = fixedConstraint(0, 2, 3);
    expect(sampleNext(logits, [], options, constraint)).toBe(3);
  });

  it("throws when the constraint allows no token at all", () => {
    const logits = [0.1, 5.0, -3.0, 2.0];
    const options: SamplerOptions = { mode: "greedy" };
    const constraint: Constraint = { nextAllowed: () => new Set() };
    expect(() => sampleNext(logits, [0], options, constraint)).toThrow(/no token/);
  });

  it("passes the prefix through to the constraint unmodified", () => {
    const logits = [1, 2, 3];
    const seen: number[][] = [];
    const constraint: Constraint = {
      nextAllowed: (prefix) => {
        seen.push([...prefix]);
        return null;
      },
    };
    sampleNext(logits, [7, 8, 9], { mode: "greedy" }, constraint);
    expect(seen).toEqual([[7, 8, 9]]);
  });
});

describe("sampleNext / temperature + top-p", () => {
  it("restricts the draw to the nucleus: a low top-p ignores rng values that would only be reachable by including long-tail tokens", () => {
    // One dominant logit and three flat, tiny ones. With topP=0.05 only the
    // dominant token is in the nucleus, so even an rng draw near 1 must still
    // return it.
    const logits = [10, -10, -10, -10];
    const options: SamplerOptions = { mode: "top-p", temperature: 1, topP: 0.05, rng: queueRng(0.9999) };
    expect(sampleNext(logits, [], options)).toBe(0);
  });

  it("an rng draw of 0 always picks the highest-probability token in the nucleus", () => {
    const logits = [1, 3, 2];
    const options: SamplerOptions = { mode: "top-p", temperature: 1, topP: 1, rng: queueRng(0) };
    expect(sampleNext(logits, [], options)).toBe(1);
  });

  it("a wide top-p and an rng draw near 1 can reach the lowest-probability token", () => {
    const logits = [1, 3, 2];
    const options: SamplerOptions = { mode: "top-p", temperature: 1, topP: 1, rng: queueRng(0.999999) };
    expect(sampleNext(logits, [], options)).toBe(0);
  });

  it("respects the constraint mask: an excluded token is never drawn even at rng=0 favoring it", () => {
    const logits = [10, 1, 1, 1]; // index 0 dominates
    const constraint = fixedConstraint(1, 2, 3); // index 0 excluded
    const options: SamplerOptions = { mode: "top-p", temperature: 1, topP: 1, rng: queueRng(0) };
    const choice = sampleNext(logits, [], options, constraint);
    expect(choice).not.toBe(0);
  });

  it("rejects a non-positive temperature", () => {
    const options: SamplerOptions = { mode: "top-p", temperature: 0, topP: 1 };
    expect(() => sampleNext([1, 2, 3], [], options)).toThrow(/temperature/);
  });

  it("rejects a top-p outside (0, 1]", () => {
    const options: SamplerOptions = { mode: "top-p", temperature: 1, topP: 0 };
    expect(() => sampleNext([1, 2, 3], [], options)).toThrow(/topP/);
  });

  it("has no finite logit to choose from when every logit is -Infinity", () => {
    const options: SamplerOptions = { mode: "top-p", temperature: 1, topP: 1 };
    expect(() => sampleNext([-Infinity, -Infinity], [], options)).toThrow(/no finite logit/);
  });
});

describe("sampleNext / no finite logit", () => {
  it("throws in greedy mode when every logit is -Infinity", () => {
    const options: SamplerOptions = { mode: "greedy" };
    expect(() => sampleNext([-Infinity, -Infinity], [], options)).toThrow(/no finite logit/);
  });
});
