import { afterAll, beforeAll, describe, expect, it } from "vitest";
import { createRunner, type Dispatch, type Runner } from "./wgsl.js";

/**
 * The harness has to fail loudly when a dispatch does not happen.
 *
 * Output buffers start zeroed, so a dispatch that never ran reads back as zeros
 * — and a test whose expected value contains zeros then passes over a kernel
 * that executed nothing. That is worse than a wrong kernel: rule 8 says an
 * incorrect kernel has negative value, and one that never ran while reporting
 * green undermines every other test here at once, because every correctness
 * claim in this repository is "it agrees with the reference".
 *
 * Two ways in, both measured on this device before the fix (issue #46):
 *
 *   - a shader that does not compile — read back `[0, 0]`, no error anywhere
 *   - a binding the shader never mentions — `layout: "auto"` drops it from the
 *     layout, the bind group then fails validation, and the readback is zeros
 *
 * The second is not hypothetical: `attention` is two WGSL files rather than one
 * file with two entry points precisely because of it.
 */

let runner: Runner | null = null;

/**
 * `expect(...).rejects` is avoided deliberately: on this binding it takes the
 * vitest worker down with it and reports nothing, while a plain try/catch around
 * the same call reports normally. Measured, not assumed.
 */
async function errorFrom(dispatch: Dispatch): Promise<string> {
  try {
    await runner!.run(dispatch);
    return "";
  } catch (error) {
    return String(error);
  }
}

describe("harness / error reporting", () => {
  // The device is acquired in a hook rather than a test of its own. As an `it()`
  // it was skipped whenever the file ran under `-t`, leaving the runner null so
  // every other test returned early and passed — green while proving nothing.
  beforeAll(async () => {
    runner = await createRunner();
  }, 60_000);
  afterAll(() => {
    runner?.destroy();
    runner = null;
  });

  it("reports a shader that does not compile", async () => {
    if (!runner) return;
    const message = await errorFrom({
      code: `
        @group(0) @binding(0) var<storage, read> input: array<f32>;
        @group(0) @binding(1) var<storage, read_write> output: array<f32>;
        @compute @workgroup_size(1)
        fn main() { output[0] = input[0] + notAFunction(1.0); }
      `,
      bindings: [
        { kind: "storage", data: Float32Array.from([42, 43]) },
        { kind: "out", type: "f32", length: 2 },
      ],
      workgroups: [1],
    });
    expect(message, "a shader that does not compile read back as a pass").toMatch(/compile/i);
  });

  it("reports a declared binding the shader never references", async () => {
    if (!runner) return;
    const message = await errorFrom({
      // `unused` is bound but never mentioned, so `layout: "auto"` omits it.
      code: `
        @group(0) @binding(0) var<storage, read> input: array<f32>;
        @group(0) @binding(1) var<storage, read_write> output: array<f32>;
        @group(0) @binding(2) var<storage, read> unused: array<f32>;
        @compute @workgroup_size(1)
        fn main() { output[0] = input[0]; }
      `,
      bindings: [
        { kind: "storage", data: Float32Array.from([42]) },
        { kind: "out", type: "f32", length: 1 },
        { kind: "storage", data: Float32Array.from([7]) },
      ],
      workgroups: [1],
    });
    expect(message, "a dropped binding read back as a pass").toMatch(/not valid/i);
  });

  it("still runs a shader that is fine", async () => {
    if (!runner) return;
    const [out] = await runner.run({
      code: `
        @group(0) @binding(0) var<storage, read> input: array<f32>;
        @group(0) @binding(1) var<storage, read_write> output: array<f32>;
        @compute @workgroup_size(1)
        fn main() { output[0] = input[0] * 2.0; }
      `,
      bindings: [
        { kind: "storage", data: Float32Array.from([21]) },
        { kind: "out", type: "f32", length: 1 },
      ],
      workgroups: [1],
    });
    expect(Array.from(out!)).toEqual([42]);
  });
});
