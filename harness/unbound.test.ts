import { beforeAll, describe, expect, it } from "vitest";
import { createRunner, type Dispatch, type Runner } from "./wgsl.js";

/**
 * A binding the shader never mentions must not read back as a pass.
 *
 * Output buffers start zeroed, so a dispatch that never ran reads back as zeros
 * — and a test whose expected value contains zeros then passes over a kernel
 * that executed nothing. Every correctness claim in this repository is "it
 * agrees with the reference", so a silent no-op reading as agreement is the one
 * failure that could hollow out all of them at once. Issue #46.
 *
 * ONE provoked error per file, and the device is never destroyed. Both are
 * measured, not stylistic: two provoked device errors in a single process
 * crashed this binding in 4 of 5 runs, and removing the destroy alone made it
 * 5 of 5. A process that provokes exactly one error and then exits is stable.
 */

let runner: Runner | null = null;

/**
 * `expect(...).rejects` is avoided deliberately: on this binding it takes the
 * worker down and reports nothing, while a plain try/catch reports normally.
 */
async function errorFrom(dispatch: Dispatch): Promise<string> {
  try {
    await runner!.run(dispatch);
    return "";
  } catch (error) {
    return String(error);
  }
}

describe("harness / a binding the shader never references", () => {
  beforeAll(async () => {
    runner = await createRunner();
  }, 60_000);

  it("is reported instead of reading back zeros", async () => {
    if (!runner) return;
    // `unused` is bound but never mentioned, so `layout: "auto"` omits it and
    // the bind group fails validation. Not hypothetical: it is why `attention`
    // is two WGSL files rather than one file with two entry points.
    const message = await errorFrom({
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
});
