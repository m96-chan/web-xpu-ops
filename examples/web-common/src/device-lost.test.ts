/**
 * The device-loss wording, which is the whole point of the module.
 *
 * Issue #211. A page reported `OperationError: Instance dropped in
 * popErrorScope` five times with five different stacks, and none of them was
 * the fault. These assertions are about a *message*, which is unusual here —
 * but the message is the deliverable: it is the difference between "out of
 * memory" and "a pipeline the adapter refused", and getting from one to the
 * other cost a model shrunk to four blocks.
 */
import { describe, expect, it } from "vitest";
import { explainFailure, isInstanceDropped } from "./device-lost.js";

const context = { allocated: 23.1e9, buffers: 912 };

describe("web-common / device loss", () => {
  it("recognises the message every call produces after a loss", () => {
    expect(isInstanceDropped(new Error("OperationError: Instance dropped in popErrorScope"))).toBe(true);
    expect(isInstanceDropped(new Error("Instance dropped error in getCompilationInfo"))).toBe(true);
    expect(isInstanceDropped("Instance Dropped")).toBe(true);
    // Not everything that fails after a loss says it, and not everything that
    // says something similar is one.
    expect(isInstanceDropped(new Error("out of GPU memory allocating 512 MB"))).toBe(false);
    expect(isInstanceDropped(new Error("pipeline is not valid: workgroup_size(512)"))).toBe(false);
  });

  it("replaces the useless message with the reason, once the reason is known", () => {
    const message = explainFailure(
      new Error("OperationError: Instance dropped in popErrorScope"),
      { reason: "unknown", message: "Device lost: Vulkan device out of memory" },
      context,
      "allocating scratch",
    ).message;
    expect(message).toMatch(/Vulkan device out of memory/);
    expect(message).toMatch(/23\.10 GB allocated across 912 buffers/);
    expect(message).toMatch(/allocating scratch/);
    // The original said nothing, so it does not survive.
    expect(message).not.toMatch(/Instance dropped/);
  });

  it("says the device is gone but silent, when the loss has not resolved yet", () => {
    // The ordinary race: the rejections arrive before `device.lost` does.
    const message = explainFailure(
      new Error("Instance dropped error in getCompilationInfo"), null, context, "building a pipeline",
    ).message;
    expect(message).toMatch(/has not yet said why/);
    expect(message).toMatch(/not a cause/);
    expect(message).toMatch(/23\.10 GB/);
  });

  it("keeps an error that is the fault rather than a consequence", () => {
    // A pipeline the adapter refused is the fault, and the wording that helped
    // here — `workgroup_size(512)` against a 256 limit — has to survive.
    const message = explainFailure(
      new Error("pipeline is not valid: Entry-point uses workgroup_size(512, 1, 1) that exceeds the maximum allowed (256, 256, 64)"),
      null, context, "building a pipeline",
    ).message;
    expect(message).toMatch(/workgroup_size\(512, 1, 1\)/);
    expect(message).toMatch(/maximum allowed \(256, 256, 64\)/);
    expect(message).toMatch(/23\.10 GB allocated/);
  });

  it("survives something thrown that is not an Error", () => {
    expect(explainFailure("a string", null, context, "doing a thing").message).toMatch(/a string/);
  });

  it("says so when the backend gives no message", () => {
    // `reason` alone is one of two words and neither is a diagnosis, so an
    // empty `message` has to read as absent rather than as an explanation.
    const message = explainFailure(new Error("Instance dropped"), { reason: "destroyed", message: "" }, context, "x").message;
    expect(message).toMatch(/destroyed/);
    expect(message).toMatch(/no message from the backend/);
  });
});
