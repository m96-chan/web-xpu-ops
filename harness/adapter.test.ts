import { beforeAll, describe, expect, it } from "vitest";
import { create, globals } from "webgpu";
import { PORTABLE, TARGETS, describeAdapter, resolve } from "./index.js";

/**
 * The one part of resolution that cannot be checked with a fake: that a real
 * `adapter.info` survives being read.
 *
 * This file asks for an adapter and stops there — no device, no dispatch — which
 * is why it does not use `useGpu`. Describing an adapter is a question about the
 * adapter, and `createRunner` is left to its own job of running a kernel.
 *
 * Nothing here asserts a particular vendor. The suite has to pass on whatever
 * hardware it runs on, so these are the claims that hold for any adapter.
 */
describe("adapter report / gpu", () => {
  let adapter: GPUAdapter | null = null;

  beforeAll(async () => {
    Object.assign(globalThis, globals);
    const gpu = create([]) as GPU;
    adapter = await gpu.requestAdapter();
  }, 60_000);

  it("reads a real adapter without losing any of its fields", () => {
    if (!adapter) return;
    const { info } = describeAdapter(adapter);
    // A spread or a JSON round-trip of the raw GPUAdapterInfo drops everything,
    // because its fields are prototype getters. This is the check that the
    // report copied them out rather than aliasing the object.
    expect(Object.keys(JSON.parse(JSON.stringify(info))).sort()).toEqual([
      "architecture",
      "description",
      "device",
      "isFallbackAdapter",
      "vendor",
    ]);
    expect(typeof info.vendor).toBe("string");
    expect(typeof info.isFallbackAdapter).toBe("boolean");
  });

  it("says which target it picked and why", () => {
    if (!adapter) return;
    const { target, why, info, features, subgroupSize } = describeAdapter(adapter);
    if (target !== null) expect(TARGETS).toContain(target);
    expect(why.length).toBeGreaterThan(0);
    console.log(
      `[adapter] vendor=${JSON.stringify(info.vendor)} arch=${JSON.stringify(info.architecture)} ` +
        `target=${target ?? "portable"} (${why}) f16=${features.f16} subgroups=${features.subgroups}`,
    );
    // The subgroup size is only defined when the feature is there; reporting a
    // number without it would be reporting a guess. The converse does not hold —
    // `subgroupMinSize` is still optional in the WebGPU types, so a device with
    // the feature may not publish a size.
    if (!features.subgroups) expect(subgroupSize).toBeNull();
    if (subgroupSize) expect(subgroupSize.min).toBeLessThanOrEqual(subgroupSize.max);
  });

  it("resolves to the portable kernel on this machine, because no op ships a variant yet", () => {
    if (!adapter) return;
    const report = describeAdapter(adapter);
    const choice = resolve({
      have: [PORTABLE],
      target: report.target,
      dtype: report.features.f16 ? "f16" : "f32",
    });
    expect(choice.name).toBe(PORTABLE);
    expect(choice.rung).toBe("portable");
  });
});
