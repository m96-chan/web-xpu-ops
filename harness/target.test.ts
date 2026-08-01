import { describe, expect, it } from "vitest";
import { TARGETS, describeAdapter, detectTarget } from "./target.js";

/**
 * Measured, not invented. `node probe` against this repository's own adapter
 * (Linux / Dawn `webgpu@0.4.x`) printed exactly this, and it is the only
 * `adapter.info` any of these rules has been checked against on real hardware.
 * Every other case below is a rule about vendor strings, which is why detection
 * is documented as a hint and why the override exists.
 */
const MEASURED = {
  vendor: "nvidia",
  architecture: "blackwell",
  device: "nvidia-geforce-rtx-5090",
  description: "NVIDIA: 610.43.02 610.43.2.0",
  isFallbackAdapter: false,
};

describe("detectTarget", () => {
  it("recognises the adapter this repository was developed on", () => {
    expect(detectTarget(MEASURED).target).toBe("nvidia");
  });

  it.each([
    ["amd", "amd"],
    ["AMD Radeon", "amd"],
    ["apple", "apple"],
    ["qualcomm", "soc"],
    ["arm", "soc"],
  ])("maps vendor %j to %j", (vendor, target) => {
    expect(detectTarget({ ...MEASURED, vendor }).target).toBe(target);
  });

  // Intel is the honest case. The same vendor string covers an on-die iGPU with
  // a handful of watts and a discrete Arc card, and `adapter.info` does not say
  // which. Guessing `soc` would be wrong half the time and invisible both ways.
  it("refuses to guess for a vendor that spans wildly different hardware", () => {
    const detected = detectTarget({ ...MEASURED, vendor: "intel" });
    expect(detected.target).toBeNull();
    expect(detected.why).toMatch(/intel/i);
  });

  // Whole tokens of the vendor string, so a driver that spells itself out still
  // lands, and a substring collision does not decide anything.
  it("matches a vendor token inside a longer string", () => {
    expect(detectTarget({ ...MEASURED, vendor: "NVIDIA Corporation" }).target).toBe("nvidia");
  });

  it("gives up on a vendor it does not know", () => {
    expect(detectTarget({ ...MEASURED, vendor: "vivante" }).target).toBeNull();
  });

  it("gives up when the vendor string is empty", () => {
    expect(detectTarget({ ...MEASURED, vendor: "" }).target).toBeNull();
  });

  // A software fallback runs none of the silicon the vendor string names, so a
  // kernel tuned for that silicon is precisely the wrong choice.
  it("ignores the vendor entirely on a fallback adapter", () => {
    const detected = detectTarget({ ...MEASURED, isFallbackAdapter: true });
    expect(detected.target).toBeNull();
    expect(detected.why).toMatch(/fallback/i);
  });

  // The spec says not to parse `description`, and drivers put version numbers in
  // it. A rule that reads it would fire on driver strings that mention a vendor
  // the adapter is not.
  it("never decides from the description", () => {
    const detected = detectTarget({
      vendor: "",
      architecture: "",
      device: "",
      description: "AMD Radeon Graphics (RADV)",
      isFallbackAdapter: false,
    });
    expect(detected.target).toBeNull();
  });

  it("always says why, whatever it decided", () => {
    for (const vendor of ["nvidia", "intel", "", "wat"]) {
      expect(detectTarget({ ...MEASURED, vendor }).why.length).toBeGreaterThan(0);
    }
  });

  it("only ever returns a target the tree can hold a kernel for", () => {
    for (const vendor of ["nvidia", "amd", "apple", "qualcomm", "arm", "intel", ""]) {
      const { target } = detectTarget({ ...MEASURED, vendor });
      if (target !== null) expect(TARGETS).toContain(target);
    }
  });
});

describe("describeAdapter", () => {
  const fake = (features: string[], info = MEASURED) => ({
    info: { ...info, subgroupMinSize: 32, subgroupMaxSize: 64 },
    features: new Set(features),
  });

  it("reports f16 as absent when the feature is not there", () => {
    expect(describeAdapter(fake(["subgroups"])).features.f16).toBe(false);
  });

  it("reports f16 as present when the feature is there", () => {
    expect(describeAdapter(fake(["shader-f16"])).features.f16).toBe(true);
  });

  it("reports the subgroup size only when subgroups exist", () => {
    expect(describeAdapter(fake(["subgroups"])).subgroupSize).toEqual({ min: 32, max: 64 });
    expect(describeAdapter(fake([])).subgroupSize).toBeNull();
  });

  it("carries the detected target and its reason", () => {
    const report = describeAdapter(fake(["subgroups"]));
    expect(report.target).toBe("nvidia");
    expect(report.why.length).toBeGreaterThan(0);
  });

  // `GPUAdapterInfo` exposes its fields as prototype getters — `Object.keys` on
  // the real one returns `["subgroupMatrixConfigs"]` and nothing else, so a
  // spread or a `JSON.stringify` of it silently loses everything. Measured.
  it("copies the info out field by field, so it survives serialisation", () => {
    const info = Object.create(
      Object.defineProperties(
        {},
        Object.fromEntries(
          Object.entries(MEASURED).map(([k, v]) => [k, { get: () => v, enumerable: false }]),
        ),
      ),
    );
    const report = describeAdapter({ info, features: new Set<string>() });
    expect(JSON.parse(JSON.stringify(report.info))).toEqual(MEASURED);
  });
});
