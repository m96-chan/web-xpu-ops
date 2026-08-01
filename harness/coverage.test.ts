import { mkdirSync, mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterAll, describe, expect, it } from "vitest";
import { unguardedOps } from "./coverage.js";

const roots: string[] = [];
function opsTree(files: Record<string, string>): string {
  const root = mkdtempSync(join(tmpdir(), "web-xpu-ops-coverage-"));
  roots.push(root);
  for (const [path, contents] of Object.entries(files)) {
    const full = join(root, path);
    mkdirSync(join(full, ".."), { recursive: true });
    writeFileSync(full, contents);
  }
  return root;
}
afterAll(() => {
  for (const root of roots) rmSync(root, { recursive: true, force: true });
});

const LOOPED = 'eachVariant(import.meta.url, ({ code }) => {\n  gpuTest("agrees", async () => {});\n});';
const FIXED = 'const code = kernel(import.meta.url);\ngpuTest("agrees", async () => {});';

describe("unguardedOps", () => {
  it("says nothing about an op that only has the portable kernel", () => {
    const root = opsTree({ "alpha/wgsl/kernel.wgsl": "", "alpha/wgsl.test.ts": FIXED });
    expect(unguardedOps(root)).toEqual([]);
  });

  // The failure this whole mechanism exists for: a target-specific kernel that
  // is fast and wrong, sitting in the tree while the suite only ever compiles
  // the portable one and stays green.
  it("catches a variant added beside a test that only loads the portable kernel", () => {
    const root = opsTree({
      "beta/wgsl/kernel.wgsl": "",
      "beta/wgsl/nvidia.wgsl": "",
      "beta/wgsl.test.ts": FIXED,
    });
    expect(unguardedOps(root)).toEqual(["beta"]);
  });

  it("is satisfied by a test that iterates the variants", () => {
    const root = opsTree({
      "gamma/wgsl/kernel.wgsl": "",
      "gamma/wgsl/nvidia.wgsl": "",
      "gamma/wgsl.test.ts": LOOPED,
    });
    expect(unguardedOps(root)).toEqual([]);
  });

  it("catches an op that has variants and no test at all", () => {
    const root = opsTree({ "delta/wgsl/kernel.wgsl": "", "delta/wgsl/amd.wgsl": "" });
    expect(unguardedOps(root)).toEqual(["delta"]);
  });

  it("accepts the loop from any test file in the op, not just wgsl.test.ts", () => {
    const root = opsTree({
      "epsilon/wgsl/kernel.wgsl": "",
      "epsilon/wgsl/amd.wgsl": "",
      "epsilon/wgsl.test.ts": FIXED,
      "epsilon/variants.test.ts": LOOPED,
    });
    expect(unguardedOps(root)).toEqual([]);
  });

  it("reports every offender, sorted", () => {
    const root = opsTree({
      "zeta/wgsl/kernel.wgsl": "",
      "zeta/wgsl/amd.wgsl": "",
      "eta/wgsl/kernel.wgsl": "",
      "eta/wgsl/apple.wgsl": "",
      "theta/wgsl/kernel.wgsl": "",
      "theta/wgsl.test.ts": FIXED,
    });
    expect(unguardedOps(root)).toEqual(["eta", "zeta"]);
  });

  // Runs against the real tree. Vacuous today — no op has a variant yet — and
  // that is the point: it starts failing the moment the first one lands without
  // a loop, rather than after someone ships a wrong kernel.
  it("passes for this repository", () => {
    expect(unguardedOps(new URL("../ops/", import.meta.url))).toEqual([]);
  });
});
