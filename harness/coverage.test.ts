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

/** A test that loops one named entry point's variants. */
const looped = (entry: string) =>
  `eachVariant(import.meta.url, ${JSON.stringify(entry)}, ({ code }) => {\n` +
  `  gpuTest("agrees", async () => {});\n});`;
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
      "beta/wgsl/kernel.nvidia.wgsl": "",
      "beta/wgsl.test.ts": FIXED,
    });
    expect(unguardedOps(root)).toEqual(["beta"]);
  });

  it("is satisfied by a test that iterates the variants", () => {
    const root = opsTree({
      "gamma/wgsl/kernel.wgsl": "",
      "gamma/wgsl/kernel.nvidia.wgsl": "",
      "gamma/wgsl.test.ts": looped("kernel"),
    });
    expect(unguardedOps(root)).toEqual([]);
  });

  it("catches an op that has variants and no test at all", () => {
    const root = opsTree({ "delta/wgsl/kernel.wgsl": "", "delta/wgsl/kernel.amd.wgsl": "" });
    expect(unguardedOps(root)).toEqual(["delta"]);
  });

  it("accepts the loop from any test file in the op, not just wgsl.test.ts", () => {
    const root = opsTree({
      "epsilon/wgsl/kernel.wgsl": "",
      "epsilon/wgsl/kernel.amd.wgsl": "",
      "epsilon/wgsl.test.ts": FIXED,
      "epsilon/variants.test.ts": looped("kernel"),
    });
    expect(unguardedOps(root)).toEqual([]);
  });

  it("reports every offender, sorted", () => {
    const root = opsTree({
      "zeta/wgsl/kernel.wgsl": "",
      "zeta/wgsl/kernel.amd.wgsl": "",
      "eta/wgsl/kernel.wgsl": "",
      "eta/wgsl/kernel.apple.wgsl": "",
      "theta/wgsl/kernel.wgsl": "",
      "theta/wgsl.test.ts": FIXED,
    });
    expect(unguardedOps(root)).toEqual(["eta", "zeta"]);
  });

  /**
   * Ops with more than one entry point are the case that broke the first version
   * of this. `stft` has `kernel` and `inverse`; `attention` has `scores` and
   * `context`. A loop over one of them says nothing about the other.
   */
  describe("ops with more than one entry point", () => {
    it("says nothing when neither entry point has a variant", () => {
      const root = opsTree({
        "stftish/wgsl/kernel.wgsl": "",
        "stftish/wgsl/inverse.wgsl": "",
        "stftish/wgsl.test.ts": FIXED,
      });
      expect(unguardedOps(root)).toEqual([]);
    });

    it("catches a variant on the entry point the tests do not loop", () => {
      const root = opsTree({
        "stftish/wgsl/kernel.wgsl": "",
        "stftish/wgsl/kernel.amd.wgsl": "",
        "stftish/wgsl/inverse.wgsl": "",
        "stftish/wgsl/inverse.amd.wgsl": "",
        "stftish/wgsl.test.ts": looped("kernel"),
      });
      expect(unguardedOps(root)).toEqual(["stftish"]);
    });

    it("is satisfied once every entry point that has variants is looped", () => {
      const root = opsTree({
        "stftish/wgsl/kernel.wgsl": "",
        "stftish/wgsl/kernel.amd.wgsl": "",
        "stftish/wgsl/inverse.wgsl": "",
        "stftish/wgsl/inverse.amd.wgsl": "",
        "stftish/wgsl.test.ts": `${looped("kernel")}\n${looped("inverse")}`,
      });
      expect(unguardedOps(root)).toEqual([]);
    });

    it("ignores an entry point that has no variants to skip", () => {
      const root = opsTree({
        "attentionish/wgsl/scores.wgsl": "",
        "attentionish/wgsl/scores.amd.wgsl": "",
        "attentionish/wgsl/context.wgsl": "",
        "attentionish/wgsl.test.ts": looped("scores"),
      });
      expect(unguardedOps(root)).toEqual([]);
    });
  });

  // Runs against the real tree. Vacuous today — no op has a variant yet — and
  // that is the point: it starts failing the moment the first one lands without
  // a loop, rather than after someone ships a wrong kernel.
  it("passes for this repository", () => {
    expect(unguardedOps(new URL("../ops/", import.meta.url))).toEqual([]);
  });
});
