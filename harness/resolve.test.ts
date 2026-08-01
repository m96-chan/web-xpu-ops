import { mkdirSync, mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { pathToFileURL } from "node:url";
import { afterAll, describe, expect, it } from "vitest";
import { PORTABLE, parseVariant, resolve, variantSuites, variantsIn } from "./resolve.js";
import { eachVariant } from "./suite.js";

/**
 * The targets here are deliberately fake.
 *
 * Resolution is string joining and first-hit-wins; it has no opinion about which
 * targets exist. Testing it with `nvidia` would let a bug that hardcodes a real
 * vendor name pass, so nothing in this file names a real one.
 */
const FAKE = ["frobnitz", "widget"] as const;

const roots: string[] = [];
function tree(files: Record<string, string>): string {
  const root = mkdtempSync(join(tmpdir(), "web-xpu-ops-variants-"));
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

describe("resolve / order", () => {
  const all = ["frobnitz.f16", "frobnitz", "widget", "f16", PORTABLE];

  it("takes target + dtype first", () => {
    const choice = resolve({ have: all, target: "frobnitz", dtype: "f16" });
    expect(choice.name).toBe("frobnitz.f16");
    expect(choice.rung).toBe("target+dtype");
  });

  it("falls back to the target when no target+dtype variant exists", () => {
    const choice = resolve({ have: ["frobnitz", "f16", PORTABLE], target: "frobnitz", dtype: "f16" });
    expect(choice.name).toBe("frobnitz");
    expect(choice.rung).toBe("target");
  });

  it("falls back to the dtype when the target has no variant at all", () => {
    const choice = resolve({ have: ["widget", "f16", PORTABLE], target: "frobnitz", dtype: "f16" });
    expect(choice.name).toBe("f16");
    expect(choice.rung).toBe("dtype");
  });

  it("falls back to portable when nothing else hits", () => {
    const choice = resolve({ have: ["widget", PORTABLE], target: "frobnitz", dtype: "f16" });
    expect(choice.name).toBe(PORTABLE);
    expect(choice.rung).toBe("portable");
  });

  // The whole reason detection is allowed to return null: an unrecognised
  // adapter must get the portable kernel, never someone else's.
  it("takes portable when the target is unknown, even though variants exist", () => {
    const choice = resolve({ have: all, target: null, dtype: null });
    expect(choice.name).toBe(PORTABLE);
    expect(choice.rung).toBe("portable");
  });

  it("still uses the dtype when the target is unknown", () => {
    const choice = resolve({ have: all, target: null, dtype: "f16" });
    expect(choice.name).toBe("f16");
    expect(choice.rung).toBe("dtype");
  });

  it("steers on the target: the same variant set resolves differently", () => {
    expect(resolve({ have: all, target: "frobnitz", dtype: null }).name).toBe("frobnitz");
    expect(resolve({ have: all, target: "widget", dtype: null }).name).toBe("widget");
  });

  it("reports the whole chain it walked, in order", () => {
    const choice = resolve({ have: [PORTABLE], target: "frobnitz", dtype: "f16" });
    expect(choice.tried).toEqual(["frobnitz.f16", "frobnitz", "f16", PORTABLE]);
  });

  it("throws when the op has no portable kernel", () => {
    expect(() => resolve({ have: ["frobnitz"], target: null, dtype: null })).toThrow(/portable/);
  });
});

describe("resolve / override", () => {
  const all = ["frobnitz.f16", "frobnitz", "widget", "f16", PORTABLE];

  it("beats a detected target", () => {
    const choice = resolve({ have: all, target: "frobnitz", dtype: null, override: "widget" });
    expect(choice.name).toBe("widget");
    expect(choice.rung).toBe("override");
  });

  it("beats a target+dtype hit", () => {
    const choice = resolve({ have: all, target: "frobnitz", dtype: "f16", override: PORTABLE });
    expect(choice.name).toBe(PORTABLE);
    expect(choice.rung).toBe("override");
  });

  it("skips the chain entirely rather than walking past itself", () => {
    const choice = resolve({ have: all, target: "frobnitz", dtype: "f16", override: "widget" });
    expect(choice.tried).toEqual(["widget"]);
  });

  // Falling through to portable on a typo would hand the caller a kernel they
  // did not ask for and never tell them. That is the invisible wrong choice.
  it("throws when it names a variant that does not exist", () => {
    expect(() => resolve({ have: all, target: "frobnitz", dtype: null, override: "widgit" })).toThrow(
      /widgit/,
    );
  });
});

describe("parseVariant", () => {
  it("reads the portable kernel", () => {
    expect(parseVariant("kernel.wgsl")).toEqual({ name: "kernel", target: null, dtype: null });
  });

  it("reads a target variant", () => {
    expect(parseVariant("nvidia.wgsl")).toEqual({ name: "nvidia", target: "nvidia", dtype: null });
  });

  it("reads a dtype variant", () => {
    expect(parseVariant("f16.wgsl")).toEqual({ name: "f16", target: null, dtype: "f16" });
  });

  it("reads a target+dtype variant", () => {
    expect(parseVariant("apple.f16.wgsl")).toEqual({
      name: "apple.f16",
      target: "apple",
      dtype: "f16",
    });
  });

  it("takes a target vocabulary, so resolution can be exercised without real vendors", () => {
    expect(parseVariant("frobnitz.f16.wgsl", FAKE)).toEqual({
      name: "frobnitz.f16",
      target: "frobnitz",
      dtype: "f16",
    });
  });

  // A misspelt variant is worse than a missing one: it sits in the tree looking
  // tuned while resolution never reaches it.
  it("throws on a misspelt target", () => {
    expect(() => parseVariant("nvidai.wgsl")).toThrow(/nvidai/);
  });

  it("throws on a misspelt dtype", () => {
    expect(() => parseVariant("nvidia.fp16.wgsl")).toThrow(/fp16/);
  });

  it("throws when target and dtype are the wrong way round", () => {
    expect(() => parseVariant("f16.nvidia.wgsl")).toThrow();
  });

  it("throws on a file that is not wgsl", () => {
    expect(() => parseVariant("nvidia.txt")).toThrow();
  });
});

describe("variantsIn", () => {
  it("discovers every variant on disk, sorted, ignoring anything else", () => {
    const root = tree({
      "wgsl/kernel.wgsl": "",
      "wgsl/nvidia.wgsl": "",
      "wgsl/nvidia.f16.wgsl": "",
      "wgsl/NOTES.md": "not a kernel",
    });
    expect(variantsIn(join(root, "wgsl")).map((v) => v.name)).toEqual([
      "kernel",
      "nvidia",
      "nvidia.f16",
    ]);
  });

  it("names the offending file when one cannot be parsed", () => {
    const root = tree({ "wgsl/kernel.wgsl": "", "wgsl/nvidai.wgsl": "" });
    expect(() => variantsIn(join(root, "wgsl"))).toThrow(/nvidai\.wgsl/);
  });

  it("returns nothing for a directory that does not exist", () => {
    expect(variantsIn(join(tree({}), "wgsl"))).toEqual([]);
  });
});

/**
 * `eachVariant` is what makes a new variant impossible to leave untested, so it
 * needs an observation point of its own rather than borrowing one from the ops
 * that will use it later. No op ships a variant yet, so without this the whole
 * mechanism could be deleted and the suite would stay green.
 *
 * The block below runs at collection time against a fixture directory holding
 * two kernels, and `defined` is therefore complete before any assertion runs.
 */
const defined: string[] = [];
const fixture = tree({ "wgsl/kernel.wgsl": "// portable", "wgsl/amd.wgsl": "// amd" });
eachVariant(new URL("wgsl.test.ts", pathToFileURL(`${fixture}/`)), ({ variant, code }) => {
  defined.push(`${variant.name}:${code}`);
  it("gets a test of its own", () => {
    expect(code.length).toBeGreaterThan(0);
  });
});

describe("eachVariant", () => {
  it("runs the block once per variant on disk, without being told they exist", () => {
    expect(defined).toEqual(["amd:// amd", "kernel:// portable"]);
  });
});

describe("variantSuites", () => {
  // What makes a new variant impossible to skip: the test loop is built from
  // the directory, not from a list someone has to remember to extend.
  it("pairs every discovered variant with its source", () => {
    const root = tree({
      "wgsl/kernel.wgsl": "// portable",
      "wgsl/amd.wgsl": "// amd",
    });
    const suites = variantSuites(join(root, "wgsl"));
    expect(suites.map((s) => [s.variant.name, s.code])).toEqual([
      ["amd", "// amd"],
      ["kernel", "// portable"],
    ]);
  });
});
