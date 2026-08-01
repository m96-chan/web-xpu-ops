import { mkdirSync, mkdtempSync, readdirSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { pathToFileURL } from "node:url";
import { afterAll, describe, expect, it } from "vitest";
import { DEFAULT_ENTRY, parseVariant, resolve, variantSuites, variantsIn } from "./resolve.js";
import { eachVariant } from "./variants.js";

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
  const all = ["kernel.frobnitz.f16", "kernel.frobnitz", "kernel.widget", "kernel.f16", "kernel"];

  it("takes target + dtype first", () => {
    const choice = resolve({ have: all, target: "frobnitz", dtype: "f16" });
    expect(choice.name).toBe("kernel.frobnitz.f16");
    expect(choice.rung).toBe("target+dtype");
  });

  it("falls back to the target when no target+dtype variant exists", () => {
    const choice = resolve({
      have: ["kernel.frobnitz", "kernel.f16", "kernel"],
      target: "frobnitz",
      dtype: "f16",
    });
    expect(choice.name).toBe("kernel.frobnitz");
    expect(choice.rung).toBe("target");
  });

  it("falls back to the dtype when the target has no variant at all", () => {
    const choice = resolve({
      have: ["kernel.widget", "kernel.f16", "kernel"],
      target: "frobnitz",
      dtype: "f16",
    });
    expect(choice.name).toBe("kernel.f16");
    expect(choice.rung).toBe("dtype");
  });

  it("falls back to the entry's own portable kernel when nothing else hits", () => {
    const choice = resolve({ have: ["kernel.widget", "kernel"], target: "frobnitz", dtype: "f16" });
    expect(choice.name).toBe("kernel");
    expect(choice.rung).toBe("portable");
  });

  // The whole reason detection is allowed to return null: an unrecognised
  // adapter must get the portable kernel, never someone else's.
  it("takes portable when the target is unknown, even though variants exist", () => {
    const choice = resolve({ have: all, target: null, dtype: null });
    expect(choice.name).toBe("kernel");
    expect(choice.rung).toBe("portable");
  });

  it("still uses the dtype when the target is unknown", () => {
    const choice = resolve({ have: all, target: null, dtype: "f16" });
    expect(choice.name).toBe("kernel.f16");
    expect(choice.rung).toBe("dtype");
  });

  it("steers on the target: the same variant set resolves differently", () => {
    expect(resolve({ have: all, target: "frobnitz", dtype: null }).name).toBe("kernel.frobnitz");
    expect(resolve({ have: all, target: "widget", dtype: null }).name).toBe("kernel.widget");
  });

  it("reports the whole chain it walked, in order", () => {
    const choice = resolve({ have: ["kernel"], target: "frobnitz", dtype: "f16" });
    expect(choice.tried).toEqual([
      "kernel.frobnitz.f16",
      "kernel.frobnitz",
      "kernel.f16",
      "kernel",
    ]);
  });

  it("throws when the entry has no portable kernel", () => {
    expect(() => resolve({ have: ["kernel.frobnitz"], target: null, dtype: null })).toThrow(
      /kernel\.wgsl/,
    );
  });
});

/**
 * An op may have more than one entry point, and two of them shipped while this
 * branch was open: `stft` has `kernel` and `inverse`, `attention` has `scores`
 * and `context`. Resolution has to stay inside the entry it was asked about.
 */
describe("resolve / named entry points", () => {
  const all = ["kernel", "kernel.frobnitz", "inverse", "inverse.frobnitz", "inverse.frobnitz.f16"];

  it("resolves a named entry's variant", () => {
    const choice = resolve({ have: all, entry: "inverse", target: "frobnitz", dtype: "f16" });
    expect(choice.name).toBe("inverse.frobnitz.f16");
    expect(choice.rung).toBe("target+dtype");
  });

  it("falls back to that entry's own portable kernel", () => {
    const choice = resolve({ have: all, entry: "inverse", target: "widget", dtype: null });
    expect(choice.name).toBe("inverse");
    expect(choice.rung).toBe("portable");
  });

  // The failure this guards against is the worst kind available here: istft
  // silently running the forward transform because the chain ran off the end of
  // its own entry point and found something that happened to compile.
  it("never falls back across entry points", () => {
    const choice = resolve({ have: ["kernel", "kernel.frobnitz", "inverse"], entry: "inverse" });
    expect(choice.name).toBe("inverse");
    expect(choice.tried).not.toContain("kernel");
  });

  it("throws rather than borrowing another entry's kernel", () => {
    expect(() => resolve({ have: ["kernel", "kernel.frobnitz"], entry: "inverse" })).toThrow(
      /inverse\.wgsl/,
    );
  });

  it("defaults to the kernel entry when none is named", () => {
    expect(resolve({ have: all }).name).toBe(DEFAULT_ENTRY);
  });

  it("names the entry in the chain it reports", () => {
    const choice = resolve({ have: ["scores"], entry: "scores", target: "frobnitz", dtype: "f16" });
    expect(choice.tried).toEqual([
      "scores.frobnitz.f16",
      "scores.frobnitz",
      "scores.f16",
      "scores",
    ]);
  });
});

describe("resolve / override", () => {
  const all = ["kernel.frobnitz.f16", "kernel.frobnitz", "kernel.widget", "kernel.f16", "kernel"];

  it("beats a detected target", () => {
    const choice = resolve({
      have: all,
      target: "frobnitz",
      dtype: null,
      override: "kernel.widget",
    });
    expect(choice.name).toBe("kernel.widget");
    expect(choice.rung).toBe("override");
  });

  it("beats a target+dtype hit", () => {
    const choice = resolve({ have: all, target: "frobnitz", dtype: "f16", override: "kernel" });
    expect(choice.name).toBe("kernel");
    expect(choice.rung).toBe("override");
  });

  it("skips the chain entirely rather than walking past itself", () => {
    const choice = resolve({
      have: all,
      target: "frobnitz",
      dtype: "f16",
      override: "kernel.widget",
    });
    expect(choice.tried).toEqual(["kernel.widget"]);
  });

  // Falling through to portable on a typo would hand the caller a kernel they
  // did not ask for and never tell them. That is the invisible wrong choice.
  it("throws when it names a variant that does not exist", () => {
    expect(() =>
      resolve({ have: all, target: "frobnitz", dtype: null, override: "kernel.widgit" }),
    ).toThrow(/widgit/);
  });
});

describe("parseVariant", () => {
  it("reads the default entry point", () => {
    expect(parseVariant("kernel.wgsl")).toEqual({
      name: "kernel",
      entry: "kernel",
      target: null,
      dtype: null,
    });
  });

  // The three names that broke the first version of this grammar, all from ops
  // that landed while this branch was open.
  it.each(["inverse", "scores", "context"])("reads the named entry point %j", (entry) => {
    expect(parseVariant(`${entry}.wgsl`)).toEqual({
      name: entry,
      entry,
      target: null,
      dtype: null,
    });
  });

  it("reads a target variant of an entry", () => {
    expect(parseVariant("kernel.nvidia.wgsl")).toEqual({
      name: "kernel.nvidia",
      entry: "kernel",
      target: "nvidia",
      dtype: null,
    });
  });

  it("reads a dtype variant of an entry", () => {
    expect(parseVariant("inverse.f16.wgsl")).toEqual({
      name: "inverse.f16",
      entry: "inverse",
      target: null,
      dtype: "f16",
    });
  });

  it("reads a target+dtype variant of an entry", () => {
    expect(parseVariant("scores.apple.f16.wgsl")).toEqual({
      name: "scores.apple.f16",
      entry: "scores",
      target: "apple",
      dtype: "f16",
    });
  });

  it("takes a target vocabulary, so resolution can be exercised without real vendors", () => {
    expect(parseVariant("kernel.frobnitz.f16.wgsl", FAKE)).toEqual({
      name: "kernel.frobnitz.f16",
      entry: "kernel",
      target: "frobnitz",
      dtype: "f16",
    });
  });

  // A misspelt variant is worse than a missing one: it sits in the tree looking
  // tuned while resolution never reaches it. This is the check that survived the
  // grammar change, because it is the one doing the work.
  it("throws on a misspelt target", () => {
    expect(() => parseVariant("kernel.nvidai.wgsl")).toThrow(/nvidai/);
  });

  it("throws on a misspelt dtype", () => {
    expect(() => parseVariant("kernel.fp16.wgsl")).toThrow(/fp16/);
  });

  it("throws when target and dtype are the wrong way round", () => {
    expect(() => parseVariant("kernel.f16.nvidia.wgsl")).toThrow();
  });

  // The spelling the first version of this grammar used. Anyone who writes it
  // means "the nvidia variant" and would otherwise get a whole new entry point
  // that nothing ever dispatches.
  it.each(["nvidia", "apple", "f16", "u32"])(
    "throws on a bare %j, which is ambiguous rather than an entry point",
    (bare) => {
      expect(() => parseVariant(`${bare}.wgsl`)).toThrow(/ambiguous/i);
    },
  );

  it("throws on a file that is not wgsl", () => {
    expect(() => parseVariant("kernel.txt")).toThrow();
  });

  it("throws on more segments than the grammar has", () => {
    expect(() => parseVariant("kernel.nvidia.f16.extra.wgsl")).toThrow();
  });
});

describe("variantsIn", () => {
  it("discovers every variant on disk, sorted, ignoring anything else", () => {
    const root = tree({
      "wgsl/kernel.wgsl": "",
      "wgsl/kernel.nvidia.wgsl": "",
      "wgsl/kernel.nvidia.f16.wgsl": "",
      "wgsl/NOTES.md": "not a kernel",
    });
    expect(variantsIn(join(root, "wgsl")).map((v) => v.name)).toEqual([
      "kernel",
      "kernel.nvidia",
      "kernel.nvidia.f16",
    ]);
  });

  it("keeps several entry points apart", () => {
    const root = tree({
      "wgsl/kernel.wgsl": "",
      "wgsl/inverse.wgsl": "",
      "wgsl/inverse.nvidia.wgsl": "",
    });
    const found = variantsIn(join(root, "wgsl"));
    expect(found.map((v) => [v.entry, v.name])).toEqual([
      ["inverse", "inverse"],
      ["inverse", "inverse.nvidia"],
      ["kernel", "kernel"],
    ]);
  });

  // Portable is the floor. A variant with no portable beside it means an
  // unrecognised device has nothing of that entry point to fall back to.
  it("throws when a variant has no portable kernel for its entry", () => {
    const root = tree({ "wgsl/kernel.wgsl": "", "wgsl/inverse.nvidia.wgsl": "" });
    expect(() => variantsIn(join(root, "wgsl"))).toThrow(/inverse\.wgsl/);
  });

  it("names the offending file when one cannot be parsed", () => {
    const root = tree({ "wgsl/kernel.wgsl": "", "wgsl/kernel.nvidai.wgsl": "" });
    expect(() => variantsIn(join(root, "wgsl"))).toThrow(/kernel\.nvidai\.wgsl/);
  });

  it("returns nothing for a directory that does not exist", () => {
    expect(variantsIn(join(tree({}), "wgsl"))).toEqual([]);
  });
});

/**
 * Run against the real tree, which is what this change exists for.
 *
 * The first version of this grammar assumed every `.wgsl` under an op was either
 * `kernel.wgsl` or a variant of it, and that was already false by the time the
 * branch was ready to merge. Fixtures alone could not have caught it — nothing
 * in a fixture knows what shipped last week — so these read the ops themselves.
 */
describe("variantsIn / the ops in this repository", () => {
  const ops = new URL("../ops/", import.meta.url);

  it("reads stft, which has a second entry point for the inverse transform", () => {
    const found = variantsIn(new URL("stft/wgsl/", ops));
    expect(found.map((v) => v.name)).toEqual(["inverse", "kernel"]);
    expect(found.every((v) => v.target === null && v.dtype === null)).toBe(true);
  });

  it("reads attention, whose two dispatches are two entry points and no kernel.wgsl", () => {
    const found = variantsIn(new URL("attention/wgsl/", ops));
    expect(found.map((v) => v.entry)).toEqual(["context", "scores"]);
  });

  it("reads every op in the tree without tripping the grammar", () => {
    const seen = new Map<string, number>();
    for (const op of readdirSync(ops)) {
      const found = variantsIn(new URL(`${op}/wgsl/`, ops));
      if (found.length > 0) seen.set(op, found.length);
    }
    // Every op ships at least one entry point, and this is the assertion that
    // fails the day one of them is named something the grammar cannot read.
    expect(seen.size).toBeGreaterThanOrEqual(14);
    expect(seen.get("stft")).toBe(2);
    expect(seen.get("attention")).toBe(2);
  });
});

describe("variantSuites", () => {
  // What makes a new variant impossible to skip: the test loop is built from
  // the directory, not from a list someone has to remember to extend.
  it("pairs every discovered variant with its source", () => {
    const root = tree({
      "wgsl/kernel.wgsl": "// portable",
      "wgsl/kernel.amd.wgsl": "// amd",
    });
    const suites = variantSuites(join(root, "wgsl"), "kernel");
    expect(suites.map((s) => [s.variant.name, s.code])).toEqual([
      ["kernel", "// portable"],
      ["kernel.amd", "// amd"],
    ]);
  });

  it("returns only the entry it was asked for", () => {
    const root = tree({
      "wgsl/kernel.wgsl": "// forward",
      "wgsl/inverse.wgsl": "// inverse",
      "wgsl/inverse.amd.wgsl": "// inverse amd",
    });
    const suites = variantSuites(join(root, "wgsl"), "inverse");
    expect(suites.map((s) => s.variant.name)).toEqual(["inverse", "inverse.amd"]);
  });

  it("throws when asked for an entry point that does not exist", () => {
    const root = tree({ "wgsl/kernel.wgsl": "" });
    expect(() => variantSuites(join(root, "wgsl"), "inverse")).toThrow(/inverse/);
  });
});

/**
 * `eachVariant` is what makes a new variant impossible to leave untested, so it
 * needs an observation point of its own rather than borrowing one from the ops
 * that will use it later. No op ships a variant yet, so without this the whole
 * mechanism could be deleted and the suite would stay green.
 *
 * The block below runs at collection time against a fixture directory holding
 * two entry points, and `defined` is therefore complete before any assertion runs.
 */
const defined: string[] = [];
const fixture = tree({
  "wgsl/kernel.wgsl": "// portable",
  "wgsl/kernel.amd.wgsl": "// amd",
  "wgsl/inverse.wgsl": "// inverse",
});
eachVariant(new URL("wgsl.test.ts", pathToFileURL(`${fixture}/`)), "kernel", ({ variant, code }) => {
  defined.push(`${variant.name}:${code}`);
  it("gets a test of its own", () => {
    expect(code.length).toBeGreaterThan(0);
  });
});

describe("eachVariant", () => {
  it("runs the block once per variant on disk, without being told they exist", () => {
    expect(defined).toEqual(["kernel:// portable", "kernel.amd:// amd"]);
  });

  it("covers one entry point at a time, so an op with two declares both", () => {
    expect(defined.some((d) => d.startsWith("inverse"))).toBe(false);
  });
});
