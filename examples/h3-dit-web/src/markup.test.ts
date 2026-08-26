/**
 * Every element the page reaches for exists in the page.
 *
 * `$("gate-action")` throws when the id is absent — but only when the page
 * loads, in a browser, which is the slowest place to learn about a typo.
 *
 * Reads both files rather than restating the list, so adding an element to one
 * side and forgetting the other is what fails.
 */
import { describe, expect, it } from "vitest";
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";

const html = readFileSync(fileURLToPath(new URL("../index.html", import.meta.url)), "utf8");
const main = readFileSync(fileURLToPath(new URL("./main.ts", import.meta.url)), "utf8");
const kernels = readFileSync(fileURLToPath(new URL("./kernels-web.ts", import.meta.url)), "utf8");

const defined = new Set([...html.matchAll(/\bid="([^"]+)"/g)].map((m) => m[1]!));
const used = [...main.matchAll(/\$<[^>]+>\("([^"]+)"\)/g)].map((m) => m[1]!);

describe("h3-dit-web / markup", () => {
  it("finds both sides", () => {
    expect(defined.size).toBeGreaterThan(10);
    expect(used.length).toBeGreaterThan(10);
  });

  it("has every element the script asks for", () => {
    expect([...new Set(used)].filter((id) => !defined.has(id))).toEqual([]);
  });

  it("loads the bundle by a relative path, so a subdirectory still works", () => {
    // `src="/dist/bundle.js"` resolves to the site root, where there is no
    // bundle, and **a module script that fails to load raises no exception a
    // page can catch** — so the page renders its static markup and nothing
    // opens the dialog.
    expect(html).toMatch(/src="\.\/dist\/bundle\.js"/);
    expect(html).not.toMatch(/(src|href)="\/[^/]/);
  });

  it("says on the page that the prompt list is fixed", () => {
    // The one thing a visitor will assume otherwise: every other generator has
    // a text field. The reason it does not is 66.7 GB of text encoder, and the
    // page has to carry that rather than leave it to a README.
    expect(html).toMatch(/prompt list is fixed/i);
    expect(html).toMatch(/Qwen3-VL-32B/);
    expect(html).toMatch(/66\.7 GB/);
  });

  it("displays the licence line the agreement requires", () => {
    // "Powered by MiniMax H3" is the agreement's own wording, not a courtesy.
    expect(html).toMatch(/Powered by MiniMax H3/);
    expect(html).toMatch(/Applicable Territory/);
  });

  it("does not offer a step count the conversion may have no table for", () => {
    // `adaln_proj` is not resident, so a step count with no precomputed table
    // has no modulation at all. The options are built from
    // `manifest.stepCounts` at runtime; a hard-coded `<option>` here would be a
    // button that throws.
    const stepsBlock = /<select id="steps">([\s\S]*?)<\/select>/.exec(html);
    expect(stepsBlock).not.toBeNull();
    expect(stepsBlock![1]!.trim()).toBe("");
    expect(main).toMatch(/ditManifest\.stepCounts/);
  });

  it("reads the adapter's limits before it uploads anything", () => {
    // The first run of this page spent 21 s uploading 23 GB and then reported
    // `Instance dropped`, five times, with five stacks and no cause. The cause
    // was `maxComputeWorkgroupSizeX = 256` against `ops/matmul`'s 512 — one
    // number, readable in the first second. Issue #211.
    const limitsAt = main.indexOf("maxComputeWorkgroupSizeX");
    const uploadAt = main.indexOf("DitGpu.create");
    expect(limitsAt).toBeGreaterThan(-1);
    expect(uploadAt).toBeGreaterThan(-1);
    expect(limitsAt).toBeLessThan(uploadAt);
    expect(main).toMatch(/cannot run the matmul/);
    expect(defined.has("limits")).toBe(true);
  });

  it("gives the DiT and the decoder the same kernel objects", () => {
    // `examples/h3-video-web` shipped without `matmulQ8` and would have
    // dispatched `undefined` at its first matmul, because nothing typechecked
    // that project. This asserts the two sets are built from one, so a kernel
    // can only be missing from both.
    expect(kernels).toMatch(/\.\.\.ditKernels/);
    expect(kernels).toMatch(/matmulQ8: matmulQ8Kernel/);
  });
});
