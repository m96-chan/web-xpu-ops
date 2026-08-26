/**
 * Every element the page reaches for exists in the page.
 *
 * Issue #212. Plus the two things this page must say and one it must not do.
 */
import { describe, expect, it } from "vitest";
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";

const html = readFileSync(fileURLToPath(new URL("../index.html", import.meta.url)), "utf8");
const main = readFileSync(fileURLToPath(new URL("./main.ts", import.meta.url)), "utf8");

const defined = new Set([...html.matchAll(/\bid="([^"]+)"/g)].map((m) => m[1]!));
const used = [...main.matchAll(/\$<[^>]+>\("([^"]+)"\)/g)].map((m) => m[1]!);

describe("h3-ref2v-web / markup", () => {
  it("finds both sides", () => {
    expect(defined.size).toBeGreaterThan(10);
    expect(used.length).toBeGreaterThan(10);
  });

  it("has every element the script asks for", () => {
    expect([...new Set(used)].filter((id) => !defined.has(id))).toEqual([]);
  });

  it("loads the bundle by a relative path", () => {
    expect(html).toMatch(/src="\.\/dist\/bundle\.js"/);
    expect(html).not.toMatch(/(src|href)="\/[^/]/);
  });

  it("takes references, which is what makes it R2V", () => {
    // A prompt-only page would be `examples/h3-dit-web`. The drop zone and the
    // file input are the difference.
    expect(defined.has("drop")).toBe(true);
    expect(defined.has("files")).toBe(true);
    expect(html).toMatch(/accept="image\/\*,video\/\*"/);
  });

  it("says the resize is the browser's, not PIL's", () => {
    // The one step in the chain that is *not* held to the model. A page that
    // did not say so would be claiming more than was measured.
    expect(html).toMatch(/bicubic/);
    expect(html).toMatch(/unmeasured/);
  });

  it("says the models run in sequence, at the size they actually are", () => {
    // 48.9 GB fits on no card this page will meet, and the staging is the
    // reason the page can exist at all. The figure is asserted rather than the
    // sentence because it is the figure that goes stale: it read 48.7 until the
    // conversion stopped keeping a text layer nobody evaluates.
    expect(html).toMatch(/48\.9 GB/);
    expect(html).toMatch(/sequence, not together/);
  });

  it("reads the adapter's limits before it uploads anything", () => {
    // Issue #211: `examples/h3-dit-web` spent 21 s on 23 GB to learn one number.
    const limitsAt = main.indexOf("maxComputeWorkgroupSizeX");
    // The **call**, not the import at the top of the file — which is what this
    // matched first, and it made the assertion pass for the wrong reason in
    // one direction and fail in the other.
    const gateAt = main.indexOf("await requireBoundFolder(");
    expect(limitsAt).toBeGreaterThan(-1);
    expect(gateAt).toBeGreaterThan(-1);
    expect(limitsAt).toBeLessThan(gateAt);
  });

  it("offers no step count of its own", () => {
    // Built from the DiT manifest, as `examples/h3-dit-web` does, because a
    // step count with no precomputed table has no modulation at all.
    const block = /<select id="steps">([\s\S]*?)<\/select>/.exec(html);
    expect(block).not.toBeNull();
    expect(block![1]!.trim()).toBe("");
    expect(main).toMatch(/dit\.stepCounts/);
  });
});
