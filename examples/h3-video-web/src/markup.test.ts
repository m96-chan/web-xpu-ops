/**
 * Every element the page reaches for exists in the page.
 *
 * `$("gate-action")` throws when the id is absent — but only when the page
 * loads, in a browser, which is the slowest place to learn about a typo. The
 * gate added for #180 is seven new ids across two files, and a mismatch in the
 * one that shows the dialog would mean a page that requires a folder and offers
 * no way to pick one.
 *
 * Reads both files rather than restating the list, so adding an element to one
 * side and forgetting the other is what fails.
 */
import { describe, expect, it } from "vitest";
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";

const html = readFileSync(fileURLToPath(new URL("../index.html", import.meta.url)), "utf8");
const main = readFileSync(fileURLToPath(new URL("./main.ts", import.meta.url)), "utf8");

/** Ids the page defines. */
const defined = new Set([...html.matchAll(/\bid="([^"]+)"/g)].map((m) => m[1]!));
/** Ids the script asks for, through its own `$` helper. */
const used = [...main.matchAll(/\$<[^>]+>\("([^"]+)"\)/g)].map((m) => m[1]!);

describe("h3-video-web / markup", () => {
  it("finds both sides", () => {
    expect(defined.size).toBeGreaterThan(10);
    expect(used.length).toBeGreaterThan(10);
  });

  it("loads the bundle by a relative path, so a subdirectory still works", () => {
    // `src="/dist/bundle.js"` resolves to the site root, where there is no
    // bundle, and **a module script that fails to load raises no exception a
    // page can catch** — so the page renders its static markup and nothing
    // opens the dialog. Measured on Pages before it was fixed.
    expect(html).toMatch(/src="\.\/dist\/bundle\.js"/);
    expect(html).not.toMatch(/(src|href)="\/[^/]/);
  });

  it("says on the page that there is no prompt, rather than offering one", () => {
    // The decoder is 9.69 GB; the transformer that would write its latent and
    // the encoder that would read a prompt are 27 GB more and are not here.
    expect(html).toMatch(/There is no text prompt here/);
    expect(html).not.toMatch(/<textarea/);
  });

  it("states the download size once, where the script reads it from", () => {
    // Another demo copied this sentence out of its neighbour's markup and
    // inherited the wrong size and the wrong licence. The number the gate shows
    // comes from `GateOptions`, so the page's own prose has to agree with it.
    expect(main).toMatch(/downloadSize: "9\.69 GB"/);
    expect(html).toMatch(/9\.69 GB/);
  });

  for (const id of [...new Set(used)]) {
    it(`#${id} is in index.html`, () => {
      expect(
        defined.has(id),
        `main.ts reaches for #${id} and index.html has no such element, so the page throws on load`,
      ).toBe(true);
    });
  }

  it("the gate is a <dialog>, so it centres and dims without hand-rolled CSS", () => {
    // The behaviour that depends on it lives in `web-common/src/gate.ts` and is
    // asserted there; this file's business is the markup, which has to provide
    // an element `showModal` can be called on.
    expect(html).toMatch(/<dialog[^>]*id="gate"/);
  });
});
