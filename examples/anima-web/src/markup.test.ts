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

describe("anima-web / markup", () => {
  it("finds both sides", () => {
    expect(defined.size).toBeGreaterThan(10);
    expect(used.length).toBeGreaterThan(10);
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
    // `showModal` is called on it; on a plain <div> that is a TypeError at the
    // moment the page decides it needs a folder.
    expect(html).toMatch(/<dialog[^>]*id="gate"/);
    expect(main).toContain("gateDialog.showModal()");
  });

  it("every URL the page loads is relative to it", () => {
    // On GitHub Pages this page lives under a subdirectory, and `src="/dist/…"`
    // resolves to the site root where there is no bundle. **A module script
    // that fails to load raises no exception the page can catch**: it simply
    // never runs, the page shows its static markup, and nothing anywhere says
    // why. That is exactly how this was found, after the dialog "existed" with
    // the right text and would not open.
    for (const m of html.matchAll(/\b(?:src|href)="(\/[^/][^"]*)"/g)) {
      expect.fail(`index.html loads ${m[1]} from the site root; make it relative so the page works from a subdirectory`);
    }
  });

  it("Escape cannot dismiss a gate that is required", () => {
    // A modal the user can close into a page that cannot work is a modal that
    // lies about being required.
    expect(main).toMatch(/gateDialog\.oncancel[\s\S]{0,120}preventDefault/);
  });
});
