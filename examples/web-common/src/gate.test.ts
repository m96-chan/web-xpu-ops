/**
 * The two promises the gate makes, held against the file that makes them.
 *
 * Both are behaviour a browser would show and Node cannot run, so they are
 * asserted against the source. That is weaker than executing it and stronger
 * than nothing — and it is what caught this file moving out of
 * `examples/anima-web/src/main.ts`, where the same two assertions used to live
 * and quietly stopped covering anything.
 */
import { describe, expect, it } from "vitest";
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";

const source = readFileSync(fileURLToPath(new URL("./gate.ts", import.meta.url)), "utf8");

/**
 * The file with its comments removed.
 *
 * The first version of these assertions ran against the raw text and three of
 * them failed on the file's own prose: the comment explaining *why*
 * `isSecureContext` is checked before `navigator.gpu` mentions `navigator.gpu`
 * first, and the comment explaining why `addEventListener` is not used contains
 * `addEventListener`. A test that reads comments as code fails when the code is
 * right and passes when a comment is deleted.
 */
const code = source.replace(/\/\*[\s\S]*?\*\//g, "").replace(/\/\/.*$/gm, "");

describe("web-common / the gate", () => {
  it("opens as a modal, so it centres, dims and traps focus", () => {
    // A `<dialog>` shown with `show()` instead is not modal: the page behind it
    // stays clickable, which for a page that cannot run is worse than useless.
    expect(code).toContain("e.dialog.showModal()");
    expect(code).not.toContain("e.dialog.show()");
  });

  it("blocks Escape while the requirement is unmet", () => {
    // A modal that can be dismissed into a page that cannot work is a modal
    // that lies about being required.
    expect(code).toMatch(/oncancel[\s\S]{0,120}required[\s\S]{0,60}preventDefault/);
  });

  it("checks the insecure context before blaming the browser for WebGPU", () => {
    // `navigator.gpu` is undefined over plain HTTP, and "this browser has no
    // WebGPU" is a lie about a browser that has it. Order matters, so order is
    // what is asserted.
    // Present, **then** first. `indexOf` returns -1 for an absent needle and
    // -1 is less than everything, so an order assertion alone passes when the
    // check is deleted — which is the mutation that survived the first version.
    expect(code).toContain("isSecureContext");
    expect(code).toContain("navigator.gpu");
    expect(code.indexOf("isSecureContext")).toBeLessThan(code.indexOf("navigator.gpu"));
  });

  it("asks for permission inside a click, never on load", () => {
    // A permission prompt outside a user gesture is refused without telling
    // anyone, and the page then reports a good folder as unavailable.
    // On load: query only. The request lives in `bindFolder`, which is reached
    // from the action button and from nowhere else.
    expect(code).toMatch(/hasPermission\(remembered, "readwrite"\)/);
    expect(code).toContain("requestPermission(handle,");
    expect(code).not.toMatch(/requireBoundFolder[\s\S]*?requestPermission\(remembered/);
  });

  it("uses onclick rather than accumulating listeners", () => {
    // `gate()` and the change-folder flow both drive the same button. With
    // `addEventListener` the first one fires alongside the second, re-binding
    // the old folder and closing the dialog out from under the new one.
    expect(code).not.toContain("addEventListener");
  });

  it("forgets a folder it could not fill", () => {
    // Otherwise it is offered as usable on the next load and rejected every
    // time, without ever saying why.
    // Counted, not matched. There are two places a bind can fail -- the first
    // bind and a later change of folder -- and a regex satisfied by either one
    // passes while the other quietly stops forgetting.
    const catches = code.match(/catch\s*\(/g) ?? [];
    const forgets = code.match(/forgetFolder\(\)/g) ?? [];
    expect(catches.length, "every catch that handles a bind failure must forget the folder").toBe(forgets.length);
    expect(forgets.length).toBeGreaterThanOrEqual(2);
  });
});
