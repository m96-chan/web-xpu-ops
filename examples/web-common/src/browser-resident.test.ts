/**
 * That the browser's resident device did not quietly drift from the Node one.
 *
 * Issue #221, and the reason it is *this* pair being held together: #217 is a
 * bug that existed because a change landed in one implementation and not in
 * another. `ropeAxes`' binding type changed, `examples/zimage/src/dit-gpu.ts`
 * was updated, and the two resident callers were not. Nobody noticed because
 * nothing compared them.
 *
 * `browser-resident.ts` is a second implementation of `harness/resident.ts` on
 * purpose — it runs against the browser's own `navigator.gpu`, with no Dawn and
 * no Node, and it is the file the pages actually go through. That makes it
 * exactly the place a check gets added on one side and forgotten on the other.
 *
 * Asserted against the source, because a browser is what running it would take.
 * Weaker than executing it and considerably stronger than nothing — the same
 * trade `gate.test.ts` makes, for the same reason.
 */
import { describe, expect, it } from "vitest";
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";

const read = (path: string) => readFileSync(fileURLToPath(new URL(path, import.meta.url)), "utf8");
/** Comments stripped: several of them name the very calls being counted. */
const strip = (s: string) => s.replace(/\/\*[\s\S]*?\*\//g, "").replace(/\/\/.*$/gm, "");

const browser = strip(read("./browser-resident.ts"));
const node = strip(read("../../../harness/resident.ts"));

describe("web-common / the browser resident device", () => {
  it("checks binding types in both bind-group paths, as the Node one does", () => {
    // Counted, not matched. There are two paths into a bind group -- whole
    // buffers and slices -- and a regex satisfied by either one passes while
    // the other silently stops checking. Anima's rope goes through the sliced
    // one, which is #217's own path.
    for (const [name, source] of [["browser", browser], ["node", node]] as const) {
      const calls = source.match(/const mismatch = bindingMismatch\(/g) ?? [];
      expect(calls.length, `${name}: both bind-group paths must check`).toBe(2);
      const throws = source.match(/if \(mismatch\) throw new Error\(mismatch\);/g) ?? [];
      expect(throws.length, `${name}: every check must be acted on`).toBe(2);
    }
  });

  it("records what was uploaded, or the check has nothing to compare", () => {
    // The check reads `uploaded`. A version that wired the bind-group half and
    // not this one would pass the test above and never fire.
    for (const [name, source] of [["browser", browser], ["node", node]] as const) {
      expect(source, `${name}: upload must record the view`).toContain("uploaded.set(buffer, data)");
      expect(source, `${name}: the pipeline must record its declaration`).toMatch(
        /declarations\.set\(pipeline, \{ kernel: kernelName\(code\), types: storageElementTypes\(code\) \}\)/,
      );
    }
  });

  it("imports the rule rather than restating it", () => {
    // The plumbing is duplicated because the two runtimes genuinely differ.
    // The rule -- which TypedArray may go into which declaration -- must not be,
    // or the next `atomic<u32>` question gets answered twice and differently.
    expect(browser).toContain("harness/binding-types.js");
    expect(browser).not.toMatch(/instanceof Float32Array/);
  });
});
