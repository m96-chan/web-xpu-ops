/**
 * Every benchmark dispatches the shipped kernel the way the shipped kernel is
 * dispatched.
 *
 * Three of them did not. `ops/matmul/tools/bench.ts`, `bench-q8.ts` and
 * `ops/attention/tools/bench.ts` each restated a grid inline — `ceil(N / 16)`,
 * or one workgroup per query — that was correct before the kernels were tiled
 * and wrong after. Each launched about **thirty-two times too many
 * workgroups**, every one of them recomputing a whole tile.
 *
 * Nothing failed. The kernels guard their stores, so the answers stayed right
 * and only the clock moved: the baseline came out ~32x too slow, and every
 * candidate measured against it looked like a triumph. That is the same class
 * of mistake as timing a program other than the one that ships, and it is the
 * reason the flash baseline once read 83.93 ms against its real 8.43.
 *
 * The fix in each case was to call the grid function the op already exports.
 * This test holds that: **a benchmark that loads a shipped `.wgsl` off disk has
 * to import its op's grid helper.**
 *
 * It looks for a **call**, because the first version of this test looked for a
 * mention and the import left behind by the mutation satisfied it — the check
 * passed on all three bugs put back. What it still cannot see is a bench that
 * calls the helper and then ignores the result; that would need the dispatch
 * builders exported, and importing a bench spins up a GPU device. The three
 * benches were made uniform (every shipped dispatch takes its grid from a
 * helper call) so that this narrower check is exact for the way they are
 * written, and `ops/flash_attention/tools/bench.ts` grew an explicit `grid`
 * parameter for that reason rather than passing a tile width.
 */
import { describe, expect, it } from "vitest";
import { readdirSync, readFileSync, existsSync, statSync } from "node:fs";
import { fileURLToPath } from "node:url";
import path from "node:path";

const root = fileURLToPath(new URL("..", import.meta.url));
const ops = path.join(root, "ops");

/** Every `ops/<op>/tools/bench*.ts`. */
function benches(): { op: string; file: string; source: string }[] {
  const found: { op: string; file: string; source: string }[] = [];
  for (const op of readdirSync(ops)) {
    const tools = path.join(ops, op, "tools");
    if (!existsSync(tools) || !statSync(tools).isDirectory()) continue;
    for (const name of readdirSync(tools)) {
      if (!/^bench.*\.ts$/.test(name)) continue;
      const file = path.join(tools, name);
      found.push({ op, file, source: readFileSync(file, "utf8") });
    }
  }
  return found;
}

/**
 * The ops whose shipped WGSL a bench reads off disk.
 *
 * Reading it is the tell: a bench that builds its candidates from a generator
 * chooses their grids itself, but one that loads the *shipped* file is
 * dispatching something whose tile it does not own.
 */
function shippedOpsRead(source: string): string[] {
  const out = new Set<string>();
  for (const m of source.matchAll(/new URL\("([^"]*\/)?\.\.\/(?:\.\.\/)?(?:([a-z_]+)\/)?wgsl\/[^"]+"/g)) {
    out.add(m[2] ?? "self");
  }
  // Template-literal paths (`../wgsl/${GENERATION}.wgsl`) as well.
  for (const m of source.matchAll(/new URL\(`([^`]*\/)?\.\.\/(?:\.\.\/)?(?:([a-z_]+)\/)?wgsl\/[^`]+`/g)) {
    out.add(m[2] ?? "self");
  }
  return [...out];
}

/**
 * The grid *functions* an op exports, read from its own `index.ts`.
 *
 * Discovered rather than listed here: a hardcoded map goes stale the moment an
 * op grows a grid helper, and a stale map makes this test skip the very case it
 * was added for. An op with no such export has no tile for a bench to get
 * wrong — `ops/attention`'s kernels are one workgroup per query row — and is
 * reported as unchecked rather than quietly passed.
 */
function gridSymbols(op: string): string[] {
  const index = path.join(ops, op, "index.ts");
  if (!existsSync(index)) return [];
  const source = readFileSync(index, "utf8");
  const names = new Set<string>();
  for (const m of source.matchAll(/export function ([A-Za-z0-9_]*Grid)\b/g)) names.add(m[1]!);
  return [...names];
}

const ALL = benches();

describe("benchmarks / the baseline is the shipped dispatch", () => {
  it("finds the benchmarks", () => {
    expect(ALL.length).toBeGreaterThanOrEqual(4);
  });

  for (const { op, file, source } of ALL) {
    const shipped = shippedOpsRead(source).map((from) => (from === "self" ? op : from));
    if (shipped.length === 0) continue;
    const short = path.relative(root, file);

    it(`${short}: takes the tile from the ops it measures, not from a literal`, () => {
      const unchecked: string[] = [];
      for (const from of shipped) {
        const symbols = gridSymbols(from);
        if (symbols.length === 0) {
          // No tile to get wrong: one workgroup per output row.
          unchecked.push(from);
          continue;
        }
        // A **call**, not a mention. An import left behind after the call was
        // replaced by a literal is exactly the regression this exists to catch,
        // and `includes(name)` passes on the import alone — verified by putting
        // each of the three bugs back and watching this fail.
        expect(
          symbols.some((name) => source.includes(`${name}(`)),
          `${short}: loads ${from}'s shipped kernel but calls none of its grid helpers ` +
            `(${symbols.map((n) => `${n}()`).join(", ")}). A grid restated inline goes stale when the kernel ` +
            "is tiled, and the only symptom is a baseline that is too slow — which reads as every " +
            "candidate being fast.",
        ).toBe(true);
      }
      // Recorded, not asserted: it is information about what this test does not
      // cover, and an empty expectation would hide it.
      if (unchecked.length) console.log(`    (${short}: ${unchecked.join(", ")} export no tile — nothing to check)`);
    });

    it(`${short}: does not restate a tile constant for the shipped grid`, () => {
      // The exact spelling all three used. A bench is free to compute grids for
      // its own generated candidates; what it may not do is invent a tile for
      // the kernel it is measuring against.
      expect(source).not.toMatch(/const TILE = \d+;/);
    });
  }
});
