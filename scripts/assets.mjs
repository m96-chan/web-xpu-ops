#!/usr/bin/env node
/**
 * Copies every op's `.wgsl` beside its compiled JavaScript, then checks that
 * what landed in `dist/` is what the package claims to export.
 *
 * `tsc` emits TypeScript and nothing else. Without this step `dist/` would hold
 * twenty-seven references and not one kernel — a package missing the entire
 * backend it exists to ship, which type-checks, packs and publishes without
 * complaint because nothing in the type system knows those files are there. The
 * failure would surface at a caller's first import, which is far too late.
 *
 * The tree is mirrored rather than flattened: `ops/<op>/wgsl/<entry>.wgsl`
 * becomes `dist/ops/<op>/wgsl/<entry>.wgsl`. That keeps one spelling of a
 * kernel's path across the source tree, the resolution grammar in
 * `harness/resolve.ts` and the published subpath, instead of three that have to
 * agree.
 *
 * The checks at the end are the other half of the job. An op whose `index.ts`
 * the build glob missed, or one that grew a reference and no kernel, is a
 * packaging bug — and packaging bugs are invisible in a repo where the tests
 * import from the source tree and never touch `dist/` at all.
 */
import { copyFileSync, existsSync, mkdirSync, readdirSync, readFileSync } from "node:fs";
import { join } from "node:path";

const OPS = "ops";
const DIST = "dist";

/** Op directories, discovered the same way `tsconfig.build.json` globs them. */
const ops = readdirSync(OPS, { withFileTypes: true })
  .filter((entry) => entry.isDirectory())
  .map((entry) => entry.name)
  .sort();

const problems = [];
let copied = 0;

for (const op of ops) {
  if (!existsSync(join(DIST, OPS, op, "index.js"))) {
    problems.push(`${op}: no compiled index.js — is ops/${op}/index.ts missing?`);
    continue;
  }

  const from = join(OPS, op, "wgsl");
  const kernels = existsSync(from)
    ? readdirSync(from).filter((file) => file.endsWith(".wgsl"))
    : [];

  if (kernels.length === 0) {
    problems.push(`${op}: no .wgsl kernel to publish`);
    continue;
  }

  const to = join(DIST, OPS, op, "wgsl");
  mkdirSync(to, { recursive: true });
  for (const kernel of kernels) {
    copyFileSync(join(from, kernel), join(to, kernel));
    copied += 1;
  }
}

// Every `exports` entry that points into `dist/` has to be there. The ops
// above are checked by walking `ops/`; the model and engine subpaths (issue
// #224) are built by a second `tsc` program, and a program that silently
// dropped an entry from its `include` would publish a subpath that resolves
// to nothing — visible only at a consumer's first import.
const promised = [];
const walkExports = (value) => {
  if (typeof value === "string") {
    if (value.startsWith("./dist/")) promised.push(value.slice("./".length));
  } else if (value && typeof value === "object") Object.values(value).forEach(walkExports);
};
walkExports(JSON.parse(readFileSync("package.json", "utf8")).exports);
for (const path of promised) {
  if (path.includes("*")) continue; // `./ops/*` patterns are covered by the walk above
  if (!existsSync(path)) problems.push(`${path}: promised by package.json#exports, not built`);
}

// Test files reach `dist/` only through a mistake in the build config, and one
// that ships is pure transfer cost for a consumer who can never call it.
const leaked = [];
const walk = (dir) => {
  for (const entry of readdirSync(dir, { withFileTypes: true })) {
    const path = join(dir, entry.name);
    if (entry.isDirectory()) walk(path);
    else if (/\.test\.[cm]?js$/.test(entry.name)) leaked.push(path);
  }
};
if (existsSync(DIST)) walk(DIST);
for (const path of leaked) problems.push(`${path}: test file emitted into the package`);

if (problems.length > 0) {
  for (const problem of problems) console.error(`✗ ${problem}`);
  process.exit(1);
}

console.log(`${ops.length} ops, ${copied} kernels copied into ${DIST}/`);
