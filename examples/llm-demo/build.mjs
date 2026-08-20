#!/usr/bin/env node
/**
 * Bundles the browser demo's entry point into `dist/bundle.js`.
 *
 * ## Why esbuild, and why a bundler at all
 *
 * The rest of this repository ships **unbundled** — `npm run build` is a
 * plain `tsc` compile (see `tsconfig.build.json`), because every published
 * entry point (`ops/*\/index.ts`, `llm/tokenizer.ts`) already resolves as
 * native ESM with explicit `.js` specifiers (`moduleResolution: "NodeNext"`)
 * and touches nothing a browser or Node can't load directly. This demo
 * cannot get away with that, for two reasons specific to it:
 *
 *   1. `llm/kernels.ts` reads a kernel's `.wgsl` source via
 *      `harness/index.ts#kernel()`, which calls `node:fs`'s `readFileSync`
 *      — fine for the test suite (Node-only), fatal in a browser (no
 *      filesystem). The demo needs those ten kernel strings available
 *      *synchronously* at module-eval time (`kernels.ts`'s `CODE` object is
 *      built with plain assignments, not awaited), which rules out an
 *      in-browser `fetch()` shim. A bundler that can inline a file's text
 *      into the module graph at build time is the only way to keep
 *      `kernels.ts` — and therefore `llm/engine-q8.ts` — completely unmodified
 *      while still running in a browser; see `harnessBrowserShim` below.
 *   2. `harness/wgsl.ts` (which `kernel`/`params` come from) imports the
 *      `webgpu` package at module scope — a native Node addon (Dawn) with no
 *      browser equivalent. `src/browser-runtime.ts` is this demo's own
 *      `Runner` over `navigator.gpu`; a bundler is what lets one import
 *      specifier (`../harness/index.js`, as `llm/kernels.ts` already spells
 *      it) resolve to the Node harness under `npm test` and to this file
 *      under the demo build, with zero changes to `kernels.ts` itself.
 *
 * esbuild specifically: it is already exactly what this repository would
 * reach for if it ever needed one (zero-config TS+ESM support, a `text`
 * loader for point 1, an `onResolve` plugin hook for point 2), it needs no
 * config file of its own beyond this script, and at ~10ms for a graph this
 * size it does not turn `npm run demo:build` into something anyone avoids
 * running. It is a `devDependency` — nothing published under `dist/` (the
 * package's own build) touches it.
 */
import * as esbuild from "esbuild";
import { fileURLToPath } from "node:url";
import path from "node:path";

const here = path.dirname(fileURLToPath(import.meta.url));
const browserHarnessShim = path.join(here, "src/browser-runtime.ts");

/**
 * Redirects `llm/kernels.ts`'s `import { kernel, params, type Runner } from
 * "../harness/index.js"` to this demo's browser-native counterpart. The
 * `type Runner` half of that import is erased by esbuild before resolution
 * even runs (TypeScript type-only imports produce no JS), so only the
 * `kernel`/`params` value import ever reaches this hook — confirmed by
 * grepping every `llm/*.ts` import of `harness/index.js`: `engine.ts` and
 * `engine-q8.ts` both import `type Runner` only (erased), and every other
 * importer is a `*.test.ts` file this bundle's entry point never reaches.
 * `src/browser-runtime.ts` exports `kernel(url, name)` and `params(fields)`
 * with the exact same signatures `harness/wgsl.ts`/`harness/suite.ts`
 * define, so `kernels.ts` needed no changes at all — see that file's own
 * module doc for how it sources its ten kernels' WGSL text.
 */
const harnessBrowserShim = {
  name: "harness-browser-shim",
  setup(build) {
    build.onResolve({ filter: /harness\/index\.js$/ }, () => ({ path: browserHarnessShim }));
  },
};

const watch = process.argv.includes("--watch");

const options = {
  entryPoints: [path.join(here, "src/main.ts")],
  outfile: path.join(here, "dist/bundle.js"),
  bundle: true,
  format: "esm",
  platform: "browser",
  target: "es2022",
  sourcemap: true,
  logLevel: "info",
  // `.wgsl` files are plain WGSL source, inlined as JS string literals — see
  // point 1 above. Every `import x from "*.wgsl"` in `src/browser-runtime.ts`
  // resolves through this, and `src/wgsl.d.ts` gives `tsc` the matching type.
  loader: { ".wgsl": "text" },
  plugins: [harnessBrowserShim],
};

if (watch) {
  const ctx = await esbuild.context(options);
  await ctx.watch();
  console.log("watching for changes — Ctrl+C to stop");
} else {
  await esbuild.build(options);
  console.log(`built ${path.relative(process.cwd(), options.outfile)}`);
}
