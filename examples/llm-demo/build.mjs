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
 *   1. `llm/kernels.ts` asks `harness/api.ts`'s kernel registry for each
 *      shader's WGSL text, and a page has no filesystem to register from. A
 *      bundler that can inline a file's text into the module graph at build
 *      time (`src/browser-runtime.ts`'s `WGSL_TABLE`, one static import per
 *      `.wgsl`) is what gives `main.ts` a table to hand to
 *      `registerKernelSources` before the first dispatch.
 *   2. The engines' imports resolve as native ESM with explicit `.js`
 *      specifiers, but a page still wants one file to load rather than the
 *      forty modules they reach — and the `.wgsl` loader above only exists
 *      inside a bundle anyway.
 *
 * esbuild specifically: it is already exactly what this repository would
 * reach for if it ever needed one (zero-config TS+ESM support, a `text`
 * loader for point 1), it needs no
 * config file of its own beyond this script, and at ~10ms for a graph this
 * size it does not turn `npm run demo:build` into something anyone avoids
 * running. It is a `devDependency` — nothing published under `dist/` (the
 * package's own build) touches it.
 */
import * as esbuild from "esbuild";
import { fileURLToPath } from "node:url";
import path from "node:path";

const here = path.dirname(fileURLToPath(import.meta.url));
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
};

if (watch) {
  const ctx = await esbuild.context(options);
  await ctx.watch();
  console.log("watching for changes — Ctrl+C to stop");
} else {
  await esbuild.build(options);
  console.log(`built ${path.relative(process.cwd(), options.outfile)}`);
}
