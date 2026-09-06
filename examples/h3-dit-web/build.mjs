#!/usr/bin/env node
/**
 * Bundles the MiniMax-H3 video-decoder demo into `dist/bundle.js`.
 *
 * Same shape as `examples/llm-demo/build.mjs` and for the same two reasons:
 * `.wgsl` files have to be inlined as strings (no filesystem in a browser), and
 * `harness/wgsl.ts` imports the `webgpu` package at module scope — a native
 * Node addon with no browser build.
 *
 * The DiT takes its kernels as a *parameter* (`DitKernels`), so no import
 * redirect is needed for them; the shim below exists only for `params()`, which
 * `model-gpu.ts` imports from the harness for its uniform packing. One small
 * function, and redirecting it is cheaper than duplicating the byte layout it
 * defines.
 */
import * as esbuild from "esbuild";
import { fileURLToPath } from "node:url";
import path from "node:path";
import { execSync } from "node:child_process";

const here = path.dirname(fileURLToPath(import.meta.url));
const browserHarnessShim = path.join(here, "../llm-demo/src/browser-runtime.ts");

const harnessBrowserShim = {
  name: "harness-browser-shim",
  setup(build) {
    build.onResolve({ filter: /harness\/(index|wgsl)\.js$/ }, () => ({ path: browserHarnessShim }));
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
  loader: { ".wgsl": "text" },
  // So the page can say which build it is. A stale cached bundle throwing an
  // error that was already fixed is otherwise indistinguishable from the fix
  // not working.
  define: {
    BUILD_STAMP: JSON.stringify(
      `${execSync("git rev-parse --short HEAD").toString().trim()} at ${new Date().toISOString().slice(11, 19)}`,
    ),
  },
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
