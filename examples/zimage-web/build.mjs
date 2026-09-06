#!/usr/bin/env node
/**
 * Bundles the Z-Image browser demo into `dist/bundle.js`.
 *
 * Same shape as `examples/llm-demo/build.mjs` and for the same reason: `.wgsl`
 * files have to be inlined as strings, since a browser has no filesystem.
 *
 * There used to be a second reason — `harness/wgsl.ts` imports the `webgpu`
 * package at module scope, and a plugin here redirected that import to a
 * browser copy for the one function (`params()`) the forwards needed from it.
 * Since issue #224 the forwards import `harness/api.ts`, which is the
 * contract with no runtime attached, so nothing in this bundle's graph reaches
 * the Node binding and there is nothing to redirect.
 */
import * as esbuild from "esbuild";
import { fileURLToPath } from "node:url";
import path from "node:path";
import { execSync } from "node:child_process";

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
  loader: { ".wgsl": "text" },
  // So the page can say which build it is. A stale cached bundle throwing an
  // error that was already fixed is otherwise indistinguishable from the fix
  // not working.
  define: {
    BUILD_STAMP: JSON.stringify(
      `${execSync("git rev-parse --short HEAD").toString().trim()} at ${new Date().toISOString().slice(11, 19)}`,
    ),
  },
};

if (watch) {
  const ctx = await esbuild.context(options);
  await ctx.watch();
  console.log("watching for changes — Ctrl+C to stop");
} else {
  await esbuild.build(options);
  console.log(`built ${path.relative(process.cwd(), options.outfile)}`);
}
