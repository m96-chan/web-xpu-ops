#!/usr/bin/env node
/**
 * Dev server for the browser demo (issue #106) — Node standard library only
 * (`node:http`, `node:fs`, `node:path`, `node:url`), per the issue's own
 * instruction: no bundler, no framework, nothing this demo needs
 * `npm install` to serve. `build.mjs` (esbuild) is a separate, explicit step
 * (`npm run demo:build`) that produces the static files this server sends —
 * this script never touches TypeScript or `.wgsl` itself.
 *
 * Two roots:
 *   - the repository itself, so `examples/llm-demo/index.html`, its built
 *     `dist/bundle.js`, and `llm/data/*.vocab.json` are all reachable by
 *     their repo-relative path;
 *   - `/weights/` -> a converted-checkpoint directory (`convert_weights.py`'s
 *     `manifest.json` / `weights.codes.bin` / `weights.scales.bin` /
 *     `weights.norms.bin`), which lives *outside* this repository entirely
 *     (issue #105's own instructions — `technologies-moe/alibi-ai`'s
 *     `third_party/webgpu-weights/`, gitignored there), so it cannot be
 *     served from the repo root. `ALIBI_WEIGHTS_DIR` overrides the default.
 *
 * No `Range` support (issue #106 says it is not required); `Content-Length`
 * is always set, since `llm/browser-weights.ts`'s progress bar reads it.
 */
import { createReadStream, existsSync, statSync } from "node:fs";
import { pipeline } from "node:stream";
import { createServer } from "node:http";
import { extname, join, resolve, sep } from "node:path";
import { fileURLToPath } from "node:url";

const HERE = fileURLToPath(new URL(".", import.meta.url)); // examples/llm-demo/
const REPO_ROOT = resolve(HERE, "..", "..");
const WEIGHTS_DIR = resolve(
  process.env.ALIBI_WEIGHTS_DIR ??
    "/home/m96-chan/project/technologies.moe/alibi-ai/third_party/webgpu-weights/sarashina2.2-1b-alibi-v1-q8",
);
const PORT = Number(process.env.PORT ?? 8770);

const MIME = {
  ".html": "text/html; charset=utf-8",
  ".js": "text/javascript; charset=utf-8",
  ".mjs": "text/javascript; charset=utf-8",
  ".map": "application/json; charset=utf-8",
  ".json": "application/json; charset=utf-8",
  ".css": "text/css; charset=utf-8",
  ".wgsl": "text/plain; charset=utf-8",
  ".bin": "application/octet-stream",
};

/** `true` when `path` is `root` itself or a descendant of it — refuses a resolved path that escaped its root via `..`. */
function within(root, path) {
  return path === root || path.startsWith(root + sep);
}

function send(res, filePath) {
  const stat = statSync(filePath);
  res.writeHead(200, {
    "Content-Type": MIME[extname(filePath)] ?? "application/octet-stream",
    "Content-Length": stat.size,
  });
  // pipeline, not .pipe(): a bare ReadStream 'error' (file replaced by
  // demo:build mid-stream, I/O error in the 1.4 GiB weights read) has no
  // listener and takes the whole server process down, and .pipe() never
  // destroys the source when the client aborts a download, leaking the fd.
  pipeline(createReadStream(filePath), res, () => {});
}

const server = createServer((req, res) => {
  try {
    const url = new URL(req.url ?? "/", `http://localhost:${PORT}`);
    let filePath;
    let root;
    if (url.pathname.startsWith("/weights/")) {
      root = WEIGHTS_DIR;
      filePath = resolve(WEIGHTS_DIR, "." + url.pathname.slice("/weights".length));
    } else {
      root = REPO_ROOT;
      const pathname = url.pathname.endsWith("/") ? `${url.pathname}index.html` : url.pathname;
      filePath = resolve(REPO_ROOT, "." + pathname);
    }

    if (!within(root, filePath)) {
      res.writeHead(403, { "Content-Type": "text/plain" });
      res.end("forbidden: path escapes its root");
      return;
    }
    if (!existsSync(filePath) || statSync(filePath).isDirectory()) {
      res.writeHead(404, { "Content-Type": "text/plain" });
      res.end(`not found: ${url.pathname}`);
      return;
    }
    send(res, filePath);
  } catch (err) {
    res.writeHead(500, { "Content-Type": "text/plain" });
    res.end(String(err instanceof Error ? err.stack : err));
  }
});

// 127.0.0.1, not the default 0.0.0.0: this server exposes the repository
// root (including .git/ and untracked local files) plus an external weights
// directory, and its only documented audience is a browser on this machine.
server.listen(PORT, "127.0.0.1", () => {
  console.log(`llm-demo dev server listening on http://localhost:${PORT}/`);
  console.log(`  demo:    http://localhost:${PORT}/examples/llm-demo/`);
  console.log(`  weights: ${WEIGHTS_DIR}  ->  /weights/`);
  if (!existsSync(join(WEIGHTS_DIR, "manifest.json"))) {
    console.warn(`  warning: ${WEIGHTS_DIR}/manifest.json not found — has convert_weights.py been run?`);
  }
  if (!existsSync(join(HERE, "dist", "bundle.js"))) {
    console.warn("  warning: dist/bundle.js not found — run `npm run demo:build` first.");
  }
});
