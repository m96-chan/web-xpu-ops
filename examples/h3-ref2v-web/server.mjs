#!/usr/bin/env node
/**
 * A static server for the MiniMax-H3 R2V demo that answers `Range`.
 *
 * The gate fills an empty folder by ranges — `examples/web-common/src/provision.ts`
 * asks for 8 MB at a time and keeps eight in flight, measured at 25.8 MB/s
 * against 3.2 MB/s sequential. A server that ignored `Range` would answer each
 * of those with the whole 20.08 GB file.
 *
 *     node examples/h3-ref2v-web/server.mjs --weights ~/h3-ref2v-web
 *
 * `--weights` is a directory `examples/h3-dit/tools/convert_dit.py`, `convert_decoder.py` and `encode_prompt.py` wrote into.
 */
import { createReadStream, existsSync, readFileSync, statSync } from "node:fs";
import http from "node:http";
import path from "node:path";
import { fileURLToPath } from "node:url";

const here = path.dirname(fileURLToPath(import.meta.url));
const repoRoot = path.resolve(here, "../..");

function arg(flag, fallback) {
  const at = process.argv.indexOf(flag);
  return at >= 0 && process.argv[at + 1] !== undefined ? process.argv[at + 1] : fallback;
}

const weightsDir = arg("--weights", null);
const port = Number(arg("--port", 8791));

const TYPES = {
  ".html": "text/html; charset=utf-8",
  ".js": "text/javascript; charset=utf-8",
  ".map": "application/json",
  ".json": "application/json",
  ".bin": "application/octet-stream",
};

/**
 * `index.html`, with the bundle's mtime appended to its `<script src>`.
 *
 * `Cache-Control: no-store` was measured not to be enough — a tab that had
 * loaded the page before the header existed kept executing the bundle it
 * already had. **The throw is the point**: `anima-web` shipped this mechanism
 * inert for months because it replaced `"./dist/bundle.js"` while its own page
 * said `"/dist/bundle.js"`, so the rewrite matched nothing and returned its
 * input. A no-op string replacement is invisible unless something looks.
 */
function stampedPage() {
  const file = path.join(here, "index.html");
  const html = readFileSync(file, "utf8");
  const bundle = path.join(here, "dist/bundle.js");
  const stamp = existsSync(bundle) ? statSync(bundle).mtimeMs.toFixed(0) : "0";
  const stamped = html.replace(/(src=")(\.?\/dist\/bundle\.js)(")/, `$1$2?v=${stamp}$3`);
  if (stamped === html) throw new Error(`${file}: no <script src=".../dist/bundle.js"> to stamp — the cache buster would be a no-op`);
  return stamped;
}

/** Serves one file, honouring a single `bytes=a-b` range. */
function serve(request, response, file) {
  if (!existsSync(file) || !statSync(file).isFile()) {
    response.writeHead(404).end("not found");
    return;
  }
  const size = statSync(file).size;
  const headers = {
    "content-type": TYPES[path.extname(file)] ?? "application/octet-stream",
    "cache-control": "no-store",
    "accept-ranges": "bytes",
  };

  const range = /^bytes=(\d+)-(\d*)$/.exec(request.headers.range ?? "");
  if (!range) {
    response.writeHead(200, { ...headers, "content-length": size });
    createReadStream(file).pipe(response);
    return;
  }
  const start = Number(range[1]);
  const end = range[2] === "" ? size - 1 : Math.min(Number(range[2]), size - 1);
  if (start > end) {
    response.writeHead(416, { "content-range": `bytes */${size}` }).end();
    return;
  }
  response.writeHead(206, {
    ...headers,
    "content-length": end - start + 1,
    "content-range": `bytes ${start}-${end}/${size}`,
  });
  createReadStream(file, { start, end }).pipe(response);
}

http
  .createServer((request, response) => {
    const name = decodeURIComponent(new URL(request.url, `http://localhost:${port}`).pathname);

    if (name === "/" || name === "/index.html") {
      response.writeHead(200, { "content-type": TYPES[".html"], "cache-control": "no-store" }).end(stampedPage());
      return;
    }
    if (name.startsWith("/dist/")) {
      serve(request, response, path.join(here, name));
      return;
    }
    if (name.startsWith("/weights/")) {
      if (!weightsDir) {
        response.writeHead(404).end("started without --weights");
        return;
      }
      // `path.basename` rather than joining the URL: a `..` in the path would
      // otherwise reach outside the directory that was opted into.
      serve(request, response, path.join(weightsDir, path.basename(name)));
      return;
    }
    if (name === "/favicon.ico") {
      // Answered rather than 404'd: a console with one harmless error in it is
      // a console nobody reads.
      response.writeHead(204).end();
      return;
    }
    // Anything else off the repo root, so a source map resolves to real sources.
    serve(request, response, path.join(repoRoot, name));
  })
  .listen(port, "127.0.0.1", () => {
    console.log(`http://localhost:${port}/`);
    console.log(weightsDir ? `weights from ${weightsDir}` : "no --weights: the gate can bind a folder but not fill one");
  });
