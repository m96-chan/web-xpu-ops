#!/usr/bin/env node
/**
 * A static server for the Anima demo that answers `Range` requests.
 *
 * The page reads every weight as a byte range, so a server that ignores `Range`
 * would return whole files — 3.76 GB per tensor — and `fetch-weights.ts`
 * refuses a 200 for exactly that reason. `node:http`'s default handler does not
 * do ranges, so this exists.
 *
 *     node examples/anima-web/server.mjs \
 *       --dit ~/anima-q8 --encoder ~/anima-src/qwen_3_06b_base.safetensors \
 *       --vae ~/anima-src/qwen_image_vae.safetensors
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

const ditDir = path.resolve(arg("--dit", process.env.ANIMA_DIT_DIR ?? ""));
const encoderFile = path.resolve(arg("--encoder", process.env.ANIMA_ENCODER ?? ""));
const vaeFile = path.resolve(arg("--vae", process.env.ANIMA_VAE ?? ""));
const port = Number(arg("--port", "8789"));

const missing = [
  [ditDir, "--dit <dir>, the output of examples/anima/tools/convert_dit.py"],
  [encoderFile, "--encoder <qwen_3_06b_base.safetensors>"],
  [vaeFile, "--vae <qwen_image_vae.safetensors>"],
].filter(([p]) => !p || !existsSync(p));
if (missing.length > 0) {
  console.error("server: missing\n" + missing.map(([, why]) => `  ${why}`).join("\n"));
  process.exit(2);
}

/**
 * URL prefix to the directory behind it. Longest prefix wins.
 *
 * The encoder and the VAE are single files rather than directories, so they are
 * mounted by their parent and reached by name — the page asks for
 * `/weights/encoder/qwen_3_06b_base.safetensors`, and a file that is not the
 * one named here simply is not found.
 */
const mounts = [
  ["/weights/dit/", ditDir],
  ["/weights/encoder/", path.dirname(encoderFile)],
  ["/weights/vae/", path.dirname(vaeFile)],
  // Both tokenizer vocabularies: the Qwen BPE is the one this repository
  // already carries for Qwen3-4B — byte-for-byte the same 151,643 pieces — and
  // the T5 unigram is Anima's own fixture.
  ["/weights/tokenizer/", path.join(repoRoot, "llm/data")],
  ["/weights/tokenizer/", path.join(repoRoot, "examples/anima/fixtures")],
  ["/dist/", path.join(here, "dist")],
  ["/", here],
];

/**
 * The demo's own files must not be cached; the weights should be.
 *
 * Without this the browser caches `bundle.js` heuristically — no
 * `Cache-Control`, so it invents one — and a rebuilt demo silently keeps
 * running the old code. That happened: a fixed `decodeGpu` call kept throwing
 * the error it had already been fixed for, and the bundle on disk, the bundle
 * the server returned and the bundle the page was executing were three
 * different questions.
 *
 * The weights are the opposite case. They are gigabytes and they never change
 * for a given conversion, so they are the one thing here worth caching hard.
 */
function cacheControl(file) {
  return file.includes("/weights/") || /\.(bin|safetensors)$/.test(file)
    ? "public, max-age=31536000, immutable"
    : "no-store";
}

const types = {
  ".html": "text/html; charset=utf-8",
  ".js": "text/javascript; charset=utf-8",
  ".map": "application/json; charset=utf-8",
  ".json": "application/json; charset=utf-8",
  ".bin": "application/octet-stream",
  ".safetensors": "application/octet-stream",
  ".png": "image/png",
};

function resolve(urlPath) {
  const clean = decodeURIComponent(urlPath.split("?")[0]);
  for (const [prefix, dir] of mounts) {
    if (!clean.startsWith(prefix)) continue;
    const rest = clean.slice(prefix.length) || "index.html";
    const full = path.join(dir, rest);
    // `path.join` collapses `..`, so this compares the resolved path against
    // the mount it claimed to be under. Without it, `/weights/dit/../../etc`
    // reads whatever it likes.
    if (!full.startsWith(path.resolve(dir))) return null;
    if (existsSync(full) && statSync(full).isFile()) return full;
  }
  return null;
}

/**
 * `index.html`, with the bundle's mtime appended to its `<script src>`.
 *
 * `Cache-Control: no-store` should already be enough, and was not: a tab that
 * had loaded the page before the header existed kept executing the bundle it
 * already had, and "reload" is advice rather than a mechanism. Changing the URL
 * is a mechanism — no cache, at any layer, can serve a different resource for a
 * name it has never seen.
 *
 * Done here rather than in the file so that `index.html` stays a plain document
 * in the repository, and so the stamp cannot go stale relative to the build.
 */
function indexWithStamp(file) {
  const bundle = path.join(here, "dist/bundle.js");
  const stamp = existsSync(bundle) ? statSync(bundle).mtimeMs.toFixed(0) : "0";
  return readFileSync(file, "utf8").replace("./dist/bundle.js", `./dist/bundle.js?v=${stamp}`);
}

/**
 * One line per request, because "is the page actually asking for anything?" is
 * otherwise unanswerable from outside the browser — and the first run of this
 * demo was diagnosed by not being able to answer it.
 */
function log(req, status, bytes) {
  const range = req.headers.range ? ` ${req.headers.range}` : "";
  console.log(`${status} ${req.url}${range}${bytes === undefined ? "" : ` (${bytes}B)`}`);
}

const server = http.createServer((req, res) => {
  const file = resolve(req.url ?? "/");
  if (!file) {
    log(req, 404);
    res.writeHead(404, { "content-type": "text/plain" });
    res.end("not found\n");
    return;
  }

  if (file.endsWith("index.html")) {
    const body = Buffer.from(indexWithStamp(file), "utf8");
    res.writeHead(200, {
      "content-type": types[".html"],
      "content-length": body.byteLength,
      "cache-control": "no-store",
    });
    log(req, 200, body.length);
    res.end(body);
    return;
  }

  const size = statSync(file).size;
  const type = types[path.extname(file)] ?? "application/octet-stream";
  const range = req.headers.range;

  if (range) {
    const match = /^bytes=(\d+)-(\d*)$/.exec(range);
    if (!match) {
      res.writeHead(416, { "content-range": `bytes */${size}` });
      res.end();
      return;
    }
    const start = Number(match[1]);
    const end = match[2] === "" ? size - 1 : Math.min(Number(match[2]), size - 1);
    if (start >= size || end < start) {
      res.writeHead(416, { "content-range": `bytes */${size}` });
      res.end();
      return;
    }
    res.writeHead(206, {
      "content-type": type,
      "content-length": end - start + 1,
      "content-range": `bytes ${start}-${end}/${size}`,
      "accept-ranges": "bytes",
      "cache-control": cacheControl(file),
    });
    log(req, 206, end - start + 1);
    createReadStream(file, { start, end }).pipe(res);
    return;
  }

  res.writeHead(200, {
    "content-type": type,
    "content-length": size,
    "accept-ranges": "bytes",
    "cache-control": cacheControl(file),
  });
  log(req, 200, size);
  createReadStream(file).pipe(res);
});

server.listen(port, "127.0.0.1", () => {
  console.log(`Anima demo on http://127.0.0.1:${port}`);
  console.log(`  DiT          ${ditDir}`);
  console.log(`  text encoder ${encoderFile}`);
  console.log(`  VAE          ${vaeFile}`);
  console.log("\nNeeds a browser with WebGPU. Nothing leaves this machine.");
});
