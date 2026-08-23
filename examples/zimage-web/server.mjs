#!/usr/bin/env node
/**
 * Serves the Z-Image browser demo, and the weights it streams.
 *
 * Two things this has to do that a plain static server does not:
 *
 *   1. **Answer `Range` requests.** Every weight read is a byte range — the DiT
 *      is 6.17 GB and the text encoder 8 GB, and the demo fetches one tensor at
 *      a time. A server that ignored `Range` would send the whole file per
 *      tensor; `fetch-weights.ts` treats a 200 as an error rather than as a slow
 *      success, so that failure is loud instead of a hang.
 *   2. **Map three separate directories under `/weights`.** The converted DiT
 *      lives wherever `convert_dit.py` wrote it, the text encoder in the
 *      Hugging Face cache, and the VAE fixture in the repository. Copying
 *      gigabytes to make one root would be the obvious alternative and is not
 *      one.
 *
 *     node examples/zimage-web/server.mjs --dit ~/zimage-q8
 *
 * Nothing here is meant for the public internet: it serves whatever the paths
 * below point at, to localhost, for a demo.
 */
import { createReadStream, existsSync, readdirSync, statSync } from "node:fs";
import http from "node:http";
import { homedir } from "node:os";
import path from "node:path";
import { fileURLToPath } from "node:url";

const here = path.dirname(fileURLToPath(import.meta.url));
const repoRoot = path.resolve(here, "../..");

function arg(flag, fallback) {
  const at = process.argv.indexOf(flag);
  return at >= 0 && process.argv[at + 1] !== undefined ? process.argv[at + 1] : fallback;
}

function modelRoot() {
  const explicit = process.env.ZIMAGE_MODEL_DIR;
  if (explicit) return explicit;
  const hub = path.join(process.env.HF_HOME ?? path.join(homedir(), ".cache", "huggingface"), "hub");
  const snapshots = path.join(hub, "models--Tongyi-MAI--Z-Image", "snapshots");
  if (!existsSync(snapshots)) return null;
  return path.join(snapshots, readdirSync(snapshots)[0]);
}

const ditDir = path.resolve(arg("--dit", process.env.ZIMAGE_DIT_DIR ?? ""));
const port = Number(arg("--port", "8788"));
const root = modelRoot();

if (!ditDir || !existsSync(ditDir)) {
  console.error(
    "server: pass --dit <dir>, the output of examples/zimage/tools/convert_dit.py.\n" +
      "That directory holds dit.manifest.json and the blobs beside it.",
  );
  process.exit(2);
}
if (!root) {
  console.error("server: no Z-Image checkpoint found. Set ZIMAGE_MODEL_DIR to a directory holding text_encoder/.");
  process.exit(2);
}

/** URL prefix to the directory behind it. Longest prefix wins. */
const mounts = [
  ["/weights/dit/", ditDir],
  ["/weights/text_encoder/", path.join(root, "text_encoder")],
  ["/weights/vae/", path.join(repoRoot, "examples/zimage-vae/fixtures-small")],
  ["/weights/tokenizer/", path.join(repoRoot, "llm/data")],
  ["/dist/", path.join(here, "dist")],
  ["/", here],
];

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

const server = http.createServer((req, res) => {
  const file = resolve(req.url ?? "/");
  if (!file) {
    res.writeHead(404, { "content-type": "text/plain" });
    res.end("not found\n");
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
    });
    createReadStream(file, { start, end }).pipe(res);
    return;
  }

  res.writeHead(200, { "content-type": type, "content-length": size, "accept-ranges": "bytes" });
  createReadStream(file).pipe(res);
});

server.listen(port, "127.0.0.1", () => {
  console.log(`Z-Image demo on http://127.0.0.1:${port}`);
  console.log(`  DiT          ${ditDir}`);
  console.log(`  text encoder ${path.join(root, "text_encoder")}`);
  console.log(`  VAE          examples/zimage-vae/fixtures-small`);
  console.log("\nNeeds a browser with WebGPU. Nothing leaves this machine.");
});
