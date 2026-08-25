#!/usr/bin/env node
/**
 * Assembles the static site GitHub Pages serves, into `_site/`.
 *
 * Issue #192. Trying a demo currently needs a clone, `npm install`, a converted
 * checkpoint on disk and a dev server. Everything `examples/anima-web` needs to
 * stop needing those is done: its weights come from a URL (#180), and it keeps
 * them in a folder the user picks rather than in the origin's storage.
 *
 * **Only the demos that actually work without a server are published.**
 * `examples/zimage-web` and `examples/llm-demo` still fetch `/weights/…` from a
 * dev server that has converted weights beside it; deploying them would put up
 * a page that 404s on its first read. The landing page lists them and says what
 * they need, which is more use than a broken link.
 *
 * **The cache-buster has to move here.** `server.mjs` stamps the bundle's mtime
 * onto the `<script src>` because `Cache-Control: no-store` was measured not to
 * be enough — a tab that had loaded the page before kept running the bundle it
 * already had. There is no server on Pages, so the stamp is baked into the file
 * this writes, and it is the bundle's **content hash** rather than a timestamp:
 * a rebuild that changes nothing should not invalidate anyone's cache.
 *
 *     node scripts/pages.mjs [--out _site]
 */
import { execFileSync } from "node:child_process";
import { copyFileSync, cpSync, mkdirSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { createHash } from "node:crypto";
import { fileURLToPath } from "node:url";
import path from "node:path";

const root = fileURLToPath(new URL("..", import.meta.url));
const at = process.argv.indexOf("--out");
const out = path.resolve(root, at >= 0 ? process.argv[at + 1] : "_site");

/**
 * What gets published, and what does not.
 *
 * `extra` are files the page fetches that are **this repository's** rather than
 * the model's — the two tokenizer vocabularies, 6.8 MB, which belong with the
 * page rather than with 5 GB of weights on a model host.
 */
const DEMOS = [
  {
    dir: "examples/anima-web",
    slug: "anima",
    title: "Anima-3.8B",
    blurb:
      "A 3.8B diffusion transformer, its Qwen3 text encoder and the Wan 2.1 VAE, all in WGSL. " +
      "Downloads 5.0 GB into a folder you pick, once.",
    extra: [
      ["llm/data/qwen-qwen3-4b.bpe-vocab.json", "weights/tokenizer/qwen-qwen3-4b.bpe-vocab.json"],
      ["examples/anima/fixtures/t5.unigram-vocab.json", "weights/tokenizer/t5.unigram-vocab.json"],
    ],
  },
];

/** Listed, not published: they cannot work without a dev server beside them. */
const NEEDS_A_CLONE = [
  ["examples/zimage-web", "Z-Image", "fetches <code>/weights/…</code> from its own server"],
  ["examples/llm-demo", "Sarashina2.2-1B", "fetches <code>/weights/…</code> from its own server"],
];

rmSync(out, { recursive: true, force: true });
mkdirSync(out, { recursive: true });

const commit = execFileSync("git", ["rev-parse", "--short", "HEAD"], { cwd: root }).toString().trim();

for (const demo of DEMOS) {
  const from = path.join(root, demo.dir);
  execFileSync("node", [path.join(from, "build.mjs")], { cwd: root, stdio: "inherit" });

  const target = path.join(out, demo.slug);
  mkdirSync(path.join(target, "dist"), { recursive: true });
  const bundle = readFileSync(path.join(from, "dist/bundle.js"));
  copyFileSync(path.join(from, "dist/bundle.js"), path.join(target, "dist/bundle.js"));

  // Content hash, not a timestamp: a rebuild that changes nothing should not
  // invalidate a cache, and a rebuild that changes something must.
  const stamp = createHash("sha256").update(bundle).digest("hex").slice(0, 12);
  const html = readFileSync(path.join(from, "index.html"), "utf8");
  const stamped = html.replace(/(src=")(\.?\/dist\/bundle\.js)(")/, `$1$2?v=${stamp}$3`);
  if (stamped === html) {
    throw new Error(
      `pages: no <script src=".../dist/bundle.js"> in ${demo.dir}/index.html to stamp. ` +
        "Without the stamp a tab keeps running the bundle it already has, and nothing says so.",
    );
  }
  writeFileSync(path.join(target, "index.html"), stamped);

  for (const [source, destination] of demo.extra ?? []) {
    const to = path.join(target, destination);
    mkdirSync(path.dirname(to), { recursive: true });
    cpSync(path.join(root, source), to);
  }
  console.log(`  ${demo.slug}/ — bundle ${stamp}`);
}

const card = (href, title, blurb, note = "") =>
  `<a class="card${href ? "" : " off"}"${href ? ` href="${href}"` : ""}>` +
  `<h2>${title}</h2><p>${blurb}</p>${note ? `<p class="need">${note}</p>` : ""}</a>`;

writeFileSync(
  path.join(out, "index.html"),
  `<!doctype html>
<meta charset="utf-8">
<title>web-xpu-ops — demos</title>
<style>
  :root { color-scheme: dark; }
  body { margin: 0 auto; padding: 3rem 2rem; max-width: 46rem;
         font: 15px/1.65 ui-sans-serif, system-ui, sans-serif; background: #14161a; color: #e6e8ec; }
  h1 { font-size: 1.4rem; margin: 0 0 .3rem; }
  .sub { color: #98a1b3; margin: 0 0 2rem; font-size: .9rem; }
  .card { display: block; text-decoration: none; color: inherit; padding: 1.1rem 1.25rem;
          border: 1px solid #2c313b; border-radius: 10px; margin-bottom: .9rem; background: #171a20; }
  .card:hover { border-color: #4f7cff; }
  .card.off { opacity: .55; }
  .card h2 { font-size: 1rem; margin: 0 0 .35rem; }
  .card p { margin: 0; color: #98a1b3; font-size: .85rem; }
  .need { margin-top: .5rem !important; color: #6b7383 !important; font-size: .78rem !important; }
  .note { color: #6b7383; font-size: .8rem; margin-top: 2rem; }
  code, a code { background: #1c1f26; padding: .1rem .3rem; border-radius: 3px; }
  a { color: #7f9cff; }
</style>
<h1>web-xpu-ops</h1>
<p class="sub">
  Inference in a browser, on WebGPU, from kernels verified one at a time against
  their own references. <a href="https://github.com/m96-chan/web-xpu-ops">Source</a>, MIT.
</p>

${DEMOS.map((d) => card(`${d.slug}/`, d.title, d.blurb)).join("\n")}
${NEEDS_A_CLONE.map(([dir, title, why]) =>
  card("", title, `Runs from a checkout: <code>${dir}</code>.`, `Not published here — it ${why}.`),
).join("\n")}

<p class="note">
  <strong>The models are not MIT.</strong> Anima is
  <strong>non-commercial</strong>: it descends from
  <a href="https://huggingface.co/circlestone-labs/Anima">circlestone-labs/Anima</a>
  under the CircleStone Labs Non-Commercial License, itself built on
  <a href="https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-open-model-license">NVIDIA Cosmos</a>.
  Running MIT code against a checkpoint does not relicense the checkpoint.
  Built on NVIDIA Cosmos.
</p>
<p class="note">Built from <code>${commit}</code>.</p>
`,
);

// Pages runs Jekyll otherwise, which drops files and directories beginning
// with an underscore without saying so.
writeFileSync(path.join(out, ".nojekyll"), "");
console.log(`site in ${path.relative(process.cwd(), out)}`);
