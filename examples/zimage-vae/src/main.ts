/**
 * Decodes a Z-Image latent to a PNG with this repository's ops.
 *
 *     node --experimental-strip-types examples/zimage-vae/src/main.ts \
 *         --fixtures examples/zimage-vae/fixtures-small
 *
 * Writes `decoded.png` next to the fixtures and prints how far it lands from
 * the model's own decode of the same latent. The number is the point as much
 * as the picture: a decoder with a transposed convolution still produces
 * something image-shaped, so "it looks right" is not a check.
 */
import { existsSync, readFileSync, writeFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { type DecoderConfig, decode } from "./decoder.js";
import { encodePng } from "./render.js";

interface Entry {
  name: string;
  shape: number[];
  offset: number;
  length: number;
}
interface Manifest {
  config: Record<string, number | number[] | boolean>;
  latent: Entry[];
  decoder: Entry[];
}

function arg(flag: string, fallback: string): string {
  const at = process.argv.indexOf(flag);
  return at >= 0 && process.argv[at + 1] !== undefined ? process.argv[at + 1]! : fallback;
}

const dir = arg("--fixtures", "examples/zimage-vae/fixtures-small");
const base = new URL(`file://${process.cwd()}/${dir}/`);

const weightsPath = fileURLToPath(new URL("decoder.bin", base));
if (!existsSync(weightsPath)) {
  console.error(
    `no decoder weights at ${weightsPath}\n` +
      `They are 190 MB and not committed. Regenerate with:\n` +
      `  /home/m96-chan/project/therdparty/musubi-tuner/.venv/bin/python \\\n` +
      `      examples/zimage-vae/tools/gen_latent.py --size 64 --out ${dir}`,
  );
  process.exit(1);
}

const manifest = JSON.parse(readFileSync(fileURLToPath(new URL("manifest.json", base)), "utf8")) as Manifest;
const read = (file: string): Float32Array => {
  const raw = readFileSync(fileURLToPath(new URL(file, base)));
  return new Float32Array(raw.buffer, raw.byteOffset, raw.byteLength / 4);
};
const latentBlob = read("latent.bin");
const weightBlob = read("decoder.bin");
const pick = (blob: Float32Array, table: Entry[], name: string): Float32Array => {
  const e = table.find((t) => t.name === name);
  if (!e) throw new Error(`fixture has no "${name}"`);
  return blob.subarray(e.offset, e.offset + e.length);
};

const c = manifest.config as unknown as {
  block_out_channels: number[];
  layers_per_block: number;
  norm_num_groups: number;
  latent_channels: number;
  out_channels: number;
  scaling_factor: number;
  shift_factor: number;
};
const cfg: DecoderConfig = {
  blockOutChannels: c.block_out_channels,
  layersPerBlock: c.layers_per_block,
  normNumGroups: c.norm_num_groups,
  latentChannels: c.latent_channels,
  outChannels: c.out_channels,
  scalingFactor: c.scaling_factor,
  shiftFactor: c.shift_factor,
};

const latentEntry = manifest.latent.find((e) => e.name === "latent")!;
const [, , lh, lw] = latentEntry.shape;

console.log(`latent ${latentEntry.shape.join("x")} -> image ${lh! * 8}x${lw! * 8}`);
const started = performance.now();
const out = decode(cfg, (n) => pick(weightBlob, manifest.decoder, n), pick(latentBlob, manifest.latent, "latent"), lh!, lw!);
const elapsed = performance.now() - started;

const png = encodePng(out.data, out.H, out.W);
const target = fileURLToPath(new URL("decoded.png", base));
writeFileSync(target, png);

const want = pick(latentBlob, manifest.latent, "reference");
let worst = 0;
for (let i = 0; i < want.length; i += 1) worst = Math.max(worst, Math.abs(out.data[i]! - want[i]!));

console.log(`decoded in ${(elapsed / 1000).toFixed(1)}s on the CPU reference path`);
console.log(`worst |ours - model| = ${worst.toExponential(3)} on pixels in [-1, 1]`);
console.log(`wrote ${target}`);
