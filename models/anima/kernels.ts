/**
 * Building an Anima kernel table from WGSL a host already holds.
 *
 * The DiT, the encoder and the VAE each take their shaders as a plain
 * `{ name: wgslSource }` object (`DitKernels`, `EncoderKernels`,
 * `VaeKernels`), and each publishes the list of `(op, entry)` pairs it will
 * read (`DIT_KERNEL_SOURCES` and friends). Node fills the tables with
 * `readFileSync`; a page fills them with whatever its bundler inlined. Both
 * are the same loop, which is the one below — a consumer that bundles or
 * fetches `web-xpu-ops/ops/<op>/wgsl/<entry>.wgsl` itself hands the loop a
 * `load(op, entry)` and gets a table whose keys the forwards recognise.
 *
 * `ANIMA_KERNEL_FILES` is the union, so a host can fetch exactly what a
 * generation needs and nothing else — 20 files, not the package's 60.
 */
import { DIT_KERNEL_SOURCES, type DitKernels } from "../../examples/zimage/src/dit-gpu.js";
import { ENCODER_KERNEL_SOURCES, type EncoderKernels } from "../../examples/zimage/src/text-encoder-gpu.js";
import { VAE_KERNEL_SOURCES, type VaeKernels } from "../../examples/anima/src/vae-gpu.js";

export type { DitKernels, EncoderKernels, VaeKernels };
export { DIT_KERNEL_SOURCES, ENCODER_KERNEL_SOURCES, VAE_KERNEL_SOURCES };

/** One `(op, entry)` the tables read, under the key a forward looks it up by. */
export interface KernelSource<K extends string = string> {
  key: K;
  op: string;
  entry: string;
}

/** `(op, entry)` → WGSL text. Synchronous: fetch before, not during. */
export type KernelLoader = (op: string, entry: string) => string;

/** Fills one table. Every key in `sources` is set; a loader that returns anything but a string is an error, not an empty shader. */
export function kernelsFromSources<K extends string>(sources: readonly KernelSource<K>[], load: KernelLoader): Record<K, string> {
  const out = {} as Record<K, string>;
  for (const { key, op, entry } of sources) {
    const source = load(op, entry);
    if (typeof source !== "string") {
      throw new Error(`kernelsFromSources: loader returned ${typeof source} for ops/${op}/wgsl/${entry}.wgsl (key ${JSON.stringify(key)})`);
    }
    out[key] = source;
  }
  return out;
}

/** The three tables a generation dispatches through, from one loader. */
export function animaKernels(load: KernelLoader): { dit: DitKernels; encoder: EncoderKernels; vae: VaeKernels } {
  return {
    dit: kernelsFromSources(DIT_KERNEL_SOURCES, load),
    encoder: kernelsFromSources(ENCODER_KERNEL_SOURCES, load),
    vae: kernelsFromSources(VAE_KERNEL_SOURCES, load),
  };
}

/** `ops/<op>/wgsl/<entry>.wgsl`, relative to the package root, for everything `animaKernels` will ask for — deduplicated, sorted. */
export const ANIMA_KERNEL_FILES: readonly string[] = [
  ...new Set(
    [...DIT_KERNEL_SOURCES, ...ENCODER_KERNEL_SOURCES, ...VAE_KERNEL_SOURCES].map(({ op, entry }) => `ops/${op}/wgsl/${entry}.wgsl`),
  ),
].sort();

/** The subpath `ANIMA_KERNEL_FILES[i]` resolves to under this package's `exports`: `web-xpu-ops/ops/<op>/wgsl/<entry>.wgsl`. */
export function kernelSubpath(op: string, entry: string): string {
  return `web-xpu-ops/ops/${op}/wgsl/${entry}.wgsl`;
}
