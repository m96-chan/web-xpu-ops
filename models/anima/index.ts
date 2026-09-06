/**
 * `web-xpu-ops/models/anima` — the Anima-3.8B port, as a library.
 *
 * Issue #224. Everything a page needs to go from a prompt to a latent to an
 * image, and nothing that needs Node: the resident DiT forward, the CPU
 * reference forwards it is held to, the two tokenizers, the Qwen3-0.6B encoder
 * (CPU and per-dispatch GPU), the adapter, the sampler, the VAE decoder, and
 * the browser device/runner the demo drives them with. The Node-only halves —
 * `weights-node.ts`, `generate.ts`, the `verify-*` scripts,
 * `zimage/src/safetensors.ts` — stay in `examples/`; the browser weight
 * loader is `web-xpu-ops/models/anima/fetch-weights`, and the WGSL a host has
 * to supply is described by `web-xpu-ops/models/anima/kernels`.
 *
 * The source files keep their places under `examples/`; this is a barrel over
 * them, so a demo and a consumer run the same code and the `verify-*` scripts
 * keep holding it to the goldens.
 */
export * from "../../examples/anima/src/dit-resident.js";
export * from "../../examples/anima/src/dit.js";
export * from "../../examples/anima/src/block.js";
export * from "../../examples/anima/src/sampler.js";
export * from "../../examples/anima/src/text-encoder.js";
export * from "../../examples/anima/src/tokenize.js";
export * from "../../examples/anima/src/vae-gpu.js";
export * from "../../examples/anima/src/vae.js";
export type * from "../../examples/anima/src/manifest.js";
export * from "../../examples/zimage/src/text-encoder.js";
export * from "../../examples/zimage/src/text-encoder-gpu.js";
export { type DitKernels, DIT_KERNEL_SOURCES, type PackedWeightSource } from "../../examples/zimage/src/dit-gpu.js";
export { dequantizeQ8 } from "../../examples/zimage/src/weights.js";
export { bf16ToF32 } from "../../examples/zimage/src/bf16.js";
export * from "../../examples/web-common/src/byte-source.js";
export * from "../../examples/web-common/src/browser-resident.js";
export * from "../../examples/web-common/src/browser-runner.js";
export type { ResidentDevice, Runner } from "../../harness/api.js";
