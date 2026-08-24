/**
 * The shape of `dit.manifest.json`, with nothing else in the module.
 *
 * Separate from `weights-node.ts` because the browser loader needs these types
 * and that file imports `node:fs`. A type-only import is erased at run time but
 * **not** at type-check time: `tsc` follows it and fails on `node:fs` under a
 * DOM program, which is how this file came to exist.
 */
export interface AnimaTensor {
  name: string;
  kind: "q8" | "f32";
  shape: number[];
  /** q8 only: where the packed codes start, in u32 words. */
  codesOffset?: number;
  /** q8 only: where the per-row scales start, in f32 elements. */
  scaleOffset?: number;
  /** f32 only: where the values start, in f32 elements. */
  offset?: number;
}

export interface AnimaManifest {
  format: { quant: string };
  blocks: number;
  /** The model's own shape, as `model_detection.detect_unet_config` reads it. */
  config?: Record<string, number | boolean | string>;
  tensors: AnimaTensor[];
}
