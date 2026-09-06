/**
 * The image processor against `transformers`' own `Qwen2VLImageProcessor`.
 *
 * Issue #212. Committed fixture — pixels and their patches, no weights, no
 * model licence. The golden is generated with **`do_resize=False`**, because
 * upstream resizes with PIL's bicubic and a browser has `drawImage`; see
 * `processor.ts` for what that leaves unmeasured.
 */
import { readFileSync } from "node:fs";
import { describe, expect, it } from "vitest";
import { patchify, smartResize, visionTokenCount, type ProcessorConfig } from "./processor.js";

const here = new URL("../fixtures/", import.meta.url);
const manifest = JSON.parse(readFileSync(new URL("processor.json", here), "utf8")) as {
  patchSize: number; mergeSize: number; temporalPatchSize: number;
  factor: number; minPixels: number; maxPixels: number;
  imageMean: [number, number, number]; imageStd: [number, number, number];
  smartResize: { height: number; width: number; resized?: [number, number]; error?: string }[];
  image: { height: number; width: number; channels: number; bytes: number };
  patches: { rows: number; cols: number; offset: number };
  grid: [number, number, number];
};
const blob = readFileSync(new URL("processor.bin", here));

const cfg: ProcessorConfig = {
  patchSize: manifest.patchSize,
  mergeSize: manifest.mergeSize,
  temporalPatchSize: manifest.temporalPatchSize,
  minPixels: manifest.minPixels,
  maxPixels: manifest.maxPixels,
  imageMean: manifest.imageMean,
  imageStd: manifest.imageStd,
};

const pixels = new Uint8Array(blob.buffer, blob.byteOffset, manifest.image.bytes);
const wantPatches = new Float32Array(
  blob.buffer.slice(
    blob.byteOffset + manifest.patches.offset,
    blob.byteOffset + manifest.patches.offset + manifest.patches.rows * manifest.patches.cols * 4,
  ),
);

describe("h3 ref2v / processor", () => {
  it("covers every branch of smartResize", () => {
    // Already conforming, above the ceiling, below the floor, and refused.
    const results = manifest.smartResize;
    expect(results.length).toBeGreaterThan(4);
    expect(results.some((r) => r.error)).toBe(true);
    expect(results.some((r) => r.resized && r.height * r.width > manifest.maxPixels)).toBe(true);
    expect(results.some((r) => r.resized && r.height * r.width < manifest.minPixels)).toBe(true);
  });

  it("reproduces smartResize", () => {
    for (const c of manifest.smartResize) {
      if (c.error) {
        expect(() => smartResize(c.height, c.width, manifest.factor, manifest.minPixels, manifest.maxPixels),
          `${c.height}x${c.width}`).toThrow(/aspect ratio/);
        continue;
      }
      expect(smartResize(c.height, c.width, manifest.factor, manifest.minPixels, manifest.maxPixels),
        `${c.height}x${c.width}`).toEqual(c.resized);
    }
  });

  it("rounds half to even, like Python and not like JavaScript", () => {
    // 48 / 32 is exactly 1.5. `Math.round` gives 2 and Python gives 2 as well —
    // but 80 / 32 is 2.5, where `Math.round` gives 3 and Python gives 2. The
    // clamp then hides it unless the result is inside the pixel range, which is
    // why this is checked directly rather than through a size.
    expect(smartResize(80, 4096, 32, 1, 1e9)).toEqual([64, 4096]);
    expect(Math.round(80 / 32) * 32).toBe(96);
  });

  it("reproduces the patches, exactly", () => {
    const { height, width } = manifest.image;
    const got = patchify([pixels], height, width, cfg);
    expect(got.grid).toEqual(manifest.grid);
    expect(got.pixelValues.length).toBe(manifest.patches.rows * manifest.patches.cols);
    let worst = 0;
    for (let i = 0; i < wantPatches.length; i += 1) {
      worst = Math.max(worst, Math.abs(got.pixelValues[i]! - wantPatches[i]!));
    }
    console.log(`h3 ref2v processor: patches worst ${worst.toExponential(3)}`);
    // The same f32 arithmetic on both sides: `x / 255`, then `(x - 0.5) / 0.5`.
    expect(worst).toBe(0);
  });

  it("repeats a still image to fill the temporal patch", () => {
    const { height, width } = manifest.image;
    const got = patchify([pixels], height, width, cfg);
    // One frame in, one grid frame out — because two patches were made from it.
    expect(got.grid[0]).toBe(1);
    const perFrame = manifest.patchSize * manifest.patchSize * 3;
    // Inside a row the layout is `C, t, py, px`, so the two time slices of a
    // channel are adjacent and identical for a still.
    const row = got.pixelValues.subarray(0, manifest.patches.cols);
    const stride = manifest.patchSize * manifest.patchSize;
    for (let c = 0; c < 3; c += 1) {
      const first = row.subarray(c * 2 * stride, c * 2 * stride + stride);
      const second = row.subarray(c * 2 * stride + stride, (c + 1) * 2 * stride);
      expect([...first], `channel ${c}`).toEqual([...second]);
    }
    expect(perFrame * 2).toBe(manifest.patches.cols);
  });

  it("lands tokens in merge-block order", () => {
    // The first two rows are the top-left merge block's first two patches,
    // which are horizontally adjacent — **not** the first two patches of the
    // raster row when the image is wider than one block.
    const { height, width } = manifest.image;
    const got = patchify([pixels], height, width, cfg);
    const cols = manifest.patches.cols;
    // Row 0 starts at pixel (0, 0) and row 1 at (0, patchSize): red channel,
    // first time slice, first pixel.
    expect(got.pixelValues[0]).toBeCloseTo(pixels[0]! / 255 * 2 - 1, 6);
    const at = (y: number, x: number): number => (pixels[(y * width + x) * 3]! / 255 - 0.5) / 0.5;
    expect(got.pixelValues[cols]).toBeCloseTo(at(0, manifest.patchSize), 6);
    // And row 2 is the block's second **row** of patches, not its third column.
    expect(got.pixelValues[2 * cols]).toBeCloseTo(at(manifest.patchSize, 0), 6);
  });

  it("counts the vision tokens a grid produces", () => {
    // What the presentation needs for its pad run: the tower's tokens divided
    // by the merge, because a merger eats `merge²` of them.
    expect(visionTokenCount(manifest.grid, manifest.mergeSize)).toBe(
      (manifest.grid[0] * manifest.grid[1] * manifest.grid[2]) / (manifest.mergeSize ** 2),
    );
  });

  it("refuses pixels that were never resized", () => {
    // The resize is not in this port, so an image that does not already
    // conform has to be refused rather than silently cropped.
    expect(() => patchify([new Uint8Array(3 * 40 * 40)], 40, 40, cfg))
      .toThrow(/not a whole number of 32x32 blocks/);
    expect(() => patchify([new Uint8Array(8)], manifest.image.height, manifest.image.width, cfg))
      .toThrow(/8 bytes, not/);
  });
});
