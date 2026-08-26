/**
 * Qwen3-VL's image processor — pixels into `pixel_values` and `image_grid_thw`.
 *
 * Issue #212. Between a reference's pixels and `./vision.ts` sit a resize, a
 * normalisation and a patchify, and each is a convention rather than a choice
 * this port gets to make. Ported from `Qwen2VLImageProcessor`, which is the
 * class `Qwen3VLProcessor` uses.
 *
 * **The resize is not here, and that is a stated limitation.**
 * `Qwen2VLImageProcessor` resamples with **PIL's bicubic** — a real filter with
 * its own support-scaling rule for downsampling. A browser has `drawImage`,
 * which is not it. So `patchify` takes pixels that are **already** the size
 * `smartResize` asked for, and what a browser's own resampler costs against
 * PIL's is **unmeasured**. Matching it is its own piece of work.
 *
 * What is here is exact:
 *
 * - **`smartResize`** decides the target, and its rounding is Python's
 *   **banker's** at a factor of `patchSize * mergeSize`. Above the pixel
 *   ceiling it uses `floor`; below the floor, `ceil` — two different rules, and
 *   using one for both is a size that is merely close.
 * - **A still image is repeated to fill the temporal patch.** One image becomes
 *   two frames, so a port that emitted one produces half a patch at the right
 *   shape.
 * - **The patchify permutation lands tokens in merge-block order**, each row
 *   laid out `C, t, py, px` — the order `./vision.ts` reads, and the reason
 *   these two files cannot be checked independently and then disagree.
 */

export interface ProcessorConfig {
  patchSize: number;
  mergeSize: number;
  temporalPatchSize: number;
  minPixels: number;
  maxPixels: number;
  /** Per channel, applied after the `1/255` rescale. */
  imageMean: [number, number, number];
  imageStd: [number, number, number];
}

/** Python's `round`: half to even. `Math.round(0.5)` is 1 and `round(0.5)` is 0. */
function roundHalfToEven(x: number): number {
  const floor = Math.floor(x);
  const rest = x - floor;
  if (rest > 0.5) return floor + 1;
  if (rest < 0.5) return floor;
  return floor % 2 === 0 ? floor : floor + 1;
}

/**
 * The size Qwen3-VL wants an image at: a multiple of `factor` on both axes,
 * within the pixel range, aspect ratio kept as closely as it can be.
 *
 * Refuses past a 200:1 aspect ratio, as upstream does — a limit worth keeping
 * because the alternative is an image whose short edge rounds to zero.
 */
export function smartResize(
  height: number, width: number, factor: number, minPixels: number, maxPixels: number,
): [number, number] {
  const long = Math.max(height, width);
  const short = Math.min(height, width);
  if (long / short > 200) {
    throw new Error(`smartResize: absolute aspect ratio must be smaller than 200, got ${long / short}`);
  }
  let h = roundHalfToEven(height / factor) * factor;
  let w = roundHalfToEven(width / factor) * factor;
  if (h * w > maxPixels) {
    // **`floor` above the ceiling.** Using `ceil` here would step back over it.
    const beta = Math.sqrt((height * width) / maxPixels);
    h = Math.max(factor, Math.floor(height / beta / factor) * factor);
    w = Math.max(factor, Math.floor(width / beta / factor) * factor);
  } else if (h * w < minPixels) {
    // **`ceil` below the floor**, for the mirror reason.
    const beta = Math.sqrt(minPixels / (height * width));
    h = Math.ceil((height * beta) / factor) * factor;
    w = Math.ceil((width * beta) / factor) * factor;
  }
  return [h, w];
}

export interface PatchifyResult {
  /** `[gridT * gridH * gridW, C * temporal * patch * patch]`, in merge-block order. */
  pixelValues: Float32Array;
  /** `[t, h, w]`, in patches. */
  grid: [number, number, number];
}

/**
 * `[frames][height][width][3]` bytes into the tower's patches.
 *
 * `frames` is the video's own frame count, **before** the temporal repeat: a
 * still image is one frame here and becomes two patches' worth. The pixels are
 * `uint8` in `[0, 255]` and are rescaled by `1/255` and normalised — which at
 * the shipped mean and std of 0.5 is `x / 127.5 - 1`, but is written as the two
 * steps upstream takes so a different checkpoint's numbers still work.
 */
export function patchify(
  frames: Uint8Array[], height: number, width: number, cfg: ProcessorConfig,
): PatchifyResult {
  const { patchSize: ps, mergeSize: m, temporalPatchSize: tp } = cfg;
  const channels = 3;
  if (height % (ps * m) || width % (ps * m)) {
    throw new Error(
      `patchify: ${height}x${width} is not a whole number of ${ps * m}x${ps * m} blocks — ` +
        "resize to what smartResize asked for first",
    );
  }
  for (const frame of frames) {
    if (frame.length !== height * width * channels) {
      throw new Error(`patchify: a frame holds ${frame.length} bytes, not ${height * width * channels}`);
    }
  }

  // **The last frame repeats to fill the temporal patch.** A still image is one
  // frame and becomes two.
  const padded = frames.slice();
  const remainder = padded.length % tp;
  if (remainder !== 0) {
    const last = padded[padded.length - 1];
    if (!last) throw new Error("patchify: no frames");
    for (let i = 0; i < tp - remainder; i += 1) padded.push(last);
  }

  const gridT = padded.length / tp;
  const gridH = height / ps;
  const gridW = width / ps;
  const rowWidth = channels * tp * ps * ps;
  const out = new Float32Array(gridT * gridH * gridW * rowWidth);

  // `(0, 3, 6, 4, 7, 2, 1, 5, 8)` written as loops: block row, block column,
  // then within the block, then channel, then time, then the patch's own rows
  // and columns.
  let at = 0;
  for (let t = 0; t < gridT; t += 1) {
    for (let by = 0; by < gridH / m; by += 1) {
      for (let bx = 0; bx < gridW / m; bx += 1) {
        for (let iy = 0; iy < m; iy += 1) {
          for (let ix = 0; ix < m; ix += 1) {
            for (let c = 0; c < channels; c += 1) {
              for (let dt = 0; dt < tp; dt += 1) {
                const frame = padded[t * tp + dt]!;
                for (let py = 0; py < ps; py += 1) {
                  for (let px = 0; px < ps; px += 1) {
                    const y = (by * m + iy) * ps + py;
                    const x = (bx * m + ix) * ps + px;
                    // **Rounded to f32 after each step**, because upstream is
                    // two numpy operations on float32 arrays and not one
                    // expression: `rescale` writes a float32, `normalize` reads
                    // it. Doing both in f64 and narrowing once disagrees by
                    // 5.9e-8 — small, and not zero, which is the difference
                    // between a check that can be exact and one that needs a
                    // tolerance nobody can justify.
                    const value = Math.fround(frame[(y * width + x) * channels + c]! / 255);
                    out[at] = Math.fround((value - cfg.imageMean[c]!) / cfg.imageStd[c]!);
                    at += 1;
                  }
                }
              }
            }
          }
        }
      }
    }
  }
  return { pixelValues: out, grid: [gridT, gridH, gridW] };
}

/** Vision tokens a grid produces once the merger has run: `t * h * w / merge²`. */
export function visionTokenCount(grid: [number, number, number], mergeSize: number): number {
  return (grid[0] * grid[1] * grid[2]) / (mergeSize * mergeSize);
}
