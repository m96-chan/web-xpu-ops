/**
 * `encode_temporal`: the path a conditioning reference actually takes.
 *
 * Issue #216. `examples/h3-encoder` reproduces `EncoderFCN3D` + `quant_conv` to
 * 9.537e-6, and that pair is `AutoencoderKLLegacy.encode` — which
 * `MiniMaxH3`'s `encode_base` calls **only for a single image**. A frame stack
 * goes through `encode_temporal` (`video_vae/klvae.py`, lines 461-493): the clip
 * is padded up to a multiple of `clip_length` by repeating its last frame, each
 * chunk is encoded **on its own** so the causal state restarts, and `token_drop`
 * latent frames come off the end of the concatenation.
 *
 * Measured with the released weights, `encode` against `encode_temporal` on the
 * same clip (`tools/measure_temporal_chunking.py`):
 *
 *     frames        encode      temporal   rms difference
 *          8      48x2x2x2      48x2x2x2             0.0%
 *         17      48x5x2x2      48x2x2x2   different shape
 *         22      48x6x2x2      48x7x2x2   different shape
 *         48     48x12x2x2     48x12x2x2            17.9%
 *         68     48x17x2x2     48x17x2x2            19.0%
 *         85     48x22x2x2     48x22x2x2            21.5%
 *
 * A video reference is 2 to 15 seconds at 24 fps — 48 to 360 frames — so this
 * is the whole of that range. Eight frames agree because the encoder is causal:
 * two latent frames depend only on the first eight pixel frames, and the padding
 * that follows them is exactly what `token_drop` removes. That coincidence is
 * why `verify-encode-gpu.ts`'s 8x32x32 clip could not see any of this.
 *
 * The per-clip encode is a parameter rather than a method, so the chunking can
 * be checked without a device, without weights and without a golden — and the
 * stand-in can be asked which frames each chunk received, which a real encoder
 * cannot.
 */

/** A tensor and the shape live in it, laid out channel-major `[C][D][H][W]`. */
export interface Moments {
  data: Float32Array;
  C: number;
  D: number;
  H: number;
  W: number;
}

/** `AutoencoderKLLegacy.encode` for one clip: `quant_conv(encoder(x))`. */
export type EncodeClip = (
  clip: Float32Array, T: number, H: number, W: number,
) => Promise<Moments>;

export interface Chunking {
  /** `vae_clip_length`. Pixel frames per chunk. */
  clipLength: number;
  /** `vae_token_drop`. Latent frames taken off the end of the concatenation. */
  tokenDrop: number;
}

export async function encodeConditioning(
  encodeClip: EncodeClip,
  video: Float32Array,
  T: number,
  H: number,
  W: number,
  { clipLength, tokenDrop }: Chunking,
): Promise<Moments> {
  // A single frame has no temporal extent to chunk, and `encode_base` sends it
  // through the spatial encoder alone. Padding it up to `clip_length` by
  // repetition would encode seventeen copies of one picture and return
  // `clip_length / 4 - token_drop` latent frames rather than one.
  if (T === 1) return encodeClip(video, T, H, W);

  const plane = H * W;
  const channels = video.length / (T * plane);
  const padded = T + ((-T % clipLength) + clipLength) % clipLength;

  let source = video;
  if (padded !== T) {
    // Channel-major, so the pad is per channel against the **new** stride.
    // Appending it once at the end of the buffer gives a result of exactly the
    // right length in which every channel's padding sits inside the next
    // channel's frames.
    source = new Float32Array(channels * padded * plane);
    for (let ch = 0; ch < channels; ch += 1) {
      source.set(video.subarray(ch * T * plane, (ch + 1) * T * plane), ch * padded * plane);
      // The **last** frame, repeated. Not the first, and not zeros.
      const last = video.subarray((ch * T + T - 1) * plane, (ch * T + T) * plane);
      for (let t = T; t < padded; t += 1) source.set(last, (ch * padded + t) * plane);
    }
  }

  const chunks: Moments[] = [];
  for (let at = 0; at < padded; at += clipLength) {
    const clip = new Float32Array(channels * clipLength * plane);
    for (let ch = 0; ch < channels; ch += 1) {
      clip.set(
        source.subarray((ch * padded + at) * plane, (ch * padded + at + clipLength) * plane),
        ch * clipLength * plane,
      );
    }
    chunks.push(await encodeClip(clip, clipLength, H, W));
  }

  const first = chunks[0]!;
  const total = chunks.reduce((n, chunk) => n + chunk.D, 0);
  // Off the end of the concatenation, not off each chunk.
  const kept = Math.max(0, total - tokenDrop);
  const voxels = first.H * first.W;
  const out = new Float32Array(first.C * kept * voxels);
  for (let ch = 0; ch < first.C; ch += 1) {
    let to = 0;
    for (const chunk of chunks) {
      for (let d = 0; d < chunk.D && to < kept; d += 1, to += 1) {
        out.set(
          chunk.data.subarray((ch * chunk.D + d) * voxels, (ch * chunk.D + d + 1) * voxels),
          (ch * kept + to) * voxels,
        );
      }
    }
  }
  return { data: out, C: first.C, D: kept, H: first.H, W: first.W };
}
