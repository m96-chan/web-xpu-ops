/**
 * A reference video does not hand the same frames to the conditioner and to the VAE.
 *
 * Issue #216, found by reading `encoders.py` after the geometry question turned
 * out to be answerable. `MiniMaxH3Ref2VAReferenceEncoderStep` and
 * `MiniMaxH3Ref2VAConditionerStep` both take `reference.frames` — the clip at
 * its **own** rate, 24 fps for the model's own request — and use it differently:
 *
 * - the **conditioner** samples it at `video_sample_fps` (2.0), every
 *   `fps / sample_fps`-th frame, which is what `sampleVideoConditionFrames`
 *   already computes here;
 * - the **VAE** gets the clip at its own rate, truncated to `17n + 5` so the
 *   chunked encode lands on whole chunks.
 *
 * This port fed the tower's 2 fps frames to both, which is a two-second
 * reference described to the VAE as twenty-four seconds of slow motion — the
 * anchor's whole temporal content, wrong by the sampling ratio.
 */

/**
 * How many of a reference's frames the VAE encodes: `17n + 5`, rounded **down**.
 *
 * `encoders.py`:
 *
 *     num_frames = max(1, (num_frames - latents_per_chunk) // frames_per_chunk)
 *                  * frames_per_chunk + latents_per_chunk
 *
 * Down rather than to the nearest, because the frames past the end do not
 * exist. `framesPerChunk` and `latentsPerChunk` are the checkpoint's
 * `vae_clip_length` and its latent frames per chunk, passed in rather than
 * written here for the reason the chunking itself is: they belong to the
 * checkpoint, and a default is how the next one is snapped to this one's
 * geometry and returns a well-shaped tensor while doing it.
 */
export function vaeConditionFrames(
  numFrames: number,
  framesPerChunk: number,
  latentsPerChunk: number,
): number {
  // `//` in Python is floor division and floors *negative* values too, so a
  // clip shorter than one chunk gives `max(1, -1) = 1` and the formula asks for
  // more frames than exist. Upstream then writes `reference.frames[:num_frames]`
  // and the slice clamps. The clamp is here instead, so the number this returns
  // is the count that will actually be encoded rather than a formula's output
  // that has to be trimmed by whoever holds the array.
  const chunks = Math.max(1, Math.floor((numFrames - latentsPerChunk) / framesPerChunk));
  return Math.min(numFrames, chunks * framesPerChunk + latentsPerChunk);
}
