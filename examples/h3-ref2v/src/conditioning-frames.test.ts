/**
 * Which frames of a reference video go to the tower, and which go to the VAE.
 *
 * Issue #216. They are **not the same frames**, and this port sent the same
 * ones to both — `encoders.py` samples the clip at `video_sample_fps` for the
 * conditioner and hands `encode_vae_condition` the clip at its own rate,
 * truncated. Sending the tower's 2 fps frames to the VAE describes a
 * two-second reference as twenty-four seconds of slow motion.
 */
import { describe, expect, it } from "vitest";
import { vaeConditionFrames } from "./conditioning-frames.js";

describe("h3 ref2v / the frames a reference hands each model", () => {
  describe("the VAE's clip", () => {
    // `encoders.py`: `max(1, (N - latents_per_chunk) // frames_per_chunk) *
    // frames_per_chunk + latents_per_chunk`, snapping **down** to `17n + 5` so
    // the chunked encode lands on whole chunks. The comment upstream says it
    // only bites when the reference is shorter than the target, whose own frame
    // count already has that form.
    it("snaps down to 17n + 5", () => {
      expect(vaeConditionFrames(48, 17, 5)).toBe(39);
      expect(vaeConditionFrames(39, 17, 5)).toBe(39);
      expect(vaeConditionFrames(56, 17, 5)).toBe(56);
      expect(vaeConditionFrames(120, 17, 5)).toBe(107);
    });

    it("never asks for frames the caller does not have", () => {
      // The first version of this test asserted the formula alone and was
      // wrong about upstream, not about the code: `//` in Python floors
      // negatives too, so a clip shorter than a chunk gives `max(1, -1) = 1`
      // and the formula returns 22 for a one-frame reference. Upstream then
      // writes `reference.frames[:22]` and the slice clamps. The clamp lives in
      // the function here so the number is the count that gets encoded.
      for (let n = 1; n <= 200; n += 1) {
        expect(vaeConditionFrames(n, 17, 5), `${n} frames`).toBeLessThanOrEqual(n);
      }
    });

    it("encodes the whole of a clip shorter than one chunk", () => {
      // `max(1, ...)` asks for a full chunk and there is less than one, so what
      // is encoded is what exists.
      expect(vaeConditionFrames(6, 17, 5)).toBe(6);
      expect(vaeConditionFrames(1, 17, 5)).toBe(1);
      expect(vaeConditionFrames(22, 17, 5)).toBe(22);
    });

    it("takes the geometry rather than assuming this checkpoint's", () => {
      // 17 and 5 belong to the released `video_vae`. A second checkpoint with a
      // different `clip_length` must not be snapped to this one's.
      expect(vaeConditionFrames(48, 8, 2)).toBe(42);
    });
  });
});
