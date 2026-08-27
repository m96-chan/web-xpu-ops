/**
 * The chunking a conditioning encode goes through, held to the released model's.
 *
 * Issue #216. `examples/h3-encoder` was held to `EncoderFCN3D` + `quant_conv` at
 * 9.537e-6 and was still wrong for every reference with more than one frame,
 * because that pair is `AutoencoderKLLegacy.encode` and `encode` is not what
 * `encode_base` calls for a frame stack. It calls `encode_temporal`.
 *
 * **Measured** with the released weights (`tools/measure_temporal_chunking.py`),
 * the two paths against each other on the same clip:
 *
 *     frames        encode      temporal   rms difference
 *          8      48x2x2x2      48x2x2x2             0.0%
 *         17      48x5x2x2      48x2x2x2   different shape
 *         22      48x6x2x2      48x7x2x2   different shape
 *         48     48x12x2x2     48x12x2x2            17.9%
 *         68     48x17x2x2     48x17x2x2            19.0%
 *         85     48x22x2x2     48x22x2x2            21.5%
 *
 * 48 pixel frames is a two-second reference, the shortest the model card allows.
 * The shapes coinciding there is arithmetic, not agreement, and it is why
 * nothing downstream ever complained.
 *
 * Every case here uses a fake per-clip encode, so the chunking is checked
 * without a device, without weights and without a golden — and the fake can be
 * asked *which frames it was given*, which a real encoder cannot.
 */
import { describe, expect, it } from "vitest";
import { encodeConditioning } from "./conditioning.js";

const CHUNKING = { clipLength: 17, tokenDrop: 3 };

/**
 * A stand-in encoder that records its calls and returns `T / 4` latent frames
 * whose single value is the first pixel of the clip it was given.
 *
 * The value is what makes the concatenation checkable: a chunk that landed in
 * the wrong place, or a dropped frame taken off the wrong end, changes which
 * numbers come out in which order.
 */
function spy(latentPerClip = 5) {
  const calls: { T: number; frames: number[] }[] = [];
  const encode = (clip: Float32Array, T: number, H: number, W: number) => {
    // One value per frame, read back out of the clip so the test can say which
    // frames this chunk actually received.
    const plane = H * W;
    calls.push({ T, frames: Array.from({ length: T }, (_, t) => clip[t * plane]!) });
    const data = new Float32Array(latentPerClip);
    for (let d = 0; d < latentPerClip; d += 1) data[d] = calls.length * 100 + d;
    return Promise.resolve({ data, C: 1, D: latentPerClip, H: 1, W: 1 });
  };
  return { calls, encode };
}

/** `[1][T][1]`: one channel, one pixel a frame, the frame's own index as its value. */
const ramp = (T: number) => Float32Array.from({ length: T }, (_, t) => t);

describe("h3 video vae / conditioning chunking", () => {
  it("sends a single image through the spatial encoder alone", async () => {
    // `encode_base` chunks only a frame *stack*. Padding one image up to
    // `clip_length` by repetition would run the temporal path over seventeen
    // copies of the same picture and return `17 / 4 - 3` latent frames rather
    // than one — which is not the conditioning the model was trained with.
    const { calls, encode } = spy(1);
    const out = await encodeConditioning(encode, ramp(1), 1, 1, 1, CHUNKING);
    expect(calls).toHaveLength(1);
    expect(calls[0]!.T).toBe(1);
    expect(out.D).toBe(1);
  });

  it("cuts a frame stack into independent clip_length chunks", async () => {
    // The point is *independent*. The encoder is causal, so one long pass makes
    // latent frame k depend on every pixel frame before it; the model restarts
    // that state every 17 frames, and chunk two has never seen chunk one.
    const { calls, encode } = spy();
    await encodeConditioning(encode, ramp(34), 34, 1, 1, CHUNKING);
    expect(calls.map((c) => c.T)).toEqual([17, 17]);
    expect(calls[0]!.frames[0]).toBe(0);
    expect(calls[0]!.frames[16]).toBe(16);
    expect(calls[1]!.frames[0]).toBe(17);
    expect(calls[1]!.frames[16]).toBe(33);
  });

  it("pads up to a multiple of clip_length by repeating the last frame", async () => {
    // The last, not the first, and not zeros. 48 frames — a two-second
    // reference — is 51 padded, so the final chunk is frames 34..47 and then
    // frame 47 three more times.
    const { calls, encode } = spy();
    await encodeConditioning(encode, ramp(48), 48, 1, 1, CHUNKING);
    expect(calls.map((c) => c.T)).toEqual([17, 17, 17]);
    expect(calls[2]!.frames.slice(0, 14)).toEqual([34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47]);
    expect(calls[2]!.frames.slice(14)).toEqual([47, 47, 47]);
  });

  it("pads each channel against the new stride, not the old one", async () => {
    // Channel-major `[C][T][plane]`. Appending the padding at each channel's
    // *old* offset puts channel 1's frames inside channel 0's padding, and the
    // buffer is still exactly the right length — so nothing but this catches it.
    //
    // **15 frames into 10-frame chunks, deliberately.** The first version of
    // this test used 20 into 10, which is already a whole number of chunks: no
    // padding happened, and the mutation it was written for survived it. The
    // count has to be one the padding path actually runs on.
    const T = 15;
    const C = 3;
    const video = new Float32Array(C * T);
    for (let ch = 0; ch < C; ch += 1) for (let t = 0; t < T; t += 1) video[ch * T + t] = ch * 1000 + t;
    const clips: Float32Array[] = [];
    const encode = (clip: Float32Array) => {
      clips.push(clip.slice());
      return Promise.resolve({ data: new Float32Array(5), C: 1, D: 5, H: 1, W: 1 });
    };
    await encodeConditioning(encode, video, T, 1, 1, { clipLength: 10, tokenDrop: 0 });
    expect(clips).toHaveLength(2);
    // Every value still carries its own channel's thousand. A stride mistake
    // shows up as channel 1's numbers sitting in channel 0's frames.
    for (const clip of clips) {
      for (let ch = 0; ch < C; ch += 1) {
        for (let t = 0; t < 10; t += 1) {
          expect(Math.floor(clip[ch * 10 + t]! / 1000), `channel ${ch}, frame ${t}`).toBe(ch);
        }
      }
    }
    // And the tail of the second chunk is that channel's own last frame, 14.
    for (let ch = 0; ch < C; ch += 1) {
      expect(Array.from(clips[1]!.slice(ch * 10 + 5, ch * 10 + 10)))
        .toEqual([ch * 1000 + 14, ch * 1000 + 14, ch * 1000 + 14, ch * 1000 + 14, ch * 1000 + 14]);
    }
  });

  it("drops token_drop latent frames off the end of the whole concatenation", async () => {
    // Off the end of the *concatenation*, not off each chunk. Two chunks of
    // five is ten, minus three is seven — and the three that go are the last
    // chunk's last three, so the values that survive are 100..104 and 200..201.
    const { encode } = spy();
    const out = await encodeConditioning(encode, ramp(34), 34, 1, 1, CHUNKING);
    expect(out.D).toBe(7);
    expect(Array.from(out.data)).toEqual([100, 101, 102, 103, 104, 200, 201]);
  });

  it("interleaves the chunks per channel, not one after the other", async () => {
    // The output is channel-major too, so concatenating along D means every
    // chunk's slice of a channel lands at *that channel's* offset. Writing the
    // chunks end to end gives a buffer of exactly the right length in which
    // channel 1 holds channel 0's second chunk.
    let call = 0;
    const encode = () => {
      call += 1;
      // Two channels, two latent frames: `[c0d0, c0d1, c1d0, c1d1]`.
      const tag = call * 10;
      return Promise.resolve({
        data: Float32Array.from([tag + 0, tag + 1, tag + 2, tag + 3]),
        C: 2, D: 2, H: 1, W: 1,
      });
    };
    const out = await encodeConditioning(encode, ramp(20), 20, 1, 1, { clipLength: 10, tokenDrop: 0 });
    expect(out.C).toBe(2);
    expect(out.D).toBe(4);
    // channel 0: chunk 1's two, then chunk 2's two. Then channel 1's four.
    expect(Array.from(out.data)).toEqual([10, 11, 20, 21, 12, 13, 22, 23]);
  });

  it("passes clip_length through rather than assuming 17", async () => {
    // The numbers belong to the checkpoint. A version that wrote them in would
    // encode the next checkpoint's references with the wrong geometry and
    // return a well-shaped tensor while doing it.
    const { calls, encode } = spy();
    await encodeConditioning(encode, ramp(12), 12, 1, 1, { clipLength: 6, tokenDrop: 1 });
    expect(calls.map((c) => c.T)).toEqual([6, 6]);
  });
});
