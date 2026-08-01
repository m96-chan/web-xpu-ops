import { describe, expect, it } from "vitest";
import { CASE_C, CASES, scoresFor } from "./cases.js";
import { CTC_PAD, ctcDecode } from "./reference.js";

/**
 * The reference held to written-down answers rather than to its own output.
 *
 * The kernel is measured against the reference, so nothing in the GPU suite can
 * catch the two being wrong together — and the way to be wrong here is a single
 * transposition, dropping blanks before collapsing repeats instead of after,
 * which agrees with the correct order on every input except the one CTC's blank
 * exists for. Touches no GPU, so it costs the suite nothing.
 *
 * `torch.argmax` → `torch.unique_consecutive` → filter the blank is the
 * `GreedyCTCDecoder` from torchaudio's ASR-inference tutorial, and the blank
 * defaults to 0 as `torch.nn.CTCLoss` has it.
 */
describe("ctc_decode / reference", () => {
  for (const [name, path, labels] of CASES) {
    it(name, () => {
      const T = path.length;
      const scores = scoresFor([path], CASE_C);
      const { tokens, lengths } = ctcDecode({ scores, B: 1, T, C: CASE_C });
      expect(Array.from(lengths)).toEqual([labels.length]);
      expect(Array.from(tokens.slice(0, labels.length))).toEqual(labels);
      // Everything past the result is padding, so no caller can mistake a stale
      // slot for a label.
      expect(Array.from(tokens.slice(labels.length))).toEqual(
        Array(T - labels.length).fill(CTC_PAD),
      );
    });
  }

  it("takes the lowest index when the top score is tied", () => {
    // torch.argmax returns the first maximal index. A tie is what a saturated
    // frame, or a zero-padded one, looks like — and it changes the emitted
    // label rather than a low-order bit, so the rule has to be pinned.
    const C = 8;
    const scores = new Float32Array(C).fill(-1);
    scores[2] = 4;
    scores[5] = 4;
    expect(Array.from(ctcDecode({ scores, B: 1, T: 1, C }).tokens)).toEqual([2]);
  });

  it("emits every frame when blank is outside the class range", () => {
    // Not an error and not a special case: no class is the blank, so nothing is
    // dropped and only the repeat collapse remains.
    const path = [1, 1, 2, 0, 0, 3];
    const { tokens, lengths } = ctcDecode({
      scores: scoresFor([path], 4),
      B: 1,
      T: path.length,
      C: 4,
      blank: 99,
    });
    expect(Array.from(lengths)).toEqual([4]);
    expect(Array.from(tokens.slice(0, 4))).toEqual([1, 2, 0, 3]);
  });

  it("never emits more labels than there are frames", () => {
    // Why [B, T] is always big enough, and therefore why the output shape does
    // not depend on the answer: at most one label leaves each frame.
    const C = 5;
    const path = [1, 2, 3, 4, 1, 2, 3, 4];
    const { lengths } = ctcDecode({ scores: scoresFor([path], C), B: 1, T: path.length, C });
    expect(lengths[0]).toBe(path.length);
  });
});
