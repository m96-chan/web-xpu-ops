/**
 * Greedy CTC decoding: `argmax` per frame, collapse repeats, then drop blanks.
 *
 * The definition of correct for this op. Every backend is measured against it,
 * and it is deliberately the slowest, plainest expression of the algorithm —
 * its job is to be obviously right, not fast.
 *
 * ## The order of the two steps is the whole op
 *
 * Repeats are collapsed **on the raw best path, blanks included**, and only then
 * are blanks removed. Doing it the other way round — strip blanks, then collapse
 * — is the plausible implementation, and it is wrong: it turns `a ␣ a` into a
 * single `a`, losing a genuine doubled letter.
 *
 * ```
 * path            collapse repeats   drop blanks   result
 * a a             a                  a             "a"     one letter
 * a ␣ a           a ␣ a              a a           "aa"    two letters
 * ```
 *
 * A blank between two identical labels is the *only* thing that lets CTC emit a
 * doubled character at all — it is why the blank symbol exists. So the two
 * orderings differ on exactly the input the blank was invented for, and they
 * agree everywhere else, which is what makes the wrong one survive testing.
 *
 * This is `torch.argmax` → `torch.unique_consecutive` → filter out the blank:
 * the `GreedyCTCDecoder` from torchaudio's ASR-inference tutorial, which is the
 * closest thing PyTorch ships to a canonical greedy decoder.
 *
 * ## Blank index
 *
 * Defaults to **0**, matching `torch.nn.CTCLoss(blank=0)`. This is a place where
 * conventions genuinely differ — TensorFlow's `ctc_greedy_decoder` puts the
 * blank *last*, at `C - 1` — so the default is stated rather than assumed, and
 * `blank` is an argument for anyone whose model was trained the other way.
 * A `blank` outside `[0, C)` is not an error and needs no special case: it
 * simply means no class is the blank, and every frame emits.
 *
 * ## Ties
 *
 * The **lowest** class index wins, matching `torch.argmax`, which documents that
 * the first maximal value's index is returned. Ties are not hypothetical here:
 * a model whose logits saturate, or an all-zero frame from a padded batch, ties
 * every class at once, and a tie broken differently changes the emitted string
 * rather than a low-order bit.
 *
 * ## Scores, not probabilities
 *
 * The input is whatever the acoustic model emits — logits, log-probabilities or
 * probabilities. `argmax` is invariant under any monotonic map, so applying a
 * softmax first would cost a pass over `[B, T, C]` and change nothing.
 *
 * ## Output shape
 *
 * A greedy decode emits at most one label per frame, so `[B, T]` is always large
 * enough and no allocation depends on the answer. Positions past a row's result
 * are filled with {@link CTC_PAD}, and the true lengths come back in their own
 * `[B]` buffer — both written by the kernel, so the caller never has to read the
 * tokens back to the CPU to find out how long they are.
 */

/** Fills `tokens` past the end of each row's result. Not a valid class index. */
export const CTC_PAD = -1;

export interface CTCDecodeArgs {
  /** Row-major `[B, T, C]` frame scores. */
  scores: Float32Array;
  /** Batch size. */
  B: number;
  /** Frames per batch item. */
  T: number;
  /** Classes per frame, blank included. */
  C: number;
  /** Class index of the CTC blank. Defaults to `0`, as `torch.nn.CTCLoss` does. */
  blank?: number;
}

export interface CTCDecoded {
  /** `[B, T]` labels, each row padded to `T` with {@link CTC_PAD}. */
  tokens: Int32Array;
  /** `[B]` label counts; row `b` holds `lengths[b]` labels. */
  lengths: Int32Array;
}

export function ctcDecode({ scores, B, T, C, blank = 0 }: CTCDecodeArgs): CTCDecoded {
  const tokens = new Int32Array(B * T).fill(CTC_PAD);
  const lengths = new Int32Array(B);
  for (let b = 0; b < B; b += 1) {
    let count = 0;
    // No frame precedes the first one, so nothing can collapse against it. -1 is
    // safe as "no previous label" for the same reason it is safe as padding: it
    // is not a class index.
    let previous = -1;
    for (let t = 0; t < T; t += 1) {
      const row = (b * T + t) * C;
      // argmax, first maximal index wins.
      let best = scores[row]!;
      let bestIndex = 0;
      for (let c = 1; c < C; c += 1) {
        if (scores[row + c]! > best) {
          best = scores[row + c]!;
          bestIndex = c;
        }
      }
      // Collapse against the previous frame's label — blank or not — and only
      // then drop the blank. Swapping these two lines is the bug this op exists
      // to not have.
      if (bestIndex !== previous && bestIndex !== blank) {
        tokens[b * T + count] = bestIndex;
        count += 1;
      }
      previous = bestIndex;
    }
    lengths[b] = count;
  }
  return { tokens, lengths };
}
