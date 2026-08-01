import { params, type Binding } from "../../harness/index.js";

/**
 * Fixtures shared by this op's three test files.
 *
 * The named cases live here rather than in one of them so that the reference
 * suite and the kernel suites all check against the *same* written-down answer.
 * A kernel test that re-derived its expectation from the reference, and a
 * reference test that only checked the reference against itself, would between
 * them assert nothing about what CTC decoding means.
 *
 * ## Why the shapes are small, and why the GPU tests are split across two files
 *
 * Measured on this machine, and not a preference — issue #51: a process aborts —
 * `std::system_error` or a futex error, the whole process rather than the
 * dispatch — once a run has allocated enough storage. A single 640 KB buffer is
 * past it; so is a dozen dispatches at 128 KB each; so is sixteen at varying
 * sizes in the tens of kilobytes. Thirty dispatches at one constant 79 KB are
 * fine. It reproduces running the *rmsnorm* kernel with none of this op
 * involved, and this machine's `/tmp` tmpfs is full, so it is the environment
 * rather than the decode. The existing suites never met it because their widest
 * operand is a few kilobytes; `[B, T, C]` multiplies out faster.
 *
 * So every shape below stays inside about 25 KB and the suites are split so
 * neither runs too long. The op is not limited to these sizes — the tests are.
 *
 * ## Why every buffer is longer than the answer
 *
 * This device discards a write past the end of a buffer and reads zero past the
 * end of one, so a buffer sized exactly to the result makes both guards
 * untestable: the padding loop could overrun and the frame scan could overrun,
 * and nothing would show. Every scores buffer therefore carries spare frames of
 * real wave data, and both outputs are allocated at a fixed length past any
 * result here.
 */

/** Every named collapse case, as a best path and the labels it must decode to. */
export const CASES: [name: string, path: number[], labels: number[]][] = [
  ["a blank run at the start is dropped", [0, 0, 3, 4], [3, 4]],
  ["a blank run at the end is dropped", [3, 4, 0, 0], [3, 4]],
  ["blanks in the middle are dropped", [3, 0, 0, 0, 4], [3, 4]],
  ["a repeat with nothing between it collapses", [3, 3, 3, 4], [3, 4]],
  // The case that separates a correct collapse from a plausible one. Strip the
  // blanks first and this comes back as a single 3.
  ["a repeat separated by a blank survives", [3, 0, 3], [3, 3]],
  ["a repeat separated by a blank run survives", [3, 0, 0, 3], [3, 3]],
  ["a path of nothing but blanks decodes to nothing", [0, 0, 0, 0], []],
  ["a path with no blank at all keeps every frame", [3, 4, 5, 4], [3, 4, 5, 4]],
  ["all of it at once", [0, 0, 3, 3, 0, 3, 0, 0, 4, 4, 0, 5, 5, 5, 0], [3, 3, 4, 5]],
];

/** Classes used by the named cases; wide enough for every label in them. */
export const CASE_C = 6;

/** Labels are integers. There is no ulp to forgive. */
export const EXACT = { rel: 0, abs: 0 };

/** Spare f32 slots past `B*T*C` in every `scores` buffer. */
export const SCORE_SPARE = 512;
/** i32 slots in every `tokens` buffer. Larger than the largest `B*T` below. */
export const TOKEN_SLOTS = 128;
/** i32 slots in every `lengths` buffer. Larger than the largest `B` below. */
export const LENGTH_SLOTS = 32;

/**
 * The wave that fills a scores buffer, both under the intended path and past
 * the end of it.
 *
 * Never zeros: a kernel that ignores the scores, reads the wrong frame, or runs
 * one frame past `T` would find a tie across every class, pick class 0 — which
 * is the blank — and emit nothing, which is indistinguishable from having
 * stopped correctly. Against a wave it emits a visibly wrong label instead.
 */
const wave = (n: number, k: number) => Float32Array.from({ length: n }, (_, i) => Math.sin(i * k) * 2);

/** `head`, followed by `SCORE_SPARE` slots of wave for a scan to overrun into. */
export function padScores(head: Float32Array): Float32Array {
  const scores = wave(head.length + SCORE_SPARE, 0.19);
  scores.set(head);
  return scores;
}

/**
 * Scores whose per-frame `argmax` is the given path, with spare slots past it.
 *
 * The winner sits at 4, clear of the wave's `[-2, 2]`, so the intended path is
 * the argmax by a margin no rounding can close.
 */
export function scoresFor(paths: readonly (readonly number[])[], C: number, k = 0.37): Float32Array {
  const T = paths[0]!.length;
  const head = wave(paths.length * T * C, k);
  paths.forEach((path, b) => {
    path.forEach((label, t) => {
      head[(b * T + t) * C + label] = 4;
    });
  });
  return padScores(head);
}

/** The four bindings, in shader order, at the fixed lengths. */
export function bindings(
  scores: Float32Array,
  { B, T, C, blank = 0 }: { B: number; T: number; C: number; blank?: number },
): Binding[] {
  return [
    { kind: "storage", data: scores },
    { kind: "out", type: "i32", length: TOKEN_SLOTS },
    { kind: "out", type: "i32", length: LENGTH_SLOTS },
    { kind: "uniform", data: params([["u32", B], ["u32", T], ["u32", C], ["u32", blank]]) },
  ];
}

/** A reference result widened to the fixed output lengths; the spare stays zero. */
export function expected(tokens: Int32Array, lengths: Int32Array): [Int32Array, Int32Array] {
  const paddedTokens = new Int32Array(TOKEN_SLOTS);
  paddedTokens.set(tokens);
  const paddedLengths = new Int32Array(LENGTH_SLOTS);
  paddedLengths.set(lengths);
  return [paddedTokens, paddedLengths];
}

/**
 * A deterministic path dense in blanks, repeats, and repeats across a blank.
 *
 * MINSTD rather than the usual glibc constants: `state * 1103515245` runs past
 * 2^53, so the low bits round away and every subsequent `state % 4` comes out
 * 0 — measured, after a first version of this emitted one label and then blanks
 * for the whole rest of the path, which would have made every shape below
 * decode to nothing. MINSTD's product stays under 2^53.
 */
export function pseudoPath(T: number, C: number, seed: number): number[] {
  const path: number[] = [];
  let state = seed;
  for (let t = 0; t < T; t += 1) {
    state = (state * 48271) % 2147483647;
    const roll = state % 4;
    if (roll === 0) path.push(0);
    else if (roll === 1 && t >= 1) path.push(path[t - 1]!);
    // Two frames back, so whatever sits between — often a blank — separates a
    // pair of identical labels.
    else if (roll === 2 && t >= 2) path.push(path[t - 2]!);
    else path.push(state % C);
  }
  return path;
}
