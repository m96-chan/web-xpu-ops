import { describe, expect } from "vitest";
import { agree, expectAgrees, gpuTest, kernel, params, useGpu, type Runner } from "../../harness/index.js";
import { attention, defaultScale, type AttentionArgs, type AttentionResult } from "./reference.js";

// Two files rather than two entry points in one, which is what the ISSUE
// suggests and what 0xBitNet does. Measured, not preferred: this harness builds
// its bind group from a contiguous 0..n-1 list, while `layout: "auto"` keeps
// only the bindings an entry point statically references. A probe with `scores`
// and `context` in one module — `scores` never naming binding 2 — read back all
// zeros with no error raised, which is issue #46 exactly. Splitting the files
// makes each pipeline's layout match what the harness binds.
const scoresCode = kernel(import.meta.url, "scores");
const contextCode = kernel(import.meta.url, "context");

/**
 * Every input, every expected value and every uniform is built here, at module
 * scope, and the tests below only dispatch and compare.
 *
 * Not a style choice — see #49. Once `useGpu`'s `beforeAll` has created the
 * device, a test body that spends even ~10ms before its first dispatch kills
 * the vitest worker with `std::system_error` and reports no results at all. The
 * same work at module scope, before the device exists, is fine at a full
 * second. This op walks B x H x L x S x D to build its expectations, which is
 * milliseconds at the multi-head and long-KV shapes, so it lands squarely in
 * that trap. `ops/stft/wgsl.test.ts` hit it first and has the measurements.
 */

/** Must match `WORKGROUP_SIZE` in both kernels; the shapes below are chosen around it. */
const WG = 256;

const wave = (n: number, k: number, phase = 0) =>
  Float32Array.from({ length: n }, (_, i) => Math.sin(i * k + phase) * 1.5);

/**
 * Looser on absolute than the harness default, and measured rather than widened
 * until green.
 *
 * The reference sums in f64; the kernels sum in f32, in three places it does
 * not — the Q.K dot products, the softmax row sum, and the probs@V sum. The
 * attention matrix itself is *not* a source of disagreement: this
 * implementation rounds it to f32 on its way between the two dispatches and the
 * reference models that exactly (its `probs` is a Float32Array).
 *
 * Worst actually observed over every case below, from an instrumented run with
 * the tolerance disabled — these are printed numbers, not estimates:
 *
 *   probs   rel 1.3e-6   abs 2.4e-7
 *   output  rel 9.7e-4   abs 4.8e-7
 *
 * The `output` relative figure is the interesting one, and it is why `abs` has
 * to be the measure that carries this op. `output` is a sum of probabilities
 * against a sine wave, so individual elements cancel to near zero, and relative
 * error against a near-zero true value stops meaning anything — 9.7e-4 relative
 * is 4.8e-7 absolute on a result that happens to land near 0. `agree` passes an
 * element on either measure, so those elements pass on absolute, and a
 * relative-only bound of 1e-5 would fail a correct kernel.
 *
 * abs 2e-6 is 4x the worst absolute deviation observed. That is far tighter
 * than it sounds: everything these tests exist to catch — a flipped mask, a
 * dropped scale, a transposed K, a crossed head stride — moves elements by
 * order 0.1 to 1, five orders of magnitude clear of it. Checked by mutation,
 * not assumed.
 */
const TOLERANCE = { rel: 1e-5, abs: 2e-6 };

type Prepared = {
  name: string;
  args: AttentionArgs;
  want: AttentionResult;
  scoresParams: ArrayBuffer;
  contextParams: ArrayBuffer;
  /**
   * What actually gets bound at binding 2.
   *
   * The mask binding is not optional, and that is the design rather than an
   * omission: an additive bias of zeros *is* "no mask", and it is what torch's
   * own reference does — `attn_bias = torch.zeros(L, S)`, added unconditionally.
   * A `has_mask` flag would buy one branch per element in exchange for a guard
   * whose only failing input is a dummy buffer nobody reads (rule 1).
   */
  maskData: Float32Array;
};

function prepare(name: string, args: AttentionArgs): Prepared {
  const [mb, mh, mr] = args.maskShape ?? [args.B, 1, 1];
  return {
    name,
    args,
    want: attention(args),
    // A zero bias where the case has no mask. `B * S` of them, the default
    // [B, 1, 1] shape — the smallest buffer that says "nothing is masked".
    maskData: args.mask ?? new Float32Array(args.B * args.S),
    scoresParams: params([
      ["u32", args.H],
      ["u32", args.L],
      ["u32", args.S],
      ["u32", args.D],
      ["f32", args.scale ?? defaultScale(args.D)],
      ["u32", args.causal ? 1 : 0],
      ["i32", args.queryOffset ?? 0],
      ["u32", mb],
      ["u32", mh],
      ["u32", mr],
    ]),
    contextParams: params([["u32", args.H], ["u32", args.L], ["u32", args.S], ["u32", args.Dv]]),
  };
}

type Case = Omit<AttentionArgs, "q" | "k" | "v"> & { why: string };

// Every shape carries a reason. The ones that matter most:
//
//   S vs 256   — the scores kernel softmaxes a row of S with a 256-wide
//                workgroup, so S below / at / not-a-multiple-of 256 are three
//                different paths through the strided loop and the tree reduce.
//   Dv vs 256  — the same for the context kernel, which strides over Dv.
//   S > L      — a KV cache longer than the query. This is the shape where the
//                causal mask has two defensible meanings, so both are tested.
//   L > S      — the other non-square direction, and not a mirror of it: here
//                the causal mask leaves whole rows fully populated instead.
//   Dv != D    — the two dispatches must not share a head-dim param.
const CASES: Case[] = [
  { B: 1, H: 1, L: 1, S: 1, D: 1, Dv: 1, why: "one of everything" },
  { B: 1, H: 1, L: 4, S: 4, D: 8, Dv: 8, why: "square sequence, one head" },
  { B: 1, H: 1, L: 4, S: 4, D: 8, Dv: 8, causal: true, why: "square sequence, causal" },
  { B: 2, H: 3, L: 5, S: 5, D: 16, Dv: 16, why: "multi-head and multi-batch" },
  { B: 2, H: 3, L: 5, S: 5, D: 16, Dv: 16, causal: true, why: "multi-head and multi-batch, causal" },
  { B: 1, H: 2, L: 3, S: 7, D: 8, Dv: 8, why: "KV cache longer than the query" },
  { B: 1, H: 2, L: 3, S: 7, D: 8, Dv: 8, causal: true, why: "KV longer than query, causal upper-left (torch is_causal)" },
  { B: 1, H: 2, L: 3, S: 7, D: 8, Dv: 8, causal: true, queryOffset: 4, why: "KV longer than query, causal bottom-right (torch causal_lower_right)" },
  { B: 2, H: 2, L: 9, S: 4, D: 8, Dv: 8, why: "query longer than the KV" },
  { B: 2, H: 2, L: 9, S: 4, D: 8, Dv: 8, causal: true, why: "query longer than the KV, causal" },
  { B: 2, H: 2, L: 4, S: 4, D: 8, Dv: 5, why: "value head dim smaller than the query head dim" },
  { B: 1, H: 1, L: 3, S: 4, D: 5, Dv: 8, why: "value head dim larger than the query head dim" },
  { B: 1, H: 2, L: 3, S: WG, D: 8, Dv: 8, why: "S exactly one workgroup wide" },
  { B: 1, H: 2, L: 3, S: WG + 44, D: 8, Dv: 8, why: "S past one workgroup, not a multiple" },
  { B: 1, H: 1, L: 3, S: WG + 44, D: 8, Dv: 8, causal: true, why: "S past one workgroup, causal, so the mask lands mid-stride" },
  { B: 1, H: 1, L: 2, S: 5, D: 8, Dv: WG + 44, why: "Dv past one workgroup, not a multiple" },
  { B: 1, H: 1, L: 2, S: 5, D: 8, Dv: WG, why: "Dv exactly one workgroup wide" },
  { B: 1, H: 1, L: 4, S: 6, D: 64, Dv: 64, why: "a head dim a transformer would actually use" },
  { B: 1, H: 1, L: 3, S: 6, D: 8, Dv: 8, scale: 0.25, why: "an explicit scale, like torch's scale=" },
];

const SHAPE_TESTS = CASES.map((c) => {
  const label = `B=${c.B} H=${c.H} L=${c.L} S=${c.S} D=${c.D} Dv=${c.Dv}${c.causal ? ` causal@${c.queryOffset ?? 0}` : ""}`;
  return prepare(`agrees with the reference at ${label} — ${c.why}`, {
    ...c,
    q: wave(c.B * c.H * c.L * c.D, 0.37),
    k: wave(c.B * c.H * c.S * c.D, 0.11, 0.5),
    v: wave(c.B * c.H * c.S * c.Dv, 0.23, 1.25),
  });
});

/**
 * The shape tests would survive a mask flipped end for end, because their
 * inputs are near-symmetric and a softmax of a near-uniform row is near-uniform
 * whichever triangle is kept. This makes direction the only thing measured:
 * every score is flat except at key 3, which is large, so a row that can see
 * key 3 is a near one-hot on it and a row that cannot is near-uniform. With
 * causal masking at offset 0, rows 0..2 cannot reach key 3 and rows 3..4 can —
 * two visibly different outputs, and the opposite grouping if the triangle is
 * the other one.
 */
const CAUSAL_DIRECTION = (() => {
  const [L, S, D, Dv] = [5, 5, 4, 3];
  const q = new Float32Array(L * D);
  const k = new Float32Array(S * D);
  for (let i = 0; i < L; i += 1) q[i * D] = 6; // every query points along axis 0
  k[3 * D] = 1; // only key 3 answers, and hard
  return prepare("masks the causal triangle at the same end the reference does", {
    q, k, v: wave(S * Dv, 0.7, 0.3), B: 1, H: 1, L, S, D, Dv, causal: true,
  });
})();

/**
 * exp(90) is already 1.2e39 and f32 tops out at 3.4e38, so a softmax that skips
 * subtracting the row maximum returns Infinity/Infinity = NaN. Real logits do
 * reach this — `scale` divides by sqrt(D), but that does not stop |q||k|D from
 * being large. Ordinary inputs never come close, so without this case the max
 * subtraction can be deleted and the suite stays green. Checked by mutation.
 */
const OVERFLOW = (() => {
  const [L, S, D, Dv] = [2, 6, 4, 4];
  const q = new Float32Array(L * D).fill(30);
  const k = new Float32Array(S * D);
  for (let j = 0; j < S; j += 1) k[j * D] = j * 5; // scores 0, 75, 150, ... after scale
  return prepare("survives scores that would overflow exp", {
    q, k, v: wave(S * Dv, 0.9), B: 1, H: 1, L, S, D, Dv,
  });
})();

/**
 * Covers the reads that leave the buffer entirely, which nothing else here does.
 *
 * Measured on this device: a storage read past the end of a buffer returns 0.
 * The consequence is not uniform across shapes, and it is worth being precise
 * about rather than repeating the usual claim, because the usual claim is not
 * what happens here. Running the Q.K dot product one element past D (mutation
 * M5) gives three different outcomes:
 *
 *   - on most shapes it lands on the *next row* of Q or K — in bounds, a live
 *     value — and the shape tests catch it on their own;
 *   - at the smallest shape (B=H=L=D=1) every over-read leaves the buffer,
 *     comes back 0, and the bug is invisible: that case passes under M5;
 *   - here it reads padding, which is 1e3 and dominates any real term.
 *
 * So this is not the only test that can see an overrun — it is the one that
 * sees it at the tail, where the device's convenient zero would otherwise hide
 * it, and without depending on a neighbouring row happening to exist. WGSL only
 * promises an out-of-bounds read stays within the resource, so the guard is
 * real even where this device is forgiving.
 */
const PADDED = (() => {
  const [B, H, L, S, D, Dv] = [2, 2, 3, 5, 7, 6];
  const PAD = 1e3;
  const q = new Float32Array(B * H * L * D + 64).fill(PAD);
  const k = new Float32Array(B * H * S * D + 64).fill(PAD);
  const v = new Float32Array(B * H * S * Dv + 64).fill(PAD);
  q.set(wave(B * H * L * D, 0.41));
  k.set(wave(B * H * S * D, 0.23, 0.75));
  v.set(wave(B * H * S * Dv, 0.31, 1.1));
  // The subarrays are views onto the padded buffers, so the dispatch is handed
  // the padding while the reference only ever sees the real elements.
  return {
    ...prepare("reads no further than D into Q and K, or S into V", {
      q: q.subarray(0, B * H * L * D),
      k: k.subarray(0, B * H * S * D),
      v: v.subarray(0, B * H * S * Dv),
      B, H, L, S, D, Dv,
    }),
    padding: { q: q.length - B * H * L * D, k: k.length - B * H * S * D, v: v.length - B * H * S * Dv },
  };
})();

/**
 * Head and batch strides are pure index arithmetic, and index arithmetic fails
 * by landing on a *neighbour* — a live, plausible value rather than noise. Here
 * every (batch, head) attends to a V of its own whose rows are all the same
 * constant, so the correct output for each is that constant, and a stride that
 * is off by one head or one batch returns a neighbour's constant instead.
 */
const STRIDES = (() => {
  const [B, H, L, S, D, Dv] = [3, 4, 2, 3, 4, 2];
  const v = new Float32Array(B * H * S * Dv);
  for (let head = 0; head < B * H; head += 1) {
    for (let s = 0; s < S; s += 1) {
      for (let c = 0; c < Dv; c += 1) v[head * S * Dv + s * Dv + c] = head + 1 + c * 0.25;
    }
  }
  return prepare("does not let one head, batch or query row bleed into another", {
    q: new Float32Array(B * H * L * D).fill(0.5),
    k: new Float32Array(B * H * S * D).fill(0.5),
    v, B, H, L, S, D, Dv,
  });
})();

/**
 * The additive key mask (#77).
 *
 * Every case here is built so the mask *moves the answer by order 1*. A bias
 * small enough to be a tolerance question would let a kernel that binds the
 * buffer and ignores it pass, which is exactly the mutation these have to catch.
 *
 * The broadcast cases are the ones with two plausible implementations. A mask
 * indexed at the wrong axis reads a live, in-bounds neighbour rather than
 * garbage, so each shape below gives every (batch, head, row) it broadcasts over
 * a *different* bias — otherwise collapsing the axis is invisible.
 */
const MASK_TESTS: Prepared[] = (() => {
  const [B, H, L, S, D, Dv] = [2, 3, 4, 5, 8, 6];
  const operands = {
    q: wave(B * H * L * D, 0.37),
    k: wave(B * H * S * D, 0.11, 0.5),
    v: wave(B * H * S * Dv, 0.23, 1.25),
    B, H, L, S, D, Dv,
  };
  const shape = (n: number, k: number, phase: number) => wave(n, k, phase);
  return [
    prepare("applies a [B, 1, 1, S] key padding mask", {
      ...operands,
      // Each sample keeps a different set of keys, so a mask read at batch 0 for
      // every sample changes the answer.
      mask: Float32Array.from([0, 0, -Infinity, -Infinity, 0, 0, -Infinity, 0, 0, -Infinity], (x) => x),
    }),
    prepare("applies a [B, H, L, S] bias element by element", {
      ...operands,
      mask: shape(B * H * L * S, 0.53, 0.2),
      maskShape: [B, H, L],
    }),
    prepare("broadcasts a [1, H, L, S] bias across the batch, the shape ops/alibi returns", {
      ...operands,
      mask: shape(H * L * S, 0.71, 0.9),
      maskShape: [1, H, L],
    }),
    prepare("broadcasts a [B, H, 1, S] bias across the query rows", {
      ...operands,
      mask: shape(B * H * S, 0.61, 0.4),
      maskShape: [B, H, 1],
    }),
    prepare("broadcasts a [B, 1, L, S] bias across the heads", {
      ...operands,
      mask: shape(B * L * S, 0.43, 1.7),
      maskShape: [B, 1, L],
    }),
    // S past one workgroup: the scores loop strides over j, so the mask index
    // has to stride with it. A mask read at `tid` instead of `j` agrees for the
    // first 256 keys and disagrees after.
    prepare("indexes the mask across a strided S", {
      q: wave(1 * 2 * 3 * 8, 0.37),
      k: wave(1 * 2 * (WG + 44) * 8, 0.11, 0.5),
      v: wave(1 * 2 * (WG + 44) * 6, 0.23, 1.25),
      B: 1, H: 2, L: 3, S: WG + 44, D: 8, Dv: 6,
      mask: wave(1 * (WG + 44), 0.09, 2.1),
    }),
  ];
})();

/**
 * A sample nobody may attend to.
 *
 * The row `ops/attention/reference.ts` used to declare unreachable. It is
 * reachable now, and it is reachable by accident — an empty sequence in a
 * padded batch is one. Torch returns zeros for it (`aten::_safe_softmax`), and
 * so must this, because the alternatives are a NaN row and a uniform row and
 * both look like results.
 *
 * Batch 0 is dead and batch 1 is alive, deliberately: an expectation of only
 * zeros cannot be told apart from a kernel that did nothing at all.
 */
const FULLY_MASKED = (() => {
  const [B, H, L, S, D, Dv] = [2, 2, 3, 4, 8, 5];
  const mask = new Float32Array(B * S);
  mask.fill(-Infinity, 0, S);
  return prepare("returns zeros for a sample whose every key is masked", {
    q: wave(B * H * L * D, 0.37),
    k: wave(B * H * S * D, 0.11, 0.5),
    v: wave(B * H * S * Dv, 0.23, 1.25),
    B, H, L, S, D, Dv, mask,
  });
})();

/**
 * One dead row inside a live sample, through a `[B, 1, L, S]` mask.
 *
 * Different from `FULLY_MASKED` in what it can catch: there the whole workgroup
 * grid for a batch is dead, so a guard applied per *dispatch* would pass. Here
 * rows 0 and 2 of every head are dead and row 1 is not, so the guard has to be
 * per query row — which is the granularity the softmax runs at.
 */
const FULLY_MASKED_ROW = (() => {
  const [B, H, L, S, D, Dv] = [1, 2, 3, 4, 8, 5];
  const mask = new Float32Array(B * L * S);
  mask.fill(-Infinity, 0 * S, 1 * S);
  mask.fill(-Infinity, 2 * S, 3 * S);
  return prepare("returns zeros for one dead query row beside live ones", {
    q: wave(B * H * L * D, 0.37),
    k: wave(B * H * S * D, 0.11, 0.5),
    v: wave(B * H * S * Dv, 0.23, 1.25),
    B, H, L, S, D, Dv, mask, maskShape: [B, 1, L],
  });
})();

/**
 * `-FLT_MAX` is not `-Infinity`, and the kernel must not confuse them.
 *
 * `-FLT_MAX` is what the causal path writes, so a fully-masked guard spelled
 * `row_max <= MASKED` would fire on a caller's mask of `-FLT_MAX` and return
 * zeros. Torch returns the mean of V there — measured on 2.10, a mask row of
 * -3.402823e38 gives `[-0.662, -0.211, 0.556]`, not zeros. The reference models
 * that, so this case fails if the comparison is widened by one character.
 */
const NEARLY_MASKED = (() => {
  const [B, H, L, S, D, Dv] = [1, 1, 2, 4, 8, 3];
  const mask = new Float32Array(B * S).fill(-3.402823e38);
  return prepare("treats a row of -FLT_MAX as finite, not as fully masked", {
    q: wave(B * H * L * D, 0.37),
    k: wave(B * H * S * D, 0.11, 0.5),
    v: wave(B * H * S * Dv, 0.23, 1.25),
    B, H, L, S, D, Dv, mask,
  });
})();

/**
 * The mask read that leaves the buffer.
 *
 * Same reasoning as `PADDED`: this device returns 0 for a storage read past the
 * end, and 0 is the *identity* for an additive bias, so an overrun off the tail
 * of the mask is invisible unless there is something non-zero sitting there.
 * The padding is 40, which is far past any logit these operands produce, so a
 * single element of overrun takes that key's probability to 1.
 */
const MASK_PADDED = (() => {
  const [B, H, L, S, D, Dv] = [2, 2, 3, 5, 4, 3];
  const PAD = 40;
  const backing = new Float32Array(B * S + 64).fill(PAD);
  const mask = wave(B * S, 0.31, 0.2);
  backing.set(mask);
  return {
    ...prepare("reads no further than S into the mask", {
      q: wave(B * H * L * D, 0.41),
      k: wave(B * H * S * D, 0.23, 0.75),
      v: wave(B * H * S * Dv, 0.31, 1.1),
      B, H, L, S, D, Dv,
      mask: backing.subarray(0, B * S),
    }),
    // Bound with the padding still attached; the reference only sees the view.
    maskData: backing,
    padding: backing.length - B * S,
  };
})();

const ALL: Prepared[] = [
  ...SHAPE_TESTS,
  CAUSAL_DIRECTION,
  OVERFLOW,
  PADDED,
  STRIDES,
  ...MASK_TESTS,
  FULLY_MASKED,
  FULLY_MASKED_ROW,
  NEARLY_MASKED,
  MASK_PADDED,
];

/**
 * Runs the real two-dispatch pipeline and checks both halves.
 *
 * The attention matrix the second dispatch consumes is the one the first
 * dispatch actually produced, not the reference's — otherwise the seam between
 * the two kernels is never exercised and a layout disagreement across them
 * stays invisible.
 */
async function check(run: Runner["run"], p: Prepared): Promise<void> {
  const { B, H, L, S, Dv } = p.args;

  const [raw] = await run({
    code: scoresCode,
    bindings: [
      { kind: "storage", data: p.args.q },
      { kind: "storage", data: p.args.k },
      { kind: "storage", data: p.maskData },
      { kind: "out", type: "f32", length: B * H * L * S },
      { kind: "uniform", data: p.scoresParams },
    ],
    workgroups: [L, H, B],
  });
  const probs = raw as Float32Array;
  const worst = agree(probs, p.want.probs, TOLERANCE);
  expect(worst, worst ? `probs: ${JSON.stringify(worst)}` : undefined).toBeNull();

  await expectAgrees(
    run,
    {
      code: contextCode,
      bindings: [
        { kind: "storage", data: probs },
        { kind: "storage", data: p.args.v },
        { kind: "out", type: "f32", length: B * H * L * Dv },
        { kind: "uniform", data: p.contextParams },
      ],
      workgroups: [L, H, B],
    },
    [p.want.output],
    TOLERANCE,
  );
}

describe("attention / wgsl", () => {
  useGpu();

  for (const p of ALL) {
    gpuTest(p.name, async (run) => {
      await check(run, p);
    });
  }

  // The padded case only means anything if the padding is really there. A
  // Float32Array view reports its own length, so `check` above would be just as
  // happy with operands sized exactly to their contents.
  gpuTest("the padded case really is padded", async () => {
    expect(PADDED.padding).toEqual({ q: 64, k: 64, v: 64 });
    expect(MASK_PADDED.padding).toBe(64);
    expect(MASK_PADDED.maskData.length).toBeGreaterThan(MASK_PADDED.args.mask!.length);
  });
});
