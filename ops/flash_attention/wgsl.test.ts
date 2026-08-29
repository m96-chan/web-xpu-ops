import { describe, expect, it } from "vitest";
import {
  agree,
  gpuTest,
  kernel,
  params,
  useGpu,
  type Binding,
  type Dispatch,
  type Runner,
} from "../../harness/index.js";
import { attention, defaultScale, type AttentionArgs } from "../attention/reference.js";
import { DEFAULT_TILE, flashAttention } from "./reference.js";
import { FLASH_GENERATION, FLASH_TOKEN_ENTRY, flashGrid, flashSupports } from "./index.js";
import { shippedKernels } from "./tools/generate.js";

/**
 * Both shipped kernels, each held to the same reference.
 *
 * FA2 and FA3 are separate kernels with separate schedules — see
 * `tools/fa3.wgsl.ts` for what the second one does differently and why two of
 * FlashAttention-3's three ideas cannot be expressed in WGSL at all. What they
 * are not free to differ in is the answer, and the only way to know that is to
 * dispatch both. The first FA3 candidates written here were wrong by 1.6e-2 on
 * a ragged shape (an unsigned expression that needed a bracket), which nothing
 * but this comparison would have caught.
 */
const GENERATIONS = ["fa2", "fa3"] as const;
const CODE: Record<(typeof GENERATIONS)[number], string> = {
  fa2: kernel(import.meta.url, "fa2"),
  fa3: kernel(import.meta.url, "fa3"),
};
const code = CODE[FLASH_GENERATION];

/**
 * Every input, every expected value and every uniform is built here, at module
 * scope; the tests below only dispatch and compare.
 *
 * Not a style choice — see #49. Once `useGpu`'s `beforeAll` has created the
 * device, a test body that spends even ~10ms before its first dispatch kills the
 * vitest worker with `std::system_error` and reports no results at all. The same
 * work before the device exists is fine at a full second. This op walks
 * B x H x L x S x D twice per case (once for the flash recurrence, once for the
 * unfused reference it is checked against), which is milliseconds at the long-KV
 * shapes, so it lands squarely in that trap.
 */

/** Must match `TILE` in `wgsl/kernel.wgsl`; the sequence lengths below sit either side of it. */
const TILE = DEFAULT_TILE;
/** Must match `WORKGROUP_SIZE` in `wgsl/kernel.wgsl`. `Dv` past it needs a second pass over the keys. */
const WG = 256;

const wave = (n: number, k: number, phase = 0) =>
  Float32Array.from({ length: n }, (_, i) => Math.sin(i * k + phase) * 1.5);

/**
 * Measured, not widened until green. Both figures below are printed numbers from
 * an instrumented run with the comparison disabled, over every case in this file.
 *
 *   flash kernel vs flash reference    rel 3.711e-3   abs 3.725e-7
 *   flash kernel vs unfused reference  rel 3.743e-3   abs 4.023e-7
 *
 * The relative figure is large and the absolute one is not, which is the shape
 * of this op rather than a problem: `output` is a convex combination of a sine
 * wave, so elements cancel to near zero and relative error against a near-zero
 * true value stops carrying information. `agree` passes an element on either
 * measure, so those elements pass on absolute; a relative-only bound would fail
 * a correct kernel.
 *
 * `abs` is 4x the worst deviation seen (4.023e-7). That is far tighter than it
 * sounds: everything these tests exist to catch — a dropped rescale, a mask at
 * the wrong end, a tile boundary off by one, a crossed head stride — moves
 * elements by order 0.1 to 1, six orders of magnitude clear of it. Checked by
 * mutation rather than assumed.
 *
 * The kernel and the reference do not even sum in the same order, let alone the
 * same precision: the reference runs the recurrence in f64 and the kernel in
 * f32, and the kernel re-derives the running statistics per pass over Dv.
 */
const TOLERANCE = { rel: 1e-5, abs: 1.6e-6 };

type Prepared = {
  name: string;
  args: AttentionArgs;
  /** What this op's own reference says, computed with the online recurrence. */
  want: Float32Array;
  /** What the unfused op says. The ISSUE asks for this comparison by name. */
  unfused: Float32Array;
  dispatch: Dispatch;
};

/**
 * The uniform block, in the order `Params` declares it. `params()` packs 4 bytes
 * per field with no padding, so the struct is deliberately all scalars.
 */
function uniforms(args: AttentionArgs): ArrayBuffer {
  return params([
    ["u32", args.H],
    ["u32", args.L],
    ["u32", args.S],
    ["u32", args.D],
    ["u32", args.Dv],
    ["f32", args.scale ?? defaultScale(args.D)],
    ["u32", args.causal ? 1 : 0],
    ["i32", args.queryOffset ?? 0],
    ["u32", (args.maskShape ?? [args.B, 1, 1])[0]],
    ["u32", (args.maskShape ?? [args.B, 1, 1])[1]],
    ["u32", (args.maskShape ?? [args.B, 1, 1])[2]],
  ]);
}

/**
 * What gets bound at binding 3.
 *
 * Not optional, and that is the design: an additive bias of zeros *is* "no
 * mask", which is exactly what torch's own reference does (`attn_bias =
 * torch.zeros(L, S)`, added unconditionally). It costs `B * S` floats — `4n` at
 * the growth shapes below, against the `4n^2` this op exists to avoid — and it
 * buys back a per-element branch and a guard nothing could trip.
 */
const maskFor = (args: AttentionArgs): Float32Array => args.mask ?? new Float32Array(args.B * args.S);

/**
 * The one place a dispatch of this kernel is described.
 *
 * Both the correctness tests and the allocation test go through it, and that is
 * the point: the buffers counted are the buffers that produced the answer. A
 * kernel that quietly materialised the score matrix would have to name a buffer
 * here to write it into, and the count would see it.
 */
function dispatchFor(args: AttentionArgs, program = code): Dispatch {
  const { B, H, L, Dv } = args;
  return {
    code: program,
    bindings: [
      { kind: "storage", data: args.q },
      { kind: "storage", data: args.k },
      { kind: "storage", data: args.v },
      { kind: "storage", data: maskFor(args) },
      { kind: "out", type: "f32", length: B * H * L * Dv },
      { kind: "uniform", data: uniforms(args) },
    ],
    workgroups: flashGrid(L, H, B),
  };
}

function prepare(name: string, args: AttentionArgs): Prepared {
  return {
    name,
    args,
    want: flashAttention(args).output,
    unfused: attention(args).output,
    dispatch: dispatchFor(args),
  };
}

type Case = Omit<AttentionArgs, "q" | "k" | "v"> & { why: string };

// Every shape carries a reason. The ones this op adds over `attention`:
//
//   S vs TILE   — the key loop folds one tile at a time, so S below / on / just
//                 past / well past a tile boundary are four different paths, and
//                 anything past the first tile is the online recurrence running
//                 at all rather than a single pass wearing its clothes.
//   Dv vs WG    — Dv past the workgroup makes the kernel walk the keys a second
//                 time for the next block of output channels, re-deriving the
//                 running max and sum as it goes.
//   S > L       — a KV cache longer than the query, where the causal mask has
//                 two defensible meanings, so both are tested.
//   L > S       — not a mirror of it: here whole rows stay fully populated.
//   Dv != D     — the head dims must not be confused for one another.
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
  { B: 1, H: 1, L: 3, S: TILE - 1, D: 8, Dv: 8, why: "S one short of a tile" },
  { B: 1, H: 1, L: 3, S: TILE, D: 8, Dv: 8, why: "S exactly one tile" },
  { B: 1, H: 2, L: 3, S: TILE + 1, D: 8, Dv: 8, why: "S one past a tile, so the second tile holds a single key" },
  { B: 1, H: 2, L: 3, S: 2 * TILE, D: 8, Dv: 8, why: "S exactly two tiles" },
  { B: 1, H: 2, L: 3, S: 3 * TILE + 17, D: 8, Dv: 8, why: "several tiles with a ragged last one" },
  { B: 1, H: 1, L: 5, S: 3 * TILE + 17, D: 8, Dv: 8, causal: true, why: "several tiles, causal, so the mask ends mid-tile" },
  { B: 1, H: 1, L: 5, S: 3 * TILE + 17, D: 8, Dv: 8, causal: true, queryOffset: 3 * TILE + 12, why: "several tiles, causal bottom-right, mask ends in the last tile" },
  { B: 1, H: 1, L: 2, S: 5, D: 8, Dv: WG, why: "Dv exactly one workgroup wide" },
  { B: 1, H: 1, L: 2, S: TILE + 9, D: 8, Dv: WG + 44, why: "Dv past one workgroup, so the keys are walked twice, across tiles" },
  { B: 1, H: 1, L: 4, S: 6, D: 64, Dv: 64, why: "a head dim a transformer would actually use" },
  { B: 1, H: 1, L: 3, S: 6, D: 8, Dv: 8, scale: 0.25, why: "an explicit scale, like torch's scale=" },
];

const SHAPE_TESTS = CASES.map((c) => {
  const label = `B=${c.B} H=${c.H} L=${c.L} S=${c.S} D=${c.D} Dv=${c.Dv}${c.causal ? ` causal@${c.queryOffset ?? 0}` : ""}`;
  return prepare(`agrees at ${label} — ${c.why}`, {
    ...c,
    q: wave(c.B * c.H * c.L * c.D, 0.37),
    k: wave(c.B * c.H * c.S * c.D, 0.11, 0.5),
    v: wave(c.B * c.H * c.S * c.Dv, 0.23, 1.25),
  });
});

/**
 * The case the rescale exists for, and the only kind of case that can see it.
 *
 * `corr = exp(m - m_new)` is exactly 1 whenever the running maximum does not
 * move, so a kernel that never rescales — the single most likely way to get an
 * online softmax wrong — is correct on every input whose largest score is in the
 * first tile. Every shape case above has near-uniform scores, so every one of
 * them survives that bug.
 *
 * Here the scores climb with the key index, so the maximum moves in every tile
 * and the final one is three tiles past where it started. Dropping the
 * correction leaves the first tile's contribution overweighted by
 * exp(m_final - m_first), which is a factor of ~2.7 here, not a rounding.
 */
const RESCALE = (() => {
  const [L, S, D, Dv] = [2, 3 * TILE + 17, 4, 3];
  const q = new Float32Array(L * D);
  const k = new Float32Array(S * D);
  for (let i = 0; i < L; i += 1) q[i * D] = 1;
  for (let j = 0; j < S; j += 1) k[j * D] = j * 0.05; // scores rise monotonically
  return prepare("folds in a tile whose maximum beats every earlier tile", {
    q, k, v: wave(S * Dv, 0.37, 0.2), B: 1, H: 1, L, S, D, Dv,
  });
})();

/**
 * The shape tests would survive a mask flipped end for end: their inputs are
 * near-symmetric, and a softmax of a near-uniform row is near-uniform whichever
 * triangle is kept. This makes direction the only thing measured. Every score is
 * flat except at key 3, which is large, so a row that can see key 3 is a near
 * one-hot on it and a row that cannot is near-uniform.
 */
const CAUSAL_DIRECTION = (() => {
  const [L, S, D, Dv] = [5, 5, 4, 3];
  const q = new Float32Array(L * D);
  const k = new Float32Array(S * D);
  for (let i = 0; i < L; i += 1) q[i * D] = 6;
  k[3 * D] = 1;
  return prepare("masks the causal triangle at the same end the reference does", {
    q, k, v: wave(S * Dv, 0.7, 0.3), B: 1, H: 1, L, S, D, Dv, causal: true,
  });
})();

/**
 * exp(90) is already 1.2e39 and f32 tops out at 3.4e38, so a softmax that skips
 * the maximum returns Infinity/Infinity = NaN. Real logits do reach this —
 * `scale` divides by sqrt(D), but that does not stop |q||k|D from being large.
 * Ordinary inputs never come close, so without this case the subtraction can be
 * deleted and the suite stays green.
 *
 * The largest score is deliberately in the *last* tile, so the running maximum
 * has to carry the overflow guard across tiles rather than only within one.
 */
const OVERFLOW = (() => {
  const [L, S, D, Dv] = [2, TILE + 6, 4, 4];
  const q = new Float32Array(L * D).fill(30);
  const k = new Float32Array(S * D);
  for (let j = 0; j < S; j += 1) k[j * D] = j * 0.4; // scores 0, 6, 12, ... after scale
  return prepare("survives scores that would overflow exp", {
    q, k, v: wave(S * Dv, 0.9), B: 1, H: 1, L, S, D, Dv,
  });
})();

/**
 * Covers the reads that leave the buffer entirely, which nothing else here does.
 *
 * Measured on this device: a storage read past the end of a buffer returns 0,
 * which means an over-read is *invisible* wherever it falls off the end — a term
 * multiplied by zero changes nothing. Padding the operands with a value that is
 * large and not zero is what turns those reads into a visible wrong answer. The
 * shapes are chosen so `S` is not a multiple of the tile: the last tile is the
 * one whose bounds a kernel gets wrong.
 */
const PADDED = (() => {
  const [B, H, L, S, D, Dv] = [2, 2, 3, TILE + 5, 7, 6];
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
 * constant, so the correct output for each is that constant, and a stride that is
 * off by one head or one batch returns a neighbour's constant instead.
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
 * The shapes the allocation claim is made on: `L` and `S` grow together, because
 * that is the only sweep where `seq x seq` and `seq` are distinguishable. Held
 * at equal spacing so linearity can be asserted as a second difference of zero
 * rather than by fitting anything.
 *
 * They are ordinary correctness cases as well, and deliberately so — a byte
 * count only means something if the dispatch it counts produced the right
 * answer.
 */
const GROWTH_N = [64, 128, 192, 256];
const GROWTH = GROWTH_N.map((n) =>
  prepare(`agrees at L=S=${n} — the shapes the allocation claim is measured on`, {
    q: wave(n * 8, 0.37),
    k: wave(n * 8, 0.11, 0.5),
    v: wave(n * 8, 0.23, 1.25),
    B: 1, H: 1, L: n, S: n, D: 8, Dv: 8,
  }),
);

/**
 * The additive key mask (#77), and the thing it interacts with that it does not
 * in `attention`: the tile boundary.
 *
 * The online recurrence carries a running maximum across tiles, so *where* the
 * masked keys fall changes which path runs. The third case below masks the whole
 * of the first tile and nothing after it, which is the only input that reaches a
 * fold with no finite score in it yet — and the only one that can tell a missing
 * `-Infinity` guard apart from a working one, since every other row has
 * something finite in tile 0 to anchor the maximum.
 */
const MASK_TESTS: Prepared[] = (() => {
  const small = { B: 2, H: 3, L: 4, S: 5, D: 8, Dv: 6 };
  const operands = (a: { B: number; H: number; L: number; S: number; D: number; Dv: number }) => ({
    q: wave(a.B * a.H * a.L * a.D, 0.37),
    k: wave(a.B * a.H * a.S * a.D, 0.11, 0.5),
    v: wave(a.B * a.H * a.S * a.Dv, 0.23, 1.25),
    ...a,
  });
  const wide = { B: 1, H: 2, L: 3, S: 3 * TILE + 17, D: 8, Dv: 6 };
  // Every key of tile 0 masked, the rest alive.
  const firstTileDead = new Float32Array(wide.B * wide.S);
  firstTileDead.fill(-Infinity, 0, TILE);
  return [
    prepare("applies a [B, 1, 1, S] key padding mask", {
      ...operands(small),
      mask: Float32Array.from([0, 0, -Infinity, -Infinity, 0, 0, -Infinity, 0, 0, -Infinity]),
    }),
    prepare("applies a [B, H, L, S] bias element by element", {
      ...operands(small),
      mask: wave(small.B * small.H * small.L * small.S, 0.53, 0.2),
      maskShape: [small.B, small.H, small.L],
    }),
    prepare("carries the running maximum across a fully masked first tile", {
      ...operands(wide),
      mask: firstTileDead,
    }),
    prepare("broadcasts a [1, H, L, S] bias across the batch", {
      ...operands(small),
      mask: wave(small.H * small.L * small.S, 0.71, 0.9),
      maskShape: [1, small.H, small.L],
    }),
    prepare("indexes the mask past a tile boundary", {
      ...operands(wide),
      mask: wave(wide.B * wide.S, 0.09, 2.1),
    }),
  ];
})();

/**
 * A sample nobody may attend to — zeros, not `0 / 0`.
 *
 * This op reaches the row differently from `attention`: there the softmax
 * denominator is 0, here the running sum `l` never leaves 0 and the final divide
 * is 0/0. Batch 1 stays alive so the expectation is not zeros alone.
 */
const FULLY_MASKED = (() => {
  const shape = { B: 2, H: 2, L: 3, S: TILE + 9, D: 8, Dv: 5 };
  const mask = new Float32Array(shape.B * shape.S);
  mask.fill(-Infinity, 0, shape.S);
  return prepare("returns zeros for a sample whose every key is masked", {
    q: wave(shape.B * shape.H * shape.L * shape.D, 0.37),
    k: wave(shape.B * shape.H * shape.S * shape.D, 0.11, 0.5),
    v: wave(shape.B * shape.H * shape.S * shape.Dv, 0.23, 1.25),
    ...shape,
    mask,
  });
})();

const ALL: Prepared[] = [
  ...SHAPE_TESTS,
  ...GROWTH,
  RESCALE,
  CAUSAL_DIRECTION,
  OVERFLOW,
  PADDED,
  STRIDES,
  ...MASK_TESTS,
  FULLY_MASKED,
];

/**
 * What the harness will really allocate for a binding, mirroring
 * `createBuffer` in `harness/wgsl.ts` — including its 16-byte floor for
 * uniforms, so the number is the device's, not an idealised one.
 */
function bindingBytes(binding: Binding): number {
  if (binding.kind === "out") return binding.length * 4;
  // Scratch is storage the harness neither uploads nor reads back, but the
  // device still allocates it, so it counts toward what this op costs.
  if (binding.kind === "scratch") return Math.max(4, binding.length * 4);
  if (binding.kind === "uniform") return Math.max(16, binding.data.byteLength);
  return Math.max(4, binding.data.byteLength);
}

const boundBytes = (dispatch: Dispatch): number =>
  dispatch.bindings.reduce((total, binding) => total + bindingBytes(binding), 0);

/**
 * Zero everywhere iff the series is linear in an equally spaced argument. Used
 * instead of fitting a line because "the second difference is zero" is a claim
 * with no fitted constants in it to be wrong about.
 */
const secondDifference = (xs: number[]): number[] =>
  xs.slice(2).map((x, index) => x - 2 * xs[index + 1]! + xs[index]!);

/**
 * What the unfused op has to allocate at the same shape, for the comparison the
 * ISSUE asks for. Counted from its binding list rather than measured, because
 * running it here would couple this file to another op's kernels; the shape of
 * the list is in `ops/attention/wgsl.test.ts`. Two dispatches, so the peak is
 * the larger of them, and the `[B, H, L, S]` attention matrix is live across
 * both because the second reads what the first wrote.
 */
function unfusedPeakBytes(a: AttentionArgs): number {
  const probs = a.B * a.H * a.L * a.S * 4;
  const mask = a.B * a.S * 4;
  // 40 and 16 are the two uniform blocks: ten fields and four, floored at 16.
  const scores = a.B * a.H * a.L * a.D * 4 + a.B * a.H * a.S * a.D * 4 + mask + probs + 40;
  const context = probs + a.B * a.H * a.S * a.Dv * 4 + a.B * a.H * a.L * a.Dv * 4 + 16;
  return Math.max(scores, context);
}

/**
 * The report the ISSUE asks for: memory, not time. Bytes bound per dispatch at
 * the growth shapes (B=H=1, L=S=n, D=Dv=8), flash against unfused.
 *
 *   n     flash    unfused peak   the score matrix alone
 *    64    8_492        20_776                   16_384
 *   128   16_940        74_280                   65_536
 *   192   25_388       160_552                  147_456
 *   256   33_836       279_592                  262_144
 *
 * Flash is `132n + 44` exactly; unfused is `4n^2 + 68n + 40`, and the whole of
 * the difference is the score matrix. At n = 256 that is already 8x, and it
 * doubles with every doubling of the sequence. The assertions below hold both
 * shapes rather than these particular numbers.
 *
 * `4n` of the linear term on each side is the mask buffer added in #77 — a
 * `[B, 1, 1, S]` bias of zeros where the case has no mask. It is linear, so it
 * changes the constants and not the claim; the previous figures, before the
 * mask existed, were `128n + 32` and `4n^2 + 64n + 28`.
 *
 * Time is **unmeasured**: the roofline harness is #3 / #4.
 */
const REPORT = GROWTH.map((p) => ({
  n: p.args.L,
  flash: boundBytes(p.dispatch),
  unfused: unfusedPeakBytes(p.args),
  scoreMatrix: p.args.B * p.args.H * p.args.L * p.args.S * 4,
}));

/**
 * The kernel's own answer compared against the *unfused* reference as well as
 * its own, computed here so the test bodies stay short. The two references are
 * separately checked against each other below; this is what the ISSUE means by
 * measuring against unfused attention rather than only against the reference.
 */
async function check(run: Runner["run"], p: Prepared, dispatch = p.dispatch): Promise<void> {
  const [got] = await run(dispatch);
  const mine = agree(got as Float32Array, p.want, TOLERANCE);
  expect(mine, mine ? `vs flash reference: ${JSON.stringify(mine)}` : undefined).toBeNull();
  const theirs = agree(got as Float32Array, p.unfused, TOLERANCE);
  expect(theirs, theirs ? `vs unfused reference: ${JSON.stringify(theirs)}` : undefined).toBeNull();
}

/**
 * The online recurrence must not depend on where the tile boundaries fall. Swept
 * at module scope over tile sizes that divide S, that do not, that exceed it,
 * and that are 1 — a tile of 1 makes every step a rescale.
 *
 * This is a property of the algorithm, checked in TS, and it is what makes the
 * kernel's fixed compile-time tile a free choice rather than a load-bearing one.
 */
const TILE_SWEEP = (() => {
  const [B, H, L, S, D, Dv] = [1, 2, 4, 70, 8, 6];
  const args: AttentionArgs = {
    q: wave(B * H * L * D, 0.37),
    k: wave(B * H * S * D, 0.11, 0.5),
    v: wave(B * H * S * Dv, 0.23, 1.25),
    B, H, L, S, D, Dv, causal: true,
  };
  const unfused = attention(args).output;
  return [1, 2, 3, 7, 64, 70, 71, 1000].map((tile) => ({
    tile,
    worst: agree(flashAttention({ ...args, tile }).output, unfused, { rel: 1e-6, abs: 1e-7 }),
  }));
})();

/**
 * Every case's own reference against the unfused one, in f64. Separates "the
 * recurrence is the same function as a plain softmax" from "the kernel
 * implements the recurrence", so a failure says which.
 *
 * The bound is tight because both run in f64 and the only real difference is
 * that `attention` rounds its probabilities to f32 on the way between its two
 * dispatches, which this one has no occasion to do.
 */
const REFERENCE_AGREEMENT = ALL.map((p) => ({
  name: p.name,
  worst: agree(p.want, p.unfused, { rel: 1e-6, abs: 1e-7 }),
}));

describe("flash_attention / wgsl", () => {
  useGpu();

  for (const generation of GENERATIONS) {
  for (const p of ALL) {
    gpuTest(`${generation} ${p.name}`, async (run) => {
      // The shipped program is generated for a bounded `Dv`, because widening
      // it costs every caller registers — measured at 5.00 s a forward against
      // 5.92 s (`flashSupports`). A case past the bound is not a failure of
      // this kernel; it is a case `ops/attention` serves instead, and asserting
      // against it here would be asserting that the trade was not made.
      if (!flashSupports(p.args.D, p.args.Dv ?? p.args.D)) return;
      await check(run, p, dispatchFor(p.args, CODE[generation]));
    });
  }
  }

  // A generator edited without regenerating ships the old kernel while the
  // sweep times the new one — the same failure as benchmarking a program other
  // than the one that runs. `shippedKernels()` also builds `fa2_token`
  // (#223) — the equivalent check for that file is in the token-major
  // describe block below, next to the fixture that loads it.
  it("the shipped .wgsl files are what the generators produce", () => {
    const { fa2, fa3 } = shippedKernels();
    expect(CODE).toEqual({ fa2, fa3 });
  });

  // Not a GPU test: the recurrence is a claim about the algorithm.
  it("computes the same function as unfused attention, at every tile size", () => {
    expect(TILE_SWEEP.filter((s) => s.worst !== null)).toEqual([]);
  });

  it("its reference agrees with the unfused reference on every case here", () => {
    expect(REFERENCE_AGREEMENT.filter((r) => r.worst !== null)).toEqual([]);
  });

  /**
   * The other half of the claim, and the half a correctness test cannot make.
   *
   * "Same answer as unfused attention" on its own would pass a kernel that
   * materialised the score matrix and then read it back — which is the thing
   * this op exists not to do. So the buffers behind the dispatches that produced
   * those answers are counted, at shapes where `L` and `S` grow together, and
   * held to growing linearly. A `[B, H, L, S]` buffer would grow as n^2 and
   * break the second difference; nothing else about the kernel would change.
   */
  it("allocates linearly in the sequence length, not quadratically", () => {
    // The sweep has to be equally spaced for a second difference to mean
    // anything, so that is checked rather than trusted.
    const spacing = GROWTH_N[1]! - GROWTH_N[0]!;
    expect(GROWTH_N.map((n, index) => n - (GROWTH_N[0]! + index * spacing))).toEqual(
      GROWTH_N.map(() => 0),
    );

    const flash = REPORT.map((r) => r.flash);
    expect(secondDifference(flash)).toEqual(flash.slice(2).map(() => 0));
    // A real slope, so a kernel that bound nothing at all could not pass this.
    expect(flash[1]! - flash[0]!).toBeGreaterThan(0);

    // The control: the same counting applied to the unfused op, which does
    // allocate seq x seq, bends upward. Without this the assertion above would
    // only show that the arithmetic can produce zeros.
    const unfused = REPORT.map((r) => r.unfused);
    expect(secondDifference(unfused).every((d) => d > 0)).toBe(true);
  });

  it("never binds a buffer the size of the score matrix", () => {
    for (const p of GROWTH) {
      const largest = Math.max(...p.dispatch.bindings.map(bindingBytes));
      const scoreMatrix = p.args.B * p.args.H * p.args.L * p.args.S * 4;
      expect(largest, `L=S=${p.args.L}: largest binding ${largest} vs score matrix ${scoreMatrix}`)
        .toBeLessThan(scoreMatrix);
    }
  });

  // The comparison the report is built from, kept as an assertion so the numbers
  // in that comment cannot quietly stop being true.
  it("stays under what the unfused op has to allocate, by a margin that grows", () => {
    const ratios = REPORT.map((r) => r.unfused / r.flash);
    expect(REPORT.every((r) => r.flash < r.scoreMatrix)).toBe(true);
    // Strictly increasing, because the gap is quadratic over linear rather than
    // a constant factor — that is the property, not the size of it at any n.
    expect(ratios.slice(1).every((r, index) => r > ratios[index]!)).toBe(true);
    expect(ratios[ratios.length - 1]).toBeGreaterThan(8);
  });

  // The tile cases only mean anything if they really span tiles, and the rescale
  // case only means anything if the maximum really moves after the first tile.
  it("the multi-tile cases really do span more than one tile", () => {
    const spans = ALL.filter((p) => Math.ceil(p.args.S / TILE) > 1).map((p) => p.args.S);
    expect(spans.length).toBeGreaterThan(5);
    expect(Math.max(...spans)).toBeGreaterThan(3 * TILE);
  });

  it("the rescale case really does move its maximum past the first tile", () => {
    const { probs } = attention(RESCALE.args);
    const S = RESCALE.args.S;
    for (let i = 0; i < RESCALE.args.L; i += 1) {
      let argmax = 0;
      for (let j = 0; j < S; j += 1) if (probs[i * S + j]! > probs[i * S + argmax]!) argmax = j;
      expect(argmax).toBeGreaterThanOrEqual(3 * TILE);
    }
  });

  // A Float32Array view reports its own length, so `check` would be just as
  // happy with operands sized exactly to their contents.
  it("the padded case really is padded", () => {
    expect(PADDED.padding).toEqual({ q: 64, k: 64, v: 64 });
  });
});

/**
 * The token-major variant (#223): `q`/`k`/`v`/`output` addressed
 * `[B, L, H, D]` instead of `[B, H, L, D]`, so `examples/h3-dit` and
 * `examples/h3-video` can dispatch straight off the layout they already hold
 * q/k/v in, without `swapLeading` copying all four in and out.
 *
 * The staged tiles (`sq`/`sk`/`sv`) and everything that reads them — scores,
 * softmax, accumulate — are the same generated code for both layouts; only
 * the four global-memory index expressions differ (`tools/parts.ts`'s
 * `qAddress`/`kAddress`/`vAddress`/`outAddress`). That is what makes
 * bit-exactness against the head-major kernel the right acceptance test
 * (test b below) rather than a tolerance: same values land in the same
 * staged slots in the same order, so the arithmetic that follows cannot see
 * which layout fed it.
 */
describe("flash_attention / wgsl / token-major layout (#223)", () => {
  useGpu();

  const TOKEN_CODE = kernel(import.meta.url, "fa2_token");

  /**
   * Q/K content: a per-head phase shift, so two heads attend differently —
   * `wave` alone would make every head's scores a near-mirror of every
   * other's, and a softmax of near-mirror scores is itself near-identical,
   * which would let a head-stride bug hide within tolerance. **Not** a large
   * per-head constant like `headFolded` below: `Q . K` is a dot product, and
   * a large constant in both operands is squared into it (`offset^2 * D`),
   * swamping legitimate signal and turning ordinary f32-vs-f64 rounding into
   * a failure that has nothing to do with layout — measured while building
   * this test, at `+ head * 100` in both Q and K: rel 2.7e-5 against this
   * op's own f64 reference, nothing to do with the kernel under test.
   */
  const perHeadWave = (B: number, H: number, L: number, D: number, freq: number, phase: number): Float32Array => {
    const out = new Float32Array(B * H * L * D);
    for (let b = 0; b < B; b += 1) {
      for (let h = 0; h < H; h += 1) {
        const head = b * H + h;
        for (let i = 0; i < L; i += 1) {
          for (let d = 0; d < D; d += 1) {
            out[(head * L + i) * D + d] = Math.sin(i * freq + d * 0.31 + phase + head * 1.7) * 1.5;
          }
        }
      }
    }
    return out;
  };

  /**
   * V content: folds the (batch, head) index into the value itself
   * (`+ head * 100`). Unlike Q/K, V only ever appears in a *linear*
   * combination (the softmax-weighted sum), so a large per-head constant
   * here is safe — it does not get squared into anything — and it is what
   * turns a stride bug into garbage rather than a near-miss: a kernel that
   * reads one head's `v` rows out of another's slot lands on a value wrong
   * by 100, six orders of magnitude past what a dropped rescale or a masked
   * key could produce (see `TOLERANCE`'s own comment above).
   */
  const headFolded = (B: number, H: number, L: number, D: number, freq: number, phase: number): Float32Array => {
    const out = new Float32Array(B * H * L * D);
    for (let b = 0; b < B; b += 1) {
      for (let h = 0; h < H; h += 1) {
        const head = b * H + h;
        for (let i = 0; i < L; i += 1) {
          for (let d = 0; d < D; d += 1) {
            out[(head * L + i) * D + d] = Math.sin(i * freq + d * 0.31 + phase) * 1.5 + head * 100;
          }
        }
      }
    }
    return out;
  };

  /** `[B, H, L, D]` (head-major, what `flashAttention` and `dispatchFor` read) -> `[B, L, H, D]`. */
  const toTokenMajor = (x: Float32Array, B: number, H: number, L: number, D: number): Float32Array => {
    const out = new Float32Array(x.length);
    for (let b = 0; b < B; b += 1) {
      for (let h = 0; h < H; h += 1) {
        for (let i = 0; i < L; i += 1) {
          for (let d = 0; d < D; d += 1) {
            out[((b * L + i) * H + h) * D + d] = x[((b * H + h) * L + i) * D + d]!;
          }
        }
      }
    }
    return out;
  };

  /** The inverse of `toTokenMajor`: `[B, L, H, D]` -> `[B, H, L, D]`. */
  const toHeadMajor = (x: Float32Array, B: number, H: number, L: number, D: number): Float32Array => {
    const out = new Float32Array(x.length);
    for (let b = 0; b < B; b += 1) {
      for (let h = 0; h < H; h += 1) {
        for (let i = 0; i < L; i += 1) {
          for (let d = 0; d < D; d += 1) {
            out[((b * H + h) * L + i) * D + d] = x[((b * L + i) * H + h) * D + d]!;
          }
        }
      }
    }
    return out;
  };

  // H >= 2: at H = 1 head-major and token-major are the same bytes in the
  // same order (there is only one head to misplace), so every index bug —
  // including both mutations below — would pass by accident. L != S and
  // D != Dv so a stride confused with the wrong dimension is also visible.
  // S spans two tiles (DEFAULT_TILE = 64) so the token-major `j` — which
  // walks with `base`, the tile offset — is exercised past the first one.
  const SHAPE = { B: 2, H: 3, L: 5, S: 70, D: 8, Dv: 6 };
  const ARGS: AttentionArgs = {
    ...SHAPE,
    q: perHeadWave(SHAPE.B, SHAPE.H, SHAPE.L, SHAPE.D, 0.37, 0),
    k: perHeadWave(SHAPE.B, SHAPE.H, SHAPE.S, SHAPE.D, 0.11, 0.5),
    v: headFolded(SHAPE.B, SHAPE.H, SHAPE.S, SHAPE.Dv, 0.23, 1.25),
  };
  const WANT_HEAD_MAJOR = flashAttention(ARGS).output;
  const Q_TOKEN = toTokenMajor(ARGS.q, SHAPE.B, SHAPE.H, SHAPE.L, SHAPE.D);
  const K_TOKEN = toTokenMajor(ARGS.k, SHAPE.B, SHAPE.H, SHAPE.S, SHAPE.D);
  const V_TOKEN = toTokenMajor(ARGS.v, SHAPE.B, SHAPE.H, SHAPE.S, SHAPE.Dv);

  function tokenDispatch(): Dispatch {
    const { B, H, L, Dv } = ARGS;
    return {
      code: TOKEN_CODE,
      bindings: [
        { kind: "storage", data: Q_TOKEN },
        { kind: "storage", data: K_TOKEN },
        { kind: "storage", data: V_TOKEN },
        { kind: "storage", data: maskFor(ARGS) },
        { kind: "out", type: "f32", length: B * H * L * Dv },
        { kind: "uniform", data: uniforms(ARGS) },
      ],
      workgroups: flashGrid(L, H, B),
    };
  }

  // (a) Against this op's own reference, transposed at the boundary: the
  // reference only ever speaks head-major, so the kernel input is built
  // head-major and transposed to token-major, and the kernel's token-major
  // output is transposed back before the comparison.
  gpuTest("agrees with the reference on token-major inputs", async (run) => {
    const [got] = await run(tokenDispatch());
    const gotHeadMajor = toHeadMajor(got as Float32Array, SHAPE.B, SHAPE.H, SHAPE.L, SHAPE.Dv);
    const mismatch = agree(gotHeadMajor, WANT_HEAD_MAJOR, TOLERANCE);
    expect(mismatch, mismatch ? JSON.stringify(mismatch) : undefined).toBeNull();
  });

  // (b) Against the head-major kernel itself, on the same logical tensors,
  // to 0 ulp — not a tolerance. See the describe block's own comment for why
  // exact equality is the right bar here rather than a generous one.
  gpuTest("is bit-identical to the head-major kernel, transposed", async (run) => {
    const [headMajorGot] = await run(dispatchFor(ARGS));
    const [tokenGot] = await run(tokenDispatch());
    const tokenAsHeadMajor = toHeadMajor(tokenGot as Float32Array, SHAPE.B, SHAPE.H, SHAPE.L, SHAPE.Dv);
    expect(Array.from(tokenAsHeadMajor)).toEqual(Array.from(headMajorGot as Float32Array));
  });

  // The same staleness guard the head-major kernels get, extended to the
  // token-major file: a generator edited without regenerating would ship the
  // old kernel while this describe block tests the new one.
  it("the shipped fa2_token.wgsl is what the generator produces", () => {
    expect(TOKEN_CODE).toEqual(shippedKernels().fa2_token);
  });

  it("FLASH_TOKEN_ENTRY names the file this describe block loaded", () => {
    expect(FLASH_TOKEN_ENTRY).toBe("fa2_token");
  });
});
