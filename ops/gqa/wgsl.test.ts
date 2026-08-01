import { describe, expect, it } from "vitest";
import { agree, expectAgrees, gpuTest, kernel, params, useGpu, type Runner } from "../../harness/index.js";
import { attention } from "../attention/reference.js";
import {
  defaultScale,
  groupedAttention,
  kvCacheBytes,
  type GroupedAttentionArgs,
  type GroupedAttentionResult,
} from "./reference.js";

// Two files rather than two entry points in one, for the reason `attention`
// measured and wrote down (#46): `layout: "auto"` drops the bindings an entry
// point does not statically name, the harness binds group 0 as a contiguous
// list, and the mismatch reads back zeros without raising.
const scoresCode = kernel(import.meta.url, "scores");
const contextCode = kernel(import.meta.url, "context");

/**
 * Every input, expected value and uniform is built at module scope; the test
 * bodies only dispatch and compare.
 *
 * Required, not stylistic — #49. After `useGpu`'s `beforeAll` has made the
 * device, a test body that spends ~10ms before its first dispatch kills the
 * vitest worker and the file reports nothing at all. The same work before the
 * device exists is fine at a full second. This op walks B x H x L x S x D twice
 * over (once for the reference, once for the expanded cross-check), so it is
 * squarely in that trap.
 */

/** Must match `WORKGROUP_SIZE` in both kernels. */
const WG = 256;

const wave = (n: number, k: number, phase = 0) =>
  Float32Array.from({ length: n }, (_, i) => Math.sin(i * k + phase) * 1.5);

/**
 * Measured, not widened until green.
 *
 * The arithmetic is `attention`'s — the same three f32 sums (the Q.K dots, the
 * softmax row sum, the probs@V sum) against an f64 reference — and sharing KV
 * heads changes which K rows are read, not how they are summed. Worst actually
 * observed across every case in this file, from an instrumented run with the
 * tolerance disabled:
 *
 *   probs   rel 1.32e-6   abs 2.38e-7
 *   output  rel 1.88e-3   abs 9.54e-7
 *
 * Identical across three runs, to every digit — this device's f32 reduction
 * order is fixed, so these are the numbers and not a sample of them.
 *
 * `abs` is the measure that carries this op, for the reason `attention` records:
 * `output` sums probabilities against a sine wave, so elements cancel to near
 * zero and a relative bound against a near-zero true value stops meaning
 * anything. That 1.88e-3 relative is 9.54e-7 absolute — one ulp at exponent
 * -20 — on an element whose true value is near zero. `agree` passes an element
 * on either measure, so it passes on absolute, and a relative-only bound of 1e-5
 * would fail a correct kernel.
 *
 * abs 2e-6 is 2.1x the worst absolute deviation observed. Deliberately not more:
 * everything this file exists to catch moves elements by order 0.1 to 10, five
 * orders of magnitude clear of the bound, so there is nothing to buy by widening
 * it. Confirmed by mutation, not assumed.
 */
const TOLERANCE = { rel: 1e-5, abs: 2e-6 };

type Prepared = {
  name: string;
  args: GroupedAttentionArgs;
  want: GroupedAttentionResult;
  scoresParams: ArrayBuffer;
  contextParams: ArrayBuffer;
};

function prepare(name: string, args: GroupedAttentionArgs): Prepared {
  return {
    name,
    args,
    want: groupedAttention(args),
    scoresParams: params([
      ["u32", args.H],
      ["u32", args.kvHeads],
      ["u32", args.L],
      ["u32", args.S],
      ["u32", args.D],
      ["f32", args.scale ?? defaultScale(args.D)],
      ["u32", args.causal ? 1 : 0],
      ["i32", args.queryOffset ?? 0],
    ]),
    contextParams: params([
      ["u32", args.H],
      ["u32", args.kvHeads],
      ["u32", args.L],
      ["u32", args.S],
      ["u32", args.Dv],
    ]),
  };
}

type Case = Omit<GroupedAttentionArgs, "q" | "k" | "v"> & { why: string };

// `kvHeads` is the axis this file exists for, so every grouping regime appears:
//
//   kvHeads == H   plain MHA. The op must not change behaviour here at all,
//                  which is also asserted directly against `attention` below.
//   kvHeads == 1   MQA.
//   1 < kv < H     GQA. The case both ends can pass by accident, so there are
//                  several: group sizes 2, 3 and 4, at two and three kv heads.
//
// The rest of the shape coverage is `attention`'s, kept because the two
// dispatches are the same shape of code: S and Dv below / at / not a multiple of
// the 256-wide workgroup, S > L and L > S, Dv != D, an explicit scale.
const CASES: Case[] = [
  { B: 1, H: 1, kvHeads: 1, L: 1, S: 1, D: 1, Dv: 1, why: "one of everything — MHA, MQA and GQA all at once" },
  { B: 1, H: 4, kvHeads: 4, L: 4, S: 4, D: 8, Dv: 8, why: "kvHeads == H, plain MHA" },
  { B: 1, H: 4, kvHeads: 1, L: 4, S: 4, D: 8, Dv: 8, why: "kvHeads == 1, MQA" },
  { B: 1, H: 4, kvHeads: 2, L: 4, S: 4, D: 8, Dv: 8, why: "GQA, two groups of two — the middle case" },
  { B: 2, H: 6, kvHeads: 3, L: 5, S: 5, D: 16, Dv: 16, why: "GQA, three groups of two, batched" },
  { B: 2, H: 6, kvHeads: 2, L: 5, S: 5, D: 16, Dv: 16, why: "GQA, two groups of three — group size is not 2" },
  { B: 2, H: 8, kvHeads: 2, L: 3, S: 4, D: 8, Dv: 8, why: "GQA, two groups of four" },
  { B: 3, H: 8, kvHeads: 4, L: 2, S: 3, D: 4, Dv: 4, why: "GQA, four groups of two, three batches" },
  { B: 2, H: 6, kvHeads: 3, L: 5, S: 5, D: 16, Dv: 16, causal: true, why: "GQA, causal" },
  { B: 1, H: 4, kvHeads: 1, L: 4, S: 4, D: 8, Dv: 8, causal: true, why: "MQA, causal" },
  { B: 1, H: 4, kvHeads: 2, L: 3, S: 7, D: 8, Dv: 8, why: "GQA with a KV cache longer than the query" },
  { B: 1, H: 4, kvHeads: 2, L: 3, S: 7, D: 8, Dv: 8, causal: true, why: "GQA, KV longer than query, causal upper-left (torch is_causal)" },
  { B: 1, H: 4, kvHeads: 2, L: 3, S: 7, D: 8, Dv: 8, causal: true, queryOffset: 4, why: "GQA, KV longer than query, causal bottom-right (torch causal_lower_right)" },
  { B: 2, H: 6, kvHeads: 3, L: 9, S: 4, D: 8, Dv: 8, causal: true, why: "GQA, query longer than the KV, causal" },
  { B: 2, H: 4, kvHeads: 2, L: 4, S: 4, D: 8, Dv: 5, why: "GQA with Dv smaller than D" },
  { B: 1, H: 4, kvHeads: 2, L: 3, S: 4, D: 5, Dv: 8, why: "GQA with Dv larger than D" },
  { B: 1, H: 4, kvHeads: 2, L: 3, S: WG, D: 8, Dv: 8, why: "S exactly one workgroup wide" },
  { B: 1, H: 4, kvHeads: 2, L: 3, S: WG + 44, D: 8, Dv: 8, why: "S past one workgroup, not a multiple" },
  { B: 1, H: 4, kvHeads: 2, L: 3, S: WG + 44, D: 8, Dv: 8, causal: true, why: "S past one workgroup, causal, so the mask lands mid-stride" },
  { B: 1, H: 4, kvHeads: 2, L: 2, S: 5, D: 8, Dv: WG + 44, why: "Dv past one workgroup, not a multiple" },
  { B: 1, H: 4, kvHeads: 2, L: 2, S: 5, D: 8, Dv: WG, why: "Dv exactly one workgroup wide" },
  { B: 1, H: 8, kvHeads: 2, L: 4, S: 6, D: 64, Dv: 64, why: "a Llama-shaped grouping and head dim" },
  { B: 1, H: 4, kvHeads: 2, L: 3, S: 6, D: 8, Dv: 8, scale: 0.25, why: "an explicit scale, like torch's scale=" },
];

const SHAPE_TESTS = CASES.map((c) => {
  const label = `B=${c.B} H=${c.H} kv=${c.kvHeads} L=${c.L} S=${c.S} D=${c.D} Dv=${c.Dv}${c.causal ? ` causal@${c.queryOffset ?? 0}` : ""}`;
  return prepare(`agrees with the reference at ${label} — ${c.why}`, {
    ...c,
    q: wave(c.B * c.H * c.L * c.D, 0.37),
    k: wave(c.B * c.kvHeads * c.S * c.D, 0.11, 0.5),
    v: wave(c.B * c.kvHeads * c.S * c.Dv, 0.23, 1.25),
  });
});

/**
 * The one thing this op adds, isolated so it is the only thing measured.
 *
 * The shape tests above would notice a badly wrong index, but they would not
 * reliably notice the *plausible* wrong index — strided groups (`h % kvHeads`)
 * instead of contiguous ones (`h / (H / kvHeads)`) — because sine-wave K and V
 * differ between heads by an amount softmax largely flattens. Here the two
 * mappings are made to give visibly different answers:
 *
 *   D = 1, every query is 1.0, and key row j of KV head g is large iff j == g.
 *     So KV head g produces a probability row that is a near one-hot on
 *     *column g*, and reading the wrong KV head puts the peak in the wrong
 *     column. That is checked on `probs`, before V is involved at all.
 *   V row j of KV head g is the constant g + 1 + ..., distinct per (kv head,
 *     row), so `output` names the KV head it read.
 *
 * `B` is 2 deliberately. The KV head is flattened as `b * kvHeads + g`, and
 * `b * H + g` — reusing the query head count for the KV stride — is the other
 * mistake available here. It is invisible at B = 1, and at B = 2 it reads off
 * the end or into the wrong batch.
 *
 * Both mappings are the identity at kvHeads == H and constant 0 at kvHeads == 1,
 * so only the middle entries can fail: 4 heads in 2 groups, 6 in 3, 6 in 2, 8 in
 * 4. The ends are here to show they are the ends, not to catch anything.
 */
const GROUPINGS = [
  { H: 4, kvHeads: 4 },
  { H: 4, kvHeads: 2 },
  { H: 4, kvHeads: 1 },
  { H: 6, kvHeads: 3 },
  { H: 6, kvHeads: 2 },
  { H: 8, kvHeads: 4 },
].map(({ H, kvHeads }) => {
  const [B, L, S, D, Dv] = [2, 3, 8, 1, 3];
  const q = new Float32Array(B * H * L * D).fill(1);
  const k = new Float32Array(B * kvHeads * S * D);
  const v = new Float32Array(B * kvHeads * S * Dv);
  for (let b = 0; b < B; b += 1) {
    for (let g = 0; g < kvHeads; g += 1) {
      const flat = b * kvHeads + g;
      k[flat * S * D + g * D] = 12; // only key row g answers this KV head, and hard
      for (let j = 0; j < S; j += 1) {
        for (let c = 0; c < Dv; c += 1) v[flat * S * Dv + j * Dv + c] = flat + 1 + c * 0.25 + j * 0.0625;
      }
    }
  }
  return prepare(`reads the KV head of its contiguous group at H=${H} kvHeads=${kvHeads}`, {
    q, k, v, B, H, kvHeads, L, S, D, Dv,
  });
});

/**
 * Reads that leave the buffer entirely, which nothing else here covers.
 *
 * Measured on this device: a storage read past the end of a buffer returns 0,
 * which makes an over-read *invisible* rather than loud — a bug that adds a
 * phantom term gets multiplied by zero and the suite stays green. So the
 * operands are allocated longer than they need to be and the slack filled with
 * 1e3, which dominates any real term instead.
 *
 * It matters more here than in `attention`. K and V are `kvHeads/H` of their MHA
 * size, so an index that forgot to divide walks off the end of a *shorter*
 * buffer, and at H=8 kvHeads=2 it runs three-quarters of the way past it. With
 * exact-length buffers that would be three-quarters zeros and a plausible-looking
 * partial answer.
 */
const PADDED = (() => {
  const [B, H, kvHeads, L, S, D, Dv] = [2, 8, 2, 3, 5, 7, 6];
  const PAD = 1e3;
  const qLen = B * H * L * D;
  const kLen = B * kvHeads * S * D;
  const vLen = B * kvHeads * S * Dv;
  const q = new Float32Array(qLen + 64).fill(PAD);
  const k = new Float32Array(kLen + 64).fill(PAD);
  const v = new Float32Array(vLen + 64).fill(PAD);
  q.set(wave(qLen, 0.41));
  k.set(wave(kLen, 0.23, 0.75));
  v.set(wave(vLen, 0.31, 1.1));
  return {
    ...prepare("reads no further than kvHeads into K and V", {
      q: q.subarray(0, qLen),
      k: k.subarray(0, kLen),
      v: v.subarray(0, vLen),
      B, H, kvHeads, L, S, D, Dv,
    }),
    padding: { q: q.length - qLen, k: k.length - kLen, v: v.length - vLen },
  };
})();

const ALL: Prepared[] = [...SHAPE_TESTS, ...GROUPINGS, PADDED];

/**
 * Runs the real two-dispatch pipeline and checks both halves.
 *
 * The attention matrix fed to the second dispatch is the one the first dispatch
 * produced, not the reference's, so the seam between them is exercised.
 */
async function check(run: Runner["run"], p: Prepared): Promise<void> {
  const { B, H, L, S, Dv } = p.args;

  const [raw] = await run({
    code: scoresCode,
    bindings: [
      { kind: "storage", data: p.args.q },
      { kind: "storage", data: p.args.k },
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

/**
 * Reference-level facts, computed here rather than in the test bodies for the
 * same #49 reason as everything else above: these call the reference several
 * times over multi-head shapes, which is milliseconds, which is fatal after the
 * device exists.
 */

/**
 * `kvHeads == H` must be `attention` — the same values, not merely close.
 *
 * This is the ISSUE's "MHA behaviour is unchanged", and it is asserted against
 * the op that was itself checked against torch rather than re-deriving the
 * answer here. Exact equality is the right bar: both references sum in f64 in
 * the same order, so any difference at all is a difference in indexing.
 */
const MHA_UNCHANGED = (() => {
  const [B, H, L, S, D, Dv] = [2, 4, 3, 6, 8, 5];
  const q = wave(B * H * L * D, 0.37);
  const k = wave(B * H * S * D, 0.11, 0.5);
  const v = wave(B * H * S * Dv, 0.23, 1.25);
  const shared = { q, k, v, B, L, S, D, Dv };
  return {
    plain: [
      attention({ ...shared, H }),
      attention({ ...shared, H, causal: true }),
      attention({ ...shared, H, causal: true, queryOffset: S - L }),
      attention({ ...shared, H, scale: 0.25 }),
    ],
    grouped: [
      groupedAttention({ ...shared, H, kvHeads: H }),
      groupedAttention({ ...shared, H, kvHeads: H, causal: true }),
      groupedAttention({ ...shared, H, kvHeads: H, causal: true, queryOffset: S - L }),
      groupedAttention({ ...shared, H, kvHeads: H, scale: 0.25 }),
    ],
  };
})();

/**
 * Sharing KV heads must equal not sharing them, given K and V expanded to match.
 *
 * The expansion is written out from a literal head map rather than from
 * `kvHeadOf`, so this does not check the reference against itself. `[0,0,1,1]`
 * and `[0,0,1,1,2,2]` are what contiguous grouping means, spelled out; the
 * strided alternative would be `[0,1,0,1]` and `[0,1,2,0,1,2]`.
 */
const EXPANSION = (() => {
  const [B, L, S, D, Dv] = [2, 3, 6, 8, 4];
  const maps: { H: number; kvHeads: number; map: number[] }[] = [
    { H: 4, kvHeads: 4, map: [0, 1, 2, 3] },
    { H: 4, kvHeads: 2, map: [0, 0, 1, 1] },
    { H: 4, kvHeads: 1, map: [0, 0, 0, 0] },
    { H: 6, kvHeads: 3, map: [0, 0, 1, 1, 2, 2] },
    { H: 6, kvHeads: 2, map: [0, 0, 0, 1, 1, 1] },
    { H: 8, kvHeads: 4, map: [0, 0, 1, 1, 2, 2, 3, 3] },
  ];
  return maps.map(({ H, kvHeads, map }) => {
    const q = wave(B * H * L * D, 0.37);
    const k = wave(B * kvHeads * S * D, 0.11, 0.5);
    const v = wave(B * kvHeads * S * Dv, 0.23, 1.25);
    const kx = new Float32Array(B * H * S * D);
    const vx = new Float32Array(B * H * S * Dv);
    for (let b = 0; b < B; b += 1) {
      for (let h = 0; h < H; h += 1) {
        const g = map[h]!;
        kx.set(k.subarray((b * kvHeads + g) * S * D, (b * kvHeads + g + 1) * S * D), (b * H + h) * S * D);
        vx.set(v.subarray((b * kvHeads + g) * S * Dv, (b * kvHeads + g + 1) * S * Dv), (b * H + h) * S * Dv);
      }
    }
    return {
      label: `H=${H} kvHeads=${kvHeads}`,
      grouped: groupedAttention({ q, k, v, B, H, kvHeads, L, S, D, Dv, causal: true }),
      expanded: attention({ q, k: kx, v: vx, B, H, L, S, D, Dv, causal: true }),
    };
  });
})();

describe("gqa / wgsl", () => {
  useGpu();

  for (const p of ALL) {
    gpuTest(p.name, async (run) => {
      await check(run, p);
    });
  }

  // The padded case only means anything if the padding is really there — a
  // Float32Array view reports its own length, so `check` cannot tell.
  gpuTest("the padded case really is padded", async () => {
    expect(PADDED.padding).toEqual({ q: 64, k: 64, v: 64 });
  });

  it("is exactly attention when kvHeads == H", () => {
    MHA_UNCHANGED.grouped.forEach((got, index) => {
      expect(Array.from(got.probs)).toEqual(Array.from(MHA_UNCHANGED.plain[index]!.probs));
      expect(Array.from(got.output)).toEqual(Array.from(MHA_UNCHANGED.plain[index]!.output));
    });
  });

  for (const { label, grouped, expanded } of EXPANSION) {
    it(`equals attention over K and V expanded by the head map at ${label}`, () => {
      expect(Array.from(grouped.output)).toEqual(Array.from(expanded.output));
    });
  }

  it("rejects a kvHeads that does not divide H", () => {
    const args = { q: new Float32Array(0), k: new Float32Array(0), v: new Float32Array(0), B: 1, H: 6, L: 1, S: 1, D: 1, Dv: 1 };
    for (const kvHeads of [4, 5, 7]) {
      expect(() => groupedAttention({ ...args, kvHeads })).toThrow(/must divide H=6/);
    }
    for (const kvHeads of [0, -1, 1.5]) {
      expect(() => groupedAttention({ ...args, kvHeads })).toThrow(/positive integer/);
    }
  });

  it("rejects K and V sized for the query heads instead of the KV heads", () => {
    const [B, H, kvHeads, L, S, D, Dv] = [1, 4, 2, 2, 3, 4, 4];
    const ok = {
      q: new Float32Array(B * H * L * D),
      k: new Float32Array(B * kvHeads * S * D),
      v: new Float32Array(B * kvHeads * S * Dv),
      B, H, kvHeads, L, S, D, Dv,
    };
    expect(() => groupedAttention(ok)).not.toThrow();
    expect(() => groupedAttention({ ...ok, k: new Float32Array(B * H * S * D) })).toThrow(/24 key elements, got 48/);
    expect(() => groupedAttention({ ...ok, v: new Float32Array(B * H * S * Dv) })).toThrow(/24 value elements, got 48/);
    expect(() => groupedAttention({ ...ok, q: new Float32Array(B * kvHeads * L * D) })).toThrow(/32 query elements, got 16/);
  });

  /**
   * The saving, in bytes, because that is the reason to reach for this op and
   * "uses less memory" is not a reason.
   *
   * One decoder layer at Llama-3-8B's shape — 32 query heads, head dim 128,
   * 8192 cached positions, batch 1, f32 — and the same layer as MQA and as MHA.
   */
  it("states the KV cache size in bytes", () => {
    const layer = { B: 1, S: 8192, D: 128, Dv: 128 };
    expect(kvCacheBytes({ ...layer, kvHeads: 32 })).toBe(268_435_456); // MHA,  256 MiB
    expect(kvCacheBytes({ ...layer, kvHeads: 8 })).toBe(67_108_864); //  GQA-8, 64 MiB
    expect(kvCacheBytes({ ...layer, kvHeads: 1 })).toBe(8_388_608); //   MQA,     8 MiB
    // Across all 32 layers: 8 GiB -> 2 GiB, a saving of 6_442_450_944 bytes.
    expect(32 * (kvCacheBytes({ ...layer, kvHeads: 32 }) - kvCacheBytes({ ...layer, kvHeads: 8 }))).toBe(6_442_450_944);
    // Linear in kvHeads, and in nothing else the caller controls.
    expect(kvCacheBytes({ ...layer, kvHeads: 8, bytesPerElement: 2 })).toBe(33_554_432);
  });
});
