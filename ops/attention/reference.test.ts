import { describe, expect, it } from "vitest";
import { agree } from "../../harness/index.js";
import { attention, keyPaddingBias, resolveMask } from "./reference.js";

/**
 * The mask half of `attention`, checked against torch rather than against
 * itself.
 *
 * No GPU here: these are the questions the kernel cannot answer — which torch
 * API the mask matches, what happens when `causal` and `mask` are both given,
 * and what a fully masked row produces. A kernel cannot throw, so the rejections
 * live in the reference and are tested here.
 *
 * Every `TORCH_*` array below is `torch.nn.functional.scaled_dot_product_
 * attention` on torch 2.10, in f64, on the inputs `wave()` builds — the same
 * closed form on both sides, so only the answers had to be carried across.
 * Regenerate with the script recorded in PR #77 if the shapes ever change.
 */

const wave = (n: number, k: number, phase = 0) =>
  Float32Array.from({ length: n }, (_, i) => Math.sin(i * k + phase) * 1.5);

const [B, H, L, S, D, Dv] = [2, 2, 3, 4, 5, 3];
const q = wave(B * H * L * D, 0.37);
const k = wave(B * H * S * D, 0.11, 0.5);
const v = wave(B * H * S * Dv, 0.23, 1.25);
const base = { q, k, v, B, H, L, S, D, Dv };

/**
 * The reference rounds `probs` to f32 between its two halves (it models the two
 * dispatches, which really do), while torch stays in f64 throughout. That is the
 * whole of the difference measured here, and it is small: over the four cases
 * below, from an instrumented run with the tolerance set to zero, the worst
 * element is
 *
 *   rel 2.0e-6  (on an output of 0.0078, where abs is 1.6e-8)
 *   abs 1.2e-7  (on an output of 1.166, where rel is 1.0e-7)
 *
 * — the two extremes are different elements, which is the usual reason `agree`
 * passes on either measure. The harness defaults, 1e-5 and 1e-6, are 5x and 8x
 * those, and everything these cases exist to catch (a mask added to the wrong
 * head, a sign, a broadcast collapsed) moves elements by order 1.
 */
const TOLERANCE = { rel: 1e-5, abs: 1e-6 };

const expectAgrees = (got: ArrayLike<number>, want: ArrayLike<number>) => {
  const worst = agree(got, want, TOLERANCE);
  expect(worst, worst ? JSON.stringify(worst) : undefined).toBeNull();
};

// attn_mask = [[1,1,0,0],[1,0,1,1]] as booleans, `True` meaning attend.
const KEEP = [1, 1, 0, 0, 1, 0, 1, 1];

const TORCH_KEY_PADDING = [
  1.4068794708522159, 1.321357806421455, 1.1662439129289601, 1.4064534382092315, 1.3169309199040113,
  1.1580493242988696, 1.417726236350568, 1.4340660712299498, 1.3748776471974045, -1.3219683223816414,
  -1.4000114929039147, -1.4043199641420534, -1.1787485966149964, -1.3479995410545147,
  -1.4462551105999017, -1.4205980325792695, -1.4358300486947404, -1.3754409021267295,
  0.7705729564385545, 1.0048540879002217, 1.1862123581058919, 0.8964800854348571, 0.6109516397720434,
  0.29324607580407835, 0.9907756603397785, 0.860902588634707, 0.6856881795137221, -1.1304299577106138,
  -1.0044180067566417, -0.8255061617238334, -0.7800169901457561, -0.9109191372186369,
  -0.9938457148083983, -0.9337754484863321, -1.0315304841550699, -1.0749576872045006,
];

const TORCH_FULL_BIAS = [
  0.8167093628794843, 0.5755275038090614, 0.3040342168409377, 1.1506288076367732, 1.021471719458891,
  0.8385165657341865, 0.30398830510110797, 0.07292609083984952, -0.1619769371613664,
  -0.9064583080185152, -0.7109766514012879, -0.47804983818725477, -1.1620679352566876,
  -1.3397735712687775, -1.4469170712648072, -0.593361680499322, -0.30069508212183055,
  0.007808287288505522, 1.0549172070090302, 1.199934500404341, 1.281754591493511, 0.9615864758702768,
  0.7208275632238843, 0.4421046739391395, 0.9900976505930126, 0.796531187407181, 0.5610136489111881,
  -1.1975961231435344, -1.0641319875300081, -0.8746229890729604, -0.5696658536241497,
  -0.8255822784376222, -1.038017588029218, -1.1223007536137475, -1.1846270569443813,
  -1.1845623583079778,
];

const TORCH_SHARED_BIAS = [
  1.1309613042007145, 0.9558453792353304, 0.7303877449645044, 0.6179535465481375,
  0.39105139513477694, 0.1435536577772641, 0.6047412013853951, 0.3989134758924121,
  0.17207609059706175, -1.059496388704633, -1.054999947481015, -0.9949396024847662,
  -1.1963350495368568, -1.2957148035681716, -1.3268528739537817, -1.033196517808294,
  -0.9115968757451046, -0.7419859696015033, 0.9795480948258745, 1.1716578090081078,
  1.3020595753277664, 0.8658508964848085, 0.5730509251369033, 0.2500699603802784, 1.1571099906301427,
  1.048860107683582, 0.8853696891963019, -0.8882623033439173, -0.8927625998301774,
  -0.8502435810006328, -1.1850102960387334, -1.2299468788106134, -1.2101055910405514,
  -0.6228340777912418, -0.8730657072820242, -1.0773154018329714,
];

const TORCH_FULLY_MASKED = [
  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.9381331879409216, 1.134380365200987,
  1.27088289355933, 0.9036934964387502, 0.6237799180508542, 0.31101359173189613, 1.064606967352948,
  0.9768016786173404, 0.8375509707561408, -1.1265913991108243, -1.0223713373180712,
  -0.8643058296157963, -0.8805981518143126, -1.0392586844896694, -1.1431843617958575,
  -0.9909155027159935, -1.1359054302352045, -1.2210703879322886,
];

describe("attention / mask", () => {
  it("adds a [B, 1, 1, S] key mask, as torch's float attn_mask", () => {
    const { output } = attention({ ...base, mask: keyPaddingBias(B, S, KEEP) });
    expectAgrees(output, TORCH_KEY_PADDING);
  });

  it("adds a [B, H, L, S] bias to every element, as torch broadcasts none", () => {
    const mask = wave(B * H * L * S, 0.53, 0.2);
    const { output } = attention({ ...base, mask, maskShape: [B, H, L] });
    expectAgrees(output, TORCH_FULL_BIAS);
  });

  // The shape `ops/alibi` produces. If this did not broadcast over the batch,
  // composing the two would mean tiling the bias B times for nothing.
  it("broadcasts a [1, H, L, S] bias across the batch", () => {
    const mask = wave(H * L * S, 0.71, 0.9);
    const { output } = attention({ ...base, mask, maskShape: [1, H, L] });
    expectAgrees(output, TORCH_SHARED_BIAS);
  });

  it("is additive and not subtractive", () => {
    const mask = wave(B * H * L * S, 0.53, 0.2);
    const negated = Float32Array.from(mask, (x) => -x);
    const { output } = attention({ ...base, mask: negated, maskShape: [B, H, L] });
    // The sign matters by order 1, not by a tolerance. Measured against torch:
    // flipping it moves the answer by 1.06 at this shape.
    expect(agree(output, TORCH_FULL_BIAS, TOLERANCE)).not.toBeNull();
  });

  it("keeps a bias of zeros equal to no mask at all", () => {
    const plain = attention(base);
    const zeroed = attention({ ...base, mask: new Float32Array(B * S) });
    expectAgrees(zeroed.output, plain.output);
    expectAgrees(zeroed.probs, plain.probs);
  });
});

describe("attention / a fully masked row", () => {
  const dead = new Float32Array(B * S);
  dead.fill(-Infinity, 0, S); // batch 0 sees no key at all
  const { probs, output } = attention({ ...base, mask: dead });

  it("gives zeros, as aten::_safe_softmax does and torch.softmax does not", () => {
    expectAgrees(output, TORCH_FULLY_MASKED);
  });

  it("gives a zero probability row rather than a uniform one", () => {
    // A uniform row is what a kernel that only guards the division returns, and
    // it is a plausible-looking wrong answer: it still sums to 1.
    expect(Array.from(probs.subarray(0, S))).toEqual([0, 0, 0, 0]);
  });

  it("puts no NaN anywhere", () => {
    expect(Array.from(probs).some(Number.isNaN)).toBe(false);
    expect(Array.from(output).some(Number.isNaN)).toBe(false);
  });

  it("leaves the other samples in the batch alone", () => {
    const alive = attention(base);
    const half = (H * L * Dv);
    expectAgrees(output.subarray(half), alive.output.subarray(half));
  });

  // -FLT_MAX is what the kernels write for a causally masked entry, and torch
  // treats it as an ordinary finite score: measured, a row of -FLT_MAX returns
  // the mean of V, not zeros. So the guard has to trigger on -Infinity alone.
  it("does not treat a row of -FLT_MAX as fully masked", () => {
    const nearly = new Float32Array(B * S);
    nearly.fill(-3.402823e38, 0, S);
    const { probs: p } = attention({ ...base, mask: nearly });
    expect(Array.from(p.subarray(0, S))).toEqual([0.25, 0.25, 0.25, 0.25]);
  });
});

describe("attention / mask validation", () => {
  it("rejects causal and mask together, as torch does", () => {
    expect(() => attention({ ...base, causal: true, mask: new Float32Array(B * S) })).toThrow(
      /causal and mask are exclusive/,
    );
  });

  it("rejects a maskShape entry that is neither 1 nor the full dimension", () => {
    expect(() => attention({ ...base, mask: new Float32Array(B * S), maskShape: [B, 3, 1] })).toThrow(
      /maskHeads must be 1 or 2/,
    );
  });

  it("rejects a mask of the wrong length", () => {
    expect(() => attention({ ...base, mask: new Float32Array(B * S + 1) })).toThrow(
      /needs 8 elements, got 9/,
    );
  });

  it("rejects a maskShape with no mask, rather than ignoring it", () => {
    expect(() => attention({ ...base, maskShape: [B, H, L] })).toThrow(/maskShape given without a mask/);
  });

  it("names the op that rejected, since three ops share the check", () => {
    expect(() => resolveMask({ B, H, L, S, mask: new Float32Array(1) }, "gqa")).toThrow(/^gqa\(\):/);
  });
});

describe("keyPaddingBias", () => {
  it("maps truthy to 0 and falsy to -Infinity, torch's attn_mask polarity", () => {
    expect(Array.from(keyPaddingBias(2, 2, [true, false, 0, 1]))).toEqual([0, -Infinity, -Infinity, 0]);
  });

  it("rejects a flag count that is not B * S", () => {
    expect(() => keyPaddingBias(2, 2, [true])).toThrow(/expected 4 flags, got 1/);
  });
});
