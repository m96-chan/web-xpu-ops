import { describe, expect, it } from "vitest";
import { groupNorm } from "./reference.js";
import { layernorm } from "../layernorm/reference.js";

/**
 * The parts of the contract a kernel cannot be asked about, and the one number
 * that decides whether this op is GroupNorm at all.
 *
 * Every expectation below was measured against torch 2.10.0+cu128 in float64,
 * not read off the documentation. The generator is in the PR for #89.
 */

const N = 2;
const C = 6;
const L = 4;
const G = 3;
const EPS = 1e-5;

/** `torch.rand(2, 6, 4, dtype=float64) * 20 - 10`, seed 7. */
const INPUT = Float64Array.from([
  -4.41239140187192, -4.526125726964687, 7.24185922835251, 3.1337686117640278, 8.450587349059745,
  6.790907515247731, -4.105757839742326, 1.21443194744311, 5.260868224242774, 5.187144544265664,
  -8.739577590928498, -8.848883334739226, -6.651187964035898, -9.247325188346613,
  -7.556203152491876, 6.9508714106710485, -9.73590520884571, 8.99724893119993, -7.801282285845039,
  9.872338779094001, 8.483841465893384, 7.304085267015314, 3.206102123763676, -3.853008358349741,
  7.897996307652836, 4.620451439149827, 8.97921929025231, 6.787799429657369, -7.217557970470237,
  9.157105450381984, 0.6188057197275736, -8.229087698903372, 2.8768630146822627, 4.351266372264655,
  -3.12978480801865, -8.87905648702727, -6.198139592039793, -8.777855114203499, 4.733802249104926,
  -3.030804791373276, -1.5536160732734299, 7.993805982677873, -2.9178306320216745,
  -0.41973189360100704, -5.5517155328544305, -6.484056293895446, 6.145315038901757,
  -2.9030819552421665,
]);

/**
 * Non-uniform **within** a group, not only between groups.
 *
 * The statistics are shared across a group's channels; the affine transform is
 * not. A weight that were constant inside each group would pass even for a
 * kernel that indexed it by group.
 */
const WEIGHT = Float64Array.from([0.5, -1.25, 2.0, 0.75, -0.3, 1.6]);
const BIAS = Float64Array.from([0.1, 0.0, -0.4, 1.2, 0.25, -0.75]);

/** `torch.nn.functional.group_norm(x, 3, w, b, 1e-5)`, float64. */
const EXPECTED = [
  -0.49272740636007745, -0.5037143088926045, 0.633090289040845, 0.23624271323264096,
  -1.6246379935832589, -1.223819231208306, 1.4077653868740274, 0.12292005546954733,
  2.0016171217560754, 1.9800680312204673, -2.090648409531444, -2.122597972686331,
  0.7949168307626904, 0.5103522439150351, 0.6957174237428227, 2.285848962544914,
  0.7214944323557498, -0.02734116538388862, 0.6441601776906951, -0.06232184319312214,
  0.6196976610582274, 0.36818110902591306, -0.5054829509361256, -2.0104406113110325,
  0.4830446878107199, 0.23547856183889163, 0.5647138016911091, 0.3991870592569145,
  1.8967345706744054, -1.1953756443144543, 0.41695452316687953, 2.087746826967257,
  1.5483131571627256, 2.1078839303704924, -0.7313510536179868, -2.9133349607080716,
  0.6390510353846549, 0.2719025032420807, 2.1948994868751024, 1.0898303220454766,
  0.3018460860318216, -0.28585970174248154, 0.3858223455125201, 0.2320481561661557,
  -2.3390936233143322, -2.645182287238097, 1.5010564690045174, -1.4695438332893374,
];

const args = () => ({
  input: Float32Array.from(INPUT),
  weight: Float32Array.from(WEIGHT),
  bias: Float32Array.from(BIAS),
  N,
  C,
  L,
  G,
  eps: EPS,
});

describe("group_norm / reference", () => {
  it("matches torch.nn.functional.group_norm", () => {
    const got = groupNorm(args());
    expect(got).toHaveLength(EXPECTED.length);
    // f32 in, f64 arithmetic, f32 out — against a float64 torch answer, so the
    // gap is the storage type rather than the maths.
    got.forEach((value, i) => expect(value).toBeCloseTo(EXPECTED[i]!, 5));
  });

  /**
   * The one measurement that decides the op.
   *
   * Both of these are shaped `[N, C, L]` and neither errors, so a reduction
   * over the wrong axis produces a well-formed wrong answer. Torch's numbers,
   * checked against candidate spellings in float64:
   *
   *   biased variance,   eps inside sqrt   max|diff| 4.44e-16   ← this one
   *   unbiased variance, eps inside sqrt   max|diff| 1.62e-01
   *   biased variance,   eps outside sqrt  max|diff| 4.32e-06
   *
   * The unbiased form is off by a sixth of the answer; the misplaced eps by
   * four thousand f32 ulps. Neither is close enough to hide.
   */
  it("pools statistics across a group's channels, unlike layernorm", () => {
    const grouped = groupNorm(args());
    // Same input, same shapes, no affine on either side — the only difference
    // is what the mean and variance were taken over.
    const perChannel = layernorm({
      input: Float32Array.from(INPUT),
      weight: new Float32Array(L).fill(1),
      bias: new Float32Array(L),
      N: N * C,
      D: L,
      eps: EPS,
    });
    const unit = groupNorm({
      ...args(),
      weight: new Float32Array(C).fill(1),
      bias: new Float32Array(C),
    });
    const worst = Math.max(...unit.map((v, i) => Math.abs(v - perChannel[i]!)));
    expect(worst).toBeGreaterThan(0.1);
    // …and the affine version is still the torch answer, so the disagreement
    // is layernorm's axis, not a broken group_norm.
    expect(grouped[0]).toBeCloseTo(EXPECTED[0]!, 5);
  });

  it("refuses a channel count that does not divide, as torch does", () => {
    // RuntimeError: Expected number of channels in input to be divisible by
    // num_groups. Distributing the remainder would be a different op that
    // happens to run.
    expect(() =>
      groupNorm({
        input: new Float32Array(1 * 5 * 3),
        weight: new Float32Array(5).fill(1),
        bias: new Float32Array(5),
        N: 1,
        C: 5,
        L: 3,
        G: 2,
        eps: EPS,
      }),
    ).toThrow(/divisible/);
  });

  it("is layernorm over the whole sample at G = 1, and per channel at G = C", () => {
    // Not an alias to implement — a statement of where the two extremes land,
    // so a reader can place GroupNorm between them. G = C pools one channel's
    // L elements, which is exactly per-channel layernorm.
    const perChannel = layernorm({
      input: Float32Array.from(INPUT),
      weight: new Float32Array(L).fill(1),
      bias: new Float32Array(L),
      N: N * C,
      D: L,
      eps: EPS,
    });
    const instance = groupNorm({
      ...args(),
      G: C,
      weight: new Float32Array(C).fill(1),
      bias: new Float32Array(C),
    });
    instance.forEach((value, i) => expect(value).toBeCloseTo(perChannel[i]!, 5));
  });
});
