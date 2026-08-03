import { describe } from "vitest";
import { expectAgrees, gpuTest, kernel, params, useGpu } from "../../harness/index.js";
import { ACTIVATION, type ActivationKind, activation } from "./reference.js";

const code = kernel(import.meta.url);

const WG = 256;
/** Not a multiple of 256, so the bounds check spans two workgroups and matters. */
const N = 300;
const DISPATCH = Math.ceil(N / WG) * WG;

/**
 * Saturating magnitudes on both sides, the knee, and either side of zero.
 *
 * `-2.699` and `2.699` are where the two GELUs are furthest apart (4.73e-4,
 * measured in the reference), so an implementation that quietly used one for
 * the other fails rather than passes by looking similar. `±1e-4` is where
 * `exp(x) - 1` cancels hardest in ELU.
 */
const PROBES = [
  -20, -8, -4, -3, -2.699, -2, -1, -0.5, -0.05, -1e-4, 0, 1e-4, 0.05, 0.5, 1, 2, 2.699, 3, 4, 8, 20,
];
const input = Float32Array.from({ length: N }, (_, i) =>
  i < PROBES.length ? PROBES[i]! : Math.sin(i * 0.37) * 8,
);

/**
 * A non-zero value past the `N` elements the kernel is meant to touch.
 *
 * The bounds check cannot be observed any other way here: this device reads 0
 * out of range, and every activation in this op maps 0 to 0 — so a kernel that
 * dropped the check would compute the expected zero by accident and the
 * mutation would survive. With a sentinel the tail says what happened.
 */
const SENTINEL = 3;
const padded = new Float32Array(DISPATCH).fill(SENTINEL);
padded.set(input);

/** Built before the device exists: CPU work between dispatches kills the worker (#49). */
function expect(kind: ActivationKind, alpha?: number): Float32Array {
  const out = new Float32Array(DISPATCH);
  out.set(activation({ input, kind, alpha }));
  return out;
}

const cases: { label: string; kind: ActivationKind; alpha?: number; expected: Float32Array }[] = [
  { label: "relu2", kind: ACTIVATION.relu2, expected: expect(ACTIVATION.relu2) },
  { label: "silu", kind: ACTIVATION.silu, expected: expect(ACTIVATION.silu) },
  { label: "elu (default alpha 1.0)", kind: ACTIVATION.elu, expected: expect(ACTIVATION.elu) },
  // A second alpha, because with only the default one a kernel that ignored
  // the uniform entirely — or hard-coded 1.0 — would pass.
  {
    label: "elu at alpha 0.7",
    kind: ACTIVATION.elu,
    alpha: 0.7,
    expected: expect(ACTIVATION.elu, 0.7),
  },
  { label: "tanh", kind: ACTIVATION.tanh, expected: expect(ACTIVATION.tanh) },
  { label: "gelu (exact, torch's default)", kind: ACTIVATION.gelu, expected: expect(ACTIVATION.gelu) },
  { label: "gelu_tanh", kind: ACTIVATION.gelu_tanh, expected: expect(ACTIVATION.gelu_tanh) },
];

describe("activation / wgsl", () => {
  useGpu();

  for (const { label, kind, alpha, expected } of cases) {
    gpuTest(`agrees with the reference for ${label}`, async (run) => {
      // Default tolerance throughout, and measured rather than assumed. Worst
      // absolute error against the f64 reference over x ∈ [-20, 20] on this
      // device: relu2 1.52e-5 (but 5.9e-8 relative — it is squaring numbers
      // near 400), tanh 6.6e-8, elu 7.6e-8, gelu 3.69e-7, gelu_tanh 4.02e-7.
      // Every one of those sits inside `rel 1e-5, abs 1e-6`, so nothing here is
      // widened. GELU's is the interesting one: `erf` does not exist in WGSL
      // and the approximation standing in for it is the largest error in this
      // op.
      await expectAgrees(
        run,
        {
          code,
          bindings: [
            { kind: "storage", data: padded },
            { kind: "out", type: "f32", length: DISPATCH },
            {
              kind: "uniform",
              data: params([
                ["u32", N],
                ["u32", kind],
                ["f32", alpha ?? 1],
              ]),
            },
          ],
          workgroups: [DISPATCH / WG],
        },
        [expected],
      );
    });
  }
});
