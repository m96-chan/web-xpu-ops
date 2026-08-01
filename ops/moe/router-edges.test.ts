import { describe } from "vitest";
import { expectAgrees, gpuTest, kernel, useGpu } from "../../harness/index.js";
import { router } from "./reference.js";
import { routerParams, tokenThreads, wave, withJunk } from "./testing.js";

const code = kernel(import.meta.url, "router");

function routed(T: number, E: number, k: number, normalize: boolean, logits: Float32Array) {
  const { expert, gate } = router({ logits, T, E, k, normalize });
  return { T, E, k, normalize, logits, expert, gate };
}

/**
 * torch.topk leaves ties open and a kernel cannot. Three rows: all equal, a tie
 * at the top with a distinct runner-up, and a tie below the winner. Without a
 * rule the GPU is free to return any of them in any order, so this is what pins
 * the kernel to the reference's choice. `reference.test.ts` pins the reference.
 */
const TIES = routed(3, 5, 3, true, Float32Array.from([
  2, 2, 2, 2, 2, //
  1, 5, 3, 5, 1,
  9, 4, 4, 4, 0,
]));

/**
 * exp(800) is Infinity in f32 and takes the row to NaN, which would make every
 * gate NaN and every comparison in the top-k scan false. Router logits are not
 * small, so subtracting the row maximum is load-bearing.
 */
const OVERFLOW = routed(1, 8, 3, true, Float32Array.from([100, 200, 300, 400, 500, 600, 700, 800]));

/**
 * k is a config value and E is a checkpoint's; they do disagree. The slots that
 * cannot be filled have to say so, because dispatch reads them.
 */
const SURPLUS = routed(4, 3, 5, true, wave(4 * 3, 0.9));

/**
 * T is not a multiple of 256, so the last workgroup has threads with no token.
 * Two things make their writes observable: output buffers as long as the
 * dispatch rather than as long as the result (a write past the end is discarded
 * by the driver, so an exactly sized buffer cannot fail a bounds check), and a
 * logits buffer with real values past T*E (a read past the end returns zero, and
 * a row of zeros softmaxes to the same 1/k a normalised two-way gate produces,
 * which is exactly what the guard is being tested for).
 */
const TAIL = (() => {
  const [T, E, k] = [300, 6, 2];
  const padded = tokenThreads(T)[0] * 256;
  const logits = withJunk(wave(T * E, 0.23), (padded - T) * E);
  const { expert, gate } = router({ logits, T, E, k, normalize: true });
  const paddedExpert = new Int32Array(padded * k);
  paddedExpert.set(expert);
  const paddedGate = new Float32Array(padded * k);
  paddedGate.set(gate);
  return { T, E, k, normalize: true, logits, expert: paddedExpert, gate: paddedGate };
})();

describe("moe router edges / wgsl", () => {
  useGpu();

  for (const [name, fixture] of [
    ["breaks ties towards the lower expert index", TIES],
    ["survives logits that would overflow exp", OVERFLOW],
    ["fills surplus slots with -1 when k exceeds the expert count", SURPLUS],
    ["leaves the tail of its output buffers alone", TAIL],
  ] as const) {
    gpuTest(name, async (run) => {
      const { T, E, k, normalize, logits, expert, gate } = fixture;
      await expectAgrees(
        run,
        {
          code,
          bindings: [
            { kind: "storage", data: logits },
            { kind: "out", type: "i32", length: expert.length },
            { kind: "out", type: "f32", length: gate.length },
            { kind: "uniform", data: routerParams(T, E, k, normalize) },
          ],
          workgroups: tokenThreads(T),
        },
        [expert, gate],
      );
    });
  }
});
