/**
 * Comparing kernel variants on a device whose clock does not hold still.
 *
 * Issue #177. Every naive way of doing this was tried here and produced a wrong
 * answer:
 *
 *   - **Timing candidates in order.** An idle RTX 5090 sits at 195 MHz and a
 *     loaded one at 2910. The first candidates in a sweep are measured on the
 *     ramp and the last ones at full clock, so the ranking is partly the order.
 *     The same kernel measured 3.58 ms, 6.44 ms and 8.43 ms on three runs.
 *   - **Warming up first.** It fixes the ramp and not the drift: the clock
 *     still moves across a fifteen-candidate sweep, and nothing in the numbers
 *     says by how much.
 *   - **Reading the clock.** `nvidia-smi` is not a thing a WebGPU library can
 *     depend on, and a figure from it describes the moment it was sampled
 *     rather than the dispatch it is attached to.
 *
 * What works is **interleaving a baseline**. Each round measures the reference
 * dispatch and every candidate, back to back; a candidate is reported as its
 * ratio to the reference *from the same round*, so a clock that moved between
 * rounds moves both and cancels. That is `harness/roofline.ts`'s own argument
 * for measuring a ceiling rather than quoting one, applied to a comparison.
 *
 * The rounds also walk the candidates in **opposite orders**, and the report
 * says whether the ranking survived. A ranking that changes when the order does
 * is a ranking of the order.
 */
import type { Dispatch, Runner } from "./wgsl.js";

export interface SweepEntry<T> {
  candidate: T;
  label: string;
  dispatch: Dispatch;
}

export interface SweepResult<T> {
  candidate: T;
  label: string;
  /** Best seconds seen, for reporting a rate. */
  seconds: number;
  /** Best ratio to the reference measured in the same round. Below 1 is faster. */
  ratio: number;
  /** Ratio spread across rounds — how much the device moved under this one. */
  spread: number;
}

export interface SweepReport<T> {
  results: SweepResult<T>[];
  /** The reference's own best time, for absolute rates. */
  referenceSeconds: number;
  /**
   * Whether the first and last round ranked the candidates the same way.
   *
   * False does not mean the sweep is worthless — it means the differences are
   * inside the device's own noise, and the top entries should be treated as
   * tied rather than ordered.
   */
  rankingStable: boolean;
  /** How far the reference itself moved between rounds, as max/min. */
  referenceDrift: number;
}

/**
 * Times `entries` against `reference`, interleaved, over several rounds.
 *
 * `rounds` is at least two, because one round cannot tell a ranking from an
 * ordering. Each candidate is warmed on its own dispatch before its first
 * timing — the clock ramps with the work, and a microsecond probe does not move
 * it.
 */
export async function sweep<T>(
  runner: Runner,
  reference: Dispatch,
  entries: SweepEntry<T>[],
  { rounds = 3, warmRounds = 3 }: { rounds?: number; warmRounds?: number } = {},
): Promise<SweepReport<T> | null> {
  if (entries.length === 0) return null;

  const warm = async (dispatch: Dispatch): Promise<void> => {
    for (let i = 0; i < warmRounds; i += 1) await runner.run(dispatch);
  };

  await warm(reference);
  for (const entry of entries) await warm(entry.dispatch);

  const ratios = new Map<string, number[]>();
  const times = new Map<string, number[]>();
  const referenceTimes: number[] = [];
  const orderPerRound: string[][] = [];

  for (let round = 0; round < Math.max(2, rounds); round += 1) {
    const referenceSeconds = await runner.time(reference);
    if (referenceSeconds === null) return null;
    referenceTimes.push(referenceSeconds);

    // Opposite orders on alternate rounds. Whatever the device does over the
    // course of a round, it does it to the candidates in both directions.
    const order = round % 2 === 0 ? entries : [...entries].reverse();
    const ranking: { label: string; ratio: number }[] = [];
    for (const entry of order) {
      const seconds = await runner.time(entry.dispatch);
      if (seconds === null) continue;
      const ratio = seconds / referenceSeconds;
      ratios.set(entry.label, [...(ratios.get(entry.label) ?? []), ratio]);
      times.set(entry.label, [...(times.get(entry.label) ?? []), seconds]);
      ranking.push({ label: entry.label, ratio });
    }
    ranking.sort((a, b) => a.ratio - b.ratio);
    orderPerRound.push(ranking.map((r) => r.label));
  }

  const results: SweepResult<T>[] = entries
    .filter((e) => (ratios.get(e.label) ?? []).length > 0)
    .map((e) => {
      const r = ratios.get(e.label)!;
      const t = times.get(e.label)!;
      return {
        candidate: e.candidate,
        label: e.label,
        seconds: Math.min(...t),
        ratio: Math.min(...r),
        spread: Math.max(...r) / Math.min(...r),
      };
    })
    .sort((a, b) => a.ratio - b.ratio);

  const first = orderPerRound[0] ?? [];
  const last = orderPerRound[orderPerRound.length - 1] ?? [];
  // Only the top few need to agree: the tail is noise either way, and demanding
  // a total order would report instability that no decision depends on.
  const top = Math.min(5, first.length, last.length);
  const rankingStable = first.slice(0, top).join() === last.slice(0, top).join();

  return {
    results,
    referenceSeconds: Math.min(...referenceTimes),
    rankingStable,
    referenceDrift: Math.max(...referenceTimes) / Math.min(...referenceTimes),
  };
}

/**
 * The lines a sweep should print, including the ones that say not to trust it.
 *
 * Separate from `sweep` so a caller cannot report the numbers without the
 * caveats: a ranking taken from an unstable sweep looks exactly like one taken
 * from a stable sweep.
 */
export function describeSweep<T>(report: SweepReport<T>, flops: number, roofline: number | null): string[] {
  const lines: string[] = [];
  lines.push(
    `  reference ${(report.referenceSeconds * 1000).toFixed(2)} ms, and it moved ` +
      `${((report.referenceDrift - 1) * 100).toFixed(0)}% between rounds`,
  );
  if (!report.rankingStable) {
    lines.push("  **the top five ranked differently in the first and last round — treat them as tied**");
  }
  if (report.referenceDrift > 1.15) {
    lines.push(
      `  **the reference itself moved ${((report.referenceDrift - 1) * 100).toFixed(0)}% — the device is not holding ` +
        "still, and every ratio below is that much less certain**",
    );
  }
  lines.push("  ratio  spread   ms      TFLOP/s  roofline  candidate");
  for (const r of report.results.slice(0, 6)) {
    const rate = flops / r.seconds;
    const share = roofline === null ? "n/a" : `${((rate / roofline) * 100).toFixed(1)}%`;
    lines.push(
      `  ${r.ratio.toFixed(2).padStart(5)}  ${`${((r.spread - 1) * 100).toFixed(0)}%`.padStart(6)}  ` +
        `${(r.seconds * 1000).toFixed(2).padStart(7)} ${(rate / 1e12).toFixed(2).padStart(8)} ${share.padStart(9)}  ${r.label}`,
    );
  }
  return lines;
}
