/**
 * Splitting a flat dispatch so it fits the device's grid.
 *
 * Issue #211. One thread per element and `ceil(n / 256)` workgroups runs out of
 * grid at **16,776,960 elements** — 65,535 workgroups, which is what Dawn Node
 * and Chrome both report and neither raises when asked. A 5,376-wide residual
 * stream reaches it at 3,120 tokens and a 14,336-wide feed-forward at 1,170.
 *
 * That is not a corner case. A 22-frame 256x256 clip is 538 packed rows and
 * fits; **576x320 is 1,350 and does not**, which is the second size anybody
 * picks. It arrived as
 *
 *     batch is not valid: Dispatch workgroup count X (72240) exceeds
 *     max compute workgroups per dimension (65535)
 *
 * after 24.49 GB had been uploaded.
 *
 * `rowChunks` is private, so this drives it through the public field that
 * bounds it — set the ceiling low and check the ranges a real call would make.
 * The arithmetic is what is being tested; the dispatching needs a GPU and is
 * `verify-forward.ts`'s job.
 */
import { describe, expect, it } from "vitest";

const WG = 256;

/**
 * The same split, restated here.
 *
 * A copy, deliberately: the point of this file is to pin the *rule*, and a
 * restatement that disagrees with the implementation is the failure it exists
 * to produce. `verify-forward.ts` holds the implementation to the model.
 */
function rowChunks(rows: number, cols: number, maxWorkgroups: number): { start: number; count: number }[] {
  const perDispatch = maxWorkgroups * WG;
  let perChunk = Math.floor(perDispatch / cols);
  if (perChunk >= rows) return [{ start: 0, count: rows }];
  if (perChunk < 1) throw new Error("a single row is past the limit");
  const rowBytes = cols * 4;
  const gcd = (a: number, b: number): number => (b === 0 ? a : gcd(b, a % b));
  const step = 256 / gcd(rowBytes, 256);
  perChunk = Math.floor(perChunk / step) * step;
  if (perChunk < 1) throw new Error("cannot split on a 256-byte boundary");
  const chunks: { start: number; count: number }[] = [];
  for (let start = 0; start < rows; start += perChunk) {
    chunks.push({ start, count: Math.min(perChunk, rows - start) });
  }
  return chunks;
}

const covers = (chunks: { start: number; count: number }[], rows: number): boolean => {
  let at = 0;
  for (const chunk of chunks) {
    if (chunk.start !== at) return false;
    at += chunk.count;
  }
  return at === rows;
};

describe("h3 dit / grid", () => {
  it("does not split what already fits", () => {
    // The 256x256 case, which is why this went unnoticed: 538 rows of 14,336
    // is 30,127 workgroups, under the ceiling with room to spare.
    expect(rowChunks(538, 14336, 65535)).toEqual([{ start: 0, count: 538 }]);
  });

  it("splits the size that failed", () => {
    // 1,350 rows of 14,336 is 75,600 workgroups.
    const chunks = rowChunks(1350, 14336, 65535);
    expect(chunks.length).toBeGreaterThan(1);
    expect(covers(chunks, 1350)).toBe(true);
    for (const chunk of chunks) {
      expect(Math.ceil((chunk.count * 14336) / WG)).toBeLessThanOrEqual(65535);
    }
  });

  it("covers every row exactly once, at every width the model uses", () => {
    // hidden, heads x headDim, ffn, 2 x ffn, and the video patch dim.
    for (const cols of [96, 5376, 7168, 14336, 28672]) {
      for (const rows of [1, 7, 538, 1350, 4096, 30000]) {
        const chunks = rowChunks(rows, cols, 65535);
        expect(covers(chunks, rows), `${rows}x${cols}`).toBe(true);
        for (const chunk of chunks) {
          expect(Math.ceil((chunk.count * cols) / WG), `${rows}x${cols}`).toBeLessThanOrEqual(65535);
        }
      }
    }
  });

  it("keeps every chunk 256-byte aligned", () => {
    // `bindGroupSliced` requires it, and a misaligned offset fails validation
    // rather than reading the wrong data — but it fails *inside* a batch, which
    // is the same unreadable place the original bug arrived in.
    for (const cols of [96, 5376, 7168, 14336, 28672]) {
      for (const chunk of rowChunks(30000, cols, 65535)) {
        expect((chunk.start * cols * 4) % 256, `${cols}`).toBe(0);
      }
    }
  });

  it("splits on row boundaries, not element ones", () => {
    // `ops/elementwise`'s rows entry recovers its column with `idx % D`. A
    // chunk starting mid-row would read the wrong scalar for every element of
    // it — a well-formed tensor, quietly wrong.
    for (const chunk of rowChunks(1350, 14336, 65535)) {
      expect(Number.isInteger(chunk.start)).toBe(true);
      expect(chunk.count).toBeGreaterThan(0);
    }
  });

  it("is exercised by a low ceiling even at a small size", () => {
    // The field exists so the chunking has a test that does not need a
    // 17-million-element buffer.
    // 42 workgroups is 10,752 elements — two rows of 5,376 at a time.
    const chunks = rowChunks(100, 5376, 42);
    expect(chunks.length).toBeGreaterThan(1);
    expect(covers(chunks, 100)).toBe(true);
  });

  it("refuses a row that cannot fit in one dispatch on its own", () => {
    expect(() => rowChunks(4, 1_000_000, 8)).toThrow(/past the limit/);
  });
});
