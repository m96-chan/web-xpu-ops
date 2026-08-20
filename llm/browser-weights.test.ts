import { describe, expect, it } from "vitest";
import { loadWeightsQ8FromUrl, type WeightFetchProgress } from "./browser-weights.js";

/**
 * `loadWeightsQ8FromUrl` against a fake `fetch` — no network, no browser, no
 * GPU. Real `Response`/`ReadableStream` objects (Node 22 has both as
 * globals), so the streaming-progress code path in `browser-weights.ts` runs
 * for real rather than being short-circuited by a mock that skips straight
 * to `arrayBuffer()`.
 *
 * The fixture is the smallest manifest `buildLlamaWeightsQ8` accepts:
 * `numLayers: 0`, so only `embedTokens` / `finalNorm` / `lmHead` are
 * required (see `weights-q8.ts#assertWeightShapesQ8`, which skips every
 * per-layer check when `weights.layers.length === 0 === numLayers`).
 */

const CONFIG = {
  numLayers: 0,
  hiddenSize: 2,
  numHeads: 1,
  numKvHeads: 1,
  headDim: 1,
  ffnHidden: 1,
  vocabSize: 3,
  ropeTheta: 10000,
  rmsNormEps: 1e-5,
  tieEmbeddings: false,
};

// embedTokens: [3, 2] int8 codes at codes[0..6), scale at scales[0..3).
// lmHead:      [3, 2] int8 codes at codes[6..12), scale at scales[3..6).
// finalNorm:   [2] f32 at norms[0..2).
const MANIFEST = {
  config: CONFIG,
  weights: [
    { name: "embedTokens", kind: "quant", shape: [3, 2], codesOffset: 0, scaleOffset: 0 },
    { name: "finalNorm", kind: "norm", shape: [2], offset: 0 },
    { name: "lmHead", kind: "quant", shape: [3, 2], codesOffset: 6, scaleOffset: 3 },
  ],
};

const CODES = Int8Array.from([1, -2, 3, -4, 5, -6, 7, -8, 9, -10, 11, -12]);
const SCALES = Float32Array.from([0.1, 0.2, 0.3, 0.4, 0.5, 0.6]);
const NORMS = Float32Array.from([1.5, -1.5]);

function fakeFetch(input: string | URL | Request): Promise<Response> {
  const url = String(input);
  const base = "http://demo.invalid/weights";
  const respond = (body: string | ArrayBuffer, contentType: string) => {
    const bytes = typeof body === "string" ? new TextEncoder().encode(body) : new Uint8Array(body);
    return Promise.resolve(
      new Response(bytes, {
        status: 200,
        headers: { "content-type": contentType, "content-length": String(bytes.byteLength) },
      }),
    );
  };
  if (url === `${base}/manifest.json`) return respond(JSON.stringify(MANIFEST), "application/json");
  if (url === `${base}/weights.codes.bin`) return respond(CODES.buffer as ArrayBuffer, "application/octet-stream");
  if (url === `${base}/weights.scales.bin`) return respond(SCALES.buffer as ArrayBuffer, "application/octet-stream");
  if (url === `${base}/weights.norms.bin`) return respond(NORMS.buffer as ArrayBuffer, "application/octet-stream");
  return Promise.resolve(new Response("not found", { status: 404 }));
}

describe("loadWeightsQ8FromUrl", () => {
  it("parses a fetched manifest into the same shape loadConvertedWeightsQ8 builds from disk", async () => {
    const { config, weights } = await loadWeightsQ8FromUrl(
      "http://demo.invalid/weights",
      /* maxSeqLen */ 128,
      undefined,
      fakeFetch,
    );
    expect(config.hiddenSize).toBe(2);
    expect(config.vocabSize).toBe(3);
    expect(config.maxSeqLen).toBe(128);
    expect(Array.from(weights.embedTokens.codes)).toEqual([1, -2, 3, -4, 5, -6]);
    expect(Array.from(weights.embedTokens.scale)).toEqual([0.1, 0.2, 0.3].map((x) => Math.fround(x)));
    expect(Array.from(weights.lmHead.codes)).toEqual([7, -8, 9, -10, 11, -12]);
    expect(Array.from(weights.finalNorm)).toEqual([1.5, -1.5]);
    expect(weights.layers).toHaveLength(0);
  });

  it("reports progress ticks with monotonically increasing loadedBytes per file, ending at the true total", async () => {
    const ticks: WeightFetchProgress[] = [];
    await loadWeightsQ8FromUrl("http://demo.invalid/weights", 128, (p) => ticks.push({ ...p }), fakeFetch);

    const codesTicks = ticks.filter((t) => t.file === "codes");
    expect(codesTicks.length).toBeGreaterThan(0);
    for (let i = 1; i < codesTicks.length; i += 1) {
      expect(codesTicks[i]!.loadedBytes).toBeGreaterThanOrEqual(codesTicks[i - 1]!.loadedBytes);
    }
    expect(codesTicks[codesTicks.length - 1]!.loadedBytes).toBe(CODES.byteLength);
    expect(codesTicks[codesTicks.length - 1]!.totalBytes).toBe(CODES.byteLength);

    // Every file must appear at least once — a silent skip would leave the
    // demo's progress bar stuck on a phase that already finished.
    for (const file of ["manifest", "codes", "scales", "norms"]) {
      expect(ticks.some((t) => t.file === file)).toBe(true);
    }
  });

  it("rejects a 404 with the URL and status in the message, not a generic fetch error", async () => {
    await expect(
      loadWeightsQ8FromUrl("http://demo.invalid/does-not-exist", 128, undefined, fakeFetch),
    ).rejects.toThrow(/404/);
  });

  it("reports totalBytes as null, never NaN, when Content-Length is unparsable", async () => {
    // WeightFetchProgress documents totalBytes as "number | null, null when
    // unknown". Number("garbage") is NaN, which is neither — it flows into
    // `p.totalBytes ? ...` consumers as falsy-but-non-null and into any
    // fraction as NaN.
    const garbageLengthFetch = async (input: string | URL | Request): Promise<Response> => {
      const res = await fakeFetch(input);
      const headers = new Headers(res.headers);
      headers.set("content-length", "garbage");
      return new Response(res.body, { status: res.status, headers });
    };
    const ticks: WeightFetchProgress[] = [];
    await loadWeightsQ8FromUrl("http://demo.invalid/weights", 128, (p) => ticks.push({ ...p }), garbageLengthFetch);
    expect(ticks.length).toBeGreaterThan(0);
    for (const tick of ticks) {
      expect(tick.totalBytes === null || Number.isFinite(tick.totalBytes)).toBe(true);
    }
  });
});
