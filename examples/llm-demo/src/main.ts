/**
 * Browser demo entry point (issue #106): loads the converted Sarashina2.2-1B-
 * alibi-v1 int8 checkpoint over HTTP, tokenizes a prompt built the same way
 * `technologies-moe/alibi-ai`'s `assets/llm.js` does, and runs
 * `LlamaEngineQ8` on a real WebGPU device — the first time this repository's
 * `llm/` engine has generated anything in a browser rather than under Node
 * (see `engine-q8.real-model.test.ts` / PR #108 / issue #107 for why Node's
 * `webgpu` (Dawn) binding could not complete this on the machine that PR was
 * built on, and why that gate moved here).
 *
 * Everything this file imports from `llm/` is unmodified library code —
 * `SentencePieceTokenizer`, `sampleNext`, `LineFormatConstraint`,
 * `LlamaEngineQ8`, `loadWeightsQ8FromUrl` — the same symbols `llm/index.ts`
 * re-exports (this issue's other half, verified by `llm/index.test.ts`).
 * Imported here from each module directly rather than through
 * `llm/index.js` itself: that barrel also re-exports
 * `real-model-weights.ts#loadConvertedWeightsQ8`, which imports `node:fs` —
 * fine for `llm/index.ts` as a package entry point (Node and browser callers
 * alike just don't call the one export that needs a filesystem), but fatal
 * for `esbuild --bundle --platform=browser`: bundling *resolves* every
 * `import` statement reachable from the entry point before it can tree-shake
 * anything, and `node:fs`/`node:path` have no browser build to resolve to.
 * Importing the six modules this demo actually uses individually keeps that
 * one Node-only file out of the bundle's graph entirely.
 *
 * The only browser-specific code in this demo is `browser-runtime.ts` (a
 * `Runner` over `navigator.gpu` in place of `harness/wgsl.ts`'s Node/Dawn
 * one) and this file's UI glue.
 */
import { LineFormatConstraint, type LineFormatSpec } from "../../../llm/constraints/line-format.js";
import type { TokenCodec } from "../../../llm/constraints/token-codec.js";
import { argmax } from "../../../llm/engine.js";
import { LlamaEngineQ8 } from "../../../llm/engine-q8.js";
import { createForwardProfile, LlamaEngineQ8Resident, type ForwardProfile } from "../../../llm/engine-q8-resident.js";
import { sampleNext, type Constraint } from "../../../llm/sampler.js";
import { SentencePieceTokenizer, type TokenizerVocab } from "../../../llm/tokenizer.js";
import { loadWeightsQ8FromUrl } from "../../../llm/browser-weights.js";
import type { LlamaConfig } from "../../../llm/config.js";
import type { LlamaWeightsQ8 } from "../../../llm/weights-q8.js";
import { createBrowserRunner } from "./browser-runtime.js";
import { createBrowserResidentDevice } from "./browser-resident-runtime.js";

// ---------------------------------------------------------------------------
// Prompt format and system prompt: copied verbatim from
// technologies-moe/alibi-ai's assets/llm.js (the SYSTEM_PROMPT / buildPrompt
// / LISTEN_* block), per this issue's own instruction to reuse that file's
// training-pinned wording rather than inventing a new one. Not imported —
// alibi-ai is a separate repository and wiring to it is explicitly out of
// this issue's scope (see #106's "スコープ外"); this is a snapshot, dated to
// this PR, of the same constants.
// ---------------------------------------------------------------------------

const SYSTEM_PROMPT =
  "あなたは技術系同人サークル「技術萌」の看板キャラ、Alibi。" +
  "渡された内容をAlibiの口調で言い直す。各行を【表情】で始める。" +
  "表情はbase/talk/talk2/smile/pout/ohのいずれか。";

const buildStylePrompt = (policy: string, text: string): string =>
  `<|system|>${SYSTEM_PROMPT}</s><|user|>[${policy}] ${text}</s><|assistant|>`;

const LISTEN_SYSTEM_PROMPT =
  "訪問者の発言を分類する。返答はしない。次の2行だけを出力する。\n" +
  "policy: full_gear|engage|brush_off のいずれか\n" +
  "topic: 発言の主題を10文字以内\n" +
  "判定基準:\n" +
  "- full_gear: 技術の質問・相談・愚痴(プログラミング、ハードウェア、AI、インフラ等)。" +
  "「〜って何？」「〜わかんない」でも対象が技術用語ならfull_gear。\n" +
  "- engage: 挨拶。Alibi自身(様子・外見・持ち物)への質問、サークル・技術萌・同人誌・" +
  "コミケ(即売会)や自分たちのイベントの話題。「〜って何？」でも対象がサークルや同人・" +
  "コミケ関連ならengage。\n" +
  "- brush_off: 上記以外の雑談。訪問者自身の趣味・日常の報告(推し活・グルメ・買い物等、" +
  "技術以外)、興味外の話題(芸能、スポーツ、天気、占い等)。";

const LISTEN_FEWSHOT: [string, string][] = [
  ["メモリリークってどう直すの？", "policy: full_gear\ntopic: メモリリーク"],
  ["ビルドがコケる", "policy: full_gear\ntopic: ビルドエラー"],
  ["ニューラルネットって何？", "policy: full_gear\ntopic: ニューラルネットとは"],
  ["こんにちは", "policy: engage\ntopic: 挨拶"],
  ["その髪型似合ってるね", "policy: engage\ntopic: 髪型の感想"],
  ["即売会ってどんなイベント？", "policy: engage\ntopic: 即売会とは"],
  ["推しのライブ最高だった", "policy: brush_off\ntopic: 推しのライブ"],
  ["昨日ケーキ食べた", "policy: brush_off\ntopic: ケーキ"],
  ["週末友達と映画観た", "policy: brush_off\ntopic: 映画"],
];

const buildListenPrompt = (text: string): string => {
  let p = `<|system|>${LISTEN_SYSTEM_PROMPT}</s>`;
  for (const [u, a] of LISTEN_FEWSHOT) p += `<|user|>${u}</s><|assistant|>${a}</s>`;
  p += `<|user|>${text}</s><|assistant|>`;
  return p;
};

/**
 * `LineFormatConstraint` version of `assets/llm.js`'s `LISTEN_GRAMMAR`
 * (`root ::= "policy: " ("full_gear" | "engage" | "brush_off") "\ntopic: "
 * [^\n]{1,24}`) — GBNF there (wllama's grammar support), the token-trie
 * `Constraint` this repository's `llm/constraints/line-format.ts` (#102)
 * built instead. `eosTokenId` is filled in once the tokenizer's vocab is
 * loaded (see `buildListenSpec` below).
 */
function buildListenSpec(eosTokenId: number): LineFormatSpec {
  return {
    eosTokenId,
    segments: [
      { kind: "literal", text: "policy: " },
      { kind: "enum", choices: ["full_gear", "engage", "brush_off"] },
      { kind: "literal", text: "\ntopic: " },
      { kind: "freeText", forbiddenChars: ["\n"], maxLength: 24 },
    ],
  };
}

// ---------------------------------------------------------------------------
// Config
// ---------------------------------------------------------------------------

const WEIGHTS_BASE_URL = "/weights";
const VOCAB_URL = "/llm/data/sarashina2.2-1b-instruct.vocab.json";
/** `assets/llm.js`'s own `N_CTX` — kept equal so the two are comparable. */
const MAX_SEQ_LEN = 1024;
/**
 * `?maxDecodeSteps=` as a validated positive integer, not a raw `Number(...)`
 * — PR #119 review, item 9: `Number("")` is `0`, not `NaN`, so
 * `?maxDecodeSteps=` with no value (a stray `&maxDecodeSteps` in a pasted
 * URL, or a browser autofill quirk) silently produced a **0-step** decode
 * loop below — not a crash, not a short run, a measurement of nothing that
 * still prints tok/s as if it measured something. A typo'd value (`"abc"`)
 * already produced `NaN`, and `NaN` steps in a `for (step = 0; step <
 * NaN; ...)` loop never runs either, so both malformed shapes shared the
 * same silent-zero-steps failure this function now rejects instead of
 * quietly running with, per rule 8's "壊れたことに気づけない" bar.
 */
function positiveIntParam(name: string, fallback: number): number {
  const raw = new URLSearchParams(location.search).get(name);
  if (raw === null) return fallback;
  const n = Number(raw);
  if (!Number.isFinite(n) || !Number.isInteger(n) || n <= 0) {
    throw new Error(`?${name}=${JSON.stringify(raw)} must be a positive integer, got ${JSON.stringify(raw)}`);
  }
  return n;
}

/**
 * `assets/llm.js`'s own `MAX_TOKENS`, overridable via `?maxDecodeSteps=` —
 * issue #117's decode retest needs a run long enough to see `sEff`'s
 * position dependence (`sEff = position + 1` grows every step, unlike the
 * pre-#117 `S = maxSeqLen` scan, which was flat regardless of position), and
 * every prompt here reaches `</s>` in 13–25 steps on its own (`?ignoreEos=1`
 * disables that stop so a long run is not left to chance).
 */
const MAX_DECODE_STEPS = positiveIntParam("maxDecodeSteps", 200);
const IGNORE_EOS = new URLSearchParams(location.search).get("ignoreEos") === "1";

// ---------------------------------------------------------------------------
// DOM
// ---------------------------------------------------------------------------

function el<T extends HTMLElement>(id: string): T {
  const found = document.getElementById(id);
  if (!found) throw new Error(`main.ts: missing #${id} in index.html`);
  return found as T;
}

const loadBtn = el<HTMLButtonElement>("loadBtn");
const loadProgress = el<HTMLProgressElement>("loadProgress");
const loadStatus = el<HTMLElement>("loadStatus");
const genSection = el<HTMLElement>("gen");

const modeRadios = document.getElementsByName("mode") as NodeListOf<HTMLInputElement>;
const styleControls = el<HTMLElement>("styleControls");
const listenControls = el<HTMLElement>("listenControls");
const policySelect = el<HTMLSelectElement>("policy");
const listenConstraintToggle = el<HTMLInputElement>("listenConstraint");
const residentEngineToggle = el<HTMLInputElement>("residentEngine");
const inputText = el<HTMLTextAreaElement>("input");
const genBtn = el<HTMLButtonElement>("genBtn");
const promptPreview = el<HTMLElement>("promptPreview");
const output = el<HTMLElement>("output");
const stats = el<HTMLElement>("stats");

function currentMode(): "style" | "listen" {
  for (const radio of modeRadios) if (radio.checked) return radio.value as "style" | "listen";
  return "style";
}

function syncModeVisibility(): void {
  const mode = currentMode();
  styleControls.hidden = mode !== "style";
  listenControls.hidden = mode !== "listen";
}
for (const radio of modeRadios) radio.addEventListener("change", syncModeVisibility);
syncModeVisibility();

// ---------------------------------------------------------------------------
// Weight + vocab loading
// ---------------------------------------------------------------------------

interface Loaded {
  engineConfig: LlamaConfig;
  weights: LlamaWeightsQ8;
  tokenizer: SentencePieceTokenizer;
  codec: TokenCodec;
  vocab: TokenizerVocab;
  runner: Awaited<ReturnType<typeof createBrowserRunner>>;
}

let loaded: Loaded | null = null;

function formatBytes(n: number): string {
  if (n < 1024) return `${n} B`;
  if (n < 1024 * 1024) return `${(n / 1024).toFixed(1)} KiB`;
  if (n < 1024 * 1024 * 1024) return `${(n / (1024 * 1024)).toFixed(1)} MiB`;
  return `${(n / (1024 * 1024 * 1024)).toFixed(2)} GiB`;
}

/**
 * `weights.codes.bin` is the checkpoint's overwhelming majority of bytes
 * (1.4 GiB of the ~1.41 GiB total per PR #108's own numbers), so a
 * byte-weighted progress bar across all four fetched files tracks it closely
 * enough without needing to know every file's exact size up front.
 */
const PROGRESS_WEIGHT: Record<string, number> = { manifest: 0, codes: 0.95, scales: 0.03, norms: 0.02 };

async function loadEverything(): Promise<void> {
  loadBtn.disabled = true;
  const t0 = performance.now();

  loadStatus.textContent = "重みを取得中… (weights.codes.bin が支配的、数百MB〜1.4GB)";
  // One [0,1] fraction per file, combined by `PROGRESS_WEIGHT` — a file not
  // yet started simply has no entry (treated as 0), and a completed one
  // holds 1 once its last progress tick reports `loadedBytes === totalBytes`.
  const fractionByFile = new Map<string, number>();
  const { config: engineConfig, weights } = await loadWeightsQ8FromUrl(WEIGHTS_BASE_URL, MAX_SEQ_LEN, (p) => {
    fractionByFile.set(p.file, p.totalBytes ? p.loadedBytes / p.totalBytes : 0);
    loadProgress.value = Object.entries(PROGRESS_WEIGHT).reduce(
      (sum, [file, weight]) => sum + weight * (fractionByFile.get(file) ?? 0),
      0,
    );
    loadStatus.textContent = `${p.file}: ${formatBytes(p.loadedBytes)}${p.totalBytes ? ` / ${formatBytes(p.totalBytes)}` : ""}`;
  });
  const weightsMs = performance.now() - t0;

  loadStatus.textContent = "語彙(vocab.json)を取得中…";
  const vocabRes = await fetch(VOCAB_URL);
  if (!vocabRes.ok) throw new Error(`fetch ${VOCAB_URL}: HTTP ${vocabRes.status}`);
  const vocab: TokenizerVocab = await vocabRes.json();
  const tokenizer = new SentencePieceTokenizer(vocab);
  const codec: TokenCodec = {
    encode: (text) => tokenizer.encode(text),
    idToToken: (id) => tokenizer.decode([id]),
    vocabSize: vocab.vocabSize,
  };

  loadStatus.textContent = "WebGPUデバイスを初期化中…";
  const runner = await createBrowserRunner();

  loaded = { engineConfig, weights, tokenizer, codec, vocab, runner };
  loadProgress.value = 1;
  loadStatus.textContent =
    `ロード完了: 重み ${weightsMs.toFixed(0)}ms, vocabSize=${vocab.vocabSize}, ` +
    `numLayers=${engineConfig.numLayers}, hiddenSize=${engineConfig.hiddenSize}`;
  genSection.hidden = false;
}

loadBtn.addEventListener("click", () => {
  loadEverything().catch((err: unknown) => {
    loadStatus.textContent = `ロード失敗: ${err instanceof Error ? err.message : String(err)}`;
    loadBtn.disabled = false;
  });
});

// ---------------------------------------------------------------------------
// Generation
// ---------------------------------------------------------------------------

genBtn.addEventListener("click", () => {
  runGeneration()
    .catch((err: unknown) => {
      stats.textContent = `生成失敗: ${err instanceof Error ? err.message : String(err)}`;
    })
    .finally(() => {
      // Re-enable on every outcome, not only success: an over-long prompt
      // (forward throws past maxSeqLen) or a constraint dead-end in
      // sampleNext would otherwise leave the button disabled until a page
      // reload — which re-downloads the ~1.4 GiB checkpoint.
      genBtn.disabled = false;
    });
});

/** Both `LlamaEngineQ8` and `LlamaEngineQ8Resident` expose exactly this — the shape the decode loop below actually needs, regardless of which one built it. */
interface ForwardEngine {
  forward(tokens: number[]): Promise<Float32Array[]>;
}

/**
 * The sampling loop, factored out of `runGeneration` so it runs identically
 * over either engine — issue #110's whole point is that the *decode
 * structure* changed, not sampling, tokenization or when to stop, so this
 * function is exactly the part that must stay byte-for-byte the same
 * between a `LlamaEngineQ8` run and a `LlamaEngineQ8Resident` run for the
 * two to be a fair before/after comparison.
 */
async function generate(
  engine: ForwardEngine,
  promptTokens: number[],
  constraint: Constraint | undefined,
): Promise<{ tokens: number[]; prefillMs: number; decodeMsTotal: number; decodeSteps: number; stepMs: number[] }> {
  if (!loaded) throw new Error("重みが未ロードです");
  const { vocab, tokenizer } = loaded;

  const prefillStart = performance.now();
  const prefillLogits = await engine.forward(promptTokens);
  const prefillMs = performance.now() - prefillStart;

  let logits = prefillLogits[prefillLogits.length - 1]!;
  const tokens: number[] = [];
  let decodeMsTotal = 0;
  let decodeSteps = 0;
  // One entry per decode step — issue #117's decode retest wants
  // early-vs-late tok/s within a single run (`sEff = position + 1` grows
  // every step, so a long-enough run should show a slope the pre-#117
  // `S = maxSeqLen` scan, flat regardless of position, could not have had).
  // `decodeMsTotal`/`decodeSteps` above stay as the single aggregate number
  // `runGeneration`'s `stats` line already reported before this existed.
  const stepMs: number[] = [];

  for (let step = 0; step < MAX_DECODE_STEPS; step += 1) {
    const next = sampleNext(logits, tokens, { mode: "greedy" }, constraint);
    tokens.push(next);
    if (next === vocab.eosId && !IGNORE_EOS) break;

    const decodeStart = performance.now();
    const [nextLogits] = await engine.forward([next]);
    const elapsed = performance.now() - decodeStart;
    decodeMsTotal += elapsed;
    decodeSteps += 1;
    stepMs.push(elapsed);
    logits = nextLogits!;

    output.textContent = tokenizer.decode(tokens);
  }
  output.textContent = tokenizer.decode(tokens);
  return { tokens, prefillMs, decodeMsTotal, decodeSteps, stepMs };
}

async function runGeneration(): Promise<void> {
  if (!loaded) throw new Error("重みが未ロードです");
  const { engineConfig, weights, tokenizer, codec, vocab, runner } = loaded;

  const mode = currentMode();
  const text = inputText.value;
  const prompt = mode === "style" ? buildStylePrompt(policySelect.value, text) : buildListenPrompt(text);
  promptPreview.textContent = prompt;

  const promptTokens = tokenizer.encode(prompt);
  const constraint: Constraint | undefined =
    mode === "listen" && listenConstraintToggle.checked
      ? new LineFormatConstraint(codec, buildListenSpec(vocab.eosId))
      : undefined;

  genBtn.disabled = true;
  output.textContent = "";
  const useResident = residentEngineToggle.checked;
  stats.textContent = useResident ? "生成中… (GPU常駐デコード, issue #110)" : "生成中… (従来エンジン)";

  // A fresh engine per generation, still — `LlamaEngineQ8` has no reset
  // path, and constructing a new one from the already-fetched `weights` is
  // the correctness-preserving way to start clean (`engine-q8.ts`'s own doc
  // — the constructor only reads `weights`, never mutates them).
  // `LlamaEngineQ8Resident` *does* have one now (`reset()`, issue #120) —
  // this button deliberately does not use it, to keep this demo's existing
  // "every click is an independent, from-scratch generation" behaviour
  // rather than introducing session state (which prompt-style toggle applied
  // to which still-alive engine, how long its ~1.4 GiB of CPU-side weights
  // stay referenced, what happens to a live `ResidentDevice` across an error)
  // that a demo page is not the right place to design. `reset()`'s own
  // effect is measured instead via `__resetBenchmark`, below, driven by a
  // CDP script rather than this button (see the README's "reset():
  // multi-generation reuse without rebuilding" section). Cost reported
  // separately from tok/s below, since it is one-time setup, not decode-time.
  const buildStart = performance.now();
  // A `ResidentDevice` is created lazily here, per generation, rather than
  // once at load time alongside `runner` — one browser tab, two live
  // `navigator.gpu` devices for the whole session, was exactly the shape
  // that reproducibly crashed this repository's Node/Dawn binding
  // (`harness/resident.ts#runnerFromResident`'s doc); a browser is far more
  // robust than that native binding, but there is no reason to keep a
  // second device alive between generations when nothing needs it to.
  //
  // `residentDevice` is assigned *inside* the `try` below, not before it
  // (PR #116 review, item 3): `createBrowserResidentDevice()` can succeed
  // while the following `LlamaEngineQ8Resident.create()` still throws (an
  // undersized GPU rejecting one of its buffer allocations, say) — with the
  // device construction outside `try`/`finally`, that throw propagated past
  // the `finally` that destroys it, leaking one live `GPUDevice` per failed
  // click. Every native resource this function creates now lives entirely
  // inside the `try` it is destroyed in.
  let residentDevice: Awaited<ReturnType<typeof createBrowserResidentDevice>> | null = null;
  try {
    const engine: ForwardEngine = useResident
      ? await (async () => {
          residentDevice = await createBrowserResidentDevice();
          return LlamaEngineQ8Resident.create(engineConfig, weights, residentDevice);
        })()
      : new LlamaEngineQ8(engineConfig, weights, runner.run);
    const buildMs = performance.now() - buildStart;

    const { tokens, prefillMs, decodeMsTotal, decodeSteps, stepMs } = await generate(engine, promptTokens, constraint);

    const prefillTokPerSec = promptTokens.length / (prefillMs / 1000);
    const decodeTokPerSec = decodeSteps > 0 ? decodeSteps / (decodeMsTotal / 1000) : 0;

    // Issue #117's own ask: decode tok/s split early vs late in one run,
    // since `sEff = position + 1` grows every step and the pre-#117
    // `S = maxSeqLen` scan did not depend on position at all. Halves of
    // `stepMs`, not a fixed window — meaningful at both a 13-step natural
    // run and a `?maxDecodeSteps=300&ignoreEos=1` long one.
    const half = Math.floor(stepMs.length / 2);
    const avgTokPerSec = (ms: number[]) => (ms.length > 0 ? 1000 / (ms.reduce((a, b) => a + b, 0) / ms.length) : 0);
    const earlyTokPerSec = avgTokPerSec(stepMs.slice(0, half));
    const lateTokPerSec = avgTokPerSec(stepMs.slice(stepMs.length - half));

    // Exposed for automated measurement (issue #117's PR) — the DOM text
    // below is for a human reading the page, this is for a script that
    // wants the exact numbers without parsing formatted text back out.
    (window as unknown as { __lastRun: unknown }).__lastRun = {
      engine: useResident ? "resident" : "legacy",
      promptTokens: promptTokens.length,
      prefillMs,
      decodeSteps,
      decodeMsTotal,
      stepMs,
      positionAtStep0: promptTokens.length,
    };

    stats.textContent =
      `engine: ${useResident ? "LlamaEngineQ8Resident (#110)" : "LlamaEngineQ8"} | ` +
      `construction: ${buildMs.toFixed(0)}ms | ` +
      `prefill: ${promptTokens.length} tok in ${prefillMs.toFixed(0)}ms (${prefillTokPerSec.toFixed(2)} tok/s) | ` +
      `decode: ${decodeSteps} tok in ${decodeMsTotal.toFixed(0)}ms (${decodeTokPerSec.toFixed(2)} tok/s) | ` +
      `decode early half: ${earlyTokPerSec.toFixed(2)} tok/s | decode late half: ${lateTokPerSec.toFixed(2)} tok/s` +
      (tokens[tokens.length - 1] === vocab.eosId ? " | stopped at </s>" : " | stopped at MAX_DECODE_STEPS");
  } finally {
    // The cast (not just `residentDevice?.destroy()`) is load-bearing: TS's
    // control-flow narrowing loses track of `residentDevice`'s declared type
    // across the `try`'s `await`/closure boundary and narrows it to `never`
    // here without it — the same workaround the pre-#116 version of this
    // `finally` already needed for the identical reason.
    (residentDevice as Awaited<ReturnType<typeof createBrowserResidentDevice>> | null)?.destroy();
  }
}

// ---------------------------------------------------------------------------
// Issue #120: measuring reset()'s own effect
// ---------------------------------------------------------------------------

/**
 * Issue #120's own completion bar: "reset経由の2生成目以降が構築コストゼロで動くこと
 * の実測", a real-hardware wall-clock comparison, not an estimate (rule 9) —
 * "構築1回+reset反復で連続 N 生成" against "都度構築 N 生成", both running
 * the exact same `promptTexts` through the exact same `generate()` this
 * file's own button uses, so the only thing that differs between the two
 * loops below is whether each generation after the first pays
 * `LlamaEngineQ8Resident.create()`'s own cost (a fresh `ResidentDevice`, a
 * fresh `device.upload()` for every persistent `matvecQ8` weight buffer) or
 * `reset()`'s (two field writes).
 *
 * Not reachable from the UI — a CDP-driven script calls this directly, the
 * same way issue #117's own prefill numbers were measured (this README's
 * "GPU-resident prefill" section) — so it lives outside `runGeneration`'s
 * own click handler entirely and never runs unless something calls it by
 * name. `policySelect.value`'s current UI selection is still used to build
 * each prompt (`buildStylePrompt`), the same style/policy the button itself
 * would use, so a driving script only has to set the text input, not
 * reimplement prompt construction.
 */
interface BenchGenDetail {
  createMs: number;
  generateMs: number;
  prefillMs: number;
  decodeMsTotal: number;
  decodeSteps: number;
  promptTokenCount: number;
}

interface BenchStrategyResult {
  totalMs: number;
  perGenMs: number[];
  perGenDetail: BenchGenDetail[];
}

/** `LlamaEngineQ8Resident.create()`'s own cost, timed on its own — separate
 * from `generate()`'s, so the two are never conflated (PR #126 review, item
 * 6: an earlier version of this file's own README write-up used `prefillMs`,
 * which structurally cannot include `create()`'s cost at all, as evidence
 * about `create()`'s cost — a category error this split makes impossible to
 * repeat). */
async function timedCreate(engineConfig: LlamaConfig, weights: LlamaWeightsQ8): Promise<{
  engine: LlamaEngineQ8Resident;
  device: Awaited<ReturnType<typeof createBrowserResidentDevice>>;
  createMs: number;
}> {
  const t0 = performance.now();
  const device = await createBrowserResidentDevice();
  const engine = await LlamaEngineQ8Resident.create(engineConfig, weights, device);
  const createMs = performance.now() - t0;
  return { engine, device, createMs };
}

/** One `ResidentDevice`, one `LlamaEngineQ8Resident`, `reset()` before every
 * generation after the first — `create()`'s cost is paid exactly once,
 * timed on its own and folded into generation 0's own `createMs`; every
 * later generation's `createMs` is `0` by construction (no `create()` call
 * happens). */
async function runResetStrategy(
  engineConfig: LlamaConfig,
  weights: LlamaWeightsQ8,
  promptTokenLists: number[][],
): Promise<BenchStrategyResult> {
  const perGenMs: number[] = [];
  const perGenDetail: BenchGenDetail[] = [];
  const start = performance.now();
  const device = await createBrowserResidentDevice();
  try {
    const createT0 = performance.now();
    const engine = await LlamaEngineQ8Resident.create(engineConfig, weights, device);
    const firstCreateMs = performance.now() - createT0;
    for (let i = 0; i < promptTokenLists.length; i += 1) {
      const createMs = i === 0 ? firstCreateMs : 0;
      const genT0 = performance.now();
      if (i > 0) engine.reset();
      // eslint-disable-next-line no-await-in-loop
      const { prefillMs, decodeMsTotal, decodeSteps } = await generate(engine, promptTokenLists[i]!, undefined);
      const generateMs = performance.now() - genT0;
      perGenMs.push(createMs + generateMs);
      perGenDetail.push({
        createMs,
        generateMs,
        prefillMs,
        decodeMsTotal,
        decodeSteps,
        promptTokenCount: promptTokenLists[i]!.length,
      });
    }
  } finally {
    device.destroy();
  }
  return { totalMs: performance.now() - start, perGenMs, perGenDetail };
}

/** `runGeneration`'s own resident branch, unchanged — a fresh
 * `ResidentDevice` and a fresh `LlamaEngineQ8Resident.create()` call for
 * every generation, `create()`'s own cost timed separately from
 * `generate()`'s every time. */
async function runRebuildStrategy(
  engineConfig: LlamaConfig,
  weights: LlamaWeightsQ8,
  promptTokenLists: number[][],
): Promise<BenchStrategyResult> {
  const perGenMs: number[] = [];
  const perGenDetail: BenchGenDetail[] = [];
  const start = performance.now();
  for (const promptTokens of promptTokenLists) {
    const t0 = performance.now();
    const { engine, device, createMs } = await timedCreate(engineConfig, weights);
    let detail: { prefillMs: number; decodeMsTotal: number; decodeSteps: number };
    try {
      // eslint-disable-next-line no-await-in-loop
      detail = await generate(engine, promptTokens, undefined);
    } finally {
      device.destroy();
    }
    const genMs = performance.now() - t0;
    perGenMs.push(genMs);
    perGenDetail.push({ createMs, generateMs: genMs - createMs, ...detail, promptTokenCount: promptTokens.length });
  }
  return { totalMs: performance.now() - start, perGenMs, perGenDetail };
}

/**
 * Issue #120's own completion bar, re-measured per PR #126 review (items
 * 5-7): the first version of this benchmark always ran the `reset()`
 * strategy before the `rebuild` strategy, in every trial — and this box's
 * own `create()` cost turned out to *rise* within a page session (measured:
 * generation 1's ~2.1-2.3s vs. each rebuild generation's own ~4.6-4.9s,
 * `create()`-only), so running `reset()` first meant the rebuild strategy
 * absorbed three of those later, larger `create()` calls while `reset()`
 * only ever paid the earliest, cheapest one — a session-position penalty
 * riding on the rebuild side of every trial, not a `reset()` vs. rebuild
 * difference at all. `order` lets a driving script alternate which strategy
 * goes first, trial by trial, so that penalty lands on both sides equally
 * across a run rather than always favouring `reset()`.
 *
 * Not reachable from the UI — a CDP-driven script calls this directly, the
 * same way issue #117's own prefill numbers were measured (this README's
 * "GPU-resident prefill" section). `policySelect.value`'s current UI
 * selection is still used to build each prompt (`buildStylePrompt`), the
 * same style/policy the button itself would use.
 */
async function __resetBenchmark(
  promptTexts: string[],
  order: "reset-first" | "rebuild-first" = "reset-first",
): Promise<{
  order: "reset-first" | "rebuild-first";
  reset: BenchStrategyResult;
  rebuild: BenchStrategyResult;
}> {
  if (!loaded) throw new Error("重みが未ロードです");
  const { engineConfig, weights, tokenizer } = loaded;
  const promptTokenLists = promptTexts.map((text) => tokenizer.encode(buildStylePrompt(policySelect.value, text)));

  const runReset = () => runResetStrategy(engineConfig, weights, promptTokenLists);
  const runRebuild = () => runRebuildStrategy(engineConfig, weights, promptTokenLists);

  let reset: BenchStrategyResult;
  let rebuild: BenchStrategyResult;
  if (order === "reset-first") {
    reset = await runReset();
    rebuild = await runRebuild();
  } else {
    rebuild = await runRebuild();
    reset = await runReset();
  }

  return { order, reset, rebuild };
}
(window as unknown as { __resetBenchmark: typeof __resetBenchmark }).__resetBenchmark = __resetBenchmark;

// ---------------------------------------------------------------------------
// Issue #111: measuring the fused decode kernels' effect on forward()'s fixed cost
// ---------------------------------------------------------------------------

/** One prompt length's own prefill and per-step decode timing — `__decodeFixedCostBenchmark`'s own return shape, below. */
interface DecodeFixedCostResult {
  promptLength: number;
  prefillMs: number;
  decodeStepMs: number[];
}

/**
 * Issue #111's own completion bar: "前後のforward固定費とtok/sを実測" —
 * measures `LlamaEngineQ8Resident.forward()`'s prefill cost at two very
 * different prompt lengths, and decode's own per-step cost, directly
 * against real weights and a real device, one prompt length at a time.
 * (PR #127 review, item 5: an earlier version of this doc's own framing
 * — "prefill barely changes between 76 and 365 tokens, which only makes
 * sense if fusing *decode*'s dispatches should help" — conflated the two:
 * prefill runs through `matmul`, never `matvecQ8`, so this PR's own fused
 * kernels do not touch it, and this benchmark's own measured results
 * (README, "Fused decode kernels (issue #111)") show exactly that —
 * prefill's own near-flat cost at 76 vs. 365 tokens is real and
 * reproducible, but it is unrelated to what this benchmark's decode
 * column measures.) This benchmark reports the comparison that *does*
 * apply — decode's per-token cost before vs. after fusing four decode
 * dispatches into one (`ops/matvec/wgsl/q8_ffn.wgsl`) or two into one
 * (`q8_residual.wgsl`) — plus prefill's own timing alongside it, so both
 * numbers come from the same run rather than being assumed to move
 * together.
 *
 * Prompt *content* does not matter for a timing measurement — only its
 * length does (`forward()` reads exactly `hiddenSize` embedding-table
 * bytes per token, regardless of which row) — so this builds synthetic
 * token id sequences (`i % vocabSize`, always in range) instead of relying
 * on `tokenizer.encode` to land on an exact requested length, which a real
 * tokenizer over real text cannot promise.
 *
 * One `LlamaEngineQ8Resident`/`ResidentDevice` for every prompt length in
 * `promptLengths`, `reset()` (issue #120) between them rather than a fresh
 * `create()` — the same "don't let `create()`'s own ~1.4 GiB weight
 * re-upload leak into a measurement that is not about `create()`" reasoning
 * `runResetStrategy` above already established, load-bearing here for the
 * same reason: `createMs` is reported once, separately, so a slow first
 * `create()` call cannot be mistaken for a slow `forward()`.
 *
 * Not reachable from the UI — a CDP-driven script calls this directly, the
 * same pattern `__resetBenchmark` above already establishes.
 *
 * (PR #127 review, item 6: an earlier version of this PR put this doc
 * comment directly above `DecodeFixedCostResult` instead of above this
 * function — the interface sitting between them meant the doc silently
 * attached to the wrong declaration. Same fix as `packQ8`'s doc in
 * `ops/matvec/reference.ts`: moved to directly precede what it documents.)
 */
async function __decodeFixedCostBenchmark(
  promptLengths: number[],
  decodeSteps: number,
): Promise<{ createMs: number; results: DecodeFixedCostResult[] }> {
  if (!loaded) throw new Error("重みが未ロードです");
  const { engineConfig, weights } = loaded;
  const device = await createBrowserResidentDevice();
  try {
    const createT0 = performance.now();
    const engine = await LlamaEngineQ8Resident.create(engineConfig, weights, device);
    const createMs = performance.now() - createT0;

    const results: DecodeFixedCostResult[] = [];
    for (let i = 0; i < promptLengths.length; i += 1) {
      const promptLength = promptLengths[i]!;
      if (i > 0) engine.reset();
      const tokens = Array.from({ length: promptLength }, (_, j) => j % engineConfig.vocabSize);

      const prefillT0 = performance.now();
      const prefillLogits = await engine.forward(tokens);
      const prefillMs = performance.now() - prefillT0;

      let logits = prefillLogits[prefillLogits.length - 1]!;
      const decodeStepMs: number[] = [];
      for (let s = 0; s < decodeSteps; s += 1) {
        const next = argmax(logits);
        const stepT0 = performance.now();
        // eslint-disable-next-line no-await-in-loop
        const [nextLogits] = await engine.forward([next]);
        decodeStepMs.push(performance.now() - stepT0);
        logits = nextLogits!;
      }
      results.push({ promptLength, prefillMs, decodeStepMs });
    }
    return { createMs, results };
  } finally {
    device.destroy();
  }
}
(window as unknown as { __decodeFixedCostBenchmark: typeof __decodeFixedCostBenchmark }).__decodeFixedCostBenchmark =
  __decodeFixedCostBenchmark;

// ---------------------------------------------------------------------------
// Issue #131: where prefill's ~1.2s fixed cost actually goes
// ---------------------------------------------------------------------------

/** One prompt length's own prefill profile plus one decode step's, for contrast — `__prefillProfileBenchmark`'s own return shape, below. */
interface PrefillProfileResult {
  promptLength: number;
  prefill: ForwardProfile;
  /** One `forward([token])` call immediately after this prompt length's prefill — issue #131's own scope asks for "1トークンdecode" alongside prefill, as a contrast case (`ForwardProfile`'s own doc: decode's `packEntries`/`bindGroupCalls` should be near-zero, unlike prefill's). */
  decodeStep: ForwardProfile;
}

/**
 * Issue #131's own measurement entry point: for each prompt length, runs one
 * real `forward()` prefill and one real decode step through the **unmodified**
 * production dispatch chain (`ForwardProfile`'s own doc — passing a profile
 * changes no arithmetic, only which `performance.now()`/timestamp-query calls
 * get made around it), against real weights and a real device, and returns
 * the full breakdown for a driving script to tabulate.
 *
 * Same shape as `__decodeFixedCostBenchmark` immediately above (one `create()`,
 * `reset()` between prompt lengths so a slow `create()` cannot be mistaken
 * for a slow `forward()`, synthetic-but-in-range token ids so prompt length
 * is exact) — not reachable from the UI, a CDP-driven script calls this
 * directly.
 */
async function __prefillProfileBenchmark(
  promptLengths: number[],
): Promise<{ createMs: number; timestampsSupported: boolean; results: PrefillProfileResult[] }> {
  if (!loaded) throw new Error("重みが未ロードです");
  const { engineConfig, weights } = loaded;
  const device = await createBrowserResidentDevice();
  try {
    const createT0 = performance.now();
    const engine = await LlamaEngineQ8Resident.create(engineConfig, weights, device);
    const createMs = performance.now() - createT0;

    const results: PrefillProfileResult[] = [];
    for (let i = 0; i < promptLengths.length; i += 1) {
      const promptLength = promptLengths[i]!;
      if (i > 0) engine.reset();
      const tokens = Array.from({ length: promptLength }, (_, j) => j % engineConfig.vocabSize);

      const prefill = createForwardProfile();
      // eslint-disable-next-line no-await-in-loop
      const prefillLogits = await engine.forward(tokens, prefill);

      const decodeStep = createForwardProfile();
      // eslint-disable-next-line no-await-in-loop
      const next = argmax(prefillLogits[prefillLogits.length - 1]!);
      // eslint-disable-next-line no-await-in-loop
      await engine.forward([next], decodeStep);

      results.push({ promptLength, prefill, decodeStep });
    }
    return { createMs, timestampsSupported: device.timestampsSupported, results };
  } finally {
    device.destroy();
  }
}
(window as unknown as { __prefillProfileBenchmark: typeof __prefillProfileBenchmark }).__prefillProfileBenchmark =
  __prefillProfileBenchmark;
