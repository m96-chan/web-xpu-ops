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
import { LlamaEngineQ8 } from "../../../llm/engine-q8.js";
import { sampleNext, type Constraint } from "../../../llm/sampler.js";
import { SentencePieceTokenizer, type TokenizerVocab } from "../../../llm/tokenizer.js";
import { loadWeightsQ8FromUrl } from "../../../llm/browser-weights.js";
import type { LlamaConfig } from "../../../llm/config.js";
import type { LlamaWeightsQ8 } from "../../../llm/weights-q8.js";
import { createBrowserRunner } from "./browser-runtime.js";

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
/** `assets/llm.js`'s own `MAX_TOKENS`. */
const MAX_DECODE_STEPS = 200;

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
  runGeneration().catch((err: unknown) => {
    stats.textContent = `生成失敗: ${err instanceof Error ? err.message : String(err)}`;
  });
});

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
  stats.textContent = "生成中…";

  // A fresh engine per generation: `LlamaEngineQ8` has no KV-cache reset, and
  // packing every projection's `matvecQ8` wire format from the already-fetched
  // `weights` is the correctness-preserving way to start a clean generation
  // (see `engine-q8.ts`'s own doc — the constructor only reads `weights`, it
  // does not mutate it, so building a second engine from the same loaded
  // weights is safe). Its cost is reported separately from tok/s below, since
  // it is a one-time setup cost, not a decode-time one.
  const buildStart = performance.now();
  const engine = new LlamaEngineQ8(engineConfig, weights, runner.run);
  const buildMs = performance.now() - buildStart;

  const prefillStart = performance.now();
  const prefillLogits = await engine.forward(promptTokens);
  const prefillMs = performance.now() - prefillStart;

  let logits = prefillLogits[prefillLogits.length - 1]!;
  const tokens: number[] = [];
  let decodeMsTotal = 0;
  let decodeSteps = 0;

  for (let step = 0; step < MAX_DECODE_STEPS; step += 1) {
    const next = sampleNext(logits, tokens, { mode: "greedy" }, constraint);
    tokens.push(next);
    if (next === vocab.eosId) break;

    const decodeStart = performance.now();
    const [nextLogits] = await engine.forward([next]);
    decodeMsTotal += performance.now() - decodeStart;
    decodeSteps += 1;
    logits = nextLogits!;

    output.textContent = tokenizer.decode(tokens);
  }
  output.textContent = tokenizer.decode(tokens);

  const prefillTokPerSec = promptTokens.length / (prefillMs / 1000);
  const decodeTokPerSec = decodeSteps > 0 ? decodeSteps / (decodeMsTotal / 1000) : 0;

  stats.textContent =
    `engine construction: ${buildMs.toFixed(0)}ms | ` +
    `prefill: ${promptTokens.length} tok in ${prefillMs.toFixed(0)}ms (${prefillTokPerSec.toFixed(2)} tok/s) | ` +
    `decode: ${decodeSteps} tok in ${decodeMsTotal.toFixed(0)}ms (${decodeTokPerSec.toFixed(2)} tok/s)` +
    (tokens[tokens.length - 1] === vocab.eosId ? " | stopped at </s>" : " | stopped at MAX_DECODE_STEPS");

  genBtn.disabled = false;
}
