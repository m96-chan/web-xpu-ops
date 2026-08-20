# Changelog

Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/); versions
follow [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

Entries record **why** a change was needed. What changed is in the diff.

## [Unreleased]

### Added

- `ops/gqa`'s `scores`/`context` kernels gain an `sEff` parameter (issue
  #117): the number of key/value positions actually resident, bounding the
  softmax scan without changing `S`, the KV cache's own address stride.
  Uniform-passed, defaults to `S` — every existing caller (including
  `ops/gqa`'s own reference and `llm/kernels.ts#runGqa`) is unaffected
  unless it opts in. `LlamaEngineQ8Resident`'s decode step now passes
  `sEff = position + 1` instead of always scanning `maxSeqLen` — the fix
  for the "gqaの全長走査" roofline gap issue #116 measured (672 MiB/token on
  Sarashina2.2-1B's shape) and explicitly left open, since removing it
  needed a parameter `ops/gqa` did not have yet.

  `context.wgsl`'s `v` read is unconditional (masked columns arrive as a
  plain `0` from `scores.wgsl`'s softmax, not a skip), so an `sEff` that
  is not honored there is numerically observable — a dedicated test poisons
  `v` past `sEff` with `+Infinity` and checks for the resulting `NaN`.
  `scores.wgsl`'s own bound has no such numeric signature under the
  `causal = true` contract `sEff < S` requires (every excluded position is
  already causally masked, by construction), so it is instead proven by
  `seffEquivalence()` (`ops/gqa/wgsl-seff.test.ts`): running with `sEff = n` must equal running with `S`
  itself shrunk to `n`. See `ops/gqa/reference.ts`'s `sEff` doc for the full
  safety contract (`sEff >= min(S, L + queryOffset)` whenever `causal` is
  true; rejected outright otherwise).

- `LlamaEngineQ8Resident`'s prefill is GPU-resident now too (issue #117),
  replacing the delegation to `LlamaEngineQ8` issue #110 left in place
  ("プリフィルは現行方式のまま"). Every prompt token's pass through every
  layer is encoded into one `queue.submit`, on the `matmul` path (not
  `matvecQ8` per token — that would multiply prefill's weight traffic by
  the prompt length). A new `ops/permute/wgsl/kernel.wgsl` kernel does `ops/gqa`'s
  required token-major/head-major reshape on the GPU, replacing the CPU
  round trip `LlamaEngineQ8`'s non-resident reshape (`llm/reshape.ts`)
  would otherwise force back in. KV writes go straight into the persistent,
  `maxSeqLen`-strided cache via one contiguous `copyBufferToBuffer` per
  head, so `runDecodeStep` continues seamlessly with no CPU round trip
  between prefill and the first decode step. Only the final prompt
  position is projected through `lm_head` — **`forward()`'s prefill return
  shape changed**: `[finalPositionLogits]`, one element, not one per prompt
  token, matching what every real caller (`llm/engine.ts#greedyGenerate`,
  `examples/llm-demo`) already read (`prefillLogits[prefillLogits.length - 1]`).
  See the PR for prefill tok/s at three prompt lengths, before/after, and the
  updated roofline table.

- Persistent IndexedDB weight cache for `llm/browser-weights.ts#loadWeightsQ8FromUrl`
  (issue #121, parent #96): `technologies-moe/alibi-ai`'s browser integration was
  re-fetching the full ~1.41 GiB int8 checkpoint over HTTP on every visit — no
  caching layer existed between `fetch` and the page. Caching is on by default and
  falls back transparently (no feature loss, only a persistence difference) when
  IndexedDB is unavailable (Node, a browser with none at all) or `indexedDB.open()`
  itself fails (Safari private mode's own failure shape) or when
  `navigator.storage.estimate()` reports too little free space for the checkpoint —
  pass `{ enabled: false }` as `loadWeightsQ8FromUrl`'s new fifth argument to opt out
  entirely.

  Versioning is a SHA-256 hash of `manifest.json`'s raw bytes (`llm/weight-cache.ts#sha256Hex`,
  via WebCrypto — available identically in Node and every browser this package
  targets), embedded in every cache key: a re-converted checkpoint under the same URL
  is detected and re-downloaded automatically, and the previous version's chunks are
  swept from the store once the new one is fully written (`sweepOrphanedChunks`,
  which doubles as a general orphan-chunk cleanup for a write interrupted mid-way, via
  one `ChunkStore.list()` pass rather than exact bookkeeping of what to delete). The
  manifest itself (tens of KB) is still fetched over the network on every load — it is
  the only way to learn whether a cached checkpoint is still current — but the three
  large binaries it may cache (`weights.codes.bin` / `.scales.bin` / `.norms.bin`,
  1.41 GiB combined) are not, on a cache hit; see the README's real-hardware section
  for the DevTools Network-panel proof.

  Each of the three cached files is split into 96 MiB chunks
  (`weight-cache.ts#DEFAULT_CHUNK_SIZE_BYTES`, the midpoint of this issue's own
  64–128 MiB range) before being written — a single ~1.4 GiB `IndexedDB` value is what
  this issue's own spec asks to avoid (browser blob-size implementation differences,
  and no way to show incremental write progress) — written and read one chunk at a
  time (`llm/weight-cache.ts#iterateChunks`, a generator) rather than materializing
  every chunk of a file up front, so a ~1.4 GiB read or write peaks at roughly the
  file's own size plus one chunk, not roughly double it. Storage is behind a small
  injected interface, `llm/chunk-store.ts#ChunkStore` (`get`/`put`/`delete`/`list`, all
  `ArrayBuffer`-valued), so every piece of cache logic — chunk splitting, manifest-hash
  comparison, stale-version eviction, the quota/fallback decision — is unit-tested in
  Node against `InMemoryChunkStore`, with no browser and no IndexedDB; only the real
  backend, `llm/idb-chunk-store.ts#createIndexedDbChunkStore` (hand-rolled minimal
  IndexedDB types, since neither this repository's `tsconfig.json` — deliberately
  `"DOM"`-lib-free — nor `@types/node` has any), needs the real-Chrome verification
  this issue's PR carries.

  The quota check (`decideCacheStrategy`) gates only the *write* path, never a read —
  a cache read frees no space and needs none, and gating it too would have made a
  device whose reported `usage` (which includes this very cache once one write
  succeeds) leaves too little headroom refetch the entire checkpoint on every visit
  from then on, forever, even though the cache it just refused to read was perfectly
  valid. A read failure (an IndexedDB `get()` rejecting — storage pressure closing the
  connection, a stale database from an older store-name layout) is caught and treated
  as a cache miss, the same fallback every other failure mode here already gets, rather
  than failing the whole load. `IndexedDbChunkStore#put`/`#delete` resolve on their
  *transaction*'s `oncomplete` (reject on `onabort`/`onerror`), not on the individual
  request's own `onsuccess` — a large write's request can report success and still have
  its transaction abort during commit (`QuotaExceededError`'s usual shape), and
  resolving on the request alone let a caller believe a chunk was durably written when
  it was not. The orphan sweep (`sweepOrphanedChunks`) now matches exact
  `namespace`/`manifestHash`/`file`/index chunk keys against a version record's own
  `chunkCount`, rather than a `namespace`/`manifestHash`/`file` *prefix* — re-writing
  the same file under the same hash with a different chunk size (a changed
  `DEFAULT_CHUNK_SIZE_BYTES`, say) used to leave every chunk beyond the new, smaller
  `chunkCount` behind forever, since they shared the kept prefix — and it now also runs
  (best-effort) after a cache *hit*, not only at the end of a successful cache-miss
  write, so chunks orphaned by an interrupted write are reclaimed the next time this
  cache is read rather than sitting unclaimed for as long as every later visit keeps
  hitting.

  Known limitation, tracked as [#124](https://github.com/m96-chan/web-xpu-ops/issues/124):
  versioning one `manifestHash` across all three binary files means a hypothetical
  future producer of this checkpoint format that keeps `manifest.json` byte-identical
  while republishing one of the binaries (this repository's own `convert_weights.py`
  does not — it regenerates `manifest.json`, and so the hash, on every run) could have
  an interrupted overwrite leave old- and new-version chunks mixed under one hash that
  this cache's own length check cannot tell apart from a correctly written file. Also
  tracked, lower priority since it cannot corrupt data either way (only causes
  redundant re-downloads and a non-deterministic "winner"):
  [#125](https://github.com/m96-chan/web-xpu-ops/issues/125), concurrent tabs open
  across a deploy window each detecting the other's freshly-written version as stale.

- `LlamaEngineQ8Resident` (`llm/engine-q8-resident.ts`, issue #110): a
  GPU-resident decode path for `LlamaEngineQ8` — one `queue.submit` and one
  logits-only readback per generated token, instead of the ~155 GPU↔CPU
  round trips (one per kernel dispatch, per layer) `LlamaEngineQ8.forward`
  pays today. Real-model measurement (`examples/llm-demo`, RTX 5090,
  Sarashina2.2-1B-alibi-v1): decode went from **0.75–0.80 tok/s to
  137–162 tok/s** (~175–200x), with byte-for-byte identical generated text
  to `LlamaEngineQ8` on both prompts tested — see the PR for the full
  before/after table, roofline decomposition and screenshots.

  `harness/resident.ts` (`ResidentDevice`) is the new low-level layer this
  runs on: build every buffer, pipeline and bind group once at construction,
  then record many dispatches (plus GPU-to-GPU `copyBufferToBuffer` for the
  KV-cache writes) into one command encoder per token. `harness/wgsl.ts`'s
  existing `Runner`/`createRunner()` is unchanged and still what every op's
  own correctness test uses (rule 8: optimize only after a reference-correct
  baseline exists) — `ResidentDevice` exists solely for this decode path,
  where the WGSL, binding order and uniform layout are already known correct
  from `llm/kernels.ts`. `harness/resident.ts#runnerFromResident` adapts one
  `ResidentDevice` back into a `Runner["run"]` — at the time this landed,
  `LlamaEngineQ8Resident` used it to delegate prefill to `LlamaEngineQ8`
  unchanged (issue #110's scope was decode only) instead of constructing a
  second native device: an earlier version took a genuinely separate
  `Runner`, and running both engines' dispatches on two devices in one
  process reproducibly crashed this repository's Node/Dawn binding (the
  same failure family issue #38/#49/#107 already document). Issue #117
  later made prefill resident too and removed that delegation entirely —
  see this file's #117 entry above; `runnerFromResident` itself remains,
  unused by this class since.

  `examples/llm-demo` gained a "GPU常駐デコード" toggle so the same page can
  run either engine against the same loaded weights for a direct comparison.

  PR #116 review: `LlamaEngineQ8Resident.forward` now throws once a decode
  step would exceed `maxSeqLen` (`LlamaEngineQ8`/`LlamaEngine`'s own decode
  paths already do), instead of a KV-cache GPU-to-GPU copy silently landing
  past the buffer it was meant for and the call resolving with the previous
  step's stale logits.

- `examples/llm-demo/`: a browser demo that loads Sarashina2.2-1B-alibi-v1
  (int8, `convert_weights.py`'s output) over HTTP and runs `LlamaEngineQ8` on
  a real WebGPU device — issue #106, and the first time this repository's
  `llm/` engine has generated anything in a browser rather than under Node.
  It closes the live-generation / `llama.cpp`-comparison / tok/s gate PR
  #108 (issue #105) moved here: #108 could not complete a live GPU dispatch
  on the machine it was built on (a Node+Dawn binding fragility triggered by
  real-model-scale CPU-bound work immediately before a dispatch — issue
  #107), and a browser's separate GPU process does not share that condition.
  `llm/index.ts` also now re-exports `llm/tokenizer.ts` (#101), `llm/sampler.ts`
  and `llm/constraints/line-format.ts` (#102) — a known gap since both landed,
  called out in #106's own text, and caught by the new `llm/index.test.ts`
  before being fixed.

  `llm/browser-weights.ts#loadWeightsQ8FromUrl` is a `fetch`-based sibling of
  `real-model-weights.ts#loadConvertedWeightsQ8`, sharing the same
  manifest-parsing core (`weights-q8-io.ts#buildLlamaWeightsQ8`) per PR #108's
  own note that a browser loader "should be a small adapter over the same
  parsing logic, not a reimplementation" — it lives in `llm/` (not
  `examples/`) and is exported from `llm/index.ts` because `fetch`/`Response`
  are standard Web APIs this package already assumes elsewhere (`llm/tokenizer.ts`'s
  `TextEncoder`/`TextDecoder`), unlike the `node:fs` the disk-based loader
  needs, and because that keeps it testable with a mocked `fetch` and no
  browser.

  The demo's own WebGPU plumbing (`examples/llm-demo/src/browser-runtime.ts`)
  is a `navigator.gpu` port of `harness/wgsl.ts#createRunner` — necessarily a
  separate implementation, not a shared import, since `harness/wgsl.ts`
  imports the `webgpu` package (a Node-native Dawn binding with no browser
  build) at module scope. `llm/kernels.ts` itself needed **zero** changes to
  run in a browser: `examples/llm-demo/build.mjs` (esbuild — the first
  bundler this repository has needed, justified in that file's own module
  doc) redirects `kernels.ts`'s `import { kernel, params } from
  "../harness/index.js"` to `browser-runtime.ts`'s matching-signature
  implementations at bundle time, and inlines all ten `.wgsl` kernel sources
  `kernels.ts` reads via `node:fs` under Node into plain JS strings via
  esbuild's `text` loader — `llm/kernels.browser-parity.test.ts` (text-based,
  no `.wgsl` import, runs under `npm test`) guards the two kernel tables
  against drifting apart. `examples/llm-demo/server.mjs` is a Node-standard-
  library-only static file server (repository root + a `/weights/` mount
  pointing at a converted checkpoint directory outside this repository,
  e.g. `technologies-moe/alibi-ai`'s `third_party/webgpu-weights/`), always
  setting `Content-Length` (no `Range` support needed) since the demo's
  progress bar reads it.

- `llm/tools/convert_weights.py`, `llm/weights-q8.ts`, `llm/engine-q8.ts`
  (`LlamaEngineQ8`): a weight converter and an int8 (W8A32) engine path,
  closing #96's loop from "an engine that runs a tiny fixture" to "an engine
  that loads and generates from a real checkpoint" — issue #105.
  `convert_weights.py` streams a HF safetensors checkpoint (bf16) one tensor
  at a time into per-row absmax int8 codes + f32 scale (`wq`/`wk` permuted
  per `permuteRopeChannels` before quantizing) plus f32 norms; converting
  Sarashina2.2-1B-alibi-v1 (2.68 GiB bf16) produced a 1.41 GiB (1345 MiB)
  int8 checkpoint in ~8s. `LlamaEngineQ8` mirrors `LlamaEngine`'s forward pass
  exactly but keeps only the packed `matvecQ8` wire format resident per
  projection (not also an unpacked or dequantized copy — packing does not
  change a weight's resident size, so keeping both would roughly double it):
  decode dispatches `matvecQ8` directly; prefill dequantizes the needed
  projection to a transient f32 matrix and runs `matmul`, once per generation
  (the prompt), not once per token — issue #105's own stated scope
  ("プリフィルは当面「行スケールdequantしてf32 matmul」でもよい"). The
  embedding table is quantized like every other weight but is gathered and
  dequantized **on the CPU**, one row per requested token, rather than
  dispatching a GPU gather against a fully-dequantized 733 MiB table.
  Verified on a real GPU against an int8-quantization-aware `transformers`
  reference (`llm/tools/gen_fixture_q8.py`, which substitutes dequantized
  weights into the tiny fixture model and re-runs its own forward pass, so
  the quantization error is baked into the reference the same way it is
  baked into `LlamaEngineQ8`'s output — worst observed diff abs 1.49e-7, rel
  7.22e-4). The real Sarashina2.2-1B-alibi-v1 checkpoint converts, loads, and
  constructs a `LlamaEngineQ8` successfully; **live GPU generation from it
  does not run yet** — a Node+Dawn binding fragility triggered by
  real-model-scale CPU-bound work before a dispatch (see #107 for the
  isolated repro), and, found in review, the real vocab's lmHead projection
  breaking WebGPU's default dispatch and buffer limits (#112). Neither is a
  numerics problem. Live generation, `llama.cpp` comparison, and tok/s move
  to #106 (browser demo), where #107's condition does not exist — #112
  applies there too and blocks it.
- `matvecQ8` gets a second call site: `llm/kernels.ts#runMatVecQ8`, the GPU
  dispatch wrapper `LlamaEngineQ8`'s decode path uses — issue #97's kernel
  finally consumed by the engine it was built for.
- `llm/weights-q8.ts`, `llm/weights-q8-io.ts`, `llm/reshape.ts#concatRowsInt8`:
  the int8 counterparts of `weights.ts` (`QuantizedLinear`,
  `LlamaWeightsQ8`, `assertWeightShapesQ8`) and `reshape.ts#concatRows`,
  plus a manifest-parsing core (`buildLlamaWeightsQ8`) shared by
  `fixture-q8.ts` (the tiny int8 fixture) and `real-model-weights.ts` (a
  real converted checkpoint) — the two write the identical manifest shape
  (`convert_weights.py`'s own module doc) on purpose, so one parser serves
  both rather than two that could drift apart. `cloneQuantizedLinear` copies
  the embedding table out of the loader's shared buffer at construction time
  rather than keeping a view into it — measured to matter: a `LlamaEngineQ8`
  built from the real checkpoint fell from ~2.96 GiB to ~1.56 GiB resident
  (RSS) once the caller also drops its own reference to the loaded weights
  and a GC runs, because a `TypedArray` view keeps its *entire* backing
  buffer alive, not just the slice it reads.
- `llm/tools/quant_common.py`: the per-row absmax int8 quantizer
  `convert_weights.py` and `gen_fixture_q8.py` both call, sharing one
  implementation rather than risking two silently-different ones (rule 7).
  Uses `floor(x) + (frac >= 0.5)`, not `np.round` and not `np.floor(x + 0.5)`
  — `Math.round` (what `ops/quantize/reference.ts#quantize` uses) rounds ties
  toward `+Infinity`; `np.round` rounds ties to even and disagrees at every
  exact `.5` boundary (`Math.round(62.5) === 63`, `np.round(62.5) === 62`);
  `np.floor(x + 0.5)` disagrees in the narrow band just below half-integers,
  where the addition itself rounds up before `floor` runs
  (`Math.round(0.49999999999999994) === 0`, `floor(… + 0.5) === 1`). Verified
  rather than assumed by `llm/quantize-parity.test.ts`, which spawns
  `quant_common.py --selftest` and diffs its output against `quantize()`
  directly, including rows at both boundaries.
- `llm/tokenizer.ts`: a SentencePiece **unigram** tokenizer (Viterbi encode,
  matching decode) for the Sarashina2.2-1B-Instruct model the LLM engine
  (#98) targets, plus `llm/tools/export_tokenizer.py` (parses
  `tokenizer.model`'s protobuf into a JSON vocab) and
  `llm/tools/gen_fixtures.py` (bakes real-tokenizer encode/decode fixtures
  from Python `sentencepiece`, never `transformers` — see below). No wasm,
  no vendored binary: this is a from-scratch TS reimplementation of the
  algorithm, CPU-only, and it never touches a GPU device.

  Nothing about this model's tokenizer config was assumed. Read directly out
  of the real `tokenizer.model`/`tokenizer_config.json`:

  - The normalizer is literally `"identity"` — **not** NFKC. Composed and
    decomposed forms of the same glyph (が vs か + combining voiced sound
    mark) tokenize to different ids; a normalizing tokenizer would not do
    that. Only the literal ASCII space gets escaped to `▁`; tabs, newlines,
    NBSP and the full-width space do not.
  - `add_dummy_prefix` is `false`: no implicit leading `▁`.
  - Control/special tokens (`<s>`, `</s>`, `<|system|>`, ...) are **not**
    matched by SentencePiece's own Viterbi search against raw text — verified
    directly (`sp.encode("<|system|>")` decomposes it byte-by-byte). They are
    recognized through a separate literal-substring pre-tokenization pass
    over `tokenizer_config.json`'s `added_tokens_decoder`, which is what
    lets the engine's chat-template fragments (`<|system|>`, `<|user|>`,
    `</s>`, ...) round-trip as single ids.
  - `transformers` 5.3.0, on this environment, silently converts this
    UNIGRAM model into an *approximate BPE* tokenizer when loaded with
    `AutoTokenizer.from_pretrained(..., use_fast=False)` and no
    `tokenizer.json` is present (e.g. it segments `"hello"` as
    `['he','ll','o']`, not the single correct unigram piece `'hello'`).
    Fixtures are therefore generated from Python `sentencepiece`'s
    `SentencePieceProcessor` directly, never from `transformers`.
  - Byte fallback is one Viterbi edge per uncovered *codepoint*, scored
    `min_score() - 10.0` (SentencePiece's own `kUnkPenalty`/`min_score()`
    formula from `unigram_model.cc`) — not per byte, and not an arbitrarily
    large sentinel. An earlier version used `-1e6` per byte, which is locally
    harmless (a byte edge never competes with a real piece at the same
    position) but not globally harmless: accumulated over a few dozen
    uncovered codepoints in one string it pushed the running score to a
    magnitude where float32 could no longer distinguish the much smaller
    (~1-2 point) gaps between competing *real* segmentations later in the
    same string, silently picking the wrong one. Caught by a fixture built
    from a Hiragana-block sweep containing unassigned codepoints; kept as a
    permanent regression case (`byte_fallback_dense_sweep`).
  - The Viterbi DP accumulates scores in float32, matching SentencePiece's
    own C++ `Lattice`, and rounds every candidate to float32 *before*
    comparing it against an already-stored (also float32) score — comparing
    an un-rounded float64 sum against an already-rounded value can flip an
    exact tie either way depending on which side happened to round which
    way. This is not cosmetic: for a run of one repeated character, more
    than one tiling by the same multiset of piece lengths can be an exact
    tie at the level of real numbers, and only float32 rounding — applied
    consistently on both sides of the comparison — reproduces which one the
    real tokenizer picks.

  80+ fixture cases (Japanese/English/mixed, code fragments, emoji including
  ZWJ/flag/skin-tone sequences, whitespace and newline patterns, NFC/NFD
  probes, chat-template-shaped special-token boundaries, and the
  byte-fallback stress case above) all match the real tokenizer exactly on
  both encode and decode. Vocab JSON: 102400 entries, 3.48 MiB raw / 1.33 MiB
  gzipped — not compressed further; see the PR for why.
- `llm/sampler.ts`: token sampling with `greedy` and `temperature` + `top-p`
  modes, plus a token-level `Constraint` interface (`nextAllowed(prefixTokens)
  -> allowed token ids, or null`) applied to the logits before either mode
  runs. Repetition penalty is **deliberately not implemented** — the
  consuming project measured it degrading Japanese output rather than
  de-looping it (technologies-moe/alibi-ai#3); see `sampler.ts`'s module
  comment for the reasoning. A caller that wants repetition control should
  express it as a `Constraint` instead.

- `llm/constraints/line-format.ts`: a `Constraint` implementation for a fixed
  line shape — literal text, an enum choice, more literal text, free text
  (forbidden characters, max length), then EOS. Built for a small,
  hand-written schema like `policy: <enum>\ntopic: <short text>`, where a
  full GBNF/grammar engine would be overkill. Tokenizer-agnostic: it takes an
  injected `TokenCodec` (`encode` / `idToToken` / `vocabSize`) rather than
  depending on any one tokenizer, since issue #101's tokenizer is a separate,
  not-yet-merged branch. Enum choices are matched by tokenizing each
  candidate once and walking a token-id trie, so they must be
  **token-prefix-free** (no choice's tokenization may be a strict prefix of
  another's) — an ambiguous spec is rejected at construction rather than
  silently misclassified at generation time.

  Neither module is wired into `llm/engine.ts` yet — that integration is
  left to a follow-up issue, matching #102's stated scope.
- `llm/`: a config-driven llama-architecture inference engine built by
  composing existing ops (`gather` → N × [`rmsnorm` → fused QKV projection →
  `rope` → `gqa` + a pre-allocated f32 KV cache → O projection → residual →
  `rmsnorm` → fused gate/up projection → SiLU-gated MLP → residual] →
  `rmsnorm` → `lm_head`), rather than a new op — issue #98. Prefill uses
  `matmul`, decode uses `matvec`; Q/K/V and gate/up are fused into single
  projections purely to keep this repository's `webgpu` (Dawn) binding within
  the dispatch count it can sustain in one device's lifetime on some
  machines, not for FLOPs. Correctness is defined by a `transformers`-generated
  fixture (`llm/tools/gen_fixture.py`, committed at `llm/fixtures/tiny.*`,
  436 KiB): an 8-token prefill and a 4-step greedy decode on a tiny random
  model, checked on a real GPU to a measured tolerance (worst observed
  absolute diff 1.49e-7, relative 3.4e-4) and exact greedy-token equality.
  `SARASHINA_2_2_1B_CONFIG` documents the real target model's dimensions;
  running it needs a weight converter and tokenizer, both later issues under
  #96. Needed because wllama's WASM CPU path is too slow for in-browser
  inference (technologies-moe/alibi-ai#3: 22.6s few-shot prefill) and WebLLM's
  4-bit-only quantization collapses on the target Japanese model.
- `matvecQ8`, a W8A32 GEMV in `ops/matvec`: the weight held as int8 instead of
  f32, packed four codes per `u32` word. Decode-time GEMV is bandwidth-bound,
  and an int8 weight is a quarter the traffic of f32 for the same values — the
  packed layout halves that again versus one code per lane, since the whole
  point of quantizing a weight nobody re-quantizes at inference is to spend
  the packing cost once rather than never taking it (#97). `packQ8` packs
  `quantize`'s existing per-row absmax codes into the layout `matvecQ8` reads,
  so the two compose instead of `matvecQ8` inventing its own quantization.

### Fixed

- `llm/kernels.ts#runMatVec`/`runMatVecQ8` now split a dispatch whose row
  count `M` exceeds WebGPU's `maxComputeWorkgroupsPerDimension` (`65535` on
  every implementation measured against this repo) into several dispatches
  and concatenate the results, instead of asking for `workgroups: [M]`
  directly. Found running Sarashina2.2-1B-alibi-v1 in a browser for the
  first time (issue #106): `lm_head`'s decode-time projection
  (`vocabSize=102400`) silently produced an all-zero logits vector on every
  decode step — a WebGPU validation error past this limit is reported
  asynchronously through the device's error callback, not by throwing where
  `dispatchWorkgroups` is called, so nothing in the existing code path ever
  saw it fail. The visible symptom was the model emitting one correct
  character from prefill (which projects `lm_head` through the tiled
  `matmul` path, unaffected) and then the same wrong token — id 0 — forever,
  since argmax over an all-zero vector always picks index 0. Verified
  end-to-end by re-running the same generation in the browser after the fix
  and getting real, `llama.cpp`-comparable output (see #106's PR); the
  chunking logic itself is proven against a mocked `Runner["run"]`
  (`llm/kernels.chunking.test.ts`) rather than on real hardware at this
  scale, because a real dispatch at 65,535 workgroups reproducibly crashed
  this repository's own Node/Dawn binding (issue #38/#49/#107's family) —
  exactly the fragility issue #106 exists to route real-model verification
  around in the first place.

## [0.1.0] - 2026-08-04

### Documentation

- The README's op count, backend table and roadmap match what is on disk:
  **twenty-seven** ops, where both count lines still said twenty-four. This is
  the second time these same lines have gone stale, and the reason is structural
  rather than careless — op PRs land in parallel and each is told to leave the
  shared count line alone, since otherwise every one of them conflicts on it. The
  count is therefore correct when it is written and wrong again once the next
  batch merges. `group_norm` had also reached the per-op table without reaching
  the roadmap's `kernel/` layer, which is the same drift wearing a different
  shape. The per-op rows were right throughout; only the totals and that one
  roadmap tick were not. The count line still states outright that speed is
  unmeasured for every op, since rule 9 treats an omission as a claim of speed.

  Nothing here stops it happening a third time. What would is a test asserting
  the count against `ops/` on disk, the way `harness/coverage.ts` already asserts
  that a kernel variant cannot exist without a test looping it — worth doing, and
  deliberately not smuggled into a release-preparation change.

- The README says how to install this and what to import. It had neither: written
  as a design document, it could tell a reader why `scatter` accumulates but not
  how to call it. An op's two halves are now given as two imports, because they
  run in different places — the reference is dependency-free TypeScript that runs
  anywhere, while the kernel is a `.wgsl` file whose loading belongs to the
  caller's bundler and is documented as such rather than guessed at here.

### Added

- `istft` synthesises `"same"` padding, the vocoder convention: `(nFft - hop) / 2`
  cropped from each end, so `T` frames give exactly `T * hop` samples. torch has
  no such mode and it cannot be composed from one — `center: false` returns edge
  samples whose `w²` envelope is zero, so the signal a caller would slice cannot
  be produced in the first place. This is the last stage of a MioCodec-style
  decoder, and the point where latents become audio. `center` keeps working and
  is unchanged; `padding` is the new spelling and giving both raises.

- `group_norm` — `torch.nn.functional.group_norm`, pooling statistics over a
  group of channels while applying the affine transform per channel. Added
  because a codec decoder's residual blocks are `GroupNorm → SiLU → Conv1d` and
  nothing here could express the reduction: `layernorm` over `[N·C, L]` gives one
  mean per channel, where GroupNorm pools `C/G` of them into each. Same input,
  same output shape, no error either way, different numbers — the shape of bug
  this library exists to keep out of a caller's code.

  Both numerical conventions were measured against torch 2.10 rather than
  inherited from `layernorm`, since `group_norm` is a separate kernel upstream
  and its agreement is worth one measurement: the biased variance matched to
  4.4e-16 while the unbiased form was out by 1.6e-1, and moving `eps` outside
  the square root moved the answer by 4.3e-6 — small, but four thousand f32
  ulps and not noise. `C % G != 0` throws, as torch does; distributing a
  remainder would be a different op that happens to run. The layout is `[N, C,
  L]` contiguous, which makes a group one unbroken run rather than a stride.
- `snake_beta`, as a second entry point of `ops/snake` — `x + sin²(α·x)/β`, with
  α and β both learned per channel. Added because "Snake" turns out to name two
  different functions and the library shipped only one of them: DACVAE's has a
  single α, while BigVGAN's `SnakeBeta` — which MioCodec's decoder is built from
  — separates the sine's frequency from the amplitude it is added at. β = α
  recovers the one-parameter form, so this generalises rather than replaces.
  Two entry points rather than one kernel with an optional binding, because an
  unreferenced binding throws since #46 and the α-only path would otherwise have
  to supply a β buffer it never reads.

  The reason this needed saying at all is that the two fail into each other
  silently. A checkpoint holds a tensor called `alpha` and nothing in it says
  which function it belongs to — and the BigVGAN family stores **logarithms**
  where DACVAE stores values, so reading one as the other turns a log-scale
  α of `0.0` into α = 0, which makes `sin(0·x)` vanish and collapses the whole
  activation to the identity. No NaN, no shape mismatch: just a vocoder that
  sounds duller than it should. Neither kernel exponentiates — that belongs to
  whoever reads the checkpoint — and both references now say so where a reader
  will look.
- `snake` — `x + sin²(α·x)/α` with a learned per-channel α, as a **separate op**
  rather than a kind of `activation`. Every neural audio codec worth decoding on
  the web is built from it (DACVAE, BigVGAN), and without it the decoder stops
  at latents. It is separate because `activation` is "one buffer in, one buffer
  out, branch on an integer" and Snake's α is a tensor: folding it in would add
  a storage binding and a channel stride that `relu2` and `silu` would then have
  to supply on every call — the pipeline layout is derived from what the shader
  declares, so an unused binding is not free, it is mandatory. `elu`'s scalar
  `alpha` went the other way for the same reason: it costs a uniform field the
  struct was padding out anyway. The `1e-9` sits inside the reciprocal at
  upstream's value, because `1/α` guarded afterwards returns NaN at α = 0 where
  upstream returns x, and a checkpoint that trained an α to nothing would be the
  one to find that out.
- `activation` gains `elu`, `tanh`, `gelu` and `gelu_tanh`. The first three
  complete the set DACVAE's encoder and decoder stages need; GELU is for the
  other half of the same model, whose text encoder is a GeGLU ModernBERT. GELU
  is **two** functions in torch, and they part company by 4.73e-4 at x = 2.699 —
  measured, in the reference — so both ship and the default is the exact `erf`
  form, matching `torch.nn.functional.gelu`'s own default. A library that picked
  one silently would be right for half its callers and quietly wrong for the
  rest.
- `rmsnorm` takes an optional group count: `weight` may be `[G, D]`, and row `n`
  uses group `n % G`. QK-norm gives every attention head its own gamma, and
  until now this op could not say that — a per-head weight flattened into `[N,
  D]` silently applied head 0's gamma to every head, with no NaN and no change
  of shape to notice it by. A wrong answer that looks well-formed is worse than
  a missing op, which is why it is in the library rather than worked around by
  the caller. `G = 1` and an unpacked group word are both today's behaviour, so
  nothing existing moves. The grouped axis must be the one immediately left of
  `D` (`[B, S, H, Dh]`); `[B, H, S, Dh]` needs a different index and is refused
  rather than served wrongly. Speed unmeasured. `layernorm` has the same gap and
  #81 tracks it, so the two do not end up disagreeing about what a weight is.
- An **additive attention mask** on `attention`, `flash_attention` and `gqa`,
  because without one this library had no answer for an encoder at all. Every
  batched encoder over variable-length sequences needs to say "key `j` is
  padding, never attend to it", and `causal` cannot: it masks by position, not by
  content. Padding positions were contributing to the softmax and every
  conditioned output was quietly wrong — not NaN, not a crash, just slightly
  wrong. The form is PyTorch's **float** `attn_mask`, added to the scores before
  the softmax, and the reason it is the float form rather than the boolean one is
  `ops/alibi`: ALiBi already produces an additive score bias and says masking is
  the caller's, so with one representation the two compose by addition instead of
  needing an op to combine them. `keyPaddingBias(B, S, keep)` converts the
  boolean mask people actually hold, with `keep` naming the polarity because
  torch's own two APIs disagree about it. All three ops take the same mask, so it
  does not decide which kernel a caller may use.
- **A fully masked row now returns zeros** rather than NaN. It used to be
  unreachable — `queryOffset >= 0` guaranteed every row saw key 0, and the
  kernels said so and skipped the guard. A key mask breaks that guarantee, and an
  empty sequence in a padded batch reaches it by accident, so it is guarded in
  all three ops: a NaN row is a wrong answer that looks like a result. Zeros is
  what torch returns, through `aten::_safe_softmax` — measured, and worth
  measuring, because plain `torch.softmax` on the same row returns NaN.
- `rope` takes a head range — `headOffset` / `headCount`, defaulting to every
  head and byte-identical there. The joint-attention blocks of a diffusion TTS
  DiT rotate half their heads and leave the other half position-free, so that a
  block attending over positioned latents and unpositioned conditioning has
  heads that cannot see order at all. The layout is `(token, head, dim)`, so the
  heads to rotate are not contiguous and the honest workaround was a gather, a
  dispatch and a scatter — three passes and a temporary, spent to do less work.
  **The range is over heads, not over the channels within a head**: both are
  called "half-RoPE" in the wild, and channel-wise partial rotary (`rotaryDim`)
  is a different op that this deliberately does not add. Heads outside the range
  are copied rather than left at zero — `rope` returns a fresh array, so an
  unwritten head is silent garbage flowing into attention.
- `ops/conv_transpose` — `convTranspose1d` and `convTranspose1dOutputLength`,
  matching `torch.nn.functional.conv_transpose1d`. `conv` covers a codec's
  encoder; nothing covered the decoder, so the library could run the transformer
  half of a diffusion TTS model and then not turn latents into a waveform. A
  DAC-style codec decodes with transposed convolutions and nothing else — no
  iSTFT head to fall back on — which made this a hard blocker rather than a
  convenience (ISSUE #75, #80). Three conventions are stated rather than chosen
  quietly, because each fails silently: the weight is `[Cin, Cout/groups, K]`
  and not `conv`'s transpose of it, which only diverges when `Cin != Cout` —
  exactly when a codec is upsampling; `padding` *crops* the output instead of
  extending the input, so getting it backwards yields a waveform that is merely
  a little too long; and `output_padding` lengthens the trailing end without
  joining the sum, which matters only at odd `stride`, which is what DACVAE's
  decoder stages use. `weight_norm` is deliberately not a flag — folding
  `weight_g`/`weight_v` into a plain weight is a load-time conversion, not
  per-frame kernel work.
- `roofline(runner)` — this device's measured bandwidth and compute ceilings,
  taken here and now rather than derived from a spec sheet, and cached for the
  session. WebGPU exposes no clock, compute-unit count or bus width, so there is
  nothing to compute a theoretical peak from; a self-calibrating ceiling also
  throttles when the device does, which is what keeps a percentage meaningful
  across machines. Null where the device cannot time a dispatch, rather than an
  estimate.
- `Runner.time(dispatch)` — GPU time for a dispatch, read from a timestamp query
  written around the compute pass, and `null` on devices that cannot report it
  rather than a guess. Host wall-clock cannot stand in: it charges the dispatch
  for buffer creation, submission and the readback round trip, about 1.2 ms here
  against a real dispatch of 0.3 ms. Taking a bandwidth ceiling that way reported
  5.3 TB/s on a 1.79 TB/s card — the authoritative-looking, unrelated figure the
  roofline design exists to avoid.
- A `scratch` binding: storage the kernel uses but nobody uploads or reads back,
  so a measurement of transfer rate does not sit next to a transfer of the same
  size. Reused across dispatches rather than reallocated, because repeated large
  allocations abort this binding — a 64 MiB buffer allocated twice kills the
  worker on the second.
- The device now asks the adapter for the limits it actually offers. A device
  requested with none caps storage bindings at 128 MiB where this adapter allows
  2 GB.

- `harness` — a WGSL runner over Dawn in Node, an `agree` comparator that passes
  an element on either relative or absolute difference, and a suite helper that
  creates exactly one GPU device per run. The comparator is not equality because
  the references run in f64 and the kernels in f32; the single device is because
  the GPU binding is a native module that does not survive vitest recycling its
  workers.
- `rmsnorm` — workgroup tree reduction, with `eps` guarding an all-zero row.
- `softmax` — max-subtracted, so real logits do not overflow `exp`.
- `activation` — `relu2` and `silu`.
- `elementwise` — `add` and `multiply`.
- `rope` — rotary position embedding, with a KV-cache offset.
- `quantize` — per-row absmax to int8, symmetric over `[-127, 127]`.
- `dequantize` — applies both the weight scale and the activation scale.
- `matvec` (GEMV) — one vector against a `[M, K]` row-major matrix, following
  `torch.mv` rather than BLAS `sgemv`: no `alpha`, no `beta`, no transpose flag.
  It exists as its own op rather than as a path through a future `matmul`
  because it reads every weight exactly once and reuses none of them, so the
  kernel is written to stream — lanes walk a row at the workgroup stride, which
  keeps each pass one contiguous burst — and the tiling that makes GEMM fast has
  nothing to capture here. Autoregressive decoding is this shape at every step.
  Speed is **unmeasured**: the bandwidth roofline it should be reported against
  does not exist yet.
- `matmul` — GEMM (`C = A @ B`) with a shared-memory tiled WGSL kernel. Separate
  from GEMV because the reuse a tile buys is the only reason this shape can be
  compute-bound, and none of it applies to a batch of one. Shapes that do not
  divide by the tile are where tiled kernels are fast and wrong, so the ragged
  tails are tested on their own and together — including against buffers longer
  than the operands, because this device reads past the end of a buffer as zero
  and would otherwise hide a missing tail guard.
- `scatter` — indexed writes where **colliding indices accumulate**, via an f32
  compare-exchange atomic. The rule had to be decided rather than discovered:
  "last write wins" is undefined behaviour on a GPU, since nothing orders the
  threads reaching a slot, and callers would have built on whichever answer
  their own device happened to give. Accumulation is the only rule that is the
  same for every ordering, and it is what gradient accumulation, MoE dispatch
  and bincount need. Matches `scatter_add_` in PyTorch, not `scatter_`.
- `transpose` — 2D, staged through a 16x16 workgroup tile. The tile is there
  because transpose computes nothing: the only thing it can get wrong is where a
  value lands, and the only thing it can be slow at is reaching memory. Turning
  the tile inside the workgroup keeps both the read and the write consecutive,
  which the obvious one-line version does for the read only. Shapes that do not
  divide by 16 are the case that matters — the leftover threads address inside
  the buffer, so an unguarded write replaces a real value instead of faulting.
- `reduce` — `sum` / `max` / `min` / `mean` along one axis, over a tensor viewed
  as `[outer, axis, inner]` so that any axis of any rank fits. rmsnorm and
  softmax each carry their own copy of the workgroup tree reduction, and a third
  copy was about to be written; this is that reduction with the combiner and the
  identity lifted out. The edges follow PyTorch and are stated rather than
  implied: an empty axis sums to `0`, means to `NaN`, and is an error for `max`
  and `min`; `mean` always divides by the axis length. Callers who get those
  wrong get them wrong quietly, which is why they are written down.
- `layernorm` — mean-subtracted normalisation with a bias term, following
  `torch.nn.functional.layer_norm`: the variance is **biased** (`1/D`, not
  `1/(D-1)`) and `eps` sits inside the square root. Both were checked against
  PyTorch in float64 rather than read off the documentation, because the two
  variance conventions differ by less than a percent on a wide row and a caller
  would never notice which one they got. The variance is computed in two passes
  — mean, then the mean of the squared deviations — instead of the one-pass
  `E[x²] - E[x]²`. That identity is not a performance choice with a numerical
  footnote; it is broken where LayerNorm is most often used. Measured on this
  device by running both shaders: on a row of `8192 ± 4`, true variance 6.678162,
  the two-pass kernel returns 6.67816 and the one-pass identity returns a
  *negative* number — the squares land past 2^26 where f32 steps by 8, the
  subtraction cancels every digit the spread had, and `inverseSqrt` of that is
  NaN. The same test at `1024 ± 4` is the more dangerous half: 6.62347 against
  6.678162, 0.8% low, a number nobody would look at twice. Both rows are in the
  suite, and with the one-pass form in place they are the only test of the eight
  that fails — which is the whole reason they had to be written.
- `stft` / `istft` — windowed transform and its inverse, the pair ONNX cannot
  express, since it has no complex tensors and so cannot carry the spectrogram a
  vocoder head emits. Every convention follows `torch.stft` / `torch.istft` and
  is checked against it numerically rather than read off the documentation:
  centred by default, reflect padding without repeating the edge sample,
  one-sided, unnormalised, `hannWindow` periodic like `torch.hann_window` and
  `scipy.signal.get_window` rather than symmetric like `np.hanning`. Named
  because each has more than one defensible answer and picking silently means
  half the callers get a subtly wrong waveform. `istft` divides by the
  overlap-added `w²` envelope instead of assuming the window is COLA — a
  periodic Hann at 50% overlap is COLA in `w` but **not** in `w²`, so a
  reconstruction that skipped the division is wrong by up to 2x and still sounds
  like audio. Two departures from torch, both in `ops/stft/reference.ts`: the
  layout is frame-major `[frames, bins]` because a vocoder head emits one row
  per frame, and asking for more samples than the frames reach raises instead of
  quietly returning a zero tail. Speed is **unmeasured**, and the kernels are a
  naive DFT rather than an FFT: 1920 is not a power of two, this sits beside a
  transformer, and correctness came first.
- Kernel resolution — an op's `wgsl/` directory holds one or more **entry
  points**, each of which may have variants, and which file runs is decided by
  `resolve()`: `explicit override → target + dtype → target → dtype → portable`,
  first hit wins, **within one entry point**. The filename carries it:
  `<entry>[.<target>][.<dtype>].wgsl`, so `kernel.wgsl` and `inverse.wgsl` are
  two entry points and `kernel.nvidia.wgsl` is a variant of the first. Entry
  points are named rather than assumed because ops already need them — `stft`
  computes the inverse transform with different arithmetic, `attention` is two
  dispatches split into two files so `layout: "auto"` cannot drop bindings an
  entry point does not reference — and because resolution must never leave the
  entry point it was asked about: `istft` falling back to the forward transform
  would be a wrong answer that still looks like a result. A suffix that is not a
  known target or dtype is an error, and so is a bare `nvidia.wgsl`, which is far
  likelier to be a mis-written variant than an entry point called "nvidia".
  Target detection reads `adapter.info` and is
  allowed to answer "I don't know", because a vendor string does not say what a
  device is good at; an unknown adapter gets the portable kernel rather than
  someone else's. Intel is the standing example — the same vendor string covers
  an on-die iGPU and a discrete Arc card. Because the hint will sometimes be
  wrong, the override beats detection outright, and one that names a variant that
  does not exist raises instead of quietly falling back. The choice is readable
  through `describeAdapter(adapter)` and the returned `Choice`, which names the
  rung that hit and every candidate tried: a wrong guess nobody can see is worse
  than a portable kernel. Reading the adapter is a call the caller makes rather
  than something `createRunner` does on the way past, so nothing changes for ops
  that only want a kernel run.
  Adding a variant cannot skip the reference test — `eachVariant(url, entry, …)`
  builds an op's test loop from its `wgsl/` directory rather than from a list, and
  `unguardedOps` fails the suite for an entry point that grows a variant no test
  iterates. Per entry point, not per op: looping `scores` says nothing about
  `context`. None of it is re-exported from `harness/index.ts`, which every op's
  test imports, so `index.ts` and `suite.ts` are byte-identical to what they were
  before this landed and no existing op's test loads anything new.
- `ctc_decode` — greedy CTC decoding: argmax per frame, collapse repeats, then
  drop blanks, **in that order**. The order is the whole op. Stripping blanks
  first and collapsing afterwards is the implementation everyone reaches for,
  and it turns `a ␣ a` into a single `a` — losing the doubled letter that the
  blank symbol exists to make expressible. The two orders agree on every other
  input, which is what lets the wrong one survive being tested. Blank defaults
  to `0`, as `torch.nn.CTCLoss` has it, and is a parameter because TensorFlow's
  `ctc_greedy_decoder` puts it last and models trained that way exist. The
  result never leaves the GPU: labels come back as `[B, T]` padded with `-1`,
  and the true lengths in their own `[B]` buffer, both written by the kernel —
  a greedy decode emits at most one label per frame, so the allocation never
  depends on the answer. Reading the labels back to find out how long they are
  would be the per-step readback this op exists to avoid. Beam search is
  deliberately absent: it is a different algorithm and needs its own decision
  about where the beam lives.
- `mel` — the filterbank and its application, in two kernels because the matrix
  depends only on scalars and is built once per configuration while the
  application runs per frame. The other half of the DSP gap next to `stft`: no
  ML kernel library ships it because it is not machine learning, so every voice
  encoder reimplements it in numpy, and that numpy is exactly what has to be
  rewritten to move a pipeline into a browser. Four conventions have more than
  one answer in wide use and all four are named rather than picked silently —
  the HTK mel scale, unnormalised triangles, a power spectrum, and a dB log
  flooring its argument at `1e-10`, which together are
  `torchaudio.transforms.MelSpectrogram` + `AmplitudeToDB(stype="power")`.
  Asking for `{ scale: "slaney", norm: "slaney" }` gives `librosa.filters.mel`'s
  defaults instead, and on the same audio those two differ by a factor of 200,
  which is what silently picking one would have cost a caller. Checked against
  torchaudio 2.10 and librosa 0.11 on a real recorded voice, not on a formula.
  `top_db` is deliberately absent: it clamps against the maximum over the whole
  spectrogram, so it cannot be computed before the last frame exists and it
  makes the answer depend on how the caller chunked their audio — that is
  `reduce` then `elementwise`. Speed is **unmeasured**.
- `gather` — row selection for embedding lookup, matching
  `torch.index_select(table, 0, indices)` rather than `torch.gather`, because
  embedding lookup is why the op exists and the two names are close enough to
  pick the wrong one by accident. An index outside `[0, rows)` gathers zeros:
  PyTorch raises there and a kernel cannot, and the alternatives — clamping or
  wrapping — hand back a real embedding for a token that was never in the
  vocabulary, which looks plausible all the way downstream.
- `conv` — 1D only, matching `torch.nn.functional.conv1d`, with `stride`,
  `padding`, `dilation`, `groups` and an optional `bias`. The convention worth
  stating out loud is that PyTorch's `conv1d` is a **cross-correlation**: it does
  not flip the kernel. `F.conv1d([[[1,2,3,4]]], [[[1,10,100]]])` is `[[[321,
  432]]]`, not `[[[123, 234]]]`. The two definitions agree on every symmetric
  kernel, so a library that quietly picks the mathematical one passes every
  hand-written test and then disagrees with PyTorch on real weights. `groups` is
  in from the start because `groups = Cin = Cout` is a depthwise conv, which is
  what the speech front-ends this op exists for actually run. 2D is deliberately
  absent until something asks for it. Speed is **unmeasured**: the roofline it
  should be reported against does not exist yet.
- `attention` — unfused scaled dot-product attention, in two dispatches:
  `softmax(mask(scale * Q @ K^T))` writes the attention matrix, then `@ V` reads
  it. Slower than a fused kernel by construction, and worth having anyway,
  because it is what makes the fused one verifiable — `flash_attention` has no
  other definition of correct to be measured against.
  Conventions follow `torch.nn.functional.scaled_dot_product_attention`, checked
  against torch 2.10 rather than read off the docs, because the two that matter
  both have a plausible wrong answer: `scale` defaults to `1/sqrt(D)` from the
  **query's** head dim even when V's differs, and `causal` is **upper-left**
  aligned, so with a KV cache longer than the query it keeps keys `0..i` rather
  than the most recent `i+1`. That second one is a trap for the case the op
  exists to serve, so masking is parameterised by `queryOffset` — the absolute
  position of query row 0 in the key sequence — which reaches `is_causal=True`
  at `0` and `causal_lower_right` at `S - L` without a second flag.
  Speed is **unmeasured**: the roofline harness it should be reported against
  does not exist yet.
- `rope` gains NTK and YaRN context scaling, as an optional `scaling` argument
  rather than as new ops. Neither scheme changes the rotation — they change
  which frequency each pair rotates at — so forking `rope` would have left three
  copies of the same rotation to keep in step. There is no PyTorch definition to
  follow, so the convention is `jquesnelle/yarn` and `transformers`, which
  agree; YaRN's attention temperature (`0.1·ln(s) + 1`) **is** included, folded
  into `cos`/`sin` as both of those do, and can be overridden for checkpoints
  fine-tuned with a different one. Omitting `scaling` leaves plain RoPE
  identical bit for bit, which is a property of the arithmetic rather than a
  tolerance: the unscaled path multiplies an exact IEEE zero. Speed is
  **unmeasured** — scaling costs one multiply-add per element over plain RoPE,
  and the roofline it should be reported against does not exist yet.
- `rope` gains an optional precomputed angle table, `ropeCache`. The angles
  depend on position and pair and on nothing else, so decoding recomputes the
  same `pow`, `sin` and `cos` for every head at every step; a table trades
  memory for those. What the table does when decoding runs past its end had to
  be decided rather than discovered, because the tempting answers are silently
  wrong: growing it needs a host that a dispatch does not have, and wrapping
  (`pos % positions`) returns a real angle for the wrong position, which comes
  back as a plausible tensor rather than as an error. So it **falls back** to
  computing the angle — past the end it is exactly the uncached op — and a
  table built for a different `thetaBase` or `scaling` is **refused**, since
  that is the same failure by another route. The transcendental saving is
  counted rather than asserted: 3 calls per (token, head, pair) become 0, and
  the table costs 3 per (position, pair) once, so a decode saves a factor of
  `numHeads`. It does **not** show up in wall time on the GPU measured — RoPE
  there is bandwidth-bound, so the calls were never what it was waiting for;
  see the PR for the numbers and the conditions.
- `flash_attention` — the same function as `attention`, in one dispatch, with the
  `[B, H, L, S]` score matrix never allocated. That is the entire difference and
  the entire reason it is a kernel rather than a composition: at any sequence
  length worth fusing for, the score matrix is the largest thing in the
  computation. Tiled online softmax — a running max and a running sum, so a tile
  of 64 keys folds in without a second pass — and the only score storage anywhere
  is that one tile in workgroup memory, which does not depend on `S`.
  Both halves of the claim are tested, because the first half alone would pass a
  kernel that materialised the matrix and read it back. Agreement is checked
  against **both** references, this op's and the unfused one's. Allocation is
  checked by counting the bytes behind the dispatches that produced those
  answers, at shapes where `L` and `S` grow together — the only sweep where
  `seq x seq` and `seq` can be told apart — and holding the total to a second
  difference of zero. Bound bytes at `B=H=1, L=S=n, D=Dv=8`: `128n + 32` here
  against `4n² + 64n + 28` unfused, which is 32_800 against 278_556 at n = 256
  and doubles its advantage with every doubling of the sequence.
  Conventions are inherited from `attention` unchanged — same `scale` default,
  same `queryOffset` mask — and the arg type is imported rather than restated so
  the two cannot drift. Speed is **unmeasured**: the roofline harness is #3 / #4.
  Memory is measured, because memory is what this op is a claim about.
- `alibi` and `pope` — two more position encodings, so that position encoding is
  a family in this repository rather than a synonym for `rope`. Models pick
  differently and none of the three can be substituted for another: `rope`
  rotates Q and K, `alibi` biases the attention scores, `pope` builds a table
  that is added to the embeddings. They run on different tensors at different
  points in the layer.

  `alibi` ships as two kernels because its two halves fail differently. The
  per-head slopes are where implementations quietly disagree, and the
  disagreement is invisible at 8 heads: for a power-of-two head count everyone
  produces the same geometric run, and for anything else the paper appends
  every other slope of the *next* run rather than interpolating — so the
  sequence is not monotonic, and an implementation that sorts or truncates is
  wrong only at head counts nobody tests. The slopes therefore have their own
  kernel, their own GPU test, and a reference test that pins the published
  numbers, so a fault in the slopes cannot hide inside a correct bias and the
  convention is checked against the paper rather than against the kernel. The
  bias itself follows the paper's relative form, `m * (j - i)`; BLOOM's
  `m * j` differs by a per-row constant that the following softmax erases, but
  this op returns the tensor and not the softmax, so the two are not
  interchangeable here. Masking is left to the caller — writing `-inf` above
  the diagonal would fold a masking policy into a bias op.

  `pope` records where its paper is silent instead of guessing. The polynomial
  order is the token position and the argument sweeps the feature index across
  `[-1, 1)`, which is the paper's Equation (14); whether positions start at 0
  or 1 is not stated, and it matters, because at order 0 the polynomial is the
  constant 1 and the first token would carry no position at all. That is a
  required `posOffset` argument rather than a default, the same way `rope`
  takes one. It is evaluated by the three-term recurrence, not by Rodrigues'
  formula, and the reason an f32 kernel can walk the recurrence is that
  `|P_n(x)| <= 1` holds on the domain — measured, not assumed: the f32
  recurrence sits 4.1e-6 from f64 at order 70 and 1.6e-5 at order 128, so the
  tolerance is stated for the order range the tests reach and not beyond it.

  Speed is **unmeasured** for both; the roofline to report against does not
  exist yet.

- `moe` — MoE routing: `router` (top-k over expert logits and the gate weights),
  `moeDispatch` (tokens reordered into per-expert contiguous runs) and
  `moeGather` (results back in token order, weighted). One op rather than three
  because none of them is usable without the other two, and because the
  decisions that matter are decisions *between* them. Three of those had to be
  made rather than inherited, and all three change the model's output:
  **ties in top-k go to the lower expert index**, which `torch.topk` explicitly
  leaves open and a GPU cannot — otherwise the same token reaches different
  experts on different runs; **capacity overflow drops by rank first, then by
  token index** (GShard, the Switch Transformer, fairseq `top2gating`), so a
  token's first-choice expert outranks another token's second choice, and not
  by arrival order, which would make the set of dropped tokens depend on the
  scheduler; and the **gate is applied in gather and only there**, since
  weighting on the way in puts it inside the expert FFN and weighting at both
  ends squares it, which stays plausible while being wrong everywhere.
  Renormalising the k gates is a caller's flag with no default, because Mixtral
  renormalises and the Switch Transformer must not — at `k = 1` renormalisation
  makes every gate exactly 1 and deletes the gate entirely. Nothing divides by
  an expert's token count: with 64 experts and a short sequence, most experts
  receive nothing on most steps, and that division is `0/0`. Speed is
  **unmeasured**.
- `gqa` — grouped-query and multi-query attention: `kvHeads` query heads share
  one K and one V. One op rather than two, because MQA is GQA with `kvHeads = 1`
  and MHA is GQA with `kvHeads = H` — splitting them would mean two copies of the
  same index arithmetic, one of them less tested.
  The reason to reach for it is a memory number, so here is the number. One
  Llama-3-8B decoder layer, batch 1, 8192 cached positions, 32 query heads, head
  dim 128, f32: the KV cache is **268,435,456 bytes** at `kvHeads = 32` (MHA),
  **67,108,864 bytes** at `kvHeads = 8` (GQA) and **8,388,608 bytes** at
  `kvHeads = 1` (MQA). Over the model's 32 layers that is 8 GiB against 2 GiB — a
  saving of **6,442,450,944 bytes**, which during decoding is usually the
  difference between the model fitting and not. `kvCacheBytes()` is exported so
  the figure stays a computation rather than a claim in a document.
  The head mapping is **contiguous** groups, `kvHead = h / (H / kvHeads)`,
  matching `enable_gqa=True` on torch 2.10 — measured, because the alternative
  reading (strided, `h % kvHeads`) agrees with it at both `kvHeads = H` and
  `kvHeads = 1` and disagrees everywhere in between, which is how an
  implementation passes the two configurations people test and fails the ones
  they ship. `H % kvHeads != 0` throws rather than picking a grouping for the
  caller, as torch does.
  Speed is **unmeasured**: the roofline harness it should be reported against
  does not exist yet.

### Fixed

- The harness refuses a shader that does not compile instead of reading back
  zeros. Output buffers start zeroed, so a dispatch that never ran produced a
  result — and where an expected value contains zeros, that is a passing test
  over a kernel that executed nothing. Every correctness claim here is "it agrees
  with the reference", so a silent no-op reading as agreement could have hollowed
  out all of them at once. Compiled modules are now cached by source as well, so
  the check costs one await per distinct shader rather than one per dispatch.

- The per-file timeout in `npm test` actually fires. It never had: `npx` starts
  vitest as a grandchild that keeps the stdout pipe open, so killing the child
  left `"close"` unfired and the run waited forever. A hanging file took an outer
  120s kill instead of the 6s limit it was given. It hid because a *GPU* hang
  brings the worker down within seconds on its own, which looked like the timeout
  working — the suite could not tell "wedged" from "slow".

### Changed

- The score kernels of `attention`, `gqa` and `flash_attention` take a **mask
  buffer at every dispatch**, not only when there is a mask, and their uniform
  blocks gained three fields for its broadcast shape. A bias of zeros *is* "no
  mask" — it is what torch's own reference does — and making it unconditional
  removes a per-element branch and a `has_mask` guard whose only failing input
  would be a dummy buffer nobody reads. It costs `B * S` floats against the
  `B * H * L * S` these kernels already move. Callers driving the WGSL directly
  must add the binding; `attention` and `gqa` bind it at index 2, `flash_attention`
  at index 3.
- `npm test` runs one test file per vitest process. A single process cannot cross
  a test-file boundary with a GPU device in play — it aborts inside Dawn's thread
  pool or hangs, with no kernel of its own required to trigger it. The runner also
  refuses to report a false pass: a crashed vitest worker can exit 0 having
  skipped most of the suite, so every file is now accounted for individually.

- The package ships **compiled JavaScript and type declarations** instead of
  TypeScript source. `exports` pointed straight at `.ts` files with no build
  behind them, which requires every consumer to own a TypeScript toolchain
  configured the way this repo's is, and bills a browser bundle for comments that
  belong in the source. `npm run build` emits `dist/`, `prepublishOnly` runs it,
  and CI runs it too — the tests import from the source tree and never touch
  `dist/`, so a broken package is invisible to them by construction.
  `removeComments` applies to the JavaScript only: the declarations keep their
  JSDoc, so the reasoning still reaches an editor's hover.

- `harness` is **no longer exported**. It imports vitest and the Dawn binding,
  both devDependencies, so that subpath resolved to code no consumer's runtime
  could load. Nothing imported it that way regardless — the tests reach it by
  relative path — so this removes a promise that was never kept rather than a
  feature anyone had. It stays in the repo as the test infrastructure it always
  was.

- The `.wgsl` kernels are published beside the JavaScript at
  `web-xpu-ops/ops/<op>/wgsl/<entry>.wgsl`, mirroring the source tree so a
  kernel's path is one string across the repo, the resolution grammar and the
  package. `scripts/assets.mjs` copies them, then fails the build if an op
  compiled to a reference with no kernel beside it: `tsc` emits no assets, so
  without that check a package missing its entire WGSL backend would type-check,
  pack and publish without a word.

- Publishing is a **tag-triggered workflow** rather than a command someone runs
  on a laptop. `.github/workflows/release.yml` fires on `v*`, repeats the lint,
  build and test that CI runs — a tag can point at a commit that never went
  through CI — and publishes with `--provenance`, so the tarball is linked to the
  workflow run and commit that produced it rather than being bytes of unknown
  origin. It refuses to publish when the tag disagrees with `package.json`: the
  tag is typed by hand while `npm publish` reads the version from the file, and
  an npm version, once taken, cannot be replaced.

  The token is an npm automation token in a repository secret. Trusted publishing
  (OIDC) would remove that secret, and is the better end state, but npm cannot
  configure a trusted publisher for a package that does not exist on the registry
  yet — so it is a follow-up to the first release, not part of it.

- `package.json` carries `keywords`, `homepage` and `bugs`. None of it changes
  what is installed; all of it is what npm's own page and search have to work
  from, and a package that cannot be found is not meaningfully published.

- The first released version is `0.1.0`, not `1.0.0`. One backend of the three
  described here exists, every op's speed is unmeasured against the roofline it
  should be reported against, and the resolution grammar has target and dtype
  rungs that no kernel yet uses. A major version would claim those are settled.

[Unreleased]: https://github.com/m96-chan/web-xpu-ops/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/m96-chan/web-xpu-ops/releases/tag/v0.1.0
