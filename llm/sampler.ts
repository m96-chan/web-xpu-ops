/**
 * Token sampling over a next-token logits vector.
 *
 * Two modes: `greedy` (argmax) and `top-p` (temperature-scaled softmax with
 * nucleus truncation). Both take an optional `Constraint`, which narrows the
 * legal next token *before* any probability math runs — the mask is applied
 * to the raw logits, so a constrained greedy pick is the argmax among only
 * the allowed ids, and a constrained top-p draw can never land on an
 * excluded one.
 *
 * ## Repetition penalty is intentionally NOT implemented
 *
 * This sampler has no repeat/frequency penalty, and that is deliberate
 * rather than a gap. The consuming project (Alibi) tried the usual
 * logit-penalty-per-seen-token approach against its Japanese model and found
 * it breaks output rather than de-looping it — see
 * technologies-moe/alibi-ai#3 for the empirical writeup. Penalizing "seen"
 * tokens does not respect multi-token kana/kanji boundaries or the way
 * natural Japanese repeats particles and sentence-final forms, so it pushes
 * the model off-distribution instead of discouraging genuine loops. Nothing
 * here silently reduces token probabilities based on generation history; a
 * caller that wants repetition control should express it as a `Constraint`
 * (this module's own tool for narrowing the next token) rather than as a
 * penalty baked into the sampler.
 */

/**
 * Token-level constraint interface: given everything generated so far,
 * returns which token ids are legal to come next.
 *
 * `null` means unconstrained — every token in the vocabulary is allowed, and
 * the sampler skips masking entirely. An empty (size 0) `Set` means the
 * opposite extreme: no legal continuation exists. `sampleNext` treats that
 * as a hard error rather than silently returning some token, because there
 * is no correct choice to make on the constraint's behalf — it means either
 * the constraint's spec is unsatisfiable from this prefix or the caller
 * kept sampling past a point where the constraint had already signalled
 * completion (typically right after EOS).
 *
 * The interface knows nothing about tokenizers, text, or grammars — it only
 * ever sees and returns token ids. `llm/constraints/line-format.ts` is one
 * implementation; nothing here depends on it.
 */
export interface Constraint {
  nextAllowed(prefixTokens: readonly number[]): ReadonlySet<number> | null;
}

export type SamplerOptions =
  | { readonly mode: "greedy" }
  | {
      readonly mode: "top-p";
      readonly temperature: number;
      readonly topP: number;
      /** Source of randomness for the nucleus draw. Defaults to `Math.random`. */
      readonly rng?: () => number;
    };

/** Replaces every logit outside `allowed` with `-Infinity`, in place of a fresh copy. `null` means no masking. */
function applyMask(logits: ArrayLike<number>, allowed: ReadonlySet<number> | null): Float64Array {
  const masked = new Float64Array(logits.length);
  for (let i = 0; i < logits.length; i += 1) masked[i] = logits[i]!;
  if (allowed === null) return masked;
  for (let i = 0; i < masked.length; i += 1) {
    if (!allowed.has(i)) masked[i] = -Infinity;
  }
  return masked;
}

function argmax(logits: ArrayLike<number>): number {
  let bestIndex = -1;
  let bestValue = -Infinity;
  for (let i = 0; i < logits.length; i += 1) {
    const v = logits[i]!;
    if (v > bestValue) {
      bestValue = v;
      bestIndex = i;
    }
  }
  return bestIndex;
}

/**
 * Temperature-scaled softmax, truncated to the smallest prefix of
 * descending-probability tokens whose cumulative mass reaches `topP` (the
 * nucleus), renormalised, and sampled from with `rng`.
 */
function sampleTopP(logits: ArrayLike<number>, temperature: number, topP: number, rng: () => number): number {
  if (!(temperature > 0)) {
    throw new Error(`sampleTopP: temperature must be > 0, got ${temperature}`);
  }
  if (!(topP > 0) || topP > 1) {
    throw new Error(`sampleTopP: topP must be in (0, 1], got ${topP}`);
  }

  const n = logits.length;
  const scaled = new Float64Array(n);
  let maxScaled = -Infinity;
  for (let i = 0; i < n; i += 1) {
    const v = logits[i]! / temperature;
    scaled[i] = v;
    if (v > maxScaled) maxScaled = v;
  }
  if (maxScaled === -Infinity) {
    throw new Error("sampleTopP: no finite logit to choose from");
  }

  const probs = new Float64Array(n);
  let total = 0;
  for (let i = 0; i < n; i += 1) {
    const p = Math.exp(scaled[i]! - maxScaled);
    probs[i] = p;
    total += p;
  }

  const order = Array.from({ length: n }, (_, i) => i)
    .filter((i) => probs[i]! > 0)
    .sort((a, b) => probs[b]! - probs[a]!);

  let cumulative = 0;
  let cutoff = order.length;
  for (let k = 0; k < order.length; k += 1) {
    cumulative += probs[order[k]!]!;
    if (cumulative / total >= topP) {
      cutoff = k + 1;
      break;
    }
  }
  const nucleus = order.slice(0, cutoff);

  let nucleusTotal = 0;
  for (const i of nucleus) nucleusTotal += probs[i]!;

  const draw = rng() * nucleusTotal;
  let acc = 0;
  for (const i of nucleus) {
    acc += probs[i]!;
    if (draw < acc) return i;
  }
  // Floating-point edge: the draw landed exactly on (or past, by an epsilon)
  // the total. The last nucleus member is the correct answer either way.
  return nucleus[nucleus.length - 1]!;
}

/**
 * Picks the next token id given `logits`, the tokens generated so far
 * (`prefixTokens` — passed straight through to `constraint`, unread
 * otherwise), the sampling mode, and an optional constraint.
 */
export function sampleNext(
  logits: ArrayLike<number>,
  prefixTokens: readonly number[],
  options: SamplerOptions,
  constraint?: Constraint,
): number {
  const allowed = constraint ? constraint.nextAllowed(prefixTokens) : null;
  if (allowed !== null && allowed.size === 0) {
    throw new Error("sampleNext: constraint allows no token to follow this prefix");
  }
  const masked = applyMask(logits, allowed);

  if (options.mode === "greedy") {
    const choice = argmax(masked);
    if (choice < 0) throw new Error("sampleNext: no finite logit to choose from");
    return choice;
  }
  return sampleTopP(masked, options.temperature, options.topP, options.rng ?? Math.random);
}
