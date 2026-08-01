import { existsSync, readFileSync, readdirSync } from "node:fs";
import { join } from "node:path";
import { fileURLToPath } from "node:url";
import { DTYPES, TARGETS, type Dtype } from "./target.js";

/**
 * Which kernel file to run, and why.
 *
 * An op's `wgsl/` directory holds one or more **entry points**, and each entry
 * point may have variants specialised for a target, a dtype, or both. The
 * filename says which is which:
 *
 * ```
 * kernel.wgsl              the default entry point, portable
 * inverse.wgsl             a second entry point, portable        (ops/stft)
 * scores.wgsl              one of two entry points, portable     (ops/attention)
 * kernel.nvidia.wgsl       a target variant of `kernel`
 * inverse.f16.wgsl         a dtype variant of `inverse`
 * scores.apple.f16.wgsl    both
 * ```
 *
 * Resolution walks one chain **within a single entry point**, first hit wins:
 *
 * ```
 * explicit override  →  target + dtype  →  target  →  dtype  →  portable
 * ```
 *
 * ## Why the entry point is part of the name
 *
 * The first version of this grammar had no entry points: a file was either
 * `kernel.wgsl` or a variant, so `inverse.wgsl` read as "a variant for a target
 * called inverse" and was rejected. That was already wrong when it was written.
 * `stft` needs a second entry point because the inverse transform is a different
 * kernel, not a tuned one, and `attention` needs two because it is two
 * dispatches with a buffer between them — split into separate files on purpose,
 * since `layout: "auto"` drops bindings an entry point does not reference (#46).
 *
 * Prefixing with the entry point keeps the strictness that made the first
 * version worth having. A suffix still has to be a target or a dtype, so
 * `kernel.nvidai.wgsl` is an error rather than a file nothing resolves to. What
 * it gives up is the bare `nvidia.wgsl` spelling, which is now rejected as
 * ambiguous: it is far more likely to be a mis-written variant than a genuine
 * entry point called "nvidia", and guessing either way would be silent.
 *
 * The one thing this cannot catch is a misspelt **entry point** — `inverze.wgsl`
 * reads as a legitimate new entry point. That asymmetry is deliberate rather
 * than overlooked: a misspelt variant fails *silently*, because resolution just
 * falls back to portable and everything still runs. A misspelt entry point fails
 * *loudly* the moment its test asks for it, because `kernel(url, "inverse")`
 * cannot read a file that is not there. Only the silent one needs a guard.
 *
 * Nothing here knows what a real target is called. The vocabulary belongs to
 * detection; resolution is name joining and set membership, which is what lets
 * the order be tested against targets that do not exist.
 */

/** The entry point an op has unless it says otherwise. */
export const DEFAULT_ENTRY = "kernel";

export interface Variant {
  /** File stem — `"kernel"`, `"inverse.nvidia"`. Names the `.wgsl` beside it. */
  name: string;
  /** Which entry point this is a spelling of. */
  entry: string;
  /**
   * Typed as a plain string rather than `Target`: the vocabulary is injectable
   * so the resolution order can be exercised without naming a real vendor.
   */
  target: string | null;
  dtype: Dtype | null;
}

/**
 * Reads a variant filename.
 *
 * Throws on anything it does not recognise. A misspelt variant is worse than a
 * missing one — `kernel.nvidai.wgsl` sits in the tree looking tuned while
 * resolution never once asks for it.
 */
export function parseVariant(
  file: string,
  targets: readonly string[] = TARGETS,
  dtypes: readonly string[] = DTYPES,
): Variant {
  const reject = (why: string): never => {
    throw new Error(`unrecognised variant file ${JSON.stringify(file)}: ${why}`);
  };
  if (!file.endsWith(".wgsl")) return reject("not a .wgsl file");
  const stem = file.slice(0, -".wgsl".length);
  const [entry, ...suffix] = stem.split(".");
  if (!entry) return reject("no entry point name");

  if (suffix.length === 0) {
    // Rejected rather than accepted as an entry point: this is the spelling the
    // first version of this grammar used for variants, so anyone who writes it
    // almost certainly means the variant and would otherwise get an entry point
    // that nothing ever dispatches.
    if (targets.includes(entry) || dtypes.includes(entry)) {
      return reject(
        `ambiguous — ${JSON.stringify(entry)} is a target or dtype, so this reads as a variant ` +
          `with no entry point. Write <entry>.${entry}.wgsl, or rename the entry point`,
      );
    }
    return { name: stem, entry, target: null, dtype: null };
  }

  if (suffix.length === 1) {
    const [only] = suffix as [string];
    if (targets.includes(only)) return { name: stem, entry, target: only, dtype: null };
    if (dtypes.includes(only)) return { name: stem, entry, target: null, dtype: only as Dtype };
    return reject(`${JSON.stringify(only)} is neither a known target nor a known dtype`);
  }

  if (suffix.length === 2) {
    const [target, dtype] = suffix as [string, string];
    if (!targets.includes(target)) return reject(`${JSON.stringify(target)} is not a known target`);
    if (!dtypes.includes(dtype)) return reject(`${JSON.stringify(dtype)} is not a known dtype`);
    return { name: stem, entry, target, dtype: dtype as Dtype };
  }

  return reject("expected <entry>[.<target>][.<dtype>].wgsl");
}

function toPath(dir: string | URL): string {
  return typeof dir === "string" ? dir : fileURLToPath(dir);
}

/**
 * Every variant present in a `wgsl/` directory, sorted by name.
 *
 * Discovery rather than a list: a list is something a person has to remember to
 * extend, and the whole failure this guards against is a variant nobody
 * remembered.
 */
export function variantsIn(dir: string | URL, targets?: readonly string[]): Variant[] {
  const path = toPath(dir);
  if (!existsSync(path)) return [];
  const found = readdirSync(path)
    .filter((file) => file.endsWith(".wgsl"))
    .map((file) => parseVariant(file, targets))
    .sort((a, b) => (a.name < b.name ? -1 : a.name > b.name ? 1 : 0));

  // Portable is the floor for every entry point, not just for the op. Without
  // it an unrecognised device has nothing of that entry point to fall back to,
  // and `resolve` would throw at run time instead of here.
  const portable = new Set(found.filter((v) => !v.target && !v.dtype).map((v) => v.entry));
  for (const variant of found) {
    if (!portable.has(variant.entry)) {
      throw new Error(
        `${variant.name}.wgsl has no portable kernel: expected ${variant.entry}.wgsl beside it`,
      );
    }
  }
  return found;
}

/** Every variant of one entry point, paired with its source. */
export function variantSuites(
  dir: string | URL,
  entry: string = DEFAULT_ENTRY,
  targets?: readonly string[],
): { variant: Variant; code: string }[] {
  const path = toPath(dir);
  const all = variantsIn(path, targets);
  const mine = all.filter((variant) => variant.entry === entry);
  if (mine.length === 0) {
    throw new Error(
      `no entry point ${JSON.stringify(entry)} in ${path}; found: ` +
        `${[...new Set(all.map((v) => v.entry))].sort().join(", ") || "(none)"}`,
    );
  }
  return mine.map((variant) => ({
    variant,
    code: readFileSync(join(path, `${variant.name}.wgsl`), "utf8"),
  }));
}

/** Which rung of the chain produced the choice. */
export type Rung = "override" | "target+dtype" | "target" | "dtype" | "portable";

export interface Choice {
  /** Variant stem to load. Feed to `kernel(url, name)`. */
  name: string;
  rung: Rung;
  /** Every name asked for, in order, ending at the one that hit. */
  tried: string[];
  entry: string;
  target: string | null;
  dtype: string | null;
}

export interface Resolution {
  /** The variants that exist, as names or as parsed variants. */
  have: Iterable<string | Variant>;
  /** Which entry point to resolve within. Resolution never leaves it. */
  entry?: string;
  /** Null when detection could not tell — which is a normal answer, not a failure. */
  target?: string | null;
  dtype?: string | null;
  /**
   * A variant named by the caller, which beats everything.
   *
   * Not an escape hatch. Whoever integrates this eventually knows something the
   * library cannot — that their sequence length is always 1, that they would
   * rather have lower peak memory than higher throughput — and a library that
   * refuses that knowledge becomes something to work around.
   */
  override?: string | null;
}

export function resolve({
  have,
  entry = DEFAULT_ENTRY,
  target = null,
  dtype = null,
  override = null,
}: Resolution): Choice {
  const names = new Set<string>();
  for (const item of have) names.add(typeof item === "string" ? item : item.name);

  if (override != null) {
    // Falling through to the chain on a typo would hand back a kernel the
    // caller did not ask for and never mention it. A wrong choice that is
    // invisible is worse than a portable kernel.
    if (!names.has(override)) {
      throw new Error(
        `override ${JSON.stringify(override)} names no variant; have: ${[...names].sort().join(", ") || "(none)"}`,
      );
    }
    return { name: override, rung: "override", tried: [override], entry, target, dtype };
  }

  // Every rung is prefixed with the entry point, and the last one is the entry
  // point itself. Nothing in this chain can reach another entry's kernel, which
  // is the point: istft quietly running the forward transform because the chain
  // ran off the end would be a wrong answer that still looks like a result.
  const chain: [string, Rung][] = [];
  if (target && dtype) chain.push([`${entry}.${target}.${dtype}`, "target+dtype"]);
  if (target) chain.push([`${entry}.${target}`, "target"]);
  if (dtype) chain.push([`${entry}.${dtype}`, "dtype"]);
  chain.push([entry, "portable"]);

  const tried: string[] = [];
  for (const [name, rung] of chain) {
    tried.push(name);
    if (names.has(name)) return { name, rung, tried, entry, target, dtype };
  }
  throw new Error(
    `no portable ${entry}.wgsl to fall back to; tried: ${tried.join(", ")}; have: ${[...names].sort().join(", ") || "(none)"}`,
  );
}
