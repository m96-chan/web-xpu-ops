import { existsSync, readFileSync, readdirSync } from "node:fs";
import { join } from "node:path";
import { fileURLToPath } from "node:url";
import { DTYPES, TARGETS, type Dtype } from "./target.js";

/**
 * Which kernel file to run, and why.
 *
 * A variant lives under an op's `wgsl/` directory and says in its name what it
 * is specialised for:
 *
 * ```
 * kernel.wgsl        portable — always present, the answer when nothing else fits
 * nvidia.wgsl        a target
 * f16.wgsl           a dtype
 * nvidia.f16.wgsl    both
 * ```
 *
 * Resolution walks one chain, first hit wins:
 *
 * ```
 * explicit override  →  target + dtype  →  target  →  dtype  →  portable
 * ```
 *
 * Nothing here knows what a real target is called. The vocabulary belongs to
 * detection; resolution is name joining and set membership, which is what lets
 * the order be tested against targets that do not exist.
 */

/** The variant every op must have. */
export const PORTABLE = "kernel";

export interface Variant {
  /** File stem — `"kernel"`, `"nvidia"`, `"nvidia.f16"`. Names the `.wgsl` beside it. */
  name: string;
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
 * missing one — `nvidai.wgsl` sits in the tree looking tuned while resolution
 * never once asks for it.
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
  const parts = stem.split(".");
  const [first, second] = parts;

  if (parts.length === 1) {
    if (first === PORTABLE) return { name: stem, target: null, dtype: null };
    if (targets.includes(first!)) return { name: stem, target: first!, dtype: null };
    if (dtypes.includes(first!)) return { name: stem, target: null, dtype: first as Dtype };
    return reject(`${JSON.stringify(first)} is neither a known target nor a known dtype`);
  }
  if (parts.length === 2) {
    if (!targets.includes(first!)) return reject(`${JSON.stringify(first)} is not a known target`);
    if (!dtypes.includes(second!)) return reject(`${JSON.stringify(second)} is not a known dtype`);
    return { name: stem, target: first!, dtype: second as Dtype };
  }
  return reject("expected <target>.<dtype>, <target>, <dtype> or kernel");
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
  return readdirSync(path)
    .filter((file) => file.endsWith(".wgsl"))
    .map((file) => parseVariant(file, targets))
    .sort((a, b) => (a.name < b.name ? -1 : a.name > b.name ? 1 : 0));
}

/** Every variant in a directory, paired with its source. */
export function variantSuites(
  dir: string | URL,
  targets?: readonly string[],
): { variant: Variant; code: string }[] {
  const path = toPath(dir);
  return variantsIn(path, targets).map((variant) => ({
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
  target: string | null;
  dtype: string | null;
}

export interface Resolution {
  /** The variants that exist, as names or as parsed variants. */
  have: Iterable<string | Variant>;
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

export function resolve({ have, target = null, dtype = null, override = null }: Resolution): Choice {
  const names = new Set<string>();
  for (const entry of have) names.add(typeof entry === "string" ? entry : entry.name);

  if (override != null) {
    // Falling through to the chain on a typo would hand back a kernel the
    // caller did not ask for and never mention it. A wrong choice that is
    // invisible is worse than a portable kernel.
    if (!names.has(override)) {
      throw new Error(
        `override ${JSON.stringify(override)} names no variant; have: ${[...names].sort().join(", ") || "(none)"}`,
      );
    }
    return { name: override, rung: "override", tried: [override], target, dtype };
  }

  const chain: [string, Rung][] = [];
  if (target && dtype) chain.push([`${target}.${dtype}`, "target+dtype"]);
  if (target) chain.push([target, "target"]);
  if (dtype) chain.push([dtype, "dtype"]);
  chain.push([PORTABLE, "portable"]);

  const tried: string[] = [];
  for (const [name, rung] of chain) {
    tried.push(name);
    if (names.has(name)) return { name, rung, tried, target, dtype };
  }
  throw new Error(
    `no portable ${PORTABLE}.wgsl to fall back to; tried: ${tried.join(", ")}; have: ${[...names].sort().join(", ") || "(none)"}`,
  );
}
