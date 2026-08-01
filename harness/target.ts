/**
 * Which machine a kernel was written for, guessed from `adapter.info`.
 *
 * The guess is a **hint**. `adapter.info` gives a vendor string, an
 * architecture string and a driver blurb, and none of those answer the question
 * a kernel actually cares about — subgroup width, whether memory is unified,
 * what the power budget is. So detection is allowed to answer "I don't know",
 * and an unknown adapter gets the portable kernel rather than someone else's.
 *
 * That is also why an explicit override exists: the hint will sometimes be
 * wrong, and the caller is often the one who knows it.
 */

/** The targets a kernel may be tuned for. Portable is the absence of one. */
export const TARGETS = ["apple", "nvidia", "amd", "soc"] as const;
export type Target = (typeof TARGETS)[number];

/** The dtypes a variant may be specialised for. `f32` is the floor. */
export const DTYPES = ["f32", "f16", "i32", "u32"] as const;
export type Dtype = (typeof DTYPES)[number];

/** The parts of `GPUAdapterInfo` detection is allowed to look at. */
export interface AdapterInfoLike {
  readonly vendor: string;
  readonly architecture: string;
  readonly device: string;
  readonly description: string;
  readonly isFallbackAdapter: boolean;
  readonly subgroupMinSize?: number | undefined;
  readonly subgroupMaxSize?: number | undefined;
}

export interface Detected {
  target: Target | null;
  /** How it got there, in words. A guess nobody can read is a guess nobody can correct. */
  why: string;
}

/**
 * Vendor tokens that name a target, and the ones that deliberately do not.
 *
 * Matched as whole tokens of the lowercased vendor string, so "AMD Radeon"
 * works and a substring collision does not. `description` is never consulted:
 * the WebGPU spec says not to parse it, and drivers put version numbers and
 * other vendors' names in it.
 */
const VENDOR_TOKENS: Record<string, Target> = {
  nvidia: "nvidia",
  amd: "amd",
  ati: "amd",
  radeon: "amd",
  apple: "apple",
  qualcomm: "soc",
  adreno: "soc",
  arm: "soc",
  mali: "soc",
  imagination: "soc",
  powervr: "soc",
  samsung: "soc",
};

/**
 * Vendors whose name covers hardware too different to share a kernel.
 *
 * Intel ships an on-die iGPU with a few watts of budget and a discrete Arc card
 * with its own memory under the same vendor string, and `adapter.info` does not
 * distinguish them. Picking either would be wrong about half the time, and
 * wrong invisibly.
 */
const AMBIGUOUS: Record<string, string> = {
  intel: "intel spans an on-die iGPU and a discrete Arc card, and adapter.info does not say which",
};

export function detectTarget(info: AdapterInfoLike): Detected {
  // A software fallback runs none of the silicon its vendor string names, so a
  // kernel tuned for that silicon is exactly the wrong choice.
  if (info.isFallbackAdapter) {
    return { target: null, why: "fallback adapter: the vendor string is not the hardware in use" };
  }
  const vendor = info.vendor.toLowerCase();
  if (vendor === "") return { target: null, why: "the adapter reported no vendor" };

  const tokens = vendor.split(/[^a-z0-9]+/).filter(Boolean);
  for (const token of tokens) {
    const target = VENDOR_TOKENS[token];
    if (target) return { target, why: `vendor ${JSON.stringify(info.vendor)} names ${target}` };
  }
  for (const token of tokens) {
    const reason = AMBIGUOUS[token];
    if (reason) return { target: null, why: `no target: ${reason}` };
  }
  return { target: null, why: `vendor ${JSON.stringify(info.vendor)} matches no known target` };
}

export interface AdapterReport {
  /** `adapter.info`, copied out. See `describeAdapter` for why it is copied. */
  info: {
    vendor: string;
    architecture: string;
    device: string;
    description: string;
    isFallbackAdapter: boolean;
  };
  target: Target | null;
  why: string;
  features: { f16: boolean; subgroups: boolean };
  /** Null unless the adapter both has subgroups and publishes a size. */
  subgroupSize: { min: number; max: number } | null;
}

/** Structural so a fake adapter can be handed to it; a real `GPUAdapter` fits. */
export interface AdapterLike {
  readonly info: AdapterInfoLike;
  readonly features: ReadonlySet<string>;
}

/**
 * Everything about an adapter that a kernel choice can be argued from.
 *
 * The fields are copied one by one rather than spread. `GPUAdapterInfo` exposes
 * them as prototype getters — measured on Dawn, `Object.keys(adapter.info)`
 * returns `["subgroupMatrixConfigs"]` and nothing else — so `{ ...info }` and
 * `JSON.stringify(info)` both come back empty.
 */
export function describeAdapter(adapter: AdapterLike): AdapterReport {
  const { info } = adapter;
  const subgroups = adapter.features.has("subgroups");
  const min = info.subgroupMinSize;
  const max = info.subgroupMaxSize;
  return {
    info: {
      vendor: info.vendor,
      architecture: info.architecture,
      device: info.device,
      description: info.description,
      isFallbackAdapter: info.isFallbackAdapter,
    },
    ...detectTarget(info),
    features: { f16: adapter.features.has("shader-f16"), subgroups },
    subgroupSize:
      subgroups && typeof min === "number" && typeof max === "number" ? { min, max } : null,
  };
}
