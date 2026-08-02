import { params, type Runner } from "./wgsl.js";

/**
 * What this device will actually give anyone, measured here and now.
 *
 * Not derived from a spec sheet. WebGPU exposes no clock, no compute-unit count
 * and no bus width, so there is nothing to compute a theoretical peak *from* —
 * and a figure lifted from a vendor page would be worse than none, because it
 * would look authoritative while describing hardware under a different driver
 * than the one in front of the user.
 *
 * Self-calibrating on purpose: if the device is throttling, or another process
 * has the GPU, the ceiling throttles with it and the fraction an op reaches
 * stays honest. That is the only reason a percentage means anything across the
 * wildly different machines a browser library has to run on.
 */
export interface Roofline {
  /** Best sustained bytes per second read from device memory. */
  bandwidth: number;
  /** Best sustained FLOP/s, counting a fused multiply-add as 2. */
  compute: number;
}

/**
 * 512 MiB, swept exactly once.
 *
 * The single sweep is the whole design, and it was arrived at the hard way. An
 * earlier version looped over the buffer many times to make the timing
 * comfortable, and reported **9 TB/s on a 1.79 TB/s card** — the repeated passes
 * were being served without reaching memory. One pass over a buffer far larger
 * than any cache is the only shape that measures what it claims to.
 */
const STREAM_BYTES = 512 * 1024 * 1024;
const STREAM_QUADS = STREAM_BYTES / 16;

const WORKGROUP = 256;
const STREAM_GROUPS = 16384;
/**
 * Deliberately modest. A dispatch large enough to take milliseconds, repeated a
 * few times, kills this binding (issue #68) — so the compute probe is sized to
 * finish in well under one, which is still far above the timer's resolution.
 */
const COMPUTE_GROUPS = 1024;

/**
 * Writes values that cannot be compressed away.
 *
 * An untouched buffer is zeros, and this hardware serves zeroed pages without
 * going to memory. Floats in `[1, 2)` are barely better: they share their
 * exponent bytes and compress nearly as well. Both were measured reporting
 * impossible bandwidth before this existed, so the fill writes a full 32-bit
 * hash per word.
 */
const FILL_WGSL = `
struct Params { words: u32 }
@group(0) @binding(0) var<storage, read_write> data: array<u32>;
@group(0) @binding(1) var<uniform> params: Params;

@compute @workgroup_size(${WORKGROUP})
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) groups: vec3<u32>) {
  let stride = ${WORKGROUP}u * groups.x;
  for (var i = gid.x; i < params.words; i += stride) {
    var h = i * 2654435761u;
    h = h ^ (h >> 15u);
    h = h * 2246822519u;
    h = h ^ (h >> 13u);
    data[i] = h;
  }
}
`;

/**
 * Reads memory and does as little as possible with it.
 *
 * `vec4<u32>` because the widest load the hardware offers is how it reaches its
 * ceiling: measured on a filled buffer at 94-95% of this card's rated 1792 GB/s,
 * against 76% for scalar loads. A roofline is meant to be the best the device
 * will give, so the wider load is the honest one to calibrate against.
 *
 * Consecutive lanes read consecutive addresses so the hardware can coalesce
 * them. The total is written under a condition that never holds — the compiler
 * cannot know it never holds, so the loads survive, but nothing is stored, which
 * keeps write traffic out of a read measurement.
 */
const STREAM_WGSL = `
struct Params { quads: u32, guard: u32 }
@group(0) @binding(0) var<storage, read> data: array<vec4<u32>>;
@group(0) @binding(1) var<storage, read_write> sink: array<u32>;
@group(0) @binding(2) var<uniform> params: Params;

@compute @workgroup_size(${WORKGROUP})
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) groups: vec3<u32>) {
  let stride = ${WORKGROUP}u * groups.x;
  var total = vec4<u32>(0u);
  for (var i = gid.x; i < params.quads; i += stride) {
    total += data[i];
  }
  let sum = total.x + total.y + total.z + total.w;
  if (sum == params.guard) { sink[gid.x % ${WORKGROUP}u] = sum; }
}
`;

/**
 * Fused multiply-adds out of registers, touching no memory at all.
 *
 * Four independent chains, because one dependent chain measures the latency of a
 * single FMA rather than throughput — each result would stall the next. The seed
 * arrives in a uniform so the loop cannot be folded to a constant.
 */
const COMPUTE_WGSL = `
struct Params { iterations: u32, seed: f32 }
@group(0) @binding(0) var<storage, read_write> sink: array<f32>;
@group(0) @binding(1) var<uniform> params: Params;

@compute @workgroup_size(${WORKGROUP})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  var a = params.seed;
  var b = a + 1.0;
  var c = a + 2.0;
  var d = a + 3.0;
  let k = params.seed * 0.5 + 1.0;
  for (var i = 0u; i < params.iterations; i += 1u) {
    a = fma(a, k, 1.0);
    b = fma(b, k, 1.0);
    c = fma(c, k, 1.0);
    d = fma(d, k, 1.0);
  }
  if (a + b + c + d == params.seed - 999.0) { sink[gid.x % ${WORKGROUP}u] = a; }
}
`;

const FMAS_PER_ITERATION = 4;
const COMPUTE_ITERATIONS = 8_000;
/** A guard value the accumulated sum will not take. */
const NEVER = 4042322160;

/**
 * Fastest of several attempts, after one untimed warm-up.
 *
 * A ceiling is a best case, so the minimum is the right statistic: scheduling
 * noise and contention can only make a run slower, never faster than the device
 * can go.
 */
async function fastest(runner: Runner, dispatch: Parameters<Runner["time"]>[0]): Promise<number | null> {
  const first = await runner.time(dispatch);
  if (first === null) return null;
  let best = Infinity;
  for (let attempt = 0; attempt < 2; attempt += 1) {
    const seconds = await runner.time(dispatch);
    if (seconds !== null && seconds > 0) best = Math.min(best, seconds);
  }
  return Number.isFinite(best) ? best : null;
}

/**
 * Measures both ceilings, or null where the device cannot time a dispatch.
 *
 * Null rather than an estimate: rule 9 says an unmeasured figure must say so,
 * and the devices least able to report timing are exactly the ones where a
 * fabricated ceiling would mislead most.
 */
export async function measureRoofline(runner: Runner): Promise<Roofline | null> {
  const stream = { kind: "scratch", length: STREAM_BYTES / 4 } as const;

  // Fill before reading. Untimed: it is setup, not measurement.
  await runner.run({
    code: FILL_WGSL,
    bindings: [stream, { kind: "uniform", data: params([["u32", STREAM_BYTES / 4]]) }],
    workgroups: [STREAM_GROUPS],
  });

  const streamSeconds = await fastest(runner, {
    code: STREAM_WGSL,
    bindings: [
      stream,
      { kind: "out", type: "u32", length: WORKGROUP },
      { kind: "uniform", data: params([["u32", STREAM_QUADS], ["u32", NEVER]]) },
    ],
    workgroups: [STREAM_GROUPS],
  });
  if (streamSeconds === null) return null;

  const threads = WORKGROUP * COMPUTE_GROUPS;
  const computeSeconds = await fastest(runner, {
    code: COMPUTE_WGSL,
    bindings: [
      { kind: "out", type: "f32", length: WORKGROUP },
      { kind: "uniform", data: params([["u32", COMPUTE_ITERATIONS], ["f32", 1.0001]]) },
    ],
    workgroups: [COMPUTE_GROUPS],
  });
  if (computeSeconds === null) return null;

  return {
    bandwidth: STREAM_BYTES / streamSeconds,
    compute: (threads * COMPUTE_ITERATIONS * FMAS_PER_ITERATION * 2) / computeSeconds,
  };
}

const CACHE = Symbol.for("web-xpu-ops.roofline");
type Holder = { [CACHE]?: Promise<Roofline | null> };

/**
 * The ceiling for this session, measured once.
 *
 * Cached because calibration costs the better part of a second and cannot change
 * within a run. Held on `globalThis` for the same reason the device is: vitest
 * gives each test file a fresh module registry, so a module-level cache would
 * re-measure per file.
 *
 * **This describes one device in one session.** Comparing it with a figure from
 * another machine, or another run in a different thermal or driver state, is
 * comparing two different denominators — which is exactly why ops are reported
 * as a fraction of it rather than in absolute units.
 */
export function roofline(runner: Runner): Promise<Roofline | null> {
  const holder = globalThis as Holder;
  holder[CACHE] ??= measureRoofline(runner);
  return holder[CACHE];
}
