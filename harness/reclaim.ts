/**
 * Waiting until the card actually has back what `destroy()` was called on.
 *
 * **`destroy()` does not free the memory; it schedules the freeing.** Dawn
 * releases a destroyed buffer's allocation when it next ticks, and it ticks
 * when it is asked to do GPU work — **not** when a timer fires. A stage that
 * destroys 25 GB and immediately allocates 20 GB gets an *invalid* buffer back,
 * and that does not throw: it fails later at `createBindGroup`, or silently,
 * with a plausible-looking number. Issue #213.
 *
 * Measured on an RTX 5090 (32 GB, Dawn/Vulkan), 25.78 GB destroyed and 20.66 GB
 * asked for, judged by **reading back a value only a live buffer can produce**
 * rather than by an error flag — an out-of-memory `createBuffer` reports
 * asynchronously, and two earlier versions of that measurement read their own
 * staleness instead of the card:
 *
 * | after the destroys | got the 20.66 GB |
 * | --- | --- |
 * | nothing | no, 3 of 3 |
 * | `setTimeout` 0 … 2000 ms | no, at every delay |
 * | one submit-and-readback | **sometimes** — 1 of 3 failed |
 * | two or more | yes, 8 of 8 |
 *
 * `examples/h3-dit/src/generate.ts` ran its two phases as separate *processes*
 * because of this, having measured process exit as the only reliable release.
 * A browser tab has no process to exit, which is what
 * `examples/h3-ref2v-web`'s three stages need.
 *
 * **This file imports nothing at runtime**, because both `ResidentDevice`
 * factories need it and one of them is in a browser bundle:
 * `harness/resident.ts` imports the native `webgpu` addon at module scope and
 * cannot be bundled, which is the whole reason `examples/web-common`'s copy
 * exists. A shared file with type-only imports is what keeps the two from
 * drifting — and that file's own doc records what happened the last time two
 * copies of a device answered the same question differently.
 */
import type { ResidentDevice } from "./resident.js";

/**
 * Round trips `reclaim` submits: twice the measured floor.
 *
 * Two were enough every time and one was not — see the table above. The cost is
 * a 4-byte copy and a map, under a millisecond, so the margin is free and the
 * failure it guards against is an invalid buffer that reads as a number.
 */
export const RECLAIM_ROUND_TRIPS = 4;

/**
 * Submit and read back `RECLAIM_ROUND_TRIPS` times, so Dawn ticks.
 *
 * A copy rather than a dispatch, so it needs no pipeline and no kernel source:
 * what makes Dawn tick is the submission and the map, not the arithmetic.
 */
export async function reclaimByRoundTrips(
  batch: ResidentDevice["batch"],
  createStorageBuffer: (bytes: number, usage?: number) => GPUBuffer,
): Promise<void> {
  const source = createStorageBuffer(4);
  const target = createStorageBuffer(4);
  const staging = createStorageBuffer(4, GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST);
  try {
    for (let i = 0; i < RECLAIM_ROUND_TRIPS; i += 1) {
      await batch(
        [{ kind: "copy", src: source, srcOffset: 0, dst: target, dstOffset: 0, size: 4 }],
        [{ staging, source: target, sourceOffset: 0, length: 1, type: "f32" }],
      );
    }
  } finally {
    source.destroy();
    target.destroy();
    staging.destroy();
  }
}
