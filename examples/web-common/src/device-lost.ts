/**
 * Turning "Instance dropped" back into the reason the device was lost.
 *
 * Issue #211. When a WebGPU device is lost, **every later call fails with the
 * same useless message**: `popErrorScope` rejects with
 * `OperationError: Instance dropped in popErrorScope`, whatever went wrong.
 * That is what a page reports — over and over, once per buffer, with a stack
 * that points at the allocation which happened to be next rather than at the
 * fault.
 *
 * The actual reason is sitting in `device.lost`, a promise that resolves with a
 * `reason` and a `message`. Nothing here read it. So the first failure the
 * `h3-dit-web` page produced said only "Instance dropped", and separating "out
 * of GPU memory" from "a pipeline the adapter refused" took shrinking the model
 * to four blocks until the underlying error fitted in one line.
 *
 * This module is the pure half, so it can be tested without a GPU: the
 * bookkeeping and the wording. `browser-resident.ts` wires it to a real
 * `device.lost`.
 */

export interface DeviceLoss {
  /** `"destroyed"` when the page called `destroy()`, `"unknown"` otherwise. */
  reason: string;
  message: string;
}

/**
 * What a device tracks so a loss can be explained rather than only announced.
 *
 * `allocated` and `buffers` are the two numbers that separate the common causes
 * from each other: a loss after 23 GB across 900 buffers reads very differently
 * from one after 40 MB across three.
 */
export interface LossContext {
  allocated: number;
  buffers: number;
}

/** `true` for the message every call produces once the device is gone. */
export function isInstanceDropped(error: unknown): boolean {
  return /instance dropped/i.test(error instanceof Error ? error.message : String(error));
}

/**
 * The message a call should throw, given what is known about the device.
 *
 * When the device is known to be lost, the original error is **replaced**: it
 * carries no information beyond "later than the fault". When it is not — the
 * loss has not been observed yet, or there was no loss — the original is kept
 * and the context appended, because then it is the fault.
 */
export function explainFailure(error: unknown, loss: DeviceLoss | null, context: LossContext, doing: string): Error {
  const original = error instanceof Error ? error.message : String(error);
  const spent =
    `${(context.allocated / 1e9).toFixed(2)} GB allocated across ${context.buffers} buffers`;

  if (loss) {
    // `reason` is one of two words and neither is a diagnosis; `message` is
    // where a backend says what it actually hit.
    return new Error(
      `the GPU device was lost (${loss.reason}) while ${doing}. ${loss.message || "no message from the backend"} ` +
        `— ${spent} before it went. On a card shared with other tabs this is usually memory: WebGPU has no way ` +
        `to ask how much is free, so the only check is outside the browser.`,
    );
  }
  if (isInstanceDropped(error)) {
    // Lost, but `device.lost` has not resolved yet — which is the ordinary
    // race, since the rejections arrive first.
    return new Error(
      `the GPU device is gone while ${doing}, and it has not yet said why — ${spent}. ` +
        `"Instance dropped" is what every call reports after a loss, not a cause.`,
    );
  }
  return new Error(`${original} (while ${doing}; ${spent})`);
}
