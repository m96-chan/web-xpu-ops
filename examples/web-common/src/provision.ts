/**
 * Filling a folder the user picked, once, so every run after it costs no network.
 *
 * Issue #180. The weights are 5.0 GB. Today they are fetched into the Cache
 * API, which sits in the origin's storage quota, is evicted by rules the page
 * does not control, and cannot be inspected or deleted by the person whose disk
 * it is on. A folder they chose has none of those properties.
 *
 * **The danger here is not failure, it is partial success.** A download that
 * stops at 60% leaves files that open, have plausible sizes, and produce a
 * wrong image rather than an error — the same class as a truncated tensor read.
 *
 * What prevents it is a receipt: `provisioned.json` is written **last**, names
 * every file with its exact length, and is verified against the folder before
 * the folder is used. A folder whose receipt does not match what is in it is
 * treated as unfilled and filled again. That is the cheap direction to be wrong
 * in.
 *
 * **Not a temporary-name-then-rename dance**, which was the first design.
 * `FileSystemFileHandle.move()` is a Chromium extension rather than part of the
 * standard, and the version that called it optionally would have left every
 * file under `<name>.partial` on a browser without it — filling a folder
 * successfully and producing nothing usable, silently. The receipt gives the
 * same guarantee without depending on a method that may not be there.
 *
 * **What the receipt does not catch**: a file that arrived at exactly the right
 * length with the wrong bytes. Only a digest would, and a digest costs a full
 * read of 5 GB every start. Stated rather than papered over.
 */
import type { ByteSource } from "./byte-source.js";

/** What a filled folder must contain. Sizes come from the source, not from here. */
export interface ProvisionPlan {
  files: string[];
}

/**
 * Written last, and read first.
 *
 * `source` is recorded so a folder can say where it came from — a folder filled
 * from one repository and then pointed at another is a confusion worth naming
 * rather than discovering as a shape mismatch fifty blocks in.
 */
export interface Receipt {
  version: 1;
  source: string;
  completedAt: string;
  sizes: Record<string, number>;
}

const RECEIPT = "provisioned.json";
/** 8 MB. Large enough that the per-request overhead disappears, small enough to show progress. */
const CHUNK = 8 * 1024 * 1024;

/**
 * How many chunks are in flight at once.
 *
 * **Measured** against `dit.q8.bin` on Hugging Face, 512 MB each way:
 *
 *     one at a time, 8 MB    3.2 MB/s
 *     eight at once, 8 MB   25.8 MB/s
 *     eight at once, 32 MB  22.2 MB/s
 *
 * A single stream is latency-bound, not bandwidth-bound — the fetch spends its
 * time waiting for the far end rather than reading from it, and eight waits
 * overlap. Bigger chunks do not help on top of that and cost progress
 * granularity, so the chunk stays where it was and only the count changed.
 *
 * Eight rather than more because a CDN is a shared thing and this is a demo,
 * not a benchmark; the win is already 8x.
 */
const IN_FLIGHT = 8;

export interface ProvisionProgress {
  file: string;
  fileIndex: number;
  fileCount: number;
  bytesDone: number;
  bytesTotal: number;
}

/**
 * The receipt in `dir`, or null if there is none, it is unreadable, it was
 * written by a different version, or it does not describe what is actually
 * there.
 *
 * Checking the sizes rather than trusting the receipt is the point: a receipt
 * is evidence that a download finished, not that nothing has touched the folder
 * since.
 */
export async function readReceipt(dir: FileSystemDirectoryHandle): Promise<Receipt | null> {
  let text: string;
  try {
    text = await (await (await dir.getFileHandle(RECEIPT)).getFile()).text();
  } catch {
    return null;
  }
  let receipt: Receipt;
  try {
    receipt = JSON.parse(text) as Receipt;
  } catch {
    return null;
  }
  if (receipt.version !== 1 || typeof receipt.sizes !== "object" || receipt.sizes === null) return null;
  for (const [name, size] of Object.entries(receipt.sizes)) {
    let actual: number;
    try {
      actual = (await (await dir.getFileHandle(name)).getFile()).size;
    } catch {
      return null;
    }
    if (actual !== size) return null;
  }
  return receipt;
}

/**
 * Copies every file in `plan` from `source` into `dir`, then writes the receipt.
 *
 * Resumable only in the coarse sense that a file already present at its full
 * length is skipped. Mid-file resume would mean trusting a length that nothing
 * has vouched for, which is the state this whole module exists to distrust.
 */
export async function provision(
  dir: FileSystemDirectoryHandle,
  source: ByteSource,
  plan: ProvisionPlan,
  onProgress?: (p: ProvisionProgress) => void,
): Promise<Receipt> {
  const sizes: Record<string, number> = {};
  for (const file of plan.files) sizes[file] = await source.size(file);
  const bytesTotal = Object.values(sizes).reduce((a, b) => a + b, 0);

  let bytesDone = 0;
  for (const [index, file] of plan.files.entries()) {
    const want = sizes[file]!;

    // Already there at the right length: skip it. The receipt is what makes a
    // folder usable, so a run interrupted before it was written re-checks
    // rather than re-downloads.
    let existing = 0;
    try {
      existing = (await (await dir.getFileHandle(file)).getFile()).size;
    } catch {
      existing = 0;
    }
    if (existing === want) {
      bytesDone += want;
      onProgress?.({ file, fileIndex: index, fileCount: plan.files.length, bytesDone, bytesTotal });
      continue;
    }

    // Written under its real name. The receipt, not the filename, is what says
    // a file is complete, and a partial file here is caught by the size check
    // above on the next attempt rather than being trusted.
    const handle = await dir.getFileHandle(file, { create: true });
    const writable = await handle.createWritable();
    /** Every read started for this file, so a failure can wait them out. */
    const pendingReads = new Set<Promise<unknown>>();
    try {
      // Fetched ahead, written in order. A `FileSystemWritableFileStream` is a
      // stream and has to be written sequentially, but nothing says the reads
      // have to be sequential — so `IN_FLIGHT` of them are started before the
      // first one is needed, and the write waits on a request that has already
      // had time to finish.
      const spans: { at: number; n: number }[] = [];
      for (let at = 0; at < want; at += CHUNK) spans.push({ at, n: Math.min(CHUNK, want - at) });

      const pending = new Map<number, Promise<ArrayBuffer>>();
      const start = (i: number): void => {
        const span = spans[i];
        if (!span) return;
        // `catch` attached here, not later: a read that rejects while the loop
        // below is awaiting an earlier one would otherwise be an unhandled
        // rejection, which in a browser is a console error nobody asked for and
        // in Node is a process that exits. The rejection is still delivered --
        // to whoever awaits this entry -- because the handler rethrows.
        const request = source.read(file, span.at, span.n);
        pending.set(i, request);
        pendingReads.add(request);
        void request.catch(() => undefined);
      };
      for (let i = 0; i < Math.min(IN_FLIGHT, spans.length); i += 1) start(i);

      for (let i = 0; i < spans.length; i += 1) {
        const chunk = await pending.get(i)!;
        pending.delete(i);
        pendingReads.delete(pending.get(i) as Promise<unknown>);
        // Started only once its predecessor has been written, so the number in
        // flight stays at `IN_FLIGHT` rather than growing to the file's length.
        start(i + IN_FLIGHT);
        await writable.write(chunk);
        bytesDone += spans[i]!.n;
        onProgress?.({ file, fileIndex: index, fileCount: plan.files.length, bytesDone, bytesTotal });
      }
      await writable.close();
    } catch (error) {
      // Everything still in flight is waited out before leaving, or its
      // rejection arrives after this function has returned and belongs to
      // nobody.
      await Promise.allSettled([...pendingReads]);
      // `abort` discards the whole write rather than leaving the target at
      // whatever length it reached. The receipt would reject either state; less
      // left behind is less to explain.
      try {
        await writable.abort();
      } catch {
        // Already closed.
      }
      throw error;
    }

    const written = (await (await dir.getFileHandle(file)).getFile()).size;
    if (written !== want) {
      throw new Error(`provision: wrote ${written} bytes of ${file}, expected ${want}.`);
    }
  }

  const receipt: Receipt = {
    version: 1,
    source: source.describe,
    completedAt: new Date().toISOString(),
    sizes,
  };
  const handle = await dir.getFileHandle(RECEIPT, { create: true });
  const writable = await handle.createWritable();
  await writable.write(JSON.stringify(receipt, null, 2));
  await writable.close();
  return receipt;
}
