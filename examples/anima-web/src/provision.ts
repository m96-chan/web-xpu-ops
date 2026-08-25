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

/** How many bytes a file has, from the source, via a `Range` for one byte. */
async function sizeOf(source: ByteSource, file: string): Promise<number> {
  // `HttpByteSource` throws on anything but 206, so this doubles as a check
  // that the host serves ranges at all — better here, on a 1-byte request,
  // than three gigabytes into the first download.
  const url = `${source.describe}/${file}`;
  const response = await fetch(url, { headers: { Range: "bytes=0-0" } });
  if (response.status !== 206) {
    throw new Error(
      `provision: ${url} answered ${response.status} for a Range request, so its length cannot be read ` +
        "without downloading it whole.",
    );
  }
  const range = response.headers.get("content-range");
  const total = range?.match(/\/(\d+)\s*$/)?.[1];
  if (!total) throw new Error(`provision: ${url} gave no usable content-range (${range ?? "absent"}).`);
  await response.arrayBuffer();
  return Number(total);
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
  for (const file of plan.files) sizes[file] = await sizeOf(source, file);
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
    try {
      for (let at = 0; at < want; at += CHUNK) {
        const n = Math.min(CHUNK, want - at);
        await writable.write(await source.read(file, at, n));
        bytesDone += n;
        onProgress?.({ file, fileIndex: index, fileCount: plan.files.length, bytesDone, bytesTotal });
      }
      await writable.close();
    } catch (error) {
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
