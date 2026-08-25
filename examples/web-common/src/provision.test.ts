/**
 * A folder is usable only when something says it is finished.
 *
 * Issue #180. Filling a folder with 5.0 GB can stop at 60%, and what it leaves
 * behind opens, has a plausible size, and produces a **wrong image rather than
 * an error**. Nothing about a file on disk distinguishes "downloaded" from
 * "downloaded so far", so `provision` writes a receipt last and `readReceipt`
 * verifies it against what is actually there.
 *
 * These run in Node against fakes, because the thing under test is that
 * arithmetic and not the File System Access API. The fake is deliberately
 * dumber than a real handle: it will happily hand back a truncated file, which
 * is the case that matters.
 */
import { describe, expect, it } from "vitest";
import { provision, readReceipt, type Receipt } from "./provision.js";
import type { ByteSource } from "./byte-source.js";

/**
 * Enough of `FileSystemDirectoryHandle` for `provision` and `readReceipt`.
 *
 * `maxBytes` makes it write **less than it was given** and say nothing, which
 * is what a real filesystem does when it runs out of room part-way through a
 * 3.5 GB file. The first version of this fake could not do that, and the
 * post-write length check in `provision` survived every mutation as a result —
 * a guard nothing exercises is a guard nobody can claim works.
 */
function fakeDir(initial: Record<string, Uint8Array> = {}, maxBytes = Infinity) {
  const files = new Map<string, Uint8Array>(Object.entries(initial));
  const handle = {
    name: "fake",
    files,
    async getFileHandle(name: string, options?: { create?: boolean }) {
      if (!files.has(name)) {
        if (!options?.create) throw new Error(`NotFoundError: ${name}`);
        files.set(name, new Uint8Array(0));
      }
      return {
        async getFile() {
          const bytes = files.get(name)!;
          return {
            size: bytes.byteLength,
            async text() {
              return new TextDecoder().decode(bytes);
            },
          };
        },
        async createWritable() {
          const parts: Uint8Array[] = [];
          return {
            async write(chunk: ArrayBuffer | string) {
              parts.push(
                typeof chunk === "string" ? new TextEncoder().encode(chunk) : new Uint8Array(chunk),
              );
            },
            async close() {
              const asked = parts.reduce((n, p) => n + p.byteLength, 0);
              const total = Math.min(asked, maxBytes);
              const out = new Uint8Array(total);
              let at = 0;
              for (const p of parts) {
                if (at >= total) break;
                out.set(p.subarray(0, Math.min(p.byteLength, total - at)), at);
                at += p.byteLength;
              }
              files.set(name, out);
            },
            async abort() {
              // Discards: the target keeps whatever it had, which for a fresh
              // create is zero bytes.
            },
          };
        },
      };
    },
  };
  return handle as unknown as FileSystemDirectoryHandle & { files: Map<string, Uint8Array> };
}

/** A source of `sizes[file]` bytes, each byte its own index modulo 251. */
function fakeSource(sizes: Record<string, number>, failAt?: { file: string; after: number }): ByteSource {
  let served = 0;
  return {
    describe: "fake://weights",
    async size(file) {
      const n = sizes[file];
      if (n === undefined) throw new Error(`no such file ${file}`);
      return n;
    },
    async read(file, offset, length) {
      if (failAt && file === failAt.file) {
        served += length;
        if (served > failAt.after) throw new Error("network went away");
      }
      const out = new Uint8Array(length);
      for (let i = 0; i < length; i += 1) out[i] = (offset + i) % 251;
      return out.buffer;
    },
  };
}

const SIZES = { "a.bin": 20 * 1024 * 1024 + 7, "b.bin": 5 };

describe("anima-web / provisioning a folder", () => {
  it("writes every file and a receipt that matches them", async () => {
    const dir = fakeDir();
    const receipt = await provision(dir, fakeSource(SIZES), { files: Object.keys(SIZES) });
    expect(receipt.sizes).toEqual(SIZES);
    expect(dir.files.get("a.bin")!.byteLength).toBe(SIZES["a.bin"]);
    // The chunking must not lose or duplicate a byte across its 8 MB boundaries.
    const a = dir.files.get("a.bin")!;
    for (const at of [0, 1, 8 * 1024 * 1024 - 1, 8 * 1024 * 1024, SIZES["a.bin"] - 1]) {
      expect(a[at], `byte ${at}`).toBe(at % 251);
    }
    expect(await readReceipt(dir)).toEqual(receipt);
  });

  it("a folder with no receipt is not usable", async () => {
    const dir = fakeDir({ "a.bin": new Uint8Array(SIZES["a.bin"]) });
    expect(await readReceipt(dir)).toBeNull();
  });

  it("a receipt that disagrees with the folder is not usable", async () => {
    // The case this module exists for: a file that was replaced or truncated
    // after the receipt was written. Its name and its receipt entry both look
    // right; only its length does not.
    const dir = fakeDir();
    await provision(dir, fakeSource(SIZES), { files: Object.keys(SIZES) });
    expect(await readReceipt(dir)).not.toBeNull();
    dir.files.set("b.bin", new Uint8Array(4));
    expect(await readReceipt(dir)).toBeNull();
  });

  it("a receipt naming a file that is gone is not usable", async () => {
    const dir = fakeDir();
    await provision(dir, fakeSource(SIZES), { files: Object.keys(SIZES) });
    dir.files.delete("a.bin");
    expect(await readReceipt(dir)).toBeNull();
  });

  it("an interrupted download leaves no receipt, so the folder stays unusable", async () => {
    const dir = fakeDir();
    await expect(
      provision(dir, fakeSource(SIZES, { file: "a.bin", after: 8 * 1024 * 1024 }), {
        files: Object.keys(SIZES),
      }),
    ).rejects.toThrow(/network went away/);
    expect(await readReceipt(dir)).toBeNull();
  });

  it("re-running skips a file that is already the right length", async () => {
    const dir = fakeDir();
    await provision(dir, fakeSource(SIZES), { files: Object.keys(SIZES) });
    let reads = 0;
    const counting: ByteSource = {
      describe: "fake://weights",
      async size(file) {
        return SIZES[file as keyof typeof SIZES];
      },
      async read(file, offset, length) {
        reads += 1;
        void file;
        void offset;
        return new ArrayBuffer(length);
      },
    };
    await provision(dir, counting, { files: Object.keys(SIZES) });
    expect(reads, "a complete folder should be re-read zero times").toBe(0);
  });

  it("a short write is an error, not a shorter file", async () => {
    // Disk full part-way through. The write reports no failure and the file
    // exists; only its length says anything. Without the check this test
    // covers, `provision` would write a receipt claiming the full size and the
    // next run would trust the folder.
    const dir = fakeDir({}, 1024);
    await expect(
      provision(dir, fakeSource(SIZES), { files: Object.keys(SIZES) }),
    ).rejects.toThrow(/wrote 1024 bytes of a\.bin, expected/);
    expect(await readReceipt(dir)).toBeNull();
  });

  it("rejects a receipt from a future version rather than guessing at it", async () => {
    const dir = fakeDir();
    const receipt: Receipt = { version: 2 as 1, source: "x", completedAt: "", sizes: {} };
    const handle = await dir.getFileHandle("provisioned.json", { create: true });
    const writable = await handle.createWritable();
    await writable.write(JSON.stringify(receipt));
    await writable.close();
    expect(await readReceipt(dir)).toBeNull();
  });
});
