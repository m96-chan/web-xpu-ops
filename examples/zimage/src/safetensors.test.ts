import { mkdtempSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { describe, expect, it } from "vitest";
import { bf16ToF32 } from "./bf16.js";
import { SafetensorsFile } from "./safetensors.js";

/**
 * The reader, against files this test writes.
 *
 * Correctness for the *format* is the specification plus the checkpoints it is
 * pointed at; what is pinned here is that this implementation reads back what
 * the format says is there — including the two cases that are silent when got
 * wrong: bf16's truncation and the data offset being relative to the end of
 * the header rather than the start of the file.
 */

function writeSafetensors(
  dir: string,
  tensors: { name: string; dtype: string; shape: number[]; bytes: Uint8Array }[],
): string {
  const header: Record<string, unknown> = {};
  let at = 0;
  const chunks: Uint8Array[] = [];
  for (const t of tensors) {
    header[t.name] = { dtype: t.dtype, shape: t.shape, data_offsets: [at, at + t.bytes.length] };
    chunks.push(t.bytes);
    at += t.bytes.length;
  }
  const headerText = new TextEncoder().encode(JSON.stringify(header));
  const out = new Uint8Array(8 + headerText.length + at);
  new DataView(out.buffer).setBigUint64(0, BigInt(headerText.length), true);
  out.set(headerText, 8);
  let cursor = 8 + headerText.length;
  for (const chunk of chunks) {
    out.set(chunk, cursor);
    cursor += chunk.length;
  }
  const path = join(dir, "model.safetensors");
  writeFileSync(path, out);
  return path;
}

/** f32 bits to the bf16 the checkpoint stores: the top 16 bits, truncated. */
function f32ToBf16(values: number[]): Uint8Array {
  const out = new Uint16Array(values.length);
  const scratch = new DataView(new ArrayBuffer(4));
  values.forEach((v, i) => {
    scratch.setFloat32(0, v, true);
    out[i] = scratch.getUint16(2, true);
  });
  return new Uint8Array(out.buffer);
}

describe("bf16ToF32", () => {
  it("puts the stored bits in the high half of the float", () => {
    // bf16 *is* f32's top 16 bits, so every bf16 value is exactly representable
    // and this is an equality, not an approximation. A reader that treated the
    // bits as f16 would produce numbers of a wildly different magnitude.
    const values = [1, -1, 0.5, 3.5, -0.0078125, 0];
    const packed = new Uint16Array(f32ToBf16(values).buffer);
    const got = bf16ToF32(packed);
    expect([...got]).toEqual(values);
  });

  it("keeps the sign of a negative that rounds toward zero", () => {
    // Truncation, not rounding: 1.1 in bf16 is slightly below 1.1, and -1.1 is
    // slightly *above* -1.1. A reader that dropped the sign bit or shifted
    // wrong would still give something near 1.1 in magnitude.
    const packed = new Uint16Array(f32ToBf16([-1.1]).buffer);
    const got = bf16ToF32(packed);
    expect(got[0]).toBeLessThan(0);
    expect(Math.abs(got[0]! - -1.1)).toBeLessThan(0.01);
  });
});

describe("SafetensorsFile", () => {
  const dir = mkdtempSync(join(tmpdir(), "safetensors-"));
  const f32Bytes = new Uint8Array(Float32Array.from([1, 2, 3, 4, 5, 6]).buffer);
  const bf16Bytes = f32ToBf16([0.5, -0.5, 2, -2]);
  const path = writeSafetensors(dir, [
    { name: "a.weight", dtype: "F32", shape: [2, 3], bytes: f32Bytes },
    { name: "b.weight", dtype: "BF16", shape: [2, 2], bytes: bf16Bytes },
  ]);

  it("lists the tensors the header declares", () => {
    const file = new SafetensorsFile(path);
    expect(file.names().sort()).toEqual(["a.weight", "b.weight"]);
    expect(file.shape("a.weight")).toEqual([2, 3]);
    file.close();
  });

  it("reads an f32 tensor", () => {
    const file = new SafetensorsFile(path);
    expect([...file.read("a.weight")]).toEqual([1, 2, 3, 4, 5, 6]);
    file.close();
  });

  it("reads a bf16 tensor as f32", () => {
    // The second tensor, so its data offset is non-zero — which is the case a
    // reader that measured from the start of the file instead of the end of
    // the header gets wrong, and only for tensors after the first.
    const file = new SafetensorsFile(path);
    expect([...file.read("b.weight")]).toEqual([0.5, -0.5, 2, -2]);
    file.close();
  });

  it("names a tensor it does not have", () => {
    const file = new SafetensorsFile(path);
    expect(() => file.read("c.weight")).toThrow(/c\.weight/);
    file.close();
  });

  it("refuses a dtype it does not implement", () => {
    // Reading F16 bytes as BF16 gives numbers, all wrong. Refused rather than
    // guessed at.
    const other = writeSafetensors(mkdtempSync(join(tmpdir(), "safetensors-")), [
      { name: "half.weight", dtype: "F16", shape: [2], bytes: new Uint8Array(4) },
    ]);
    const file = new SafetensorsFile(other);
    expect(() => file.read("half.weight")).toThrow(/F16/);
    file.close();
  });

  it("refuses a tensor whose byte range does not match its shape", () => {
    // A truncated download, or a header that disagrees with the payload. Both
    // otherwise read as a short tensor padded with whatever follows.
    const bad = join(mkdtempSync(join(tmpdir(), "safetensors-")), "model.safetensors");
    const headerText = new TextEncoder().encode(
      JSON.stringify({ "x.weight": { dtype: "F32", shape: [4], data_offsets: [0, 8] } }),
    );
    const out = new Uint8Array(8 + headerText.length + 8);
    new DataView(out.buffer).setBigUint64(0, BigInt(headerText.length), true);
    out.set(headerText, 8);
    writeFileSync(bad, out);
    const file = new SafetensorsFile(bad);
    expect(() => file.read("x.weight")).toThrow(/16 bytes|elements/i);
    file.close();
  });
});
