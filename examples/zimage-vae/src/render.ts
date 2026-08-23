/**
 * Writes a decoded `[1, 3, H, W]` map out as a PNG, with no dependencies.
 *
 * The demo's whole point is that a person can look at the result, so the output
 * has to leave the process as an image rather than as a number. Node has no
 * image encoder built in and this repository has no runtime dependencies, so
 * the PNG is assembled here: it is a container format, and a stored (level 0)
 * deflate stream plus CRCs is about sixty lines. That is cheaper than taking a
 * dependency for one file, and it keeps `npm install` honest.
 *
 * Stored deflate means the file is roughly `W * H * 3` bytes with no
 * compression. For a 256x256 preview that is 196 KB, which is fine for looking
 * at and wrong for shipping — nothing here ships.
 */

function crc32(bytes: Uint8Array): number {
  let crc = ~0;
  for (const byte of bytes) {
    crc ^= byte;
    for (let bit = 0; bit < 8; bit += 1) crc = (crc >>> 1) ^ (0xedb88320 & -(crc & 1));
  }
  return ~crc >>> 0;
}

function adler32(bytes: Uint8Array): number {
  let a = 1;
  let b = 0;
  for (const byte of bytes) {
    a = (a + byte) % 65521;
    b = (b + a) % 65521;
  }
  return ((b << 16) | a) >>> 0;
}

function chunk(type: string, body: Uint8Array): Uint8Array {
  const out = new Uint8Array(12 + body.length);
  const view = new DataView(out.buffer);
  view.setUint32(0, body.length);
  for (let i = 0; i < 4; i += 1) out[4 + i] = type.charCodeAt(i);
  out.set(body, 8);
  view.setUint32(8 + body.length, crc32(out.subarray(4, 8 + body.length)));
  return out;
}

/**
 * `[3, H, W]` in `[-1, 1]` to PNG bytes.
 *
 * Values outside the range are clamped rather than wrapped — a decoder that
 * overshoots should look like an overexposed image, not like noise, so the
 * failure stays readable.
 */
export function encodePng(planes: Float32Array, H: number, W: number): Uint8Array {
  const hw = H * W;
  // PNG scanlines carry a filter byte each; 0 means "no filter".
  const raw = new Uint8Array(H * (1 + W * 3));
  for (let y = 0; y < H; y += 1) {
    const row = y * (1 + W * 3);
    raw[row] = 0;
    for (let x = 0; x < W; x += 1) {
      for (let c = 0; c < 3; c += 1) {
        const v = planes[c * hw + y * W + x]!;
        raw[row + 1 + x * 3 + c] = Math.max(0, Math.min(255, Math.round((Math.max(-1, Math.min(1, v)) + 1) * 127.5)));
      }
    }
  }

  // zlib: 2-byte header, stored deflate blocks (max 65535 bytes each), adler32.
  const blocks: Uint8Array[] = [new Uint8Array([0x78, 0x01])];
  for (let off = 0; off < raw.length; off += 65535) {
    const len = Math.min(65535, raw.length - off);
    const header = new Uint8Array(5);
    header[0] = off + len >= raw.length ? 1 : 0;
    header[1] = len & 0xff;
    header[2] = len >>> 8;
    header[3] = ~len & 0xff;
    header[4] = (~len >>> 8) & 0xff;
    blocks.push(header, raw.subarray(off, off + len));
  }
  const tail = new Uint8Array(4);
  new DataView(tail.buffer).setUint32(0, adler32(raw));
  blocks.push(tail);

  let zlibLen = 0;
  for (const b of blocks) zlibLen += b.length;
  const zlib = new Uint8Array(zlibLen);
  let at = 0;
  for (const b of blocks) {
    zlib.set(b, at);
    at += b.length;
  }

  const ihdr = new Uint8Array(13);
  const ihdrView = new DataView(ihdr.buffer);
  ihdrView.setUint32(0, W);
  ihdrView.setUint32(4, H);
  ihdr[8] = 8; // bit depth
  ihdr[9] = 2; // truecolour
  const parts = [
    new Uint8Array([0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a]),
    chunk("IHDR", ihdr),
    chunk("IDAT", zlib),
    chunk("IEND", new Uint8Array(0)),
  ];
  let total = 0;
  for (const p of parts) total += p.length;
  const png = new Uint8Array(total);
  let cursor = 0;
  for (const p of parts) {
    png.set(p, cursor);
    cursor += p.length;
  }
  return png;
}
