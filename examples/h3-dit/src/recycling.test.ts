/**
 * That a buffer is only handed back once nothing is going to read it again.
 *
 * Issue #223. The attention path wrote eleven buffers at `heads * head_dim`
 * wide and kept every one of them until the block's submit — 6.0 GB of the
 * 31.3 GB that filled a 32 GB card at 19,027 rows. Almost all of them are
 * written once, read once and then dead, so `recycle` hands each back the
 * moment its last reader has been recorded.
 *
 * **This is the kind of change that is silently wrong.** Recycling a buffer one
 * step too early gives a later dispatch the same memory a pending one still
 * reads: the shapes stay right, nothing errors, and the numbers move a little.
 * The golden comparison catches it — `verify-ref2va-forward.ts` against the
 * real-content golden is bit-identical either way, worst 1.394e+0 — but that
 * needs a GPU and 20 GB of weights, so it is a script, and a script is the
 * thing nobody runs.
 *
 * So the ordering is asserted here instead, against the source. Weaker than
 * executing it and considerably stronger than nothing, and it is what stops a
 * later edit from moving a `recycle` above the call that reads the buffer.
 */
import { describe, expect, it } from "vitest";
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";

const source = readFileSync(fileURLToPath(new URL("./model-gpu.ts", import.meta.url)), "utf8");
/** Comments stripped: several of them name the very calls being ordered. */
const code = source.replace(/\/\*[\s\S]*?\*\//g, "").replace(/\/\/.*$/gm, "");

/** The body of one method, so an ordering assertion cannot drift into another. */
function method(name: string): string {
  const at = code.indexOf(`private async ${name}(`);
  expect(at, `${name} is not in model-gpu.ts`).toBeGreaterThan(-1);
  const rest = code.slice(at);
  const end = rest.indexOf("\n  }\n");
  expect(end, `${name} has no end`).toBeGreaterThan(-1);
  return rest.slice(0, end);
}

describe("h3 dit / recycling an intermediate", () => {
  it("gives q, k and v back only after the head swaps have read them", () => {
    // The three swaps are the last readers. Recycling above them would hand a
    // later dispatch the memory `swapLeading` is still reading, which is a
    // wrong number rather than an error.
    const body = method("attention");
    for (const name of ["q", "k", "v"]) {
      const swap = body.indexOf(`const ${name}h = await this.swapLeading(ops, ${name}.buffer`);
      const back = body.indexOf(`this.recycle(${name}.buffer)`);
      expect(swap, `${name} is not swapped`).toBeGreaterThan(-1);
      expect(back, `${name} is never given back`).toBeGreaterThan(-1);
      expect(back, `${name} is recycled before its swap reads it`).toBeGreaterThan(swap);
    }
  });

  it("gives the head-swapped copies back only after the attention dispatch", () => {
    const body = method("attention");
    const dispatch = body.indexOf("this.kernels.flashAttention");
    expect(dispatch).toBeGreaterThan(-1);
    for (const name of ["qh", "kh", "vh"]) {
      const back = body.indexOf(`this.recycle(${name})`);
      expect(back, `${name} is never given back`).toBeGreaterThan(-1);
      expect(back, `${name} is recycled before the attention reads it`).toBeGreaterThan(dispatch);
    }
  });

  it("gives the attended rows back only after the merge has read them", () => {
    const body = method("attention");
    const merge = body.indexOf("const merged = await this.swapLeading(ops, attended");
    const back = body.indexOf("this.recycle(attended)");
    expect(merge).toBeGreaterThan(-1);
    expect(back).toBeGreaterThan(-1);
    expect(back, "attended is recycled before the merge reads it").toBeGreaterThan(merge);
  });

  it("never gives back what it is about to return", () => {
    // `merged` is the method's result. Handing it to the pool would let the
    // caller's next `take` return the same memory and overwrite the answer.
    const body = method("attention");
    expect(body).not.toContain("this.recycle(merged)");
  });

  it("recycles through `consume`, which cannot free a buffer it also returns", () => {
    // `qkNorm` and `rope` are `x = consume(x, f(x))`. If either ever returns
    // its input -- an in-place version, which is the obvious next optimisation
    // -- freeing the old handle would free the live one. `consume` compares
    // them, so that change stays safe.
    expect(code).toMatch(
      /private consume\(dead: Mat, made: Mat\): Mat \{\s*if \(dead\.buffer !== made\.buffer\) this\.recycle\(dead\.buffer\);/,
    );
    const body = method("block");
    expect(body).toContain("q = this.consume(q, await this.qkNorm(");
    expect(body).toContain("k = this.consume(k, await this.rope(");
  });

  it("gives the feed-forward's widest buffers back as each is finished with", () => {
    // Four at `ffn_dim` -- value, gate, activated, gated -- and at 19,027 rows
    // each is 1,095 MB. Once the attention path stopped being the wall, these
    // were.
    const body = method("feedForward");
    const activation = body.indexOf("this.kernels.activation");
    const multiply = body.indexOf("ELEMENTWISE.multiply");
    const last = body.indexOf("ff.net.2.weight");
    expect(activation).toBeGreaterThan(-1);
    expect(multiply).toBeGreaterThan(activation);
    expect(last).toBeGreaterThan(multiply);
    expect(body.indexOf("this.recycle(gate.buffer)"), "gate before its activation")
      .toBeGreaterThan(activation);
    expect(body.indexOf("this.recycle(value.buffer)"), "value before the multiply")
      .toBeGreaterThan(multiply);
    expect(body.indexOf("this.recycle(activated)"), "activated before the multiply")
      .toBeGreaterThan(multiply);
    expect(body.indexOf("this.recycle(gated.buffer)"), "gated before the last projection")
      .toBeGreaterThan(last);
  });

  it("keeps the residual it is about to return", () => {
    // `hidden` is read by the add that produces the block's result. Recycling
    // it would hand the pool the buffer the answer is being written from.
    const body = method("block");
    expect(body).not.toMatch(/this\.recycle\(hidden\.buffer\)/);
  });

  it("takes a buffer out of `lent` when it holds it", () => {
    // Otherwise `release` pools it a second time at the next flush and the pool
    // hands the same buffer to two callers at once.
    expect(code).toMatch(/const at = this\.lent\.indexOf\(buffer\);[\s\S]{0,120}this\.lent\.splice\(at, 1\);/);
  });

  it("does NOT put it back in the pool until the next flush", () => {
    // **The thing that was wrong and looked right.** Handing the buffer
    // straight to the pool lets a later dispatch in the same pass *write* what
    // an earlier one still *reads*. Dawn barriers read-after-write and does not
    // barrier write-after-read, so nothing errors and the numbers move:
    // `examples/h3-video`'s twelve-frame golden went 1.753e-1 -> 3.512e+0, 10
    // wrong pixel levels to 201, while its two-frame golden stayed green.
    //
    // A submit is a real barrier, so the buffer waits in quarantine for one.
    // The memory win is therefore the number of flushes, which is what
    // `splitBlockAboveRows` is for.
    for (const [name, body] of [
      ["dit", code],
      ["decoder", readFileSync(
        fileURLToPath(new URL("../../h3-video/src/decoder-gpu.ts", import.meta.url)), "utf8",
      ).replace(/\/\*[\s\S]*?\*\//g, "").replace(/\/\/.*$/gm, "")],
    ] as const) {
      const recycle = body.slice(body.indexOf("private recycle(buffer: GPUBuffer)"));
      const end = recycle.indexOf("\n  }\n");
      const inside = recycle.slice(0, end);
      expect(inside, `${name}: recycle must not pool directly`).not.toContain("this.pool.set(");
      expect(inside, `${name}: recycle must quarantine`).toContain("this.quarantine.push(buffer)");
      expect(body, `${name}: the flush is what releases the quarantine`)
        .toMatch(/for \(const buffer of this\.quarantine\)[\s\S]{0,200}this\.pool\.set\(/);
    }
  });
});
