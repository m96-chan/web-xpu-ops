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
  // Issue #223: `q`, `k` and `v` arrive already token-major, so
  // `flashAttentionToken` reads them directly and there is no head-swap step
  // to give them back after — the dispatch itself is the last reader.
  it("gives q, k and v back only after the flash dispatch has read them", () => {
    const body = method("attention");
    const dispatch = body.indexOf("this.kernels.flashAttentionToken");
    expect(dispatch, "the attention dispatches the token-major kernel").toBeGreaterThan(-1);
    for (const name of ["q", "k", "v"]) {
      const back = body.indexOf(`this.recycle(${name}.buffer)`);
      expect(back, `${name} is never given back`).toBeGreaterThan(-1);
      expect(back, `${name} is recycled before the dispatch reads it`).toBeGreaterThan(dispatch);
    }
  });

  it("never gives back what it is about to return", () => {
    // `attended` is the method's result. Handing it to the pool would let the
    // caller's next `take` return the same memory and overwrite the answer.
    const body = method("attention");
    expect(body).not.toContain("this.recycle(attended)");
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

  it("splits the block wherever a group of buffers has just died", () => {
    // The quarantine means the peak falls with the number of *submits*, not
    // with the bookkeeping, so a split has to sit where a group is already
    // dead. Issue #223 shrank the group inside `attention` from eight
    // buffers (three head swaps plus the five that fed them) to three (q, k
    // and v themselves) — smaller, but still worth a split rather than
    // waiting for the caller's own one after `attention` returns.
    const attention = method("attention");
    expect(attention, "the attention must split once q, k and v are recycled")
      .toMatch(/this\.recycle\(v\.buffer\);[\s\S]{0,200}await this\.flush\(ops, \[attended/);
    // Four in the block -- before the attention, after the attention, after
    // the residual add and after the feed-forward -- plus the one inside
    // `attention` above. Counted rather than matched: a split that is deleted
    // costs memory and nothing else, so nothing else would notice.
    const block = method("block");
    const splits = block.match(/await this\.flush\(ops, \[/g) ?? [];
    expect(splits.length, "the block's four split points").toBe(4);
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

  it("evicts the free pool, largest-first, after the quarantine drain", () => {
    // Issue #223: the pool only ever grew, so resident memory was the SUM of
    // every size class's own peak. `evictionPlan` bounds the FREE half of the
    // pool; it must only run once `lent` and `quarantine` have both been
    // folded in, and it must only ever destroy what it plans to.
    for (const [name, body] of [
      ["dit", code],
      ["decoder", readFileSync(
        fileURLToPath(new URL("../../h3-video/src/decoder-gpu.ts", import.meta.url)), "utf8",
      ).replace(/\/\*[\s\S]*?\*\//g, "").replace(/\/\/.*$/gm, "")],
    ] as const) {
      const at = body.indexOf("private release(keep");
      expect(at, `${name}: release() is not in the file`).toBeGreaterThan(-1);
      const rest = body.slice(at);
      const end = rest.indexOf("\n  }\n");
      expect(end, `${name}: release() has no end`).toBeGreaterThan(-1);
      const inside = rest.slice(0, end);

      const quarantineDrain = inside.indexOf("this.quarantine.length = 0");
      expect(quarantineDrain, `${name}: release() must drain the quarantine`).toBeGreaterThan(-1);

      const evictionCall = inside.indexOf("evictionPlan(");
      expect(evictionCall, `${name}: release() must call evictionPlan`).toBeGreaterThan(-1);
      expect(evictionCall, `${name}: eviction must run after the quarantine drain`)
        .toBeGreaterThan(quarantineDrain);

      expect(inside, `${name}: the planned buffers must actually be destroyed`).toMatch(/\.destroy\(\)/);
    }
  });
});
