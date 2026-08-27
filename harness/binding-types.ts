/**
 * Does the TypedArray the host uploaded agree with the type the kernel declared?
 *
 * Issue #221, out of #217. `ropeAxes`' `positions` binding was changed from
 * `array<i32>` to `array<f32>` when it learned fractional positions; one caller
 * was updated and two were not, and **nothing anywhere said so**. WebGPU has no
 * opinion about this — `queue.writeBuffer` takes an `ArrayBufferView` and copies
 * its bytes, and a small integer's bit pattern read as an `f32` is a denormal
 * near zero. Every rotation angle became zero, every token got the identity
 * rotation, and the DiT returned a well-formed latent with no position in it.
 * The page drew one flat colour for eighteen commits.
 *
 * **The type was written down the whole time.** It is in the WGSL, one line
 * above the code that reads it. Nothing on the host read that line.
 *
 * ## Why here and not in a bigger test
 *
 * `examples/anima/src/verify-forward-gpu.ts` catches this bug at the first
 * block — measured, with the fix reverted, at 8.275e-2 against 1.968e-5. It
 * could have caught it on the day it landed. It did not, because it needs a GPU
 * and 5 GB of weights that cannot go in CI, so it is a script somebody has to
 * remember to run.
 *
 * So this is deliberately the cheapest thing that works: two pure functions
 * over a string and an array. No device, no weights, no fixture. They are
 * called from the two places a buffer meets a pipeline —
 * `harness/wgsl.ts#createRunner` and `harness/resident.ts`'s bind group — so
 * every op test and every resident model pays for the check by existing.
 */

/** The scalar a storage binding's array is made of. */
export type ElementType = "f32" | "i32" | "u32";

/**
 * A `@group(0)` storage binding's element type per binding number, `null` where
 * the binding is a uniform, is in another group, or is not a storage array.
 *
 * A regex rather than a WGSL parser. Every binding in this repo is one line of
 * the form `@group(0) @binding(N) var<storage, ACCESS> name: array<T>;` — the
 * survey that decided this is in the commit that added the file — and a real
 * parser would be several hundred lines standing between a check and being run.
 * The cost is stated rather than hidden: a declaration this does not match
 * reads as `null`, which means *unchecked*, not *wrong*. Failing open is the
 * right direction for a check that is trying to be cheap enough to always be on.
 */
export function storageElementTypes(code: string): (ElementType | null)[] {
  // Comments first. Several kernels here quote their own binding lines while
  // explaining them, and a quoted line that shifted every index past it would
  // be worse than no check at all.
  const source = code.replace(/\/\*[\s\S]*?\*\//g, "").replace(/\/\/.*$/gm, "");
  const out: (ElementType | null)[] = [];
  // **Every** `@group(0)` binding, not only the storage arrays, so the array's
  // length is the kernel's binding count rather than "up to the last one I
  // recognised". A uniform sitting last is the common shape here, and an array
  // that stopped short of it could not be compared against a call site's
  // binding list at all.
  const declaration = /@group\(\s*0\s*\)\s*@binding\(\s*(\d+)\s*\)\s*var\s*(<[^>]*>)?\s*\w+\s*:\s*([^;]+);/g;
  for (const match of source.matchAll(declaration)) {
    const at = Number(match[1]);
    while (out.length <= at) out.push(null);
    const isStorage = (match[2] ?? "").includes("storage");
    // **`atomic<T>` is deliberately not read.** The first version treated
    // `array<atomic<u32>>` as `u32` -- an atomic u32 is a u32, so what could go
    // wrong -- and it turned `ops/scatter`'s eight tests red. Scatter is an f32
    // scatter-add: WGSL has no f32 atomic, so it declares `atomic<u32>` and does
    // `bitcast<u32>(bitcast<f32>(old) + value)` inside a compare-exchange loop.
    // The buffer holds floats. The `u32` is there for `atomicCompareExchangeWeak`
    // and for nothing else.
    //
    // That is the shape of every atomic worth writing: an atomic exists to do a
    // read-modify-write, and in WGSL the only way to do one on a float is to
    // pun it. So the declared type of an atomic says how it is *updated*, not
    // what it *holds*, and it is the one place a mismatch here is the point
    // rather than the bug. Unknown, therefore unchecked. Scatter's `indices:
    // array<i32>` is still checked, which is the binding that could be wrong.
    const element = isStorage ? /^array\s*<\s*(f32|i32|u32)\s*>/.exec(match[3]!.trim()) : null;
    out[at] = element ? (element[1] as ElementType) : null;
  }
  return out;
}

/**
 * Something to call a kernel by in an error message, out of its own source.
 *
 * The call sites hold `code`, a multi-kilobyte string, and nothing else that
 * identifies it — `kernel(import.meta.url)` has already thrown the path away by
 * then. Every WGSL file here opens with a comment naming the op (`// RMSNorm:
 * x_i * w_i / …`), so the first clause of the first line is the name its author
 * already wrote. `"a kernel"` when there is no comment: a vague name in a
 * message that also carries the binding number and both types is a small loss,
 * and threading a path through two APIs to avoid it is not worth it.
 */
export function kernelName(code: string): string {
  const first = /^\s*\/\/\s*(.+)$/m.exec(code);
  if (!first) return "a kernel";
  return first[1]!.split(/[:(]/)[0]!.trim().slice(0, 60) || "a kernel";
}

/** What each TypedArray claims its elements are. `null` means "raw bytes". */
function elementOf(view: ArrayBufferView): ElementType | null {
  if (view instanceof Float32Array) return "f32";
  if (view instanceof Int32Array) return "i32";
  if (view instanceof Uint32Array) return "u32";
  // **`Uint8Array` is the deliberate escape hatch, and it is load-bearing.**
  // `ops/matvec/wgsl/q8.wgsl` declares `weight: array<u32>` and is fed four
  // packed int8 weights per word; every resident weight upload in the repo is
  // `new Uint8Array(<safetensors slice>)`. A `Uint8Array` is the host saying
  // "these are bytes", and there is no element type in that claim to disagree
  // with. Rejecting it would reject every correct weight upload here.
  //
  // The exposure is real and worth naming: `new Uint8Array(ints.buffer)` would
  // reintroduce #217 and this would not see it. What it does cover is the far
  // more likely slip -- passing the array you already had, of the type you
  // already had -- which is exactly what happened.
  return null;
}

/**
 * The first disagreement between what a kernel declared and what was uploaded
 * into it, as a message, or `null` when there is nothing to say.
 *
 * Returns rather than throws so both call sites can decide how to raise it, and
 * so the whole thing is testable without catching.
 */
export function bindingTypeMismatch(
  kernel: string,
  declared: readonly (ElementType | null)[],
  /**
   * A `TypedArray` for a binding the host uploaded into, or an `ElementType`
   * for one it only intends to *read back* as that type — the same hazard with
   * the arrow reversed. `{ kind: "out", type: "i32" }` on a binding declared
   * `array<f32>` reads float bits as integers and returns enormous nonsense,
   * with no error anywhere, which is #217's whole shape. Passed by name rather
   * than as an empty `Float32Array` so the message can say what actually
   * happened instead of claiming an array nobody handed over.
   */
  supplied: readonly (ArrayBufferView | ElementType | null | undefined)[],
): string | null {
  // The shorter of the two. They come from different places -- one from the
  // WGSL, one from the call site -- and a kernel whose bindings were miscounted
  // is a thing to report elsewhere, not to read off the end of an array here.
  const n = Math.min(declared.length, supplied.length);
  for (let i = 0; i < n; i += 1) {
    const want = declared[i];
    const view = supplied[i];
    if (!want || !view) continue;
    const got = typeof view === "string" ? view : elementOf(view);
    if (got === null || got === want) continue;
    const what = typeof view === "string" ? `read back as ${view}` : `given a ${view.constructor.name}`;
    return (
      `${kernel}: binding ${i} is declared \`array<${want}>\` and was ${what}. ` +
      "WebGPU copies the bytes without complaint, so this does not fail — it " +
      "silently reads as the wrong numbers (issue #217)."
    );
  }
  return null;
}
