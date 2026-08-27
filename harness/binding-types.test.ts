/**
 * The check that would have caught #217, held to the shape of that bug.
 *
 * Every case here runs without a GPU and without a byte of weights, which is
 * the whole point: the thing that *could* have caught #217 —
 * `examples/anima/src/verify-forward-gpu.ts` — needed 5 GB of weights and a
 * device, so it was a script nobody ran, and the bug lived for eighteen
 * commits. A check nobody can afford to run is not a check.
 */
import { describe, expect, it } from "vitest";
import { bindingTypeMismatch, kernelName, storageElementTypes } from "./binding-types.js";
import { gpuTest, kernel, useGpu } from "./suite.js";
import { params } from "./wgsl.js";

describe("harness / binding types", () => {
  describe("reading what the kernel declared", () => {
    it("indexes by the binding number, not by the order of the lines", () => {
      // Out of order deliberately: the array is indexed by `@binding(n)`, and a
      // parser that pushed in source order would agree with this on every
      // kernel in the repo and disagree the first time someone reordered.
      const types = storageElementTypes(`
        @group(0) @binding(2) var<storage, read_write> out: array<i32>;
        @group(0) @binding(0) var<storage, read> input: array<f32>;
      `);
      expect(types).toEqual(["f32", null, "i32"]);
    });

    it("reads read and read_write alike", () => {
      const types = storageElementTypes(`
        @group(0) @binding(0) var<storage, read> a: array<f32>;
        @group(0) @binding(1) var<storage, read_write> b: array<u32>;
      `);
      expect(types).toEqual(["f32", "u32"]);
    });

    it("leaves an atomic unknown, because that is where bit-punning lives", () => {
      // The first version of this said `atomic<u32>` is a `u32` -- what could
      // go wrong -- and turned `ops/scatter`'s eight tests red. Scatter is an
      // f32 scatter-add; WGSL has no f32 atomic, so it declares `atomic<u32>`
      // and runs `bitcast<u32>(bitcast<f32>(old) + value)` inside a
      // compare-exchange loop. The buffer holds floats and reading it back as
      // f32 is right.
      //
      // Which is the shape of every atomic worth writing: the declared type
      // says how the slot is *updated*, not what it *holds*.
      expect(storageElementTypes("@group(0) @binding(0) var<storage, read_write> o: array<atomic<u32>>;"))
        .toEqual([null]);
      // But only the atomic. A plain binding beside it is still checked -- the
      // exemption must not spread to the rest of the kernel.
      expect(storageElementTypes(`
        @group(0) @binding(0) var<storage, read> i: array<i32>;
        @group(0) @binding(1) var<storage, read_write> o: array<atomic<u32>>;
      `)).toEqual(["i32", null]);
    });

    it("leaves a uniform unknown rather than guessing at its struct", () => {
      // `var<uniform> params: Params` is a struct whose fields are mixed types,
      // packed by `harness/wgsl.ts#params` into raw bytes. There is nothing
      // here to compare a TypedArray against, and inventing a comparison would
      // reject every correct uniform in the repo.
      const types = storageElementTypes(`
        @group(0) @binding(0) var<storage, read> a: array<f32>;
        @group(0) @binding(1) var<uniform> params: Params;
      `);
      expect(types).toEqual(["f32", null]);
    });

    it("ignores a binding in a group other than 0", () => {
      // Every kernel here uses `@group(0)` and `layout: "auto"` builds one bind
      // group. A `@group(1)` entry would be a different bind group with its own
      // binding numbering, and folding it into this array would silently
      // shift every index past it.
      expect(storageElementTypes(`
        @group(0) @binding(0) var<storage, read> a: array<f32>;
        @group(1) @binding(0) var<storage, read> b: array<i32>;
      `)).toEqual(["f32"]);
    });

    it("is not fooled by the declaration appearing in a comment", () => {
      // Kernels in this repo carry long comments, and several of them quote a
      // binding line while explaining it.
      const types = storageElementTypes(`
        // @group(0) @binding(1) var<storage, read> ghost: array<i32>;
        /* @group(0) @binding(2) var<storage, read> ghost2: array<u32>; */
        @group(0) @binding(0) var<storage, read> real: array<f32>;
      `);
      expect(types).toEqual(["f32"]);
    });
  });

  describe("comparing it against what the host uploaded", () => {
    const f32 = ["f32"] as const;

    it("passes the matching type", () => {
      expect(bindingTypeMismatch("rope", ["f32"], [new Float32Array(4)])).toBeNull();
      expect(bindingTypeMismatch("gather", ["i32"], [new Int32Array(4)])).toBeNull();
      expect(bindingTypeMismatch("matvec", ["u32"], [new Uint32Array(4)])).toBeNull();
    });

    it("catches #217 exactly: an Int32Array bound as array<f32>", () => {
      // The bug this file exists for. `ropeAxes`' `positions` became
      // `array<f32>` and two callers kept uploading an `Int32Array`; WebGPU
      // copied the bytes, a small integer read as a float is a denormal, and
      // every rotation angle became zero.
      const message = bindingTypeMismatch("ropeAxes", ["f32", "u32"], [new Int32Array(4), new Uint32Array(2)]);
      expect(message).toContain("ropeAxes");
      expect(message).toContain("binding 0");
      expect(message).toContain("Int32Array");
      expect(message).toContain("array<f32>");
    });

    it("names the first mismatch and the binding it is at", () => {
      // Not just "something is wrong": the message has to say which binding,
      // because a kernel with eight of them and a wrong one is exactly the
      // situation where the error is otherwise useless.
      const message = bindingTypeMismatch("k", ["f32", "f32", "i32"], [
        new Float32Array(1), new Float32Array(1), new Float32Array(1),
      ]);
      expect(message).toContain("binding 2");
      expect(message).toContain("array<i32>");
    });

    it("lets a Uint8Array through, because packed bytes are the point", () => {
      // `ops/matvec/wgsl/q8.wgsl` declares `weight: array<u32>` and is fed a
      // `Uint8Array` of four packed int8 weights per word, deliberately — and
      // every resident weight upload in the repo goes through
      // `new Uint8Array(data)` over a safetensors slice. A `Uint8Array` is the
      // host saying "these are bytes, I know what I am doing"; there is no
      // element type to disagree with.
      expect(bindingTypeMismatch("matvecQ8", ["u32"], [new Uint8Array(16)])).toBeNull();
      expect(bindingTypeMismatch("anything", ["f32"], [new Uint8Array(16)])).toBeNull();
    });

    it("says nothing about a binding nobody uploaded", () => {
      // Output and scratch buffers are created, never written from the host.
      expect(bindingTypeMismatch("k", ["f32", "f32"], [new Float32Array(1), null])).toBeNull();
    });

    it("says nothing about a binding the kernel did not declare as a storage array", () => {
      // The uniform. Its bytes are a packed struct.
      expect(bindingTypeMismatch("k", [...f32, null], [new Float32Array(1), new Uint8Array(16)])).toBeNull();
    });

    it("does not read past either array", () => {
      // The two arrays come from different places -- one from the WGSL, one
      // from the call site -- and a kernel whose bindings were miscounted must
      // not turn into an out-of-bounds read here.
      expect(bindingTypeMismatch("k", ["f32"], [new Float32Array(1), new Int32Array(1)])).toBeNull();
      expect(bindingTypeMismatch("k", ["f32", "i32"], [new Float32Array(1)])).toBeNull();
    });
  });

  describe("the same mistake in the other direction: reading back", () => {
    // An `out` binding says what to interpret the result as, and it is the same
    // hazard with the arrow reversed -- `{ kind: "out", type: "i32" }` on a
    // binding the kernel declared `array<f32>` reads float bits as integers and
    // returns enormous nonsense, again without an error. Passing the element
    // type by name rather than by an empty TypedArray keeps the message honest:
    // nobody handed a `Float32Array` over, so it must not say they did.
    it("catches a readback type that disagrees with the declaration", () => {
      const message = bindingTypeMismatch("gather", ["f32", "i32"], [null, "f32"]);
      expect(message).toContain("binding 1");
      expect(message).toContain("array<i32>");
      expect(message).toContain("read back as f32");
      expect(message).not.toContain("Array");
    });

    it("passes a readback that agrees", () => {
      expect(bindingTypeMismatch("gather", ["f32", "i32"], [null, "i32"])).toBeNull();
    });
  });

  describe("naming the kernel in the message", () => {
    it("takes the name its author already wrote at the top of the file", () => {
      expect(kernelName("// RMSNorm: x_i * w_i / sqrt(mean(x²) + eps)\n@compute fn main() {}"))
        .toBe("RMSNorm");
      expect(kernelName("// matvecQ8 (W8A32 GEMV): out[i] = ...")).toBe("matvecQ8");
    });

    it("falls back rather than putting a kilobyte of shader in the message", () => {
      // A kernel with no leading comment still has to produce something short.
      // The failure mode being avoided is an error whose first line is the
      // whole file.
      expect(kernelName("@compute @workgroup_size(256) fn main() {}")).toBe("a kernel");
      expect(kernelName(`// ${"x".repeat(200)}`)).toHaveLength(60);
    });
  });

  describe("against the kernels actually in the repo", () => {
    it("finds a binding for every op's own WGSL", async () => {
      // The parser is only worth having if it reads the real files. A regex
      // that works on the fixtures above and returns nothing but nulls on the
      // repo would pass every other test here.
      const { readFileSync } = await import("node:fs");
      const { globSync } = await import("node:fs");
      const files = globSync("ops/*/wgsl/*.wgsl");
      expect(files.length).toBeGreaterThan(20);
      for (const file of files) {
        const types = storageElementTypes(readFileSync(file, "utf8"));
        expect(types.filter((t) => t !== null).length, `${file} has no storage binding`).toBeGreaterThan(0);
      }
    });
  });
});

/**
 * That `harness/wgsl.ts#createRunner` actually calls the two functions above.
 *
 * Every case in the file above is pure, and every one of them would still pass
 * with the call site deleted -- which is the failure rule 1 names: a test whose
 * observation point is in the wrong place. This is the observation point. It
 * needs a device and nothing else: two floats, no weights, no golden.
 */
describe("harness / binding types, wired into the runner", () => {
  useGpu();
  const elementwise = kernel(new URL("../ops/elementwise/index.ts", import.meta.url));
  const uniform = params([["u32", 2], ["u32", 0]]);

  gpuTest("refuses a dispatch whose upload disagrees with the declaration", async (run) => {
    await expect(run({
      code: elementwise,
      bindings: [
        { kind: "storage", data: new Float32Array([1, 2]) },
        // `b` is `array<f32>`. This is #217's exact shape, one op down.
        { kind: "storage", data: new Int32Array([3, 4]) },
        { kind: "out", type: "f32", length: 2 },
        { kind: "uniform", data: uniform },
      ],
      workgroups: [1],
    })).rejects.toThrow(/binding 1.*array<f32>.*Int32Array/s);
  });

  gpuTest("refuses a readback type that disagrees with the declaration", async (run) => {
    await expect(run({
      code: elementwise,
      bindings: [
        { kind: "storage", data: new Float32Array([1, 2]) },
        { kind: "storage", data: new Float32Array([3, 4]) },
        // Reading f32 bits as integers returns enormous nonsense, silently.
        { kind: "out", type: "i32", length: 2 },
        { kind: "uniform", data: uniform },
      ],
      workgroups: [1],
    })).rejects.toThrow(/binding 2.*array<f32>.*read back as i32/s);
  });

  gpuTest("still runs the correct dispatch", async (run) => {
    // The other half of the mutation. A check that refused everything would
    // pass both cases above and be useless, and the 854 op tests that already
    // go through this path would catch that -- but not in this file, next to
    // the thing being asserted.
    const [out] = await run({
      code: elementwise,
      bindings: [
        { kind: "storage", data: new Float32Array([1, 2]) },
        { kind: "storage", data: new Float32Array([3, 4]) },
        { kind: "out", type: "f32", length: 2 },
        { kind: "uniform", data: uniform },
      ],
      workgroups: [1],
    });
    expect(Array.from(out as Float32Array)).toEqual([4, 6]);
  });
});
