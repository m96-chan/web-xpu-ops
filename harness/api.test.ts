/**
 * The kernel-source registry in `harness/api.ts` — the one piece of the
 * harness that a published consumer has to call before an engine can run.
 *
 * `llm/kernels.ts` and `llm/engine-q8-resident.ts` used to read their WGSL
 * through `harness/suite.ts#kernel`, which is `readFileSync` — so the engine
 * could not be published: every browser demo replaced the whole harness with an
 * esbuild shim to get around it. The registry is what replaces the shim. These
 * tests pin the contract the shim used to hold implicitly: an op name derived
 * from the `ops/<op>/index.ts` URL shape, an entry defaulting to `kernel`, and
 * an error that names what is missing rather than an `undefined` that compiles
 * as an empty shader.
 */
import { afterEach, describe, expect, it } from "vitest";
import { kernelFromUrl, opKernel, registerKernelSources, resetKernelSources } from "./api.js";

afterEach(() => resetKernelSources());

describe("kernel-source registry", () => {
  it("has nothing registered until something registers", () => {
    expect(() => opKernel("rmsnorm")).toThrow(/registerKernelSources/);
  });

  it("serves a registered table, entry defaulting to `kernel`", () => {
    registerKernelSources({ rmsnorm: { kernel: "// rms" }, matvec: { q8: "// q8" } });
    expect(opKernel("rmsnorm")).toBe("// rms");
    expect(opKernel("matvec", "q8")).toBe("// q8");
  });

  it("names the missing op and entry, not just the failure", () => {
    registerKernelSources({ rmsnorm: { kernel: "// rms" } });
    expect(() => opKernel("matvec", "q8")).toThrow(/matvec.*q8/);
    expect(() => opKernel("rmsnorm", "fused")).toThrow(/rmsnorm.*fused/);
  });

  it("accepts a resolver function, for a host that reads files lazily", () => {
    const asked: string[] = [];
    registerKernelSources((op, entry) => { asked.push(`${op}/${entry}`); return `// ${op}/${entry}`; });
    expect(opKernel("gqa", "scores")).toBe("// gqa/scores");
    expect(asked).toEqual(["gqa/scores"]);
  });

  it("derives the op from an `ops/<op>/index.ts` URL, as the shim did", () => {
    registerKernelSources({ rope: { kernel: "// rope" } });
    expect(kernelFromUrl(new URL("file:///anywhere/ops/rope/index.ts"))).toBe("// rope");
    expect(() => kernelFromUrl("file:///anywhere/rope.ts")).toThrow(/ops\/<op>\/index\.ts/);
  });
});
