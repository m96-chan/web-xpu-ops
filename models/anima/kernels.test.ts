import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { describe, expect, it } from "vitest";
import { ditKernels, encoderKernels } from "../../examples/zimage/src/kernels-node.js";
import { vaeKernels } from "../../examples/anima/src/vae-kernels-node.js";
import { ANIMA_KERNEL_FILES, animaKernels, kernelsFromSources } from "./kernels.js";

const root = new URL("../../", import.meta.url);
const fromDisk = (op: string, entry: string): string =>
  readFileSync(fileURLToPath(new URL(`ops/${op}/wgsl/${entry}.wgsl`, root)), "utf8");

describe("animaKernels", () => {
  it("builds the same three tables the Node loaders build", () => {
    const built = animaKernels(fromDisk);
    expect(built.dit).toEqual(ditKernels());
    expect(built.encoder).toEqual(encoderKernels());
    expect(built.vae).toEqual(vaeKernels());
  });

  it("asks for exactly the files ANIMA_KERNEL_FILES names", () => {
    const asked = new Set<string>();
    animaKernels((op, entry) => { asked.add(`ops/${op}/wgsl/${entry}.wgsl`); return "// wgsl"; });
    expect([...asked].sort()).toEqual([...ANIMA_KERNEL_FILES]);
  });

  it("names files that exist", () => {
    for (const file of ANIMA_KERNEL_FILES) {
      expect(() => readFileSync(fileURLToPath(new URL(file, root))), file).not.toThrow();
    }
  });

  it("refuses a loader that comes back empty-handed, naming the file", () => {
    expect(() => kernelsFromSources([{ key: "rmsnorm", op: "rmsnorm", entry: "kernel" }], () => undefined as unknown as string))
      .toThrow(/ops\/rmsnorm\/wgsl\/kernel\.wgsl/);
  });
});
