import { defineConfig } from "vitest/config";

export default defineConfig({
  test: {
    include: ["ops/**/*.test.ts", "harness/*.test.ts", "llm/**/*.test.ts", "examples/anima/src/**/*.test.ts", "examples/anima-web/src/**/*.test.ts", "examples/web-common/src/**/*.test.ts", "examples/zimage-web/src/**/*.test.ts", "examples/zimage/src/**/*.test.ts", "examples/zimage-vae/src/**/*.test.ts", "examples/h3-audio/src/**/*.test.ts", "examples/h3-audio-web/src/**/*.test.ts"],
    pool: "forks",
    poolOptions: { forks: { singleFork: true } },
    isolate: false,
    testTimeout: 30_000,
    hookTimeout: 60_000,
  },
});
