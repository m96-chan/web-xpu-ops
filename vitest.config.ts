import { defineConfig } from "vitest/config";

export default defineConfig({
  test: {
    include: ["ops/**/*.test.ts", "harness/*.test.ts"],
    pool: "forks",
    poolOptions: { forks: { singleFork: true } },
    isolate: false,
    testTimeout: 30_000,
    hookTimeout: 60_000,
  },
});
