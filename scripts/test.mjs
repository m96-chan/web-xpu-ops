#!/usr/bin/env node
/**
 * Runs the suite one test file per vitest process.
 *
 * Not a preference. Measured on Linux / Dawn (`webgpu@0.4.x`) / vitest 2.1.9:
 * a single vitest process cannot cross a test-file boundary with a GPU device
 * in play. It dies with a glibc assertion out of Dawn's thread pool
 * (`pthread_mutex_lock`, `__pthread_tpp_change_priority`), or with
 * `std::system_error`, or it hangs. See issue #38.
 *
 * What was tried and did not fix it, so nobody repeats it:
 *
 *   - `afterAll` teardown moved to `process.once("exit")`, so one device really
 *     does live for the whole run rather than being rebuilt per file
 *   - `pool: "forks"` with `isolate: true` (a process per file, recycled)
 *   - `pool: "forks"` with `maxForks: 1`, `fileParallelism: false`
 *   - `pool: "threads"` with `singleThread: true`
 *
 * All four still died at the first file boundary. The native module is not the
 * cause either — it is evaluated exactly once (verified by counting evaluations
 * on `globalThis`). Meanwhile 16 create/destroy cycles *inside* one file are
 * fine. The boundary is the thing that breaks.
 *
 * So each file gets a process that starts, makes one device, and exits. The
 * cost is process startup per op, which is seconds for the whole suite and buys
 * a result that can be believed.
 *
 * The other half of this script's job is refusing to report a false pass. A
 * crashed vitest worker prints `Test Files 1 passed (2)` and can still exit 0 —
 * the remaining files never ran. Here every file is accounted for individually
 * and a non-zero exit anywhere fails the run.
 */
import { spawn } from "node:child_process";
import { readdirSync, statSync } from "node:fs";
import { join } from "node:path";

const OPS = "ops";
const HARNESS = "harness";

/** Test files, discovered the same way `vitest.config.ts` globs them. */
function testFiles() {
  const found = [];
  for (const entry of readdirSync(OPS)) {
    const dir = join(OPS, entry);
    if (!statSync(dir).isDirectory()) continue;
    for (const file of readdirSync(dir)) {
      if (file.endsWith(".test.ts")) found.push(join(dir, file));
    }
  }
  // The harness has tests of its own — kernel resolution, target detection —
  // and they live flat rather than one directory per op. Most need no GPU, but
  // they get a process each anyway: the alternative is a second code path.
  for (const file of readdirSync(HARNESS)) {
    if (file.endsWith(".test.ts")) found.push(join(HARNESS, file));
  }
  return found.sort();
}

/**
 * A wedged file has to fail, not stall the run.
 *
 * Dawn can hang instead of aborting, and an unbounded wait turns one bad file
 * into a suite that never returns — which in CI is indistinguishable from slow.
 * A whole file is seconds when it is healthy, so a minute is far past normal.
 */
const PER_FILE_TIMEOUT_MS = 60_000;

function run(file) {
  return new Promise((resolve) => {
    const child = spawn("npx", ["vitest", "run", file], {
      stdio: ["ignore", "pipe", "pipe"],
      env: process.env,
    });
    let output = "";
    let timedOut = false;
    const timer = setTimeout(() => {
      timedOut = true;
      child.kill("SIGKILL");
    }, PER_FILE_TIMEOUT_MS);
    child.stdout.on("data", (chunk) => (output += chunk));
    child.stderr.on("data", (chunk) => (output += chunk));
    child.on("close", (code) => {
      clearTimeout(timer);
      if (timedOut) output += `\n[runner] killed after ${PER_FILE_TIMEOUT_MS / 1000}s\n`;
      resolve({ code: timedOut ? "timeout" : code, output });
    });
  });
}

/**
 * Strips ANSI SGR sequences before parsing.
 *
 * Not cosmetic. CI runs with colour forced on, so the summary arrives as
 * `\x1b[2m Tests \x1b[22m…` and an anchored `^\s*Tests` never matches — every
 * file then looks like it produced no summary, gets retried, and fails. It
 * failed safe (a false FAIL, never a false pass) but it made a genuinely green
 * suite unreportable, and it was invisible locally because vitest turns colour
 * off by itself when its output is a pipe.
 */
function plain(text) {
  return text.replace(/\[[0-9;]*m/g, "");
}

/** `Tests  5 passed (5)` → `{ passed: 5, total: 5 }`; null when vitest printed no summary. */
function counts(output) {
  const line = /^\s*Tests\s+(.*)$/m.exec(plain(output));
  if (!line) return null;
  const passed = /(\d+) passed/.exec(line[1]);
  const total = /\((\d+)\)/.exec(line[1]);
  if (!passed || !total) return null;
  return { passed: Number(passed[1]), total: Number(total[1]) };
}

const files = testFiles();
if (files.length === 0) {
  console.error("no test files found under ops/");
  process.exit(1);
}

let failed = 0;
let tests = 0;

for (const file of files) {
  let { code, output } = await run(file);
  let seen = counts(output);

  // Retried only when vitest produced **no summary at all** — the process died
  // or hung before any test reported. Nothing was learned about the kernel, so
  // running it again asks the question rather than re-asking until it says yes.
  //
  // A file whose tests ran and failed is never retried. That distinction is the
  // whole point: retrying a real assertion failure would launder a broken
  // kernel into a green suite, which is exactly what this repository's first
  // rule exists to prevent. The retry is also announced, so a run that needed
  // one cannot be mistaken for a clean one.
  if (seen === null) {
    console.log(`  ....  ${file} produced no summary (exit=${code}); retrying once — see #38`);
    ({ code, output } = await run(file));
    seen = counts(output);
  }

  const ok = code === 0 && seen !== null && seen.passed === seen.total;
  if (ok) {
    tests += seen.passed;
    console.log(`  ok   ${file}  ${seen.passed}/${seen.total}`);
    continue;
  }
  failed += 1;
  // A file with no summary at all crashed before reporting; show the output so
  // the reason is in the log rather than inferred from an exit code.
  console.log(`  FAIL ${file}  exit=${code}${seen ? `  ${seen.passed}/${seen.total}` : "  (no summary — crashed or hung)"}`);
  console.log(
    plain(output)
      .split("\n")
      .filter((l) => l.trim())
      .map((l) => `       ${l}`)
      .join("\n"),
  );
}

console.log(
  failed === 0
    ? `\n${files.length} files, ${tests} tests, all passed`
    : `\n${files.length} files, ${failed} failed`,
);
process.exit(failed === 0 ? 0 : 1);
