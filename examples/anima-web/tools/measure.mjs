/**
 * Runs the Anima demo in a real Chrome and prints the profile it reports.
 *
 * Issue #186. Every browser number in this repository so far arrived by asking
 * a person to click Generate and paste a table back. That is slow, it is the
 * reason four separate hypotheses were argued about before anyone measured
 * them, and it makes "re-measure after the change" a favour rather than a step.
 *
 * **Non-headless, and driven with raw CDP over Node's own `WebSocket`.** Both
 * are the practice this repository already documents for real-hardware checks
 * (README, "Real-hardware verification"): headless Chrome's WebGPU is not the
 * same stack a user gets, and a driver that measures a different stack is the
 * same mistake as a benchmark that times a different kernel. No puppeteer — one
 * dependency for one socket.
 *
 * A dedicated `--user-data-dir` so this cannot disturb a browser someone is
 * using, and so the 5 GB of weights the Cache API holds are this tool's own.
 * **The browser is killed on every exit path**, including the failures: a
 * measurement tool that leaks a Chrome per run is a tool that stops being run.
 *
 *     node examples/anima-web/tools/measure.mjs [--steps 2] [--size 832x1216]
 *                                              [--url http://127.0.0.1:8789]
 *                                              [--keep] [--timeout 900]
 *
 * `--steps 2` is the minimum that profiles anything: the first forward is the
 * upload, so the second is the one measured.
 */
import { spawn } from "node:child_process";
import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import path from "node:path";

const arg = (name, fallback) => {
  const at = process.argv.indexOf(`--${name}`);
  return at >= 0 ? process.argv[at + 1] : fallback;
};
const flag = (name) => process.argv.includes(`--${name}`);

const STEPS = Number(arg("steps", "2"));
const SIZE = arg("size", "832x1216");
const URL_ = arg("url", "http://127.0.0.1:8789");
const TIMEOUT_MS = Number(arg("timeout", "900")) * 1000;
const PORT = Number(arg("port", "9333"));

if (STEPS < 2) {
  console.error("measure: --steps must be at least 2 — the first forward is the upload, so the second is the one profiled.");
  process.exit(2);
}

/** Chrome and its scratch profile, tracked so `cleanup` can always reach them. */
let chrome = null;
let profileDir = null;
let socket = null;
let cleaned = false;

function cleanup() {
  if (cleaned) return;
  cleaned = true;
  try {
    socket?.close();
  } catch {
    // A socket that is already gone is the state we wanted.
  }
  if (chrome && chrome.exitCode === null) {
    // The whole process group: Chrome forks renderers and the GPU process, and
    // killing only the parent leaves them holding VRAM.
    try {
      process.kill(-chrome.pid, "SIGTERM");
    } catch {
      try {
        chrome.kill("SIGTERM");
      } catch {
        // Already gone.
      }
    }
    // **Wait for it.** The first version removed the profile directory
    // immediately after signalling, Chrome still had its files open, `rmSync`
    // failed with the error swallowed, and a 5 GB directory was left behind
    // reported as a clean run. Synchronous, because `process.on("exit")`
    // cannot await.
    const deadline = Date.now() + 10_000;
    while (chrome.exitCode === null && Date.now() < deadline) {
      try {
        // Signal 0 tests for existence without delivering anything.
        process.kill(-chrome.pid, 0);
      } catch {
        break;
      }
      // Sleeping without a timer, since this runs inside an exit handler.
      Atomics.wait(new Int32Array(new SharedArrayBuffer(4)), 0, 0, 100);
    }
    try {
      process.kill(-chrome.pid, "SIGKILL");
    } catch {
      // Gone, which is what the wait was for.
    }
  }
  if (profileDir && !flag("keep")) {
    try {
      rmSync(profileDir, { recursive: true, force: true });
    } catch (error) {
      // Reported, never swallowed: a tool that leaks 5 GB per run and says
      // nothing is a tool that fills a disk before anyone notices.
      console.error(`measure: could not remove ${profileDir} — ${error.message}. Remove it by hand.`);
    }
  }
}

for (const signal of ["exit", "SIGINT", "SIGTERM", "uncaughtException"]) {
  process.on(signal, (error) => {
    cleanup();
    if (signal === "uncaughtException") {
      console.error(error);
      process.exit(1);
    }
    if (signal !== "exit") process.exit(130);
  });
}

/** Chrome's DevTools endpoint, polled because the port opens after the process does. */
async function debuggerUrl() {
  const deadline = Date.now() + 30_000;
  while (Date.now() < deadline) {
    try {
      const response = await fetch(`http://127.0.0.1:${PORT}/json/version`);
      if (response.ok) return (await response.json()).webSocketDebuggerUrl;
    } catch {
      // Not listening yet.
    }
    await new Promise((resolve) => setTimeout(resolve, 250));
  }
  throw new Error(`measure: Chrome never opened its debugging port ${PORT}`);
}

let nextId = 1;
const pending = new Map();

function send(method, params = {}, sessionId) {
  const id = nextId++;
  socket.send(JSON.stringify({ id, method, params, sessionId }));
  return new Promise((resolve, reject) => pending.set(id, { resolve, reject }));
}

/** Evaluates an expression in the page and returns its value. */
async function evaluate(sessionId, expression) {
  const result = await send(
    "Runtime.evaluate",
    { expression, returnByValue: true, awaitPromise: true },
    sessionId,
  );
  if (result.exceptionDetails) {
    throw new Error(`measure: page threw — ${result.exceptionDetails.exception?.description ?? "unknown"}`);
  }
  return result.result.value;
}

async function main() {
  // Reachable before a browser is launched, so a missing server is one line
  // rather than a Chrome that starts and shows an error page.
  try {
    const probe = await fetch(URL_);
    if (!probe.ok) throw new Error(String(probe.status));
  } catch (error) {
    console.error(`measure: ${URL_} did not answer (${error.message}). Start examples/anima-web/server.mjs first.`);
    process.exit(2);
  }

  profileDir = mkdtempSync(path.join(tmpdir(), "anima-measure-"));
  chrome = spawn(
    "google-chrome-stable",
    [
      `--remote-debugging-port=${PORT}`,
      `--user-data-dir=${profileDir}`,
      "--no-first-run",
      "--no-default-browser-check",
      // WebGPU is the point; a software adapter would measure nothing.
      "--enable-unsafe-webgpu",
      URL_,
    ],
    { detached: true, stdio: "ignore" },
  );

  socket = new WebSocket(await debuggerUrl());
  const attached = new Promise((resolve) => {
    socket.addEventListener("message", (event) => {
      const message = JSON.parse(event.data);
      if (message.id && pending.has(message.id)) {
        const { resolve: ok, reject } = pending.get(message.id);
        pending.delete(message.id);
        if (message.error) reject(new Error(`${message.method ?? "cdp"}: ${message.error.message}`));
        else ok(message.result);
      }
    });
    socket.addEventListener("open", resolve);
  });
  await attached;

  const { targetInfos } = await send("Target.getTargets");
  const page = targetInfos.find((t) => t.type === "page" && t.url.startsWith(URL_));
  if (!page) throw new Error("measure: Chrome opened no page for the demo");
  const { sessionId } = await send("Target.attachToTarget", { targetId: page.targetId, flatten: true });

  // The page disables Generate until the weights are open. Waiting for the
  // button rather than for a fixed delay: the load is minutes on a cold cache
  // and seconds on a warm one.
  // Printed on change rather than every second: a captured run of the first
  // version was a wall of the same line repeated a hundred times, because a
  // `\r` that redraws a terminal line just appends when the output is a pipe.
  let lastStatus = "";
  console.log(`measure: waiting for ${URL_} to be ready …`);
  const readyBy = Date.now() + TIMEOUT_MS;
  while (Date.now() < readyBy) {
    const ready = await evaluate(sessionId, "!document.getElementById('go')?.disabled");
    if (ready) break;
    const status = await evaluate(sessionId, "document.getElementById('status')?.textContent ?? ''");
    if (status !== lastStatus) console.log(`  ${status}`);
    lastStatus = status;
    await new Promise((resolve) => setTimeout(resolve, 1000));
  }

  await evaluate(
    sessionId,
    `(() => {
      const size = document.getElementById('size');
      const wanted = [...size.options].find((o) => o.value === ${JSON.stringify(SIZE)});
      if (!wanted) throw new Error('no size option ' + ${JSON.stringify(SIZE)} + '; have ' + [...size.options].map((o) => o.value).join(', '));
      size.value = wanted.value;
      document.getElementById('steps').value = '${STEPS}';
      document.getElementById('profile').checked = true;
      document.getElementById('go').click();
      return true;
    })()`,
  );

  console.log(`measure: generating at ${SIZE}, ${STEPS} steps, profiling on …`);
  const doneBy = Date.now() + TIMEOUT_MS;
  while (Date.now() < doneBy) {
    // The button re-enables when `generate()` returns, success or failure.
    const done = await evaluate(sessionId, "!document.getElementById('go').disabled");
    const status = await evaluate(sessionId, "document.getElementById('status')?.textContent ?? ''");
    if (status !== lastStatus) console.log(`  ${status}`);
    lastStatus = status;
    if (done) break;
    await new Promise((resolve) => setTimeout(resolve, 1000));
  }

  const report = await evaluate(
    sessionId,
    `(async () => {
      const text = (el) => (el?.innerText ?? '').trim();
      // A digest of the pixels, so "the change did not alter the image" is
      // something checked rather than asserted. A change that removes work
      // which turns out to have been load-bearing shows up here and nowhere
      // else — it would not throw, it would draw something different.
      const canvas = document.getElementById('out');
      let image = null;
      if (canvas && canvas.width > 0) {
        const bytes = new Uint8Array(await (await new Promise((r) => canvas.toBlob(r, 'image/png'))).arrayBuffer());
        const digest = await crypto.subtle.digest('SHA-256', bytes);
        image = [...new Uint8Array(digest)].slice(0, 8).map((b) => b.toString(16).padStart(2, '0')).join('');
      }
      return {
        status: text(document.getElementById('status')),
        detail: text(document.getElementById('detail')),
        profile: text(document.getElementById('profile-out')),
        image,
      };
    })()`,
  );

  console.log(`\n${report.status}`);
  console.log(`image sha256[0:8] ${report.image ?? "(canvas empty)"}`);
  if (report.detail) console.log(report.detail);
  console.log();
  if (!report.profile) {
    console.log("measure: the page reported no profile. Nothing to read — not a profile of zeros.");
    process.exitCode = 1;
  } else {
    console.log(report.profile);
  }
}

try {
  await main();
} finally {
  cleanup();
}
