import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { describe, expect, it } from "vitest";

/**
 * Stops anyone from deleting the one reference that keeps this binding alive.
 *
 * `wgsl.ts#createRunner` and `resident.ts#createResidentDevice` each push their
 * `GPU` instance and adapter into a module-scope array. Nothing reads those
 * arrays. To a reader — or a linter, or someone tidying up — they look like
 * dead code, and removing them costs nothing visible: the suite still compiles,
 * the types still check, and a short GPU test still passes. What breaks is
 * everything longer, in a way that does not name its cause (issue #107: the
 * process aborts on a glibc futex assertion, segfaults, or hangs, and which one
 * varies per run).
 *
 * So this asserts the retention exists, by reading the source. That is a
 * deliberately shallow check — it cannot tell whether the array is reachable at
 * the moment the collector runs — and it is chosen over the behavioural
 * alternative on purpose:
 *
 * A test that asserted "removing the retention makes dispatches fail" would
 * have to treat SIGABRT, SIGSEGV and a hang as three spellings of success,
 * hand the runner the job of interpreting signals and timeouts, and would still
 * be wrong on a slower machine or a different driver where the collector does
 * not get to run inside the loop. It would be a new source of flake, added to
 * fix a source of flake.
 *
 * The behavioural half is already covered without writing anything: with the
 * retention removed, `ops/gqa/wgsl.test.ts` goes from 46/46 to producing no
 * test result at all. The existing GPU suite is the sentry; this file only
 * keeps the line from being deleted by someone who has no way to know what it
 * is for.
 */

const root = new URL("../", import.meta.url);

function source(name: string): string {
  return readFileSync(fileURLToPath(new URL(name, root)), "utf8");
}

describe("GPU instance retention (#107)", () => {
  // Each entry: the file that creates a device, and the array it must push into.
  const sites = [
    { file: "harness/wgsl.ts", holder: "retainedInstances" },
    { file: "harness/resident.ts", holder: "retainedResidentInstances" },
  ];

  for (const { file, holder } of sites) {
    it(`${file} keeps its GPU instance and adapter reachable`, () => {
      const text = source(file);

      // The holder has to exist at module scope, not inside the factory —
      // an array declared inside the function it protects dies with the call.
      expect(text, `${file}: no module-scope \`${holder}\``).toMatch(
        new RegExp(`^const ${holder}\\b`, "m"),
      );

      // Both objects, not just the adapter: the instance is the one the
      // binding fails to keep alive, and the adapter is what reaches it.
      expect(text, `${file}: \`${holder}\` must be given the instance and the adapter`).toMatch(
        new RegExp(`${holder}\\.push\\([^)]*\\bgpu\\b[^)]*\\badapter\\b[^)]*\\)`),
      );

      // The push has to happen where the device is built. A push guarded by a
      // branch that the adapter-less path takes would retain nothing.
      const pushAt = text.indexOf(`${holder}.push(`);
      const deviceAt = text.indexOf("requestDevice(");
      expect(pushAt, `${file}: no push found`).toBeGreaterThan(-1);
      expect(pushAt, `${file}: retention must precede requestDevice()`).toBeLessThan(deviceAt);
    });
  }
});
