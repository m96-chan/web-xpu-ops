import { describe, expect, it } from "vitest";
import { compilationFailure } from "./wgsl.js";

/**
 * Why this op exists at all.
 *
 * Output buffers start zeroed, so a dispatch that never ran reads back as
 * zeros. Where an expected value contains zeros, that is a passing test over a
 * kernel that executed nothing — and since every correctness claim in this
 * repository is "it agrees with the reference", a silent no-op reading as
 * agreement is the one failure that could hollow out all of them at once.
 * Measured before the fix (issue #46): a shader with an undefined function read
 * back `[0, 0]` with no error raised anywhere.
 *
 * Why it is tested here rather than end to end.
 *
 * Provoking a real compile failure through the harness crashes this Dawn
 * binding in roughly four runs in five — measured across ten runs each for the
 * error scope, `getCompilationInfo`, validation before and after buffer
 * allocation, `beforeAll` versus in-test device creation, and both file
 * locations. The same sequence written inline without going through
 * `createRunner` is 5 of 5, so it is not the check itself. An end-to-end test
 * therefore cannot be kept green, and a test that must be re-run until it
 * passes is exactly what rule 1 forbids.
 *
 * So the split is deliberate: whether Dawn reports a bad shader is platform
 * behaviour, verified by hand and recorded in #46. Whether *we* refuse when it
 * does is ours, and that is what these assert — with no device, so they cannot
 * flake.
 */
describe("compilationFailure", () => {
  it("passes a shader that compiled", () => {
    expect(compilationFailure([])).toBeNull();
  });

  it("passes when the only messages are warnings and info", () => {
    // A warning must not fail a dispatch: WGSL emits them for things like
    // unreachable code, and refusing on those would block working kernels.
    expect(
      compilationFailure([
        { type: "warning", lineNum: 3, linePos: 5, message: "unreachable statement" },
        { type: "info", lineNum: 0, linePos: 0, message: "compiled in 2ms" },
      ]),
    ).toBeNull();
  });

  it("refuses when any message is an error, and says where", () => {
    const failure = compilationFailure([
      { type: "error", lineNum: 5, linePos: 36, message: "unresolved call to 'notAFunction'" },
    ]);
    expect(failure).toContain("failed to compile");
    expect(failure).toContain("5:36");
    expect(failure).toContain("notAFunction");
  });

  it("refuses on an error even when warnings outnumber it", () => {
    // The filter has to select errors rather than look at the first or last
    // message: a real failure usually arrives buried in warnings.
    const failure = compilationFailure([
      { type: "warning", lineNum: 1, linePos: 1, message: "first" },
      { type: "error", lineNum: 9, linePos: 2, message: "the real problem" },
      { type: "warning", lineNum: 12, linePos: 1, message: "last" },
    ]);
    expect(failure).toContain("the real problem");
    expect(failure).not.toContain("first");
    expect(failure).not.toContain("last");
  });

  it("reports every error, not just the first", () => {
    const failure = compilationFailure([
      { type: "error", lineNum: 2, linePos: 1, message: "alpha" },
      { type: "error", lineNum: 7, linePos: 4, message: "beta" },
    ]);
    expect(failure).toContain("alpha");
    expect(failure).toContain("beta");
  });
});
