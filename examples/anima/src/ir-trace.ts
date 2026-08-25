/**
 * What an IR would have to carry, taken from a forward rather than imagined.
 *
 * Issue #185. The question is whether a model's forward can be data instead of
 * 1,035 lines of TypeScript, and the honest way to answer it is to record what
 * the TypeScript actually produces and look at what varies.
 *
 * Every dispatch is written down with its kernel, its workgroup counts, the
 * **decoded** uniform values, and an identity for each buffer it binds. Run two
 * resolutions and diff:
 *
 *   - what is **identical** is structure, and an IR can hold it as a constant;
 *   - what **differs** is shape-derived, and an IR needs an expression for it;
 *   - what differs *unpredictably* is control flow, and an IR needs to say so.
 *
 * The buffer identities are the part `ResidentOp` throws away — a `GPUBindGroup`
 * cannot be asked what went into it, so a trace taken after the fact has nodes
 * and no edges. Recorded here at the moment they are known.
 *
 * **A probe.** Off unless a driver turns it on, and nothing on a shipping path
 * reads it.
 */

export interface IrDispatch {
  index: number;
  kernel: string;
  workgroups: number[];
  /** Buffer identities, in binding order. Stable within one forward. */
  buffers: number[];
  /** Which of those the shader may write. Unknown here, so the trace records all of them. */
  uniforms: number[];
  /** Label from the profiler's own naming — `net.blocks.7.` and so on. */
  label: string | null;
}

export interface IrTrace {
  dispatches: IrDispatch[];
  /** Buffer id to the byte length it was created with, so shapes can be checked. */
  bufferBytes: Map<number, number>;
}

let trace: IrTrace | null = null;
const ids = new WeakMap<GPUBuffer, number>();
let nextId = 0;

export function startIrTrace(): IrTrace {
  trace = { dispatches: [], bufferBytes: new Map() };
  return trace;
}

export function stopIrTrace(): IrTrace | null {
  const done = trace;
  trace = null;
  return done;
}

export function irTracing(): boolean {
  return trace !== null;
}

/**
 * Every dispatch must be noted, and this is how that is enforced.
 *
 * The first version only hooked `record()`, so the dispatches that go through
 * `bindGroupSliced` were invisible. The probe reported `rmsnorm` falling from
 * 209 to 53 between two shapes and I read that as control flow. It was the
 * instrument: a trace that silently omits a third of the work looks exactly
 * like a trace of different work.
 *
 * `push` counts what actually goes into the batch; `noteDispatch` counts what
 * the trace saw. A mismatch is an unrouted call site, and it throws rather than
 * producing a plausible number.
 */
export function checkTraceCoverage(dispatchesPushed: number): void {
  if (!trace) return;
  if (trace.dispatches.length !== dispatchesPushed) {
    throw new Error(
      `ir-trace: ${dispatchesPushed} dispatches were recorded into the batch but ${trace.dispatches.length} ` +
        "reached the trace. Some call site does not go through `noteDispatch`, and a trace missing part of " +
        "the work reads as a trace of different work.",
    );
  }
}

/** A stable small integer per buffer, so a trace can be diffed between runs. */
function idOf(buffer: GPUBuffer): number {
  let id = ids.get(buffer);
  if (id === undefined) {
    id = nextId++;
    ids.set(buffer, id);
    trace?.bufferBytes.set(id, buffer.size);
  }
  return id;
}

/**
 * Records one dispatch.
 *
 * `uniforms` are the decoded words of whichever binding is a uniform buffer —
 * the caller knows which, because it packed them. Passed in rather than read
 * back, since reading a uniform buffer from the GPU would need a copy and a
 * map, and the values are in hand at the call site anyway.
 */
export function noteDispatch(
  kernel: string,
  buffers: GPUBuffer[],
  workgroups: readonly number[],
  uniforms: number[],
  label: string | null,
): void {
  if (!trace) return;
  trace.dispatches.push({
    index: trace.dispatches.length,
    kernel,
    workgroups: [...workgroups],
    buffers: buffers.map(idOf),
    uniforms,
    label,
  });
}
