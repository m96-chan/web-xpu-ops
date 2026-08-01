// Greedy CTC decode: argmax per frame, collapse repeats, then drop blanks.
//
// One workgroup per batch row. Within a row the frames are walked in order —
// the collapse is a running comparison against the previous frame's label, so
// the scan is serial by construction — and the parallelism is spent on the
// argmax across classes, which is where the reads are.
//
// Layout:
//   scores:  [B, T, C] f32   frame scores; logits, log-probs or probabilities
//   tokens:  [B, T]    i32   labels, each row padded to T with -1
//   lengths: [B]       i32   labels emitted per row
//
// The lengths are written here rather than inferred by the caller. A greedy
// decode emits at most one label per frame, so `[B, T]` always fits and the
// allocation never depends on the answer — which is what lets the result stay
// on the GPU. Reading the tokens back to find out how long they are would be
// the readback this op exists to avoid.

struct Params {
  B: u32,
  T: u32,
  C: u32,
  // Class index of the blank. 0 by default in the caller, as torch.nn.CTCLoss
  // has it; a value outside [0, C) simply means no class is the blank.
  blank: u32,
}

@group(0) @binding(0) var<storage, read> scores: array<f32>;
@group(0) @binding(1) var<storage, read_write> tokens: array<i32>;
@group(0) @binding(2) var<storage, read_write> lengths: array<i32>;
@group(0) @binding(3) var<uniform> params: Params;

const WORKGROUP_SIZE: u32 = 256u;

/** Fills tokens past the end of a row's result. Not a valid class index. */
const PAD: i32 = -1;

// The identity for the argmax. It has to be below every score a model can
// produce, and acoustic models emit log-probabilities — negative everywhere —
// so zero is not it.
// (-3.40282347e+38, the exact f32 minimum, is rejected by Dawn: the decimal
// literal rounds just past it. One ulp in is still below every real score.)
const LOWEST: f32 = -3.4028234e+38;

var<workgroup> top_score: array<f32, 256>;
var<workgroup> top_class: array<u32, 256>;

@compute @workgroup_size(256)
fn main(
  @builtin(workgroup_id) wg_id: vec3<u32>,
  @builtin(local_invocation_id) local_id: vec3<u32>,
) {
  let row = wg_id.x;
  let tid = local_id.x;
  let base = row * params.T * params.C;

  // Seeded with the blank rather than a "no label yet" sentinel, and the two
  // are equivalent: at t = 0 the frame is emitted exactly when its label is not
  // the blank under either seed, because the blank test already covers it.
  var previous: u32 = params.blank;
  // Held in a register, not workgroup memory: every lane runs the same
  // comparison on the same reduced label, so every lane keeps the same count.
  // Only lane 0 writes, but all of them agree on where.
  var count: u32 = 0u;

  for (var t: u32 = 0u; t < params.T; t += 1u) {
    let frame = base + t * params.C;

    // argmax over the classes, lanes striding by the workgroup width.
    var best: f32 = LOWEST;
    var best_c: u32 = 0u;
    for (var c = tid; c < params.C; c += WORKGROUP_SIZE) {
      let value = scores[frame + c];
      // Strictly greater, so a lane keeps the first of its own tied maxima —
      // `c` only increases here, which is the whole tie-break within a lane.
      // Across lanes it is not, and the reduction below has to say so.
      if (value > best) {
        best = value;
        best_c = c;
      }
    }
    top_score[tid] = best;
    top_class[tid] = best_c;
    workgroupBarrier();

    for (var stride = WORKGROUP_SIZE / 2u; stride > 0u; stride >>= 1u) {
      if (tid < stride) {
        let other = top_score[tid + stride];
        let other_c = top_class[tid + stride];
        // Ties fall to the lower class index, as torch.argmax does. Lane order
        // is not class order once C passes 256 — class 260 lands on lane 4 and
        // class 5 on lane 5 — so the comparison has to be on the class, not on
        // which slot it arrived in.
        if (other > top_score[tid] || (other == top_score[tid] && other_c < top_class[tid])) {
          top_score[tid] = other;
          top_class[tid] = other_c;
        }
      }
      workgroupBarrier();
    }

    let label = top_class[0];
    // Collapse against the previous frame's label — blank included — and only
    // then drop the blank. The other order loses a doubled label that a blank
    // was there to separate, which is the one thing the blank is for.
    if (label != previous && label != params.blank) {
      if (tid == 0u) {
        tokens[row * params.T + count] = i32(label);
      }
      count += 1u;
    }
    previous = label;

    // The next frame overwrites the scratch this frame was just read from.
    workgroupBarrier();
  }

  if (tid == 0u) {
    lengths[row] = i32(count);
  }
  for (var i = count + tid; i < params.T; i += WORKGROUP_SIZE) {
    tokens[row * params.T + i] = PAD;
  }
}
