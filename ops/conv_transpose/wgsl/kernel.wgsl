// conv_transpose1d, matching torch.nn.functional.conv_transpose1d.
//
// The reference states this as a scatter: every input sample times every kernel
// tap, deposited where that tap reaches. A shader cannot do that — many threads
// would accumulate into one address — so this inverts it into a gather, one
// thread per output element, asking which samples reach *here*. The two are
// compared against each other rather than sharing an index expression, which is
// the point: the inversion is where a transposed conv goes wrong.
//
// Inverting `ol = l*stride + k*dilation - padding` gives
// `l = (ol + padding - k*dilation) / stride`, and the tap only contributes when
// that division is exact — a strided transposed conv leaves stride-1 output
// positions between the samples that any given tap can reach.
//
// The kernel is NOT flipped, for the same reason conv1d's is not: tap k of
// sample l lands forward, at l*stride + k*dilation. See reference.ts.
//
// Layout:
//   input:  [N, Cin, L]              f32
//   weight: [Cin, Cout/groups, K]    f32 — Cin leads. The transpose of conv1d's
//                                    [Cout, Cin/groups, K], and the reason this
//                                    op is called transposed
//   bias:   [Cout]                   f32 — required here; PyTorch's bias=None
//                                    is passed as zeros, which costs one add
//                                    and saves a branch nothing can observe
//   output: [N, Cout, Lout]          f32
//
// `output_padding` is not a parameter here. It contributes nothing to the sum —
// it only widens Lout, which the host passes. The elements it appends fall out
// of the loop below on their own: they are either reached by a real tap (which
// happens whenever padding already cropped the tail) or by none, in which case
// they keep the bias. A branch for it would be a branch that never fires.
//
// Dispatch: x over Lout in 256-wide workgroups, y = Cout, z = N.

struct Params {
  Cin: u32,
  Cout: u32,
  L: u32,
  K: u32,
  Lout: u32,
  stride: u32,
  padding: u32,
  dilation: u32,
  // Cin / groups and Cout / groups. The kernel never needs `groups` itself,
  // only the two sizes it divides into, and dividing on the host keeps an
  // integer division out of every thread.
  in_per_group: u32,
  out_per_group: u32,
  // A uniform struct rounds up to a multiple of 16 bytes. Named rather than
  // implied, so the host packing twelve words is obviously deliberate.
  reserved_0: u32,
  reserved_1: u32,
}

@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read> weight: array<f32>;
@group(0) @binding(2) var<storage, read> bias: array<f32>;
@group(0) @binding(3) var<storage, read_write> output: array<f32>;
@group(0) @binding(4) var<uniform> params: Params;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let ol = gid.x;
  let oc = gid.y;
  let n = gid.z;

  // Lout is rarely a multiple of 256, so the last workgroup of each row runs
  // surplus threads. Unguarded they walk into the next channel's output.
  if (ol >= params.Lout) {
    return;
  }

  let group = oc / params.out_per_group;
  let oc_local = oc - group * params.out_per_group;
  let ic_base = group * params.in_per_group;

  // Where this output element sits in the uncropped convolution. `padding`
  // *crops* the output, so undoing the crop adds it back — the opposite sign
  // from conv1d, where padding extends the input.
  let full_ol = i32(ol) + i32(params.padding);

  var acc = bias[oc];
  for (var ic_local = 0u; ic_local < params.in_per_group; ic_local += 1u) {
    let ic = ic_base + ic_local;
    let in_row = (n * params.Cin + ic) * params.L;
    let w_row = (ic * params.out_per_group + oc_local) * params.K;
    for (var k = 0u; k < params.K; k += 1u) {
      let pos = full_ol - i32(k * params.dilation);
      // Only samples landing exactly on this output contribute. With stride > 1
      // most taps fall between samples, and truncating instead of rejecting
      // would fold a neighbouring sample in.
      if (pos % i32(params.stride) != 0) {
        continue;
      }
      // One unsigned comparison, not two. A tap reaching in from before the
      // start of the row gives a negative index, and u32() sends it to the far
      // end of the range, so `>= L` rejects both ends at once. Neither end can
      // be left to the hardware: this device reads past the end of a buffer as
      // zero, which is the right answer by accident, but one row's out-of-range
      // index is the next row's valid data, and that is what actually comes
      // back.
      let l = u32(pos / i32(params.stride));
      if (l >= params.L) {
        continue;
      }
      acc += input[in_row + l] * weight[w_row + k];
    }
  }

  output[(n * params.Cout + oc) * params.Lout + ol] = acc;
}
