// conv2d, matching torch.nn.functional.conv2d.
//
// A cross-correlation, not a true convolution: tap (kh, kw) reads down and
// right from the window's top-left corner and the kernel is flipped on neither
// axis. See reference.ts for the measurement that settles it.
//
// One thread per output element, as the 1D kernel beside it. That is all the
// parallelism this shape has before tiling, and rule 8 says the plain version
// has to agree with the reference before anything clever gets written. Tiling
// is what would actually pay here — every thread in a workgroup re-reads its
// neighbours' rows — but ISSUE #134 owns picking a tile shape *by measurement*,
// and choosing one here would be the "probably faster" rule 2 forbids.
//
// Layout:
//   input:  [N, Cin, H, W]                f32 — NCHW, W contiguous
//   weight: [Cout, Cin/groups, KH, KW]    f32
//   bias:   [Cout]                        f32 — required here; PyTorch's
//                                         bias=None is passed as zeros, which
//                                         costs one add and saves a branch
//                                         nothing can observe
//   output: [N, Cout, Hout, Wout]         f32
//
// Dispatch: x over Wout in 256-wide workgroups, y = Hout, z = N * Cout.
//
// Two decisions in that line, neither of them measured for speed:
//
//   - 256 over **Wout** rather than over Hout, because W is the contiguous
//     axis: consecutive threads then read and write consecutive addresses.
//     Unmeasured as a *number* — it is 1D's workgroup size, kept so that any
//     tuning #134 does lands on one shape rather than two.
//   - The output has four axes and a dispatch has three, so z carries two.
//     N is the slower of the pair (z = n * Cout + oc), which keeps one
//     sample's channels adjacent in z.

struct Params {
  Cin: u32,
  Cout: u32,
  H: u32,
  W: u32,
  KH: u32,
  KW: u32,
  Hout: u32,
  Wout: u32,
  stride_h: u32,
  stride_w: u32,
  pad_h: u32,
  pad_w: u32,
  dilation_h: u32,
  dilation_w: u32,
  // Cin / groups and Cout / groups. The kernel never needs `groups` itself,
  // only the two sizes it divides into, and dividing on the host keeps an
  // integer division out of every thread.
  in_per_group: u32,
  out_per_group: u32,
  // Sixteen words is 64 bytes exactly, so unlike the 1D kernel's twelve this
  // struct needs no reserved tail to reach a multiple of 16.
}

@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read> weight: array<f32>;
@group(0) @binding(2) var<storage, read> bias: array<f32>;
@group(0) @binding(3) var<storage, read_write> output: array<f32>;
@group(0) @binding(4) var<uniform> params: Params;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let ow = gid.x;
  let oh = gid.y;
  let n = gid.z / params.Cout;
  let oc = gid.z % params.Cout;

  // Wout is rarely a multiple of 256, so the last workgroup of each row runs
  // surplus threads. Unguarded they walk into the next row's output — or, on
  // the very last row, past the end of the buffer. Only x needs this: y and z
  // are dispatched at exactly one thread per unit, so a guard there would be a
  // branch nothing could ever take.
  if (ow >= params.Wout) {
    return;
  }

  let group = oc / params.out_per_group;
  let ic_base = group * params.in_per_group;
  // Where this output's window starts in the input, before the pad is trimmed.
  // Signed: with padding both are negative for the first few outputs.
  let row0 = i32(oh * params.stride_h) - i32(params.pad_h);
  let col0 = i32(ow * params.stride_w) - i32(params.pad_w);

  var acc = bias[oc];
  for (var ic_local = 0u; ic_local < params.in_per_group; ic_local += 1u) {
    let in_plane = (n * params.Cin + ic_base + ic_local) * params.H * params.W;
    let w_plane = (oc * params.in_per_group + ic_local) * params.KH * params.KW;
    for (var kh = 0u; kh < params.KH; kh += 1u) {
      let ih = row0 + i32(kh * params.dilation_h);
      // The zero pad, in four halves — two here and two below. None of them can
      // be left to the hardware: this device reads past the end of a buffer as
      // zero, which is the right answer by accident, but one row's
      // out-of-range index is the next row's valid data, and one plane's is the
      // next plane's, and that is what actually comes back.
      if (ih < 0) {
        continue;
      }
      if (ih >= i32(params.H)) {
        continue;
      }
      let in_row = in_plane + u32(ih) * params.W;
      let w_row = w_plane + kh * params.KW;
      for (var kw = 0u; kw < params.KW; kw += 1u) {
        let iw = col0 + i32(kw * params.dilation_w);
        if (iw < 0) {
          continue;
        }
        if (iw >= i32(params.W)) {
          continue;
        }
        acc += input[in_row + u32(iw)] * weight[w_row + kw];
      }
    }
  }

  output[((n * params.Cout + oc) * params.Hout + oh) * params.Wout + ow] = acc;
}
