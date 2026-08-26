// conv3d, matching torch.nn.functional.conv3d.
//
// A cross-correlation, not a true convolution: tap (kd, kh, kw) reads forward
// in time and down and right in space from the window's corner, and the kernel
// is flipped on none of the three axes. See reference.ts for the measurement
// that settles it — in 3D that measurement is worth more than in 2D, because
// an axis read in the wrong order usually still produces a well-formed tensor.
//
// One thread per output element, as the 1D and 2D kernels beside it. Rule 8:
// the plain version has to agree with the reference before anything clever gets
// written, and tiling a 3D window is where clever would actually pay — every
// thread re-reads its neighbours' rows *and* its neighbours' frames. Issue #134
// owns picking a tile shape by measurement.
//
// Layout:
//   input:  [N, Cin, D, H, W]                 f32 — NCDHW, W contiguous
//   weight: [Cout, Cin/groups, KD, KH, KW]    f32
//   bias:   [Cout]                            f32 — required here; PyTorch's
//                                             bias=None is passed as zeros
//   output: [N, Cout, Dout, Hout, Wout]       f32
//
// Dispatch: x over Wout in 256-wide workgroups, y = Hout, z = N*Cout*Dout.
//
// The output has **five** axes against a dispatch's three, so z carries three
// of them: `z = (n * Cout + oc) * Dout + od`. Frames are fastest, which keeps
// one channel's frames adjacent in z — a decoder reads them together. The
// ceiling is real and worth naming: `maxComputeWorkgroupsPerDimension` is
// 65,535 on the device this was written against, so `N * Cout * Dout` has to
// stay under it. H3's visual VAE at its widest is Cout=1024 with ~12 latent
// frames, which is 12,288; a caller that exceeds it must split the dispatch,
// the way `dit-resident.ts` splits rmsnorm (issue #112).

struct Params {
  Cin: u32,
  Cout: u32,
  D: u32,
  H: u32,
  W: u32,
  KD: u32,
  KH: u32,
  KW: u32,
  Dout: u32,
  Hout: u32,
  Wout: u32,
  stride_d: u32,
  stride_h: u32,
  stride_w: u32,
  pad_d: u32,
  pad_h: u32,
  pad_w: u32,
  dilation_d: u32,
  dilation_h: u32,
  dilation_w: u32,
  // Cin / groups and Cout / groups. The kernel never needs `groups` itself,
  // only the two sizes it divides into — an integer division kept off every
  // thread, as in the 2D kernel.
  in_per_group: u32,
  out_per_group: u32,
  // Twenty-two words is 88 bytes; two reserved words reach 96, a multiple of
  // 16. Named rather than left implicit so `params()` on the host has a slot
  // to fill and the two cannot silently disagree about the size.
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
  let ow = gid.x;
  let oh = gid.y;
  let od = gid.z % params.Dout;
  let nc = gid.z / params.Dout;
  let n = nc / params.Cout;
  let oc = nc % params.Cout;

  // Wout is rarely a multiple of 256, so the last workgroup of each row runs
  // surplus threads. Unguarded they walk into the next row's output — or, at
  // the very end, past the buffer. Only x needs this: y and z are dispatched at
  // exactly one thread per unit.
  if (ow >= params.Wout) {
    return;
  }

  let group = oc / params.out_per_group;
  let ic_base = group * params.in_per_group;
  // Where this output's window starts in the input, before the pad is trimmed.
  // Signed: with padding all three are negative for the first few outputs.
  let frame0 = i32(od * params.stride_d) - i32(params.pad_d);
  let row0 = i32(oh * params.stride_h) - i32(params.pad_h);
  let col0 = i32(ow * params.stride_w) - i32(params.pad_w);

  var acc = bias[oc];
  for (var ic_local = 0u; ic_local < params.in_per_group; ic_local += 1u) {
    let in_vol = (n * params.Cin + ic_base + ic_local) * params.D * params.H * params.W;
    let w_vol = (oc * params.in_per_group + ic_local) * params.KD * params.KH * params.KW;
    for (var kd = 0u; kd < params.KD; kd += 1u) {
      let id = frame0 + i32(kd * params.dilation_d);
      // The zero pad, in six halves — two per axis. None can be left to the
      // hardware: this device reads past the end of a buffer as zero, which is
      // the right answer by accident, but one frame's out-of-range index is the
      // *next frame's* valid data, one row's is the next row's, and one
      // volume's is the next channel's. That is what actually comes back.
      if (id < 0) {
        continue;
      }
      if (id >= i32(params.D)) {
        continue;
      }
      let in_plane = in_vol + u32(id) * params.H * params.W;
      let w_plane = w_vol + kd * params.KH * params.KW;
      for (var kh = 0u; kh < params.KH; kh += 1u) {
        let ih = row0 + i32(kh * params.dilation_h);
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
  }

  output[(((n * params.Cout + oc) * params.Dout + od) * params.Hout + oh) * params.Wout + ow] = acc;
}
