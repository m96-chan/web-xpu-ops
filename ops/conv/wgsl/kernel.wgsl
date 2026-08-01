// conv1d, matching torch.nn.functional.conv1d.
//
// A cross-correlation, not a true convolution: tap k reads forward from the
// window start and the kernel is never flipped. See reference.ts for the
// measurement that settles it.
//
// One thread per output element. That is all the parallelism this shape has
// before tiling, and rule 8 says the plain version has to agree with the
// reference before anything clever gets written.
//
// Layout:
//   input:  [N, Cin, L]              f32
//   weight: [Cout, Cin/groups, K]    f32
//   bias:   [Cout]                   f32 — required here; PyTorch's bias=None
//                                    is passed as zeros, which costs one add
//                                    and saves a branch nothing can observe
//   output: [N, Cout, Lout]          f32
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
  let ic_base = group * params.in_per_group;
  // Where this output's window starts in the input, before the pad is trimmed.
  // Signed: with padding it is negative for the first few outputs.
  let window = i32(ol * params.stride) - i32(params.padding);

  var acc = bias[oc];
  for (var ic_local = 0u; ic_local < params.in_per_group; ic_local += 1u) {
    let in_row = (n * params.Cin + ic_base + ic_local) * params.L;
    let w_row = (oc * params.in_per_group + ic_local) * params.K;
    for (var k = 0u; k < params.K; k += 1u) {
      let il = window + i32(k * params.dilation);
      // The zero pad, in two halves. Neither can be left to the hardware: this
      // device reads past the end of a buffer as zero, which is the right
      // answer by accident, but one row's out-of-range index is the next row's
      // valid data, and that is what actually comes back.
      if (il < 0) {
        continue;
      }
      if (il >= i32(params.L)) {
        continue;
      }
      acc += input[in_row + u32(il)] * weight[w_row + k];
    }
  }

  output[(n * params.Cout + oc) * params.Lout + ol] = acc;
}
