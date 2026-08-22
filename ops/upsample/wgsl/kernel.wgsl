// nearestUpsample2d, matching
// torch.nn.functional.interpolate(x, size=(outH, outW), mode='nearest').
//
// No weights and no arithmetic on the values: every output element is a bitwise
// copy of an input element, chosen by floor(dst * scale) per axis. See
// reference.ts for the measurements that settle the index formula — in
// particular why the multiply is f32 rather than exact integer arithmetic, and
// why `align_corners` has no meaning here.
//
// One thread per output element. That is all the parallelism this shape has,
// and rule 8 says the plain version has to agree with the reference before
// anything clever gets written.
//
// Layout:
//   input:  [N, C, H, W]        f32
//   output: [N, C, outH, outW]  f32
//
// N and C never appear apart — the resample is per (n, c) plane — so they are
// one flat plane index here, exactly as in the reference. That also keeps this
// off the 65535 limit on dispatch dimensions y and z for a batch times channel
// count that a decoder can reach.
//
// Dispatch: x over outW in 256-wide workgroups, y = outH, z = N * C.

struct Params {
  H: u32,
  W: u32,
  out_h: u32,
  out_w: u32,
  // inSize / outSize, rounded to f32 **on the host**. Not computed here:
  // WGSL requires f32 `*` to be correctly rounded but allows `/` up to 2.5 ULP
  // of error, and this ratio's last bit decides whole rows of output (H=14 ->
  // 46 in reference.ts). A shader-side division would make the answer depend on
  // a rounding this code does not control; a multiply does not.
  //
  // Measured, not assumed. Replacing `params.scale_h` with
  // `f32(params.H) / f32(params.out_h)` in the line below turns the 14 -> 46
  // case red on this machine: output index 1150 (destination row 23, the row
  // whose exact source is the integer 7) comes back as source row 7 where
  // torch and the reference both say 6. NVIDIA GeForce RTX 5090 (blackwell,
  // NVIDIA 610.57.04), Dawn via the `webgpu` package, f32. So the divided
  // scale here is not a theoretical hazard from the spec — this GPU's `/`
  // really does land on the other side of the boundary.
  scale_h: f32,
  scale_w: f32,
  // A uniform struct rounds up to a multiple of 16 bytes. Named rather than
  // implied, so the host packing six words is obviously deliberate.
  reserved_0: u32,
  reserved_1: u32,
}

@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;
@group(0) @binding(2) var<uniform> params: Params;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let ow = gid.x;
  let oh = gid.y;
  let plane = gid.z;

  // outW is rarely a multiple of 256, so the last workgroup of each row runs
  // surplus threads. Unguarded they walk into the next row's output — an index
  // that is still inside the buffer, so nothing faults and a real value is
  // replaced instead.
  if (ow >= params.out_w) {
    return;
  }

  // f32(u32) is correctly rounded and the product is a correctly rounded f32
  // multiply, which is the same pair of operations torch performs in
  // `nearest_neighbor_compute_source_index` and `Math.fround(oh * scaleH)`
  // performs in the reference. floor() of a non-negative value is trunc(), but
  // it is written as floor because that is what the formula says.
  let ih = u32(floor(f32(oh) * params.scale_h));
  let iw = u32(floor(f32(ow) * params.scale_w));

  // No clamp to H-1 / W-1: outSize >= inSize makes the source index provably
  // in range. reference.ts has the algebra and the search that back that.
  let in_plane = plane * params.H * params.W;
  let out_plane = plane * params.out_h * params.out_w;
  output[out_plane + oh * params.out_w + ow] = input[in_plane + ih * params.W + iw];
}
