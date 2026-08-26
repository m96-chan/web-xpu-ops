// pad, matching torch.nn.functional.pad on one axis.
//
// The tensor is viewed as [outer, L, inner] and only the middle axis is padded;
// see reference.ts for why one axis rather than several, and for the
// measurement that padding W and then H equals padding both at once.
//
// The three modes are the same table the reference states, and the one worth
// naming here is `reflect`: it does **not** repeat the edge element. Getting it
// and `replicate` the wrong way round produces a tensor of the right shape
// whose whole interior is correct.
//
// Layout:
//   input:  [outer, L, inner]      f32
//   output: [outer, Lout, inner]   f32, Lout = before + L + after
//
// Dispatch: one thread per *output* element, over a flat range tiled into
// (x, y). A pad is pure data movement, so there is nothing to reuse and nothing
// to tile for; the only question is whether every thread gets an element, and
// the flat range is what makes that true whatever the shape.
//
// **Why flat rather than one axis per dispatch dimension.** The obvious mapping
// -- x over `inner`, y over `Lout`, z over `outer` -- collapses when `inner` is
// 1, which is exactly the 1D audio case: 255 of every 256 threads would return
// immediately, and the workgroup count would go up by the same factor. The flat
// range costs two integer divisions per thread instead. That trade is
// **unmeasured**: `ops/flash_attention` learned this session that an integer
// division a GPU has no instruction for is worth about 4% there, so this is a
// real cost, not a free one -- it is simply the smaller of the two here, and
// nothing has yet measured by how much.

struct Params {
  // Elements before and after the padded axis, in the flattened view.
  outer: u32,
  L: u32,
  inner: u32,
  before: u32,
  Lout: u32,
  // 0 = constant, 1 = reflect, 2 = replicate. An integer rather than three
  // kernels: the branch is uniform across every thread in a dispatch, so it
  // costs a predictable jump and not a divergence.
  mode: u32,
  // `constant` only.
  value: f32,
  // How far one row of the (x, y) tile advances the flat index: the x extent of
  // the dispatch in threads. The host owns it, because only the host knows how
  // many workgroups it asked for.
  stride_y: u32,
}

@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;
@group(0) @binding(2) var<uniform> params: Params;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let flat = gid.x + gid.y * params.stride_y;
  let slab = params.Lout * params.inner;
  let total = params.outer * slab;
  // The flat range is almost never a multiple of the tile, so the last
  // workgroup runs surplus threads. Unguarded they write past the end of the
  // buffer, which this device drops silently -- so the guard is not what keeps
  // it from crashing, it is what keeps a wrong `stride_y` from being invisible.
  if (flat >= total) {
    return;
  }

  let o = flat / slab;
  let rem = flat - o * slab;
  let position = rem / params.inner;
  let k = rem - position * params.inner;

  // Signed: `position - before` is negative for every element before the data,
  // and the whole op is about what happens there.
  let source = i32(position) - i32(params.before);
  var reads = source;
  if (source < 0 || source >= i32(params.L)) {
    switch params.mode {
      case 0u: {
        output[flat] = params.value;
        return;
      }
      case 2u: {
        // replicate: clamp to the nearest real element.
        reads = select(i32(params.L) - 1, 0, source < 0);
      }
      default: {
        // reflect: mirror about the edge *element*, so it is not repeated.
        // -1 reads 1, and L reads L-2. The reference refuses a padding wider
        // than the axis, which is what keeps this index on the axis.
        reads = select(2 * (i32(params.L) - 1) - source, -source, source < 0);
      }
    }
  }

  output[flat] = input[(o * params.L + u32(reads)) * params.inner + k];
}
