// Transpose of a row-major 2D matrix: output[c][r] = input[r][c].
//
// Layout:
//   input:  [rows, cols] f32
//   output: [cols, rows] f32
//
// There is no arithmetic here, so the only two things this kernel can do badly
// are put a value in the wrong place and reach memory badly. One thread per
// element, reading input[y * cols + x] and writing output[x * rows + y], gets
// the first right and the second half right: neighbouring threads read
// neighbouring addresses but write addresses `rows` apart, so every write costs
// its own transaction.
//
// So the turn happens in workgroup memory instead of in the buffer. A 16x16
// tile — the 256-wide workgroup, square — is read row-wise, then read back out
// column-wise and written row-wise into the output. Both halves of the trip
// through global memory are consecutive; the strided step is against workgroup
// storage, which does not care.
//
// Dispatch is the *input* in tiles: [ceil(cols / 16), ceil(rows / 16)].

struct Params {
  rows: u32,
  cols: u32,
}

@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;
@group(0) @binding(2) var<uniform> params: Params;

const TILE: u32 = 16u;

// Row stride 16, not the customary 17. Padding the row is the usual answer to
// the bank conflict on the column-wise read below, and it was measured here
// rather than assumed: at 4096x4096 the two strides came out at 0.090 ms
// against 0.090 ms, a ratio of 1.002, over three interleaved rounds of 60
// dispatches each. A transpose moves every byte exactly twice and computes
// nothing, so global bandwidth binds it long before workgroup storage does.
// The padding buys nothing measurable and costs a line no test can hold — worth
// revisiting under #3/#4, where there is a roofline to compare against.
var<workgroup> tile: array<array<f32, 16>, 16>;

@compute @workgroup_size(16, 16)
fn main(
  @builtin(workgroup_id) wg: vec3<u32>,
  @builtin(local_invocation_id) lid: vec3<u32>,
) {
  let lx = lid.x;
  let ly = lid.y;

  // Read the tile. x runs along a row of the input, so the sixteen lanes of one
  // ly read sixteen consecutive f32.
  //
  // No bounds check on this read, deliberately. WGSL bounds-checks storage
  // accesses — an out-of-range read yields some in-range value rather than
  // faulting — and the write below already discards exactly the elements a
  // check here would discard: it is the same pair of conditions seen from the
  // other side, because (lx, ly) -> (ly, lx) is a bijection on the 16x16 thread
  // block. A guard here would be a line no test could ever turn red.
  let in_x = wg.x * TILE + lx;
  let in_y = wg.y * TILE + ly;
  tile[ly][lx] = input[in_y * params.cols + in_x];

  workgroupBarrier();

  // Write it out turned. Reading tile[lx][ly] is where the transpose happens;
  // the address is again consecutive across the sixteen lanes of one ly, now
  // along a row of the output.
  //
  // The guard is not about faulting. On a shape that does not divide by 16 the
  // leftover threads compute an index that is still inside the output buffer,
  // so writing anyway does not crash — it replaces a real value somewhere else
  // in the matrix.
  let out_x = wg.y * TILE + lx;
  let out_y = wg.x * TILE + ly;
  if (out_x < params.rows && out_y < params.cols) {
    output[out_y * params.rows + out_x] = tile[lx][ly];
  }
}
