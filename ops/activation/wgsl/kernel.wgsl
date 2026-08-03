// Elementwise activations.
//
// Every kind here is a pure function of one element: no layout, no second
// buffer, `N` is all the kernel is told. Snake looks like it belongs and does
// not — its alpha is a trained per-channel tensor, so it needs a storage
// binding and a channel stride, and it lives in `ops/snake`. See
// `ops/snake/reference.ts` for that decision.
//
//   0 ReLU²      max(0, x)²                     BitNet 2B-4T FFN
//   1 SiLU       x · sigmoid(x)                 torch.nn.SiLU
//   2 ELU        x if x > 0 else α(eˣ - 1)      torch.nn.ELU, α default 1.0
//   3 Tanh       tanh(x)                        torch.tanh
//   4 GELU       0.5x(1 + erf(x/√2))            torch gelu, approximate="none"
//   5 GELU tanh  0.5x(1 + tanh(√(2/π)(x + 0.044715x³)))   approximate="tanh"
//
// 4 and 5 are different functions, not two spellings of one: they part company
// by 4.73e-4 at x = 2.699. torch's default is 4, so this op's default is 4.
//
// Layout:
//   input:  [N] f32
//   output: [N] f32

struct Params {
  N: u32,
  activation_type: u32,
  alpha: f32,  // ELU's negative-branch scale; ignored by every other kind
}

@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;
@group(0) @binding(2) var<uniform> params: Params;

/// erfc for u >= 0: Abramowitz & Stegun 7.1.26, whose stated error bound is
/// 1.5e-7 — the ulp of 1 in f32, so it is as close as this type gets. WGSL has
/// no erf and no erfc, so the exact GELU has to carry one.
fn erfc_nonneg(u: f32) -> f32 {
  let t = 1.0 / (1.0 + 0.3275911 * u);
  let poly = t * (0.254829592
      + t * (-0.284496736
      + t * (1.421413741
      + t * (-1.453152027
      + t * 1.061405429))));
  return poly * exp(-u * u);
}

@compute @workgroup_size(256)
fn main(
  @builtin(global_invocation_id) gid: vec3<u32>,
) {
  let idx = gid.x;
  if (idx >= params.N) {
    return;
  }

  let x = input[idx];
  var y: f32;

  switch (params.activation_type) {
    case 0u: {
      // ReLU²: max(0, x)²
      let relu_x = max(0.0, x);
      y = relu_x * relu_x;
    }
    case 1u: {
      // SiLU: x * sigmoid(x)
      y = x / (1.0 + exp(-x));
    }
    case 2u: {
      // ELU. `x > 0` as torch writes it; at x = 0 both arms are 0 anyway.
      if (x > 0.0) {
        y = x;
      } else {
        y = params.alpha * (exp(x) - 1.0);
      }
    }
    case 3u: {
      y = tanh(x);
    }
    case 4u: {
      // GELU, exact. Written through erfc rather than erf so neither side
      // cancels: for x >= 0, 1 + erf(u) is 2 - erfc(u), and for x < 0 it is
      // erfc(|u|) outright, rather than 1 + (1 - erfc) — which for x = -8 is a
      // subtraction between two f32 numbers a hair either side of 1, and
      // returns exactly zero where the answer is -3e-14.
      //
      // Stated as a contract, not as something these tests check: the naive
      // arrangement was written in and the suite stayed green (measured). It
      // cannot catch it, for two compounding reasons — the harness passes an
      // element on absolute agreement below 1e-6, and every value this affects
      // is far below that; and the f64 reference computes `1 + erf` and so
      // cancels in the same place. Kept because it is free and right, not
      // because a red test demanded it.
      let u = x * 0.70710678118654752;
      let q = erfc_nonneg(abs(u));
      if (x >= 0.0) {
        y = x * (1.0 - 0.5 * q);
      } else {
        y = 0.5 * x * q;
      }
    }
    default: {
      // GELU, tanh approximation. √(2/π) = 0.7978845608028654.
      y = 0.5 * x * (1.0 + tanh(0.7978845608028654 * (x + 0.044715 * x * x * x)));
    }
  }

  output[idx] = y;
}
