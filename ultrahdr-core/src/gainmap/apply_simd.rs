//! Row-level gain map application kernels (scalar + SIMD).
//!
//! Two scalar kernels, both row-level:
//!
//! | Kernel | Gain input | Offsets | Used by |
//! |---|---|---|---|
//! | [`apply_gain_row_presampled`] | Pre-sampled `[f32; 3]` per pixel | base + alternate, per channel | `apply_gainmap`, streaming decoder |
//! | [`apply_gain_row_scalar`] | Raw `u8` byte + 256-entry LUT | none — plain multiply | SIMD consistency tests |
//!
//! The presampled kernel is what the real decode path uses: the caller
//! first runs bilinear interpolation + per-channel LUT lookup into an
//! `[f32; 3]` buffer, then applies the full ISO 21496-1 formula:
//! `hdr_i = (sdr_i + base_offset_i) * gain_i - alternate_offset_i`.
//!
//! The byte-input kernel is retained because it maps 1:1 onto the SIMD
//! dispatch path below and is useful for same-resolution single-channel
//! gain maps where bilinear interpolation is a no-op.
//!
//! SIMD dispatch (`simd` feature): via `#[magetypes]` generics and `incant!`:
//!
//! - **AVX2+FMA** on x86_64: 8 pixels per iteration
//! - **NEON** on aarch64: 8 pixels per iteration (generic f32x8)
//! - **WASM SIMD128**: 8 pixels per iteration (generic f32x8)
//! - **Scalar** everywhere else: 8 pixels per iteration (scalar f32x8)
//!
//! All kernels operate on pre-linearized `[f32; 3]` RGB pixels.

/// Apply the ISO 21496-1 gain formula to a row of pre-sampled gains.
///
/// For each pixel `i` and channel `c`:
/// `output[i][c] = (sdr[i][c] + base_offset[c]) * gains[i][c] - alternate_offset[c]`.
///
/// `gains` is already post-bilinear-interpolation and post-LUT per channel.
/// For single-channel gain maps, the caller broadcasts the same gain to
/// `[g, g, g]` before calling.
///
/// # Panics
///
/// Panics if `sdr`, `gains`, and `output` have different lengths.
pub fn apply_gain_row_presampled(
    sdr: &[[f32; 3]],
    gains: &[[f32; 3]],
    base_offset: [f32; 3],
    alternate_offset: [f32; 3],
    output: &mut [[f32; 3]],
) {
    assert_eq!(sdr.len(), output.len());
    assert_eq!(sdr.len(), gains.len());

    for ((sdr_px, gain_px), out_px) in sdr.iter().zip(gains.iter()).zip(output.iter_mut()) {
        out_px[0] = (sdr_px[0] + base_offset[0]) * gain_px[0] - alternate_offset[0];
        out_px[1] = (sdr_px[1] + base_offset[1]) * gain_px[1] - alternate_offset[1];
        out_px[2] = (sdr_px[2] + base_offset[2]) * gain_px[2] - alternate_offset[2];
    }
}

/// Scalar reference implementation for gain map application.
///
/// Applies a single-channel gain LUT to each pixel:
///   `output[i] = sdr[i] * lut[gainmap[i]]`
///
/// Both `sdr` and `output` are `[f32; 3]` RGB pixels. The gain map is
/// single-channel (one `u8` per pixel), and the LUT maps each byte value
/// to a linear gain multiplier.
///
/// # Panics
///
/// Panics if `sdr`, `gainmap`, and `output` have different lengths.
pub fn apply_gain_row_scalar(
    sdr: &[[f32; 3]],
    gainmap: &[u8],
    lut: &[f32; 256],
    output: &mut [[f32; 3]],
) {
    assert_eq!(sdr.len(), output.len());
    assert_eq!(sdr.len(), gainmap.len());

    for (i, (sdr_px, out_px)) in sdr.iter().zip(output.iter_mut()).enumerate() {
        let g = lut[gainmap[i] as usize];
        out_px[0] = sdr_px[0] * g;
        out_px[1] = sdr_px[1] * g;
        out_px[2] = sdr_px[2] * g;
    }
}

// ============================================================================
// SIMD dispatch (requires `simd` feature)
// ============================================================================

// On aarch64 the generic kernel (and therefore the archmage/magetypes
// imports) is not referenced at all — `apply_gain_row_simd` routes to the
// measured-faster scalar kernel there — so everything below is compiled out
// to keep `-D warnings` builds green (dead_code / unused_imports).
#[cfg(all(feature = "simd", not(target_arch = "aarch64")))]
use archmage::prelude::*;
#[cfg(all(feature = "simd", not(target_arch = "aarch64")))]
use magetypes::simd::generic::f32x8 as GenericF32x8;

/// Generic SIMD gain map application, dispatched via `#[magetypes]`.
///
/// Processes 8 pixels per iteration using `GenericF32x8<Token>`, which maps
/// to native AVX2, NEON, WASM SIMD128, or scalar depending on the token.
/// Remainder pixels are handled with a scalar tail loop.
///
/// Not compiled on aarch64: nothing dispatches to it there (see
/// [`apply_gain_row_simd`]'s bandwidth-bound analysis), and an uncalled
/// kernel is a `dead_code` error under CI's `-D warnings`.
#[cfg(all(feature = "simd", not(target_arch = "aarch64")))]
#[magetypes(v3, neon, wasm128, scalar)]
fn apply_gain_inner(
    token: Token,
    sdr: &[[f32; 3]],
    gainmap: &[u8],
    lut: &[f32; 256],
    output: &mut [[f32; 3]],
) {
    #[allow(non_camel_case_types)]
    type f32x8 = GenericF32x8<Token>;
    const LANES: usize = 8;

    assert_eq!(sdr.len(), output.len());
    assert_eq!(sdr.len(), gainmap.len());

    let chunks = sdr.len() / LANES;

    for chunk_idx in 0..chunks {
        let base = chunk_idx * LANES;

        // Gather gains from LUT (8 scalar lookups -> SIMD vector)
        let gains: [f32; LANES] = core::array::from_fn(|i| lut[gainmap[base + i] as usize]);
        let g = f32x8::from_array(token, gains);

        // Load R channel (strided gather - every 3rd element starting at [0])
        let r: [f32; LANES] = core::array::from_fn(|i| sdr[base + i][0]);
        let r_v = f32x8::from_array(token, r);

        // Load G channel
        let g_ch: [f32; LANES] = core::array::from_fn(|i| sdr[base + i][1]);
        let g_v = f32x8::from_array(token, g_ch);

        // Load B channel
        let b: [f32; LANES] = core::array::from_fn(|i| sdr[base + i][2]);
        let b_v = f32x8::from_array(token, b);

        // Apply gain: output = sdr * gain
        let r_out = r_v * g;
        let g_out = g_v * g;
        let b_out = b_v * g;

        // Store back (strided scatter)
        let r_arr = r_out.to_array();
        let g_arr = g_out.to_array();
        let b_arr = b_out.to_array();
        for i in 0..LANES {
            output[base + i] = [r_arr[i], g_arr[i], b_arr[i]];
        }
    }

    // Handle remainder pixels with scalar
    let remainder_start = chunks * LANES;
    for i in remainder_start..sdr.len() {
        let g_val = lut[gainmap[i] as usize];
        output[i][0] = sdr[i][0] * g_val;
        output[i][1] = sdr[i][1] * g_val;
        output[i][2] = sdr[i][2] * g_val;
    }
}

/// SIMD-accelerated gain map application with runtime dispatch.
///
/// Applies a single-channel gain LUT to each pixel using the best available
/// SIMD instruction set. Falls back to scalar when no SIMD is available.
///
/// On x86_64 with AVX2+FMA, processes 8 pixels per iteration (~3-4x faster
/// than scalar for large rows). On aarch64 and WASM, processes 8 pixels per
/// iteration using the generic f32x8 type.
///
/// # Performance on aarch64: prefer [`apply_gain_row_scalar`]
///
/// MEASURED 2026-07-28 on Apple M4 Pro (release, no `-C target-cpu=native`,
/// `benches/simd_xplat.rs`): this function is **slower** than
/// [`apply_gain_row_scalar`] on ARM, at both sizes benchmarked —
///
/// | size      | `apply_gain_row_scalar` | this function |
/// |-----------|-------------------------|---------------|
/// | 512x512   | 94.06 us                | 168.40 us     |
/// | 1920x1080 | 750.93 us               | 1.3319 ms     |
///
/// i.e. ~1.8x slower. The cause is the LUT: the scalar path is one table load
/// per pixel (`lut[gainmap[i]]`), and AArch64 NEON has no gather instruction,
/// so the vector path must do eight scalar loads and lane-inserts to build each
/// vector of gains — strictly more work than just doing the arithmetic scalar.
/// A LUT-indexed kernel is the classic case where SIMD loses.
///
/// This is NOT on the `apply_gainmap` path — that calls
/// [`apply_gain_row_presampled`], which is a plain scalar loop — so the crate's
/// own gain-map application is unaffected. It matters only for callers who
/// select this function directly expecting it to be the fast one.
///
/// Not yet measured on Ampere/Graviton or in-order Cortex-A5x; the gather
/// limitation is architectural to NEON, so the conclusion is likely to hold,
/// but that is reasoning, not a measurement.
///
/// # Panics
///
/// Panics if `sdr`, `gainmap`, and `output` have different lengths.
#[cfg(feature = "simd")]
pub fn apply_gain_row_simd(
    sdr: &[[f32; 3]],
    gainmap: &[u8],
    lut: &[f32; 256],
    output: &mut [[f32; 3]],
) {
    // aarch64 runs the SCALAR kernel. That is not a fallback — it is the
    // fastest implementation on this architecture, and it is 1.79x faster than
    // what this function used to do. Measured on Apple M4 Pro, 512x512
    // (release, no -C target-cpu=native), via benches/simd_xplat.rs:
    //
    //   generic `apply_gain_inner` (what shipped)   168.40 us   1.79x slower
    //   vld3q_f32/vst3q_f32 structure-load kernel    97.67 us   1.04x slower
    //   apply_gain_row_scalar                        94.13 us   <- shipped
    //
    // Two SIMD attempts, both lose, and the reason is not the kernels:
    //
    //  1. `apply_gain_inner` deinterleaves RGB in SCALAR code — per 8 pixels it
    //     issues 32 scalar loads, builds four `[f32; 8]` stack arrays, calls
    //     `from_array` on each, then scatters back through three `to_array`
    //     round-trips and 24 scalar stores, all to feed three vector multiplies.
    //     Strictly more work than the scalar kernel does.
    //  2. Rewriting it with `vld3q_f32`/`vst3q_f32` (the instruction pair that
    //     deinterleaves interleaved RGB natively) removed all of that and got
    //     within 4% of scalar — but no further, because:
    //
    // THE KERNEL IS MEMORY-BANDWIDTH-BOUND. It touches 25 B/px (12 B RGB in,
    // 1 B gainmap, 12 B out) for three multiplies. At 512x512 that is 6.55 MB
    // in ~94 us = 69.6 GB/s scalar / 67.1 GB/s SIMD — both sitting on this
    // machine's single-core bandwidth ceiling (~67-70 GB/s, independently
    // measured on the same host in garb's cross-bpp sweep). Arithmetic
    // throughput is not the limit, so widening it cannot help; the SIMD form
    // only adds the LUT-gather overhead that AArch64 cannot vectorise (no
    // gather instruction, and a 256-entry f32 table is far past vqtbl4q_u8).
    //
    // Do not re-add a SIMD kernel here without first showing this op is NOT
    // bandwidth-bound on the target — on a lower-bandwidth core it is bound
    // harder, not less. x86_64/wasm32 keep the generic path unchanged.
    #[cfg(target_arch = "aarch64")]
    apply_gain_row_scalar(sdr, gainmap, lut, output);

    #[cfg(not(target_arch = "aarch64"))]
    incant!(
        apply_gain_inner(sdr, gainmap, lut, output),
        [v3, neon, wasm128, scalar]
    );
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    extern crate std;
    use std::vec;
    #[cfg(feature = "simd")]
    use std::vec::Vec;

    use super::*;

    /// Build a simple gain LUT for testing.
    ///
    /// Maps byte values linearly from `min_gain` to `max_gain`:
    ///   lut[i] = min_gain + (max_gain - min_gain) * (i / 255.0)
    fn build_test_lut(min_gain: f32, max_gain: f32) -> [f32; 256] {
        let mut lut = [0.0f32; 256];
        for (i, entry) in lut.iter_mut().enumerate() {
            *entry = min_gain + (max_gain - min_gain) * (i as f32 / 255.0);
        }
        lut
    }

    #[test]
    fn test_scalar_basic() {
        let sdr = vec![[0.5f32, 0.25, 0.75], [1.0, 0.0, 0.5]];
        let gainmap = vec![128u8, 255];
        let lut = build_test_lut(1.0, 4.0);
        let mut output = vec![[0.0f32; 3]; 2];

        apply_gain_row_scalar(&sdr, &gainmap, &lut, &mut output);

        // Pixel 0: gain = 1.0 + 3.0 * (128/255) ≈ 2.506
        let g0 = lut[128];
        assert!(
            (output[0][0] - 0.5 * g0).abs() < 1e-6,
            "R0: {}",
            output[0][0]
        );
        assert!(
            (output[0][1] - 0.25 * g0).abs() < 1e-6,
            "G0: {}",
            output[0][1]
        );
        assert!(
            (output[0][2] - 0.75 * g0).abs() < 1e-6,
            "B0: {}",
            output[0][2]
        );

        // Pixel 1: gain = lut[255] = 4.0
        let g1 = lut[255];
        assert!(
            (output[1][0] - 1.0 * g1).abs() < 1e-6,
            "R1: {}",
            output[1][0]
        );
        assert!(
            (output[1][1] - 0.0 * g1).abs() < 1e-6,
            "G1: {}",
            output[1][1]
        );
        assert!(
            (output[1][2] - 0.5 * g1).abs() < 1e-6,
            "B1: {}",
            output[1][2]
        );
    }

    #[cfg(feature = "simd")]
    #[test]
    fn test_simd_matches_scalar() {
        // Test all 256 gain byte values to ensure SIMD and scalar produce
        // identical results.
        let pixel_count = 256;
        let sdr: Vec<[f32; 3]> = (0..pixel_count)
            .map(|i| {
                let v = i as f32 / 255.0;
                [v, v * 0.5, 1.0 - v]
            })
            .collect();

        // Each pixel gets a different gain byte (0..255)
        let gainmap: Vec<u8> = (0..pixel_count).map(|i| i as u8).collect();
        let lut = build_test_lut(0.5, 8.0);

        let mut scalar_output = vec![[0.0f32; 3]; pixel_count];
        let mut simd_output = vec![[0.0f32; 3]; pixel_count];

        apply_gain_row_scalar(&sdr, &gainmap, &lut, &mut scalar_output);
        apply_gain_row_simd(&sdr, &gainmap, &lut, &mut simd_output);

        for i in 0..pixel_count {
            for ch in 0..3 {
                assert!(
                    (scalar_output[i][ch] - simd_output[i][ch]).abs() < 1e-6,
                    "Mismatch at pixel {} channel {}: scalar={}, simd={}",
                    i,
                    ch,
                    scalar_output[i][ch],
                    simd_output[i][ch],
                );
            }
        }
    }

    #[cfg(feature = "simd")]
    #[test]
    fn test_simd_non_aligned_length() {
        // Test row widths that aren't multiples of 8 to exercise the
        // scalar remainder path.
        for width in [1, 3, 7, 9, 13, 15, 17, 31, 33] {
            let sdr: Vec<[f32; 3]> = (0..width)
                .map(|i| {
                    let v = (i as f32 * 7.0) % 1.0;
                    [v, v, v]
                })
                .collect();
            let gainmap: Vec<u8> = (0..width).map(|i| ((i * 13) % 256) as u8).collect();
            let lut = build_test_lut(1.0, 4.0);

            let mut scalar_output = vec![[0.0f32; 3]; width];
            let mut simd_output = vec![[0.0f32; 3]; width];

            apply_gain_row_scalar(&sdr, &gainmap, &lut, &mut scalar_output);
            apply_gain_row_simd(&sdr, &gainmap, &lut, &mut simd_output);

            for i in 0..width {
                for ch in 0..3 {
                    assert!(
                        (scalar_output[i][ch] - simd_output[i][ch]).abs() < 1e-6,
                        "width={}, pixel={}, ch={}: scalar={}, simd={}",
                        width,
                        i,
                        ch,
                        scalar_output[i][ch],
                        simd_output[i][ch],
                    );
                }
            }
        }
    }

    #[cfg(feature = "simd")]
    #[test]
    fn test_simd_empty() {
        let sdr: &[[f32; 3]] = &[];
        let gainmap: &[u8] = &[];
        let lut = build_test_lut(1.0, 4.0);
        let mut output: Vec<[f32; 3]> = vec![];

        // Should not panic on empty input
        apply_gain_row_simd(sdr, gainmap, &lut, &mut output);
        assert!(output.is_empty());
    }

    #[cfg(feature = "simd")]
    #[test]
    fn test_simd_single_pixel() {
        let sdr = vec![[0.8f32, 0.4, 0.2]];
        let gainmap = vec![200u8];
        let lut = build_test_lut(1.0, 4.0);

        let mut scalar_output = vec![[0.0f32; 3]; 1];
        let mut simd_output = vec![[0.0f32; 3]; 1];

        apply_gain_row_scalar(&sdr, &gainmap, &lut, &mut scalar_output);
        apply_gain_row_simd(&sdr, &gainmap, &lut, &mut simd_output);

        for ch in 0..3 {
            assert!(
                (scalar_output[0][ch] - simd_output[0][ch]).abs() < 1e-6,
                "ch={}: scalar={}, simd={}",
                ch,
                scalar_output[0][ch],
                simd_output[0][ch],
            );
        }
    }

    #[cfg(feature = "simd")]
    #[test]
    fn test_simd_gain_endpoints() {
        // Byte 0 should give min gain, byte 255 should give max gain
        let min_gain = 0.5f32;
        let max_gain = 8.0f32;
        let lut = build_test_lut(min_gain, max_gain);

        let sdr = vec![[1.0f32; 3]; 2];
        let gainmap = vec![0u8, 255];
        let mut output = vec![[0.0f32; 3]; 2];

        apply_gain_row_simd(&sdr, &gainmap, &lut, &mut output);

        // Pixel 0: gain = min_gain = 0.5
        for (ch, val) in output[0].iter().enumerate() {
            assert!(
                (val - min_gain).abs() < 1e-6,
                "byte 0 ch={}: expected {}, got {}",
                ch,
                min_gain,
                val,
            );
        }

        // Pixel 1: gain = max_gain = 8.0
        for (ch, val) in output[1].iter().enumerate() {
            assert!(
                (val - max_gain).abs() < 1e-6,
                "byte 255 ch={}: expected {}, got {}",
                ch,
                max_gain,
                val,
            );
        }
    }
}
