//! SIMD-accelerated gain map application.
//!
//! Provides [`apply_gain_row_scalar`] (always available) and
//! `apply_gain_row_simd` (requires `simd` feature) which dispatches
//! to the best available SIMD implementation at runtime:
//!
//! - **AVX2+FMA** on x86_64: 8 pixels per iteration
//! - **NEON** on aarch64: 4 pixels per iteration
//! - **Scalar** everywhere else
//!
//! All functions operate on pre-linearized `[f32; 3]` RGB pixels with a
//! precomputed LUT mapping gain map bytes to linear gain multipliers.

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

/// SIMD-accelerated gain map application with runtime dispatch.
///
/// Applies a single-channel gain LUT to each pixel using the best available
/// SIMD instruction set. Falls back to scalar when no SIMD is available.
///
/// On x86_64 with AVX2+FMA, processes 8 pixels per iteration (~3-4x faster
/// than scalar for large rows).
///
/// # Panics
///
/// Panics if `sdr`, `gainmap`, and `output` have different lengths.
#[cfg(all(feature = "simd", target_arch = "x86_64"))]
pub fn apply_gain_row_simd(
    sdr: &[[f32; 3]],
    gainmap: &[u8],
    lut: &[f32; 256],
    output: &mut [[f32; 3]],
) {
    use archmage::SimdToken;

    if let Some(token) = archmage::X64V3Token::try_new() {
        apply_gain_avx2(token, sdr, gainmap, lut, output);
    } else {
        apply_gain_row_scalar(sdr, gainmap, lut, output);
    }
}

/// SIMD-accelerated gain map application with runtime dispatch.
///
/// On aarch64 with NEON, processes 4 pixels per iteration (~2-3x faster
/// than scalar for large rows).
///
/// # Panics
///
/// Panics if `sdr`, `gainmap`, and `output` have different lengths.
#[cfg(all(feature = "simd", target_arch = "aarch64"))]
pub fn apply_gain_row_simd(
    sdr: &[[f32; 3]],
    gainmap: &[u8],
    lut: &[f32; 256],
    output: &mut [[f32; 3]],
) {
    use archmage::SimdToken;

    if let Some(token) = archmage::NeonToken::try_new() {
        apply_gain_neon(token, sdr, gainmap, lut, output);
    } else {
        apply_gain_row_scalar(sdr, gainmap, lut, output);
    }
}

/// SIMD-accelerated gain map application with runtime dispatch.
///
/// On architectures without SIMD support, falls back to scalar.
///
/// # Panics
///
/// Panics if `sdr`, `gainmap`, and `output` have different lengths.
#[cfg(all(
    feature = "simd",
    not(any(target_arch = "x86_64", target_arch = "aarch64"))
))]
pub fn apply_gain_row_simd(
    sdr: &[[f32; 3]],
    gainmap: &[u8],
    lut: &[f32; 256],
    output: &mut [[f32; 3]],
) {
    apply_gain_row_scalar(sdr, gainmap, lut, output);
}

// ============================================================================
// AVX2+FMA implementation (x86_64, requires `simd` feature)
// ============================================================================

#[cfg(all(feature = "simd", target_arch = "x86_64"))]
#[archmage::arcane]
fn apply_gain_avx2(
    token: archmage::X64V3Token,
    sdr: &[[f32; 3]],
    gainmap: &[u8],
    lut: &[f32; 256],
    output: &mut [[f32; 3]],
) {
    use magetypes::simd::v3::f32x8;

    assert_eq!(sdr.len(), output.len());
    assert_eq!(sdr.len(), gainmap.len());

    // Process 8 pixels at a time
    let chunks = sdr.len() / 8;

    for chunk_idx in 0..chunks {
        let base = chunk_idx * 8;

        // Gather gains from LUT (8 scalar lookups -> SIMD vector)
        let gains: [f32; 8] = core::array::from_fn(|i| lut[gainmap[base + i] as usize]);
        let g = f32x8::from_array(token, gains);

        // Load R channel (strided gather - every 3rd element starting at [0])
        let r: [f32; 8] = core::array::from_fn(|i| sdr[base + i][0]);
        let r_v = f32x8::from_array(token, r);

        // Load G channel
        let g_ch: [f32; 8] = core::array::from_fn(|i| sdr[base + i][1]);
        let g_v = f32x8::from_array(token, g_ch);

        // Load B channel
        let b: [f32; 8] = core::array::from_fn(|i| sdr[base + i][2]);
        let b_v = f32x8::from_array(token, b);

        // Apply gain: output = sdr * gain
        let r_out = r_v * g;
        let g_out = g_v * g;
        let b_out = b_v * g;

        // Store back (strided scatter)
        let r_arr = r_out.to_array();
        let g_arr = g_out.to_array();
        let b_arr = b_out.to_array();
        for i in 0..8 {
            output[base + i] = [r_arr[i], g_arr[i], b_arr[i]];
        }
    }

    // Handle remainder pixels with scalar
    let remainder_start = chunks * 8;
    for i in remainder_start..sdr.len() {
        let g = lut[gainmap[i] as usize];
        output[i][0] = sdr[i][0] * g;
        output[i][1] = sdr[i][1] * g;
        output[i][2] = sdr[i][2] * g;
    }
}

// ============================================================================
// NEON implementation (aarch64, requires `simd` feature)
// ============================================================================

#[cfg(all(feature = "simd", target_arch = "aarch64"))]
#[archmage::arcane]
fn apply_gain_neon(
    token: archmage::NeonToken,
    sdr: &[[f32; 3]],
    gainmap: &[u8],
    lut: &[f32; 256],
    output: &mut [[f32; 3]],
) {
    use magetypes::simd::f32x4;

    assert_eq!(sdr.len(), output.len());
    assert_eq!(sdr.len(), gainmap.len());

    // Process 4 pixels at a time
    let chunks = sdr.len() / 4;

    for chunk_idx in 0..chunks {
        let base = chunk_idx * 4;

        // Gather gains from LUT (4 scalar lookups -> SIMD vector)
        let gains: [f32; 4] = core::array::from_fn(|i| lut[gainmap[base + i] as usize]);
        let g = f32x4::from_array(token, gains);

        // Load R channel (strided gather - every 3rd element starting at [0])
        let r: [f32; 4] = core::array::from_fn(|i| sdr[base + i][0]);
        let r_v = f32x4::from_array(token, r);

        // Load G channel
        let g_ch: [f32; 4] = core::array::from_fn(|i| sdr[base + i][1]);
        let g_v = f32x4::from_array(token, g_ch);

        // Load B channel
        let b: [f32; 4] = core::array::from_fn(|i| sdr[base + i][2]);
        let b_v = f32x4::from_array(token, b);

        // Apply gain: output = sdr * gain
        let r_out = r_v * g;
        let g_out = g_v * g;
        let b_out = b_v * g;

        // Store back (strided scatter)
        let r_arr = r_out.to_array();
        let g_arr = g_out.to_array();
        let b_arr = b_out.to_array();
        for i in 0..4 {
            output[base + i] = [r_arr[i], g_arr[i], b_arr[i]];
        }
    }

    // Handle remainder pixels with scalar
    let remainder_start = chunks * 4;
    for i in remainder_start..sdr.len() {
        let g = lut[gainmap[i] as usize];
        output[i][0] = sdr[i][0] * g;
        output[i][1] = sdr[i][1] * g;
        output[i][2] = sdr[i][2] * g;
    }
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
