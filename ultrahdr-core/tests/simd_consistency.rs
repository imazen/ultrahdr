//! SIMD tier consistency tests for ultrahdr-core gain map application.
//!
//! Runs `apply_gain_row_simd` under every archmage SIMD tier permutation
//! and verifies all produce identical output. The operation is pure
//! multiply+scatter so results should be byte-exact across tiers.

#![forbid(unsafe_code)]
#![cfg(feature = "simd")]

use archmage::testing::{CompileTimePolicy, for_each_token_permutation};
use ultrahdr_core::gainmap::apply_simd::{apply_gain_row_scalar, apply_gain_row_simd};

/// FNV-1a hash of a byte slice.
fn hash_bytes(data: &[u8]) -> u64 {
    let mut h: u64 = 0xcbf29ce484222325;
    for &b in data {
        h ^= b as u64;
        h = h.wrapping_mul(0x100000001b3);
    }
    h
}

/// Hash a slice of `[f32; 3]` as raw bytes.
fn hash_f32x3(data: &[[f32; 3]]) -> u64 {
    hash_bytes(bytemuck::cast_slice(data))
}

/// Build a test LUT mapping byte values linearly from `min` to `max`.
fn build_test_lut(min: f32, max: f32) -> [f32; 256] {
    let mut lut = [0.0f32; 256];
    for (i, entry) in lut.iter_mut().enumerate() {
        *entry = min + (max - min) * (i as f32 / 255.0);
    }
    lut
}

/// Generate SDR pixel data with varied values.
fn generate_sdr(count: usize) -> Vec<[f32; 3]> {
    (0..count)
        .map(|i| {
            let r = ((i * 7 + 3) % 256) as f32 / 255.0;
            let g = ((i * 11 + 50) % 256) as f32 / 255.0;
            let b = ((i * 5 + 100) % 256) as f32 / 255.0;
            [r, g, b]
        })
        .collect()
}

/// Generate gain map bytes with varied values.
fn generate_gainmap(count: usize) -> Vec<u8> {
    (0..count).map(|i| ((i * 13 + 7) % 256) as u8).collect()
}

const PIXELS: usize = 512;

#[test]
fn apply_gain_all_tiers_match() {
    let sdr = generate_sdr(PIXELS);
    let gainmap = generate_gainmap(PIXELS);
    let lut = build_test_lut(0.5, 4.0);
    let mut reference_hash: Option<u64> = None;

    let _ = for_each_token_permutation(CompileTimePolicy::Warn, |perm| {
        let mut output = vec![[0.0f32; 3]; PIXELS];
        apply_gain_row_simd(&sdr, &gainmap, &lut, &mut output);
        let h = hash_f32x3(&output);

        if let Some(ref_h) = reference_hash {
            assert_eq!(
                h, ref_h,
                "apply_gain_row_simd output differs under '{}'",
                perm.label,
            );
        } else {
            reference_hash = Some(h);
        }
    });
}

#[test]
fn apply_gain_simd_matches_scalar() {
    let sdr = generate_sdr(PIXELS);
    let gainmap = generate_gainmap(PIXELS);
    let lut = build_test_lut(0.5, 4.0);

    let mut scalar_output = vec![[0.0f32; 3]; PIXELS];
    apply_gain_row_scalar(&sdr, &gainmap, &lut, &mut scalar_output);

    let _ = for_each_token_permutation(CompileTimePolicy::Warn, |perm| {
        let mut simd_output = vec![[0.0f32; 3]; PIXELS];
        apply_gain_row_simd(&sdr, &gainmap, &lut, &mut simd_output);

        assert_eq!(
            hash_f32x3(&simd_output),
            hash_f32x3(&scalar_output),
            "SIMD output differs from scalar under '{}'",
            perm.label,
        );
    });
}

#[test]
fn apply_gain_roundtrip_stability() {
    let sdr = generate_sdr(PIXELS);
    let gainmap = generate_gainmap(PIXELS);
    let lut = build_test_lut(1.0, 3.0);

    let _ = for_each_token_permutation(CompileTimePolicy::Warn, |perm| {
        let mut out1 = vec![[0.0f32; 3]; PIXELS];
        let mut out2 = vec![[0.0f32; 3]; PIXELS];
        apply_gain_row_simd(&sdr, &gainmap, &lut, &mut out1);
        apply_gain_row_simd(&sdr, &gainmap, &lut, &mut out2);

        assert_eq!(
            hash_f32x3(&out1),
            hash_f32x3(&out2),
            "apply_gain not deterministic under '{}'",
            perm.label,
        );
    });
}
