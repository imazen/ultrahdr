//! Edge case tests (Phase 6 from TESTING_PLAN.md).
//!
//! Tests unusual dimensions, extreme values, and boundary conditions.

mod common;

use common::{create_hdr_solid, create_sdr_solid};
use ultrahdr_rs::{ColorGamut, ColorTransfer, Encoder, PixelFormat, RawImage};

// ============================================================================
// Dimension Edge Cases
// ============================================================================

/// Test 1x1 pixel image (minimum size).
#[test]
fn test_dimension_1x1() {
    let hdr = create_hdr_solid(1, 1, 0.5, 0.5, 0.5);
    let sdr = create_sdr_solid(1, 1, 186, 186, 186);

    let mut encoder = Encoder::new();
    encoder.set_hdr_image(hdr).set_sdr_image(sdr);

    let result = encoder.encode();
    assert!(
        result.is_ok(),
        "Should encode 1x1 image: {:?}",
        result.err()
    );
}

/// Test odd width dimension.
#[test]
fn test_dimension_odd_width() {
    let hdr = create_hdr_solid(63, 64, 0.5, 0.5, 0.5);
    let sdr = create_sdr_solid(63, 64, 186, 186, 186);

    let mut encoder = Encoder::new();
    encoder.set_hdr_image(hdr).set_sdr_image(sdr);

    let result = encoder.encode();
    assert!(
        result.is_ok(),
        "Should encode 63x64 image: {:?}",
        result.err()
    );
}

/// Test odd height dimension.
#[test]
fn test_dimension_odd_height() {
    let hdr = create_hdr_solid(64, 63, 0.5, 0.5, 0.5);
    let sdr = create_sdr_solid(64, 63, 186, 186, 186);

    let mut encoder = Encoder::new();
    encoder.set_hdr_image(hdr).set_sdr_image(sdr);

    let result = encoder.encode();
    assert!(
        result.is_ok(),
        "Should encode 64x63 image: {:?}",
        result.err()
    );
}

/// Test both dimensions odd.
#[test]
fn test_dimension_both_odd() {
    let hdr = create_hdr_solid(63, 63, 0.5, 0.5, 0.5);
    let sdr = create_sdr_solid(63, 63, 186, 186, 186);

    let mut encoder = Encoder::new();
    encoder.set_hdr_image(hdr).set_sdr_image(sdr);

    let result = encoder.encode();
    assert!(
        result.is_ok(),
        "Should encode 63x63 image: {:?}",
        result.err()
    );
}

/// Test prime number dimensions.
#[test]
fn test_dimension_prime() {
    let hdr = create_hdr_solid(67, 71, 0.5, 0.5, 0.5);
    let sdr = create_sdr_solid(67, 71, 186, 186, 186);

    let mut encoder = Encoder::new();
    encoder.set_hdr_image(hdr).set_sdr_image(sdr);

    let result = encoder.encode();
    assert!(
        result.is_ok(),
        "Should encode 67x71 (prime) image: {:?}",
        result.err()
    );
}

/// Test non-MCU aligned dimensions (not multiple of 8).
#[test]
fn test_dimension_non_mcu_aligned() {
    // MCU (Minimum Coded Unit) is typically 8x8 or 16x16
    let dimensions = [(9, 9), (15, 15), (17, 17), (33, 33)];

    for (w, h) in dimensions {
        let hdr = create_hdr_solid(w, h, 0.5, 0.5, 0.5);
        let sdr = create_sdr_solid(w, h, 186, 186, 186);

        let mut encoder = Encoder::new();
        encoder.set_hdr_image(hdr).set_sdr_image(sdr);

        let result = encoder.encode();
        assert!(
            result.is_ok(),
            "Should encode {}x{} (non-MCU aligned): {:?}",
            w,
            h,
            result.err()
        );
    }
}

/// Test very small dimensions.
#[test]
fn test_dimension_very_small() {
    let dimensions = [(2, 2), (3, 3), (4, 4), (5, 5)];

    for (w, h) in dimensions {
        let hdr = create_hdr_solid(w, h, 0.5, 0.5, 0.5);
        let sdr = create_sdr_solid(w, h, 186, 186, 186);

        let mut encoder = Encoder::new();
        encoder.set_hdr_image(hdr).set_sdr_image(sdr);

        let result = encoder.encode();
        assert!(
            result.is_ok(),
            "Should encode {}x{}: {:?}",
            w,
            h,
            result.err()
        );
    }
}

/// Test asymmetric dimensions.
#[test]
fn test_dimension_asymmetric() {
    let dimensions = [(10, 100), (100, 10), (1, 100), (100, 1)];

    for (w, h) in dimensions {
        let hdr = create_hdr_solid(w, h, 0.5, 0.5, 0.5);
        let sdr = create_sdr_solid(w, h, 186, 186, 186);

        let mut encoder = Encoder::new();
        encoder.set_hdr_image(hdr).set_sdr_image(sdr);

        let result = encoder.encode();
        assert!(
            result.is_ok(),
            "Should encode {}x{} (asymmetric): {:?}",
            w,
            h,
            result.err()
        );
    }
}

// ============================================================================
// Value Edge Cases
// ============================================================================

/// Test very bright HDR values.
#[test]
fn test_value_very_bright() {
    let hdr = create_hdr_solid(32, 32, 100.0, 100.0, 100.0); // Very bright
    let sdr = create_sdr_solid(32, 32, 255, 255, 255);

    let mut encoder = Encoder::new();
    encoder.set_hdr_image(hdr).set_sdr_image(sdr);

    let result = encoder.encode();
    assert!(
        result.is_ok(),
        "Should encode very bright HDR: {:?}",
        result.err()
    );
}

/// Test pure black HDR.
#[test]
fn test_value_pure_black() {
    let hdr = create_hdr_solid(32, 32, 0.0, 0.0, 0.0);
    let sdr = create_sdr_solid(32, 32, 0, 0, 0);

    let mut encoder = Encoder::new();
    encoder.set_hdr_image(hdr).set_sdr_image(sdr);

    let result = encoder.encode();
    assert!(
        result.is_ok(),
        "Should encode pure black: {:?}",
        result.err()
    );
}

/// Test pure white HDR.
#[test]
fn test_value_pure_white() {
    let hdr = create_hdr_solid(32, 32, 1.0, 1.0, 1.0);
    let sdr = create_sdr_solid(32, 32, 255, 255, 255);

    let mut encoder = Encoder::new();
    encoder.set_hdr_image(hdr).set_sdr_image(sdr);

    let result = encoder.encode();
    assert!(
        result.is_ok(),
        "Should encode pure white: {:?}",
        result.err()
    );
}

/// Test very small but non-zero values.
#[test]
fn test_value_very_small() {
    let hdr = create_hdr_solid(32, 32, 0.001, 0.001, 0.001);
    let sdr = create_sdr_solid(32, 32, 1, 1, 1);

    let mut encoder = Encoder::new();
    encoder.set_hdr_image(hdr).set_sdr_image(sdr);

    let result = encoder.encode();
    assert!(
        result.is_ok(),
        "Should encode very small values: {:?}",
        result.err()
    );
}

/// Test negative HDR values (out of gamut).
#[test]
fn test_value_negative() {
    // Some HDR content can have negative values (out of gamut)
    let mut data = Vec::with_capacity(32 * 32 * 16);
    for _ in 0..(32 * 32) {
        data.extend_from_slice(&(-0.1f32).to_le_bytes()); // Negative R
        data.extend_from_slice(&0.5f32.to_le_bytes()); // G
        data.extend_from_slice(&0.5f32.to_le_bytes()); // B
        data.extend_from_slice(&1.0f32.to_le_bytes()); // A
    }

    let hdr = RawImage::from_data(
        32,
        32,
        PixelFormat::Rgba32F,
        ColorGamut::Bt709,
        ColorTransfer::Linear,
        data,
    )
    .unwrap();

    let sdr = create_sdr_solid(32, 32, 0, 128, 128);

    let mut encoder = Encoder::new();
    encoder.set_hdr_image(hdr).set_sdr_image(sdr);

    let result = encoder.encode();
    // Should either succeed or gracefully fail, not panic
    match result {
        Ok(_) => {} // Success is fine
        Err(e) => eprintln!("Negative values produced error (acceptable): {:?}", e),
    }
}

/// Test NaN HDR values (should be handled gracefully).
#[test]
fn test_value_nan() {
    let mut data = Vec::with_capacity(32 * 32 * 16);
    for i in 0..(32 * 32) {
        let val = if i == 0 { f32::NAN } else { 0.5 };
        data.extend_from_slice(&val.to_le_bytes());
        data.extend_from_slice(&0.5f32.to_le_bytes());
        data.extend_from_slice(&0.5f32.to_le_bytes());
        data.extend_from_slice(&1.0f32.to_le_bytes());
    }

    let hdr = RawImage::from_data(
        32,
        32,
        PixelFormat::Rgba32F,
        ColorGamut::Bt709,
        ColorTransfer::Linear,
        data,
    )
    .unwrap();

    let sdr = create_sdr_solid(32, 32, 128, 128, 128);

    let mut encoder = Encoder::new();
    encoder.set_hdr_image(hdr).set_sdr_image(sdr);

    let result = encoder.encode();
    // Should handle gracefully (either succeed or error, not panic)
    match result {
        Ok(_) => {} // May succeed with NaN clamped/replaced
        Err(e) => eprintln!("NaN values produced error (acceptable): {:?}", e),
    }
}

/// Test infinity HDR values.
#[test]
fn test_value_infinity() {
    let mut data = Vec::with_capacity(32 * 32 * 16);
    for i in 0..(32 * 32) {
        let val = if i == 0 { f32::INFINITY } else { 0.5 };
        data.extend_from_slice(&val.to_le_bytes());
        data.extend_from_slice(&0.5f32.to_le_bytes());
        data.extend_from_slice(&0.5f32.to_le_bytes());
        data.extend_from_slice(&1.0f32.to_le_bytes());
    }

    let hdr = RawImage::from_data(
        32,
        32,
        PixelFormat::Rgba32F,
        ColorGamut::Bt709,
        ColorTransfer::Linear,
        data,
    )
    .unwrap();

    let sdr = create_sdr_solid(32, 32, 128, 128, 128);

    let mut encoder = Encoder::new();
    encoder.set_hdr_image(hdr).set_sdr_image(sdr);

    let result = encoder.encode();
    // Should handle gracefully
    match result {
        Ok(_) => {}
        Err(e) => eprintln!("Infinity values produced error (acceptable): {:?}", e),
    }
}

// ============================================================================
// Quality Parameter Edge Cases
// ============================================================================

/// Test minimum quality (1).
#[test]
fn test_quality_minimum() {
    let hdr = create_hdr_solid(64, 64, 0.5, 0.5, 0.5);
    let sdr = create_sdr_solid(64, 64, 186, 186, 186);

    let mut encoder = Encoder::new();
    encoder
        .set_hdr_image(hdr)
        .set_sdr_image(sdr)
        .set_quality(1, 1);

    let result = encoder.encode();
    assert!(
        result.is_ok(),
        "Should encode at quality 1: {:?}",
        result.err()
    );
}

/// Test maximum quality (100).
#[test]
fn test_quality_maximum() {
    let hdr = create_hdr_solid(64, 64, 0.5, 0.5, 0.5);
    let sdr = create_sdr_solid(64, 64, 186, 186, 186);

    let mut encoder = Encoder::new();
    encoder
        .set_hdr_image(hdr)
        .set_sdr_image(sdr)
        .set_quality(100, 100);

    let result = encoder.encode();
    assert!(
        result.is_ok(),
        "Should encode at quality 100: {:?}",
        result.err()
    );
}

/// Test mismatched base/gainmap quality.
#[test]
fn test_quality_mismatched() {
    let hdr = create_hdr_solid(64, 64, 0.5, 0.5, 0.5);
    let sdr = create_sdr_solid(64, 64, 186, 186, 186);

    // High base, low gainmap
    let mut encoder = Encoder::new();
    encoder
        .set_hdr_image(hdr.clone())
        .set_sdr_image(sdr.clone())
        .set_quality(95, 20);
    assert!(encoder.encode().is_ok());

    // Low base, high gainmap
    let mut encoder = Encoder::new();
    encoder
        .set_hdr_image(hdr)
        .set_sdr_image(sdr)
        .set_quality(20, 95);
    assert!(encoder.encode().is_ok());
}

// ============================================================================
// Gain Map Scale Edge Cases
// ============================================================================

/// Test gain map scale factor 1 (full resolution).
#[test]
fn test_gainmap_scale_1() {
    let hdr = create_hdr_solid(64, 64, 0.5, 0.5, 0.5);
    let sdr = create_sdr_solid(64, 64, 186, 186, 186);

    let mut encoder = Encoder::new();
    encoder
        .set_hdr_image(hdr)
        .set_sdr_image(sdr)
        .set_gainmap_scale(1);

    let result = encoder.encode();
    assert!(
        result.is_ok(),
        "Should encode with scale 1: {:?}",
        result.err()
    );
}

/// Test large gain map scale factor.
#[test]
fn test_gainmap_scale_large() {
    let hdr = create_hdr_solid(128, 128, 0.5, 0.5, 0.5);
    let sdr = create_sdr_solid(128, 128, 186, 186, 186);

    let mut encoder = Encoder::new();
    encoder
        .set_hdr_image(hdr)
        .set_sdr_image(sdr)
        .set_gainmap_scale(64); // 128/64 = 2 pixel gain map

    let result = encoder.encode();
    assert!(
        result.is_ok(),
        "Should encode with large scale: {:?}",
        result.err()
    );
}

// ============================================================================
// Color Space Edge Cases
// ============================================================================

/// Test P3 gamut input.
#[test]
fn test_color_p3_gamut() {
    let mut data = Vec::with_capacity(64 * 64 * 16);
    for _ in 0..(64 * 64) {
        data.extend_from_slice(&0.5f32.to_le_bytes());
        data.extend_from_slice(&0.5f32.to_le_bytes());
        data.extend_from_slice(&0.5f32.to_le_bytes());
        data.extend_from_slice(&1.0f32.to_le_bytes());
    }

    let hdr = RawImage::from_data(
        64,
        64,
        PixelFormat::Rgba32F,
        ColorGamut::DisplayP3,
        ColorTransfer::Linear,
        data,
    )
    .unwrap();

    let sdr = create_sdr_solid(64, 64, 186, 186, 186);

    let mut encoder = Encoder::new();
    encoder.set_hdr_image(hdr).set_sdr_image(sdr);

    let result = encoder.encode();
    assert!(result.is_ok(), "Should encode P3 input: {:?}", result.err());
}

/// Test BT.2100 gamut input.
#[test]
fn test_color_bt2100_gamut() {
    let mut data = Vec::with_capacity(64 * 64 * 16);
    for _ in 0..(64 * 64) {
        data.extend_from_slice(&0.5f32.to_le_bytes());
        data.extend_from_slice(&0.5f32.to_le_bytes());
        data.extend_from_slice(&0.5f32.to_le_bytes());
        data.extend_from_slice(&1.0f32.to_le_bytes());
    }

    let hdr = RawImage::from_data(
        64,
        64,
        PixelFormat::Rgba32F,
        ColorGamut::Bt2100,
        ColorTransfer::Linear,
        data,
    )
    .unwrap();

    let sdr = create_sdr_solid(64, 64, 186, 186, 186);

    let mut encoder = Encoder::new();
    encoder.set_hdr_image(hdr).set_sdr_image(sdr);

    let result = encoder.encode();
    assert!(
        result.is_ok(),
        "Should encode BT.2100 input: {:?}",
        result.err()
    );
}
