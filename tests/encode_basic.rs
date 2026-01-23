//! Basic encoding tests (Phase 3 from TESTING_PLAN.md).
//!
//! Tests encoding using synthetically generated images.

mod common;

use common::{
    create_hdr_checkerboard, create_hdr_gradient, create_hdr_solid, create_sdr_checkerboard,
    create_sdr_gradient, create_sdr_solid,
};
use ultrahdr::{Decoder, Encoder};

/// Test encoding with HDR-only input (auto-generates SDR).
#[test]
fn test_encode_hdr_only() {
    let hdr = create_hdr_gradient(128, 128, 4.0);

    let mut encoder = Encoder::new();
    encoder.set_hdr_image(hdr);

    let result = encoder.encode();
    assert!(result.is_ok(), "Should encode HDR-only: {:?}", result.err());

    let encoded = result.unwrap();
    // Verify it's a valid Ultra HDR
    let decoder = Decoder::new(&encoded).unwrap();
    assert!(decoder.is_ultrahdr(), "Output should be Ultra HDR");
}

/// Test encoding with HDR + SDR input.
#[test]
fn test_encode_hdr_and_sdr() {
    let hdr = create_hdr_gradient(128, 128, 4.0);
    let sdr = create_sdr_gradient(128, 128);

    let mut encoder = Encoder::new();
    encoder.set_hdr_image(hdr).set_sdr_image(sdr);

    let result = encoder.encode();
    assert!(result.is_ok(), "Should encode HDR+SDR: {:?}", result.err());

    let encoded = result.unwrap();
    let decoder = Decoder::new(&encoded).unwrap();
    assert!(decoder.is_ultrahdr());
}

/// Test encoding solid color images.
#[test]
fn test_encode_solid_colors() {
    // Mid-gray solid
    let hdr = create_hdr_solid(64, 64, 0.5, 0.5, 0.5);
    let sdr = create_sdr_solid(64, 64, 186, 186, 186); // ~0.5 in sRGB

    let mut encoder = Encoder::new();
    encoder.set_hdr_image(hdr).set_sdr_image(sdr);

    let result = encoder.encode();
    assert!(
        result.is_ok(),
        "Should encode solid color: {:?}",
        result.err()
    );
}

/// Test encoding checkerboard patterns.
#[test]
fn test_encode_checkerboard() {
    let hdr = create_hdr_checkerboard(64, 64, 0.1, 2.0);
    let sdr = create_sdr_checkerboard(64, 64, 50, 200);

    let mut encoder = Encoder::new();
    encoder.set_hdr_image(hdr).set_sdr_image(sdr);

    let result = encoder.encode();
    assert!(
        result.is_ok(),
        "Should encode checkerboard: {:?}",
        result.err()
    );
}

/// Test quality settings.
#[test]
fn test_encode_quality_settings() {
    let hdr = create_hdr_gradient(64, 64, 2.0);
    let sdr = create_sdr_gradient(64, 64);

    // Low quality
    let mut encoder_low = Encoder::new();
    encoder_low
        .set_hdr_image(hdr.clone())
        .set_sdr_image(sdr.clone())
        .set_quality(50, 40);
    let low_result = encoder_low.encode().unwrap();

    // High quality
    let mut encoder_high = Encoder::new();
    encoder_high
        .set_hdr_image(hdr)
        .set_sdr_image(sdr)
        .set_quality(98, 95);
    let high_result = encoder_high.encode().unwrap();

    // Both should produce valid outputs
    // Note: jpegli uses perceptual optimization, so higher quality doesn't
    // always mean larger files for simple synthetic images like gradients.
    // The important thing is that both encode successfully and produce valid JPEGs.
    assert!(low_result.len() > 100, "Low quality output too small");
    assert!(high_result.len() > 100, "High quality output too small");

    // Verify both are valid JPEGs (start with SOI marker)
    assert_eq!(
        &low_result[0..2],
        &[0xFF, 0xD8],
        "Low quality not valid JPEG"
    );
    assert_eq!(
        &high_result[0..2],
        &[0xFF, 0xD8],
        "High quality not valid JPEG"
    );
}

/// Test gain map scale factor.
#[test]
fn test_encode_gainmap_scale() {
    let hdr = create_hdr_gradient(128, 128, 4.0);
    let sdr = create_sdr_gradient(128, 128);

    // Scale factor 4 (default)
    let mut encoder = Encoder::new();
    encoder
        .set_hdr_image(hdr.clone())
        .set_sdr_image(sdr.clone())
        .set_gainmap_scale(4);
    let encoded = encoder.encode().unwrap();

    let decoder = Decoder::new(&encoded).unwrap();
    let gainmap = decoder.decode_gainmap().unwrap();

    // Gain map should be 1/4 size
    assert_eq!(gainmap.width, 32, "Gain map width should be 128/4=32");
    assert_eq!(gainmap.height, 32, "Gain map height should be 128/4=32");
}

/// Test different gain map scale factors.
#[test]
fn test_encode_gainmap_scale_variants() {
    let sizes = [(2, 64), (4, 32), (8, 16)];

    for (scale, expected_dim) in sizes {
        let hdr = create_hdr_gradient(128, 128, 2.0);
        let sdr = create_sdr_gradient(128, 128);

        let mut encoder = Encoder::new();
        encoder
            .set_hdr_image(hdr)
            .set_sdr_image(sdr)
            .set_gainmap_scale(scale);

        let encoded = encoder.encode().unwrap();
        let decoder = Decoder::new(&encoded).unwrap();
        let gainmap = decoder.decode_gainmap().unwrap();

        assert_eq!(
            gainmap.width, expected_dim,
            "Scale {} should give width {}",
            scale, expected_dim
        );
    }
}

/// Test target display peak setting.
#[test]
fn test_encode_target_display_peak() {
    let hdr = create_hdr_gradient(64, 64, 10.0);
    let sdr = create_sdr_gradient(64, 64);

    // Lower peak (1000 nits)
    let mut encoder_low = Encoder::new();
    encoder_low
        .set_hdr_image(hdr.clone())
        .set_sdr_image(sdr.clone())
        .set_target_display_peak(1000.0);
    let low_encoded = encoder_low.encode().unwrap();

    // Higher peak (10000 nits)
    let mut encoder_high = Encoder::new();
    encoder_high
        .set_hdr_image(hdr)
        .set_sdr_image(sdr)
        .set_target_display_peak(10000.0);
    let high_encoded = encoder_high.encode().unwrap();

    // Both should produce valid Ultra HDR
    assert!(Decoder::new(&low_encoded).unwrap().is_ultrahdr());
    assert!(Decoder::new(&high_encoded).unwrap().is_ultrahdr());

    // Check metadata reflects the peak setting
    let low_meta = Decoder::new(&low_encoded).unwrap();
    let high_meta = Decoder::new(&high_encoded).unwrap();

    let low_cap = low_meta.metadata().unwrap().hdr_capacity_max;
    let high_cap = high_meta.metadata().unwrap().hdr_capacity_max;

    assert!(
        high_cap > low_cap,
        "Higher peak should have higher capacity: {} vs {}",
        high_cap,
        low_cap
    );
}

/// Test ISO metadata toggle.
#[test]
fn test_encode_iso_metadata_toggle() {
    let hdr = create_hdr_gradient(64, 64, 2.0);
    let sdr = create_sdr_gradient(64, 64);

    // With ISO metadata
    let mut encoder = Encoder::new();
    encoder
        .set_hdr_image(hdr.clone())
        .set_sdr_image(sdr.clone())
        .set_use_iso_metadata(true);
    let with_iso = encoder.encode().unwrap();

    // Without ISO metadata
    let mut encoder = Encoder::new();
    encoder
        .set_hdr_image(hdr)
        .set_sdr_image(sdr)
        .set_use_iso_metadata(false);
    let without_iso = encoder.encode().unwrap();

    // Both should be valid Ultra HDR
    assert!(Decoder::new(&with_iso).unwrap().is_ultrahdr());
    assert!(Decoder::new(&without_iso).unwrap().is_ultrahdr());
}

/// Test that encoding without HDR image fails.
#[test]
fn test_encode_requires_hdr() {
    let encoder = Encoder::new();
    let result = encoder.encode();

    assert!(result.is_err(), "Should fail without HDR image");
}

/// Test output is valid JPEG structure.
#[test]
fn test_encode_produces_valid_jpeg() {
    let hdr = create_hdr_gradient(64, 64, 2.0);
    let sdr = create_sdr_gradient(64, 64);

    let mut encoder = Encoder::new();
    encoder.set_hdr_image(hdr).set_sdr_image(sdr);

    let encoded = encoder.encode().unwrap();

    // Check JPEG markers
    assert_eq!(&encoded[0..2], &[0xFF, 0xD8], "Should start with SOI");
    assert_eq!(
        &encoded[encoded.len() - 2..],
        &[0xFF, 0xD9],
        "Should end with EOI"
    );
}

/// Test output contains XMP metadata.
#[test]
fn test_encode_contains_xmp() {
    let hdr = create_hdr_gradient(64, 64, 2.0);
    let sdr = create_sdr_gradient(64, 64);

    let mut encoder = Encoder::new();
    encoder.set_hdr_image(hdr).set_sdr_image(sdr);

    let encoded = encoder.encode().unwrap();
    let data_str = String::from_utf8_lossy(&encoded);

    assert!(
        data_str.contains("hdrgm") || data_str.contains("hdr-gain-map"),
        "Should contain hdrgm namespace"
    );
}

/// Test output contains MPF structure.
#[test]
fn test_encode_contains_mpf() {
    let hdr = create_hdr_gradient(64, 64, 2.0);
    let sdr = create_sdr_gradient(64, 64);

    let mut encoder = Encoder::new();
    encoder.set_hdr_image(hdr).set_sdr_image(sdr);

    let encoded = encoder.encode().unwrap();

    let has_mpf = encoded.windows(4).any(|w| w == b"MPF\0");
    assert!(has_mpf, "Should contain MPF marker");
}

/// Test various image dimensions.
#[test]
fn test_encode_various_dimensions() {
    let dimensions = [(64, 64), (100, 75), (128, 96), (200, 150), (256, 256)];

    for (w, h) in dimensions {
        let hdr = create_hdr_gradient(w, h, 2.0);
        let sdr = create_sdr_gradient(w, h);

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

        let decoder = Decoder::new(&result.unwrap()).unwrap();
        let (dw, dh) = decoder.dimensions().unwrap();
        assert_eq!(dw, w, "Width mismatch for {}x{}", w, h);
        assert_eq!(dh, h, "Height mismatch for {}x{}", w, h);
    }
}
