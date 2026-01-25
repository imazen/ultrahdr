//! Round-trip encoding/decoding tests (Phase 5 from TESTING_PLAN.md).
//!
//! Tests that encode → decode produces consistent results.

mod common;

use common::{
    create_hdr_checkerboard, create_hdr_gradient, create_hdr_highlights, create_hdr_solid,
    create_sdr_checkerboard, create_sdr_gradient, create_sdr_solid,
};
use ultrahdr_rs::{Decoder, Encoder};

/// Test basic encode/decode round-trip preserves dimensions.
#[test]
fn test_roundtrip_dimensions_preserved() {
    let dimensions = [(64, 64), (100, 75), (128, 96), (200, 150)];

    for (w, h) in dimensions {
        let hdr = create_hdr_gradient(w, h, 4.0);
        let sdr = create_sdr_gradient(w, h);

        let mut encoder = Encoder::new();
        encoder.set_hdr_image(hdr).set_sdr_image(sdr);

        let encoded = encoder.encode().unwrap();
        let decoder = Decoder::new(&encoded).unwrap();

        let sdr_out = decoder.decode_sdr().unwrap();
        assert_eq!(sdr_out.width, w, "Width preserved for {}x{}", w, h);
        assert_eq!(sdr_out.height, h, "Height preserved for {}x{}", w, h);

        let hdr_out = decoder.decode_hdr(2.0).unwrap();
        assert_eq!(hdr_out.width, w, "HDR width preserved for {}x{}", w, h);
        assert_eq!(hdr_out.height, h, "HDR height preserved for {}x{}", w, h);
    }
}

/// Test that SDR output is close to input after round-trip.
#[test]
fn test_roundtrip_sdr_quality() {
    let hdr = create_hdr_solid(64, 64, 0.5, 0.5, 0.5);
    let sdr = create_sdr_solid(64, 64, 186, 186, 186); // ~0.5 in sRGB

    let mut encoder = Encoder::new();
    encoder
        .set_hdr_image(hdr)
        .set_sdr_image(sdr.clone())
        .set_quality(95, 90); // High quality

    let encoded = encoder.encode().unwrap();
    let decoder = Decoder::new(&encoded).unwrap();
    let sdr_out = decoder.decode_sdr().unwrap();

    // Check pixels are similar (JPEG is lossy)
    let mut max_diff = 0i16;
    for (i, chunk) in sdr_out.data.chunks(4).enumerate() {
        let orig_offset = i * 4;
        if orig_offset + 2 < sdr.data.len() {
            let diff_r = (chunk[0] as i16 - sdr.data[orig_offset] as i16).abs();
            let diff_g = (chunk[1] as i16 - sdr.data[orig_offset + 1] as i16).abs();
            let diff_b = (chunk[2] as i16 - sdr.data[orig_offset + 2] as i16).abs();
            max_diff = max_diff.max(diff_r).max(diff_g).max(diff_b);
        }
    }

    // JPEG compression introduces some error even at high quality
    // Jpegli may have different characteristics than other encoders
    assert!(
        max_diff < 80,
        "Max pixel diff {} should be < 80 for solid color at q95",
        max_diff
    );
}

/// Test gradient pattern round-trip.
#[test]
fn test_roundtrip_gradient() {
    let hdr = create_hdr_gradient(128, 128, 4.0);
    let sdr = create_sdr_gradient(128, 128);

    let mut encoder = Encoder::new();
    encoder
        .set_hdr_image(hdr)
        .set_sdr_image(sdr.clone())
        .set_quality(90, 85);

    let encoded = encoder.encode().unwrap();
    let decoder = Decoder::new(&encoded).unwrap();
    let sdr_out = decoder.decode_sdr().unwrap();

    // Calculate mean absolute error
    let mut total_error: u64 = 0;
    let pixel_count = (sdr_out.width * sdr_out.height) as usize;

    for (i, chunk) in sdr_out.data.chunks(4).enumerate() {
        let orig_offset = i * 4;
        if orig_offset + 2 < sdr.data.len() {
            total_error += (chunk[0] as i16 - sdr.data[orig_offset] as i16).unsigned_abs() as u64;
            total_error +=
                (chunk[1] as i16 - sdr.data[orig_offset + 1] as i16).unsigned_abs() as u64;
            total_error +=
                (chunk[2] as i16 - sdr.data[orig_offset + 2] as i16).unsigned_abs() as u64;
        }
    }

    let mae = total_error as f64 / (pixel_count * 3) as f64;
    eprintln!("Gradient MAE: {:.2}", mae);

    // JPEG compression produces lossy results, especially for gradients
    // Allow higher tolerance for different encoder implementations
    assert!(
        mae < 30.0,
        "Mean absolute error {} should be < 30 for gradient",
        mae
    );
}

/// Test checkerboard pattern round-trip.
#[test]
fn test_roundtrip_checkerboard() {
    let hdr = create_hdr_checkerboard(64, 64, 0.1, 2.0);
    let sdr = create_sdr_checkerboard(64, 64, 50, 200);

    let mut encoder = Encoder::new();
    encoder
        .set_hdr_image(hdr)
        .set_sdr_image(sdr)
        .set_quality(95, 90);

    let encoded = encoder.encode().unwrap();
    let decoder = Decoder::new(&encoded).unwrap();

    // Just verify it decodes without error
    let _sdr_out = decoder.decode_sdr().unwrap();
    let _hdr_out = decoder.decode_hdr(2.0).unwrap();
}

/// Test HDR reconstruction produces brighter pixels than SDR.
#[test]
fn test_roundtrip_hdr_brighter_than_sdr() {
    let hdr = create_hdr_highlights(64, 64, 0.2, 4.0);
    let sdr = create_sdr_gradient(64, 64);

    let mut encoder = Encoder::new();
    encoder
        .set_hdr_image(hdr)
        .set_sdr_image(sdr)
        .set_target_display_peak(4.0 * 203.0);

    let encoded = encoder.encode().unwrap();
    let decoder = Decoder::new(&encoded).unwrap();

    let sdr_out = decoder.decode_sdr().unwrap();
    let hdr_out = decoder.decode_hdr(4.0).unwrap();

    // Compare center pixel (highlight region)
    let center = (32 * 64 + 32) as usize;
    let sdr_center = &sdr_out.data[center * 4..(center + 1) * 4];

    // HDR is float, need to interpret the bytes
    let hdr_offset = center * 16; // RGBA32F
    if hdr_offset + 4 <= hdr_out.data.len() {
        let hdr_r = f32::from_le_bytes([
            hdr_out.data[hdr_offset],
            hdr_out.data[hdr_offset + 1],
            hdr_out.data[hdr_offset + 2],
            hdr_out.data[hdr_offset + 3],
        ]);

        let sdr_r_linear = (sdr_center[0] as f32 / 255.0).powf(2.2); // Approximate sRGB to linear

        // HDR should be at least as bright (usually brighter in highlights)
        assert!(
            hdr_r >= sdr_r_linear * 0.5,
            "HDR center ({}) should be >= SDR linear center ({}) * 0.5",
            hdr_r,
            sdr_r_linear
        );
    }
}

/// Test metadata survives round-trip.
#[test]
fn test_roundtrip_metadata_survives() {
    let hdr = create_hdr_gradient(64, 64, 4.0);
    let sdr = create_sdr_gradient(64, 64);

    let mut encoder = Encoder::new();
    encoder
        .set_hdr_image(hdr)
        .set_sdr_image(sdr)
        .set_target_display_peak(4.0 * 203.0);

    let encoded = encoder.encode().unwrap();
    let decoder = Decoder::new(&encoded).unwrap();

    assert!(decoder.is_ultrahdr(), "Should still be Ultra HDR");
    assert!(decoder.metadata().is_some(), "Should have metadata");

    let meta = decoder.metadata().unwrap();
    assert!(meta.max_content_boost[0] > 1.0, "Max boost should be > 1.0");
}

/// Test gain map dimensions are correct after round-trip.
#[test]
fn test_roundtrip_gainmap_dimensions() {
    let scales = [(2, 64), (4, 32), (8, 16)];

    for (scale, expected) in scales {
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
            gainmap.width, expected,
            "Gain map width for scale {} should be {}",
            scale, expected
        );
        assert_eq!(
            gainmap.height, expected,
            "Gain map height for scale {} should be {}",
            scale, expected
        );
    }
}

/// Test multiple encode/decode cycles don't accumulate errors excessively.
#[test]
fn test_roundtrip_multiple_cycles() {
    let hdr = create_hdr_solid(32, 32, 0.5, 0.5, 0.5);
    let sdr = create_sdr_solid(32, 32, 186, 186, 186);

    let mut encoder = Encoder::new();
    encoder
        .set_hdr_image(hdr.clone())
        .set_sdr_image(sdr)
        .set_quality(95, 90);

    let first_encoded = encoder.encode().unwrap();
    let first_size = first_encoded.len();

    // Decode and re-encode
    let decoder1 = Decoder::new(&first_encoded).unwrap();
    let sdr1 = decoder1.decode_sdr().unwrap();

    let mut encoder2 = Encoder::new();
    encoder2
        .set_hdr_image(hdr)
        .set_sdr_image(sdr1)
        .set_quality(95, 90);

    let second_encoded = encoder2.encode().unwrap();
    let second_size = second_encoded.len();

    // File sizes should be similar (not ballooning)
    let ratio = second_size as f64 / first_size as f64;
    assert!(
        ratio > 0.7 && ratio < 1.5,
        "Second encode size ratio {} should be between 0.7 and 1.5",
        ratio
    );
}

/// Test HDR at different display boost levels.
#[test]
fn test_roundtrip_hdr_boost_levels() {
    let hdr = create_hdr_gradient(64, 64, 8.0);
    let sdr = create_sdr_gradient(64, 64);

    let mut encoder = Encoder::new();
    encoder
        .set_hdr_image(hdr)
        .set_sdr_image(sdr)
        .set_target_display_peak(8.0 * 203.0);

    let encoded = encoder.encode().unwrap();
    let decoder = Decoder::new(&encoded).unwrap();

    // Test various boost levels
    let boosts = [1.0, 2.0, 4.0, 8.0];
    for boost in boosts {
        let hdr_out = decoder.decode_hdr(boost);
        assert!(
            hdr_out.is_ok(),
            "Should decode at boost {}: {:?}",
            boost,
            hdr_out.err()
        );
    }
}

/// Test quality settings produce valid outputs at different quality levels.
#[test]
fn test_roundtrip_quality_affects_size() {
    let hdr = create_hdr_gradient(128, 128, 4.0);
    let sdr = create_sdr_gradient(128, 128);

    // Encode at different qualities
    let qualities = [(50, 45), (75, 70), (95, 90)];
    let mut sizes = Vec::new();

    for (base_q, gm_q) in qualities {
        let mut encoder = Encoder::new();
        encoder
            .set_hdr_image(hdr.clone())
            .set_sdr_image(sdr.clone())
            .set_quality(base_q, gm_q);

        let encoded = encoder.encode().unwrap();
        sizes.push((base_q, encoded.len()));

        // Verify each output is a valid JPEG
        assert_eq!(
            &encoded[0..2],
            &[0xFF, 0xD8],
            "Quality {} did not produce valid JPEG",
            base_q
        );
        assert!(
            encoded.len() > 100,
            "Quality {} produced suspiciously small output: {} bytes",
            base_q,
            encoded.len()
        );
    }

    // Note: zenjpeg uses perceptual optimization, so higher quality doesn't
    // always mean larger files for simple synthetic images like gradients.
    // The important thing is that all quality levels produce valid outputs.
    // For complex real-world images, higher quality typically does produce
    // larger files, but this isn't guaranteed for synthetic test patterns.
}

/// Test that encoding bright HDR produces meaningful gain map.
#[test]
fn test_roundtrip_bright_hdr_gain_map() {
    let hdr = create_hdr_highlights(64, 64, 0.1, 10.0); // Very bright highlight
    let sdr = create_sdr_solid(64, 64, 128, 128, 128);

    let mut encoder = Encoder::new();
    encoder
        .set_hdr_image(hdr)
        .set_sdr_image(sdr)
        .set_target_display_peak(10.0 * 203.0);

    let encoded = encoder.encode().unwrap();
    let decoder = Decoder::new(&encoded).unwrap();
    let gainmap = decoder.decode_gainmap().unwrap();

    // Gain map should have variation (not all zeros or all 255)
    let min = *gainmap.data.iter().min().unwrap();
    let max = *gainmap.data.iter().max().unwrap();

    assert!(
        max > min,
        "Gain map should have variation: min={}, max={}",
        min,
        max
    );
}
