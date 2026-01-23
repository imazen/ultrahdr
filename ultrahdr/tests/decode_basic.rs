//! Basic decoding tests (Phase 2 from TESTING_PLAN.md).
//!
//! Tests the decoder against the existing test_ultrahdr.jpg sample file.

use ultrahdr::{Decoder, PixelFormat};

const TEST_ULTRAHDR: &[u8] = include_bytes!("../../test_ultrahdr.jpg");

/// Test that the decoder recognizes a valid Ultra HDR file.
#[test]
fn test_recognizes_ultrahdr() {
    let decoder = Decoder::new(TEST_ULTRAHDR).expect("Should parse Ultra HDR");
    assert!(decoder.is_ultrahdr(), "Should recognize as Ultra HDR");
}

/// Test that metadata can be extracted.
#[test]
fn test_extracts_metadata() {
    let decoder = Decoder::new(TEST_ULTRAHDR).unwrap();
    let metadata = decoder.metadata();

    assert!(metadata.is_some(), "Should have metadata");

    let meta = metadata.unwrap();
    assert!(
        meta.max_content_boost[0] >= 1.0,
        "Max boost should be >= 1.0, got {}",
        meta.max_content_boost[0]
    );
    assert!(
        meta.hdr_capacity_max >= 1.0,
        "HDR capacity max should be >= 1.0, got {}",
        meta.hdr_capacity_max
    );
}

/// Test SDR decoding produces valid output.
#[test]
fn test_decode_sdr() {
    let decoder = Decoder::new(TEST_ULTRAHDR).unwrap();
    let sdr = decoder.decode_sdr().expect("Should decode SDR");

    assert!(sdr.width > 0, "Width should be positive");
    assert!(sdr.height > 0, "Height should be positive");
    assert_eq!(sdr.format, PixelFormat::Rgba8, "SDR should be RGBA8");
    assert_eq!(
        sdr.data.len(),
        (sdr.width * sdr.height * 4) as usize,
        "Data size should match dimensions"
    );
}

/// Test gain map extraction.
#[test]
fn test_decode_gainmap() {
    let decoder = Decoder::new(TEST_ULTRAHDR).unwrap();

    assert!(decoder.is_ultrahdr());

    let gainmap = decoder.decode_gainmap().expect("Should decode gain map");

    assert!(gainmap.width > 0, "Gain map width should be positive");
    assert!(gainmap.height > 0, "Gain map height should be positive");
    assert_eq!(gainmap.channels, 1, "Gain map should be single-channel");
}

/// Test that gain map is smaller than primary image.
#[test]
fn test_gainmap_scaled() {
    let decoder = Decoder::new(TEST_ULTRAHDR).unwrap();
    let sdr = decoder.decode_sdr().unwrap();
    let gainmap = decoder.decode_gainmap().unwrap();

    // Gain map should be smaller (typically 1/4 size)
    assert!(
        gainmap.width <= sdr.width,
        "Gain map width {} should be <= SDR width {}",
        gainmap.width,
        sdr.width
    );
    assert!(
        gainmap.height <= sdr.height,
        "Gain map height {} should be <= SDR height {}",
        gainmap.height,
        sdr.height
    );
}

/// Test HDR reconstruction at SDR boost (1.0).
#[test]
fn test_decode_hdr_sdr_boost() {
    let decoder = Decoder::new(TEST_ULTRAHDR).unwrap();
    let hdr = decoder
        .decode_hdr(1.0)
        .expect("Should decode HDR at 1.0 boost");

    assert!(hdr.width > 0);
    assert!(hdr.height > 0);
    // At 1.0 boost, should be similar to SDR
}

/// Test HDR reconstruction at 4x boost.
#[test]
fn test_decode_hdr_4x_boost() {
    let decoder = Decoder::new(TEST_ULTRAHDR).unwrap();
    let hdr = decoder
        .decode_hdr(4.0)
        .expect("Should decode HDR at 4.0 boost");

    assert!(hdr.width > 0);
    assert!(hdr.height > 0);
    assert!(
        hdr.format.is_hdr(),
        "HDR output should be an HDR format, got {:?}",
        hdr.format
    );
}

/// Test dimensions are consistent across decode methods.
#[test]
fn test_dimensions_consistent() {
    let decoder = Decoder::new(TEST_ULTRAHDR).unwrap();

    let (w, h) = decoder.dimensions().expect("Should get dimensions");
    let sdr = decoder.decode_sdr().unwrap();
    let hdr = decoder.decode_hdr(2.0).unwrap();

    assert_eq!(w, sdr.width, "dimensions() width should match SDR");
    assert_eq!(h, sdr.height, "dimensions() height should match SDR");
    assert_eq!(sdr.width, hdr.width, "SDR and HDR width should match");
    assert_eq!(sdr.height, hdr.height, "SDR and HDR height should match");
}

/// Test raw gain map JPEG is accessible.
#[test]
fn test_gainmap_jpeg_accessible() {
    let decoder = Decoder::new(TEST_ULTRAHDR).unwrap();

    let gm_jpeg = decoder.gainmap_jpeg();
    assert!(gm_jpeg.is_some(), "Should access gain map JPEG");

    let data = gm_jpeg.unwrap();
    // Should be a valid JPEG
    assert_eq!(&data[0..2], &[0xFF, 0xD8], "Should start with JPEG SOI");
}

/// Test ICC profile extraction (if present).
#[test]
fn test_icc_profile_extraction() {
    let decoder = Decoder::new(TEST_ULTRAHDR).unwrap();

    // ICC profile may or may not be present
    let _icc = decoder.icc_profile();
    // Just verify the method doesn't crash
}

/// Test decoder rejects invalid data.
#[test]
fn test_rejects_invalid_data() {
    let result = Decoder::new(&[0, 1, 2, 3]);
    assert!(result.is_err(), "Should reject non-JPEG data");
}

/// Test decoder handles minimal JPEG (not Ultra HDR).
#[test]
fn test_handles_regular_jpeg() {
    // Minimal valid JPEG structure
    let regular_jpeg = [0xFF, 0xD8, 0xFF, 0xD9];
    let decoder = Decoder::new(&regular_jpeg);

    assert!(decoder.is_ok(), "Should accept minimal JPEG");
    assert!(
        !decoder.unwrap().is_ultrahdr(),
        "Regular JPEG should not be Ultra HDR"
    );
}

/// Test metadata values are in valid ranges.
#[test]
fn test_metadata_value_ranges() {
    let decoder = Decoder::new(TEST_ULTRAHDR).unwrap();
    let metadata = decoder.metadata().unwrap();

    // Check all channels
    for i in 0..3 {
        assert!(
            metadata.max_content_boost[i] >= metadata.min_content_boost[i],
            "max_content_boost[{}] should be >= min_content_boost[{}]",
            i,
            i
        );
        assert!(metadata.gamma[i] > 0.0, "gamma[{}] should be positive", i);
        assert!(
            metadata.offset_sdr[i] >= 0.0,
            "offset_sdr[{}] should be non-negative",
            i
        );
        assert!(
            metadata.offset_hdr[i] >= 0.0,
            "offset_hdr[{}] should be non-negative",
            i
        );
    }

    assert!(
        metadata.hdr_capacity_max >= metadata.hdr_capacity_min,
        "hdr_capacity_max should be >= hdr_capacity_min"
    );
}

/// Test pixel values are in valid range.
#[test]
fn test_pixel_values_valid() {
    let decoder = Decoder::new(TEST_ULTRAHDR).unwrap();
    let sdr = decoder.decode_sdr().unwrap();

    // Check first few pixels are valid RGBA
    for chunk in sdr.data.chunks(4).take(100) {
        assert_eq!(chunk.len(), 4, "Each pixel should be 4 bytes");
        // Alpha should typically be 255 for opaque images
    }
}
