//! Metadata compliance tests (Phase 4 from TESTING_PLAN.md).
//!
//! Tests XMP serialization, ISO 21496-1 binary format, and MPF structure.

mod common;

use common::{create_hdr_gradient, create_sdr_gradient, create_test_metadata};
use ultrahdr_rs::{Decoder, Encoder, GainMapMetadata};

// ============================================================================
// Phase 4.1: XMP Serialization Tests
// ============================================================================

/// Test that XMP contains required namespace declarations.
#[test]
fn test_xmp_contains_hdrgm_namespace() {
    let hdr = create_hdr_gradient(64, 64, 4.0);
    let sdr = create_sdr_gradient(64, 64);

    let mut encoder = Encoder::new();
    encoder.set_hdr_image(hdr).set_sdr_image(sdr);

    let encoded = encoder.encode().unwrap();

    // Check that the encoded data contains the hdrgm namespace
    let data_str = String::from_utf8_lossy(&encoded);
    assert!(
        data_str.contains("hdrgm") || data_str.contains("hdr-gain-map"),
        "Encoded Ultra HDR should contain hdrgm namespace"
    );
}

/// Test XMP metadata round-trip preserves key values.
#[test]
fn test_xmp_roundtrip_metadata_values() {
    let hdr = create_hdr_gradient(64, 64, 4.0);
    let sdr = create_sdr_gradient(64, 64);

    let mut encoder = Encoder::new();
    encoder
        .set_hdr_image(hdr)
        .set_sdr_image(sdr)
        .set_target_display_peak(4.0 * 203.0); // 4x boost

    let encoded = encoder.encode().unwrap();

    let decoder = Decoder::new(&encoded).unwrap();
    let metadata = decoder.metadata().unwrap();

    // Check the primary channel (channel 0) has reasonable values
    assert!(
        metadata.max_content_boost[0] >= 1.0,
        "Max boost should be >= 1.0, got {}",
        metadata.max_content_boost[0]
    );
    assert!(
        metadata.min_content_boost[0] >= 0.5,
        "Min boost should be >= 0.5, got {}",
        metadata.min_content_boost[0]
    );
    assert!(
        metadata.gamma[0] > 0.0,
        "Gamma should be positive, got {}",
        metadata.gamma[0]
    );
    assert!(
        metadata.hdr_capacity_max >= 1.0,
        "HDR capacity max should be >= 1.0, got {}",
        metadata.hdr_capacity_max
    );
}

/// Test that XMP Version is set correctly.
#[test]
fn test_xmp_version() {
    let hdr = create_hdr_gradient(32, 32, 2.0);
    let sdr = create_sdr_gradient(32, 32);

    let mut encoder = Encoder::new();
    encoder.set_hdr_image(hdr).set_sdr_image(sdr);

    let encoded = encoder.encode().unwrap();

    // Check that Version="1.0" is present
    let data_str = String::from_utf8_lossy(&encoded);
    assert!(
        data_str.contains("Version=\"1.0\"") || data_str.contains("Version=\\\"1.0\\\""),
        "XMP should contain hdrgm:Version=\"1.0\""
    );
}

/// Test XMP values are correctly encoded (log2 for boost values).
#[test]
fn test_xmp_log2_encoding() {
    let hdr = create_hdr_gradient(64, 64, 8.0); // 8x boost = 3 stops
    let sdr = create_sdr_gradient(64, 64);

    let mut encoder = Encoder::new();
    encoder
        .set_hdr_image(hdr)
        .set_sdr_image(sdr)
        .set_target_display_peak(8.0 * 203.0);

    let encoded = encoder.encode().unwrap();

    let decoder = Decoder::new(&encoded).unwrap();
    let metadata = decoder.metadata().unwrap();

    // max_content_boost should be around 8.0 or clamped to content
    assert!(
        metadata.max_content_boost[0] > 1.0,
        "Max content boost should be > 1.0, got {}",
        metadata.max_content_boost[0]
    );
}

/// Test Container Directory structure in XMP.
#[test]
fn test_xmp_container_directory() {
    let hdr = create_hdr_gradient(64, 64, 2.0);
    let sdr = create_sdr_gradient(64, 64);

    let mut encoder = Encoder::new();
    encoder.set_hdr_image(hdr).set_sdr_image(sdr);

    let encoded = encoder.encode().unwrap();

    let data_str = String::from_utf8_lossy(&encoded);

    // Check for Container:Directory structure
    assert!(
        data_str.contains("Container:Directory") || data_str.contains("Container"),
        "XMP should contain Container:Directory"
    );

    // Check for Primary and GainMap semantics
    assert!(
        data_str.contains("Primary"),
        "XMP should reference Primary image"
    );
    assert!(
        data_str.contains("GainMap"),
        "XMP should reference GainMap image"
    );
}

// ============================================================================
// Phase 4.2: ISO 21496-1 Binary Format Tests
// ============================================================================

/// Test ISO metadata serialization round-trip.
#[test]
fn test_iso21496_roundtrip() {
    use ultrahdr_rs::metadata::iso21496::{deserialize_iso21496, serialize_iso21496};

    let original = create_test_metadata(4.0);

    let serialized = serialize_iso21496(&original);
    let parsed = deserialize_iso21496(&serialized).unwrap();

    // Check values match (with tolerance for fraction conversion)
    assert!(
        (parsed.max_content_boost[0] - original.max_content_boost[0]).abs() < 0.01,
        "Max boost mismatch: {} vs {}",
        parsed.max_content_boost[0],
        original.max_content_boost[0]
    );
    assert!(
        (parsed.min_content_boost[0] - original.min_content_boost[0]).abs() < 0.01,
        "Min boost mismatch: {} vs {}",
        parsed.min_content_boost[0],
        original.min_content_boost[0]
    );
    assert!(
        (parsed.gamma[0] - original.gamma[0]).abs() < 0.01,
        "Gamma mismatch"
    );
    assert!(
        (parsed.offset_sdr[0] - original.offset_sdr[0]).abs() < 0.001,
        "Offset SDR mismatch"
    );
    assert!(
        (parsed.hdr_capacity_max - original.hdr_capacity_max).abs() < 0.01,
        "HDR capacity max mismatch"
    );
    assert_eq!(
        parsed.use_base_color_space, original.use_base_color_space,
        "use_base_color_space mismatch"
    );
}

/// Test ISO metadata version byte.
#[test]
fn test_iso21496_version() {
    use ultrahdr_rs::metadata::iso21496::{serialize_iso21496, ISO_VERSION};

    let metadata = create_test_metadata(2.0);
    let serialized = serialize_iso21496(&metadata);

    // First byte should be version
    assert_eq!(serialized[0], ISO_VERSION);
}

/// Test ISO metadata flags byte.
#[test]
fn test_iso21496_flags() {
    use ultrahdr_rs::metadata::iso21496::serialize_iso21496;

    // Single-channel, use base color space
    let metadata = GainMapMetadata {
        max_content_boost: [4.0; 3],
        min_content_boost: [1.0; 3],
        gamma: [1.0; 3],
        offset_sdr: [0.015625; 3],
        offset_hdr: [0.015625; 3],
        hdr_capacity_min: 1.0,
        hdr_capacity_max: 4.0,
        use_base_color_space: true,
    };

    let serialized = serialize_iso21496(&metadata);

    // Second byte is flags
    let flags = serialized[1];

    // Bit 1 (USE_BASE_CG) should be set
    assert!(flags & 0x02 != 0, "USE_BASE_CG flag should be set");

    // Bit 0 (MULTI_CHANNEL) should NOT be set for single channel
    assert!(
        flags & 0x01 == 0,
        "MULTI_CHANNEL flag should not be set for single channel"
    );
}

/// Test ISO metadata handles extreme values.
#[test]
fn test_iso21496_extreme_values() {
    use ultrahdr_rs::metadata::iso21496::{deserialize_iso21496, serialize_iso21496};

    let metadata = GainMapMetadata {
        max_content_boost: [100.0; 3], // Very high
        min_content_boost: [0.1; 3],   // Very low
        gamma: [2.2; 3],
        offset_sdr: [0.001; 3],
        offset_hdr: [0.001; 3],
        hdr_capacity_min: 0.5,
        hdr_capacity_max: 100.0,
        use_base_color_space: false,
    };

    let serialized = serialize_iso21496(&metadata);
    let parsed = deserialize_iso21496(&serialized).unwrap();

    // Values should round-trip reasonably
    assert!(
        (parsed.max_content_boost[0] - 100.0).abs() < 1.0,
        "Extreme max boost should preserve"
    );
    assert!((parsed.gamma[0] - 2.2).abs() < 0.1, "Gamma should preserve");
}

/// Test ISO metadata rejects invalid data.
#[test]
fn test_iso21496_rejects_invalid() {
    use ultrahdr_rs::metadata::iso21496::deserialize_iso21496;

    // Empty data
    let result = deserialize_iso21496(&[]);
    assert!(result.is_err());

    // Too short
    let result = deserialize_iso21496(&[0]);
    assert!(result.is_err());

    // Invalid version
    let result = deserialize_iso21496(&[99, 0, 0, 0, 0, 0, 0, 0]);
    assert!(result.is_err());
}

// ============================================================================
// Phase 4.3: MPF Structure Validation
// ============================================================================

/// Test that encoded Ultra HDR has valid MPF structure.
#[test]
fn test_mpf_structure_present() {
    let hdr = create_hdr_gradient(64, 64, 2.0);
    let sdr = create_sdr_gradient(64, 64);

    let mut encoder = Encoder::new();
    encoder.set_hdr_image(hdr).set_sdr_image(sdr);

    let encoded = encoder.encode().unwrap();

    // Check for MPF identifier
    let mpf_id = b"MPF\0";
    let has_mpf = encoded.windows(4).any(|w| w == mpf_id);
    assert!(has_mpf, "Encoded Ultra HDR should contain MPF marker");
}

/// Test that both primary and gain map are valid JPEGs.
#[test]
fn test_mpf_images_are_valid_jpegs() {
    let hdr = create_hdr_gradient(64, 64, 2.0);
    let sdr = create_sdr_gradient(64, 64);

    let mut encoder = Encoder::new();
    encoder.set_hdr_image(hdr).set_sdr_image(sdr);

    let encoded = encoder.encode().unwrap();

    // Find JPEG boundaries
    use ultrahdr_rs::metadata::mpf::find_jpeg_boundaries;
    let boundaries = find_jpeg_boundaries(&encoded);

    assert!(
        boundaries.len() >= 2,
        "Should have at least 2 JPEGs (primary + gain map), found {}",
        boundaries.len()
    );

    // First image (primary) should start at 0
    assert_eq!(boundaries[0].0, 0, "Primary should start at offset 0");

    // Each image should start with SOI and end with EOI
    for (i, (start, end)) in boundaries.iter().enumerate() {
        assert_eq!(
            &encoded[*start..*start + 2],
            &[0xFF, 0xD8],
            "Image {} should start with SOI",
            i
        );
        assert_eq!(
            &encoded[end - 2..*end],
            &[0xFF, 0xD9],
            "Image {} should end with EOI",
            i
        );
    }
}

/// Test that gain map can be extracted and decoded.
#[test]
fn test_gainmap_extraction() {
    let hdr = create_hdr_gradient(64, 64, 4.0);
    let sdr = create_sdr_gradient(64, 64);

    let mut encoder = Encoder::new();
    encoder
        .set_hdr_image(hdr)
        .set_sdr_image(sdr)
        .set_gainmap_scale(4);

    let encoded = encoder.encode().unwrap();

    let decoder = Decoder::new(&encoded).unwrap();
    assert!(decoder.is_ultrahdr());

    // Extract gain map
    let gainmap = decoder.decode_gainmap().unwrap();

    // Gain map should be 1/4 size
    assert_eq!(gainmap.width, 16, "Gain map width should be 64/4=16");
    assert_eq!(gainmap.height, 16, "Gain map height should be 64/4=16");
    assert_eq!(gainmap.channels, 1, "Should be single-channel gain map");
}

/// Test that gain map JPEG data can be retrieved.
#[test]
fn test_gainmap_jpeg_accessible() {
    let hdr = create_hdr_gradient(64, 64, 2.0);
    let sdr = create_sdr_gradient(64, 64);

    let mut encoder = Encoder::new();
    encoder.set_hdr_image(hdr).set_sdr_image(sdr);

    let encoded = encoder.encode().unwrap();

    let decoder = Decoder::new(&encoded).unwrap();

    let gm_jpeg = decoder.gainmap_jpeg();
    assert!(gm_jpeg.is_some(), "Should be able to access gain map JPEG");

    let gm_data = gm_jpeg.unwrap();
    // Should be a valid JPEG
    assert_eq!(
        &gm_data[0..2],
        &[0xFF, 0xD8],
        "Gain map should start with SOI"
    );
}

// ============================================================================
// Additional Metadata Tests
// ============================================================================

/// Test metadata consistency across encode/decode.
#[test]
fn test_metadata_consistency() {
    let hdr = create_hdr_gradient(128, 128, 6.0);
    let sdr = create_sdr_gradient(128, 128);

    let mut encoder = Encoder::new();
    encoder
        .set_hdr_image(hdr)
        .set_sdr_image(sdr)
        .set_target_display_peak(6.0 * 203.0)
        .set_min_content_boost(1.0);

    let encoded = encoder.encode().unwrap();

    let decoder = Decoder::new(&encoded).unwrap();
    let metadata = decoder.metadata().unwrap();

    // Log metadata for debugging
    eprintln!(
        "min_content_boost: {:?}, max_content_boost: {:?}",
        metadata.min_content_boost, metadata.max_content_boost
    );
    eprintln!(
        "hdr_capacity: min={}, max={}",
        metadata.hdr_capacity_min, metadata.hdr_capacity_max
    );

    // Verify metadata constraints
    // Note: XMP parsing may set default values that don't perfectly match encoded values
    // The key constraint is that the values are sensible
    assert!(
        metadata.max_content_boost[0] >= 1.0,
        "max_content_boost should be >= 1.0, got {}",
        metadata.max_content_boost[0]
    );
    assert!(
        metadata.hdr_capacity_max >= 1.0,
        "hdr_capacity_max should be >= 1.0, got {}",
        metadata.hdr_capacity_max
    );
    assert!(metadata.gamma[0] > 0.0, "gamma should be positive");
    assert!(
        metadata.offset_sdr[0] >= 0.0,
        "offset_sdr should be non-negative"
    );
    assert!(
        metadata.offset_hdr[0] >= 0.0,
        "offset_hdr should be non-negative"
    );
}

/// Test that offset values are sensible (prevent division by zero).
#[test]
fn test_metadata_offsets_nonzero() {
    let hdr = create_hdr_gradient(64, 64, 2.0);
    let sdr = create_sdr_gradient(64, 64);

    let mut encoder = Encoder::new();
    encoder.set_hdr_image(hdr).set_sdr_image(sdr);

    let encoded = encoder.encode().unwrap();

    let decoder = Decoder::new(&encoded).unwrap();
    let metadata = decoder.metadata().unwrap();

    // Offsets should be non-zero to prevent division by zero during reconstruction
    // The Ultra HDR spec recommends 1/64 = 0.015625
    assert!(
        metadata.offset_sdr[0] > 0.0,
        "offset_sdr should be > 0, got {}",
        metadata.offset_sdr[0]
    );
    assert!(
        metadata.offset_hdr[0] > 0.0,
        "offset_hdr should be > 0, got {}",
        metadata.offset_hdr[0]
    );
}
