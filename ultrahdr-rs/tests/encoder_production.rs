//! Tests for the production encoder API (`encode_ultrahdr` and `encode_from_jpegs`).
//!
//! These tests don't require zenjpeg or the `_test-helpers` feature.
//! They work with pre-built JPEG bytes and the public `encode_ultrahdr` function.

use ultrahdr_rs::{ColorGamut, Decoder, Encoder, GainMapMetadata, encode_ultrahdr};

/// Minimal valid JPEG for testing (SOI + DQT + SOF + DHT + SOS + scan + EOI).
///
/// This is a 1x1 black pixel JPEG. It's syntactically valid and decodable.
fn minimal_jpeg() -> Vec<u8> {
    vec![
        0xFF, 0xD8, // SOI
        // DQT (quantization table)
        0xFF, 0xDB, 0x00, 0x43, 0x00, // DQT marker, length 67, table 0
        16, 11, 10, 16, 24, 40, 51, 61, 12, 12, 14, 19, 26, 58, 60, 55, 14, 13, 16, 24, 40, 57, 69,
        56, 14, 17, 22, 29, 51, 87, 80, 62, 18, 22, 37, 56, 68, 109, 103, 77, 24, 35, 55, 64, 81,
        104, 113, 92, 49, 64, 78, 87, 103, 121, 120, 101, 72, 92, 95, 98, 112, 100, 103,
        99, // 64 QT values
        // SOF0 (start of frame, baseline, 1x1 grayscale)
        0xFF, 0xC0, 0x00, 0x0B, 0x08, 0x00, 0x01, 0x00, 0x01, 0x01, 0x01, 0x11, 0x00,
        // DHT (Huffman table - DC)
        0xFF, 0xC4, 0x00, 0x1F, 0x00, // class 0 (DC), table 0
        0x00, 0x01, 0x05, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09, 0x0A, 0x0B,
        // DHT (Huffman table - AC)
        0xFF, 0xC4, 0x00, 0xB5, 0x10, // class 1 (AC), table 0
        0x00, 0x02, 0x01, 0x03, 0x03, 0x02, 0x04, 0x03, 0x05, 0x05, 0x04, 0x04, 0x00, 0x00, 0x01,
        0x7D, 0x01, 0x02, 0x03, 0x00, 0x04, 0x11, 0x05, 0x12, 0x21, 0x31, 0x41, 0x06, 0x13, 0x51,
        0x61, 0x07, 0x22, 0x71, 0x14, 0x32, 0x81, 0x91, 0xA1, 0x08, 0x23, 0x42, 0xB1, 0xC1, 0x15,
        0x52, 0xD1, 0xF0, 0x24, 0x33, 0x62, 0x72, 0x82, 0x09, 0x0A, 0x16, 0x17, 0x18, 0x19, 0x1A,
        0x25, 0x26, 0x27, 0x28, 0x29, 0x2A, 0x34, 0x35, 0x36, 0x37, 0x38, 0x39, 0x3A, 0x43, 0x44,
        0x45, 0x46, 0x47, 0x48, 0x49, 0x4A, 0x53, 0x54, 0x55, 0x56, 0x57, 0x58, 0x59, 0x5A, 0x63,
        0x64, 0x65, 0x66, 0x67, 0x68, 0x69, 0x6A, 0x73, 0x74, 0x75, 0x76, 0x77, 0x78, 0x79, 0x7A,
        0x83, 0x84, 0x85, 0x86, 0x87, 0x88, 0x89, 0x8A, 0x92, 0x93, 0x94, 0x95, 0x96, 0x97, 0x98,
        0x99, 0x9A, 0xA2, 0xA3, 0xA4, 0xA5, 0xA6, 0xA7, 0xA8, 0xA9, 0xAA, 0xB2, 0xB3, 0xB4, 0xB5,
        0xB6, 0xB7, 0xB8, 0xB9, 0xBA, 0xC2, 0xC3, 0xC4, 0xC5, 0xC6, 0xC7, 0xC8, 0xC9, 0xCA, 0xD2,
        0xD3, 0xD4, 0xD5, 0xD6, 0xD7, 0xD8, 0xD9, 0xDA, 0xE1, 0xE2, 0xE3, 0xE4, 0xE5, 0xE6, 0xE7,
        0xE8, 0xE9, 0xEA, 0xF1, 0xF2, 0xF3, 0xF4, 0xF5, 0xF6, 0xF7, 0xF8, 0xF9, 0xFA,
        // SOS (start of scan)
        0xFF, 0xDA, 0x00, 0x08, 0x01, 0x01, 0x00, 0x00, 0x3F, 0x00,
        // Scan data (compressed 1x1 black pixel)
        0xFB, 0xD2, 0x8A, 0x00, // EOI
        0xFF, 0xD9,
    ]
}

/// Simplest possible JPEG for structure testing (may not be decodable).
fn stub_jpeg() -> Vec<u8> {
    vec![
        0xFF, 0xD8, // SOI
        0xFF, 0xE0, 0x00, 0x07, // APP0, length 7
        b'J', b'F', b'I', b'F', 0x00, // JFIF
        0xFF, 0xD9, // EOI
    ]
}

fn test_metadata() -> GainMapMetadata {
    GainMapMetadata {
        gain_map_max: [2.0; 3],
        gain_map_min: [0.0; 3],
        gamma: [1.0; 3],
        base_offset: [1.0 / 64.0; 3],
        alternate_offset: [1.0 / 64.0; 3],
        base_hdr_headroom: 0.0,
        alternate_hdr_headroom: 2.0,
        use_base_color_space: true,
    }
}

// ============================================================================
// encode_ultrahdr tests
// ============================================================================

/// Basic encode_ultrahdr produces valid output.
#[test]
fn test_encode_ultrahdr_basic() {
    let base = stub_jpeg();
    let gainmap = stub_jpeg();
    let metadata = test_metadata();

    let result = encode_ultrahdr(&base, &gainmap, &metadata, ColorGamut::Bt709);
    assert!(result.is_ok(), "encode_ultrahdr failed: {:?}", result.err());

    let encoded = result.unwrap();
    // Must be valid JPEG
    assert_eq!(&encoded[0..2], &[0xFF, 0xD8], "Should start with SOI");
    // Must contain XMP with hdrgm namespace
    let s = String::from_utf8_lossy(&encoded);
    assert!(
        s.contains("hdrgm") || s.contains("hdr-gain-map"),
        "Should contain hdrgm namespace"
    );
    // Must contain MPF
    assert!(
        encoded.windows(4).any(|w| w == b"MPF\0"),
        "Should contain MPF marker"
    );
    // Must contain gain map JPEG after primary
    assert!(
        encoded.len() > base.len() + gainmap.len(),
        "Output should be larger than both inputs combined"
    );
}

/// encode_ultrahdr output is recognized by Decoder.
#[test]
fn test_encode_ultrahdr_decoder_roundtrip() {
    let base = stub_jpeg();
    let gainmap = stub_jpeg();
    let metadata = test_metadata();

    let encoded = encode_ultrahdr(&base, &gainmap, &metadata, ColorGamut::Bt709).unwrap();
    let decoder = Decoder::new(&encoded).unwrap();

    assert!(
        decoder.is_ultrahdr(),
        "Output should be recognized as UltraHDR"
    );
    assert!(decoder.metadata().is_some(), "Should have metadata");
    assert!(decoder.gainmap_jpeg().is_some(), "Should have gainmap JPEG");
    assert!(decoder.primary_jpeg().is_some(), "Should have primary JPEG");
}

/// encode_ultrahdr with all gamut variants.
#[test]
fn test_encode_ultrahdr_gamuts() {
    let base = stub_jpeg();
    let gainmap = stub_jpeg();
    let metadata = test_metadata();

    for gamut in [ColorGamut::Bt709, ColorGamut::DisplayP3, ColorGamut::Bt2020] {
        let result = encode_ultrahdr(&base, &gainmap, &metadata, gamut);
        assert!(
            result.is_ok(),
            "encode_ultrahdr with {:?} failed: {:?}",
            gamut,
            result.err()
        );
    }
}

/// encode_ultrahdr embeds ICC profile.
#[test]
fn test_encode_ultrahdr_has_icc() {
    let base = stub_jpeg();
    let gainmap = stub_jpeg();
    let metadata = test_metadata();

    let encoded = encode_ultrahdr(&base, &gainmap, &metadata, ColorGamut::Bt709).unwrap();

    // Should contain ICC_PROFILE marker
    assert!(
        encoded.windows(12).any(|w| w.starts_with(b"ICC_PROFILE\0")),
        "Should contain ICC_PROFILE marker"
    );
}

/// encode_ultrahdr preserves metadata values through XMP.
#[test]
fn test_encode_ultrahdr_metadata_preserved() {
    let base = stub_jpeg();
    let gainmap = stub_jpeg();
    let metadata = GainMapMetadata {
        gain_map_max: [3.0; 3],
        gain_map_min: [0.0; 3],
        gamma: [1.0; 3],
        base_offset: [1.0 / 64.0; 3],
        alternate_offset: [1.0 / 64.0; 3],
        base_hdr_headroom: 0.0,
        alternate_hdr_headroom: 3.0,
        use_base_color_space: true,
    };

    let encoded = encode_ultrahdr(&base, &gainmap, &metadata, ColorGamut::Bt709).unwrap();
    let decoder = Decoder::new(&encoded).unwrap();
    let parsed = decoder.metadata().unwrap();

    // gain_map_max roundtrips through XMP (log2 domain)
    assert!(
        (parsed.gain_map_max[0] - 3.0).abs() < 0.1,
        "gain_map_max should roundtrip: got {}",
        parsed.gain_map_max[0]
    );
    assert!(
        (parsed.alternate_hdr_headroom - 3.0).abs() < 0.1,
        "alternate_hdr_headroom should roundtrip: got {}",
        parsed.alternate_hdr_headroom
    );
}

// ============================================================================
// Encoder::encode_from_jpegs tests
// ============================================================================

/// encode_from_jpegs requires all three components.
#[test]
fn test_encode_from_jpegs_requires_all() {
    // No base JPEG
    let mut encoder = Encoder::new();
    encoder.set_gainmap_jpeg(stub_jpeg(), test_metadata());
    assert!(encoder.encode_from_jpegs().is_err());

    // No gainmap JPEG
    let mut encoder = Encoder::new();
    encoder.set_base_jpeg(stub_jpeg());
    assert!(encoder.encode_from_jpegs().is_err());

    // No metadata (implicit — set_base_jpeg alone)
    let encoder = Encoder::new();
    assert!(encoder.encode_from_jpegs().is_err());
}

/// encode_from_jpegs produces valid UltraHDR.
#[test]
fn test_encode_from_jpegs_basic() {
    let mut encoder = Encoder::new();
    encoder
        .set_base_jpeg(stub_jpeg())
        .set_gainmap_jpeg(stub_jpeg(), test_metadata());

    let result = encoder.encode_from_jpegs();
    assert!(
        result.is_ok(),
        "encode_from_jpegs failed: {:?}",
        result.err()
    );

    let encoded = result.unwrap();
    let decoder = Decoder::new(&encoded).unwrap();
    assert!(decoder.is_ultrahdr());
}

/// Quality settings don't affect encode_from_jpegs (pre-encoded JPEGs).
#[test]
fn test_encode_from_jpegs_ignores_quality() {
    let mut encoder = Encoder::new();
    encoder
        .set_base_jpeg(stub_jpeg())
        .set_gainmap_jpeg(stub_jpeg(), test_metadata())
        .set_quality(1, 1);

    let result = encoder.encode_from_jpegs();
    assert!(result.is_ok());
}

/// Encoder::set_compressed_sdr and set_base_jpeg are aliases.
#[test]
fn test_set_compressed_sdr_alias() {
    let mut encoder = Encoder::new();
    encoder
        .set_compressed_sdr(stub_jpeg())
        .set_existing_gainmap_jpeg(stub_jpeg(), test_metadata());

    let result = encoder.encode_from_jpegs();
    assert!(result.is_ok());
}
