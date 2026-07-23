//! Decoder robustness tests for non-UltraHDR and malformed inputs.
//!
//! These tests pin the decoder's graceful degradation behavior:
//! - Non-UltraHDR JPEGs should parse without error but not claim is_ultrahdr
//! - Malformed data should produce errors, not panics
//! - Gain map access on plain JPEGs should return None

mod common;

use ultrahdr_rs::{Decoder, Error, ResourceLimits};

/// Helper: Build a JPEG with custom APP segments before EOI.
fn build_jpeg_with_segments(segments: &[Vec<u8>]) -> Vec<u8> {
    let mut data = vec![0xFF, 0xD8]; // SOI
    for seg in segments {
        data.extend_from_slice(seg);
    }
    data.extend_from_slice(&[0xFF, 0xD9]); // EOI
    data
}

/// Build an APP1 XMP segment with the given XML content.
fn xmp_segment(xml: &str) -> Vec<u8> {
    let namespace = b"http://ns.adobe.com/xap/1.0/\0";
    let payload_len = namespace.len() + xml.len();
    let total_len = 2 + payload_len; // length field includes itself (2 bytes) + payload
    let mut seg = Vec::with_capacity(4 + payload_len);
    seg.push(0xFF);
    seg.push(0xE1); // APP1
    seg.push(((total_len >> 8) & 0xFF) as u8);
    seg.push((total_len & 0xFF) as u8);
    seg.extend_from_slice(namespace);
    seg.extend_from_slice(xml.as_bytes());
    seg
}

/// Build an APP0 JFIF segment.
fn jfif_segment() -> Vec<u8> {
    vec![
        0xFF, 0xE0, 0x00, 0x10, // APP0, length 16
        b'J', b'F', b'I', b'F', 0x00, // JFIF identifier
        0x01, 0x01, 0x00, // version 1.1, units=none
        0x00, 0x01, 0x00, 0x01, // density 1x1
        0x00, 0x00, // no thumbnail
    ]
}

// ============================================================================
// Non-UltraHDR JPEG handling
// ============================================================================

/// JPEG with XMP but no hdrgm namespace (e.g. camera JPEG with EXIF/XMP).
#[test]
fn test_jpeg_xmp_without_hdrgm() {
    let xml = r#"<?xpacket begin="" id="W5M0MpCehiHzreSzNTczkc9d"?>
<x:xmpmeta xmlns:x="adobe:ns:meta/">
  <rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
    <rdf:Description rdf:about=""
        xmlns:xmp="http://ns.adobe.com/xap/1.0/"
        xmp:CreatorTool="TestCamera"
        xmp:CreateDate="2024-01-01T00:00:00"/>
  </rdf:RDF>
</x:xmpmeta>
<?xpacket end="w"?>"#;

    let jpeg = build_jpeg_with_segments(&[jfif_segment(), xmp_segment(xml)]);
    let decoder = Decoder::new(&jpeg).unwrap();

    assert!(!decoder.is_ultrahdr(), "Camera XMP should not be UltraHDR");
    assert!(decoder.metadata().is_none(), "Should have no metadata");
    assert!(
        decoder.gainmap_jpeg().is_none(),
        "Should have no gain map JPEG"
    );
}

/// JPEG with APP0 JFIF only — basic camera output.
#[test]
fn test_plain_jpeg_with_jfif() {
    let jpeg = build_jpeg_with_segments(&[jfif_segment()]);
    let decoder = Decoder::new(&jpeg).unwrap();

    assert!(!decoder.is_ultrahdr());
    assert!(decoder.metadata().is_none());
    assert!(decoder.gainmap_jpeg().is_none());
    assert!(
        decoder.primary_jpeg().is_some(),
        "Primary should be whole file"
    );
}

/// Gain map JPEG and metadata are both None on plain JPEG.
#[test]
fn test_plain_jpeg_no_gainmap() {
    let jpeg = vec![0xFF, 0xD8, 0xFF, 0xD9];
    let decoder = Decoder::new(&jpeg).unwrap();

    assert!(!decoder.is_ultrahdr());
    assert!(decoder.gainmap_jpeg().is_none());
    assert!(decoder.metadata().is_none());
}

/// XMP with hdrgm namespace but malformed XML — should still parse without panic.
#[test]
fn test_xmp_malformed_xml() {
    // Has hdrgm: but the XML is invalid (unclosed tag)
    let xml = r#"<rdf:Description hdrgm:Version="1.0" hdrgm:GainMapMax="2.0">"#;
    let jpeg = build_jpeg_with_segments(&[xmp_segment(xml)]);
    let decoder = Decoder::new(&jpeg).unwrap();

    // Should not panic — may or may not detect as UltraHDR depending on parser
    // The important thing is no panic
    let _ = decoder.is_ultrahdr();
    let _ = decoder.metadata();
}

/// XMP claiming gain map exists but no MPF or secondary image.
#[test]
fn test_xmp_hdrgm_but_no_gainmap() {
    // XMP says it's UltraHDR, but there's no actual gain map image
    let xml = r#"<?xpacket begin="" id="W5M0MpCehiHzreSzNTczkc9d"?>
<x:xmpmeta xmlns:x="adobe:ns:meta/">
  <rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
    <rdf:Description rdf:about=""
        xmlns:hdrgm="http://ns.adobe.com/hdr-gain-map/1.0/"
        hdrgm:Version="1.0"
        hdrgm:GainMapMax="2.0"
        hdrgm:GainMapMin="0.0"
        hdrgm:Gamma="1.0"
        hdrgm:OffsetSDR="0.015625"
        hdrgm:OffsetHDR="0.015625"
        hdrgm:HDRCapacityMax="2.0"
        hdrgm:HDRCapacityMin="0.0"/>
  </rdf:RDF>
</x:xmpmeta>
<?xpacket end="w"?>"#;

    let jpeg = build_jpeg_with_segments(&[xmp_segment(xml)]);
    let decoder = Decoder::new(&jpeg).unwrap();

    // XMP is valid UltraHDR, but no gain map image was found
    // The metadata parse should succeed, but gainmap_jpeg should be None
    assert!(
        decoder.metadata().is_some(),
        "XMP metadata should parse successfully"
    );
    // is_ultrahdr may be true (has metadata) even without the actual gain map image
}

// ============================================================================
// Truncated / corrupted data
// ============================================================================

/// Empty input should error.
#[test]
fn test_empty_input() {
    assert!(Decoder::new(&[]).is_err());
}

/// Just SOI, too short.
#[test]
fn test_just_soi() {
    assert!(Decoder::new(&[0xFF, 0xD8]).is_err());
}

/// Truncated mid-segment — should not panic.
#[test]
fn test_truncated_mid_segment() {
    // Start of an APP0 segment but truncated before length is complete
    let data = vec![
        0xFF, 0xD8, // SOI
        0xFF, 0xE0, 0x00, // APP0 truncated
    ];
    // Should not panic — may error or succeed with partial parse
    let _ = Decoder::new(&data);
}

/// JPEG with SOI but no EOI — should parse without panic.
#[test]
fn test_no_eoi() {
    let data = vec![
        0xFF, 0xD8, // SOI
        0xFF, 0xE0, 0x00, 0x07, b'J', b'F', b'I', b'F', 0x00, // APP0
              // No EOI
    ];
    let result = Decoder::new(&data);
    // Should not panic
    if let Ok(decoder) = result {
        assert!(!decoder.is_ultrahdr());
    }
}

/// Random garbage bytes.
#[test]
fn test_garbage_bytes() {
    let data = vec![0x42, 0x43, 0x44, 0x45, 0x46, 0x47, 0x48, 0x49];
    assert!(Decoder::new(&data).is_err());
}

/// Truncated file that cuts off mid-APP2 (MPF segment).
#[test]
fn test_truncated_mpf_segment() {
    let data = vec![
        0xFF, 0xD8, // SOI
        0xFF, 0xE2, 0x00, 0x20, // APP2, length 32
        b'M', b'P', b'F', 0x00, // MPF identifier
        b'M', b'M', // Big-endian
        0x00, 0x2A, // TIFF magic
              // Truncated here — should not panic
    ];
    let result = Decoder::new(&data);
    // Should not panic
    if let Ok(decoder) = result {
        assert!(!decoder.is_ultrahdr());
    }
}

// ============================================================================
// MPF edge cases
// ============================================================================

/// JPEG with MPF segment but only 1 image (not UltraHDR).
#[test]
fn test_mpf_single_image() {
    // Build MPF header claiming 1 image (just primary, no gain map)
    let mut mpf_data = Vec::new();
    mpf_data.extend_from_slice(b"MPF\0"); // identifier
    mpf_data.extend_from_slice(b"MM"); // big-endian
    mpf_data.extend_from_slice(&0x002Au16.to_be_bytes()); // TIFF magic
    mpf_data.extend_from_slice(&8u32.to_be_bytes()); // IFD offset
    // IFD: 2 entries
    mpf_data.extend_from_slice(&2u16.to_be_bytes());
    // Entry 1: Number of images = 1
    mpf_data.extend_from_slice(&0xB001u16.to_be_bytes()); // tag
    mpf_data.extend_from_slice(&4u16.to_be_bytes()); // type LONG
    mpf_data.extend_from_slice(&1u32.to_be_bytes()); // count
    mpf_data.extend_from_slice(&1u32.to_be_bytes()); // value: 1 image
    // Entry 2: MP Entry offset
    let mp_entry_offset = (8 + 2 + 24 + 4) as u32; // after IFD header + 2 entries + next IFD ptr
    mpf_data.extend_from_slice(&0xB002u16.to_be_bytes()); // tag
    mpf_data.extend_from_slice(&7u16.to_be_bytes()); // type UNDEFINED
    mpf_data.extend_from_slice(&16u32.to_be_bytes()); // count: 1 entry * 16 bytes
    mpf_data.extend_from_slice(&mp_entry_offset.to_be_bytes());
    // Next IFD: 0
    mpf_data.extend_from_slice(&0u32.to_be_bytes());
    // MP Entry: primary only
    mpf_data.extend_from_slice(&0x03_0000u32.to_be_bytes()); // attr: primary
    mpf_data.extend_from_slice(&1000u32.to_be_bytes()); // size
    mpf_data.extend_from_slice(&0u32.to_be_bytes()); // offset 0
    mpf_data.extend_from_slice(&0u32.to_be_bytes()); // dependent entries

    // Build APP2 segment
    let total_len = 2 + mpf_data.len();
    let mut app2 = vec![
        0xFF,
        0xE2,
        ((total_len >> 8) & 0xFF) as u8,
        (total_len & 0xFF) as u8,
    ];
    app2.extend_from_slice(&mpf_data);

    let jpeg = build_jpeg_with_segments(&[app2]);
    let decoder = Decoder::new(&jpeg).unwrap();

    // MPF has only 1 image — not UltraHDR
    assert!(
        !decoder.is_ultrahdr(),
        "Single-image MPF should not be UltraHDR"
    );
    assert!(decoder.gainmap_jpeg().is_none());
}

/// Two concatenated JPEGs without MPF — should detect via boundary scan.
#[test]
fn test_two_jpegs_no_mpf() {
    let data = vec![
        0xFF, 0xD8, // SOI 1
        0xFF, 0xE0, 0x00, 0x07, b'J', b'F', b'I', b'F', 0x00, // APP0
        0xFF, 0xD9, // EOI 1
        0xFF, 0xD8, // SOI 2
        0xFF, 0xD9, // EOI 2
    ];
    let decoder = Decoder::new(&data).unwrap();
    assert!(decoder.primary_jpeg().is_some());
    assert!(decoder.gainmap_jpeg().is_some());
}

// ============================================================================
// ICC profile on non-UltraHDR
// ============================================================================

/// ICC profile extraction returns None on plain JPEG.
#[test]
fn test_icc_profile_none_on_plain_jpeg() {
    let jpeg = build_jpeg_with_segments(&[jfif_segment()]);
    let decoder = Decoder::new(&jpeg).unwrap();
    assert!(decoder.icc_profile().is_none());
}

/// ICC profile with ICC_PROFILE marker is detected.
#[test]
fn test_icc_profile_detected() {
    // Build a fake ICC profile APP2 segment
    let mut icc_seg = Vec::new();
    let icc_data = vec![0u8; 100]; // fake ICC data
    let payload = {
        let mut p = Vec::new();
        p.extend_from_slice(b"ICC_PROFILE\0");
        p.push(1); // chunk 1
        p.push(1); // total 1
        p.extend_from_slice(&icc_data);
        p
    };
    let total_len = 2 + payload.len();
    icc_seg.push(0xFF);
    icc_seg.push(0xE2);
    icc_seg.push(((total_len >> 8) & 0xFF) as u8);
    icc_seg.push((total_len & 0xFF) as u8);
    icc_seg.extend_from_slice(&payload);

    let jpeg = build_jpeg_with_segments(&[icc_seg]);
    let decoder = Decoder::new(&jpeg).unwrap();
    let icc = decoder.icc_profile();
    assert!(icc.is_some(), "Should find ICC profile");
    assert_eq!(icc.unwrap(), icc_data, "ICC data should match");
}

/// Regression for issue #26: a real 7.6 KB UltraHDR sample whose MPF APP2 is
/// the FIRST marker after SOI (before any APP1), carrying attribute-form
/// hdrgm XMP (GainMapMax=2.072094, HDRCapacityMax=2.300448). zenjpeg's
/// `UltraHdrExtras` decodes this file's gain map fine; this decoder must
/// agree — two readers diverging on one file is how HDR renditions silently
/// vanish downstream.
#[test]
fn mpf_first_sample_detected_as_ultrahdr() {
    let data = include_bytes!("images/mpf_first_attribute_xmp.jpg");
    let d = Decoder::new(data).expect("Decoder::new must not error on a valid UltraHDR file");
    assert!(d.is_ultrahdr(), "is_ultrahdr must detect this layout");
    let meta = d.metadata().expect("hdrgm metadata must parse");
    assert!(
        (meta.channels[0].max - 2.072094).abs() < 1e-5,
        "gain max should be ~2.072094, got {}",
        meta.channels[0].max
    );
    assert!(
        (meta.alternate_hdr_headroom - 2.300448).abs() < 1e-5,
        "alternate headroom should be ~2.300448, got {}",
        meta.alternate_hdr_headroom
    );
    let gm = d.decode_gainmap().expect("gain map must decode");
    assert!(gm.width > 0 && gm.height > 0);
}

// ---------------------------------------------------------------------------
// Resource limits (issue #28): the front-door Decoder must be able to reject
// untrusted over-budget input with a clean Err — never a panic and never an
// unbounded allocation.
// ---------------------------------------------------------------------------

/// Build a syntactically valid JPEG prefix whose SOF0 header declares the
/// given (huge) dimensions. The scan data is absent — irrelevant, because a
/// limited decoder must reject at header-parse time, before allocating
/// pixel planes.
fn jpeg_with_sof_dimensions(width: u16, height: u16) -> Vec<u8> {
    let mut data = vec![0xFF, 0xD8]; // SOI
    // SOF0: len(17) precision(8) height width 3 components
    data.extend_from_slice(&[0xFF, 0xC0, 0x00, 0x11, 0x08]);
    data.extend_from_slice(&height.to_be_bytes());
    data.extend_from_slice(&width.to_be_bytes());
    data.extend_from_slice(&[
        0x03, // 3 components
        0x01, 0x22, 0x00, // Y:  id 1, 2x2 sampling, quant table 0
        0x02, 0x11, 0x01, // Cb: id 2, 1x1 sampling, quant table 1
        0x03, 0x11, 0x01, // Cr: id 3, 1x1 sampling, quant table 1
    ]);
    data.extend_from_slice(&[0xFF, 0xD9]); // EOI
    data
}

/// A JPEG header declaring 30000x30000 (900 MP — 1.8x the 500 MP hard cap)
/// must be rejected by the limited path with a clean typed error, at header
/// cost, without attempting a multi-gigabyte allocation.
#[test]
fn limits_reject_huge_sof_dimensions_cleanly() {
    let bomb = jpeg_with_sof_dimensions(30000, 30000);
    let decoder = Decoder::new_with_limits(&bomb, ResourceLimits::default())
        .expect("container parse of a plain JPEG prefix must succeed");
    let err = decoder
        .decode_sdr()
        .expect_err("900 MP header must not decode under default limits");
    assert!(
        matches!(err.error(), Error::LimitExceeded(_)),
        "expected LimitExceeded, got: {err:?}"
    );
}

/// Dimensions beyond even the JPEG format cap (65535x65535, ~4.29 GP) must
/// also come back as a clean Err on the limited path — never a panic or an
/// attempted multi-gigabyte allocation. (zenjpeg rejects these with its own
/// format-level dimension error before the pixel-cap check, so the exact
/// variant is not pinned — only that it is a clean typed error.)
#[test]
fn limits_reject_over_format_cap_dimensions_cleanly() {
    let bomb = jpeg_with_sof_dimensions(65535, 65535);
    let decoder = Decoder::new_with_limits(&bomb, ResourceLimits::default())
        .expect("container parse of a plain JPEG prefix must succeed");
    assert!(decoder.decode_sdr().is_err());
}

/// A caller-tightened pixel cap rejects an image that the default cap allows.
#[test]
fn limits_reject_over_caller_pixel_cap() {
    let encoded = encode_small_ultrahdr(64, 64);
    // Base is 64x64 = 4096 px; the gain map at the default 4x scale is
    // 16x16 = 256 px. Cap at 100 px so BOTH are over budget.
    let decoder = Decoder::new_with_limits(&encoded, ResourceLimits::new().with_max_pixels(100))
        .expect("parse must succeed");
    let err = decoder
        .decode_sdr()
        .expect_err("64x64 must be rejected under a 100-pixel cap");
    assert!(
        matches!(err.error(), Error::LimitExceeded(_)),
        "expected LimitExceeded, got: {err:?}"
    );
    // The gain map decode path is capped too (16x16 = 256 px > 100).
    let err = decoder
        .decode_gainmap()
        .expect_err("16x16 gain map must be rejected under a 100-pixel cap");
    assert!(
        matches!(err.error(), Error::LimitExceeded(_)),
        "expected LimitExceeded on gainmap, got: {err:?}"
    );
    // And the HDR reconstruction path.
    let err = decoder
        .decode_hdr(4.0)
        .expect_err("HDR reconstruction must be rejected under the cap");
    assert!(
        matches!(err.error(), Error::LimitExceeded(_)),
        "expected LimitExceeded on HDR, got: {err:?}"
    );
}

/// A caller memory cap bounds the decode output allocation.
#[test]
fn limits_reject_over_memory_cap() {
    let encoded = encode_small_ultrahdr(64, 64);
    // SDR output is 64x64x4 = 16384 bytes; cap below it.
    let decoder = Decoder::new_with_limits(&encoded, ResourceLimits::new().with_max_memory(4096))
        .expect("parse must succeed");
    let err = decoder
        .decode_sdr()
        .expect_err("64x64 RGBA (16 KiB) must be rejected under a 4 KiB memory cap");
    assert!(
        matches!(err.error(), Error::LimitExceeded(_)),
        "expected LimitExceeded, got: {err:?}"
    );
}

/// In-budget input must decode byte-identically through the limited path.
#[test]
fn limits_valid_input_decodes_byte_identical() {
    let encoded = encode_small_ultrahdr(64, 48);

    let plain = Decoder::new(&encoded).unwrap();
    let limited = Decoder::new_with_limits(&encoded, ResourceLimits::default()).unwrap();

    let sdr_plain = plain.decode_sdr().unwrap();
    let sdr_limited = limited.decode_sdr().unwrap();
    assert_eq!(sdr_plain.width(), sdr_limited.width());
    assert_eq!(sdr_plain.height(), sdr_limited.height());
    assert_eq!(
        sdr_plain.as_slice().as_strided_bytes(),
        sdr_limited.as_slice().as_strided_bytes(),
        "SDR pixels must be byte-identical with and without limits"
    );

    let gm_plain = plain.decode_gainmap().unwrap();
    let gm_limited = limited.decode_gainmap().unwrap();
    assert_eq!(gm_plain.data, gm_limited.data);

    let hdr_plain = plain.decode_hdr(4.0).unwrap();
    let hdr_limited = limited.decode_hdr(4.0).unwrap();
    assert_eq!(
        hdr_plain.as_slice().as_strided_bytes(),
        hdr_limited.as_slice().as_strided_bytes(),
        "HDR pixels must be byte-identical with and without limits"
    );
}

/// Encode a small real Ultra HDR JPEG for limit tests.
fn encode_small_ultrahdr(w: u32, h: u32) -> Vec<u8> {
    let hdr = common::create_hdr_gradient(w, h, 4.0);
    let sdr = common::create_sdr_gradient(w, h);
    let mut encoder = ultrahdr_rs::Encoder::new();
    encoder.set_hdr_image(hdr).set_sdr_image(sdr);
    encoder.encode().expect("test encode must succeed")
}

// ---------------------------------------------------------------------------
// Cooperative cancellation (issue #28): a cancelled Stop token must surface
// as a typed Error::Stopped from every decode path.
// ---------------------------------------------------------------------------

/// A pre-cancelled stop token cancels every decode path with Error::Stopped.
#[test]
fn stop_cancels_all_decode_paths() {
    let encoded = encode_small_ultrahdr(64, 64);
    let decoder = Decoder::new(&encoded).unwrap();

    let err = decoder
        .decode_sdr_with_stop(common::AlwaysStop)
        .expect_err("cancelled SDR decode must error");
    assert!(
        matches!(err.error(), Error::Stopped(_)),
        "expected Stopped, got: {err:?}"
    );

    let err = decoder
        .decode_gainmap_with_stop(common::AlwaysStop)
        .expect_err("cancelled gain-map decode must error");
    assert!(
        matches!(err.error(), Error::Stopped(_)),
        "expected Stopped on gainmap, got: {err:?}"
    );

    let err = decoder
        .decode_hdr_with_stop(4.0, common::AlwaysStop)
        .expect_err("cancelled HDR decode must error");
    assert!(
        matches!(err.error(), Error::Stopped(_)),
        "expected Stopped on HDR, got: {err:?}"
    );
}

/// `*_with_stop(Unstoppable)` must decode byte-identically to the plain
/// methods.
#[test]
fn stop_unstoppable_matches_plain_decode() {
    let encoded = encode_small_ultrahdr(48, 32);
    let decoder = Decoder::new(&encoded).unwrap();

    let a = decoder.decode_sdr().unwrap();
    let b = decoder
        .decode_sdr_with_stop(ultrahdr_rs::Unstoppable)
        .unwrap();
    assert_eq!(
        a.as_slice().as_strided_bytes(),
        b.as_slice().as_strided_bytes()
    );

    let h1 = decoder.decode_hdr(4.0).unwrap();
    let h2 = decoder
        .decode_hdr_with_stop(4.0, ultrahdr_rs::Unstoppable)
        .unwrap();
    assert_eq!(
        h1.as_slice().as_strided_bytes(),
        h2.as_slice().as_strided_bytes()
    );
}
