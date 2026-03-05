//! ISO 21496-1 binary metadata format for gain maps.
//!
//! This is the standardized binary format for gain map metadata,
//! as an alternative to XMP.

use alloc::format;
use alloc::vec::Vec;
use core::mem;

use crate::types::{Error, Fraction, GainMapMetadata, Result};

/// ISO 21496-1 metadata version.
pub const ISO_VERSION: u8 = 0;

/// Flags byte layout:
/// - Bit 0: Multi-channel gain map (0 = single channel)
/// - Bit 1: Use base color space (1 = gain map uses base image color space)
/// - Bit 2: Backward direction (0 = base is SDR, 1 = base is HDR)
/// - Bits 3-7: Reserved
const FLAG_MULTI_CHANNEL: u8 = 0x01;
const FLAG_USE_BASE_CG: u8 = 0x02;
const FLAG_BACKWARD_DIR: u8 = 0x04;

/// Serialize gain map metadata to ISO 21496-1 binary format.
pub fn serialize_iso21496(metadata: &GainMapMetadata) -> Vec<u8> {
    let mut data = Vec::with_capacity(128);

    // Version (1 byte)
    data.push(ISO_VERSION);

    // Flags (1 byte)
    let mut flags = 0u8;
    if !metadata.is_single_channel() {
        flags |= FLAG_MULTI_CHANNEL;
    }
    if metadata.use_base_color_space {
        flags |= FLAG_USE_BASE_CG;
    }
    // Backward direction is false (base is SDR)
    data.push(flags);

    let channels = if flags & FLAG_MULTI_CHANNEL != 0 {
        3
    } else {
        1
    };

    // Base HDR headroom (fraction) - log2(hdr_capacity_min)
    let base_headroom = Fraction::from_f32(metadata.hdr_capacity_min.log2());
    write_fraction(&mut data, base_headroom);

    // Alternate HDR headroom (fraction) - log2(hdr_capacity_max)
    let alt_headroom = Fraction::from_f32(metadata.hdr_capacity_max.log2());
    write_fraction(&mut data, alt_headroom);

    // Per-channel values
    for i in 0..channels {
        // Gain map min (fraction) - log2(min_content_boost)
        let min_val = Fraction::from_f32(metadata.min_content_boost[i].log2());
        write_fraction(&mut data, min_val);

        // Gain map max (fraction) - log2(max_content_boost)
        let max_val = Fraction::from_f32(metadata.max_content_boost[i].log2());
        write_fraction(&mut data, max_val);

        // Gamma (fraction)
        let gamma = Fraction::from_f32(metadata.gamma[i]);
        write_fraction(&mut data, gamma);

        // Base offset (fraction) - offset_sdr
        let base_offset = Fraction::from_f32(metadata.offset_sdr[i]);
        write_fraction(&mut data, base_offset);

        // Alternate offset (fraction) - offset_hdr
        let alt_offset = Fraction::from_f32(metadata.offset_hdr[i]);
        write_fraction(&mut data, alt_offset);
    }

    data
}

/// Deserialize ISO 21496-1 binary metadata.
pub fn deserialize_iso21496(data: &[u8]) -> Result<GainMapMetadata> {
    if data.len() < 2 {
        return Err(Error::InvalidMetadata("ISO metadata too short".into()));
    }

    let mut pos = 0;

    // Version
    let version = data[pos];
    pos += 1;
    if version > ISO_VERSION {
        return Err(Error::InvalidMetadata(format!(
            "Unsupported ISO version: {}",
            version
        )));
    }

    // Flags
    let flags = data[pos];
    pos += 1;
    let multi_channel = flags & FLAG_MULTI_CHANNEL != 0;
    let use_base_cg = flags & FLAG_USE_BASE_CG != 0;
    let backward_dir = flags & FLAG_BACKWARD_DIR != 0;

    let channels = if multi_channel { 3 } else { 1 };

    // We need at least: 2 + 8*2 (headrooms) + channels * 5 * 8 (per-channel fractions)
    let min_size = 2 + 16 + channels * 40;
    if data.len() < min_size {
        return Err(Error::InvalidMetadata("ISO metadata truncated".into()));
    }

    // Base HDR headroom
    let (base_headroom, new_pos) = read_fraction(data, pos)?;
    pos = new_pos;
    let hdr_capacity_min = 2.0f32.powf(base_headroom.to_f32());

    // Alternate HDR headroom
    let (alt_headroom, new_pos) = read_fraction(data, pos)?;
    pos = new_pos;
    let hdr_capacity_max = 2.0f32.powf(alt_headroom.to_f32());

    let mut metadata = GainMapMetadata {
        hdr_capacity_min,
        hdr_capacity_max,
        use_base_color_space: use_base_cg,
        ..Default::default()
    };

    // Per-channel values
    for i in 0..channels {
        let idx = if multi_channel { i } else { 0 };

        // Gain map min
        let (min_frac, new_pos) = read_fraction(data, pos)?;
        pos = new_pos;
        let min_val = 2.0f32.powf(min_frac.to_f32());

        // Gain map max
        let (max_frac, new_pos) = read_fraction(data, pos)?;
        pos = new_pos;
        let max_val = 2.0f32.powf(max_frac.to_f32());

        // Gamma
        let (gamma_frac, new_pos) = read_fraction(data, pos)?;
        pos = new_pos;

        // Base offset
        let (base_offset_frac, new_pos) = read_fraction(data, pos)?;
        pos = new_pos;

        // Alternate offset
        let (alt_offset_frac, new_pos) = read_fraction(data, pos)?;
        pos = new_pos;

        if multi_channel {
            metadata.min_content_boost[idx] = min_val;
            metadata.max_content_boost[idx] = max_val;
            metadata.gamma[idx] = gamma_frac.to_f32();
            metadata.offset_sdr[idx] = base_offset_frac.to_f32();
            metadata.offset_hdr[idx] = alt_offset_frac.to_f32();
        } else {
            // Single channel - apply to all
            metadata.min_content_boost = [min_val; 3];
            metadata.max_content_boost = [max_val; 3];
            metadata.gamma = [gamma_frac.to_f32(); 3];
            metadata.offset_sdr = [base_offset_frac.to_f32(); 3];
            metadata.offset_hdr = [alt_offset_frac.to_f32(); 3];
        }
    }

    // Handle backward direction (swap SDR/HDR interpretation)
    if backward_dir {
        mem::swap(&mut metadata.offset_sdr, &mut metadata.offset_hdr);
    }

    Ok(metadata)
}

/// Write a fraction to the buffer (8 bytes: 4 for numerator, 4 for denominator).
fn write_fraction(buf: &mut Vec<u8>, frac: Fraction) {
    buf.extend_from_slice(&frac.numerator.to_be_bytes());
    buf.extend_from_slice(&frac.denominator.to_be_bytes());
}

/// Read a fraction from the buffer.
fn read_fraction(data: &[u8], pos: usize) -> Result<(Fraction, usize)> {
    if pos + 8 > data.len() {
        return Err(Error::InvalidMetadata("Unexpected end of ISO data".into()));
    }

    let numerator = i32::from_be_bytes([data[pos], data[pos + 1], data[pos + 2], data[pos + 3]]);
    let denominator =
        u32::from_be_bytes([data[pos + 4], data[pos + 5], data[pos + 6], data[pos + 7]]);

    Ok((Fraction::new(numerator, denominator), pos + 8))
}

/// Create APP2 marker with ISO 21496-1 data.
pub fn create_iso_app2_marker(iso_data: &[u8]) -> Vec<u8> {
    // ISO 21496-1 uses a specific APP2 marker format
    let namespace = b"urn:iso:std:iso:ts:21496:-1\0";

    let total_length = 2 + namespace.len() + iso_data.len();

    let mut marker = Vec::with_capacity(2 + total_length);
    marker.push(0xFF);
    marker.push(0xE2); // APP2
    marker.push(((total_length >> 8) & 0xFF) as u8);
    marker.push((total_length & 0xFF) as u8);
    marker.extend_from_slice(namespace);
    marker.extend_from_slice(iso_data);

    marker
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_serialize_deserialize_single_channel() {
        let original = GainMapMetadata {
            min_content_boost: [1.0; 3],
            max_content_boost: [4.0; 3],
            gamma: [1.0; 3],
            offset_sdr: [0.015625; 3],
            offset_hdr: [0.015625; 3],
            hdr_capacity_min: 1.0,
            hdr_capacity_max: 4.0,
            use_base_color_space: true,
        };

        let serialized = serialize_iso21496(&original);
        let parsed = deserialize_iso21496(&serialized).unwrap();

        // Check values match (with tolerance for fraction conversion)
        assert!((parsed.max_content_boost[0] - 4.0).abs() < 0.01);
        assert!((parsed.hdr_capacity_max - 4.0).abs() < 0.01);
        assert!((parsed.gamma[0] - 1.0).abs() < 0.01);
        assert!(parsed.use_base_color_space);
    }

    #[test]
    fn test_fraction_roundtrip() {
        let values = [0.0, 0.5, 1.0, 2.0, -1.0, 0.015625];

        for &v in &values {
            let frac = Fraction::from_f32(v);
            let back = frac.to_f32();
            assert!(
                (v - back).abs() < 0.0001,
                "Fraction roundtrip failed for {}: got {}",
                v,
                back
            );
        }
    }

    /// Multi-channel roundtrip with C++ libultrahdr reference values.
    ///
    /// Per-channel: max_content_boost=[100.5, 101.5, 102.5],
    /// min_content_boost=[1.5, 1.6, 1.7], gamma=[1.0, 1.01, 1.02],
    /// offset_sdr/hdr=[0.0625, 0.0875, 0.1125],
    /// hdr_capacity_max=10000.0/203.0, use_base_cg=false
    #[test]
    fn test_iso21496_multichannel_cpp_reference() {
        let original = GainMapMetadata {
            max_content_boost: [100.5, 101.5, 102.5],
            min_content_boost: [1.5, 1.6, 1.7],
            gamma: [1.0, 1.01, 1.02],
            offset_sdr: [0.0625, 0.0875, 0.1125],
            offset_hdr: [0.0625, 0.0875, 0.1125],
            hdr_capacity_min: 1.0,
            hdr_capacity_max: 10000.0 / 203.0, // ~49.26
            use_base_color_space: false,
        };

        let serialized = serialize_iso21496(&original);
        let parsed = deserialize_iso21496(&serialized).unwrap();

        // Verify flags: multi-channel set, use_base_cg NOT set
        assert_eq!(
            serialized[1] & 0x01,
            0x01,
            "MULTI_CHANNEL flag should be set"
        );
        assert_eq!(
            serialized[1] & 0x02,
            0x00,
            "USE_BASE_CG flag should NOT be set"
        );
        assert!(!parsed.use_base_color_space);

        // Per-channel values must roundtrip within fraction precision
        let tol = 0.05;
        for i in 0..3 {
            assert!(
                (parsed.max_content_boost[i] - original.max_content_boost[i]).abs()
                    / original.max_content_boost[i]
                    < tol,
                "max_content_boost[{}]: {} vs {}",
                i,
                parsed.max_content_boost[i],
                original.max_content_boost[i]
            );
            assert!(
                (parsed.min_content_boost[i] - original.min_content_boost[i]).abs()
                    / original.min_content_boost[i]
                    < tol,
                "min_content_boost[{}]: {} vs {}",
                i,
                parsed.min_content_boost[i],
                original.min_content_boost[i]
            );
            assert!(
                (parsed.gamma[i] - original.gamma[i]).abs() < 0.01,
                "gamma[{}]: {} vs {}",
                i,
                parsed.gamma[i],
                original.gamma[i]
            );
            assert!(
                (parsed.offset_sdr[i] - original.offset_sdr[i]).abs() < 0.001,
                "offset_sdr[{}]: {} vs {}",
                i,
                parsed.offset_sdr[i],
                original.offset_sdr[i]
            );
            assert!(
                (parsed.offset_hdr[i] - original.offset_hdr[i]).abs() < 0.001,
                "offset_hdr[{}]: {} vs {}",
                i,
                parsed.offset_hdr[i],
                original.offset_hdr[i]
            );
        }

        // HDR capacity roundtrip
        assert!(
            (parsed.hdr_capacity_max - original.hdr_capacity_max).abs() / original.hdr_capacity_max
                < tol,
            "hdr_capacity_max: {} vs {}",
            parsed.hdr_capacity_max,
            original.hdr_capacity_max
        );

        // Verify channels are distinct (not collapsed to single)
        assert_ne!(parsed.max_content_boost[0], parsed.max_content_boost[1]);
        assert_ne!(parsed.max_content_boost[1], parsed.max_content_boost[2]);
        assert_ne!(parsed.min_content_boost[0], parsed.min_content_boost[1]);
    }

    /// Multi-channel with negative offsets, C++ libultrahdr reference.
    ///
    /// offset_sdr/hdr=[-0.0625, -0.0615, -0.0605],
    /// hdr_capacity_max=1000.0/203.0, use_base_cg=true
    #[test]
    fn test_iso21496_negative_offsets_cpp_reference() {
        let original = GainMapMetadata {
            max_content_boost: [10.0, 11.0, 12.0],
            min_content_boost: [0.5, 0.6, 0.7],
            gamma: [1.0, 1.1, 1.2],
            offset_sdr: [-0.0625, -0.0615, -0.0605],
            offset_hdr: [-0.0625, -0.0615, -0.0605],
            hdr_capacity_min: 1.0,
            hdr_capacity_max: 1000.0 / 203.0, // ~4.926
            use_base_color_space: true,
        };

        let serialized = serialize_iso21496(&original);
        let parsed = deserialize_iso21496(&serialized).unwrap();

        // Verify flags: multi-channel set, use_base_cg set
        assert_eq!(
            serialized[1] & 0x01,
            0x01,
            "MULTI_CHANNEL flag should be set"
        );
        assert_eq!(serialized[1] & 0x02, 0x02, "USE_BASE_CG flag should be set");
        assert!(parsed.use_base_color_space);

        // Negative offsets must survive roundtrip
        let tol = 0.001;
        for i in 0..3 {
            assert!(
                (parsed.offset_sdr[i] - original.offset_sdr[i]).abs() < tol,
                "offset_sdr[{}]: {} vs {}",
                i,
                parsed.offset_sdr[i],
                original.offset_sdr[i]
            );
            assert!(
                (parsed.offset_hdr[i] - original.offset_hdr[i]).abs() < tol,
                "offset_hdr[{}]: {} vs {}",
                i,
                parsed.offset_hdr[i],
                original.offset_hdr[i]
            );
        }

        // Verify all per-channel values roundtrip
        let rel_tol = 0.05;
        for i in 0..3 {
            assert!(
                (parsed.max_content_boost[i] - original.max_content_boost[i]).abs()
                    / original.max_content_boost[i]
                    < rel_tol,
                "max_content_boost[{}]: {} vs {}",
                i,
                parsed.max_content_boost[i],
                original.max_content_boost[i]
            );
            assert!(
                (parsed.gamma[i] - original.gamma[i]).abs() < 0.01,
                "gamma[{}]: {} vs {}",
                i,
                parsed.gamma[i],
                original.gamma[i]
            );
        }
    }

    /// Data too short: 0 bytes and 1 byte should both fail.
    #[test]
    fn test_deserialize_truncated_data() {
        let result_empty = deserialize_iso21496(&[]);
        assert!(result_empty.is_err(), "0 bytes should fail");

        let result_one = deserialize_iso21496(&[0x00]);
        assert!(result_one.is_err(), "1 byte should fail");
    }

    /// Version greater than ISO_VERSION should be rejected.
    #[test]
    fn test_deserialize_version_mismatch() {
        // Build a valid serialized blob, then set version to ISO_VERSION + 1
        let metadata = GainMapMetadata::new();
        let mut serialized = serialize_iso21496(&metadata);
        serialized[0] = ISO_VERSION + 1;

        let result = deserialize_iso21496(&serialized);
        assert!(result.is_err(), "version > ISO_VERSION should be rejected");
        let msg = result.unwrap_err().to_string();
        assert!(
            msg.contains("version") || msg.contains("Version"),
            "error should mention version: {}",
            msg
        );
    }

    /// Valid header (version + flags) but data truncated mid-fraction.
    #[test]
    fn test_deserialize_truncated_fractions() {
        // 2 bytes: valid version + flags for single-channel
        // Single-channel needs 2 + 16 + 1*40 = 58 bytes total.
        // Provide only the header + a few extra bytes (not enough for fractions).
        let mut data = vec![ISO_VERSION, 0x00]; // version=0, flags=0 (single channel)
        // Add just 4 bytes - not enough for even one complete fraction (needs 8)
        data.extend_from_slice(&[0x00, 0x00, 0x00, 0x01]);

        let result = deserialize_iso21496(&data);
        assert!(
            result.is_err(),
            "truncated fraction data should fail: got {:?}",
            result
        );
    }

    /// Serialize with backward_dir=false (default), manually set the backward
    /// direction flag bit, deserialize, verify offset_sdr and offset_hdr are swapped.
    #[test]
    fn test_backward_direction_flag() {
        let original = GainMapMetadata {
            min_content_boost: [1.0; 3],
            max_content_boost: [4.0; 3],
            gamma: [1.0; 3],
            offset_sdr: [0.1; 3],
            offset_hdr: [0.2; 3],
            hdr_capacity_min: 1.0,
            hdr_capacity_max: 4.0,
            use_base_color_space: true,
        };

        let mut serialized = serialize_iso21496(&original);

        // Verify backward_dir flag is NOT set by default
        assert_eq!(
            serialized[1] & FLAG_BACKWARD_DIR,
            0,
            "backward_dir should not be set by serialize"
        );

        // Manually set the backward direction flag (bit 2)
        serialized[1] |= FLAG_BACKWARD_DIR;

        let parsed = deserialize_iso21496(&serialized).unwrap();

        // With backward_dir set, offset_sdr and offset_hdr should be swapped
        let tol = 0.001;
        for i in 0..3 {
            assert!(
                (parsed.offset_sdr[i] - original.offset_hdr[i]).abs() < tol,
                "offset_sdr[{}] should contain original offset_hdr: {} vs {}",
                i,
                parsed.offset_sdr[i],
                original.offset_hdr[i]
            );
            assert!(
                (parsed.offset_hdr[i] - original.offset_sdr[i]).abs() < tol,
                "offset_hdr[{}] should contain original offset_sdr: {} vs {}",
                i,
                parsed.offset_hdr[i],
                original.offset_sdr[i]
            );
        }
    }

    /// Verify APP2 marker structure: starts with FF E2, correct length,
    /// contains namespace, contains data.
    #[test]
    fn test_create_iso_app2_marker() {
        let iso_data = serialize_iso21496(&GainMapMetadata::new());
        let marker = create_iso_app2_marker(&iso_data);

        // Must start with APP2 marker bytes
        assert_eq!(marker[0], 0xFF, "first byte should be 0xFF");
        assert_eq!(marker[1], 0xE2, "second byte should be 0xE2 (APP2)");

        // Length field (big-endian u16 at bytes 2..4)
        let namespace = b"urn:iso:std:iso:ts:21496:-1\0";
        let expected_length = 2 + namespace.len() + iso_data.len();
        let actual_length = ((marker[2] as usize) << 8) | (marker[3] as usize);
        assert_eq!(
            actual_length, expected_length,
            "length field mismatch: expected {}, got {}",
            expected_length, actual_length
        );

        // Total marker size = 2 (FF E2) + length field content
        assert_eq!(marker.len(), 2 + expected_length);

        // Namespace is present after the length field
        let ns_start = 4;
        let ns_end = ns_start + namespace.len();
        assert_eq!(
            &marker[ns_start..ns_end],
            namespace,
            "namespace not found at expected position"
        );

        // ISO data follows the namespace
        assert_eq!(
            &marker[ns_end..],
            &iso_data,
            "ISO data not found after namespace"
        );
    }

    /// Create marker and verify it contains the ISO namespace string.
    #[test]
    fn test_create_iso_app2_marker_roundtrip() {
        let iso_data = serialize_iso21496(&GainMapMetadata::new());
        let marker = create_iso_app2_marker(&iso_data);

        let namespace_str = b"urn:iso:std:iso:ts:21496:-1\0";

        // Verify the namespace string appears in the marker
        let found = marker
            .windows(namespace_str.len())
            .any(|w| w == namespace_str);
        assert!(
            found,
            "marker should contain the ISO namespace string \"urn:iso:std:iso:ts:21496:-1\\0\""
        );

        // Extract the ISO payload from the marker (skip 2 marker bytes + 2 length bytes + namespace)
        let payload_start = 4 + namespace_str.len();
        let extracted_data = &marker[payload_start..];
        assert_eq!(
            extracted_data, &iso_data,
            "extracted payload should match original ISO data"
        );

        // Deserialize the extracted payload to verify full roundtrip
        let parsed = deserialize_iso21496(extracted_data).unwrap();
        let original = GainMapMetadata::new();
        assert!(
            (parsed.hdr_capacity_min - original.hdr_capacity_min).abs() < 0.01,
            "roundtrip hdr_capacity_min mismatch"
        );
        assert!(parsed.use_base_color_space);
    }

    /// Single-channel metadata should produce identical values in all 3 channels.
    #[test]
    fn test_single_channel_all_same() {
        let original = GainMapMetadata {
            min_content_boost: [2.0; 3],
            max_content_boost: [8.0; 3],
            gamma: [1.5; 3],
            offset_sdr: [0.03; 3],
            offset_hdr: [0.05; 3],
            hdr_capacity_min: 1.0,
            hdr_capacity_max: 8.0,
            use_base_color_space: false,
        };

        // Confirm it serializes as single-channel
        let serialized = serialize_iso21496(&original);
        assert_eq!(
            serialized[1] & FLAG_MULTI_CHANNEL,
            0,
            "should serialize as single channel"
        );

        let parsed = deserialize_iso21496(&serialized).unwrap();

        // All three channels must be identical
        for i in 1..3 {
            assert_eq!(
                parsed.min_content_boost[0], parsed.min_content_boost[i],
                "min_content_boost[0] != min_content_boost[{}]",
                i
            );
            assert_eq!(
                parsed.max_content_boost[0], parsed.max_content_boost[i],
                "max_content_boost[0] != max_content_boost[{}]",
                i
            );
            assert_eq!(parsed.gamma[0], parsed.gamma[i], "gamma[0] != gamma[{}]", i);
            assert_eq!(
                parsed.offset_sdr[0], parsed.offset_sdr[i],
                "offset_sdr[0] != offset_sdr[{}]",
                i
            );
            assert_eq!(
                parsed.offset_hdr[0], parsed.offset_hdr[i],
                "offset_hdr[0] != offset_hdr[{}]",
                i
            );
        }
    }

    /// Fraction::new with denominator 0 should not panic; to_f32 returns 0.0.
    #[test]
    fn test_fraction_zero_denominator() {
        let frac = Fraction::new(42, 0);
        // Must not panic
        let val = frac.to_f32();
        assert_eq!(
            val, 0.0,
            "Fraction with denominator 0 should return 0.0, got {}",
            val
        );

        let frac_neg = Fraction::new(-100, 0);
        let val_neg = frac_neg.to_f32();
        assert_eq!(
            val_neg, 0.0,
            "Fraction with negative numerator and denominator 0 should return 0.0, got {}",
            val_neg
        );
    }

    /// Extreme values: very large max_content_boost, very small gamma,
    /// and hdr_capacity_max of 10000/203.
    #[test]
    fn test_serialize_deserialize_extreme_values() {
        let original = GainMapMetadata {
            min_content_boost: [0.01; 3],
            max_content_boost: [10000.0; 3],
            gamma: [0.01; 3],
            offset_sdr: [0.0; 3],
            offset_hdr: [0.0; 3],
            hdr_capacity_min: 1.0,
            hdr_capacity_max: 10000.0 / 203.0, // ~49.26
            use_base_color_space: true,
        };

        let serialized = serialize_iso21496(&original);
        let parsed = deserialize_iso21496(&serialized).unwrap();

        // Large max_content_boost: use relative tolerance
        let rel_tol = 0.05;
        assert!(
            (parsed.max_content_boost[0] - original.max_content_boost[0]).abs()
                / original.max_content_boost[0]
                < rel_tol,
            "max_content_boost: {} vs {}",
            parsed.max_content_boost[0],
            original.max_content_boost[0]
        );

        // Small min_content_boost
        assert!(
            (parsed.min_content_boost[0] - original.min_content_boost[0]).abs()
                / original.min_content_boost[0]
                < rel_tol,
            "min_content_boost: {} vs {}",
            parsed.min_content_boost[0],
            original.min_content_boost[0]
        );

        // Very small gamma
        assert!(
            (parsed.gamma[0] - original.gamma[0]).abs() < 0.001,
            "gamma: {} vs {}",
            parsed.gamma[0],
            original.gamma[0]
        );

        // hdr_capacity_max = 10000/203
        let expected_cap = 10000.0f32 / 203.0;
        assert!(
            (parsed.hdr_capacity_max - expected_cap).abs() / expected_cap < rel_tol,
            "hdr_capacity_max: {} vs {}",
            parsed.hdr_capacity_max,
            expected_cap
        );

        // Zero offsets should roundtrip exactly (or very close)
        assert!(
            (parsed.offset_sdr[0]).abs() < 0.001,
            "offset_sdr should be ~0.0: {}",
            parsed.offset_sdr[0]
        );
        assert!(
            (parsed.offset_hdr[0]).abs() < 0.001,
            "offset_hdr should be ~0.0: {}",
            parsed.offset_hdr[0]
        );
    }
}
