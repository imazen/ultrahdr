//! ISO 21496-1 binary metadata format for gain maps.
//!
//! This implements the standardized binary format used in AVIF `tmap` item
//! payloads and JXL `jhgm` boxes. The wire format matches the reference
//! implementation in `zenavif-parse::parse_tone_map_image()`.
//!
//! # Wire format (all multi-byte integers are big-endian)
//!
//! ```text
//! version:          u8   (must be 0)
//! minimum_version:  u16  (must be 0)
//! writer_version:   u16  (>= minimum_version)
//! flags:            u8   (bit 7 = multichannel, bit 6 = use_base_colour_space)
//!
//! base_hdr_headroom_n:      u32
//! base_hdr_headroom_d:      u32
//! alternate_hdr_headroom_n: u32
//! alternate_hdr_headroom_d: u32
//!
//! Per channel (1 or 3 times):
//!   gain_map_min_n:      i32
//!   gain_map_min_d:      u32
//!   gain_map_max_n:      i32
//!   gain_map_max_d:      u32
//!   gamma_n:             u32
//!   gamma_d:             u32
//!   base_offset_n:       i32
//!   base_offset_d:       u32
//!   alternate_offset_n:  i32
//!   alternate_offset_d:  u32
//! ```

use alloc::format;
use alloc::vec::Vec;

use crate::types::{Error, Fraction, GainMapMetadata, Result, UnsignedFraction};

/// Current ISO 21496-1 metadata version.
pub const ISO_VERSION: u8 = 0;

/// Flag bit: multichannel gain map (bit 7 of flags byte).
const FLAG_MULTI_CHANNEL: u8 = 0x80;

/// Flag bit: gain map uses base image colour space (bit 6 of flags byte).
const FLAG_USE_BASE_COLOUR_SPACE: u8 = 0x40;

/// Header size: version (1) + minimum_version (2) + writer_version (2) + flags (1).
const HEADER_SIZE: usize = 6;

/// Size of one fraction pair (numerator + denominator = 8 bytes).
const FRACTION_SIZE: usize = 8;

/// Number of fraction pairs in the headroom section (base + alternate).
const HEADROOM_FRACTIONS: usize = 2;

/// Number of fraction pairs per channel (min, max, gamma, base_offset, alt_offset).
const FRACTIONS_PER_CHANNEL: usize = 5;

/// Parse ISO 21496-1 binary gain map metadata.
///
/// This is the format used in AVIF `tmap` item payloads and JXL `jhgm` boxes.
/// The wire format matches `zenavif-parse::parse_tone_map_image()`.
pub fn parse_iso21496(data: &[u8]) -> Result<GainMapMetadata> {
    if data.len() < HEADER_SIZE {
        return Err(Error::IsoParse(format!(
            "data too short: need at least {} bytes, got {}",
            HEADER_SIZE,
            data.len()
        )));
    }

    let mut pos = 0;

    // version (u8) — must be 0
    let version = data[pos];
    pos += 1;
    if version != ISO_VERSION {
        return Err(Error::IsoParse(format!(
            "unsupported version {}, expected {}",
            version, ISO_VERSION
        )));
    }

    // minimum_version (u16 BE) — must be 0
    let minimum_version = read_u16_be(data, pos);
    pos += 2;
    if minimum_version > 0 {
        return Err(Error::IsoParse(format!(
            "unsupported minimum_version {}",
            minimum_version
        )));
    }

    // writer_version (u16 BE) — informational, must be >= minimum_version
    let writer_version = read_u16_be(data, pos);
    pos += 2;
    if writer_version < minimum_version {
        return Err(Error::IsoParse(format!(
            "writer_version {} < minimum_version {}",
            writer_version, minimum_version
        )));
    }

    // flags (u8)
    let flags = data[pos];
    pos += 1;
    let is_multichannel = (flags & FLAG_MULTI_CHANNEL) != 0;
    let use_base_colour_space = (flags & FLAG_USE_BASE_COLOUR_SPACE) != 0;

    let channel_count: usize = if is_multichannel { 3 } else { 1 };

    // Validate remaining data length
    let required = HEADER_SIZE
        + HEADROOM_FRACTIONS * FRACTION_SIZE
        + channel_count * FRACTIONS_PER_CHANNEL * FRACTION_SIZE;
    if data.len() < required {
        return Err(Error::IsoParse(format!(
            "data truncated: need {} bytes for {} channel(s), got {}",
            required,
            channel_count,
            data.len()
        )));
    }

    // base_hdr_headroom (u32/u32)
    let (base_headroom, new_pos) = read_unsigned_fraction(data, pos)?;
    pos = new_pos;

    // alternate_hdr_headroom (u32/u32)
    let (alt_headroom, new_pos) = read_unsigned_fraction(data, pos)?;
    pos = new_pos;

    let hdr_capacity_min = 2.0f32.powf(base_headroom.to_f32());
    let hdr_capacity_max = 2.0f32.powf(alt_headroom.to_f32());

    let mut metadata = GainMapMetadata {
        hdr_capacity_min,
        hdr_capacity_max,
        use_base_color_space: use_base_colour_space,
        ..Default::default()
    };

    // Per-channel values
    for ch in 0..channel_count {
        // gain_map_min (i32/u32)
        let (min_frac, new_pos) = read_signed_fraction(data, pos)?;
        pos = new_pos;
        let min_val = 2.0f32.powf(min_frac.to_f32());

        // gain_map_max (i32/u32)
        let (max_frac, new_pos) = read_signed_fraction(data, pos)?;
        pos = new_pos;
        let max_val = 2.0f32.powf(max_frac.to_f32());

        // gamma (u32/u32)
        let (gamma_frac, new_pos) = read_unsigned_fraction(data, pos)?;
        pos = new_pos;

        // base_offset (i32/u32)
        let (base_offset_frac, new_pos) = read_signed_fraction(data, pos)?;
        pos = new_pos;

        // alternate_offset (i32/u32)
        let (alt_offset_frac, new_pos) = read_signed_fraction(data, pos)?;
        pos = new_pos;

        if is_multichannel {
            metadata.min_content_boost[ch] = min_val;
            metadata.max_content_boost[ch] = max_val;
            metadata.gamma[ch] = gamma_frac.to_f32();
            metadata.offset_sdr[ch] = base_offset_frac.to_f32();
            metadata.offset_hdr[ch] = alt_offset_frac.to_f32();
        } else {
            // Single channel — replicate to all three
            metadata.min_content_boost = [min_val; 3];
            metadata.max_content_boost = [max_val; 3];
            metadata.gamma = [gamma_frac.to_f32(); 3];
            metadata.offset_sdr = [base_offset_frac.to_f32(); 3];
            metadata.offset_hdr = [alt_offset_frac.to_f32(); 3];
        }
    }

    Ok(metadata)
}

/// Serialize gain map metadata to ISO 21496-1 binary format.
///
/// Produces a blob suitable for AVIF `tmap` item payloads or JXL `jhgm` boxes.
/// The wire format matches `zenavif-parse::parse_tone_map_image()`.
pub fn serialize_iso21496(metadata: &GainMapMetadata) -> Vec<u8> {
    let is_multichannel = !metadata.is_single_channel();
    let channel_count: usize = if is_multichannel { 3 } else { 1 };

    let capacity = HEADER_SIZE
        + HEADROOM_FRACTIONS * FRACTION_SIZE
        + channel_count * FRACTIONS_PER_CHANNEL * FRACTION_SIZE;
    let mut buf = Vec::with_capacity(capacity);

    // version (u8)
    buf.push(ISO_VERSION);

    // minimum_version (u16 BE)
    buf.extend_from_slice(&0u16.to_be_bytes());

    // writer_version (u16 BE)
    buf.extend_from_slice(&0u16.to_be_bytes());

    // flags (u8)
    let mut flags = 0u8;
    if is_multichannel {
        flags |= FLAG_MULTI_CHANNEL;
    }
    if metadata.use_base_color_space {
        flags |= FLAG_USE_BASE_COLOUR_SPACE;
    }
    buf.push(flags);

    // base_hdr_headroom (u32/u32) — log2(hdr_capacity_min)
    let base_headroom = UnsignedFraction::from_f32(metadata.hdr_capacity_min.log2());
    write_unsigned_fraction(&mut buf, base_headroom);

    // alternate_hdr_headroom (u32/u32) — log2(hdr_capacity_max)
    let alt_headroom = UnsignedFraction::from_f32(metadata.hdr_capacity_max.log2());
    write_unsigned_fraction(&mut buf, alt_headroom);

    // Per-channel values
    for ch in 0..channel_count {
        // gain_map_min (i32/u32) — log2(min_content_boost)
        let min_val = Fraction::from_f32(metadata.min_content_boost[ch].log2());
        write_signed_fraction(&mut buf, min_val);

        // gain_map_max (i32/u32) — log2(max_content_boost)
        let max_val = Fraction::from_f32(metadata.max_content_boost[ch].log2());
        write_signed_fraction(&mut buf, max_val);

        // gamma (u32/u32)
        let gamma = UnsignedFraction::from_f32(metadata.gamma[ch]);
        write_unsigned_fraction(&mut buf, gamma);

        // base_offset (i32/u32) — offset_sdr
        let base_offset = Fraction::from_f32(metadata.offset_sdr[ch]);
        write_signed_fraction(&mut buf, base_offset);

        // alternate_offset (i32/u32) — offset_hdr
        let alt_offset = Fraction::from_f32(metadata.offset_hdr[ch]);
        write_signed_fraction(&mut buf, alt_offset);
    }

    buf
}

/// Alias for backward compatibility.
pub fn deserialize_iso21496(data: &[u8]) -> Result<GainMapMetadata> {
    parse_iso21496(data)
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

// ============================================================================
// Internal helpers
// ============================================================================

/// Read a u16 big-endian from a byte slice at the given offset.
#[inline]
fn read_u16_be(data: &[u8], pos: usize) -> u16 {
    u16::from_be_bytes([data[pos], data[pos + 1]])
}

/// Read a signed fraction (i32 numerator, u32 denominator) from the buffer.
fn read_signed_fraction(data: &[u8], pos: usize) -> Result<(Fraction, usize)> {
    if pos + FRACTION_SIZE > data.len() {
        return Err(Error::IsoParse(
            "unexpected end of data reading fraction".into(),
        ));
    }

    let numerator = i32::from_be_bytes([data[pos], data[pos + 1], data[pos + 2], data[pos + 3]]);
    let denominator =
        u32::from_be_bytes([data[pos + 4], data[pos + 5], data[pos + 6], data[pos + 7]]);

    if denominator == 0 {
        return Err(Error::IsoParse(
            "zero denominator in signed fraction".into(),
        ));
    }

    Ok((Fraction::new(numerator, denominator), pos + FRACTION_SIZE))
}

/// Read an unsigned fraction (u32 numerator, u32 denominator) from the buffer.
fn read_unsigned_fraction(data: &[u8], pos: usize) -> Result<(UnsignedFraction, usize)> {
    if pos + FRACTION_SIZE > data.len() {
        return Err(Error::IsoParse(
            "unexpected end of data reading unsigned fraction".into(),
        ));
    }

    let numerator = u32::from_be_bytes([data[pos], data[pos + 1], data[pos + 2], data[pos + 3]]);
    let denominator =
        u32::from_be_bytes([data[pos + 4], data[pos + 5], data[pos + 6], data[pos + 7]]);

    if denominator == 0 {
        return Err(Error::IsoParse(
            "zero denominator in unsigned fraction".into(),
        ));
    }

    Ok((
        UnsignedFraction::new(numerator, denominator),
        pos + FRACTION_SIZE,
    ))
}

/// Write a signed fraction (i32 numerator, u32 denominator) to the buffer.
fn write_signed_fraction(buf: &mut Vec<u8>, frac: Fraction) {
    buf.extend_from_slice(&frac.numerator.to_be_bytes());
    buf.extend_from_slice(&frac.denominator.to_be_bytes());
}

/// Write an unsigned fraction (u32 numerator, u32 denominator) to the buffer.
fn write_unsigned_fraction(buf: &mut Vec<u8>, frac: UnsignedFraction) {
    buf.extend_from_slice(&frac.numerator.to_be_bytes());
    buf.extend_from_slice(&frac.denominator.to_be_bytes());
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    /// Byte offset of the flags byte in the serialized format.
    const FLAGS_OFFSET: usize = 5;

    // ========================================================================
    // Roundtrip tests
    // ========================================================================

    #[test]
    fn test_roundtrip_single_channel() {
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
        let parsed = parse_iso21496(&serialized).unwrap();

        assert!((parsed.max_content_boost[0] - 4.0).abs() < 0.01);
        assert!((parsed.min_content_boost[0] - 1.0).abs() < 0.01);
        assert!((parsed.hdr_capacity_max - 4.0).abs() < 0.01);
        assert!((parsed.gamma[0] - 1.0).abs() < 0.01);
        assert!((parsed.offset_sdr[0] - 0.015625).abs() < 0.001);
        assert!(parsed.use_base_color_space);
    }

    #[test]
    fn test_roundtrip_multi_channel() {
        let original = GainMapMetadata {
            max_content_boost: [100.5, 101.5, 102.5],
            min_content_boost: [1.5, 1.6, 1.7],
            gamma: [1.0, 1.01, 1.02],
            offset_sdr: [0.0625, 0.0875, 0.1125],
            offset_hdr: [0.0625, 0.0875, 0.1125],
            hdr_capacity_min: 1.0,
            hdr_capacity_max: 10000.0 / 203.0,
            use_base_color_space: false,
        };

        let serialized = serialize_iso21496(&original);
        let parsed = parse_iso21496(&serialized).unwrap();

        // Verify multichannel flag is set
        assert_ne!(
            serialized[FLAGS_OFFSET] & FLAG_MULTI_CHANNEL,
            0,
            "MULTI_CHANNEL flag should be set"
        );
        assert_eq!(
            serialized[FLAGS_OFFSET] & FLAG_USE_BASE_COLOUR_SPACE,
            0,
            "USE_BASE_COLOUR_SPACE flag should NOT be set"
        );
        assert!(!parsed.use_base_color_space);

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

        // Verify channels are distinct
        assert_ne!(parsed.max_content_boost[0], parsed.max_content_boost[1]);
        assert_ne!(parsed.max_content_boost[1], parsed.max_content_boost[2]);
    }

    #[test]
    fn test_roundtrip_negative_offsets() {
        let original = GainMapMetadata {
            max_content_boost: [10.0, 11.0, 12.0],
            min_content_boost: [0.5, 0.6, 0.7],
            gamma: [1.0, 1.1, 1.2],
            offset_sdr: [-0.0625, -0.0615, -0.0605],
            offset_hdr: [-0.0625, -0.0615, -0.0605],
            hdr_capacity_min: 1.0,
            hdr_capacity_max: 1000.0 / 203.0,
            use_base_color_space: true,
        };

        let serialized = serialize_iso21496(&original);
        let parsed = parse_iso21496(&serialized).unwrap();

        assert!(parsed.use_base_color_space);

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
    }

    // ========================================================================
    // Wire format verification
    // ========================================================================

    #[test]
    fn test_header_layout() {
        let metadata = GainMapMetadata::new();
        let serialized = serialize_iso21496(&metadata);

        // Byte 0: version
        assert_eq!(serialized[0], ISO_VERSION);

        // Bytes 1-2: minimum_version (u16 BE) = 0
        assert_eq!(&serialized[1..3], &[0, 0]);

        // Bytes 3-4: writer_version (u16 BE) = 0
        assert_eq!(&serialized[3..5], &[0, 0]);

        // Byte 5: flags
        let flags = serialized[FLAGS_OFFSET];
        // Default metadata is single-channel, use_base_color_space=true
        assert_eq!(flags & FLAG_MULTI_CHANNEL, 0);
        assert_ne!(flags & FLAG_USE_BASE_COLOUR_SPACE, 0);
    }

    #[test]
    fn test_single_channel_size() {
        let metadata = GainMapMetadata::new();
        let serialized = serialize_iso21496(&metadata);

        // Header (6) + headroom (2*8=16) + 1 channel * 5 fractions * 8 = 62
        assert_eq!(
            serialized.len(),
            HEADER_SIZE
                + HEADROOM_FRACTIONS * FRACTION_SIZE
                + FRACTIONS_PER_CHANNEL * FRACTION_SIZE
        );
        assert_eq!(serialized.len(), 62);
    }

    #[test]
    fn test_multi_channel_size() {
        let metadata = GainMapMetadata {
            max_content_boost: [4.0, 5.0, 6.0],
            min_content_boost: [1.0, 1.5, 2.0],
            gamma: [1.0, 1.1, 1.2],
            offset_sdr: [0.01; 3],
            offset_hdr: [0.01; 3],
            hdr_capacity_min: 1.0,
            hdr_capacity_max: 6.0,
            use_base_color_space: true,
        };
        let serialized = serialize_iso21496(&metadata);

        // Header (6) + headroom (16) + 3 channels * 5 fractions * 8 = 142
        assert_eq!(
            serialized.len(),
            HEADER_SIZE
                + HEADROOM_FRACTIONS * FRACTION_SIZE
                + 3 * FRACTIONS_PER_CHANNEL * FRACTION_SIZE
        );
        assert_eq!(serialized.len(), 142);
    }

    // ========================================================================
    // Known-value test: hand-crafted binary blob
    // ========================================================================

    #[test]
    fn test_parse_known_blob() {
        // Single-channel blob with known values:
        // version=0, min_ver=0, writer_ver=0,
        // flags=0x40 (use_base_colour_space, single channel)
        // base_hdr_headroom = 0/1 (log2(1.0) = 0 → 2^0 = 1.0)
        // alt_hdr_headroom  = 2/1 (log2(4.0) = 2 → 2^2 = 4.0)
        // gain_map_min = 0/1 (log2(1.0) = 0 → 2^0 = 1.0)
        // gain_map_max = 2/1 (log2(4.0) = 2 → 2^2 = 4.0)
        // gamma = 1/1 (1.0)
        // base_offset = 1/64 (0.015625)
        // alt_offset  = 1/64 (0.015625)
        #[rustfmt::skip]
        let blob: Vec<u8> = [
            // Header
            0x00,                   // version = 0
            0x00, 0x00,             // minimum_version = 0
            0x00, 0x00,             // writer_version = 0
            0x40,                   // flags = 0x40 (use_base_colour_space)
            // base_hdr_headroom = 0/1
            0x00, 0x00, 0x00, 0x00, // numerator = 0
            0x00, 0x00, 0x00, 0x01, // denominator = 1
            // alt_hdr_headroom = 2/1
            0x00, 0x00, 0x00, 0x02, // numerator = 2
            0x00, 0x00, 0x00, 0x01, // denominator = 1
            // gain_map_min = 0/1
            0x00, 0x00, 0x00, 0x00, // numerator = 0 (i32)
            0x00, 0x00, 0x00, 0x01, // denominator = 1
            // gain_map_max = 2/1
            0x00, 0x00, 0x00, 0x02, // numerator = 2 (i32)
            0x00, 0x00, 0x00, 0x01, // denominator = 1
            // gamma = 1/1
            0x00, 0x00, 0x00, 0x01, // numerator = 1
            0x00, 0x00, 0x00, 0x01, // denominator = 1
            // base_offset = 1/64
            0x00, 0x00, 0x00, 0x01, // numerator = 1 (i32)
            0x00, 0x00, 0x00, 0x40, // denominator = 64
            // alt_offset = 1/64
            0x00, 0x00, 0x00, 0x01, // numerator = 1 (i32)
            0x00, 0x00, 0x00, 0x40, // denominator = 64
        ].to_vec();

        let parsed = parse_iso21496(&blob).unwrap();

        assert_eq!(parsed.hdr_capacity_min, 1.0); // 2^0
        assert!((parsed.hdr_capacity_max - 4.0).abs() < 0.001); // 2^2
        assert_eq!(parsed.min_content_boost, [1.0; 3]); // 2^0
        assert!((parsed.max_content_boost[0] - 4.0).abs() < 0.001); // 2^2
        assert_eq!(parsed.gamma, [1.0; 3]);
        assert_eq!(parsed.offset_sdr, [0.015625; 3]); // 1/64
        assert_eq!(parsed.offset_hdr, [0.015625; 3]); // 1/64
        assert!(parsed.use_base_color_space);
    }

    // ========================================================================
    // Single-channel replication
    // ========================================================================

    #[test]
    fn test_single_channel_replicates_to_all() {
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

        let serialized = serialize_iso21496(&original);
        assert_eq!(
            serialized[FLAGS_OFFSET] & FLAG_MULTI_CHANNEL,
            0,
            "should serialize as single channel"
        );

        let parsed = parse_iso21496(&serialized).unwrap();

        for i in 1..3 {
            assert_eq!(
                parsed.min_content_boost[0], parsed.min_content_boost[i],
                "min_content_boost[0] != [{}]",
                i
            );
            assert_eq!(
                parsed.max_content_boost[0], parsed.max_content_boost[i],
                "max_content_boost[0] != [{}]",
                i
            );
            assert_eq!(parsed.gamma[0], parsed.gamma[i], "gamma[0] != [{}]", i);
            assert_eq!(
                parsed.offset_sdr[0], parsed.offset_sdr[i],
                "offset_sdr[0] != [{}]",
                i
            );
            assert_eq!(
                parsed.offset_hdr[0], parsed.offset_hdr[i],
                "offset_hdr[0] != [{}]",
                i
            );
        }
    }

    // ========================================================================
    // Edge cases
    // ========================================================================

    #[test]
    fn test_zero_headroom() {
        // hdr_capacity_min = 1.0 → log2 = 0.0, hdr_capacity_max = 1.0 → log2 = 0.0
        let original = GainMapMetadata {
            min_content_boost: [1.0; 3],
            max_content_boost: [1.0; 3],
            gamma: [1.0; 3],
            offset_sdr: [0.0; 3],
            offset_hdr: [0.0; 3],
            hdr_capacity_min: 1.0,
            hdr_capacity_max: 1.0,
            use_base_color_space: true,
        };

        let serialized = serialize_iso21496(&original);
        let parsed = parse_iso21496(&serialized).unwrap();

        assert!((parsed.hdr_capacity_min - 1.0).abs() < 0.001);
        assert!((parsed.hdr_capacity_max - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_gamma_one() {
        let original = GainMapMetadata {
            gamma: [1.0; 3],
            ..GainMapMetadata::new()
        };

        let serialized = serialize_iso21496(&original);
        let parsed = parse_iso21496(&serialized).unwrap();

        assert!((parsed.gamma[0] - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_extreme_boost_values() {
        let original = GainMapMetadata {
            min_content_boost: [0.01; 3],
            max_content_boost: [10000.0; 3],
            gamma: [0.01; 3],
            offset_sdr: [0.0; 3],
            offset_hdr: [0.0; 3],
            hdr_capacity_min: 1.0,
            hdr_capacity_max: 10000.0 / 203.0,
            use_base_color_space: true,
        };

        let serialized = serialize_iso21496(&original);
        let parsed = parse_iso21496(&serialized).unwrap();

        let rel_tol = 0.05;
        assert!(
            (parsed.max_content_boost[0] - 10000.0).abs() / 10000.0 < rel_tol,
            "max_content_boost: {} vs 10000.0",
            parsed.max_content_boost[0]
        );
        assert!(
            (parsed.min_content_boost[0] - 0.01).abs() / 0.01 < rel_tol,
            "min_content_boost: {} vs 0.01",
            parsed.min_content_boost[0]
        );
        assert!(
            (parsed.gamma[0] - 0.01).abs() < 0.001,
            "gamma: {} vs 0.01",
            parsed.gamma[0]
        );
    }

    // ========================================================================
    // Error cases
    // ========================================================================

    #[test]
    fn test_empty_data() {
        assert!(parse_iso21496(&[]).is_err());
    }

    #[test]
    fn test_too_short() {
        // Just the version byte — not enough for the full header
        assert!(parse_iso21496(&[0x00]).is_err());
        assert!(parse_iso21496(&[0x00, 0x00]).is_err());
        assert!(parse_iso21496(&[0x00, 0x00, 0x00, 0x00, 0x00]).is_err());
    }

    #[test]
    fn test_invalid_version() {
        let mut blob = vec![0u8; 62];
        blob[0] = 1; // version 1 — unsupported
        let result = parse_iso21496(&blob);
        assert!(result.is_err());
        let msg = result.unwrap_err().to_string();
        assert!(
            msg.contains("version"),
            "error should mention version: {}",
            msg
        );
    }

    #[test]
    fn test_truncated_fractions() {
        // Valid 6-byte header, but not enough data for headroom fractions
        let data = vec![
            0x00, // version
            0x00, 0x00, // minimum_version
            0x00, 0x00, // writer_version
            0x00, // flags (single channel)
            0x00, 0x00, 0x00, 0x01, // partial headroom
        ];
        assert!(parse_iso21496(&data).is_err());
    }

    #[test]
    fn test_zero_denominator_in_headroom() {
        // Build a valid-length blob but with zero denominator in base_hdr_headroom
        let mut data = vec![0u8; 62];
        data[0] = 0; // version
        // min_ver, writer_ver, flags all 0
        // base_hdr_headroom: numerator=0, denominator=0
        // denominator bytes at offset 10..14 are already 0
        let result = parse_iso21496(&data);
        assert!(result.is_err());
        let msg = result.unwrap_err().to_string();
        assert!(
            msg.contains("denominator"),
            "error should mention denominator: {}",
            msg
        );
    }

    #[test]
    fn test_zero_denominator_in_channel_fraction() {
        // Build a blob with valid headroom but zero denominator in gain_map_min_d
        let metadata = GainMapMetadata::new();
        let mut data = serialize_iso21496(&metadata);
        // gain_map_min_d is at offset: 6 (header) + 16 (headroom) + 4 (min_n) = 26..30
        data[26] = 0;
        data[27] = 0;
        data[28] = 0;
        data[29] = 0;
        let result = parse_iso21496(&data);
        assert!(result.is_err());
        let msg = result.unwrap_err().to_string();
        assert!(
            msg.contains("denominator"),
            "error should mention denominator: {}",
            msg
        );
    }

    // ========================================================================
    // APP2 marker tests
    // ========================================================================

    #[test]
    fn test_app2_marker_structure() {
        let iso_data = serialize_iso21496(&GainMapMetadata::new());
        let marker = create_iso_app2_marker(&iso_data);

        assert_eq!(marker[0], 0xFF);
        assert_eq!(marker[1], 0xE2); // APP2

        let namespace = b"urn:iso:std:iso:ts:21496:-1\0";
        let expected_length = 2 + namespace.len() + iso_data.len();
        let actual_length = ((marker[2] as usize) << 8) | (marker[3] as usize);
        assert_eq!(actual_length, expected_length);

        assert_eq!(marker.len(), 2 + expected_length);

        let ns_start = 4;
        let ns_end = ns_start + namespace.len();
        assert_eq!(&marker[ns_start..ns_end], namespace);
        assert_eq!(&marker[ns_end..], &iso_data);
    }

    #[test]
    fn test_app2_marker_roundtrip() {
        let iso_data = serialize_iso21496(&GainMapMetadata::new());
        let marker = create_iso_app2_marker(&iso_data);

        let namespace = b"urn:iso:std:iso:ts:21496:-1\0";
        let payload_start = 4 + namespace.len();
        let extracted = &marker[payload_start..];
        assert_eq!(extracted, &iso_data);

        let parsed = parse_iso21496(extracted).unwrap();
        let original = GainMapMetadata::new();
        assert!((parsed.hdr_capacity_min - original.hdr_capacity_min).abs() < 0.01);
        assert!(parsed.use_base_color_space);
    }

    // ========================================================================
    // Fraction tests
    // ========================================================================

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

    #[test]
    fn test_unsigned_fraction_roundtrip() {
        let values = [0.0, 0.5, 1.0, 2.0, 0.015625, 49.26];
        for &v in &values {
            let frac = UnsignedFraction::from_f32(v);
            let back = frac.to_f32();
            assert!(
                (v - back).abs() < 0.001,
                "UnsignedFraction roundtrip failed for {}: got {}",
                v,
                back
            );
        }
    }

    #[test]
    fn test_unsigned_fraction_clamps_negative() {
        let frac = UnsignedFraction::from_f32(-5.0);
        assert_eq!(frac.numerator, 0);
    }

    #[test]
    fn test_unsigned_fraction_zero_denominator() {
        let frac = UnsignedFraction::new(42, 0);
        assert_eq!(frac.to_f32(), 0.0);
    }
}
