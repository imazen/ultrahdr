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
/// The `format` parameter selects the wire format variant:
/// - [`crate::Iso21496Format::AvifTmap`]: expects `version(u8)` prefix (AVIF `tmap` item payload)
/// - [`crate::Iso21496Format::JpegApp2`]: no version prefix (JPEG APP2, JXL `jhgm`)
pub fn parse_iso21496(data: &[u8], format: crate::Iso21496Format) -> Result<GainMapMetadata> {
    match format {
        crate::Iso21496Format::AvifTmap => parse_iso21496_avif(data),
        crate::Iso21496Format::JpegApp2 => parse_iso21496_jpeg(data),
    }
}

/// Serialize gain map metadata to ISO 21496-1 binary format.
///
/// The `format` parameter selects the wire format variant:
/// - [`crate::Iso21496Format::AvifTmap`]: includes `version(u8)` prefix
/// - [`crate::Iso21496Format::JpegApp2`]: no version prefix (also correct for JXL `jhgm`)
pub fn serialize_iso21496(metadata: &GainMapMetadata, format: crate::Iso21496Format) -> Vec<u8> {
    match format {
        crate::Iso21496Format::AvifTmap => serialize_iso21496_avif(metadata),
        crate::Iso21496Format::JpegApp2 => serialize_iso21496_jpeg(metadata),
    }
}

/// Parse ISO 21496-1 from AVIF `tmap` item payload (with version byte prefix).
fn parse_iso21496_avif(data: &[u8]) -> Result<GainMapMetadata> {
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

    read_payload(data, pos, channel_count, use_base_colour_space)
}

/// Serialize gain map metadata with AVIF `tmap` version byte prefix.
fn serialize_iso21496_avif(metadata: &GainMapMetadata) -> Vec<u8> {
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

    write_payload(&mut buf, metadata, channel_count);
    buf
}

/// JPEG APP2 header size: minimum_version (2) + writer_version (2) + flags (1).
///
/// The JPEG APP2 variant omits the version byte that AVIF/JXL box formats include,
/// because the APP2 URN namespace already identifies the format.
const JPEG_HEADER_SIZE: usize = 5;

/// Serialize gain map metadata without version byte prefix (JPEG APP2 / JXL jhgm).
fn serialize_iso21496_jpeg(metadata: &GainMapMetadata) -> Vec<u8> {
    let is_multichannel = !metadata.is_single_channel();
    let channel_count: usize = if is_multichannel { 3 } else { 1 };

    let capacity = JPEG_HEADER_SIZE
        + HEADROOM_FRACTIONS * FRACTION_SIZE
        + channel_count * FRACTIONS_PER_CHANNEL * FRACTION_SIZE;
    let mut buf = Vec::with_capacity(capacity);

    // No version byte — JPEG APP2 URN identifies the format.

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

    write_payload(&mut buf, metadata, channel_count);
    buf
}

/// Parse ISO 21496-1 binary gain map metadata from JPEG APP2 payload.
///
/// This is the wire format used by libultrahdr in JPEG APP2 markers.
/// Unlike [`parse_iso21496`], this format has no version byte prefix —
/// Parse gain map metadata without version byte prefix (JPEG APP2 / JXL jhgm).
fn parse_iso21496_jpeg(data: &[u8]) -> Result<GainMapMetadata> {
    if data.len() < JPEG_HEADER_SIZE {
        return Err(Error::IsoParse(format!(
            "JPEG ISO data too short: need at least {} bytes, got {}",
            JPEG_HEADER_SIZE,
            data.len()
        )));
    }

    let mut pos = 0;

    // minimum_version (u16 BE) — must be 0
    let minimum_version = read_u16_be(data, pos);
    pos += 2;
    if minimum_version > 0 {
        return Err(Error::IsoParse(format!(
            "unsupported minimum_version {}",
            minimum_version
        )));
    }

    // writer_version (u16 BE)
    let _writer_version = read_u16_be(data, pos);
    pos += 2;

    // flags (u8)
    let flags = data[pos];
    pos += 1;
    let is_multichannel = (flags & FLAG_MULTI_CHANNEL) != 0;
    let use_base_colour_space = (flags & FLAG_USE_BASE_COLOUR_SPACE) != 0;

    let channel_count: usize = if is_multichannel { 3 } else { 1 };

    let required = JPEG_HEADER_SIZE
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

    read_payload(data, pos, channel_count, use_base_colour_space)
}

/// The two ISO 21496-1 APP2 markers needed for a canonical Ultra HDR JPEG.
///
/// Returned by [`create_jpeg_iso_markers`].
pub struct JpegIsoMarkers {
    /// APP2 marker for the **primary** JPEG codestream.
    ///
    /// This is a 4-byte version-only block (`min_version=0, writer_version=0`)
    /// that signals ISO 21496-1 awareness. It does NOT carry gain map parameters.
    pub primary: Vec<u8>,

    /// APP2 marker for the **gain map** (secondary) JPEG codestream.
    ///
    /// This carries the full serialized gain map metadata (headroom, per-channel
    /// min/max/gamma/offsets) in canonical continued-fraction encoding.
    pub gain_map: Vec<u8>,
}

/// Create both ISO 21496-1 APP2 markers for a canonical Ultra HDR JPEG.
///
/// A spec-compliant Ultra HDR JPEG needs two ISO APP2 markers:
/// 1. A version-only block in the **primary** JPEG (signals ISO 21496-1 support)
/// 2. The full gain map metadata in the **secondary** (gain map) JPEG
///
/// This function produces both in one call so callers can't forget either one.
///
/// # Example
///
/// ```
/// use ultrahdr_core::{GainMapMetadata, metadata::iso21496::create_jpeg_iso_markers};
///
/// let metadata = GainMapMetadata::new();
/// let markers = create_jpeg_iso_markers(&metadata);
///
/// // Insert markers.primary into the primary JPEG after SOI
/// // Insert markers.gain_map into the gain map JPEG after SOI
/// ```
pub fn create_jpeg_iso_markers(metadata: &crate::GainMapMetadata) -> JpegIsoMarkers {
    let iso_payload = serialize_iso21496_jpeg(metadata);
    JpegIsoMarkers {
        primary: create_version_only_iso_app2(),
        gain_map: create_iso_app2_marker(&iso_payload),
    }
}

/// Create the version-only ISO 21496-1 APP2 marker for the primary JPEG.
///
/// Canonical Ultra HDR JPEGs include a 4-byte version-only ISO APP2 block
/// (`min_version=0, writer_version=0`) in the primary JPEG codestream.
/// This signals ISO 21496-1 awareness without carrying gain map parameters
/// (those live in the secondary/gain-map JPEG's APP2).
///
/// Prefer [`create_jpeg_iso_markers`] which produces both the primary and
/// gain map markers in one call.
pub fn create_version_only_iso_app2() -> Vec<u8> {
    let version_payload: &[u8] = &[0x00, 0x00, 0x00, 0x00]; // min_version=0, writer_version=0
    create_iso_app2_marker(version_payload)
}

/// Create a raw APP2 marker with ISO 21496-1 namespace and arbitrary payload.
///
/// Low-level building block — most callers should use [`create_jpeg_iso_markers`]
/// instead, which handles both primary and gain map markers correctly.
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

/// Read headroom + per-channel payload starting at `pos`.
/// Shared between the AVIF/JXL and JPEG parsers.
fn read_payload(
    data: &[u8],
    mut pos: usize,
    channel_count: usize,
    use_base_colour_space: bool,
) -> Result<GainMapMetadata> {
    let is_multichannel = channel_count == 3;

    let (base_headroom, new_pos) = read_unsigned_fraction(data, pos)?;
    pos = new_pos;
    let (alt_headroom, new_pos) = read_unsigned_fraction(data, pos)?;
    pos = new_pos;

    let mut metadata = GainMapMetadata {
        base_hdr_headroom: base_headroom.to_f32() as f64,
        alternate_hdr_headroom: alt_headroom.to_f32() as f64,
        use_base_color_space: use_base_colour_space,
        ..Default::default()
    };

    for ch in 0..channel_count {
        let (min_frac, p) = read_signed_fraction(data, pos)?;
        pos = p;
        let (max_frac, p) = read_signed_fraction(data, pos)?;
        pos = p;
        let (gamma_frac, p) = read_unsigned_fraction(data, pos)?;
        pos = p;
        let (base_offset_frac, p) = read_signed_fraction(data, pos)?;
        pos = p;
        let (alt_offset_frac, p) = read_signed_fraction(data, pos)?;
        pos = p;

        let min_val = min_frac.to_f32() as f64;
        let max_val = max_frac.to_f32() as f64;
        let gamma_val = gamma_frac.to_f32() as f64;
        let base_off = base_offset_frac.to_f32() as f64;
        let alt_off = alt_offset_frac.to_f32() as f64;

        if is_multichannel {
            metadata.gain_map_min[ch] = min_val;
            metadata.gain_map_max[ch] = max_val;
            metadata.gamma[ch] = gamma_val;
            metadata.base_offset[ch] = base_off;
            metadata.alternate_offset[ch] = alt_off;
        } else {
            metadata.gain_map_min = [min_val; 3];
            metadata.gain_map_max = [max_val; 3];
            metadata.gamma = [gamma_val; 3];
            metadata.base_offset = [base_off; 3];
            metadata.alternate_offset = [alt_off; 3];
        }
    }

    Ok(metadata)
}

/// Write headroom + per-channel payload to `buf`.
/// Shared between the AVIF/JXL and JPEG serializers.
fn write_payload(buf: &mut Vec<u8>, metadata: &GainMapMetadata, channel_count: usize) {
    let base_headroom = UnsignedFraction::from_f32(metadata.base_hdr_headroom as f32);
    write_unsigned_fraction(buf, base_headroom);

    let alt_headroom = UnsignedFraction::from_f32(metadata.alternate_hdr_headroom as f32);
    write_unsigned_fraction(buf, alt_headroom);

    for ch in 0..channel_count {
        write_signed_fraction(buf, Fraction::from_f32(metadata.gain_map_min[ch] as f32));
        write_signed_fraction(buf, Fraction::from_f32(metadata.gain_map_max[ch] as f32));
        write_unsigned_fraction(buf, UnsignedFraction::from_f32(metadata.gamma[ch] as f32));
        write_signed_fraction(buf, Fraction::from_f32(metadata.base_offset[ch] as f32));
        write_signed_fraction(
            buf,
            Fraction::from_f32(metadata.alternate_offset[ch] as f32),
        );
    }
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
            gain_map_min: [0.0; 3],
            gain_map_max: [2.0; 3],
            gamma: [1.0; 3],
            base_offset: [0.015625; 3],
            alternate_offset: [0.015625; 3],
            base_hdr_headroom: 0.0,
            alternate_hdr_headroom: 2.0,
            use_base_color_space: true,
        };

        let serialized = serialize_iso21496_avif(&original);
        let parsed = parse_iso21496_avif(&serialized).unwrap();

        assert!((parsed.gain_map_max[0] - 2.0).abs() < 0.01);
        assert!((parsed.gain_map_min[0] - 0.0).abs() < 0.01);
        assert!((parsed.alternate_hdr_headroom - 2.0).abs() < 0.01);
        assert!((parsed.gamma[0] - 1.0).abs() < 0.01);
        assert!((parsed.base_offset[0] - 0.015625).abs() < 0.001);
        assert!(parsed.use_base_color_space);
    }

    #[test]
    fn test_roundtrip_multi_channel() {
        let original = GainMapMetadata {
            gain_map_max: [100.5, 101.5, 102.5],
            gain_map_min: [1.5, 1.6, 1.7],
            gamma: [1.0, 1.01, 1.02],
            base_offset: [0.0625, 0.0875, 0.1125],
            alternate_offset: [0.0625, 0.0875, 0.1125],
            base_hdr_headroom: 0.0,
            alternate_hdr_headroom: 10000.0 / 203.0,
            use_base_color_space: false,
        };

        let serialized = serialize_iso21496_avif(&original);
        let parsed = parse_iso21496_avif(&serialized).unwrap();

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
                (parsed.gain_map_max[i] - original.gain_map_max[i]).abs()
                    / original.gain_map_max[i]
                    < tol,
                "max_content_boost[{}]: {} vs {}",
                i,
                parsed.gain_map_max[i],
                original.gain_map_max[i]
            );
            assert!(
                (parsed.gain_map_min[i] - original.gain_map_min[i]).abs()
                    / original.gain_map_min[i]
                    < tol,
                "min_content_boost[{}]: {} vs {}",
                i,
                parsed.gain_map_min[i],
                original.gain_map_min[i]
            );
            assert!(
                (parsed.gamma[i] - original.gamma[i]).abs() < 0.01,
                "gamma[{}]: {} vs {}",
                i,
                parsed.gamma[i],
                original.gamma[i]
            );
            assert!(
                (parsed.base_offset[i] - original.base_offset[i]).abs() < 0.001,
                "offset_sdr[{}]: {} vs {}",
                i,
                parsed.base_offset[i],
                original.base_offset[i]
            );
            assert!(
                (parsed.alternate_offset[i] - original.alternate_offset[i]).abs() < 0.001,
                "offset_hdr[{}]: {} vs {}",
                i,
                parsed.alternate_offset[i],
                original.alternate_offset[i]
            );
        }

        // Verify channels are distinct
        assert_ne!(parsed.gain_map_max[0], parsed.gain_map_max[1]);
        assert_ne!(parsed.gain_map_max[1], parsed.gain_map_max[2]);
    }

    #[test]
    fn test_roundtrip_negative_offsets() {
        let original = GainMapMetadata {
            gain_map_max: [10.0, 11.0, 12.0],
            gain_map_min: [0.5, 0.6, 0.7],
            gamma: [1.0, 1.1, 1.2],
            base_offset: [-0.0625, -0.0615, -0.0605],
            alternate_offset: [-0.0625, -0.0615, -0.0605],
            base_hdr_headroom: 0.0,
            alternate_hdr_headroom: 1000.0 / 203.0,
            use_base_color_space: true,
        };

        let serialized = serialize_iso21496_avif(&original);
        let parsed = parse_iso21496_avif(&serialized).unwrap();

        assert!(parsed.use_base_color_space);

        let tol = 0.001;
        for i in 0..3 {
            assert!(
                (parsed.base_offset[i] - original.base_offset[i]).abs() < tol,
                "offset_sdr[{}]: {} vs {}",
                i,
                parsed.base_offset[i],
                original.base_offset[i]
            );
            assert!(
                (parsed.alternate_offset[i] - original.alternate_offset[i]).abs() < tol,
                "offset_hdr[{}]: {} vs {}",
                i,
                parsed.alternate_offset[i],
                original.alternate_offset[i]
            );
        }
    }

    // ========================================================================
    // Wire format verification
    // ========================================================================

    #[test]
    fn test_header_layout() {
        let metadata = GainMapMetadata::new();
        let serialized = serialize_iso21496_avif(&metadata);

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
        let serialized = serialize_iso21496_avif(&metadata);

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
            gain_map_max: [4.0, 5.0, 6.0],
            gain_map_min: [1.0, 1.5, 2.0],
            gamma: [1.0, 1.1, 1.2],
            base_offset: [0.01; 3],
            alternate_offset: [0.01; 3],
            base_hdr_headroom: 0.0,
            alternate_hdr_headroom: 2.585,
            use_base_color_space: true,
        };
        let serialized = serialize_iso21496_avif(&metadata);

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

        let parsed = parse_iso21496_avif(&blob).unwrap();

        assert_eq!(parsed.base_hdr_headroom, 0.0); // wire: 0/1 → log2 = 0
        assert!((parsed.alternate_hdr_headroom - 2.0).abs() < 0.001); // wire: 2/1 → log2 = 2
        assert_eq!(parsed.gain_map_min, [0.0; 3]); // wire: 0/1 → log2 = 0
        assert!((parsed.gain_map_max[0] - 2.0).abs() < 0.001); // wire: 2/1 → log2 = 2
        assert_eq!(parsed.gamma, [1.0; 3]);
        assert_eq!(parsed.base_offset, [0.015625; 3]); // 1/64
        assert_eq!(parsed.alternate_offset, [0.015625; 3]); // 1/64
        assert!(parsed.use_base_color_space);
    }

    // ========================================================================
    // Single-channel replication
    // ========================================================================

    #[test]
    fn test_single_channel_replicates_to_all() {
        let original = GainMapMetadata {
            gain_map_min: [2.0; 3],
            gain_map_max: [3.0; 3],
            gamma: [1.5; 3],
            base_offset: [0.03; 3],
            alternate_offset: [0.05; 3],
            base_hdr_headroom: 0.0,
            alternate_hdr_headroom: 3.0,
            use_base_color_space: false,
        };

        let serialized = serialize_iso21496_avif(&original);
        assert_eq!(
            serialized[FLAGS_OFFSET] & FLAG_MULTI_CHANNEL,
            0,
            "should serialize as single channel"
        );

        let parsed = parse_iso21496_avif(&serialized).unwrap();

        for i in 1..3 {
            assert_eq!(
                parsed.gain_map_min[0], parsed.gain_map_min[i],
                "min_content_boost[0] != [{}]",
                i
            );
            assert_eq!(
                parsed.gain_map_max[0], parsed.gain_map_max[i],
                "max_content_boost[0] != [{}]",
                i
            );
            assert_eq!(parsed.gamma[0], parsed.gamma[i], "gamma[0] != [{}]", i);
            assert_eq!(
                parsed.base_offset[0], parsed.base_offset[i],
                "offset_sdr[0] != [{}]",
                i
            );
            assert_eq!(
                parsed.alternate_offset[0], parsed.alternate_offset[i],
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
            gain_map_min: [0.0; 3],
            gain_map_max: [1.0; 3],
            gamma: [1.0; 3],
            base_offset: [0.0; 3],
            alternate_offset: [0.0; 3],
            base_hdr_headroom: 0.0,
            alternate_hdr_headroom: 1.0,
            use_base_color_space: true,
        };

        let serialized = serialize_iso21496_avif(&original);
        let parsed = parse_iso21496_avif(&serialized).unwrap();

        assert!((parsed.base_hdr_headroom - 0.0).abs() < 0.001);
        assert!((parsed.alternate_hdr_headroom - 1.0).abs() < 0.001); // log2 = 1.0 → 2× boost
    }

    #[test]
    fn test_gamma_one() {
        let original = GainMapMetadata {
            gamma: [1.0; 3],
            ..GainMapMetadata::new()
        };

        let serialized = serialize_iso21496_avif(&original);
        let parsed = parse_iso21496_avif(&serialized).unwrap();

        assert!((parsed.gamma[0] - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_extreme_boost_values() {
        // log2(10000) ≈ 13.29, log2(0.01) ≈ −6.64
        let original = GainMapMetadata {
            gain_map_min: [-6.644; 3], // log2(0.01)
            gain_map_max: [13.288; 3], // log2(10000)
            gamma: [0.01; 3],
            base_offset: [0.0; 3],
            alternate_offset: [0.0; 3],
            base_hdr_headroom: 0.0,
            alternate_hdr_headroom: 5.623, // log2(10000/203)
            use_base_color_space: true,
        };

        let serialized = serialize_iso21496_avif(&original);
        let parsed = parse_iso21496_avif(&serialized).unwrap();

        let tol = 0.01;
        assert!(
            (parsed.gain_map_max[0] - 13.288).abs() < tol,
            "gain_map_max: {} vs 13.288",
            parsed.gain_map_max[0]
        );
        assert!(
            (parsed.gain_map_min[0] - (-6.644)).abs() < tol,
            "gain_map_min: {} vs -6.644",
            parsed.gain_map_min[0]
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
        assert!(parse_iso21496_avif(&[]).is_err());
    }

    #[test]
    fn test_too_short() {
        // Just the version byte — not enough for the full header
        assert!(parse_iso21496_avif(&[0x00]).is_err());
        assert!(parse_iso21496_avif(&[0x00, 0x00]).is_err());
        assert!(parse_iso21496_avif(&[0x00, 0x00, 0x00, 0x00, 0x00]).is_err());
    }

    #[test]
    fn test_invalid_version() {
        let mut blob = vec![0u8; 62];
        blob[0] = 1; // version 1 — unsupported
        let result = parse_iso21496_avif(&blob);
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
        assert!(parse_iso21496_avif(&data).is_err());
    }

    #[test]
    fn test_zero_denominator_in_headroom() {
        // Build a valid-length blob but with zero denominator in base_hdr_headroom
        let mut data = vec![0u8; 62];
        data[0] = 0; // version
        // min_ver, writer_ver, flags all 0
        // base_hdr_headroom: numerator=0, denominator=0
        // denominator bytes at offset 10..14 are already 0
        let result = parse_iso21496_avif(&data);
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
        let mut data = serialize_iso21496_avif(&metadata);
        // gain_map_min_d is at offset: 6 (header) + 16 (headroom) + 4 (min_n) = 26..30
        data[26] = 0;
        data[27] = 0;
        data[28] = 0;
        data[29] = 0;
        let result = parse_iso21496_avif(&data);
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
        let iso_data = serialize_iso21496_avif(&GainMapMetadata::new());
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
        // APP2 markers use JPEG format (no version byte)
        let iso_data = serialize_iso21496_jpeg(&GainMapMetadata::new());
        let marker = create_iso_app2_marker(&iso_data);

        let namespace = b"urn:iso:std:iso:ts:21496:-1\0";
        let payload_start = 4 + namespace.len();
        let extracted = &marker[payload_start..];
        assert_eq!(extracted, &iso_data);

        let parsed = parse_iso21496_jpeg(extracted).unwrap();
        let original = GainMapMetadata::new();
        assert!((parsed.base_hdr_headroom - original.base_hdr_headroom).abs() < 0.01);
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

    // ========================================================================
    // Continued fraction canonicality tests
    // ========================================================================

    #[test]
    fn test_fraction_produces_canonical_denominators() {
        // Exact binary float values should produce compact fractions,
        // matching libultrahdr's continued fraction output.
        let cases: &[(f32, i32, u32)] = &[
            (0.0, 0, 1),
            (1.0, 1, 1),
            (2.0, 2, 1),
            (4.0, 4, 1),
            (-1.0, -1, 1),
            (0.5, 1, 2),
            (0.25, 1, 4),
            (0.015625, 1, 64), // 1/64, common offset
            (-0.015625, -1, 64),
        ];
        for &(value, exp_num, exp_den) in cases {
            let frac = Fraction::from_f32(value);
            assert_eq!(
                (frac.numerator, frac.denominator),
                (exp_num, exp_den),
                "Fraction::from_f32({value}): expected {exp_num}/{exp_den}, got {}/{}",
                frac.numerator,
                frac.denominator
            );
        }
    }

    #[test]
    fn test_unsigned_fraction_produces_canonical_denominators() {
        let cases: &[(f32, u32, u32)] = &[
            (0.0, 0, 1),
            (1.0, 1, 1),
            (2.0, 2, 1),
            (4.0, 4, 1),
            (0.5, 1, 2),
            (0.25, 1, 4),
            (0.015625, 1, 64),
        ];
        for &(value, exp_num, exp_den) in cases {
            let frac = UnsignedFraction::from_f32(value);
            assert_eq!(
                (frac.numerator, frac.denominator),
                (exp_num, exp_den),
                "UnsignedFraction::from_f32({value}): expected {exp_num}/{exp_den}, got {}/{}",
                frac.numerator,
                frac.denominator
            );
        }
    }

    // ========================================================================
    // Canonical Adobe fixture tests
    // ========================================================================

    /// The exact ISO 21496-1 APP2 payload from Adobe Photoshop 27.3's
    /// `fullColor-fullRes-IDEAL.jpg` primary JPEG (version-only block).
    const ADOBE_PRIMARY_ISO_PAYLOAD: [u8; 4] = [0x00, 0x00, 0x00, 0x00];

    /// The exact ISO 21496-1 APP2 payload from Adobe Photoshop 27.3's
    /// `fullColor-fullRes-IDEAL.jpg` secondary (gain map) JPEG.
    /// Single-channel, use_base_colour_space=true.
    ///
    /// Metadata: base_headroom=0/1, alt_headroom=4/1,
    /// gain_map_min=0/1, gain_map_max=5895489/1048576 (~5.622),
    /// gamma=1/1, base_offset=0/1, alt_offset=0/1.
    #[rustfmt::skip]
    const ADOBE_GAINMAP_ISO_PAYLOAD: [u8; 61] = [
        // Header (JPEG format: no version byte)
        0x00, 0x00,             // minimum_version = 0
        0x00, 0x00,             // writer_version = 0
        0x40,                   // flags = 0x40 (use_base_colour_space)
        // base_hdr_headroom = 0/1
        0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x01,
        // alternate_hdr_headroom = 4/1
        0x00, 0x00, 0x00, 0x04,
        0x00, 0x00, 0x00, 0x01,
        // gain_map_min = 0/1
        0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x01,
        // gain_map_max = 5895489/1048576
        0x00, 0x59, 0xF5, 0x41,
        0x00, 0x10, 0x00, 0x00,
        // gamma = 1/1
        0x00, 0x00, 0x00, 0x01,
        0x00, 0x00, 0x00, 0x01,
        // base_offset = 0/1
        0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x01,
        // alternate_offset = 0/1
        0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x01,
    ];

    #[test]
    fn test_parse_adobe_primary_version_block() {
        // The 4-byte version-only block is NOT valid gain map metadata —
        // it should fail to parse as a gain map payload (too short).
        let result = parse_iso21496_jpeg(&ADOBE_PRIMARY_ISO_PAYLOAD);
        assert!(
            result.is_err(),
            "version-only block should not parse as gain map metadata"
        );
    }

    #[test]
    fn test_parse_adobe_gainmap_payload() {
        let parsed = parse_iso21496_jpeg(&ADOBE_GAINMAP_ISO_PAYLOAD)
            .expect("should parse canonical Adobe payload");

        assert!(parsed.use_base_color_space);
        assert_eq!(parsed.base_hdr_headroom, 0.0);
        assert!((parsed.alternate_hdr_headroom - 4.0).abs() < 0.001);
        assert_eq!(parsed.gain_map_min, [0.0; 3]);
        // 5895489/1048576 ≈ 5.6224
        let expected_max = 5_895_489.0 / 1_048_576.0;
        assert!(
            (parsed.gain_map_max[0] - expected_max).abs() < 0.001,
            "gain_map_max: {} vs {}",
            parsed.gain_map_max[0],
            expected_max
        );
        assert_eq!(parsed.gamma, [1.0; 3]);
        assert_eq!(parsed.base_offset, [0.0; 3]);
        assert_eq!(parsed.alternate_offset, [0.0; 3]);
    }

    #[test]
    fn test_serialize_matches_adobe_fixture_structure() {
        // Parse the Adobe payload, re-serialize, and verify the structure matches.
        let parsed = parse_iso21496_jpeg(&ADOBE_GAINMAP_ISO_PAYLOAD).unwrap();
        let reserialized = serialize_iso21496_jpeg(&parsed);

        // Same length
        assert_eq!(
            reserialized.len(),
            ADOBE_GAINMAP_ISO_PAYLOAD.len(),
            "serialized length mismatch"
        );

        // Header must match exactly
        assert_eq!(
            &reserialized[..5],
            &ADOBE_GAINMAP_ISO_PAYLOAD[..5],
            "header mismatch"
        );

        // Headroom fractions must match exactly (0/1 and 4/1 are exact)
        assert_eq!(
            &reserialized[5..21],
            &ADOBE_GAINMAP_ISO_PAYLOAD[5..21],
            "headroom fractions mismatch"
        );

        // All exact-value fractions (0/1, 1/1) must match
        // gain_map_min = 0/1 at offset 21..29
        assert_eq!(
            &reserialized[21..29],
            &ADOBE_GAINMAP_ISO_PAYLOAD[21..29],
            "gain_map_min mismatch"
        );

        // gamma = 1/1 at offset 37..45
        assert_eq!(
            &reserialized[37..45],
            &ADOBE_GAINMAP_ISO_PAYLOAD[37..45],
            "gamma mismatch"
        );

        // base_offset = 0/1 at offset 45..53
        assert_eq!(
            &reserialized[45..53],
            &ADOBE_GAINMAP_ISO_PAYLOAD[45..53],
            "base_offset mismatch"
        );

        // alt_offset = 0/1 at offset 53..61
        assert_eq!(
            &reserialized[53..61],
            &ADOBE_GAINMAP_ISO_PAYLOAD[53..61],
            "alt_offset mismatch"
        );

        // gain_map_max: the value goes through f32 roundtrip so the
        // continued fraction may not reproduce the exact same numerator.
        // Verify the decoded value matches within f32 precision.
        let our_max_n = i32::from_be_bytes([
            reserialized[29],
            reserialized[30],
            reserialized[31],
            reserialized[32],
        ]);
        let our_max_d = u32::from_be_bytes([
            reserialized[33],
            reserialized[34],
            reserialized[35],
            reserialized[36],
        ]);
        let our_val = our_max_n as f64 / our_max_d as f64;
        let adobe_val = 5_895_489.0 / 1_048_576.0;
        assert!(
            (our_val - adobe_val).abs() < 1e-6,
            "gain_map_max value mismatch: {our_val} vs {adobe_val}"
        );
    }

    #[test]
    fn test_version_only_app2_marker() {
        let marker = create_version_only_iso_app2();

        // Must start with APP2 marker
        assert_eq!(marker[0], 0xFF);
        assert_eq!(marker[1], 0xE2);

        // Extract payload after marker + length + namespace
        let namespace = b"urn:iso:std:iso:ts:21496:-1\0";
        let payload_start = 4 + namespace.len();
        let payload = &marker[payload_start..];

        // Must be exactly the 4-byte version-only block
        assert_eq!(payload, &ADOBE_PRIMARY_ISO_PAYLOAD);
    }

    #[test]
    fn test_serialize_canonical_simple_metadata() {
        // Metadata with values that should produce exact canonical fractions.
        let metadata = GainMapMetadata {
            gain_map_min: [0.0; 3],
            gain_map_max: [2.0; 3],
            gamma: [1.0; 3],
            base_offset: [0.015625; 3], // 1/64
            alternate_offset: [0.015625; 3],
            base_hdr_headroom: 0.0,
            alternate_hdr_headroom: 2.0,
            use_base_color_space: true,
        };

        let serialized = serialize_iso21496_jpeg(&metadata);

        // Verify exact bytes for known fractions
        let pos = 5; // after header
        // base_hdr_headroom = 0/1
        assert_eq!(&serialized[pos..pos + 8], &[0, 0, 0, 0, 0, 0, 0, 1]);
        // alt_hdr_headroom = 2/1
        assert_eq!(&serialized[pos + 8..pos + 16], &[0, 0, 0, 2, 0, 0, 0, 1]);
        // gain_map_min = 0/1
        assert_eq!(&serialized[pos + 16..pos + 24], &[0, 0, 0, 0, 0, 0, 0, 1]);
        // gain_map_max = 2/1
        assert_eq!(&serialized[pos + 24..pos + 32], &[0, 0, 0, 2, 0, 0, 0, 1]);
        // gamma = 1/1
        assert_eq!(&serialized[pos + 32..pos + 40], &[0, 0, 0, 1, 0, 0, 0, 1]);
        // base_offset = 1/64
        assert_eq!(&serialized[pos + 40..pos + 48], &[0, 0, 0, 1, 0, 0, 0, 64]);
        // alt_offset = 1/64
        assert_eq!(&serialized[pos + 48..pos + 56], &[0, 0, 0, 1, 0, 0, 0, 64]);
    }

    #[test]
    fn test_create_jpeg_iso_markers() {
        let metadata = GainMapMetadata {
            gain_map_min: [0.0; 3],
            gain_map_max: [2.0; 3],
            gamma: [1.0; 3],
            base_offset: [0.0; 3],
            alternate_offset: [0.0; 3],
            base_hdr_headroom: 0.0,
            alternate_hdr_headroom: 4.0,
            use_base_color_space: true,
        };

        let markers = create_jpeg_iso_markers(&metadata);
        let namespace = b"urn:iso:std:iso:ts:21496:-1\0";

        // Primary marker: APP2 with version-only payload
        assert_eq!(markers.primary[0], 0xFF);
        assert_eq!(markers.primary[1], 0xE2);
        let primary_payload = &markers.primary[4 + namespace.len()..];
        assert_eq!(primary_payload, &[0x00, 0x00, 0x00, 0x00]);

        // Gain map marker: APP2 with full metadata payload
        assert_eq!(markers.gain_map[0], 0xFF);
        assert_eq!(markers.gain_map[1], 0xE2);
        let gm_payload = &markers.gain_map[4 + namespace.len()..];

        // Should be parseable as JPEG ISO 21496-1
        let parsed = parse_iso21496_jpeg(gm_payload).unwrap();
        assert!(parsed.use_base_color_space);
        assert!((parsed.alternate_hdr_headroom - 4.0).abs() < 0.001);
        assert!((parsed.gain_map_max[0] - 2.0).abs() < 0.001);
    }

    // ========================================================================
    // Canonical interop regression tests
    // ========================================================================
    // These tests detect the class of bug reported in jcayzac/ultrajpeg#6:
    // serialized ISO bytes that are technically valid but cause browser
    // rendering failures due to non-canonical fraction encoding.

    #[test]
    fn test_no_fraction_uses_million_denominator() {
        // Regression test: our old Fraction::from_f32 used a fixed 1,000,000
        // denominator for every value. Browsers (Chromium) choked on this.
        let test_values: &[f32] = &[
            0.0,
            1.0,
            2.0,
            4.0,
            0.5,
            0.015625,
            -0.015625,
            0.25,
            10000.0 / 203.0,
            5.622376,
        ];
        for &v in test_values {
            let frac = Fraction::from_f32(v);
            assert_ne!(
                frac.denominator, 1_000_000,
                "Fraction::from_f32({v}) produced denominator 1000000 \
                 ({}/1000000) — this is the non-canonical encoding that \
                 causes browser interop failures",
                frac.numerator
            );
        }
        for &v in test_values {
            if v < 0.0 {
                continue;
            }
            let frac = UnsignedFraction::from_f32(v);
            assert_ne!(
                frac.denominator, 1_000_000,
                "UnsignedFraction::from_f32({v}) produced denominator 1000000 \
                 ({}/1000000)",
                frac.numerator
            );
        }
    }

    #[test]
    fn test_serialized_payload_has_no_million_denominators() {
        // End-to-end: serialize real metadata, scan every fraction in the
        // output, verify none have the 1M denominator anti-pattern.
        let metadata = GainMapMetadata {
            gain_map_min: [0.0, -0.5, -1.0],
            gain_map_max: [2.0, 3.0, 4.0],
            gamma: [1.0, 1.0, 1.0],
            base_offset: [0.015625; 3],
            alternate_offset: [0.015625; 3],
            base_hdr_headroom: 0.0,
            alternate_hdr_headroom: 10000.0_f64 / 203.0,
            use_base_color_space: true,
        };

        for format in [
            crate::Iso21496Format::JpegApp2,
            crate::Iso21496Format::AvifTmap,
        ] {
            let bytes = serialize_iso21496(&metadata, format);

            // Skip the header to reach fractions
            let header_size = match format {
                crate::Iso21496Format::AvifTmap => HEADER_SIZE,
                crate::Iso21496Format::JpegApp2 => JPEG_HEADER_SIZE,
            };

            // Scan every 8-byte fraction pair (4-byte num + 4-byte den)
            let frac_data = &bytes[header_size..];
            assert_eq!(
                frac_data.len() % 8,
                0,
                "payload not aligned to fraction pairs"
            );

            for (i, chunk) in frac_data.chunks(8).enumerate() {
                let denom = u32::from_be_bytes([chunk[4], chunk[5], chunk[6], chunk[7]]);
                assert_ne!(
                    denom, 1_000_000,
                    "fraction[{i}] in {format:?} has denominator 1000000 — \
                     non-canonical encoding"
                );
                assert_ne!(denom, 0, "fraction[{i}] in {format:?} has zero denominator");
            }
        }
    }

    #[test]
    fn test_parse_roundtrip_preserves_values_across_formats() {
        // Serialize with JPEG format, parse back, re-serialize with AVIF format,
        // parse back again — values should be identical within f32 precision.
        let original = GainMapMetadata {
            gain_map_min: [-1.0, 0.0, 0.5],
            gain_map_max: [2.0, 3.5, 5.622376],
            gamma: [1.0, 0.5, 2.0],
            base_offset: [0.015625, 0.0, 0.03125],
            alternate_offset: [0.015625, 0.0, 0.0625],
            base_hdr_headroom: 0.0,
            alternate_hdr_headroom: 4.0,
            use_base_color_space: false,
        };

        // JPEG round-trip
        let jpeg_bytes = serialize_iso21496(&original, crate::Iso21496Format::JpegApp2);
        let from_jpeg = parse_iso21496(&jpeg_bytes, crate::Iso21496Format::JpegApp2).unwrap();

        // AVIF round-trip
        let avif_bytes = serialize_iso21496(&original, crate::Iso21496Format::AvifTmap);
        let from_avif = parse_iso21496(&avif_bytes, crate::Iso21496Format::AvifTmap).unwrap();

        // Both must produce the same values (within f32 precision)
        let tol = 0.001;
        for ch in 0..3 {
            assert!(
                (from_jpeg.gain_map_min[ch] - from_avif.gain_map_min[ch]).abs() < tol,
                "gain_map_min[{ch}] differs: jpeg={} avif={}",
                from_jpeg.gain_map_min[ch],
                from_avif.gain_map_min[ch]
            );
            assert!(
                (from_jpeg.gain_map_max[ch] - from_avif.gain_map_max[ch]).abs() < tol,
                "gain_map_max[{ch}] differs: jpeg={} avif={}",
                from_jpeg.gain_map_max[ch],
                from_avif.gain_map_max[ch]
            );
            assert!(
                (from_jpeg.gamma[ch] - from_avif.gamma[ch]).abs() < tol,
                "gamma[{ch}] differs: jpeg={} avif={}",
                from_jpeg.gamma[ch],
                from_avif.gamma[ch]
            );
            assert!(
                (from_jpeg.base_offset[ch] - from_avif.base_offset[ch]).abs() < tol,
                "base_offset[{ch}] differs: jpeg={} avif={}",
                from_jpeg.base_offset[ch],
                from_avif.base_offset[ch]
            );
            assert!(
                (from_jpeg.alternate_offset[ch] - from_avif.alternate_offset[ch]).abs() < tol,
                "alternate_offset[{ch}] differs: jpeg={} avif={}",
                from_jpeg.alternate_offset[ch],
                from_avif.alternate_offset[ch]
            );
        }
        assert!((from_jpeg.alternate_hdr_headroom - from_avif.alternate_hdr_headroom).abs() < tol);
        assert_eq!(
            from_jpeg.use_base_color_space,
            from_avif.use_base_color_space
        );
    }
}
