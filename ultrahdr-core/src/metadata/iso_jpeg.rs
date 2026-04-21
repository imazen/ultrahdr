//! JPEG-specific ISO 21496-1 APP2 marker helpers.
//!
//! This module owns the JPEG APP2 *envelope* — the `FF E2` marker byte,
//! the big-endian length field, and the `urn:iso:std:iso:ts:21496:-1\0`
//! namespace string — around the ISO 21496-1 binary payload that
//! [`zencodec::gainmap`] produces.
//!
//! # Provisional location
//!
//! This implementation is expected to migrate to `zencodec` in a future
//! release. `zencodec` already owns the inner payload
//! (flags + fractions); the APP2 envelope is also ISO 21496-1-specific,
//! it is not JPEG-codec-specific, and every other tool emitting an
//! Ultra HDR-class JPEG would otherwise reimplement the same four-field
//! wrap. When the move happens the public API of this module is intended
//! to stay stable:
//!
//! - [`create_iso_app2_marker`], [`create_version_only_iso_app2`],
//!   [`create_jpeg_iso_markers`], [`parse_iso21496`], and
//!   [`serialize_iso21496`] are **stable entry points**. Callers at this
//!   path are supported across the migration; the bodies will become
//!   thin `pub use` or delegation to zencodec.
//! - [`ISO_21496_1_URN`] is pinned here now so downstream code can
//!   reference a single constant instead of re-declaring the byte
//!   literal.
//!
//! Nothing outside this module should depend on *how* the envelope is
//! currently built.

use alloc::vec::Vec;

use crate::Iso21496Format;

// ---------------------------------------------------------------------------
// Stable surface
// ---------------------------------------------------------------------------

/// The ISO 21496-1 namespace string that prefixes every gain-map APP2
/// marker payload. 28 bytes including the trailing NUL.
///
/// Defined by ISO/IEC 21496-1. libultrahdr writes the same byte sequence
/// at `libultrahdr/lib/src/jpegr.cpp:69` (via `kIsoNameSpace` + the
/// explicit `.size() + 1` null-terminator accounting at line 1129).
pub const ISO_21496_1_URN: &[u8; 28] = b"urn:iso:std:iso:ts:21496:-1\0";

/// The two ISO 21496-1 APP2 markers needed for a canonical Ultra HDR JPEG.
///
/// Returned by [`create_jpeg_iso_markers`].
pub struct JpegIsoMarkers {
    /// APP2 marker for the **primary** JPEG codestream.
    ///
    /// Signals ISO 21496-1 awareness. Payload is 4 bytes of zeros
    /// (`min_version=0, writer_version=0`); it does NOT carry gain map
    /// parameters — those live in the secondary JPEG's APP2.
    pub primary: Vec<u8>,

    /// APP2 marker for the **gain map** (secondary) JPEG codestream.
    ///
    /// Carries the full serialized gain map metadata (headroom, per-channel
    /// min/max/gamma/offsets) as produced by
    /// [`zencodec::gainmap::serialize_iso21496_fmt`].
    pub gain_map: Vec<u8>,
}

/// Create both ISO 21496-1 APP2 markers for a canonical Ultra HDR JPEG.
///
/// A spec-compliant Ultra HDR JPEG needs two ISO APP2 markers:
/// 1. A version-only block in the **primary** JPEG (signals ISO 21496-1 support).
/// 2. The full gain map metadata in the **secondary** (gain map) JPEG.
///
/// This function produces both in one call so callers can't forget either.
///
/// # Example
///
/// ```
/// use ultrahdr_core::{GainMapMetadata, metadata::iso_jpeg::create_jpeg_iso_markers};
///
/// let metadata = GainMapMetadata::default();
/// let markers = create_jpeg_iso_markers(&metadata);
///
/// // Insert markers.primary into the primary JPEG after SOI.
/// // Insert markers.gain_map into the gain map JPEG after SOI.
/// ```
pub fn create_jpeg_iso_markers(metadata: &crate::GainMapMetadata) -> JpegIsoMarkers {
    let iso_payload = zencodec::gainmap::serialize_iso21496_fmt(metadata, Iso21496Format::JxlJhgm);
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
    create_iso_app2_marker(&[0x00, 0x00, 0x00, 0x00])
}

/// Create a raw APP2 marker with the ISO 21496-1 URN namespace and an
/// arbitrary payload.
///
/// Low-level building block — most callers should use
/// [`create_jpeg_iso_markers`] instead, which handles both primary and
/// gain map markers correctly.
///
/// The payload is not validated. Pass whatever
/// [`zencodec::gainmap::serialize_iso21496_fmt`] or
/// [`create_version_only_iso_app2`] produces.
pub fn create_iso_app2_marker(iso_data: &[u8]) -> Vec<u8> {
    let total_length = 2 + ISO_21496_1_URN.len() + iso_data.len();
    let mut marker = Vec::with_capacity(2 + total_length);
    marker.push(0xFF);
    marker.push(0xE2);
    marker.push(((total_length >> 8) & 0xFF) as u8);
    marker.push((total_length & 0xFF) as u8);
    marker.extend_from_slice(ISO_21496_1_URN);
    marker.extend_from_slice(iso_data);
    marker
}

/// Parse ISO 21496-1 binary gain map metadata (convenience wrapper).
///
/// The input MUST be the bare payload (after the JPEG APP2 segment
/// header and the [`ISO_21496_1_URN`] namespace have been stripped).
/// Delegates to [`zencodec::gainmap::parse_iso21496_fmt`] with error
/// mapping to [`crate::Error::IsoParse`].
pub fn parse_iso21496(
    data: &[u8],
    format: Iso21496Format,
) -> crate::Result<crate::GainMapMetadata> {
    zencodec::gainmap::parse_iso21496_fmt(data, format)
        .map_err(|e| crate::Error::IsoParse(alloc::string::ToString::to_string(&e)))
}

/// Serialize gain map metadata to ISO 21496-1 binary format (convenience wrapper).
///
/// Produces the bare payload (no APP2 marker, no URN). To get a full
/// JPEG APP2 marker ready for splicing into a JPEG bitstream, use
/// [`create_iso_app2_marker`] or [`create_jpeg_iso_markers`].
///
/// Delegates to [`zencodec::gainmap::serialize_iso21496_fmt`].
pub fn serialize_iso21496(metadata: &crate::GainMapMetadata, format: Iso21496Format) -> Vec<u8> {
    zencodec::gainmap::serialize_iso21496_fmt(metadata, format)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn test_metadata() -> crate::GainMapMetadata {
        crate::types::metadata_from_arrays(
            [0.0; 3],
            [2.0; 3],
            [1.0; 3],
            [1.0 / 64.0; 3],
            [1.0 / 64.0; 3],
            0.0,
            2.0,
            true,
            false,
        )
    }

    #[test]
    fn urn_is_28_bytes_with_null_terminator() {
        assert_eq!(ISO_21496_1_URN.len(), 28);
        assert_eq!(ISO_21496_1_URN[27], 0);
        assert_eq!(&ISO_21496_1_URN[..27], b"urn:iso:std:iso:ts:21496:-1");
    }

    #[test]
    fn test_roundtrip_avif() {
        let original = test_metadata();
        let bytes = serialize_iso21496(&original, Iso21496Format::AvifTmap);
        let parsed = parse_iso21496(&bytes, Iso21496Format::AvifTmap).unwrap();
        assert!((parsed.channels[0].max - 2.0).abs() < 0.01);
        assert!((parsed.alternate_hdr_headroom - 2.0).abs() < 0.01);
    }

    #[test]
    fn test_roundtrip_jpeg() {
        let original = test_metadata();
        let bytes = serialize_iso21496(&original, Iso21496Format::JxlJhgm);
        let parsed = parse_iso21496(&bytes, Iso21496Format::JxlJhgm).unwrap();
        assert!((parsed.channels[0].max - 2.0).abs() < 0.01);
        assert!((parsed.alternate_hdr_headroom - 2.0).abs() < 0.01);
    }

    #[test]
    fn test_create_jpeg_iso_markers() {
        let metadata = test_metadata();
        let markers = create_jpeg_iso_markers(&metadata);
        assert!(markers.primary.len() > 4);
        assert!(markers.gain_map.len() > 4);
        assert_eq!(markers.primary[0], 0xFF);
        assert_eq!(markers.primary[1], 0xE2);
        assert_eq!(markers.gain_map[0], 0xFF);
        assert_eq!(markers.gain_map[1], 0xE2);
    }

    #[test]
    fn test_version_only_app2() {
        let marker = create_version_only_iso_app2();
        assert_eq!(marker[0], 0xFF);
        assert_eq!(marker[1], 0xE2);
        // Exact on-disk size: 2 (marker) + 2 (length) + 28 (URN) + 4 (version payload) = 36.
        assert_eq!(marker.len(), 36);
        // Length field counts itself + URN + payload: 2 + 28 + 4 = 34.
        assert_eq!(u16::from_be_bytes([marker[2], marker[3]]), 34);
        // URN at offset 4.
        assert_eq!(&marker[4..4 + ISO_21496_1_URN.len()], ISO_21496_1_URN);
        // Version payload of four zero bytes at the tail.
        assert_eq!(&marker[32..36], &[0x00, 0x00, 0x00, 0x00]);
    }
}
