//! JPEG-specific ISO 21496-1 APP2 marker helpers.
//!
//! These wrap [`zencodec::gainmap::serialize_iso21496_fmt`] with JPEG APP2
//! marker framing. The binary payload (flags + fractions) is produced by zencodec;
//! this module handles the JPEG marker envelope (FF E2 + length + URN namespace).

use alloc::vec::Vec;

use crate::Iso21496Format;

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
/// use ultrahdr_core::{GainMapMetadata, metadata::iso_jpeg::create_jpeg_iso_markers};
///
/// let metadata = GainMapMetadata::default();
/// let markers = create_jpeg_iso_markers(&metadata);
///
/// // Insert markers.primary into the primary JPEG after SOI
/// // Insert markers.gain_map into the gain map JPEG after SOI
/// ```
pub fn create_jpeg_iso_markers(metadata: &crate::GainMapMetadata) -> JpegIsoMarkers {
    let iso_payload = zencodec::gainmap::serialize_iso21496_fmt(metadata, Iso21496Format::JpegApp2);
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

/// Parse ISO 21496-1 binary gain map metadata (convenience wrapper).
///
/// Delegates to [`zencodec::gainmap::parse_iso21496_fmt`] with error mapping
/// to [`crate::Error::IsoParse`].
pub fn parse_iso21496(
    data: &[u8],
    format: Iso21496Format,
) -> crate::Result<crate::GainMapMetadata> {
    zencodec::gainmap::parse_iso21496_fmt(data, format)
        .map_err(|e| crate::Error::IsoParse(alloc::string::ToString::to_string(&e)))
}

/// Serialize gain map metadata to ISO 21496-1 binary format (convenience wrapper).
///
/// Delegates to [`zencodec::gainmap::serialize_iso21496_fmt`].
pub fn serialize_iso21496(metadata: &crate::GainMapMetadata, format: Iso21496Format) -> Vec<u8> {
    zencodec::gainmap::serialize_iso21496_fmt(metadata, format)
}

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
        let bytes = serialize_iso21496(&original, Iso21496Format::JpegApp2);
        let parsed = parse_iso21496(&bytes, Iso21496Format::JpegApp2).unwrap();
        assert!((parsed.channels[0].max - 2.0).abs() < 0.01);
        assert!((parsed.alternate_hdr_headroom - 2.0).abs() < 0.01);
    }

    #[test]
    fn test_create_jpeg_iso_markers() {
        let metadata = test_metadata();
        let markers = create_jpeg_iso_markers(&metadata);
        assert!(markers.primary.len() > 4);
        assert!(markers.gain_map.len() > 4);
        // Check APP2 marker bytes
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
        // Should contain the namespace string
        let namespace = b"urn:iso:std:iso:ts:21496:-1\0";
        let ns_start = 4; // after FF E2 LL LL
        assert_eq!(&marker[ns_start..ns_start + namespace.len()], namespace);
    }
}
