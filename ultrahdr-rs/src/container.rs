//! JPEG container utilities for Ultra HDR.
//!
//! Thin adapter layer over [`zenjpeg::container`] that exposes the pieces
//! ultrahdr-rs's decoder actually consumes:
//!
//! - [`scan_segments`] — collect `APPn` marker payloads into convenience
//!   [`AppSegment`] values with `is_mpf` / `is_xmp` / … predicates.
//! - [`primary_bounds`] — byte range of the first JPEG in a multi-image
//!   file (SOI → EOI).
//! - [`parse_mpf`] — MPF directory entries for a full JPEG buffer.
//! - [`MpfEntry`] / [`MpImageType`] re-exports from zenjpeg's canonical types.
//!
//! The parallel MPF / marker parsers that used to live here were deleted
//! in favor of zenjpeg's (length- and entropy-aware) implementation.

use std::ops::Range;
use ultrahdr_core::{Error, Result};
use zenjpeg::container::marker::{MarkerKind, MarkerSpan, find_jpeg_boundaries, iter};

/// MPF directory entry.
///
/// Re-exported from [`zenjpeg::container::types::MpfEntry`] — fields are
/// `image_type: MpImageType`, `offset: usize`, `size: usize` (absolute
/// file offsets/sizes).
pub use zenjpeg::container::types::MpfEntry;

/// MPF image type code (CIPA DC-007 Individual Image Attribute).
///
/// Re-exported from [`zenjpeg::container::types::MpImageType`]. Ultra HDR
/// gain maps use the `Undefined` variant (attribute code `0x000000`);
/// consumers distinguish gain map vs depth map vs other uses by looking
/// at which XMP / ISO 21496-1 metadata accompanies the file, not by the
/// MPF type code alone.
pub use zenjpeg::container::types::MpImageType;

/// An APP segment extracted from a JPEG.
///
/// Built by [`scan_segments`] from the length-aware marker iterator in
/// [`zenjpeg::container::marker`]. The predicates (`is_mpf`, `is_xmp`, …)
/// match on the leading identifier bytes in [`data`](Self::data).
#[derive(Debug, Clone)]
pub struct AppSegment {
    /// APPn marker index (`0`–`15` for `APP0`–`APP15`).
    pub marker_num: u8,
    /// Segment payload (excluding the `FF En` marker and the 2-byte length).
    pub data: Vec<u8>,
    /// Byte offset of the leading `FF` in the original file.
    pub offset: usize,
}

impl AppSegment {
    /// `true` if this is `APP2` with `MPF\0` identifier.
    #[must_use]
    pub fn is_mpf(&self) -> bool {
        self.marker_num == 2 && self.data.starts_with(b"MPF\0")
    }

    /// `true` if this is `APP1` with the XMP namespace identifier.
    #[must_use]
    pub fn is_xmp(&self) -> bool {
        self.marker_num == 1 && self.data.starts_with(b"http://ns.adobe.com/xap/1.0/\0")
    }

    /// `true` if this is `APP1` with the `Exif\0\0` identifier.
    #[must_use]
    pub fn is_exif(&self) -> bool {
        self.marker_num == 1 && self.data.starts_with(b"Exif\0\0")
    }

    /// `true` if this is `APP2` with the `ICC_PROFILE\0` identifier.
    #[must_use]
    pub fn is_icc(&self) -> bool {
        self.marker_num == 2 && self.data.starts_with(b"ICC_PROFILE\0")
    }

    /// `true` if this is `APP0` with the `JFIF\0` identifier.
    #[must_use]
    pub fn is_jfif(&self) -> bool {
        self.marker_num == 0 && self.data.starts_with(b"JFIF\0")
    }
}

fn app_segment_from_span(span: &MarkerSpan<'_>) -> Option<AppSegment> {
    match span.kind {
        MarkerKind::App(n) => Some(AppSegment {
            marker_num: n,
            data: span.payload.to_vec(),
            offset: span.offset,
        }),
        _ => None,
    }
}

/// Byte range of the primary JPEG in a (possibly multi-image) buffer.
///
/// Returns the range `0 .. EOI_inclusive` of the first JPEG's SOI-to-EOI,
/// or `None` if the buffer isn't a valid JPEG. Ultra HDR files concatenate
/// a second JPEG after the primary's EOI; this scan finds only the first.
#[must_use]
pub fn primary_bounds(data: &[u8]) -> Option<Range<usize>> {
    find_jpeg_boundaries(data).into_iter().next()
}

/// Collect all `APPn` segments from a JPEG buffer.
///
/// Walks the length-aware + entropy-aware [`zenjpeg::container::marker::iter`]
/// — does not get confused by `FF xx` byte patterns inside entropy-coded
/// scan data.
#[must_use]
pub fn scan_segments(data: &[u8]) -> Vec<AppSegment> {
    iter(data)
        .filter_map(|span| app_segment_from_span(&span))
        .collect()
}

/// Parse all MPF directory entries from a full JPEG buffer.
///
/// Delegates to [`zenjpeg::container::mpf::parse_mpf`] for the parse and
/// maps its error to [`ultrahdr_core::Error::MpfParse`]. Returns entries
/// in MPF-declared order; entry `[0]` is conventionally the primary image
/// at `offset == 0`, remaining entries carry absolute byte offsets into
/// `data`.
///
/// Returns an empty `Vec` (wrapped in `Ok`) rather than `Err` when no
/// MPF segment is present, so callers can treat "not multi-image" as a
/// non-fatal signal.
pub fn parse_mpf(data: &[u8]) -> Result<Vec<MpfEntry>> {
    use zenjpeg::container::mpf::{MpfError, parse_mpf as zen_parse_mpf};
    match zen_parse_mpf(data) {
        Ok(entries) => Ok(entries),
        Err(MpfError::NotFound) => Ok(Vec::new()),
        Err(e) => Err(Error::MpfParse(e.to_string())),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn app_segment_predicates() {
        let mpf = AppSegment {
            marker_num: 2,
            data: b"MPF\0...tiff...".to_vec(),
            offset: 10,
        };
        assert!(mpf.is_mpf());
        assert!(!mpf.is_icc());
        assert!(!mpf.is_xmp());

        let xmp = AppSegment {
            marker_num: 1,
            data: b"http://ns.adobe.com/xap/1.0/\0<xmp>".to_vec(),
            offset: 10,
        };
        assert!(xmp.is_xmp());
        assert!(!xmp.is_exif());

        let icc = AppSegment {
            marker_num: 2,
            data: b"ICC_PROFILE\0\x01\x01...".to_vec(),
            offset: 10,
        };
        assert!(icc.is_icc());
        assert!(!icc.is_mpf());

        let jfif = AppSegment {
            marker_num: 0,
            data: b"JFIF\0\x01\x01\x00".to_vec(),
            offset: 10,
        };
        assert!(jfif.is_jfif());
    }

    #[test]
    fn scan_segments_empty_jpeg() {
        // Minimal valid JPEG: just SOI + EOI, no APPn.
        let data = [0xFF, 0xD8, 0xFF, 0xD9];
        let segs = scan_segments(&data);
        assert!(segs.is_empty());
    }

    #[test]
    fn scan_segments_app0() {
        // SOI + APP0 (JFIF) + EOI.
        let data: &[u8] = &[
            0xFF, 0xD8, // SOI
            0xFF, 0xE0, 0x00, 0x10, // APP0, length=16
            b'J', b'F', b'I', b'F', 0, 1, 1, 0, 0, 1, 0, 1, 0, 0, 0xFF, 0xD9, // EOI
        ];
        let segs = scan_segments(data);
        assert_eq!(segs.len(), 1);
        assert_eq!(segs[0].marker_num, 0);
        assert!(segs[0].is_jfif());
    }

    #[test]
    fn primary_bounds_minimal() {
        let data = [0xFF, 0xD8, 0xFF, 0xD9];
        let range = primary_bounds(&data).unwrap();
        assert_eq!(range.start, 0);
        assert_eq!(range.end, 4);
    }

    #[test]
    fn primary_bounds_rejects_non_jpeg() {
        assert!(primary_bounds(b"not a jpeg").is_none());
    }

    #[test]
    fn parse_mpf_on_plain_jpeg_returns_empty() {
        // No MPF segment — return empty, not error.
        let data = [0xFF, 0xD8, 0xFF, 0xD9];
        let entries = parse_mpf(&data).unwrap();
        assert!(entries.is_empty());
    }
}
