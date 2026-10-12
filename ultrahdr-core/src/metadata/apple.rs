//! Apple iOS MakerNote HDR gain-map metadata extraction.
//!
//! Apple HEIC/JPEG captures encode their HDR gain-map *headroom* inside the
//! EXIF MakerNote (TIFF tag `0x927C`), **not** in XMP — the embedded gain-map
//! image's XMP carries only `HDRGainMapVersion`. The headroom is derived from
//! two Apple MakerNote tags, per exiftool's `Image::ExifTool::Apple`:
//!
//! | Tag      | Name          | Type          | Role             |
//! |----------|---------------|---------------|------------------|
//! | `0x0021` | `HDRHeadroom` | `rational64s` | "maker33"        |
//! | `0x0030` | `HDRGain`     | `rational64s` | "maker48"        |
//! | `0x000A` | `HDRImageType`| `int32s`      | 3 = HDR, 4 = SDR |
//!
//! The headroom (in stops, i.e. log2 luminance ratio) is computed from
//! `maker33` and `maker48` with the community/Apple-derived piecewise formula
//! (see [`AppleHdrInfo::headroom_stops`]). The exact constants are not in a
//! published Apple specification; final fidelity is validated by the gain-map
//! round-trip test, not asserted to be bit-exact here.
//!
//! References:
//! - exiftool `lib/Image/ExifTool/Apple.pm`
//! - <https://juniperphoton.substack.com/p/decoding-some-hidden-magic-of-makerapple>
//! - <https://photoinvestigator.co/blog/the-mystery-of-maker-apple-metadata/>
//!
//! The shared reader in zenmeta performs byte parsing with `no_std` + `alloc`.
//! These wrappers retain ultrahdr-core's published types and tolerant parsing
//! behavior; strict inspection is available directly from zenmeta.

#[cfg(test)]
use alloc::vec::Vec;

use crate::GainMapMetadata;

/// Apple MakerNote tag IDs (per exiftool `Apple.pm`).
pub mod tags {
    /// `HDRImageType`: 3 = HDR Image, 4 = Original (SDR) Image.
    pub const HDR_IMAGE_TYPE: u16 = zenmeta::apple::tags::HDR_IMAGE_TYPE;
    /// `HDRHeadroom` (`rational64s`) — "maker33" in community notation.
    pub const HDR_HEADROOM: u16 = zenmeta::apple::tags::HDR_HEADROOM;
    /// `HDRGain` (`rational64s`) — "maker48" in community notation.
    pub const HDR_GAIN: u16 = zenmeta::apple::tags::HDR_GAIN;
}

/// Standard EXIF/TIFF tag for the Exif sub-IFD pointer.
#[cfg(test)]
const TIFF_TAG_EXIF_IFD: u16 = 0x8769;
/// Standard EXIF/TIFF tag for the MakerNote (UNDEFINED blob).
#[cfg(test)]
const TIFF_TAG_MAKERNOTE: u16 = 0x927C;

/// HDR information recovered from an Apple MakerNote.
#[derive(Clone, Copy, Debug, PartialEq, Default)]
pub struct AppleHdrInfo {
    /// `0x21 HDRHeadroom` ("maker33"). `None` when the tag is absent — which
    /// means there is no Apple HDR gain-map signal at all.
    pub hdr_headroom: Option<f64>,
    /// `0x30 HDRGain` ("maker48"). Defaults to `0.0` when the tag is absent
    /// (Apple omits it on many captures; the formula treats absent as zero).
    pub hdr_gain: f64,
    /// `0x0a HDRImageType`: `Some(3)` = HDR Image, `Some(4)` = Original/SDR.
    pub hdr_image_type: Option<i32>,
}

impl AppleHdrInfo {
    /// HDR headroom in **stops** (log2 luminance ratio), or `None` when no
    /// `HDRHeadroom` (`0x21`) tag was present.
    ///
    /// Piecewise mapping of `maker33` (`HDRHeadroom`) and `maker48`
    /// (`HDRGain`), clamped to be non-negative:
    ///
    /// ```text
    /// if maker33 < 1:  stops = maker48 <= 0.01 ? -20·maker48 + 1.8  : -0.101·maker48 + 1.601
    /// else:            stops = maker48 <= 0.01 ? -70·maker48 + 3.0  : -0.44 ·maker48 + 2.86
    /// stops = max(stops, 0)
    /// ```
    pub fn headroom_stops(&self) -> Option<f64> {
        zenmeta::apple::compatibility::headroom_stops(&self.as_shared())
    }

    fn as_shared(&self) -> zenmeta::apple::AppleHdrInfo {
        zenmeta::apple::AppleHdrInfo {
            hdr_headroom: self.hdr_headroom,
            hdr_gain: self.hdr_gain,
            hdr_image_type: self.hdr_image_type,
        }
    }

    fn from_shared(info: zenmeta::apple::AppleHdrInfo) -> Self {
        Self {
            hdr_headroom: info.hdr_headroom,
            hdr_gain: info.hdr_gain,
            hdr_image_type: info.hdr_image_type,
        }
    }

    /// Whether `HDRImageType` marks this as the HDR rendition (`3`).
    pub fn is_hdr(&self) -> bool {
        self.hdr_image_type == Some(3)
    }

    /// Whether any Apple HDR gain-map signal is present (the `0x21` tag).
    pub fn has_gain_map(&self) -> bool {
        self.hdr_headroom.is_some()
    }
}

/// Map recovered Apple headroom to the canonical [`GainMapMetadata`]
/// (= `zencodec::GainMapParams`).
///
/// The base image is SDR (`base_hdr_headroom = 0`); the alternate (HDR)
/// headroom is the computed stops value (already log2-domain, matching
/// `GainMapParams`). Per-channel gain spans `[0, stops]` in log2 with unit
/// gamma — the Apple convention where the `[0,1]` gain-map image scales
/// luminance from SDR up to the headroom. Returns `None` when no `0x21`
/// headroom tag is present.
///
/// **Note:** the per-channel curve is the documented Apple convention; the
/// gain-map round-trip test is the authority on fidelity.
pub fn from_apple_headroom(info: &AppleHdrInfo) -> Option<GainMapMetadata> {
    let stops = info.headroom_stops()?;
    // `GainMapParams` is `#[non_exhaustive]`: build via Default + field set.
    let mut params = GainMapMetadata::default();
    params.base_hdr_headroom = 0.0;
    params.alternate_hdr_headroom = stops;
    for ch in &mut params.channels {
        ch.min = 0.0;
        ch.max = stops;
        ch.gamma = 1.0;
    }
    Some(params)
}

/// Extract Apple HDR info directly from EXIF TIFF bytes (e.g. the payload of a
/// HEIF `Exif` item or a JPEG `APP1` segment, with any leading TIFF-offset
/// prefix tolerated).
///
/// Walks IFD0 → Exif sub-IFD (`0x8769`) → MakerNote (`0x927C`) → Apple iOS
/// IFD. Returns `None` if the bytes are not a parseable TIFF, lack an Apple
/// MakerNote, or carry no HDR tags.
pub fn parse_exif_for_apple_hdr(exif: &[u8]) -> Option<AppleHdrInfo> {
    zenmeta::apple::compatibility::parse_exif_for_apple_hdr(exif).map(AppleHdrInfo::from_shared)
}

/// Parse an Apple iOS MakerNote blob (`"Apple iOS\0"` + version + byte-order
/// marker + IFD) and pull out the HDR tags.
///
/// Addressing has two bases (verified against iPhone 8/13/16/17 captures):
/// the byte-order marker and IFD live at offset 12, but the entries' *value
/// offsets* for out-of-line data are relative to the MakerNote start
/// (`maker[0]`). Returns `None` only if the blob is not a recognizable Apple
/// MakerNote; a malformed individual entry is skipped, not fatal.
pub fn parse_apple_makernote(maker: &[u8]) -> Option<AppleHdrInfo> {
    zenmeta::apple::compatibility::parse_apple_makernote(maker).map(AppleHdrInfo::from_shared)
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;

    #[test]
    fn headroom_formula_all_branches() {
        // maker33 >= 1, maker48 <= 0.01  ->  -70*0 + 3.0
        let hi = AppleHdrInfo {
            hdr_headroom: Some(1.686),
            hdr_gain: 0.0,
            hdr_image_type: Some(3),
        };
        assert!((hi.headroom_stops().unwrap() - 3.0).abs() < 1e-9);

        // maker33 >= 1, maker48 > 0.01   ->  -0.44*0.5 + 2.86 = 2.64
        let hi = AppleHdrInfo {
            hdr_headroom: Some(2.0),
            hdr_gain: 0.5,
            hdr_image_type: None,
        };
        assert!((hi.headroom_stops().unwrap() - 2.64).abs() < 1e-9);

        // maker33 < 1, maker48 <= 0.01   ->  -20*0 + 1.8
        let hi = AppleHdrInfo {
            hdr_headroom: Some(0.5),
            hdr_gain: 0.0,
            hdr_image_type: None,
        };
        assert!((hi.headroom_stops().unwrap() - 1.8).abs() < 1e-9);

        // maker33 < 1, maker48 > 0.01    ->  -0.101*0.05 + 1.601
        let hi = AppleHdrInfo {
            hdr_headroom: Some(0.5),
            hdr_gain: 0.05,
            hdr_image_type: None,
        };
        assert!((hi.headroom_stops().unwrap() - (-0.101 * 0.05 + 1.601)).abs() < 1e-9);
    }

    #[test]
    fn no_headroom_tag_means_no_signal() {
        let hi = AppleHdrInfo::default();
        assert_eq!(hi.headroom_stops(), None);
        assert!(!hi.has_gain_map());
        assert_eq!(from_apple_headroom(&hi), None);
    }

    #[test]
    fn maps_to_gain_map_params() {
        let hi = AppleHdrInfo {
            hdr_headroom: Some(1.686),
            hdr_gain: 0.0,
            hdr_image_type: Some(3),
        };
        let p = from_apple_headroom(&hi).unwrap();
        assert!((p.alternate_hdr_headroom - 3.0).abs() < 1e-9);
        assert_eq!(p.base_hdr_headroom, 0.0);
        for ch in &p.channels {
            assert_eq!(ch.min, 0.0);
            assert!((ch.max - 3.0).abs() < 1e-9);
            assert_eq!(ch.gamma, 1.0);
        }
        assert!(p.validate().is_ok());
    }

    /// Build a big-endian Apple MakerNote blob carrying the three HDR tags and
    /// confirm the IFD walk recovers them.
    #[test]
    fn parse_apple_makernote_extracts_hdr_tags() {
        let maker = build_apple_makernote_be(1686, 1000, 0, 1, 3);
        let hi = parse_apple_makernote(&maker).unwrap();
        assert_eq!(hi.hdr_headroom, Some(1.686));
        assert_eq!(hi.hdr_gain, 0.0);
        assert_eq!(hi.hdr_image_type, Some(3));
        assert!(hi.is_hdr());
        assert!((hi.headroom_stops().unwrap() - 3.0).abs() < 1e-9);
    }

    /// Wrap the MakerNote in a full EXIF TIFF (IFD0 → Exif IFD → MakerNote)
    /// and confirm the top-level walk recovers the HDR info.
    #[test]
    fn parse_full_exif_walk() {
        let exif = build_exif_with_apple_makernote();
        let hi = parse_exif_for_apple_hdr(&exif).unwrap();
        assert_eq!(hi.hdr_headroom, Some(1.686));
        assert_eq!(hi.hdr_image_type, Some(3));
    }

    #[test]
    fn rejects_non_apple_makernote() {
        assert_eq!(parse_apple_makernote(b"Nikon\0\0\0not apple here!!"), None);
        assert_eq!(parse_apple_makernote(b"short"), None);
    }

    // ── test fixtures: hand-built big-endian TIFF/MakerNote ───────────────

    fn be16(v: u16) -> [u8; 2] {
        v.to_be_bytes()
    }
    fn be32(v: u32) -> [u8; 4] {
        v.to_be_bytes()
    }

    /// One IFD entry: tag, type, count=1, and either an inline value (≤4 B,
    /// left-justified) or a 4-byte offset into the out-of-line area.
    fn ifd_entry(tag: u16, dtype: u16, count: u32, inline_or_off: [u8; 4]) -> Vec<u8> {
        let mut e = Vec::new();
        e.extend_from_slice(&be16(tag));
        e.extend_from_slice(&be16(dtype));
        e.extend_from_slice(&be32(count));
        e.extend_from_slice(&inline_or_off);
        e
    }

    /// Build an Apple iOS MakerNote (big-endian, custom layout) with
    /// HDRHeadroom (0x21, srational), HDRGain (0x30, srational), and
    /// HDRImageType (0x0a, short).
    fn build_apple_makernote_be(
        hr_num: i32,
        hr_den: i32,
        g_num: i32,
        g_den: i32,
        img_type: u16,
    ) -> Vec<u8> {
        // Header: "Apple iOS\0" + version(2). The byte-order marker "MM" is the
        // first 2 bytes of `tiff` (appended below) and thus lands at offset 12.
        let mut blob = Vec::new();
        blob.extend_from_slice(b"Apple iOS\0");
        blob.extend_from_slice(&be16(14)); // version

        // From here offsets are relative to offset 12 (the "MM").
        // Custom Apple layout: entry count starts at byte 2 of `tiff`.
        // tiff[0..2] = "MM"; tiff[2..] = count + entries.
        // Two srationals go out-of-line; lay them after the entries.
        let n_entries: u16 = 3;
        // tiff offsets: 0:MM, 2:count, 4:entries(3*12=36) -> 40, then ool data.
        let ool_tiff = 2 + 2 + (n_entries as usize) * 12; // = 40 (within tiff)
        // Out-of-line offsets are relative to the MakerNote start (maker[0]);
        // tiff begins at maker[12], so add 12.
        let hr_off = (12 + ool_tiff) as u32; // headroom srational (8 B)
        let g_off = (12 + ool_tiff + 8) as u32; // gain srational (8 B)

        let mut tiff = Vec::new();
        tiff.extend_from_slice(b"MM");
        tiff.extend_from_slice(&be16(n_entries));
        // HDRImageType (short) inline, left-justified in the 4-byte value cell.
        let mut img_inline = [0u8; 4];
        img_inline[..2].copy_from_slice(&be16(img_type));
        tiff.extend_from_slice(&ifd_entry(tags::HDR_IMAGE_TYPE, 3, 1, img_inline));
        // HDRHeadroom srational, out-of-line.
        tiff.extend_from_slice(&ifd_entry(tags::HDR_HEADROOM, 10, 1, be32(hr_off)));
        // HDRGain srational, out-of-line.
        tiff.extend_from_slice(&ifd_entry(tags::HDR_GAIN, 10, 1, be32(g_off)));
        // Out-of-line rationals.
        tiff.extend_from_slice(&be32(hr_num as u32));
        tiff.extend_from_slice(&be32(hr_den as u32));
        tiff.extend_from_slice(&be32(g_num as u32));
        tiff.extend_from_slice(&be32(g_den as u32));

        blob.extend_from_slice(&tiff);
        blob
    }

    /// Build a minimal EXIF TIFF whose IFD0 points to an Exif sub-IFD that
    /// holds the Apple MakerNote.
    fn build_exif_with_apple_makernote() -> Vec<u8> {
        let maker = build_apple_makernote_be(1686, 1000, 0, 1, 3);

        // Layout (all offsets relative to the TIFF header start "MM\0*"):
        //   0  : "MM\0*"  (4)
        //   4  : ifd0_offset = 8  (4)
        //   8  : IFD0: count=1 (2) + 1 entry (12) + next=0 (4)  -> ends at 26
        //   26 : Exif IFD: count=1 (2) + 1 entry (12) + next=0 (4) -> ends at 44
        //   44 : MakerNote bytes
        let exif_ifd_off: u32 = 26;
        let maker_off: u32 = 44;

        let mut t = Vec::new();
        t.extend_from_slice(b"MM\x00\x2a"); // big-endian TIFF header
        t.extend_from_slice(&be32(8)); // IFD0 at offset 8

        // IFD0: one entry — Exif sub-IFD pointer (type LONG).
        t.extend_from_slice(&be16(1));
        t.extend_from_slice(&ifd_entry(TIFF_TAG_EXIF_IFD, 4, 1, be32(exif_ifd_off)));
        t.extend_from_slice(&be32(0)); // next IFD = none  (offset 22..26)

        // Exif IFD: one entry — MakerNote (UNDEFINED, count = maker.len()).
        t.extend_from_slice(&be16(1));
        t.extend_from_slice(&ifd_entry(
            TIFF_TAG_MAKERNOTE,
            7,
            maker.len() as u32,
            be32(maker_off),
        ));
        t.extend_from_slice(&be32(0)); // next IFD = none

        debug_assert_eq!(t.len() as u32, maker_off);
        t.extend_from_slice(&maker);
        t
    }

    #[test]
    fn tolerates_leading_tiff_offset_prefix() {
        // HEIF Exif items prepend a 4-byte offset before the TIFF header.
        let mut exif = vec![0u8, 0, 0, 0];
        exif.extend_from_slice(&build_exif_with_apple_makernote());
        let hi = parse_exif_for_apple_hdr(&exif).unwrap();
        assert_eq!(hi.hdr_headroom, Some(1.686));
    }
}
