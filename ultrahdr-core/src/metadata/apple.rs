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
//! This module is pure byte parsing: `no_std` + `alloc`, no transcendental
//! math, zero new dependencies. The IFD-walk mechanics mirror the proven
//! reader in `zenraw::apple`, with the corrected exiftool tag IDs.

use alloc::vec::Vec;

use crate::GainMapMetadata;

/// Apple MakerNote tag IDs (per exiftool `Apple.pm`).
pub mod tags {
    /// `HDRImageType`: 3 = HDR Image, 4 = Original (SDR) Image.
    pub const HDR_IMAGE_TYPE: u16 = 0x000A;
    /// `HDRHeadroom` (`rational64s`) — "maker33" in community notation.
    pub const HDR_HEADROOM: u16 = 0x0021;
    /// `HDRGain` (`rational64s`) — "maker48" in community notation.
    pub const HDR_GAIN: u16 = 0x0030;
}

/// Standard EXIF/TIFF tag for the Exif sub-IFD pointer.
const TIFF_TAG_EXIF_IFD: u16 = 0x8769;
/// Standard EXIF/TIFF tag for the MakerNote (UNDEFINED blob).
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
        let maker33 = self.hdr_headroom?;
        let maker48 = self.hdr_gain;
        let stops = if maker33 < 1.0 {
            if maker48 <= 0.01 {
                -20.0 * maker48 + 1.8
            } else {
                -0.101 * maker48 + 1.601
            }
        } else if maker48 <= 0.01 {
            -70.0 * maker48 + 3.0
        } else {
            -0.44 * maker48 + 2.86
        };
        // Clamp to non-negative without pulling in `f64::max` (std-only in no_std).
        Some(if stops > 0.0 { stops } else { 0.0 })
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
    let (tiff, endian) = tiff_start(exif)?;
    // IFD0 offset is the u32 at byte 4 of the TIFF header.
    let ifd0_off = endian.u32(tiff, 4)? as usize;
    // Find the Exif sub-IFD pointer in IFD0.
    let (_, _, exif_ptr) = ifd_find(tiff, endian, ifd0_off, TIFF_TAG_EXIF_IFD)?;
    let exif_ifd_off = endian.u32(&exif_ptr, 0)? as usize;
    // Find the MakerNote blob in the Exif sub-IFD.
    let (_, _, maker) = ifd_find(tiff, endian, exif_ifd_off, TIFF_TAG_MAKERNOTE)?;
    parse_apple_makernote(&maker)
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
    if maker.len() < 20 || &maker[..10] != b"Apple iOS\0" {
        return None;
    }
    let bo = 12;
    let endian = match (maker[bo], maker[bo + 1]) {
        (b'M', b'M') => Endian::Big,
        (b'I', b'I') => Endian::Little,
        _ => return None,
    };
    // IFD data is relative to the byte-order marker.
    let tiff = &maker[bo..];
    // Apple uses either a standard TIFF magic (42 → IFD offset at byte 4) or a
    // custom layout where the entry count begins at byte 2.
    let ifd_off = if endian.u16(tiff, 2)? == 42 {
        endian.u32(tiff, 4)? as usize
    } else {
        2
    };

    let mut info = AppleHdrInfo::default();
    let count = endian.u16(tiff, ifd_off)? as usize;
    let entries = ifd_off + 2;
    // Walk entries defensively: a single malformed entry (Apple ships some with
    // invalid TIFF types, e.g. format 16) must not abort the whole parse. We
    // only read the value for the three HDR tags and skip everything else.
    for i in 0..count {
        let e = entries + i * 12;
        if e + 12 > tiff.len() {
            break;
        }
        let Some(tag) = endian.u16(tiff, e) else {
            break;
        };
        if tag != tags::HDR_HEADROOM && tag != tags::HDR_GAIN && tag != tags::HDR_IMAGE_TYPE {
            continue;
        }
        let (Some(dtype), Some(n)) = (endian.u16(tiff, e + 2), endian.u32(tiff, e + 4)) else {
            continue;
        };
        let total = type_size(dtype).saturating_mul(n as usize);
        let value = if total <= 4 {
            // Inline value lives in the entry's 4-byte value cell.
            match tiff.get(e + 8..e + 8 + total.min(4)) {
                Some(v) => v.to_vec(),
                None => continue,
            }
        } else {
            // Out-of-line: the offset is relative to the MakerNote start
            // (`maker[0]`, the "Apple iOS\0" byte), not the byte-order marker
            // at offset 12. (Verified against iPhone 8/13/16/17 captures.)
            let Some(off) = endian.u32(tiff, e + 8) else {
                continue;
            };
            match maker.get(off as usize..(off as usize).saturating_add(total)) {
                Some(v) => v.to_vec(),
                None => continue,
            }
        };
        match tag {
            tags::HDR_HEADROOM => info.hdr_headroom = read_rational(&value, dtype, endian),
            tags::HDR_GAIN => {
                if let Some(g) = read_rational(&value, dtype, endian) {
                    info.hdr_gain = g;
                }
            }
            tags::HDR_IMAGE_TYPE => info.hdr_image_type = read_int(&value, dtype, endian),
            _ => {}
        }
    }
    Some(info)
}

// ── TIFF/IFD primitives ──────────────────────────────────────────────────

/// Endianness of a TIFF/IFD structure.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Endian {
    Big,
    Little,
}

impl Endian {
    fn u16(self, b: &[u8], o: usize) -> Option<u16> {
        let s = b.get(o..o + 2)?;
        let a = [s[0], s[1]];
        Some(match self {
            Endian::Big => u16::from_be_bytes(a),
            Endian::Little => u16::from_le_bytes(a),
        })
    }

    fn u32(self, b: &[u8], o: usize) -> Option<u32> {
        let s = b.get(o..o + 4)?;
        let a = [s[0], s[1], s[2], s[3]];
        Some(match self {
            Endian::Big => u32::from_be_bytes(a),
            Endian::Little => u32::from_le_bytes(a),
        })
    }

    fn i32(self, b: &[u8], o: usize) -> Option<i32> {
        self.u32(b, o).map(|v| v as i32)
    }
}

/// TIFF data type → element size in bytes (TIFF 6.0 + BigTIFF extensions used
/// by EXIF). Unknown types fall back to 1 to stay within bounds.
fn type_size(dtype: u16) -> usize {
    match dtype {
        1 | 2 | 6 | 7 => 1, // BYTE, ASCII, SBYTE, UNDEFINED
        3 | 8 => 2,         // SHORT, SSHORT
        4 | 9 | 11 => 4,    // LONG, SLONG, FLOAT
        5 | 10 | 12 => 8,   // RATIONAL, SRATIONAL, DOUBLE
        _ => 1,
    }
}

/// Locate the TIFF header within an EXIF blob, tolerating a leading offset
/// prefix (HEIF `Exif` items prepend a 4-byte `tiff_header_offset`). Returns
/// the TIFF slice (starting at the byte-order marker) and its endianness.
fn tiff_start(exif: &[u8]) -> Option<(&[u8], Endian)> {
    let window = exif.len().min(16);
    for base in 0..window {
        match exif.get(base..base + 4) {
            Some(b"II\x2a\x00") => return Some((&exif[base..], Endian::Little)),
            Some(b"MM\x00\x2a") => return Some((&exif[base..], Endian::Big)),
            _ => {}
        }
    }
    None
}

/// Find an IFD entry by tag and return `(dtype, count, value_bytes)`. Value
/// bytes are read inline (≤ 4 bytes) or from the pointed-to offset (relative
/// to the TIFF start). `None` if the tag is absent or the data is truncated.
fn ifd_find(tiff: &[u8], endian: Endian, ifd_off: usize, want: u16) -> Option<(u16, u32, Vec<u8>)> {
    let count = endian.u16(tiff, ifd_off)? as usize;
    let entries = ifd_off + 2;
    for i in 0..count {
        let e = entries + i * 12;
        if e + 12 > tiff.len() {
            break;
        }
        if endian.u16(tiff, e)? != want {
            continue;
        }
        let dtype = endian.u16(tiff, e + 2)?;
        let n = endian.u32(tiff, e + 4)?;
        let total = type_size(dtype) * n as usize;
        let value = if total <= 4 {
            tiff.get(e + 8..e + 8 + total.min(4))?.to_vec()
        } else {
            let off = endian.u32(tiff, e + 8)? as usize;
            tiff.get(off..off + total)?.to_vec()
        };
        return Some((dtype, n, value));
    }
    None
}

/// Read a rational (`5` = unsigned, `10` = signed) as `f64`. Apple writes
/// `HDRHeadroom`/`HDRGain` as `rational64s` (type 10). Returns `None` on a
/// zero denominator or short data.
fn read_rational(value: &[u8], dtype: u16, endian: Endian) -> Option<f64> {
    if value.len() < 8 {
        return None;
    }
    match dtype {
        10 => {
            let num = endian.i32(value, 0)?;
            let den = endian.i32(value, 4)?;
            if den == 0 {
                None
            } else {
                Some(num as f64 / den as f64)
            }
        }
        5 => {
            let num = endian.u32(value, 0)?;
            let den = endian.u32(value, 4)?;
            if den == 0 {
                None
            } else {
                Some(num as f64 / den as f64)
            }
        }
        _ => None,
    }
}

/// Read a small integer tag (`HDRImageType` is `int32s`/SHORT) as `i32`.
fn read_int(value: &[u8], dtype: u16, endian: Endian) -> Option<i32> {
    match dtype {
        3 | 8 => endian.u16(value, 0).map(|v| v as i32),
        4 | 9 => endian.i32(value, 0),
        1 | 6 => value.first().map(|&b| b as i32),
        _ => None,
    }
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
