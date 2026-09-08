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

#[cfg(test)]
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
        if !maker33.is_finite() || !maker48.is_finite() {
            return None;
        }
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
    let (kind, count, exif_ptr) = ifd_find(tiff, endian, ifd0_off, TIFF_TAG_EXIF_IFD)?;
    if kind != 4 || count != 1 {
        return None;
    }
    let exif_ifd_off = endian.u32(exif_ptr, 0)? as usize;
    // Find the MakerNote blob in the Exif sub-IFD.
    let (kind, _, maker) = ifd_find(tiff, endian, exif_ifd_off, TIFF_TAG_MAKERNOTE)?;
    if kind != 7 {
        return None;
    }
    parse_apple_makernote(maker)
}

/// One borrowed vendor entry, including unknown/private tags.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MakerNoteEntry<'a> {
    tag: u16,
    kind: u16,
    count: u32,
    value: &'a [u8],
}
impl<'a> MakerNoteEntry<'a> {
    /// Numeric vendor tag, including unknown tags.
    pub fn tag(&self) -> u16 {
        self.tag
    }
    /// TIFF field type.
    pub fn kind(&self) -> u16 {
        self.kind
    }
    /// Number of typed values.
    pub fn count(&self) -> u32 {
        self.count
    }
    /// Raw value bytes in the MakerNote byte order.
    pub fn value(&self) -> &'a [u8] {
        self.value
    }
}
/// An entry that could not be read. Inspection does not silently hide it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub struct MakerNoteEntryError {
    /// Zero-based entry index in the original note.
    pub index: usize,
    /// Numeric tag when the entry header is readable.
    pub tag: Option<u16>,
}
/// Apple MakerNote inspection view. Other vendors remain opaque.
/// No allocation or value copies; unknown entries are visible for audit/diff.
#[derive(Debug)]
pub struct AppleMakerNote<'a> {
    data: &'a [u8],
    endian: Endian,
    entries: usize,
    count: usize,
}
impl<'a> AppleMakerNote<'a> {
    /// Recognize an Apple note, bounded to 16 MiB and 4096 entries.
    pub fn parse(data: &'a [u8]) -> Option<Self> {
        if data.len() > 16 * 1024 * 1024 || data.get(..10)? != b"Apple iOS\0" {
            return None;
        }
        let endian = match data.get(12..14)? {
            b"MM" => Endian::Big,
            b"II" => Endian::Little,
            _ => return None,
        };
        let ifd = if endian.u16(data, 14)? == 42 {
            12usize.checked_add(endian.u32(data, 16)? as usize)?
        } else {
            14
        };
        let count = endian.u16(data, ifd)? as usize;
        if count > 4096 {
            return None;
        }
        Some(Self {
            data,
            endian,
            entries: ifd.checked_add(2)?,
            count,
        })
    }
    /// Byte order for interpreting numeric value bytes.
    pub fn byte_order(&self) -> zencodec::exif::ByteOrder {
        match self.endian {
            Endian::Big => zencodec::exif::ByteOrder::Big,
            Endian::Little => zencodec::exif::ByteOrder::Little,
        }
    }
    /// Borrow all entries in order, reporting malformed entries individually.
    pub fn entries(
        &self,
    ) -> impl Iterator<Item = core::result::Result<MakerNoteEntry<'a>, MakerNoteEntryError>> + '_
    {
        (0..self.count).map(|index| {
            let offset = self.entries + index * 12;
            let tag = self.endian.u16(self.data, offset);
            let error = MakerNoteEntryError { index, tag };
            let tag = tag.ok_or(error)?;
            let kind = self.endian.u16(self.data, offset + 2).ok_or(error)?;
            let count = self.endian.u32(self.data, offset + 4).ok_or(error)?;
            if !(1..=13).contains(&kind) && !(16..=18).contains(&kind) {
                return Err(error);
            }
            let size = type_size(kind).checked_mul(count as usize).ok_or(error)?;
            let start = if size <= 4 {
                offset + 8
            } else {
                self.endian.u32(self.data, offset + 8).ok_or(error)? as usize
            };
            let value = self
                .data
                .get(start..start.checked_add(size).ok_or(error)?)
                .ok_or(error)?;
            Ok(MakerNoteEntry {
                tag,
                kind,
                count,
                value,
            })
        })
    }
}

/// Parse an Apple iOS MakerNote blob (`"Apple iOS\0"` + version + byte-order
/// marker + IFD) and pull out the HDR tags.
///
/// Addressing has two bases (verified against iPhone 8/13/16/17 captures):
/// the byte-order marker and IFD live at offset 12, but the entries' *value
/// offsets* for out-of-line data are relative to the MakerNote start
/// (`maker[0]`). Unrecognized notes, malformed entries or duplicate rendering
/// tags refuse extraction. The inspection view reports malformed entries individually.
pub fn parse_apple_makernote(maker: &[u8]) -> Option<AppleHdrInfo> {
    let view = AppleMakerNote::parse(maker)?;
    let mut info = AppleHdrInfo::default();
    let mut seen = 0u8;
    for entry in view.entries() {
        let entry = entry.ok()?;
        let bit = match entry.tag {
            tags::HDR_HEADROOM => 1,
            tags::HDR_GAIN => 2,
            tags::HDR_IMAGE_TYPE => 4,
            _ => continue,
        };
        if seen & bit != 0 || entry.count != 1 {
            return None;
        }
        seen |= bit;
        match entry.tag {
            tags::HDR_HEADROOM => {
                info.hdr_headroom = Some(read_rational(entry.value, entry.kind, view.endian)?)
            }
            tags::HDR_GAIN => info.hdr_gain = read_rational(entry.value, entry.kind, view.endian)?,
            tags::HDR_IMAGE_TYPE => {
                info.hdr_image_type = Some(read_int(entry.value, entry.kind, view.endian)?)
            }
            _ => unreachable!(),
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
        let s = b.get(o..o.checked_add(2)?)?;
        let a = [s[0], s[1]];
        Some(match self {
            Endian::Big => u16::from_be_bytes(a),
            Endian::Little => u16::from_le_bytes(a),
        })
    }

    fn u32(self, b: &[u8], o: usize) -> Option<u32> {
        let s = b.get(o..o.checked_add(4)?)?;
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
        1 | 2 | 6 | 7 => 1,         // BYTE, ASCII, SBYTE, UNDEFINED
        3 | 8 => 2,                 // SHORT, SSHORT
        4 | 9 | 11 | 13 => 4,       // LONG, SLONG, FLOAT
        5 | 10 | 12 | 16..=18 => 8, // RATIONAL, SRATIONAL, DOUBLE
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
fn ifd_find(tiff: &[u8], endian: Endian, ifd_off: usize, want: u16) -> Option<(u16, u32, &[u8])> {
    let count = endian.u16(tiff, ifd_off)? as usize;
    let entries = ifd_off.checked_add(2)?;
    let mut result = None;
    for i in 0..count {
        let e = entries + i * 12;
        if e + 12 > tiff.len() {
            return None;
        }
        if endian.u16(tiff, e)? != want {
            continue;
        }
        let dtype = endian.u16(tiff, e + 2)?;
        let n = endian.u32(tiff, e + 4)?;
        let total = type_size(dtype).checked_mul(n as usize)?;
        let value = if total <= 4 {
            tiff.get(e + 8..e + 8 + total.min(4))?
        } else {
            let off = endian.u32(tiff, e + 8)? as usize;
            tiff.get(off..off.checked_add(total)?)?
        };
        if result.replace((dtype, n, value)).is_some() {
            return None;
        }
    }
    result
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
        3 => endian.u16(value, 0).map(|v| v as i32),
        8 => endian.u16(value, 0).map(|v| v as i16 as i32),
        4 | 9 => endian.i32(value, 0),
        1 => value.first().map(|&b| b as i32),
        6 => value.first().map(|&b| b as i8 as i32),
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

    #[test]
    fn inspect_unknown_entries_and_report_truncation() {
        let mut bytes = build_apple_makernote_be(1686, 1000, 0, 1, 3);
        let view = AppleMakerNote::parse(&bytes).unwrap();
        assert_eq!(view.entries().count(), 3);
        assert!(view.entries().all(|e| e.is_ok()));
        // First entry becomes an unknown, still inspectable vendor field.
        bytes[16..18].copy_from_slice(&0xf123u16.to_be_bytes());
        let view = AppleMakerNote::parse(&bytes).unwrap();
        assert_eq!(view.entries().next().unwrap().unwrap().tag(), 0xf123);
        let truncated = &bytes[..24];
        let view = AppleMakerNote::parse(truncated).unwrap();
        assert!(view.entries().any(|e| e.is_err()));
    }
    #[test]
    fn invalid_critical_rational_refuses_instead_of_guessing() {
        let bytes = build_apple_makernote_be(1686, 0, 0, 1, 3);
        assert!(parse_apple_makernote(&bytes).is_none());
        assert!(
            AppleHdrInfo {
                hdr_headroom: Some(f64::NAN),
                ..Default::default()
            }
            .headroom_stops()
            .is_none()
        );
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
