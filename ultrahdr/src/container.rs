//! JPEG container utilities for codec-agnostic Ultra HDR support.
//!
//! This module provides low-level JPEG container manipulation for codecs that
//! don't natively support MPF (Multi-Picture Format) or segment preservation.
//!
//! # API Levels
//!
//! - **Level 1**: Full codec support (like jpegli-rs) - use `DecodedExtras`/`EncoderSegments`
//! - **Level 2**: Codec provides APP segments - use these functions with segment data
//! - **Level 3**: Codec blind to segments - use `scan_segments` to extract them
//!
//! # Usage
//!
//! ```ignore
//! use ultrahdr::container::{scan_segments, parse_mpf_segment, extract_secondary_images, assemble};
//!
//! // Level 3: Scan raw bytes for segments
//! let segments = scan_segments(&jpeg_bytes);
//!
//! // Find and parse MPF
//! let mpf_segment = segments.iter().find(|s| s.is_mpf()).unwrap();
//! let mpf = parse_mpf_segment(&mpf_segment.data)?;
//!
//! // Extract secondary images (gain map, etc.)
//! let secondaries = extract_secondary_images(&jpeg_bytes, &mpf);
//!
//! // For encoding: assemble primary + secondaries into multi-image JPEG
//! let output = assemble(&primary_jpeg, &[&gainmap_jpeg], &[MpfImageType::GainMap]);
//! ```

use ultrahdr_core::{Error, Result};
use std::ops::Range;

/// MPF (Multi-Picture Format) directory parsed from APP2 segment.
#[derive(Debug, Clone)]
pub struct MpfDirectory {
    /// Image entries in the MPF directory.
    pub entries: Vec<MpfEntry>,
    /// Offset of the MPF marker in the original file (needed for offset calculations).
    pub mpf_marker_offset: usize,
}

/// A single entry in the MPF directory.
#[derive(Debug, Clone, Copy)]
pub struct MpfEntry {
    /// Image type flags.
    pub image_type: MpfImageType,
    /// Image size in bytes.
    pub size: u32,
    /// Offset from MPF marker (0 for primary image).
    pub offset: u32,
    /// Entry index in the directory.
    pub index: u32,
}

/// MPF image type flags.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MpfImageType {
    /// Primary baseline image.
    Primary,
    /// Large thumbnail (full-size alternative).
    LargeThumbnail,
    /// Multi-frame panorama component.
    MultiFramePanorama,
    /// Multi-frame disparity (stereo).
    MultiFrameDisparity,
    /// Multi-frame multi-angle.
    MultiFrameMultiAngle,
    /// Gain map for HDR reconstruction.
    GainMap,
    /// Depth map.
    DepthMap,
    /// Unknown/other type.
    Unknown(u32),
}

impl MpfImageType {
    /// Convert from MPF attribute flags.
    ///
    /// MPF attribute format (4 bytes):
    /// - Bits 31: Dependent image flag
    /// - Bits 30: Representative image flag
    /// - Bits 29-27: Reserved
    /// - Bits 26-24: Image type
    /// - Bits 23-16: MP Format (0x03 = JPEG)
    /// - Bits 15-0: Reserved
    pub fn from_attribute(attr: u32) -> Self {
        // Ultra HDR uses 0x030000 for primary, 0x000000 for dependent
        match attr {
            0x03_0000 => MpfImageType::Primary,
            0x00_0000 => MpfImageType::GainMap, // Default dependent type
            _ => {
                // Try to decode based on known patterns
                let type_code = (attr >> 24) & 0x07;
                match type_code {
                    0 => {
                        if attr == 0 {
                            MpfImageType::GainMap
                        } else {
                            MpfImageType::Primary
                        }
                    }
                    1 => MpfImageType::LargeThumbnail,
                    2 => MpfImageType::MultiFramePanorama,
                    3 => MpfImageType::MultiFrameDisparity,
                    4 => MpfImageType::MultiFrameMultiAngle,
                    _ => MpfImageType::Unknown(attr),
                }
            }
        }
    }

    /// Convert to MPF attribute flags.
    pub fn to_attribute(self) -> u32 {
        match self {
            // Baseline MP primary image (matches existing mpf.rs)
            MpfImageType::Primary => 0x03_0000,
            // Dependent child image (matches existing mpf.rs)
            MpfImageType::GainMap => 0x00_0000,
            MpfImageType::DepthMap => 0x00_0000,
            MpfImageType::LargeThumbnail => 0x01_0001,
            MpfImageType::MultiFramePanorama => 0x02_0002,
            MpfImageType::MultiFrameDisparity => 0x03_0003,
            MpfImageType::MultiFrameMultiAngle => 0x04_0004,
            MpfImageType::Unknown(attr) => attr,
        }
    }
}

/// An APP segment extracted from a JPEG.
#[derive(Debug, Clone)]
pub struct AppSegment {
    /// Marker number (0-15 for APP0-APP15).
    pub marker_num: u8,
    /// Segment data (excluding marker and length bytes).
    pub data: Vec<u8>,
    /// Offset in the original file.
    pub offset: usize,
}

impl AppSegment {
    /// Check if this is an MPF segment (APP2 with "MPF\0" identifier).
    pub fn is_mpf(&self) -> bool {
        self.marker_num == 2 && self.data.starts_with(b"MPF\0")
    }

    /// Check if this is an XMP segment (APP1 with XMP namespace).
    pub fn is_xmp(&self) -> bool {
        self.marker_num == 1 && self.data.starts_with(b"http://ns.adobe.com/xap/1.0/\0")
    }

    /// Check if this is an EXIF segment (APP1 with "Exif\0\0").
    pub fn is_exif(&self) -> bool {
        self.marker_num == 1 && self.data.starts_with(b"Exif\0\0")
    }

    /// Check if this is an ICC profile segment (APP2 with "ICC_PROFILE\0").
    pub fn is_icc(&self) -> bool {
        self.marker_num == 2 && self.data.starts_with(b"ICC_PROFILE\0")
    }

    /// Check if this is a JFIF segment (APP0 with "JFIF\0").
    pub fn is_jfif(&self) -> bool {
        self.marker_num == 0 && self.data.starts_with(b"JFIF\0")
    }
}

/// Find the bounds of the primary JPEG image (SOI to first EOI).
///
/// Returns the byte range of the primary image, or None if not a valid JPEG.
///
/// # Example
///
/// ```ignore
/// let bounds = primary_bounds(&multi_image_jpeg)?;
/// let primary = &multi_image_jpeg[bounds];
/// ```
pub fn primary_bounds(data: &[u8]) -> Option<Range<usize>> {
    // Check for SOI
    if data.len() < 4 || data[0] != 0xFF || data[1] != 0xD8 {
        return None;
    }

    // Scan for EOI
    let mut pos = 2;
    while pos < data.len() - 1 {
        if data[pos] == 0xFF && data[pos + 1] == 0xD9 {
            return Some(0..pos + 2);
        }

        // Skip to next marker
        if data[pos] == 0xFF {
            let marker = data[pos + 1];

            // Markers without length
            if marker == 0x00 || marker == 0x01 || (0xD0..=0xD9).contains(&marker) || marker == 0xFF
            {
                pos += 2;
                continue;
            }

            // Marker with length
            if pos + 4 <= data.len() {
                let len = u16::from_be_bytes([data[pos + 2], data[pos + 3]]) as usize;
                if len >= 2 {
                    pos += 2 + len;
                    continue;
                }
            }
        }

        pos += 1;
    }

    None
}

/// Scan a JPEG for APP segments.
///
/// This is the Level 3 API - use when your codec doesn't expose segments.
///
/// # Returns
///
/// Vector of APP segments found in the JPEG, in order.
pub fn scan_segments(data: &[u8]) -> Vec<AppSegment> {
    let mut segments = Vec::new();

    // Check for SOI
    if data.len() < 4 || data[0] != 0xFF || data[1] != 0xD8 {
        return segments;
    }

    let mut pos = 2;

    while pos < data.len() - 3 {
        // Find marker
        if data[pos] != 0xFF {
            pos += 1;
            continue;
        }

        // Skip padding FF bytes
        while pos < data.len() - 1 && data[pos + 1] == 0xFF {
            pos += 1;
        }

        if pos >= data.len() - 1 {
            break;
        }

        let marker = data[pos + 1];
        let offset = pos;

        // Stop at SOS (start of scan) - no more APP segments after this
        if marker == 0xDA {
            break;
        }

        // Skip markers without length (SOI, EOI, RST0-RST7, TEM)
        if marker == 0xD8
            || marker == 0xD9
            || (0xD0..=0xD7).contains(&marker)
            || marker == 0x01
            || marker == 0x00
        {
            pos += 2;
            continue;
        }

        // Read length
        if pos + 4 > data.len() {
            break;
        }

        let length = u16::from_be_bytes([data[pos + 2], data[pos + 3]]) as usize;
        if length < 2 || pos + 2 + length > data.len() {
            break;
        }

        // Check if this is an APP marker (0xE0-0xEF)
        if (0xE0..=0xEF).contains(&marker) {
            let marker_num = marker - 0xE0;
            let segment_data = data[pos + 4..pos + 2 + length].to_vec();

            segments.push(AppSegment {
                marker_num,
                data: segment_data,
                offset,
            });
        }

        pos += 2 + length;
    }

    segments
}

/// Parse an MPF directory from APP2 segment data.
///
/// The data should be the segment content *after* the "MPF\0" identifier,
/// or the full segment data (identifier will be skipped if present).
///
/// # Arguments
///
/// * `data` - The APP2 segment data
/// * `mpf_marker_offset` - Offset of the MPF marker in the original file
pub fn parse_mpf_segment(data: &[u8], mpf_marker_offset: usize) -> Result<MpfDirectory> {
    // Skip "MPF\0" identifier if present
    let mpf_data = if data.starts_with(b"MPF\0") {
        &data[4..]
    } else {
        data
    };

    if mpf_data.len() < 8 {
        return Err(Error::MpfParse("MPF data too short".into()));
    }

    // Check endianness
    let big_endian = &mpf_data[0..2] == b"MM";
    if !big_endian && &mpf_data[0..2] != b"II" {
        return Err(Error::MpfParse("Invalid MPF endianness marker".into()));
    }

    // Read IFD offset
    let ifd_offset = read_u32(mpf_data, 4, big_endian) as usize;

    if ifd_offset + 2 > mpf_data.len() {
        return Err(Error::MpfParse("Invalid IFD offset".into()));
    }

    // Read number of IFD entries
    let num_entries = read_u16(mpf_data, ifd_offset, big_endian) as usize;

    let mut mp_entry_offset = 0usize;
    let mut mp_entry_count = 0u32;

    // Parse IFD entries to find MP Entry tag
    let entry_start = ifd_offset + 2;
    for i in 0..num_entries {
        let offset = entry_start + i * 12;
        if offset + 12 > mpf_data.len() {
            break;
        }

        let tag = read_u16(mpf_data, offset, big_endian);
        let value_offset = read_u32(mpf_data, offset + 8, big_endian);

        match tag {
            0xB001 => {
                // Number of images
                mp_entry_count = value_offset;
            }
            0xB002 => {
                // MP Entry offset
                mp_entry_offset = value_offset as usize;
            }
            _ => {}
        }
    }

    // Parse MP Entry array
    let mut entries = Vec::with_capacity(mp_entry_count as usize);

    if mp_entry_offset > 0 && mp_entry_count > 0 {
        for i in 0..mp_entry_count {
            let entry_pos = mp_entry_offset + (i as usize) * 16;
            if entry_pos + 16 > mpf_data.len() {
                break;
            }

            // Attribute (4 bytes)
            let attr = read_u32(mpf_data, entry_pos, big_endian);

            // Size (4 bytes)
            let size = read_u32(mpf_data, entry_pos + 4, big_endian);

            // Offset (4 bytes)
            let offset = read_u32(mpf_data, entry_pos + 8, big_endian);

            entries.push(MpfEntry {
                image_type: MpfImageType::from_attribute(attr),
                size,
                offset,
                index: i,
            });
        }
    }

    if entries.is_empty() {
        return Err(Error::MpfParse("No images found in MPF".into()));
    }

    Ok(MpfDirectory {
        entries,
        mpf_marker_offset,
    })
}

/// Extract secondary images from a multi-image JPEG using MPF directory.
///
/// # Arguments
///
/// * `data` - The complete multi-image JPEG data
/// * `mpf` - Parsed MPF directory
///
/// # Returns
///
/// Vector of byte slices for each secondary image (excludes primary).
pub fn extract_secondary_images<'a>(data: &'a [u8], mpf: &MpfDirectory) -> Vec<&'a [u8]> {
    let mut images = Vec::new();

    for entry in &mpf.entries {
        // Skip primary image (index 0, offset 0)
        if entry.index == 0 {
            continue;
        }

        // Calculate actual offset
        // Secondary images have offsets relative to the MPF marker
        let actual_offset = mpf.mpf_marker_offset + entry.offset as usize;
        let end = actual_offset + entry.size as usize;

        if actual_offset < data.len() && end <= data.len() {
            images.push(&data[actual_offset..end]);
        }
    }

    images
}

/// Assemble a multi-image JPEG with MPF header.
///
/// Creates a valid multi-image JPEG by:
/// 1. Inserting an MPF APP2 segment into the primary image
/// 2. Appending secondary images after the primary's EOI
///
/// # Arguments
///
/// * `primary` - The primary JPEG image (complete, with SOI and EOI)
/// * `secondaries` - Secondary images to append (gain maps, thumbnails, etc.)
/// * `types` - Image types for each secondary
///
/// # Returns
///
/// Complete multi-image JPEG with proper MPF header.
pub fn assemble(primary: &[u8], secondaries: &[&[u8]], types: &[MpfImageType]) -> Result<Vec<u8>> {
    if secondaries.len() != types.len() {
        return Err(Error::MpfParse(
            "Mismatched secondaries and types count".into(),
        ));
    }

    if secondaries.is_empty() {
        // No secondaries, just return primary
        return Ok(primary.to_vec());
    }

    // Find where to insert the MPF header (after SOI and existing APP segments)
    let insert_pos = find_mpf_insert_position(primary)?;

    // Calculate sizes for MPF header
    // Primary size includes the MPF header we're about to add
    let mpf_header = create_mpf_header_with_placeholder();
    let primary_with_mpf_size = primary.len() + mpf_header.len();

    // Build MPF directory
    let mut entries = Vec::with_capacity(1 + secondaries.len());

    // Primary entry
    entries.push((MpfImageType::Primary, primary_with_mpf_size as u32, 0u32));

    // Secondary entries - offsets are relative to MPF marker position
    // But actually, they're from start of file for the calculation, then we adjust
    let mut offset = primary_with_mpf_size as u32;
    for (i, secondary) in secondaries.iter().enumerate() {
        let img_type = types.get(i).copied().unwrap_or(MpfImageType::GainMap);
        // Offset is relative to MPF marker, not start of file
        // MPF marker will be at insert_pos
        let relative_offset = offset - insert_pos as u32;
        entries.push((img_type, secondary.len() as u32, relative_offset));
        offset += secondary.len() as u32;
    }

    // Create the actual MPF header
    let mpf_header = create_mpf_header(&entries, insert_pos);

    // Assemble the output
    let total_size =
        primary.len() + mpf_header.len() + secondaries.iter().map(|s| s.len()).sum::<usize>();
    let mut output = Vec::with_capacity(total_size);

    // Primary up to insert position
    output.extend_from_slice(&primary[..insert_pos]);

    // MPF header
    output.extend_from_slice(&mpf_header);

    // Rest of primary
    output.extend_from_slice(&primary[insert_pos..]);

    // Secondary images
    for secondary in secondaries {
        output.extend_from_slice(secondary);
    }

    Ok(output)
}

/// Generate MPF APP2 segment data.
///
/// Use this when you need to create an MPF header separately from assembly.
///
/// # Arguments
///
/// * `primary_size` - Size of the primary image in bytes
/// * `secondary_sizes` - Sizes of secondary images
/// * `types` - Types for each secondary image
/// * `mpf_offset` - Offset where the MPF marker will be placed
pub fn generate_mpf(
    primary_size: usize,
    secondary_sizes: &[usize],
    types: &[MpfImageType],
    mpf_offset: usize,
) -> Vec<u8> {
    let mut entries = Vec::with_capacity(1 + secondary_sizes.len());

    // Primary entry
    entries.push((MpfImageType::Primary, primary_size as u32, 0u32));

    // Secondary entries
    let mut offset = primary_size as u32;
    for (i, &size) in secondary_sizes.iter().enumerate() {
        let img_type = types.get(i).copied().unwrap_or(MpfImageType::GainMap);
        let relative_offset = offset - mpf_offset as u32;
        entries.push((img_type, size as u32, relative_offset));
        offset += size as u32;
    }

    create_mpf_header(&entries, mpf_offset)
}

// ============================================================================
// Internal helpers
// ============================================================================

fn read_u16(data: &[u8], offset: usize, big_endian: bool) -> u16 {
    if big_endian {
        u16::from_be_bytes([data[offset], data[offset + 1]])
    } else {
        u16::from_le_bytes([data[offset], data[offset + 1]])
    }
}

fn read_u32(data: &[u8], offset: usize, big_endian: bool) -> u32 {
    if big_endian {
        u32::from_be_bytes([
            data[offset],
            data[offset + 1],
            data[offset + 2],
            data[offset + 3],
        ])
    } else {
        u32::from_le_bytes([
            data[offset],
            data[offset + 1],
            data[offset + 2],
            data[offset + 3],
        ])
    }
}

/// Find the position to insert the MPF header.
/// Should be after SOI and after any existing APP0/APP1 segments.
fn find_mpf_insert_position(data: &[u8]) -> Result<usize> {
    if data.len() < 4 || data[0] != 0xFF || data[1] != 0xD8 {
        return Err(Error::JpegDecode("Not a valid JPEG".into()));
    }

    let mut pos = 2;

    // Skip existing APP segments that should come before MPF
    // APP0 (JFIF), APP1 (EXIF/XMP) typically come first
    while pos < data.len() - 3 {
        if data[pos] != 0xFF {
            break;
        }

        let marker = data[pos + 1];

        // Stop if we hit a non-APP marker or APP2+ (where MPF goes)
        if !(0xE0..=0xE1).contains(&marker) {
            break;
        }

        // Skip this APP segment
        let length = u16::from_be_bytes([data[pos + 2], data[pos + 3]]) as usize;
        pos += 2 + length;
    }

    Ok(pos)
}

/// Create a placeholder MPF header (for size calculation).
fn create_mpf_header_with_placeholder() -> Vec<u8> {
    // Fixed size MPF header for 2 images
    vec![0u8; 82] // Typical size for 2-image MPF
}

/// Create the actual MPF header.
fn create_mpf_header(entries: &[(MpfImageType, u32, u32)], _mpf_offset: usize) -> Vec<u8> {
    let mut mpf = Vec::with_capacity(128);

    // Build MPF data (TIFF-like structure)
    // Endianness marker (big-endian: MM)
    mpf.extend_from_slice(b"MM");

    // Fixed value 0x002A for TIFF header
    mpf.push(0x00);
    mpf.push(0x2A);

    // Offset to first IFD (8 bytes from start of TIFF header)
    mpf.extend_from_slice(&8u32.to_be_bytes());

    // IFD (Image File Directory)
    // Number of entries: 3 (Version, NumberOfImages, MPEntry)
    mpf.extend_from_slice(&3u16.to_be_bytes());

    // Entry 1: Version tag (0xB000)
    // Type: UNDEFINED (7), Count: 4, Value: inline "0100"
    mpf.extend_from_slice(&0xB000u16.to_be_bytes()); // Tag
    mpf.extend_from_slice(&7u16.to_be_bytes()); // Type (UNDEFINED)
    mpf.extend_from_slice(&4u32.to_be_bytes()); // Count
    mpf.extend_from_slice(b"0100"); // Value (inline)

    // Entry 2: Number of images (0xB001)
    // Type: LONG (4), Count: 1, Value: number of entries
    mpf.extend_from_slice(&0xB001u16.to_be_bytes()); // Tag
    mpf.extend_from_slice(&4u16.to_be_bytes()); // Type (LONG)
    mpf.extend_from_slice(&1u32.to_be_bytes()); // Count
    mpf.extend_from_slice(&(entries.len() as u32).to_be_bytes()); // Value

    // Entry 3: MP Entry (0xB002)
    // Type: UNDEFINED (7), Count: entries * 16, Offset: after IFD
    let mp_entry_size = (entries.len() * 16) as u32;
    let mp_entry_offset: u32 = 8 + 2 + 36 + 4; // TIFF header + num entries + 3 IFD entries + next IFD ptr
    mpf.extend_from_slice(&0xB002u16.to_be_bytes()); // Tag
    mpf.extend_from_slice(&7u16.to_be_bytes()); // Type (UNDEFINED)
    mpf.extend_from_slice(&mp_entry_size.to_be_bytes()); // Count
    mpf.extend_from_slice(&mp_entry_offset.to_be_bytes()); // Offset

    // Next IFD offset (0 = no more IFDs)
    mpf.extend_from_slice(&0u32.to_be_bytes());

    // MP Entry data (16 bytes per image)
    for (i, (img_type, size, offset)) in entries.iter().enumerate() {
        // Attribute (4 bytes)
        let attr = if i == 0 {
            MpfImageType::Primary.to_attribute()
        } else {
            img_type.to_attribute()
        };
        mpf.extend_from_slice(&attr.to_be_bytes());

        // Size (4 bytes)
        mpf.extend_from_slice(&size.to_be_bytes());

        // Offset (4 bytes) - 0 for primary, relative to MPF for others
        mpf.extend_from_slice(&offset.to_be_bytes());

        // Dependent image entries (4 bytes total - 2 x u16)
        mpf.extend_from_slice(&0u32.to_be_bytes());
    }

    // Create APP2 marker wrapper
    let mut marker = Vec::with_capacity(4 + 4 + mpf.len());
    marker.push(0xFF);
    marker.push(0xE2); // APP2

    let length = 2 + 4 + mpf.len(); // length field + "MPF\0" + mpf data
    marker.push(((length >> 8) & 0xFF) as u8);
    marker.push((length & 0xFF) as u8);

    marker.extend_from_slice(b"MPF\0");
    marker.extend_from_slice(&mpf);

    marker
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_primary_bounds() {
        // Minimal JPEG: SOI + APP0 + EOI
        let jpeg = vec![
            0xFF, 0xD8, // SOI
            0xFF, 0xE0, 0x00, 0x10, // APP0 with length 16
            0x4A, 0x46, 0x49, 0x46, 0x00, // JFIF identifier
            0x01, 0x01, 0x00, 0x00, 0x01, 0x00, 0x01, 0x00, 0x00, // JFIF data
            0xFF, 0xD9, // EOI
        ];

        let bounds = primary_bounds(&jpeg).unwrap();
        assert_eq!(bounds.start, 0);
        assert_eq!(bounds.end, jpeg.len());
    }

    #[test]
    fn test_primary_bounds_multi_image() {
        // Two JPEGs concatenated
        let jpeg1 = vec![
            0xFF, 0xD8, // SOI
            0xFF, 0xD9, // EOI
        ];
        let jpeg2 = vec![
            0xFF, 0xD8, // SOI
            0xFF, 0xD9, // EOI
        ];

        let mut data = jpeg1.clone();
        data.extend_from_slice(&jpeg2);

        let bounds = primary_bounds(&data).unwrap();
        assert_eq!(bounds, 0..4); // Just the first JPEG
    }

    #[test]
    fn test_scan_segments() {
        let jpeg = vec![
            0xFF, 0xD8, // SOI
            0xFF, 0xE0, 0x00, 0x07, // APP0 with length 7
            b'J', b'F', b'I', b'F', 0x00, // JFIF identifier
            0xFF, 0xE1, 0x00, 0x06, // APP1 with length 6
            b'T', b'E', b'S', b'T', // Test data
            0xFF, 0xDA, // SOS
            0x00, 0x00, // (scan data would follow)
        ];

        let segments = scan_segments(&jpeg);
        assert_eq!(segments.len(), 2);
        assert_eq!(segments[0].marker_num, 0); // APP0
        assert!(segments[0].is_jfif());
        assert_eq!(segments[1].marker_num, 1); // APP1
    }

    #[test]
    fn test_mpf_image_type_roundtrip() {
        // Test primary - matches existing mpf.rs MpImageType::BaselinePrimary
        let attr = MpfImageType::Primary.to_attribute();
        assert_eq!(attr, 0x03_0000);
        let back = MpfImageType::from_attribute(attr);
        assert!(matches!(back, MpfImageType::Primary));

        // Test gain map - matches existing mpf.rs MpImageType::DependentChild
        let attr = MpfImageType::GainMap.to_attribute();
        assert_eq!(attr, 0x00_0000);
        let back = MpfImageType::from_attribute(attr);
        assert!(matches!(back, MpfImageType::GainMap));
    }

    #[test]
    fn test_assemble_basic() {
        let primary = vec![
            0xFF, 0xD8, // SOI
            0xFF, 0xE0, 0x00, 0x07, // APP0
            b'J', b'F', b'I', b'F', 0x00, 0xFF, 0xD9, // EOI
        ];

        let secondary = vec![
            0xFF, 0xD8, // SOI
            0xFF, 0xD9, // EOI
        ];

        let result = assemble(&primary, &[&secondary], &[MpfImageType::GainMap]).unwrap();

        // Should start with SOI
        assert_eq!(result[0], 0xFF);
        assert_eq!(result[1], 0xD8);

        // Should contain MPF marker
        let has_mpf = result
            .windows(6)
            .any(|w| w[0] == 0xFF && w[1] == 0xE2 && &w[4..] == b"MP");
        assert!(has_mpf);

        // Should end with the secondary image
        assert_eq!(result[result.len() - 2], 0xFF);
        assert_eq!(result[result.len() - 1], 0xD9);
    }

    #[test]
    fn test_generate_mpf() {
        let mpf_data = generate_mpf(50000, &[10000], &[MpfImageType::GainMap], 100);

        // Should be an APP2 marker
        assert_eq!(mpf_data[0], 0xFF);
        assert_eq!(mpf_data[1], 0xE2);

        // Should contain "MPF\0"
        assert!(mpf_data.windows(4).any(|w| w == b"MPF\0"));
    }
}
