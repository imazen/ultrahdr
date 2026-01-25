//! Multi-Picture Format (MPF) for bundling primary image and gain map.
//!
//! MPF is defined by CIPA DC-007 and allows multiple images to be stored
//! in a single JPEG file.

use alloc::vec::Vec;

use crate::types::{Error, Result};

/// MPF marker identifier.
pub const MPF_IDENTIFIER: &[u8] = b"MPF\0";

/// MPF version string.
pub const MPF_VERSION: &[u8] = b"0100";

// MPF tag IDs
const TAG_VERSION: u16 = 0xB000;
const TAG_NUMBER_OF_IMAGES: u16 = 0xB001;
const TAG_MP_ENTRY: u16 = 0xB002;

// Type constants
const TYPE_UNDEFINED: u16 = 7;
const TYPE_LONG: u16 = 4;

/// Image type flags for MP Entry.
#[derive(Debug, Clone, Copy)]
#[repr(u32)]
pub enum MpImageType {
    /// Baseline MP primary image.
    BaselinePrimary = 0x030000,
    /// Dependent child image (like gain map).
    DependentChild = 0x000000,
}

/// Create MPF APP2 marker that bundles primary and gain map images.
///
/// The primary JPEG should already have the XMP metadata inserted.
/// This function creates the MPF header that goes after the primary JPEG's
/// SOI and metadata markers.
///
/// # Arguments
/// * `primary_length` - Total length of the primary JPEG in bytes
/// * `gainmap_length` - Total length of the gain map JPEG in bytes
/// * `mpf_insert_offset` - File offset where this MPF segment will be inserted.
///   The TIFF header will be at `mpf_insert_offset + 8` (after marker, length, and "MPF\0").
///   If `None`, assumes MPF is at the very start of file (offset 0).
pub fn create_mpf_header(
    primary_length: usize,
    gainmap_length: usize,
    mpf_insert_offset: Option<usize>,
) -> Vec<u8> {
    let mut mpf = Vec::with_capacity(128);

    // Calculate offsets per CIPA DC-007:
    // - Primary image offset is always 0 (from start of file)
    // - Secondary image offsets are relative to the TIFF header position
    //   (TIFF header is at mpf_insert_offset + 4 (marker+length) + 4 ("MPF\0"))
    let primary_offset = 0u32;
    let tiff_header_pos = mpf_insert_offset.unwrap_or(0) + 4 + MPF_IDENTIFIER.len();
    // Use saturating_sub to handle the case where primary_length is 0 (size estimation calls)
    let gainmap_offset = primary_length.saturating_sub(tiff_header_pos) as u32;

    // Build MPF data
    // Endianness marker (big-endian: MM)
    mpf.extend_from_slice(b"MM");

    // Fixed value 0x002A for TIFF header
    mpf.push(0x00);
    mpf.push(0x2A);

    // Offset to first IFD (8 bytes from start of TIFF header)
    mpf.extend_from_slice(&8u32.to_be_bytes());

    // IFD (Image File Directory)
    // Number of entries: 3
    mpf.extend_from_slice(&3u16.to_be_bytes());

    // Entry 1: Version tag (value "0100" stored inline in the 4-byte value field)
    // For TYPE_UNDEFINED with count<=4, value is stored inline as big-endian bytes
    let version_value = u32::from_be_bytes([
        MPF_VERSION[0],
        MPF_VERSION[1],
        MPF_VERSION[2],
        MPF_VERSION[3],
    ]);
    write_ifd_entry(&mut mpf, TAG_VERSION, TYPE_UNDEFINED, 4, version_value);

    // Entry 2: Number of images
    write_ifd_entry(&mut mpf, TAG_NUMBER_OF_IMAGES, TYPE_LONG, 1, 2);

    // Entry 3: MP Entry (array of image entries)
    // Each MP Entry is 16 bytes, we have 2 images
    let mp_entry_size = 32u32;
    // MP entry data follows: this entry (12 bytes) + next IFD pointer (4 bytes)
    let mp_entry_offset = mpf.len() as u32 + 12 + 4;
    write_ifd_entry(
        &mut mpf,
        TAG_MP_ENTRY,
        TYPE_UNDEFINED,
        mp_entry_size,
        mp_entry_offset,
    );

    // Next IFD offset (0 = no more IFDs)
    mpf.extend_from_slice(&0u32.to_be_bytes());

    // MP Entry data (16 bytes per image)
    // Image 1: Primary
    write_mp_entry(
        &mut mpf,
        MpImageType::BaselinePrimary,
        primary_length as u32,
        primary_offset,
    );

    // Image 2: Gain map
    write_mp_entry(
        &mut mpf,
        MpImageType::DependentChild,
        gainmap_length as u32,
        gainmap_offset,
    );

    // Create APP2 marker
    let mut marker = Vec::with_capacity(4 + 4 + mpf.len());
    marker.push(0xFF);
    marker.push(0xE2); // APP2

    let length = 2 + MPF_IDENTIFIER.len() + mpf.len();
    marker.push(((length >> 8) & 0xFF) as u8);
    marker.push((length & 0xFF) as u8);

    marker.extend_from_slice(MPF_IDENTIFIER);
    marker.extend_from_slice(&mpf);

    marker
}

/// Write an IFD entry.
fn write_ifd_entry(buf: &mut Vec<u8>, tag: u16, type_id: u16, count: u32, value_or_offset: u32) {
    buf.extend_from_slice(&tag.to_be_bytes());
    buf.extend_from_slice(&type_id.to_be_bytes());
    buf.extend_from_slice(&count.to_be_bytes());
    buf.extend_from_slice(&value_or_offset.to_be_bytes());
}

/// Write an MP Entry (16 bytes).
fn write_mp_entry(buf: &mut Vec<u8>, image_type: MpImageType, size: u32, offset: u32) {
    // Attribute (4 bytes): image type flags
    buf.extend_from_slice(&(image_type as u32).to_be_bytes());

    // Size (4 bytes)
    buf.extend_from_slice(&size.to_be_bytes());

    // Data offset (4 bytes) - 0 for primary image,
    // relative to TIFF header for secondary images (per CIPA DC-007)
    buf.extend_from_slice(&offset.to_be_bytes());

    // Dependent image 1 entry number (2 bytes) - 0 if none
    buf.extend_from_slice(&0u16.to_be_bytes());

    // Dependent image 2 entry number (2 bytes) - 0 if none
    buf.extend_from_slice(&0u16.to_be_bytes());
}

/// Parse MPF header to find image locations.
///
/// Returns `(start, end)` byte ranges for each image in the file.
/// The first entry is the primary image, subsequent entries are secondary images (gain maps, etc.).
pub fn parse_mpf(data: &[u8]) -> Result<Vec<(usize, usize)>> {
    // Find APP2 marker with MPF identifier
    let mut pos = 0;
    while pos + 4 < data.len() {
        if data[pos] == 0xFF && data[pos + 1] == 0xE2 {
            let length = u16::from_be_bytes([data[pos + 2], data[pos + 3]]) as usize;
            if pos + 4 + length <= data.len() {
                let marker_data = &data[pos + 4..pos + 2 + length];
                if marker_data.starts_with(MPF_IDENTIFIER) {
                    // TIFF header starts after marker (2) + length (2) + "MPF\0" (4)
                    let tiff_header_pos = pos + 4 + MPF_IDENTIFIER.len();
                    return parse_mpf_data(&marker_data[4..], tiff_header_pos);
                }
            }
        }
        pos += 1;
    }

    Err(Error::MpfParse("MPF marker not found".into()))
}

/// Parse MPF data to extract image entries.
///
/// `tiff_header_pos` is the absolute file position of the TIFF header (after "MPF\0").
/// Per CIPA DC-007, secondary image offsets are relative to this position.
fn parse_mpf_data(mpf_data: &[u8], tiff_header_pos: usize) -> Result<Vec<(usize, usize)>> {
    if mpf_data.len() < 8 {
        return Err(Error::MpfParse("MPF data too short".into()));
    }

    // Check endianness
    let big_endian = &mpf_data[0..2] == b"MM";
    if !big_endian && &mpf_data[0..2] != b"II" {
        return Err(Error::MpfParse("Invalid MPF endianness marker".into()));
    }

    // Skip to IFD
    let ifd_offset = if big_endian {
        u32::from_be_bytes([mpf_data[4], mpf_data[5], mpf_data[6], mpf_data[7]])
    } else {
        u32::from_le_bytes([mpf_data[4], mpf_data[5], mpf_data[6], mpf_data[7]])
    } as usize;

    if ifd_offset + 2 > mpf_data.len() {
        return Err(Error::MpfParse("Invalid IFD offset".into()));
    }

    // Read number of IFD entries
    let num_entries = if big_endian {
        u16::from_be_bytes([mpf_data[ifd_offset], mpf_data[ifd_offset + 1]])
    } else {
        u16::from_le_bytes([mpf_data[ifd_offset], mpf_data[ifd_offset + 1]])
    } as usize;

    let mut images = Vec::new();
    let mut mp_entry_offset = 0usize;
    let mut mp_entry_count = 0u32;

    // Parse IFD entries
    let entry_start = ifd_offset + 2;
    for i in 0..num_entries {
        let offset = entry_start + i * 12;
        if offset + 12 > mpf_data.len() {
            break;
        }

        let tag = if big_endian {
            u16::from_be_bytes([mpf_data[offset], mpf_data[offset + 1]])
        } else {
            u16::from_le_bytes([mpf_data[offset], mpf_data[offset + 1]])
        };

        let _count = if big_endian {
            u32::from_be_bytes([
                mpf_data[offset + 4],
                mpf_data[offset + 5],
                mpf_data[offset + 6],
                mpf_data[offset + 7],
            ])
        } else {
            u32::from_le_bytes([
                mpf_data[offset + 4],
                mpf_data[offset + 5],
                mpf_data[offset + 6],
                mpf_data[offset + 7],
            ])
        };

        let value_offset = if big_endian {
            u32::from_be_bytes([
                mpf_data[offset + 8],
                mpf_data[offset + 9],
                mpf_data[offset + 10],
                mpf_data[offset + 11],
            ])
        } else {
            u32::from_le_bytes([
                mpf_data[offset + 8],
                mpf_data[offset + 9],
                mpf_data[offset + 10],
                mpf_data[offset + 11],
            ])
        };

        match tag {
            TAG_NUMBER_OF_IMAGES => {
                mp_entry_count = value_offset;
            }
            TAG_MP_ENTRY => {
                mp_entry_offset = value_offset as usize;
            }
            _ => {}
        }
    }

    // Parse MP Entry array
    if mp_entry_offset > 0 && mp_entry_count > 0 && mp_entry_offset + 16 <= mpf_data.len() {
        for i in 0..mp_entry_count as usize {
            let entry_pos = mp_entry_offset + i * 16;
            if entry_pos + 16 > mpf_data.len() {
                break;
            }

            // Image size (bytes 4-7)
            let size = if big_endian {
                u32::from_be_bytes([
                    mpf_data[entry_pos + 4],
                    mpf_data[entry_pos + 5],
                    mpf_data[entry_pos + 6],
                    mpf_data[entry_pos + 7],
                ])
            } else {
                u32::from_le_bytes([
                    mpf_data[entry_pos + 4],
                    mpf_data[entry_pos + 5],
                    mpf_data[entry_pos + 6],
                    mpf_data[entry_pos + 7],
                ])
            } as usize;

            // Data offset (bytes 8-11)
            let offset = if big_endian {
                u32::from_be_bytes([
                    mpf_data[entry_pos + 8],
                    mpf_data[entry_pos + 9],
                    mpf_data[entry_pos + 10],
                    mpf_data[entry_pos + 11],
                ])
            } else {
                u32::from_le_bytes([
                    mpf_data[entry_pos + 8],
                    mpf_data[entry_pos + 9],
                    mpf_data[entry_pos + 10],
                    mpf_data[entry_pos + 11],
                ])
            } as usize;

            // First image offset is 0 (primary image starts at file offset 0)
            // Subsequent image offsets are relative to the TIFF header position
            // (per CIPA DC-007 spec)
            let start = if i == 0 { 0 } else { tiff_header_pos + offset };
            let end = start + size;

            images.push((start, end));
        }
    }

    if images.is_empty() {
        return Err(Error::MpfParse("No images found in MPF".into()));
    }

    Ok(images)
}

/// Find JPEG boundaries (SOI and EOI markers) in data.
pub fn find_jpeg_boundaries(data: &[u8]) -> Vec<(usize, usize)> {
    let mut boundaries = Vec::new();
    let mut pos = 0;

    while pos < data.len() - 1 {
        // Look for SOI (Start Of Image) marker
        if data[pos] == 0xFF && data[pos + 1] == 0xD8 {
            let start = pos;

            // Find EOI (End Of Image) marker
            pos += 2;
            while pos < data.len() - 1 {
                if data[pos] == 0xFF && data[pos + 1] == 0xD9 {
                    boundaries.push((start, pos + 2));
                    pos += 2;
                    break;
                }
                pos += 1;
            }
        } else {
            pos += 1;
        }
    }

    boundaries
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_create_mpf_header() {
        // Test with MPF at file start (legacy behavior)
        let header = create_mpf_header(50000, 10000, None);

        // Should start with APP2 marker
        assert_eq!(header[0], 0xFF);
        assert_eq!(header[1], 0xE2);

        // Should contain MPF identifier
        assert!(header.windows(4).any(|w| w == MPF_IDENTIFIER));
    }

    #[test]
    fn test_mpf_roundtrip() {
        // Create a fake file structure:
        // [0..2]: SOI
        // [2..mpf_insert_pos]: other metadata
        // [mpf_insert_pos..mpf_insert_pos+header_len]: MPF header
        // [mpf_insert_pos+header_len..primary_total]: rest of primary + EOI
        // [primary_total..]: gain map

        let mpf_insert_pos = 100;
        let gainmap_length = 10000;

        // First, calculate the MPF header size
        let header_estimate = create_mpf_header(0, 0, Some(mpf_insert_pos)).len();

        // Primary length = content before MPF + MPF header + content after MPF
        // For simplicity, use: 100 (before) + header_len + 49900 (after) = ~50000
        let primary_without_header = 50000;
        let primary_length = primary_without_header + header_estimate;

        let header = create_mpf_header(primary_length, gainmap_length, Some(mpf_insert_pos));
        let header_len = header.len();

        // Build fake file
        let total_size = primary_length + gainmap_length;
        let mut file = vec![0u8; total_size];

        // SOI at start
        file[0] = 0xFF;
        file[1] = 0xD8;

        // Insert MPF header at the insertion position
        file[mpf_insert_pos..mpf_insert_pos + header_len].copy_from_slice(&header);

        // Gain map SOI at primary_length
        file[primary_length] = 0xFF;
        file[primary_length + 1] = 0xD8;

        // Parse it back
        let images = parse_mpf(&file).expect("should parse");
        assert_eq!(images.len(), 2);
        // Primary: start=0, end=primary_length
        assert_eq!(images[0], (0, primary_length));
        // Gain map: start=primary_length, end=primary_length+gainmap_length
        assert_eq!(images[1], (primary_length, primary_length + gainmap_length));
    }

    #[test]
    fn test_find_jpeg_boundaries() {
        // Create fake JPEG data
        let mut data = Vec::new();

        // First JPEG
        data.extend_from_slice(&[0xFF, 0xD8]); // SOI
        data.extend_from_slice(&[0x00; 100]); // Content
        data.extend_from_slice(&[0xFF, 0xD9]); // EOI

        // Second JPEG
        data.extend_from_slice(&[0xFF, 0xD8]); // SOI
        data.extend_from_slice(&[0x00; 50]); // Content
        data.extend_from_slice(&[0xFF, 0xD9]); // EOI

        let boundaries = find_jpeg_boundaries(&data);

        assert_eq!(boundaries.len(), 2);
        assert_eq!(boundaries[0], (0, 104));
        assert_eq!(boundaries[1], (104, 158));
    }
}
