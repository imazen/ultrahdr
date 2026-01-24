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
pub fn create_mpf_header(primary_length: usize, gainmap_length: usize) -> Vec<u8> {
    let mut mpf = Vec::with_capacity(128);

    // Calculate offsets
    // Primary image offset is always 0 (from start of file)
    // Gain map offset is primary_length
    let primary_offset = 0u32;
    let gainmap_offset = primary_length as u32;

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

    // Entry 1: Version tag
    write_ifd_entry(&mut mpf, TAG_VERSION, TYPE_UNDEFINED, 4, 0); // Value inline
    mpf.extend_from_slice(MPF_VERSION);

    // Entry 2: Number of images
    write_ifd_entry(&mut mpf, TAG_NUMBER_OF_IMAGES, TYPE_LONG, 1, 2);

    // Entry 3: MP Entry (array of image entries)
    // Each MP Entry is 16 bytes, we have 2 images
    let mp_entry_size = 32u32;
    let mp_entry_offset = mpf.len() as u32 + 4 + 4; // After IFD end + next IFD pointer
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

    // Data offset (4 bytes) - relative to start of file for first image,
    // relative to MPF marker for subsequent images
    buf.extend_from_slice(&offset.to_be_bytes());

    // Dependent image 1 entry number (2 bytes) - 0 if none
    buf.extend_from_slice(&0u16.to_be_bytes());

    // Dependent image 2 entry number (2 bytes) - 0 if none
    buf.extend_from_slice(&0u16.to_be_bytes());
}

/// Parse MPF header to find gain map location.
///
/// Returns (offset, length) of the gain map image.
pub fn parse_mpf(data: &[u8]) -> Result<Vec<(usize, usize)>> {
    // Find APP2 marker with MPF identifier
    let mut pos = 0;
    while pos + 4 < data.len() {
        if data[pos] == 0xFF && data[pos + 1] == 0xE2 {
            let length = u16::from_be_bytes([data[pos + 2], data[pos + 3]]) as usize;
            if pos + 4 + length <= data.len() {
                let marker_data = &data[pos + 4..pos + 2 + length];
                if marker_data.starts_with(MPF_IDENTIFIER) {
                    return parse_mpf_data(&marker_data[4..], pos);
                }
            }
        }
        pos += 1;
    }

    Err(Error::MpfParse("MPF marker not found".into()))
}

/// Parse MPF data to extract image entries.
fn parse_mpf_data(mpf_data: &[u8], mpf_marker_offset: usize) -> Result<Vec<(usize, usize)>> {
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

            // First image offset is relative to start of file
            // Subsequent images are relative to MPF marker
            let actual_offset = if i == 0 {
                0
            } else {
                mpf_marker_offset + offset
            };

            images.push((actual_offset, size));
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
        let header = create_mpf_header(50000, 10000);

        // Should start with APP2 marker
        assert_eq!(header[0], 0xFF);
        assert_eq!(header[1], 0xE2);

        // Should contain MPF identifier
        assert!(header.windows(4).any(|w| w == MPF_IDENTIFIER));
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
