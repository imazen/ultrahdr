//! Multi-Picture Format (MPF) for bundling primary and secondary images.
//!
//! MPF is defined by CIPA DC-007 and allows multiple images to be stored
//! in a single JPEG file. Used for Ultra HDR gain maps, iPhone depth maps,
//! panoramas, and other multi-image use cases.

use alloc::vec::Vec;

use crate::types::{Error, Result};

use super::container::{MpImageType, MpfEntry};

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

/// Create MPF APP2 marker for N secondary images.
///
/// Each entry in `secondary_images` is `(image_type, byte_length)`.
/// The primary image is always entry 0 with `BaselinePrimary` type.
///
/// # Arguments
/// * `primary_length` - Total length of the primary JPEG in bytes
/// * `secondary_images` - Type and byte length of each secondary image
/// * `mpf_insert_offset` - File offset where this MPF segment will be inserted
pub fn create_mpf_header_typed(
    primary_length: usize,
    secondary_images: &[(MpImageType, usize)],
    mpf_insert_offset: Option<usize>,
) -> Vec<u8> {
    let num_images = 1 + secondary_images.len(); // primary + secondaries
    let mut mpf = Vec::with_capacity(128);

    let tiff_header_pos = mpf_insert_offset.unwrap_or(0) + 4 + MPF_IDENTIFIER.len();

    // TIFF header (big-endian)
    mpf.extend_from_slice(b"MM");
    mpf.push(0x00);
    mpf.push(0x2A);
    mpf.extend_from_slice(&8u32.to_be_bytes()); // IFD at offset 8

    // IFD: 3 entries
    mpf.extend_from_slice(&3u16.to_be_bytes());

    // Entry 1: Version
    let version_value = u32::from_be_bytes([
        MPF_VERSION[0],
        MPF_VERSION[1],
        MPF_VERSION[2],
        MPF_VERSION[3],
    ]);
    write_ifd_entry(&mut mpf, TAG_VERSION, TYPE_UNDEFINED, 4, version_value);

    // Entry 2: Number of images
    write_ifd_entry(
        &mut mpf,
        TAG_NUMBER_OF_IMAGES,
        TYPE_LONG,
        1,
        num_images as u32,
    );

    // Entry 3: MP Entry array
    let mp_entry_size = (num_images * 16) as u32;
    let mp_entry_offset = mpf.len() as u32 + 12 + 4; // this entry + next IFD ptr
    write_ifd_entry(
        &mut mpf,
        TAG_MP_ENTRY,
        TYPE_UNDEFINED,
        mp_entry_size,
        mp_entry_offset,
    );

    // Next IFD offset (0 = none)
    mpf.extend_from_slice(&0u32.to_be_bytes());

    // MP Entry: primary
    write_mp_entry(
        &mut mpf,
        MpImageType::BaselinePrimary.type_code(),
        primary_length as u32,
        0,
    );

    // MP Entry: secondaries (offsets accumulate after primary)
    let mut offset_from_tiff = primary_length.saturating_sub(tiff_header_pos) as u32;
    for (image_type, length) in secondary_images {
        write_mp_entry(
            &mut mpf,
            image_type.type_code(),
            *length as u32,
            offset_from_tiff,
        );
        offset_from_tiff += *length as u32;
    }

    // Wrap in APP2 marker
    let mut marker = Vec::with_capacity(4 + 4 + mpf.len());
    marker.push(0xFF);
    marker.push(0xE2);
    let length = 2 + MPF_IDENTIFIER.len() + mpf.len();
    marker.push(((length >> 8) & 0xFF) as u8);
    marker.push((length & 0xFF) as u8);
    marker.extend_from_slice(MPF_IDENTIFIER);
    marker.extend_from_slice(&mpf);

    marker
}

/// Create MPF APP2 marker for primary + single gain map (convenience wrapper).
///
/// Equivalent to `create_mpf_header_typed(primary_length, &[(MpImageType::Undefined, gainmap_length)], mpf_insert_offset)`.
pub fn create_mpf_header(
    primary_length: usize,
    gainmap_length: usize,
    mpf_insert_offset: Option<usize>,
) -> Vec<u8> {
    create_mpf_header_typed(
        primary_length,
        &[(MpImageType::Undefined, gainmap_length)],
        mpf_insert_offset,
    )
}

/// Write an IFD entry.
fn write_ifd_entry(buf: &mut Vec<u8>, tag: u16, type_id: u16, count: u32, value_or_offset: u32) {
    buf.extend_from_slice(&tag.to_be_bytes());
    buf.extend_from_slice(&type_id.to_be_bytes());
    buf.extend_from_slice(&count.to_be_bytes());
    buf.extend_from_slice(&value_or_offset.to_be_bytes());
}

/// Write an MP Entry (16 bytes).
fn write_mp_entry(buf: &mut Vec<u8>, type_code: u32, size: u32, offset: u32) {
    // Attribute (4 bytes): image type flags
    buf.extend_from_slice(&type_code.to_be_bytes());

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

/// Parse MPF header to find image locations with type information.
///
/// Returns typed [`MpfEntry`] values with image type, offset, and size.
/// The first entry is the primary image.
pub fn parse_mpf_entries(data: &[u8]) -> Result<Vec<MpfEntry>> {
    let mut pos = 0;
    while pos + 4 < data.len() {
        if data[pos] == 0xFF && data[pos + 1] == 0xE2 {
            let length = u16::from_be_bytes([data[pos + 2], data[pos + 3]]) as usize;
            if pos + 4 + length <= data.len() {
                let marker_data = &data[pos + 4..pos + 2 + length];
                if marker_data.starts_with(MPF_IDENTIFIER) {
                    let tiff_header_pos = pos + 4 + MPF_IDENTIFIER.len();
                    return parse_mpf_data_typed(&marker_data[4..], tiff_header_pos);
                }
            }
        }
        pos += 1;
    }

    Err(Error::MpfParse("MPF marker not found".into()))
}

/// Parse MPF header to find image locations (legacy convenience).
///
/// Returns `(start, end)` byte ranges for each image in the file.
/// For typed entries with image type info, use [`parse_mpf_entries`].
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

/// Parse MPF data to extract typed image entries.
fn parse_mpf_data_typed(mpf_data: &[u8], tiff_header_pos: usize) -> Result<Vec<MpfEntry>> {
    if mpf_data.len() < 8 {
        return Err(Error::MpfParse("MPF data too short".into()));
    }

    let big_endian = &mpf_data[0..2] == b"MM";
    if !big_endian && &mpf_data[0..2] != b"II" {
        return Err(Error::MpfParse("Invalid MPF endianness marker".into()));
    }

    let read_u16 = |data: &[u8], off: usize| -> u16 {
        if big_endian {
            u16::from_be_bytes([data[off], data[off + 1]])
        } else {
            u16::from_le_bytes([data[off], data[off + 1]])
        }
    };
    let read_u32 = |data: &[u8], off: usize| -> u32 {
        if big_endian {
            u32::from_be_bytes([data[off], data[off + 1], data[off + 2], data[off + 3]])
        } else {
            u32::from_le_bytes([data[off], data[off + 1], data[off + 2], data[off + 3]])
        }
    };

    let ifd_offset = read_u32(mpf_data, 4) as usize;
    if ifd_offset + 2 > mpf_data.len() {
        return Err(Error::MpfParse("Invalid IFD offset".into()));
    }

    let num_entries = read_u16(mpf_data, ifd_offset) as usize;
    let mut mp_entry_offset = 0usize;
    let mut mp_entry_count = 0u32;

    let entry_start = ifd_offset + 2;
    for i in 0..num_entries {
        let offset = entry_start + i * 12;
        if offset + 12 > mpf_data.len() {
            break;
        }
        let tag = read_u16(mpf_data, offset);
        let value_offset = read_u32(mpf_data, offset + 8);

        match tag {
            TAG_NUMBER_OF_IMAGES => mp_entry_count = value_offset,
            TAG_MP_ENTRY => mp_entry_offset = value_offset as usize,
            _ => {}
        }
    }

    let mut entries = Vec::new();

    if mp_entry_offset > 0 && mp_entry_count > 0 && mp_entry_offset + 16 <= mpf_data.len() {
        for i in 0..mp_entry_count as usize {
            let entry_pos = mp_entry_offset + i * 16;
            if entry_pos + 16 > mpf_data.len() {
                break;
            }

            let attribute = read_u32(mpf_data, entry_pos);
            let size = read_u32(mpf_data, entry_pos + 4) as usize;
            let data_offset = read_u32(mpf_data, entry_pos + 8) as usize;

            let abs_offset = if i == 0 {
                0
            } else {
                tiff_header_pos + data_offset
            };

            entries.push(MpfEntry {
                image_type: MpImageType::from_type_code(attribute),
                offset: abs_offset,
                size,
            });
        }
    }

    if entries.is_empty() {
        return Err(Error::MpfParse("No images found in MPF".into()));
    }

    Ok(entries)
}

/// Find JPEG boundaries (SOI and EOI markers) in data.
pub fn find_jpeg_boundaries(data: &[u8]) -> Vec<(usize, usize)> {
    let mut boundaries = Vec::new();
    let mut pos = 0;

    while pos + 1 < data.len() {
        // Look for SOI (Start Of Image) marker
        if data[pos] == 0xFF && data[pos + 1] == 0xD8 {
            let start = pos;

            // Find EOI (End Of Image) marker
            pos += 2;
            while pos + 1 < data.len() {
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

    #[test]
    fn test_parse_mpf_no_marker() {
        // Data without APP2 marker should error
        let data = vec![0xFF, 0xD8, 0x00, 0x00, 0x00, 0x00, 0xFF, 0xD9];
        let result = parse_mpf(&data);
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(
            matches!(err, Error::MpfParse(_)),
            "expected MpfParse error, got {:?}",
            err
        );
    }

    #[test]
    fn test_parse_mpf_data_too_short() {
        // MPF data < 8 bytes should error (call parse_mpf_data directly)
        let short_data = [b'M', b'M', 0x00, 0x2A]; // Only 4 bytes
        let result = parse_mpf_data(&short_data, 0);
        assert!(result.is_err());
        let err_msg = result.unwrap_err().to_string();
        assert!(
            err_msg.contains("too short"),
            "expected 'too short' in error, got: {}",
            err_msg
        );
    }

    #[test]
    fn test_parse_mpf_invalid_endianness() {
        // Neither "MM" nor "II" should error
        let mut bad_data = vec![0u8; 16];
        bad_data[0] = b'X';
        bad_data[1] = b'X';
        // Fill rest with valid-looking offsets
        bad_data[4] = 0x00;
        bad_data[5] = 0x00;
        bad_data[6] = 0x00;
        bad_data[7] = 0x08;

        let result = parse_mpf_data(&bad_data, 0);
        assert!(result.is_err());
        let err_msg = result.unwrap_err().to_string();
        assert!(
            err_msg.contains("endianness"),
            "expected 'endianness' in error, got: {}",
            err_msg
        );
    }

    #[test]
    fn test_find_jpeg_boundaries_empty() {
        // Empty input should return empty vec, not panic.
        let boundaries = find_jpeg_boundaries(&[]);
        assert!(boundaries.is_empty());

        // Single byte — also safe
        let boundaries = find_jpeg_boundaries(&[0xFF]);
        assert!(boundaries.is_empty());
    }

    #[test]
    fn test_find_jpeg_boundaries_single() {
        let mut data = Vec::new();
        data.extend_from_slice(&[0xFF, 0xD8]); // SOI
        data.extend_from_slice(&[0x00; 20]); // Content
        data.extend_from_slice(&[0xFF, 0xD9]); // EOI

        let boundaries = find_jpeg_boundaries(&data);
        assert_eq!(boundaries.len(), 1);
        assert_eq!(boundaries[0], (0, 24));
    }

    #[test]
    fn test_find_jpeg_boundaries_no_eoi() {
        // SOI without matching EOI should return empty
        let mut data = Vec::new();
        data.extend_from_slice(&[0xFF, 0xD8]); // SOI
        data.extend_from_slice(&[0x00; 50]); // Content, no EOI

        let boundaries = find_jpeg_boundaries(&data);
        assert!(
            boundaries.is_empty(),
            "expected no boundaries for SOI without EOI, got {:?}",
            boundaries
        );
    }

    #[test]
    fn test_create_mpf_header_structure() {
        let header = create_mpf_header(50000, 10000, Some(100));

        // Starts with APP2 marker: FF E2
        assert_eq!(header[0], 0xFF);
        assert_eq!(header[1], 0xE2);

        // Contains MPF\0 identifier after length bytes
        let mpf_id_pos = 4; // after FF E2 + 2-byte length
        assert_eq!(&header[mpf_id_pos..mpf_id_pos + 4], b"MPF\0");

        // TIFF header starts after MPF\0: "MM" for big-endian
        let tiff_start = mpf_id_pos + 4;
        assert_eq!(&header[tiff_start..tiff_start + 2], b"MM");

        // Fixed TIFF magic 0x002A
        assert_eq!(header[tiff_start + 2], 0x00);
        assert_eq!(header[tiff_start + 3], 0x2A);

        // IFD offset = 8 (from start of TIFF header)
        let ifd_offset = u32::from_be_bytes([
            header[tiff_start + 4],
            header[tiff_start + 5],
            header[tiff_start + 6],
            header[tiff_start + 7],
        ]);
        assert_eq!(ifd_offset, 8);

        // Number of IFD entries = 3
        let ifd_pos = tiff_start + 8;
        let num_entries = u16::from_be_bytes([header[ifd_pos], header[ifd_pos + 1]]);
        assert_eq!(num_entries, 3);
    }

    #[test]
    fn test_mpf_image_types() {
        assert_eq!(MpImageType::BaselinePrimary.type_code(), 0x030000);
        assert_eq!(MpImageType::Undefined.type_code(), 0x000000);
        assert_eq!(MpImageType::Disparity.type_code(), 0x020002);
    }

    #[test]
    fn test_parse_mpf_entries_roundtrip() {
        let mpf_insert_pos = 100;
        let header_estimate =
            create_mpf_header_typed(0, &[(MpImageType::Undefined, 0)], Some(mpf_insert_pos)).len();
        let primary_length = 50000 + header_estimate;
        let gainmap_length = 10000;

        let header = create_mpf_header_typed(
            primary_length,
            &[(MpImageType::Undefined, gainmap_length)],
            Some(mpf_insert_pos),
        );

        let mut file = vec![0u8; primary_length + gainmap_length];
        file[0] = 0xFF;
        file[1] = 0xD8;
        file[mpf_insert_pos..mpf_insert_pos + header.len()].copy_from_slice(&header);
        file[primary_length] = 0xFF;
        file[primary_length + 1] = 0xD8;

        let entries = parse_mpf_entries(&file).unwrap();
        assert_eq!(entries.len(), 2);
        assert_eq!(entries[0].image_type, MpImageType::BaselinePrimary);
        assert_eq!(entries[0].offset, 0);
        assert_eq!(entries[0].size, primary_length);
        assert_eq!(entries[1].image_type, MpImageType::Undefined);
        assert_eq!(entries[1].offset, primary_length);
        assert_eq!(entries[1].size, gainmap_length);
    }

    #[test]
    fn test_create_mpf_header_multi_secondary() {
        let primary_len = 50000;
        let images = [
            (MpImageType::Undefined, 10000usize), // gain map
            (MpImageType::Disparity, 5000usize),  // depth map
        ];
        let header = create_mpf_header_typed(primary_len, &images, Some(100));

        // Should contain APP2 marker
        assert_eq!(header[0], 0xFF);
        assert_eq!(header[1], 0xE2);
        assert!(header.windows(4).any(|w| w == MPF_IDENTIFIER));
    }
}
