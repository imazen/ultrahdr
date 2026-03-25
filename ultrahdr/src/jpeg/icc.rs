//! ICC profile handling for JPEG images.

use moxcms::ColorProfile;
use ultrahdr_core::ColorGamut;

/// ICC profile APP2 identifier.
pub const ICC_IDENTIFIER: &[u8] = b"ICC_PROFILE\0";

/// Embed ICC profile data into JPEG APP2 marker(s).
///
/// Large ICC profiles may be split across multiple APP2 markers.
pub fn create_icc_markers(icc_data: &[u8]) -> Vec<Vec<u8>> {
    let max_chunk_size = 65533 - ICC_IDENTIFIER.len() - 2; // Max APP2 payload minus header

    let chunks: Vec<&[u8]> = icc_data.chunks(max_chunk_size).collect();
    let num_chunks = chunks.len() as u8;

    chunks
        .iter()
        .enumerate()
        .map(|(i, chunk)| {
            let mut marker = Vec::with_capacity(4 + ICC_IDENTIFIER.len() + 2 + chunk.len());

            // APP2 marker
            marker.push(0xFF);
            marker.push(0xE2);

            // Length (2 bytes)
            let length = 2 + ICC_IDENTIFIER.len() + 2 + chunk.len();
            marker.push(((length >> 8) & 0xFF) as u8);
            marker.push((length & 0xFF) as u8);

            // ICC_PROFILE identifier
            marker.extend_from_slice(ICC_IDENTIFIER);

            // Chunk index (1-based) and total chunks
            marker.push((i + 1) as u8);
            marker.push(num_chunks);

            // ICC data chunk
            marker.extend_from_slice(chunk);

            marker
        })
        .collect()
}

/// Extract ICC profile from JPEG data.
pub fn extract_icc_profile(data: &[u8]) -> Option<Vec<u8>> {
    let mut chunks: Vec<(u8, Vec<u8>)> = Vec::new();
    let mut pos = 0;

    while pos + 4 < data.len() {
        if data[pos] == 0xFF && data[pos + 1] == 0xE2 {
            let length = u16::from_be_bytes([data[pos + 2], data[pos + 3]]) as usize;

            if pos + 4 + ICC_IDENTIFIER.len() + 2 < data.len() {
                let marker_data = &data[pos + 4..];

                if marker_data.starts_with(ICC_IDENTIFIER) {
                    let chunk_num = marker_data[ICC_IDENTIFIER.len()];
                    let _total_chunks = marker_data[ICC_IDENTIFIER.len() + 1];

                    let data_start = ICC_IDENTIFIER.len() + 2;
                    let data_end = length - 2;

                    if data_start < data_end {
                        let chunk_data = marker_data[data_start..data_end].to_vec();
                        chunks.push((chunk_num, chunk_data));
                    }
                }
            }

            pos += 2 + length;
        } else {
            pos += 1;
        }
    }

    if chunks.is_empty() {
        return None;
    }

    // Sort by chunk number and concatenate
    chunks.sort_by_key(|(num, _)| *num);

    let mut profile = Vec::new();
    for (_, chunk) in chunks {
        profile.extend(chunk);
    }

    Some(profile)
}

/// Get the appropriate ICC profile bytes for a color gamut.
///
/// Uses moxcms to generate correct ICC v2 profiles with proper
/// chromatic adaptation and transfer curves.
pub fn get_icc_profile_for_gamut(gamut: ColorGamut) -> Vec<u8> {
    let profile = match gamut {
        ColorGamut::Bt709 => ColorProfile::new_srgb(),
        ColorGamut::DisplayP3 => ColorProfile::new_display_p3(),
        ColorGamut::Bt2020 => {
            // BT.2100 typically uses PQ or HLG, not sRGB TRC.
            // For the SDR base image's ICC tag we use BT.2020 primaries
            // with a gamma 2.2 TRC (the SDR rendition is always 8-bit).
            ColorProfile::new_bt2020()
        }
    };

    profile
        .encode()
        .expect("moxcms profile encoding should never fail for built-in profiles")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_create_icc_markers_small() {
        let profile = vec![0u8; 1000];
        let markers = create_icc_markers(&profile);

        assert_eq!(markers.len(), 1);
        assert_eq!(markers[0][0], 0xFF);
        assert_eq!(markers[0][1], 0xE2);
    }

    #[test]
    fn test_create_icc_markers_large() {
        let profile = vec![0u8; 100000];
        let markers = create_icc_markers(&profile);

        // Should be split across multiple markers
        assert!(markers.len() > 1);

        // Each should be valid APP2
        for marker in &markers {
            assert_eq!(marker[0], 0xFF);
            assert_eq!(marker[1], 0xE2);
        }
    }

    #[test]
    fn test_icc_roundtrip() {
        let original = get_icc_profile_for_gamut(ColorGamut::Bt709);
        let markers = create_icc_markers(&original);

        // Build fake JPEG with ICC markers
        let mut jpeg = vec![0xFF, 0xD8]; // SOI
        for m in &markers {
            jpeg.extend_from_slice(m);
        }
        jpeg.extend_from_slice(&[0xFF, 0xD9]); // EOI

        let extracted = extract_icc_profile(&jpeg).expect("should extract ICC");
        assert_eq!(extracted, original);
    }

    #[test]
    fn test_get_icc_profile_srgb() {
        let profile_bytes = get_icc_profile_for_gamut(ColorGamut::Bt709);
        // moxcms generates valid ICC profiles — verify basic structure
        assert!(profile_bytes.len() > 128, "ICC profile too short");
        // ICC signature at offset 36: 'acsp'
        assert_eq!(&profile_bytes[36..40], b"acsp");
    }

    #[test]
    fn test_get_icc_profile_p3() {
        let profile_bytes = get_icc_profile_for_gamut(ColorGamut::DisplayP3);
        assert!(profile_bytes.len() > 128);
        assert_eq!(&profile_bytes[36..40], b"acsp");
        // Should be parseable by moxcms
        let parsed = ColorProfile::new_from_slice(&profile_bytes);
        assert!(parsed.is_ok(), "P3 profile should be valid ICC");
    }

    #[test]
    fn test_get_icc_profile_bt2100() {
        let profile_bytes = get_icc_profile_for_gamut(ColorGamut::Bt2020);
        assert!(profile_bytes.len() > 128);
        assert_eq!(&profile_bytes[36..40], b"acsp");
        let parsed = ColorProfile::new_from_slice(&profile_bytes);
        assert!(parsed.is_ok(), "BT.2020 profile should be valid ICC");
    }

    #[test]
    fn test_extract_icc_profile_no_icc() {
        // JPEG with no ICC markers
        let jpeg = vec![0xFF, 0xD8, 0xFF, 0xD9];
        assert!(extract_icc_profile(&jpeg).is_none());
    }

    #[test]
    fn test_extract_icc_profile_empty_data() {
        assert!(extract_icc_profile(&[]).is_none());
    }

    #[test]
    fn test_create_icc_markers_empty() {
        let markers = create_icc_markers(&[]);
        // Empty input produces no chunks
        assert_eq!(markers.len(), 0);
    }

    #[test]
    fn test_create_icc_markers_exact_chunk_boundary() {
        // Create data that's exactly one chunk
        let max_chunk_size = 65533 - ICC_IDENTIFIER.len() - 2;
        let profile = vec![0xABu8; max_chunk_size];
        let markers = create_icc_markers(&profile);
        assert_eq!(markers.len(), 1);

        // One byte over should split
        let profile = vec![0xABu8; max_chunk_size + 1];
        let markers = create_icc_markers(&profile);
        assert_eq!(markers.len(), 2);
    }
}
