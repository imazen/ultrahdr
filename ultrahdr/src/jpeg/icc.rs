//! ICC profile handling for JPEG images.

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

/// Generate a minimal sRGB ICC profile.
///
/// This creates a basic ICC v2 profile for sRGB color space.
pub fn generate_srgb_profile() -> Vec<u8> {
    // Use moxcms to generate the profile if available,
    // otherwise return a pre-computed minimal profile
    generate_matrix_profile(
        "sRGB",
        // sRGB primaries (D65)
        [0.6400, 0.3300], // Red
        [0.3000, 0.6000], // Green
        [0.1500, 0.0600], // Blue
        [0.3127, 0.3290], // D65 white point
        2.2,              // Approximate gamma
    )
}

/// Generate a Display P3 ICC profile.
pub fn generate_p3_profile() -> Vec<u8> {
    generate_matrix_profile(
        "Display P3",
        // P3 primaries (D65)
        [0.6800, 0.3200], // Red
        [0.2650, 0.6900], // Green
        [0.1500, 0.0600], // Blue
        [0.3127, 0.3290], // D65 white point
        2.2,              // Approximate gamma
    )
}

/// Generate a minimal matrix-based ICC profile.
fn generate_matrix_profile(
    description: &str,
    red_xy: [f32; 2],
    green_xy: [f32; 2],
    blue_xy: [f32; 2],
    white_xy: [f32; 2],
    gamma: f32,
) -> Vec<u8> {
    // ICC profile header (128 bytes)
    let mut profile = Vec::with_capacity(1024);

    // We'll build a minimal ICC v2.1 profile
    // This is a simplified implementation

    // Profile size placeholder (will update at end)
    profile.extend_from_slice(&[0u8; 4]);

    // Preferred CMM type (0 = any)
    profile.extend_from_slice(&[0u8; 4]);

    // Profile version (2.1.0)
    profile.extend_from_slice(&[0x02, 0x10, 0x00, 0x00]);

    // Profile/Device class: Display
    profile.extend_from_slice(b"mntr");

    // Color space: RGB
    profile.extend_from_slice(b"RGB ");

    // PCS: XYZ
    profile.extend_from_slice(b"XYZ ");

    // Date/time (zeros for simplicity)
    profile.extend_from_slice(&[0u8; 12]);

    // Profile signature 'acsp'
    profile.extend_from_slice(b"acsp");

    // Primary platform: Apple (common for Display profiles)
    profile.extend_from_slice(b"APPL");

    // Flags (0)
    profile.extend_from_slice(&[0u8; 4]);

    // Device manufacturer (0)
    profile.extend_from_slice(&[0u8; 4]);

    // Device model (0)
    profile.extend_from_slice(&[0u8; 4]);

    // Device attributes (0)
    profile.extend_from_slice(&[0u8; 8]);

    // Rendering intent: Perceptual
    profile.extend_from_slice(&[0u8; 4]);

    // PCS illuminant (D50 XYZ)
    profile.extend_from_slice(&xyz_to_bytes(0.9642, 1.0, 0.8249));

    // Profile creator (0)
    profile.extend_from_slice(&[0u8; 4]);

    // Profile ID (zeros, not computed)
    profile.extend_from_slice(&[0u8; 16]);

    // Reserved (28 bytes to reach 128)
    profile.extend_from_slice(&[0u8; 28]);

    // Tag table
    let num_tags = 9u32;
    profile.extend_from_slice(&num_tags.to_be_bytes());

    // Calculate tag data offsets
    let tag_table_size = 4 + num_tags as usize * 12;
    let mut data_offset = 128 + tag_table_size;

    // Tag entries and data
    let mut tag_data = Vec::new();

    // Convert primaries from xy to XYZ
    let (red_xyz, green_xyz, blue_xyz) = xy_to_xyz_primaries(red_xy, green_xy, blue_xy, white_xy);

    // 1. Profile description (desc)
    let desc_data = create_text_tag(description);
    add_tag_entry(&mut profile, b"desc", data_offset, desc_data.len());
    data_offset += align4(desc_data.len());
    tag_data.extend(desc_data);
    pad_to_align4(&mut tag_data);

    // 2. Red colorant (rXYZ)
    let rxyz_data = create_xyz_tag(red_xyz);
    add_tag_entry(&mut profile, b"rXYZ", data_offset, rxyz_data.len());
    data_offset += align4(rxyz_data.len());
    tag_data.extend(rxyz_data);
    pad_to_align4(&mut tag_data);

    // 3. Green colorant (gXYZ)
    let gxyz_data = create_xyz_tag(green_xyz);
    add_tag_entry(&mut profile, b"gXYZ", data_offset, gxyz_data.len());
    data_offset += align4(gxyz_data.len());
    tag_data.extend(gxyz_data);
    pad_to_align4(&mut tag_data);

    // 4. Blue colorant (bXYZ)
    let bxyz_data = create_xyz_tag(blue_xyz);
    add_tag_entry(&mut profile, b"bXYZ", data_offset, bxyz_data.len());
    data_offset += align4(bxyz_data.len());
    tag_data.extend(bxyz_data);
    pad_to_align4(&mut tag_data);

    // 5. Red TRC (rTRC)
    let trc_data = create_gamma_trc(gamma);
    add_tag_entry(&mut profile, b"rTRC", data_offset, trc_data.len());
    data_offset += align4(trc_data.len());
    tag_data.extend(trc_data.clone());
    pad_to_align4(&mut tag_data);

    // 6. Green TRC (gTRC) - same as red
    add_tag_entry(&mut profile, b"gTRC", data_offset, trc_data.len());
    data_offset += align4(trc_data.len());
    tag_data.extend(trc_data.clone());
    pad_to_align4(&mut tag_data);

    // 7. Blue TRC (bTRC) - same as red
    add_tag_entry(&mut profile, b"bTRC", data_offset, trc_data.len());
    data_offset += align4(trc_data.len());
    tag_data.extend(trc_data);
    pad_to_align4(&mut tag_data);

    // 8. White point (wtpt)
    let white_xyz = xy_to_xyz(white_xy[0], white_xy[1], 1.0);
    let wtpt_data = create_xyz_tag(white_xyz);
    add_tag_entry(&mut profile, b"wtpt", data_offset, wtpt_data.len());
    data_offset += align4(wtpt_data.len());
    tag_data.extend(wtpt_data);
    pad_to_align4(&mut tag_data);

    // 9. Copyright (cprt)
    let cprt_data = create_text_tag("Public Domain");
    add_tag_entry(&mut profile, b"cprt", data_offset, cprt_data.len());
    tag_data.extend(cprt_data);
    pad_to_align4(&mut tag_data);

    // Append tag data
    profile.extend(tag_data);

    // Update profile size
    let size = profile.len() as u32;
    profile[0..4].copy_from_slice(&size.to_be_bytes());

    profile
}

fn add_tag_entry(profile: &mut Vec<u8>, sig: &[u8; 4], offset: usize, size: usize) {
    profile.extend_from_slice(sig);
    profile.extend_from_slice(&(offset as u32).to_be_bytes());
    profile.extend_from_slice(&(size as u32).to_be_bytes());
}

fn create_text_tag(text: &str) -> Vec<u8> {
    let mut data = Vec::new();
    data.extend_from_slice(b"desc");
    data.extend_from_slice(&[0u8; 4]); // Reserved
    data.extend_from_slice(&((text.len() + 1) as u32).to_be_bytes());
    data.extend_from_slice(text.as_bytes());
    data.push(0); // Null terminator
    data
}

fn create_xyz_tag(xyz: [f32; 3]) -> Vec<u8> {
    let mut data = Vec::new();
    data.extend_from_slice(b"XYZ ");
    data.extend_from_slice(&[0u8; 4]); // Reserved
    data.extend_from_slice(&xyz_to_bytes(xyz[0], xyz[1], xyz[2]));
    data
}

fn create_gamma_trc(gamma: f32) -> Vec<u8> {
    let mut data = Vec::new();
    data.extend_from_slice(b"curv");
    data.extend_from_slice(&[0u8; 4]); // Reserved
    data.extend_from_slice(&1u32.to_be_bytes()); // Count = 1 (gamma value)

    // Gamma as u8.8 fixed point
    let gamma_fixed = (gamma * 256.0) as u16;
    data.extend_from_slice(&gamma_fixed.to_be_bytes());

    data
}

fn xyz_to_bytes(x: f32, y: f32, z: f32) -> [u8; 12] {
    // Convert to s15Fixed16 format
    let x_fixed = (x * 65536.0) as i32;
    let y_fixed = (y * 65536.0) as i32;
    let z_fixed = (z * 65536.0) as i32;

    let mut bytes = [0u8; 12];
    bytes[0..4].copy_from_slice(&x_fixed.to_be_bytes());
    bytes[4..8].copy_from_slice(&y_fixed.to_be_bytes());
    bytes[8..12].copy_from_slice(&z_fixed.to_be_bytes());
    bytes
}

fn xy_to_xyz(x: f32, y: f32, big_y: f32) -> [f32; 3] {
    if y == 0.0 {
        return [0.0, 0.0, 0.0];
    }
    let big_x = x * big_y / y;
    let big_z = (1.0 - x - y) * big_y / y;
    [big_x, big_y, big_z]
}

fn xy_to_xyz_primaries(
    red_xy: [f32; 2],
    green_xy: [f32; 2],
    blue_xy: [f32; 2],
    _white_xy: [f32; 2],
) -> ([f32; 3], [f32; 3], [f32; 3]) {
    // Simplified - just convert each primary
    // A full implementation would compute the proper matrix using white_xy
    let red_xyz = xy_to_xyz(red_xy[0], red_xy[1], 0.4);
    let green_xyz = xy_to_xyz(green_xy[0], green_xy[1], 0.7);
    let blue_xyz = xy_to_xyz(blue_xy[0], blue_xy[1], 0.1);
    (red_xyz, green_xyz, blue_xyz)
}

fn align4(size: usize) -> usize {
    (size + 3) & !3
}

fn pad_to_align4(data: &mut Vec<u8>) {
    while data.len() % 4 != 0 {
        data.push(0);
    }
}

/// Get the appropriate ICC profile for a color gamut.
pub fn get_icc_profile_for_gamut(gamut: ColorGamut) -> Vec<u8> {
    match gamut {
        ColorGamut::Bt709 => generate_srgb_profile(),
        ColorGamut::DisplayP3 => generate_p3_profile(),
        ColorGamut::Bt2100 => {
            // BT.2100 typically uses PQ or HLG, not ICC
            // Return sRGB as fallback for SDR representation
            generate_srgb_profile()
        }
    }
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
}
