//! ISO 21496-1 binary metadata format for gain maps.
//!
//! This is the standardized binary format for gain map metadata,
//! as an alternative to XMP.

use crate::types::{Error, Fraction, GainMapMetadata, Result};

/// ISO 21496-1 metadata version.
pub const ISO_VERSION: u8 = 0;

/// Flags byte layout:
/// - Bit 0: Multi-channel gain map (0 = single channel)
/// - Bit 1: Use base color space (1 = gain map uses base image color space)
/// - Bit 2: Backward direction (0 = base is SDR, 1 = base is HDR)
/// - Bits 3-7: Reserved
const FLAG_MULTI_CHANNEL: u8 = 0x01;
const FLAG_USE_BASE_CG: u8 = 0x02;
const FLAG_BACKWARD_DIR: u8 = 0x04;

/// Serialize gain map metadata to ISO 21496-1 binary format.
pub fn serialize_iso21496(metadata: &GainMapMetadata) -> Vec<u8> {
    let mut data = Vec::with_capacity(128);

    // Version (1 byte)
    data.push(ISO_VERSION);

    // Flags (1 byte)
    let mut flags = 0u8;
    if !metadata.is_single_channel() {
        flags |= FLAG_MULTI_CHANNEL;
    }
    if metadata.use_base_color_space {
        flags |= FLAG_USE_BASE_CG;
    }
    // Backward direction is false (base is SDR)
    data.push(flags);

    let channels = if flags & FLAG_MULTI_CHANNEL != 0 {
        3
    } else {
        1
    };

    // Base HDR headroom (fraction) - log2(hdr_capacity_min)
    let base_headroom = Fraction::from_f32(metadata.hdr_capacity_min.log2());
    write_fraction(&mut data, base_headroom);

    // Alternate HDR headroom (fraction) - log2(hdr_capacity_max)
    let alt_headroom = Fraction::from_f32(metadata.hdr_capacity_max.log2());
    write_fraction(&mut data, alt_headroom);

    // Per-channel values
    for i in 0..channels {
        // Gain map min (fraction) - log2(min_content_boost)
        let min_val = Fraction::from_f32(metadata.min_content_boost[i].log2());
        write_fraction(&mut data, min_val);

        // Gain map max (fraction) - log2(max_content_boost)
        let max_val = Fraction::from_f32(metadata.max_content_boost[i].log2());
        write_fraction(&mut data, max_val);

        // Gamma (fraction)
        let gamma = Fraction::from_f32(metadata.gamma[i]);
        write_fraction(&mut data, gamma);

        // Base offset (fraction) - offset_sdr
        let base_offset = Fraction::from_f32(metadata.offset_sdr[i]);
        write_fraction(&mut data, base_offset);

        // Alternate offset (fraction) - offset_hdr
        let alt_offset = Fraction::from_f32(metadata.offset_hdr[i]);
        write_fraction(&mut data, alt_offset);
    }

    data
}

/// Deserialize ISO 21496-1 binary metadata.
pub fn deserialize_iso21496(data: &[u8]) -> Result<GainMapMetadata> {
    if data.len() < 2 {
        return Err(Error::InvalidMetadata("ISO metadata too short".into()));
    }

    let mut pos = 0;

    // Version
    let version = data[pos];
    pos += 1;
    if version > ISO_VERSION {
        return Err(Error::InvalidMetadata(format!(
            "Unsupported ISO version: {}",
            version
        )));
    }

    // Flags
    let flags = data[pos];
    pos += 1;
    let multi_channel = flags & FLAG_MULTI_CHANNEL != 0;
    let use_base_cg = flags & FLAG_USE_BASE_CG != 0;
    let backward_dir = flags & FLAG_BACKWARD_DIR != 0;

    let channels = if multi_channel { 3 } else { 1 };

    // We need at least: 2 + 8*2 (headrooms) + channels * 5 * 8 (per-channel fractions)
    let min_size = 2 + 16 + channels * 40;
    if data.len() < min_size {
        return Err(Error::InvalidMetadata("ISO metadata truncated".into()));
    }

    // Base HDR headroom
    let (base_headroom, new_pos) = read_fraction(data, pos)?;
    pos = new_pos;
    let hdr_capacity_min = 2.0f32.powf(base_headroom.to_f32());

    // Alternate HDR headroom
    let (alt_headroom, new_pos) = read_fraction(data, pos)?;
    pos = new_pos;
    let hdr_capacity_max = 2.0f32.powf(alt_headroom.to_f32());

    let mut metadata = GainMapMetadata {
        hdr_capacity_min,
        hdr_capacity_max,
        use_base_color_space: use_base_cg,
        ..Default::default()
    };

    // Per-channel values
    for i in 0..channels {
        let idx = if multi_channel { i } else { 0 };

        // Gain map min
        let (min_frac, new_pos) = read_fraction(data, pos)?;
        pos = new_pos;
        let min_val = 2.0f32.powf(min_frac.to_f32());

        // Gain map max
        let (max_frac, new_pos) = read_fraction(data, pos)?;
        pos = new_pos;
        let max_val = 2.0f32.powf(max_frac.to_f32());

        // Gamma
        let (gamma_frac, new_pos) = read_fraction(data, pos)?;
        pos = new_pos;

        // Base offset
        let (base_offset_frac, new_pos) = read_fraction(data, pos)?;
        pos = new_pos;

        // Alternate offset
        let (alt_offset_frac, new_pos) = read_fraction(data, pos)?;
        pos = new_pos;

        if multi_channel {
            metadata.min_content_boost[idx] = min_val;
            metadata.max_content_boost[idx] = max_val;
            metadata.gamma[idx] = gamma_frac.to_f32();
            metadata.offset_sdr[idx] = base_offset_frac.to_f32();
            metadata.offset_hdr[idx] = alt_offset_frac.to_f32();
        } else {
            // Single channel - apply to all
            metadata.min_content_boost = [min_val; 3];
            metadata.max_content_boost = [max_val; 3];
            metadata.gamma = [gamma_frac.to_f32(); 3];
            metadata.offset_sdr = [base_offset_frac.to_f32(); 3];
            metadata.offset_hdr = [alt_offset_frac.to_f32(); 3];
        }
    }

    // Handle backward direction (swap SDR/HDR interpretation)
    if backward_dir {
        std::mem::swap(&mut metadata.offset_sdr, &mut metadata.offset_hdr);
    }

    Ok(metadata)
}

/// Write a fraction to the buffer (8 bytes: 4 for numerator, 4 for denominator).
fn write_fraction(buf: &mut Vec<u8>, frac: Fraction) {
    buf.extend_from_slice(&frac.numerator.to_be_bytes());
    buf.extend_from_slice(&frac.denominator.to_be_bytes());
}

/// Read a fraction from the buffer.
fn read_fraction(data: &[u8], pos: usize) -> Result<(Fraction, usize)> {
    if pos + 8 > data.len() {
        return Err(Error::InvalidMetadata("Unexpected end of ISO data".into()));
    }

    let numerator = i32::from_be_bytes([data[pos], data[pos + 1], data[pos + 2], data[pos + 3]]);
    let denominator =
        u32::from_be_bytes([data[pos + 4], data[pos + 5], data[pos + 6], data[pos + 7]]);

    Ok((Fraction::new(numerator, denominator), pos + 8))
}

/// Create APP2 marker with ISO 21496-1 data.
pub fn create_iso_app2_marker(iso_data: &[u8]) -> Vec<u8> {
    // ISO 21496-1 uses a specific APP2 marker format
    let namespace = b"urn:iso:std:iso:ts:21496:-1\0";

    let total_length = 2 + namespace.len() + iso_data.len();

    let mut marker = Vec::with_capacity(2 + total_length);
    marker.push(0xFF);
    marker.push(0xE2); // APP2
    marker.push(((total_length >> 8) & 0xFF) as u8);
    marker.push((total_length & 0xFF) as u8);
    marker.extend_from_slice(namespace);
    marker.extend_from_slice(iso_data);

    marker
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_serialize_deserialize_single_channel() {
        let original = GainMapMetadata {
            min_content_boost: [1.0; 3],
            max_content_boost: [4.0; 3],
            gamma: [1.0; 3],
            offset_sdr: [0.015625; 3],
            offset_hdr: [0.015625; 3],
            hdr_capacity_min: 1.0,
            hdr_capacity_max: 4.0,
            use_base_color_space: true,
        };

        let serialized = serialize_iso21496(&original);
        let parsed = deserialize_iso21496(&serialized).unwrap();

        // Check values match (with tolerance for fraction conversion)
        assert!((parsed.max_content_boost[0] - 4.0).abs() < 0.01);
        assert!((parsed.hdr_capacity_max - 4.0).abs() < 0.01);
        assert!((parsed.gamma[0] - 1.0).abs() < 0.01);
        assert!(parsed.use_base_color_space);
    }

    #[test]
    fn test_fraction_roundtrip() {
        let values = [0.0, 0.5, 1.0, 2.0, -1.0, 0.015625];

        for &v in &values {
            let frac = Fraction::from_f32(v);
            let back = frac.to_f32();
            assert!(
                (v - back).abs() < 0.0001,
                "Fraction roundtrip failed for {}: got {}",
                v,
                back
            );
        }
    }
}
