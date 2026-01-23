//! Gain map computation from HDR and SDR images.

use crate::color::gamut::rgb_to_luminance;
use crate::color::transfer::{apply_eotf, pq_eotf, srgb_eotf};
use crate::types::{ColorTransfer, GainMap, GainMapMetadata, PixelFormat, RawImage, Result};

/// Configuration for gain map computation.
#[derive(Debug, Clone)]
pub struct GainMapConfig {
    /// Scale factor for gain map (1 = same size as image, 4 = 1/4 size).
    pub scale_factor: u8,
    /// Gamma to apply to the gain map encoding.
    pub gamma: f32,
    /// Use multi-channel (RGB) gain map instead of single-channel luminance.
    pub multi_channel: bool,
    /// Minimum content boost (allows darkening, typically 1.0).
    pub min_content_boost: f32,
    /// Maximum content boost (HDR peak / SDR peak).
    pub max_content_boost: f32,
    /// Offset for SDR values to avoid division by zero.
    pub offset_sdr: f32,
    /// Offset for HDR values.
    pub offset_hdr: f32,
    /// Minimum HDR capacity (for metadata).
    pub hdr_capacity_min: f32,
    /// Maximum HDR capacity (for metadata).
    pub hdr_capacity_max: f32,
}

impl Default for GainMapConfig {
    fn default() -> Self {
        Self {
            scale_factor: 4,
            gamma: 1.0,
            multi_channel: false,
            min_content_boost: 1.0,
            max_content_boost: 6.0, // ~2.5 stops
            offset_sdr: 1.0 / 64.0,
            offset_hdr: 1.0 / 64.0,
            hdr_capacity_min: 1.0,
            hdr_capacity_max: 6.0,
        }
    }
}

/// Compute a gain map from HDR and SDR images.
///
/// The gain map represents the ratio between HDR and SDR pixel values,
/// encoded as 8-bit values in the range `[0, 255]`.
pub fn compute_gainmap(
    hdr: &RawImage,
    sdr: &RawImage,
    config: &GainMapConfig,
) -> Result<(GainMap, GainMapMetadata)> {
    // Validate inputs
    if hdr.width != sdr.width || hdr.height != sdr.height {
        return Err(crate::types::Error::DimensionMismatch {
            hdr_w: hdr.width,
            hdr_h: hdr.height,
            sdr_w: sdr.width,
            sdr_h: sdr.height,
        });
    }

    let scale = config.scale_factor.max(1) as u32;
    let gm_width = hdr.width.div_ceil(scale);
    let gm_height = hdr.height.div_ceil(scale);

    // Track actual min/max boost values found
    let mut actual_min_boost = f32::MAX;
    let mut actual_max_boost = f32::MIN;

    // Compute gain map
    let gainmap = if config.multi_channel {
        compute_multichannel_gainmap(
            hdr,
            sdr,
            gm_width,
            gm_height,
            scale,
            config,
            &mut actual_min_boost,
            &mut actual_max_boost,
        )?
    } else {
        compute_luminance_gainmap(
            hdr,
            sdr,
            gm_width,
            gm_height,
            scale,
            config,
            &mut actual_min_boost,
            &mut actual_max_boost,
        )?
    };

    // Clamp actual values to configured range
    actual_min_boost = actual_min_boost.max(config.min_content_boost);
    actual_max_boost = actual_max_boost.min(config.max_content_boost);

    // Build metadata
    let metadata = GainMapMetadata {
        max_content_boost: [actual_max_boost; 3],
        min_content_boost: [actual_min_boost; 3],
        gamma: [config.gamma; 3],
        offset_sdr: [config.offset_sdr; 3],
        offset_hdr: [config.offset_hdr; 3],
        hdr_capacity_min: config.hdr_capacity_min,
        hdr_capacity_max: config.hdr_capacity_max.max(actual_max_boost),
        use_base_color_space: true,
    };

    Ok((gainmap, metadata))
}

/// Compute single-channel (luminance) gain map.
#[allow(clippy::too_many_arguments)]
fn compute_luminance_gainmap(
    hdr: &RawImage,
    sdr: &RawImage,
    gm_width: u32,
    gm_height: u32,
    scale: u32,
    config: &GainMapConfig,
    actual_min_boost: &mut f32,
    actual_max_boost: &mut f32,
) -> Result<GainMap> {
    let mut gainmap = GainMap::new(gm_width, gm_height)?;

    let log_min = config.min_content_boost.ln();
    let log_max = config.max_content_boost.ln();
    let log_range = log_max - log_min;

    for gy in 0..gm_height {
        for gx in 0..gm_width {
            // Sample center pixel of the block
            let x = (gx * scale + scale / 2).min(hdr.width - 1);
            let y = (gy * scale + scale / 2).min(hdr.height - 1);

            // Get linear RGB values
            let hdr_rgb = get_linear_rgb(hdr, x, y);
            let sdr_rgb = get_linear_rgb(sdr, x, y);

            // Compute luminance
            let hdr_lum = rgb_to_luminance(hdr_rgb, hdr.gamut);
            let sdr_lum = rgb_to_luminance(sdr_rgb, sdr.gamut);

            // Compute gain (with offsets to avoid division by zero)
            let gain = (hdr_lum + config.offset_hdr) / (sdr_lum + config.offset_sdr);

            // Track actual range
            *actual_min_boost = actual_min_boost.min(gain);
            *actual_max_boost = actual_max_boost.max(gain);

            // Clamp and encode
            let gain_clamped = gain.clamp(config.min_content_boost, config.max_content_boost);
            let log_gain = gain_clamped.ln();

            // Normalize to `[0, 1]`
            let normalized = if log_range > 0.0 {
                (log_gain - log_min) / log_range
            } else {
                0.5
            };

            // Apply gamma and convert to 8-bit
            let gamma_corrected = normalized.powf(config.gamma);
            let encoded = (gamma_corrected * 255.0).round().clamp(0.0, 255.0) as u8;

            gainmap.data[(gy * gm_width + gx) as usize] = encoded;
        }
    }

    Ok(gainmap)
}

/// Compute multi-channel (RGB) gain map.
#[allow(clippy::too_many_arguments)]
fn compute_multichannel_gainmap(
    hdr: &RawImage,
    sdr: &RawImage,
    gm_width: u32,
    gm_height: u32,
    scale: u32,
    config: &GainMapConfig,
    actual_min_boost: &mut f32,
    actual_max_boost: &mut f32,
) -> Result<GainMap> {
    let mut gainmap = GainMap::new_multichannel(gm_width, gm_height)?;

    let log_min = config.min_content_boost.ln();
    let log_max = config.max_content_boost.ln();
    let log_range = log_max - log_min;

    for gy in 0..gm_height {
        for gx in 0..gm_width {
            let x = (gx * scale + scale / 2).min(hdr.width - 1);
            let y = (gy * scale + scale / 2).min(hdr.height - 1);

            let hdr_rgb = get_linear_rgb(hdr, x, y);
            let sdr_rgb = get_linear_rgb(sdr, x, y);

            for c in 0..3 {
                let gain =
                    (hdr_rgb[c] + config.offset_hdr) / (sdr_rgb[c] + config.offset_sdr).max(0.001);

                *actual_min_boost = actual_min_boost.min(gain);
                *actual_max_boost = actual_max_boost.max(gain);

                let gain_clamped = gain.clamp(config.min_content_boost, config.max_content_boost);
                let log_gain = gain_clamped.ln();

                let normalized = if log_range > 0.0 {
                    (log_gain - log_min) / log_range
                } else {
                    0.5
                };

                let gamma_corrected = normalized.powf(config.gamma);
                let encoded = (gamma_corrected * 255.0).round().clamp(0.0, 255.0) as u8;

                let idx = (gy * gm_width + gx) as usize * 3 + c;
                gainmap.data[idx] = encoded;
            }
        }
    }

    Ok(gainmap)
}

/// Extract linear RGB `[0,1]` from a raw image at the given pixel position.
fn get_linear_rgb(img: &RawImage, x: u32, y: u32) -> [f32; 3] {
    match img.format {
        PixelFormat::Rgba8 | PixelFormat::Rgb8 => {
            let bpp = if img.format == PixelFormat::Rgba8 {
                4
            } else {
                3
            };
            let idx = (y * img.stride + x * bpp as u32) as usize;
            let r = img.data[idx] as f32 / 255.0;
            let g = img.data[idx + 1] as f32 / 255.0;
            let b = img.data[idx + 2] as f32 / 255.0;

            // Apply EOTF based on transfer function
            match img.transfer {
                ColorTransfer::Srgb => [srgb_eotf(r), srgb_eotf(g), srgb_eotf(b)],
                ColorTransfer::Linear => [r, g, b],
                _ => [srgb_eotf(r), srgb_eotf(g), srgb_eotf(b)], // Assume sRGB for 8-bit
            }
        }

        PixelFormat::Rgba16F => {
            let idx = (y * img.stride + x * 8) as usize;
            // Read as f16 (half float) - assuming little-endian
            let r = half_to_f32(&img.data[idx..idx + 2]);
            let g = half_to_f32(&img.data[idx + 2..idx + 4]);
            let b = half_to_f32(&img.data[idx + 4..idx + 6]);
            [r, g, b] // Already linear
        }

        PixelFormat::Rgba32F => {
            let idx = (y * img.stride + x * 16) as usize;
            let r = f32::from_le_bytes([
                img.data[idx],
                img.data[idx + 1],
                img.data[idx + 2],
                img.data[idx + 3],
            ]);
            let g = f32::from_le_bytes([
                img.data[idx + 4],
                img.data[idx + 5],
                img.data[idx + 6],
                img.data[idx + 7],
            ]);
            let b = f32::from_le_bytes([
                img.data[idx + 8],
                img.data[idx + 9],
                img.data[idx + 10],
                img.data[idx + 11],
            ]);
            [r, g, b] // Already linear
        }

        PixelFormat::Rgba1010102Pq | PixelFormat::Rgba1010102Hlg => {
            let idx = (y * img.stride + x * 4) as usize;
            let packed = u32::from_le_bytes([
                img.data[idx],
                img.data[idx + 1],
                img.data[idx + 2],
                img.data[idx + 3],
            ]);

            let r = (packed & 0x3FF) as f32 / 1023.0;
            let g = ((packed >> 10) & 0x3FF) as f32 / 1023.0;
            let b = ((packed >> 20) & 0x3FF) as f32 / 1023.0;

            // Apply EOTF
            match img.format {
                PixelFormat::Rgba1010102Pq => [pq_eotf(r), pq_eotf(g), pq_eotf(b)],
                _ => [
                    apply_eotf(r, ColorTransfer::Hlg),
                    apply_eotf(g, ColorTransfer::Hlg),
                    apply_eotf(b, ColorTransfer::Hlg),
                ],
            }
        }

        PixelFormat::P010 => {
            // P010: 10-bit YUV 4:2:0, Y plane followed by interleaved UV
            let y_idx = (y * img.stride * 2 + x * 2) as usize;
            let y_val = u16::from_le_bytes([img.data[y_idx], img.data[y_idx + 1]]);
            let y_lum = (y_val >> 6) as f32 / 1023.0;

            // UV plane starts after Y plane
            let uv_offset = (img.height * img.stride * 2) as usize;
            let uv_y = y / 2;
            let uv_x = x / 2;
            let uv_idx =
                uv_offset + (uv_y as usize * img.stride as usize * 2) + (uv_x as usize * 4);

            let u_val = u16::from_le_bytes([img.data[uv_idx], img.data[uv_idx + 1]]);
            let v_val = u16::from_le_bytes([img.data[uv_idx + 2], img.data[uv_idx + 3]]);

            let u = (u_val >> 6) as f32 / 1023.0 - 0.5;
            let v = (v_val >> 6) as f32 / 1023.0 - 0.5;

            // BT.2020 YUV to RGB (for HDR content)
            let r = y_lum + 1.4746 * v;
            let g = y_lum - 0.1646 * u - 0.5714 * v;
            let b = y_lum + 1.8814 * u;

            // Apply PQ EOTF (P010 is typically PQ encoded)
            [pq_eotf(r), pq_eotf(g), pq_eotf(b)]
        }

        PixelFormat::Yuv420 => {
            // 8-bit YUV 4:2:0
            let y_idx = (y * img.stride + x) as usize;
            let y_val = img.data[y_idx] as f32 / 255.0;

            // U and V planes
            let uv_size = (img.stride / 2) * (img.height / 2);
            let u_offset = (img.height * img.stride) as usize;
            let v_offset = u_offset + uv_size as usize;

            let uv_x = x / 2;
            let uv_y = y / 2;
            let uv_idx = (uv_y * img.stride / 2 + uv_x) as usize;

            let u = img.data[u_offset + uv_idx] as f32 / 255.0 - 0.5;
            let v = img.data[v_offset + uv_idx] as f32 / 255.0 - 0.5;

            // BT.709 YUV to RGB
            let r = y_val + 1.5748 * v;
            let g = y_val - 0.1873 * u - 0.4681 * v;
            let b = y_val + 1.8556 * u;

            [srgb_eotf(r), srgb_eotf(g), srgb_eotf(b)]
        }

        PixelFormat::Gray8 => {
            let idx = (y * img.stride + x) as usize;
            let v = img.data[idx] as f32 / 255.0;
            let linear = srgb_eotf(v);
            [linear, linear, linear]
        }
    }
}

/// Convert half-precision float bytes to f32.
fn half_to_f32(bytes: &[u8]) -> f32 {
    let bits = u16::from_le_bytes([bytes[0], bytes[1]]);
    half::f16::from_bits(bits).to_f32()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ColorGamut;

    #[test]
    fn test_gainmap_config_default() {
        let config = GainMapConfig::default();
        assert_eq!(config.scale_factor, 4);
        assert_eq!(config.gamma, 1.0);
        assert!(!config.multi_channel);
    }

    #[test]
    fn test_compute_gainmap_basic() {
        // Create simple test images
        let mut hdr = RawImage::new(8, 8, PixelFormat::Rgba8).unwrap();
        hdr.gamut = ColorGamut::Bt709;
        hdr.transfer = ColorTransfer::Srgb;
        // Fill with mid-gray
        for i in 0..hdr.data.len() / 4 {
            hdr.data[i * 4] = 180; // R - brighter
            hdr.data[i * 4 + 1] = 180; // G
            hdr.data[i * 4 + 2] = 180; // B
            hdr.data[i * 4 + 3] = 255; // A
        }

        let mut sdr = RawImage::new(8, 8, PixelFormat::Rgba8).unwrap();
        sdr.gamut = ColorGamut::Bt709;
        sdr.transfer = ColorTransfer::Srgb;
        // Fill with darker gray
        for i in 0..sdr.data.len() / 4 {
            sdr.data[i * 4] = 128; // R
            sdr.data[i * 4 + 1] = 128; // G
            sdr.data[i * 4 + 2] = 128; // B
            sdr.data[i * 4 + 3] = 255; // A
        }

        let config = GainMapConfig {
            scale_factor: 2,
            ..Default::default()
        };

        let (gainmap, metadata) = compute_gainmap(&hdr, &sdr, &config).unwrap();

        // Check dimensions
        assert_eq!(gainmap.width, 4);
        assert_eq!(gainmap.height, 4);
        assert_eq!(gainmap.channels, 1);

        // Check metadata is populated
        assert!(metadata.max_content_boost[0] >= 1.0);
    }
}
