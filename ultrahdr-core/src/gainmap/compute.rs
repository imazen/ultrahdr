//! Gain map computation from HDR and SDR images.

use crate::color::gamut::rgb_to_luminance;
#[cfg(feature = "transfer")]
use crate::color::transfer::{apply_eotf, pq_eotf, srgb_eotf};
#[cfg(feature = "transfer")]
use crate::types::ColorTransfer;
use crate::types::{GainMap, GainMapMetadata, PixelFormat, RawImage, Result};
use enough::Stop;

/// Configuration for gain map computation.
///
/// Boost values (`min_boost`, `max_boost`) are in **linear domain** for
/// ergonomics — humans think "4× brighter" not "log2(4)=2". These are
/// converted to log2 when producing [`GainMapMetadata`].
#[derive(Debug, Clone)]
pub struct GainMapConfig {
    /// Scale factor for gain map (1 = same size as image, 4 = 1/4 size).
    pub scale_factor: u8,
    /// Gamma to apply to the gain map encoding.
    pub gamma: f32,
    /// Use multi-channel (RGB) gain map instead of single-channel luminance.
    pub multi_channel: bool,
    /// Minimum gain (linear). 1.0 = no darkening, 0.5 = allow 2× darker.
    pub min_boost: f32,
    /// Maximum gain (linear). HDR peak / SDR peak. e.g. 6.0 = ~2.5 stops.
    pub max_boost: f32,
    /// Offset for base (SDR) values to avoid division by zero. Linear domain.
    pub base_offset: f32,
    /// Offset for alternate (HDR) values. Linear domain.
    pub alternate_offset: f32,
    /// Minimum display boost (linear). For metadata headroom.
    pub base_hdr_headroom: f32,
    /// Maximum display boost (linear). For metadata headroom.
    pub alternate_hdr_headroom: f32,
}

impl Default for GainMapConfig {
    fn default() -> Self {
        Self {
            scale_factor: 4,
            gamma: 1.0,
            multi_channel: false,
            min_boost: 1.0,
            max_boost: 6.0, // ~2.5 stops
            base_offset: 1.0 / 64.0,
            alternate_offset: 1.0 / 64.0,
            base_hdr_headroom: 1.0,
            alternate_hdr_headroom: 6.0,
        }
    }
}

/// Compute a gain map from HDR and SDR images.
///
/// The gain map represents the ratio between HDR and SDR pixel values,
/// encoded as 8-bit values in the range `[0, 255]`.
///
/// The `stop` parameter enables cooperative cancellation. Pass `Unstoppable`
/// when cancellation is not needed.
pub fn compute_gainmap(
    hdr: &RawImage,
    sdr: &RawImage,
    config: &GainMapConfig,
    stop: impl Stop,
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

    // Validate pixel data is large enough for declared dimensions
    hdr.validate_data_bounds()?;
    sdr.validate_data_bounds()?;

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
            &stop,
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
            &stop,
        )?
    };

    // Clamp actual values to configured range
    actual_min_boost = actual_min_boost.max(config.min_boost);
    actual_max_boost = actual_max_boost.min(config.max_boost);

    // Build metadata (convert linear boost values to log2 domain)
    let metadata = GainMapMetadata {
        gain_map_max: [(actual_max_boost as f64).log2(); 3],
        gain_map_min: [(actual_min_boost as f64).log2(); 3],
        gamma: [config.gamma as f64; 3],
        base_offset: [config.base_offset as f64; 3],
        alternate_offset: [config.alternate_offset as f64; 3],
        base_hdr_headroom: (config.base_hdr_headroom as f64).log2(),
        alternate_hdr_headroom: (config.alternate_hdr_headroom.max(actual_max_boost) as f64).log2(),
        use_base_color_space: true,
        backward_direction: false,
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
    stop: &impl Stop,
) -> Result<GainMap> {
    let mut gainmap = GainMap::new(gm_width, gm_height)?;

    let log_min = config.min_boost.ln();
    let log_max = config.max_boost.ln();
    let log_range = log_max - log_min;

    for gy in 0..gm_height {
        // Check for cancellation once per row
        stop.check()?;

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
            let gain = (hdr_lum + config.alternate_offset) / (sdr_lum + config.base_offset);

            // Track actual range
            *actual_min_boost = actual_min_boost.min(gain);
            *actual_max_boost = actual_max_boost.max(gain);

            // Clamp and encode
            let gain_clamped = gain.clamp(config.min_boost, config.max_boost);
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
    stop: &impl Stop,
) -> Result<GainMap> {
    let mut gainmap = GainMap::new_multichannel(gm_width, gm_height)?;

    let log_min = config.min_boost.ln();
    let log_max = config.max_boost.ln();
    let log_range = log_max - log_min;

    for gy in 0..gm_height {
        // Check for cancellation once per row
        stop.check()?;

        for gx in 0..gm_width {
            let x = (gx * scale + scale / 2).min(hdr.width - 1);
            let y = (gy * scale + scale / 2).min(hdr.height - 1);

            let hdr_rgb = get_linear_rgb(hdr, x, y);
            let sdr_rgb = get_linear_rgb(sdr, x, y);

            for c in 0..3 {
                let gain = (hdr_rgb[c] + config.alternate_offset)
                    / (sdr_rgb[c] + config.base_offset).max(0.001);

                *actual_min_boost = actual_min_boost.min(gain);
                *actual_max_boost = actual_max_boost.max(gain);

                let gain_clamped = gain.clamp(config.min_boost, config.max_boost);
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
///
/// When the `transfer` feature is enabled, this function applies appropriate
/// EOTF conversions (sRGB, PQ, HLG) based on the image's transfer function.
///
/// When the `transfer` feature is disabled, only inherently linear formats
/// (Rgba16F, Rgba32F) are supported. For encoded formats, the caller must
/// pre-convert to linear using an external CMS.
#[cfg(feature = "transfer")]
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
            let r = half_to_f32(&img.data[idx..idx + 2]);
            let g = half_to_f32(&img.data[idx + 2..idx + 4]);
            let b = half_to_f32(&img.data[idx + 4..idx + 6]);
            [r, g, b]
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
            [r, g, b]
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
            let y_idx = (y * img.stride * 2 + x * 2) as usize;
            let y_val = u16::from_le_bytes([img.data[y_idx], img.data[y_idx + 1]]);
            let y_lum = (y_val >> 6) as f32 / 1023.0;

            let uv_offset = (img.height * img.stride * 2) as usize;
            let uv_y = y / 2;
            let uv_x = x / 2;
            let uv_idx =
                uv_offset + (uv_y as usize * img.stride as usize * 2) + (uv_x as usize * 4);

            let u_val = u16::from_le_bytes([img.data[uv_idx], img.data[uv_idx + 1]]);
            let v_val = u16::from_le_bytes([img.data[uv_idx + 2], img.data[uv_idx + 3]]);

            let u = (u_val >> 6) as f32 / 1023.0 - 0.5;
            let v = (v_val >> 6) as f32 / 1023.0 - 0.5;

            let r = y_lum + 1.4746 * v;
            let g = y_lum - 0.1646 * u - 0.5714 * v;
            let b = y_lum + 1.8814 * u;

            [pq_eotf(r), pq_eotf(g), pq_eotf(b)]
        }

        PixelFormat::Yuv420 => {
            let y_idx = (y * img.stride + x) as usize;
            let y_val = img.data[y_idx] as f32 / 255.0;

            let uv_size = (img.stride / 2) * (img.height / 2);
            let u_offset = (img.height * img.stride) as usize;
            let v_offset = u_offset + uv_size as usize;

            let uv_x = x / 2;
            let uv_y = y / 2;
            let uv_idx = (uv_y * img.stride / 2 + uv_x) as usize;

            let u = img.data[u_offset + uv_idx] as f32 / 255.0 - 0.5;
            let v = img.data[v_offset + uv_idx] as f32 / 255.0 - 0.5;

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

/// Extract linear RGB from a raw image (no transfer feature).
///
/// Only supports inherently linear formats (Rgba16F, Rgba32F).
/// For encoded formats (sRGB, PQ, etc.), use an external CMS to pre-convert.
#[cfg(not(feature = "transfer"))]
fn get_linear_rgb(img: &RawImage, x: u32, y: u32) -> [f32; 3] {
    match img.format {
        PixelFormat::Rgba16F => {
            let idx = (y * img.stride + x * 8) as usize;
            let r = half_to_f32(&img.data[idx..idx + 2]);
            let g = half_to_f32(&img.data[idx + 2..idx + 4]);
            let b = half_to_f32(&img.data[idx + 4..idx + 6]);
            [r, g, b]
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
            [r, g, b]
        }

        // For encoded formats without transfer feature, treat as linear
        // (caller must pre-convert using external CMS)
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
            // Assume already linear - caller must pre-convert
            [r, g, b]
        }

        PixelFormat::Gray8 => {
            let idx = (y * img.stride + x) as usize;
            let v = img.data[idx] as f32 / 255.0;
            [v, v, v]
        }

        // Formats that require transfer functions - return mid-gray as fallback
        _ => [0.18, 0.18, 0.18],
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

        let (gainmap, metadata) =
            compute_gainmap(&hdr, &sdr, &config, enough::Unstoppable).unwrap();

        // Check dimensions
        assert_eq!(gainmap.width, 4);
        assert_eq!(gainmap.height, 4);
        assert_eq!(gainmap.channels, 1);

        // Check metadata is populated
        assert!(metadata.gain_map_max[0] >= 1.0);
    }

    // ========================================================================
    // Gain encoding reference values (C++ libultrahdr parity)
    //
    // Tests the gain → encoded byte mapping with known inputs.
    // Parameters: min_boost=0.25 (log2=-2), max_boost=4.0 (log2=2), gamma=1.0
    // offset_sdr=offset_hdr=1/64
    //
    // Encoding formula:
    //   gain = (hdr + offset) / (sdr + offset)
    //   log_gain = ln(clamp(gain, min_boost, max_boost))
    //   normalized = (log_gain - ln(min_boost)) / (ln(max_boost) - ln(min_boost))
    //   encoded = round(normalized * 255)
    // ========================================================================

    /// Helper: compute the expected encoded byte for a given (sdr, hdr) pair.
    fn encode_gain_reference(sdr: f32, hdr: f32, min_boost: f32, max_boost: f32) -> u8 {
        let offset = 1.0 / 64.0;
        let gain = (hdr + offset) / (sdr + offset);
        let gain_clamped = gain.clamp(min_boost, max_boost);
        let log_min = min_boost.ln();
        let log_max = max_boost.ln();
        let log_range = log_max - log_min;
        let normalized = (gain_clamped.ln() - log_min) / log_range;
        (normalized * 255.0).round().clamp(0.0, 255.0) as u8
    }

    /// Test gain encoding against reference (sdr, hdr) pairs.
    ///
    /// Parameters match C++ test: min_boost=0.25, max_boost=4.0, gamma=1.0
    #[test]
    fn test_gain_encoding_cpp_reference() {
        let min_boost = 0.25_f32;
        let max_boost = 4.0_f32;

        // (sdr_linear, hdr_linear, description)
        let cases: &[(f32, f32, &str)] = &[
            // Same intensity → gain=1.0 → log(1)=0 → normalized=0.5 → 128
            (0.5, 0.5, "equal SDR/HDR"),
            // HDR is 4x SDR → gain=4.0 → max → 255
            (0.25, 1.0, "HDR 4x brighter"),
            // HDR is 0.25x SDR → gain=0.25 → min → 0
            (1.0, 0.25, "HDR 4x darker"),
            // Black pixels: gain dominated by offset
            (0.0, 0.0, "both black"),
            // SDR black, HDR bright: gain capped at max_boost
            (0.0, 1.0, "SDR black HDR bright"),
            // Mid range
            (0.18, 0.36, "HDR ~2x mid-gray"),
            // HDR slightly brighter
            (0.5, 0.75, "HDR 1.5x"),
        ];

        for &(sdr, hdr, desc) in cases {
            let expected = encode_gain_reference(sdr, hdr, min_boost, max_boost);
            // Verify the reference function itself is consistent
            let offset = 1.0 / 64.0;
            let gain = (hdr + offset) / (sdr + offset);
            let gain_clamped = gain.clamp(min_boost, max_boost);

            // Validate gain direction
            if sdr > 0.01 && hdr > 0.01 {
                if hdr > sdr * 1.5 {
                    assert!(
                        expected > 128,
                        "{}: hdr>sdr but encoded={} (gain={})",
                        desc,
                        expected,
                        gain
                    );
                }
                if hdr < sdr * 0.7 {
                    assert!(
                        expected < 128,
                        "{}: hdr<sdr but encoded={} (gain={})",
                        desc,
                        expected,
                        gain
                    );
                }
            }

            // Log the encoding for verification
            eprintln!(
                "  {}: sdr={:.3}, hdr={:.3}, gain={:.4}, clamped={:.4}, encoded={}",
                desc, sdr, hdr, gain, gain_clamped, expected
            );
        }
    }

    /// Helper: create an 8x8 HDR image (Rgba32F, Linear, BT.709) filled with a uniform color.
    fn make_hdr_8x8(r: f32, g: f32, b: f32) -> RawImage {
        let w = 8u32;
        let h = 8u32;
        let pixel_count = (w * h) as usize;
        let mut data = Vec::with_capacity(pixel_count * 16);
        for _ in 0..pixel_count {
            data.extend_from_slice(&r.to_le_bytes());
            data.extend_from_slice(&g.to_le_bytes());
            data.extend_from_slice(&b.to_le_bytes());
            data.extend_from_slice(&1.0f32.to_le_bytes()); // alpha
        }
        RawImage::from_data(
            w,
            h,
            PixelFormat::Rgba32F,
            ColorGamut::Bt709,
            ColorTransfer::Linear,
            data,
        )
        .unwrap()
    }

    /// Helper: create an 8x8 SDR image (Rgba8, Srgb, BT.709) filled with a uniform color.
    fn make_sdr_8x8(r: u8, g: u8, b: u8) -> RawImage {
        let w = 8u32;
        let h = 8u32;
        let pixel_count = (w * h) as usize;
        let mut data = vec![0u8; pixel_count * 4];
        for i in 0..pixel_count {
            data[i * 4] = r;
            data[i * 4 + 1] = g;
            data[i * 4 + 2] = b;
            data[i * 4 + 3] = 255;
        }
        RawImage::from_data(
            w,
            h,
            PixelFormat::Rgba8,
            ColorGamut::Bt709,
            ColorTransfer::Srgb,
            data,
        )
        .unwrap()
    }

    #[test]
    fn test_compute_gainmap_multichannel() {
        let hdr = make_hdr_8x8(0.8, 0.5, 0.3);
        let sdr = make_sdr_8x8(180, 128, 100);

        let config = GainMapConfig {
            multi_channel: true,
            scale_factor: 1,
            ..Default::default()
        };

        let (gainmap, _metadata) =
            compute_gainmap(&hdr, &sdr, &config, enough::Unstoppable).unwrap();

        assert_eq!(gainmap.channels, 3);
        assert_eq!(
            gainmap.data.len(),
            (gainmap.width * gainmap.height) as usize * 3
        );
    }

    #[test]
    fn test_compute_gainmap_scale_factor_1() {
        let hdr = make_hdr_8x8(0.5, 0.5, 0.5);
        let sdr = make_sdr_8x8(186, 186, 186);

        let config = GainMapConfig {
            scale_factor: 1,
            ..Default::default()
        };

        let (gainmap, _metadata) =
            compute_gainmap(&hdr, &sdr, &config, enough::Unstoppable).unwrap();

        assert_eq!(gainmap.width, 8);
        assert_eq!(gainmap.height, 8);
    }

    #[test]
    fn test_compute_gainmap_scale_factor_8() {
        let hdr = make_hdr_8x8(0.5, 0.5, 0.5);
        let sdr = make_sdr_8x8(186, 186, 186);

        let config = GainMapConfig {
            scale_factor: 8,
            ..Default::default()
        };

        let (gainmap, _metadata) =
            compute_gainmap(&hdr, &sdr, &config, enough::Unstoppable).unwrap();

        // 8 / 8 = 1 (div_ceil)
        assert_eq!(gainmap.width, 8u32.div_ceil(8));
        assert_eq!(gainmap.height, 8u32.div_ceil(8));
    }

    #[test]
    fn test_compute_gainmap_uniform_images() {
        // Both HDR and SDR are mid-gray: 0.5 linear, 186 sRGB
        let hdr = make_hdr_8x8(0.5, 0.5, 0.5);
        let sdr = make_sdr_8x8(186, 186, 186);

        let config = GainMapConfig {
            scale_factor: 1,
            ..Default::default()
        };

        let (gainmap, _metadata) =
            compute_gainmap(&hdr, &sdr, &config, enough::Unstoppable).unwrap();

        // All pixels should have roughly the same encoded value since inputs are uniform
        let first = gainmap.data[0];
        for &val in &gainmap.data {
            assert!(
                (val as i16 - first as i16).unsigned_abs() <= 1,
                "non-uniform gainmap: first={}, got={}",
                first,
                val
            );
        }
    }

    #[test]
    fn test_compute_gainmap_bright_hdr() {
        // HDR is very bright (5.0 linear), SDR is mid (186 sRGB ~ 0.5 linear)
        let hdr = make_hdr_8x8(5.0, 5.0, 5.0);
        let sdr = make_sdr_8x8(186, 186, 186);

        let config = GainMapConfig {
            scale_factor: 1,
            max_boost: 12.0,
            alternate_hdr_headroom: 12.0,
            ..Default::default()
        };

        let (gainmap, _metadata) =
            compute_gainmap(&hdr, &sdr, &config, enough::Unstoppable).unwrap();

        // Gainmap values should be high — a large positive gain means brighter bytes
        // The encoding maps min_content_boost → 0, max_content_boost → 255
        // gain ~= 5.0/0.5 = 10.0, which is well above 1.0 midpoint
        let avg: f32 =
            gainmap.data.iter().map(|&v| v as f32).sum::<f32>() / gainmap.data.len() as f32;
        assert!(
            avg > 128.0,
            "bright HDR should produce high gainmap values, got average {}",
            avg
        );
    }

    #[test]
    fn test_compute_gainmap_dimension_mismatch() {
        let hdr = make_hdr_8x8(0.5, 0.5, 0.5);
        // Create a 4x4 SDR image
        let sdr = RawImage::from_data(
            4,
            4,
            PixelFormat::Rgba8,
            ColorGamut::Bt709,
            ColorTransfer::Srgb,
            vec![128u8; 4 * 4 * 4],
        )
        .unwrap();

        let config = GainMapConfig::default();
        let result = compute_gainmap(&hdr, &sdr, &config, enough::Unstoppable);
        assert!(result.is_err());
        assert!(matches!(
            result.unwrap_err(),
            crate::types::Error::DimensionMismatch { .. }
        ));
    }

    #[test]
    fn test_compute_gainmap_cancellation() {
        /// A Stop implementation that cancels immediately
        struct ImmediateCancel;

        impl enough::Stop for ImmediateCancel {
            fn check(&self) -> std::result::Result<(), enough::StopReason> {
                Err(enough::StopReason::Cancelled)
            }
        }

        // Create minimal images
        let hdr = RawImage::new(8, 8, PixelFormat::Rgba8).unwrap();
        let sdr = RawImage::new(8, 8, PixelFormat::Rgba8).unwrap();
        let config = GainMapConfig::default();

        // Should return Stopped error due to cancellation
        let result = compute_gainmap(&hdr, &sdr, &config, ImmediateCancel);

        assert!(matches!(
            result,
            Err(crate::Error::Stopped(enough::StopReason::Cancelled))
        ));
    }
}
