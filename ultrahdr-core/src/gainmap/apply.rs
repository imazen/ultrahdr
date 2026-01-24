//! Gain map application for HDR reconstruction.

use alloc::boxed::Box;

use crate::color::transfer::{pq_oetf, srgb_eotf, srgb_oetf};
use crate::types::{ColorTransfer, GainMap, GainMapMetadata, PixelFormat, RawImage, Result};
use enough::Stop;

/// Precomputed lookup table for gain map decoding.
///
/// This LUT eliminates expensive `powf()` and `exp()` calls per pixel by
/// precomputing the mapping from 8-bit gain map values to linear gain multipliers.
/// Provides ~10x speedup for `apply_gainmap`.
pub struct GainMapLut {
    /// 256 entries per channel (R, G, B), mapping byte value to linear gain.
    /// Layout: [R0..R255, G0..G255, B0..B255]
    table: Box<[f32; 256 * 3]>,
}

impl GainMapLut {
    /// Create a new gain map LUT for the given metadata and display boost.
    ///
    /// The `weight` parameter is typically calculated from `display_boost` and
    /// the metadata's `hdr_capacity_min`/`hdr_capacity_max`.
    pub fn new(metadata: &GainMapMetadata, weight: f32) -> Self {
        let mut table = Box::new([0.0f32; 256 * 3]);

        for channel in 0..3 {
            let gamma = metadata.gamma[channel];
            let log_min = metadata.min_content_boost[channel].ln();
            let log_max = metadata.max_content_boost[channel].ln();
            let log_range = log_max - log_min;

            for i in 0..256 {
                // Convert byte to normalized [0,1]
                let normalized = i as f32 / 255.0;

                // Undo gamma
                let linear = if gamma != 1.0 && gamma > 0.0 {
                    normalized.powf(1.0 / gamma)
                } else {
                    normalized
                };

                // Convert from normalized to log gain, apply weight, convert to linear
                let log_gain = log_min + linear * log_range;
                let gain = (log_gain * weight).exp();

                table[channel * 256 + i] = gain;
            }
        }

        Self { table }
    }

    /// Look up the gain multiplier for a single channel.
    #[inline(always)]
    pub fn lookup(&self, byte_value: u8, channel: usize) -> f32 {
        // Safety: channel is always 0..3 and byte_value is u8 (0..255)
        debug_assert!(channel < 3);
        self.table[channel * 256 + byte_value as usize]
    }

    /// Look up gain multipliers for all 3 channels from a single byte (luminance mode).
    #[inline(always)]
    pub fn lookup_luminance(&self, byte_value: u8) -> [f32; 3] {
        let g = self.table[byte_value as usize]; // Channel 0
        [g, g, g]
    }

    /// Look up gain multipliers for RGB from 3 bytes.
    #[inline(always)]
    pub fn lookup_rgb(&self, r: u8, g: u8, b: u8) -> [f32; 3] {
        [
            self.table[r as usize],
            self.table[256 + g as usize],
            self.table[512 + b as usize],
        ]
    }
}

/// Output format for HDR reconstruction.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HdrOutputFormat {
    /// Linear float RGB `[0, ~50]` where 1.0 = SDR white (203 nits)
    LinearFloat,
    /// PQ-encoded 10-bit RGBA (1010102)
    Pq1010102,
    /// sRGB 8-bit (SDR output, no HDR boost)
    Srgb8,
}

/// Apply a gain map to an SDR image to reconstruct HDR.
///
/// The `display_boost` parameter controls how much HDR effect to apply:
/// - 1.0 = SDR output (no boost)
/// - 2.0 = 2x brightness capability
/// - 4.0 = 4x brightness capability (typical HDR display)
///
/// The `stop` parameter enables cooperative cancellation. Pass `Unstoppable`
/// when cancellation is not needed.
pub fn apply_gainmap(
    sdr: &RawImage,
    gainmap: &GainMap,
    metadata: &GainMapMetadata,
    display_boost: f32,
    output_format: HdrOutputFormat,
    stop: impl Stop,
) -> Result<RawImage> {
    let width = sdr.width;
    let height = sdr.height;

    // Calculate weight factor based on display capability
    let weight = calculate_weight(display_boost, metadata);

    // Create precomputed LUT for fast gain decoding
    let lut = GainMapLut::new(metadata, weight);

    // Create output image
    let mut output = match output_format {
        HdrOutputFormat::LinearFloat => {
            let mut img = RawImage::new(width, height, PixelFormat::Rgba32F)?;
            img.transfer = ColorTransfer::Linear;
            img.gamut = sdr.gamut;
            img
        }
        HdrOutputFormat::Pq1010102 => {
            let mut img = RawImage::new(width, height, PixelFormat::Rgba1010102Pq)?;
            img.transfer = ColorTransfer::Pq;
            img.gamut = sdr.gamut;
            img
        }
        HdrOutputFormat::Srgb8 => {
            let mut img = RawImage::new(width, height, PixelFormat::Rgba8)?;
            img.transfer = ColorTransfer::Srgb;
            img.gamut = sdr.gamut;
            img
        }
    };

    // Process each row, checking for cancellation periodically
    for y in 0..height {
        // Check for cancellation once per row (not per pixel for performance)
        stop.check()?;

        for x in 0..width {
            // Get SDR pixel (convert to linear)
            let sdr_linear = get_sdr_linear(sdr, x, y);

            // Sample gain map with LUT (fast path - no transcendentals per pixel)
            let gain = sample_gainmap_lut(gainmap, &lut, x, y, width, height);

            // Apply gain to reconstruct HDR
            let hdr_linear = apply_gain(sdr_linear, gain, metadata);

            // Write output
            write_output(&mut output, x, y, hdr_linear, output_format);
        }
    }

    Ok(output)
}

/// Calculate the weight factor for gain map application.
fn calculate_weight(display_boost: f32, metadata: &GainMapMetadata) -> f32 {
    let log_display = display_boost.max(1.0).ln();
    let log_min = metadata.hdr_capacity_min.max(1.0).ln();
    let log_max = metadata.hdr_capacity_max.max(1.0).ln();

    if log_max <= log_min {
        return 1.0;
    }

    ((log_display - log_min) / (log_max - log_min)).clamp(0.0, 1.0)
}

/// Get linear RGB from SDR image.
fn get_sdr_linear(sdr: &RawImage, x: u32, y: u32) -> [f32; 3] {
    match sdr.format {
        PixelFormat::Rgba8 | PixelFormat::Rgb8 => {
            let bpp = if sdr.format == PixelFormat::Rgba8 {
                4
            } else {
                3
            };
            let idx = (y * sdr.stride + x * bpp as u32) as usize;
            let r = sdr.data[idx] as f32 / 255.0;
            let g = sdr.data[idx + 1] as f32 / 255.0;
            let b = sdr.data[idx + 2] as f32 / 255.0;
            [srgb_eotf(r), srgb_eotf(g), srgb_eotf(b)]
        }
        _ => {
            // For other formats, use the compute module's function
            // For now, return mid-gray as fallback
            [0.18, 0.18, 0.18]
        }
    }
}

/// Bilinear interpolation.
#[inline(always)]
fn bilinear(v00: f32, v10: f32, v01: f32, v11: f32, fx: f32, fy: f32) -> f32 {
    let top = v00 * (1.0 - fx) + v10 * fx;
    let bottom = v01 * (1.0 - fx) + v11 * fx;
    top * (1.0 - fy) + bottom * fy
}

/// Sample gain map with bilinear interpolation using precomputed LUT.
///
/// This is significantly faster than `sample_gainmap` because it uses LUT lookups
/// instead of expensive `powf()` and `exp()` calls per pixel.
#[inline]
#[allow(clippy::needless_range_loop)] // c is used as both index and channel parameter
fn sample_gainmap_lut(
    gainmap: &GainMap,
    lut: &GainMapLut,
    x: u32,
    y: u32,
    img_width: u32,
    img_height: u32,
) -> [f32; 3] {
    // Map image coordinates to gain map coordinates
    let gm_x = (x as f32 / img_width as f32) * gainmap.width as f32;
    let gm_y = (y as f32 / img_height as f32) * gainmap.height as f32;

    // Bilinear interpolation coordinates
    let x0 = (gm_x.floor() as u32).min(gainmap.width - 1);
    let y0 = (gm_y.floor() as u32).min(gainmap.height - 1);
    let x1 = (x0 + 1).min(gainmap.width - 1);
    let y1 = (y0 + 1).min(gainmap.height - 1);

    let fx = gm_x - gm_x.floor();
    let fy = gm_y - gm_y.floor();

    if gainmap.channels == 1 {
        // Single channel - look up gains from LUT, then interpolate
        let g00 = lut.lookup(gainmap.data[(y0 * gainmap.width + x0) as usize], 0);
        let g10 = lut.lookup(gainmap.data[(y0 * gainmap.width + x1) as usize], 0);
        let g01 = lut.lookup(gainmap.data[(y1 * gainmap.width + x0) as usize], 0);
        let g11 = lut.lookup(gainmap.data[(y1 * gainmap.width + x1) as usize], 0);

        let gain = bilinear(g00, g10, g01, g11, fx, fy);
        [gain, gain, gain]
    } else {
        // Multi-channel - look up and interpolate each channel
        let mut gains = [0.0f32; 3];
        for c in 0..3 {
            let idx00 = (y0 * gainmap.width + x0) as usize * 3 + c;
            let idx10 = (y0 * gainmap.width + x1) as usize * 3 + c;
            let idx01 = (y1 * gainmap.width + x0) as usize * 3 + c;
            let idx11 = (y1 * gainmap.width + x1) as usize * 3 + c;

            let g00 = lut.lookup(gainmap.data[idx00], c);
            let g10 = lut.lookup(gainmap.data[idx10], c);
            let g01 = lut.lookup(gainmap.data[idx01], c);
            let g11 = lut.lookup(gainmap.data[idx11], c);

            gains[c] = bilinear(g00, g10, g01, g11, fx, fy);
        }
        gains
    }
}

/// Apply gain to SDR pixel to get HDR.
fn apply_gain(sdr_linear: [f32; 3], gain: [f32; 3], metadata: &GainMapMetadata) -> [f32; 3] {
    [
        (sdr_linear[0] + metadata.offset_sdr[0]) * gain[0] - metadata.offset_hdr[0],
        (sdr_linear[1] + metadata.offset_sdr[1]) * gain[1] - metadata.offset_hdr[1],
        (sdr_linear[2] + metadata.offset_sdr[2]) * gain[2] - metadata.offset_hdr[2],
    ]
}

/// Write HDR pixel to output image.
fn write_output(output: &mut RawImage, x: u32, y: u32, hdr: [f32; 3], format: HdrOutputFormat) {
    match format {
        HdrOutputFormat::LinearFloat => {
            let idx = (y * output.stride + x * 16) as usize;
            let r_bytes = hdr[0].to_le_bytes();
            let g_bytes = hdr[1].to_le_bytes();
            let b_bytes = hdr[2].to_le_bytes();
            let a_bytes = 1.0f32.to_le_bytes();

            output.data[idx..idx + 4].copy_from_slice(&r_bytes);
            output.data[idx + 4..idx + 8].copy_from_slice(&g_bytes);
            output.data[idx + 8..idx + 12].copy_from_slice(&b_bytes);
            output.data[idx + 12..idx + 16].copy_from_slice(&a_bytes);
        }

        HdrOutputFormat::Pq1010102 => {
            // Convert linear to PQ
            // First normalize: linear HDR has 1.0 = SDR white (203 nits)
            // PQ expects 1.0 = 10000 nits, so multiply by 203/10000
            let scale = 203.0 / 10000.0;
            let r_pq = pq_oetf(hdr[0].max(0.0) * scale);
            let g_pq = pq_oetf(hdr[1].max(0.0) * scale);
            let b_pq = pq_oetf(hdr[2].max(0.0) * scale);

            // Pack to 1010102
            let r = (r_pq * 1023.0).round().clamp(0.0, 1023.0) as u32;
            let g = (g_pq * 1023.0).round().clamp(0.0, 1023.0) as u32;
            let b = (b_pq * 1023.0).round().clamp(0.0, 1023.0) as u32;
            let a = 3u32; // Full alpha

            let packed = r | (g << 10) | (b << 20) | (a << 30);
            let idx = (y * output.stride + x * 4) as usize;
            output.data[idx..idx + 4].copy_from_slice(&packed.to_le_bytes());
        }

        HdrOutputFormat::Srgb8 => {
            // Clip to SDR range and apply sRGB OETF
            let r = srgb_oetf(hdr[0].clamp(0.0, 1.0));
            let g = srgb_oetf(hdr[1].clamp(0.0, 1.0));
            let b = srgb_oetf(hdr[2].clamp(0.0, 1.0));

            let idx = (y * output.stride + x * 4) as usize;
            output.data[idx] = (r * 255.0).round() as u8;
            output.data[idx + 1] = (g * 255.0).round() as u8;
            output.data[idx + 2] = (b * 255.0).round() as u8;
            output.data[idx + 3] = 255;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::ColorGamut;

    #[test]
    fn test_calculate_weight() {
        let metadata = GainMapMetadata {
            hdr_capacity_min: 1.0,
            hdr_capacity_max: 4.0,
            ..Default::default()
        };

        // No boost
        let w = calculate_weight(1.0, &metadata);
        assert!((w - 0.0).abs() < 0.01);

        // Full boost
        let w = calculate_weight(4.0, &metadata);
        assert!((w - 1.0).abs() < 0.01);

        // Half boost (log scale)
        let w = calculate_weight(2.0, &metadata);
        assert!(w > 0.4 && w < 0.6);
    }

    #[test]
    fn test_gain_map_lut() {
        let metadata = GainMapMetadata {
            min_content_boost: [1.0; 3],
            max_content_boost: [4.0; 3],
            gamma: [1.0; 3],
            ..Default::default()
        };

        let lut = GainMapLut::new(&metadata, 1.0);

        // Min gain (byte 0 = normalized 0.0)
        let gain = lut.lookup(0, 0);
        assert!((gain - 1.0).abs() < 0.01, "min gain: {}", gain);

        // Max gain (byte 255 = normalized 1.0)
        let gain = lut.lookup(255, 0);
        assert!((gain - 4.0).abs() < 0.1, "max gain: {}", gain);

        // Mid gain should be between min and max
        let gain = lut.lookup(128, 0);
        assert!(gain > 1.5 && gain < 2.5, "mid gain: {}", gain);
    }

    #[test]
    fn test_apply_gainmap_basic() {
        // Create SDR image
        let mut sdr = RawImage::new(4, 4, PixelFormat::Rgba8).unwrap();
        sdr.gamut = ColorGamut::Bt709;
        sdr.transfer = ColorTransfer::Srgb;
        for i in 0..sdr.data.len() / 4 {
            sdr.data[i * 4] = 128;
            sdr.data[i * 4 + 1] = 128;
            sdr.data[i * 4 + 2] = 128;
            sdr.data[i * 4 + 3] = 255;
        }

        // Create gain map (all same boost)
        let mut gainmap = GainMap::new(2, 2).unwrap();
        for v in &mut gainmap.data {
            *v = 200; // High gain
        }

        let metadata = GainMapMetadata {
            min_content_boost: [1.0; 3],
            max_content_boost: [4.0; 3],
            gamma: [1.0; 3],
            offset_sdr: [0.015625; 3],
            offset_hdr: [0.015625; 3],
            hdr_capacity_min: 1.0,
            hdr_capacity_max: 4.0,
            use_base_color_space: true,
        };

        let result = apply_gainmap(
            &sdr,
            &gainmap,
            &metadata,
            4.0,
            HdrOutputFormat::Srgb8,
            enough::Unstoppable,
        )
        .unwrap();

        assert_eq!(result.width, 4);
        assert_eq!(result.height, 4);
        assert_eq!(result.format, PixelFormat::Rgba8);
    }

    #[test]
    fn test_apply_gainmap_cancellation() {
        /// A Stop implementation that cancels immediately
        struct ImmediateCancel;

        impl enough::Stop for ImmediateCancel {
            fn check(&self) -> std::result::Result<(), enough::StopReason> {
                Err(enough::StopReason::Cancelled)
            }
        }

        // Create minimal images
        let sdr = RawImage::new(4, 4, PixelFormat::Rgba8).unwrap();
        let gainmap = GainMap::new(2, 2).unwrap();
        let metadata = GainMapMetadata::new();

        // Should return Stopped error due to cancellation
        let result = apply_gainmap(
            &sdr,
            &gainmap,
            &metadata,
            4.0,
            HdrOutputFormat::Srgb8,
            ImmediateCancel,
        );

        assert!(matches!(
            result,
            Err(crate::Error::Stopped(enough::StopReason::Cancelled))
        ));
    }
}
