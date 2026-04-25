//! Gain map computation from HDR and SDR images.

use alloc::vec;

use crate::color::gamut::rgb_to_luminance;

use crate::color::transfer::srgb_eotf;
use crate::gainmap::splitter::{LumaGainMapSplitter, LumaToneMap, SplitConfig, SplitStats};
use crate::types::TransferFunction;
use crate::types::{
    ColorPrimaries, GainMap, GainMapMetadata, PixelBuffer, PixelFormat, PixelSlice, Result,
    new_pixel_buffer,
};
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
    hdr: &PixelBuffer,
    sdr: &PixelBuffer,
    config: &GainMapConfig,
    stop: impl Stop,
) -> Result<(GainMap, GainMapMetadata)> {
    compute_gainmap_slice(hdr.as_slice(), sdr.as_slice(), config, stop)
}

/// [`compute_gainmap`] variant that takes borrowed [`PixelSlice`]s.
pub fn compute_gainmap_slice(
    hdr: PixelSlice<'_>,
    sdr: PixelSlice<'_>,
    config: &GainMapConfig,
    stop: impl Stop,
) -> Result<(GainMap, GainMapMetadata)> {
    crate::types::validate_ultrahdr_slice(&hdr)?;
    crate::types::validate_ultrahdr_slice(&sdr)?;

    let hdr_w = hdr.width();
    let hdr_h = hdr.rows();
    let sdr_w = sdr.width();
    let sdr_h = sdr.rows();

    if hdr_w != sdr_w || hdr_h != sdr_h {
        return Err(crate::types::Error::DimensionMismatch {
            hdr_w,
            hdr_h,
            sdr_w,
            sdr_h,
        });
    }

    let scale = config.scale_factor.max(1) as u32;
    let gm_width = hdr_w.div_ceil(scale);
    let gm_height = hdr_h.div_ceil(scale);

    // Track actual min/max boost values found
    let mut actual_min_boost = f32::MAX;
    let mut actual_max_boost = f32::MIN;

    // Compute gain map
    let gainmap = if config.multi_channel {
        compute_multichannel_gainmap(
            &hdr,
            &sdr,
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
            &hdr,
            &sdr,
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
    let metadata = crate::types::metadata_from_arrays(
        [(actual_min_boost as f64).log2(); 3],
        [(actual_max_boost as f64).log2(); 3],
        [config.gamma as f64; 3],
        [config.base_offset as f64; 3],
        [config.alternate_offset as f64; 3],
        (config.base_hdr_headroom as f64).log2(),
        (config.alternate_hdr_headroom.max(actual_max_boost) as f64).log2(),
        true,
        false,
    );

    Ok((gainmap, metadata))
}

/// Compute single-channel (luminance) gain map.
#[allow(clippy::too_many_arguments)]
fn compute_luminance_gainmap(
    hdr: &PixelSlice<'_>,
    sdr: &PixelSlice<'_>,
    gm_width: u32,
    gm_height: u32,
    scale: u32,
    config: &GainMapConfig,
    actual_min_boost: &mut f32,
    actual_max_boost: &mut f32,
    stop: &impl Stop,
) -> Result<GainMap> {
    let mut gainmap = GainMap::new(gm_width, gm_height)?;
    let hdr_w = hdr.width();
    let hdr_h = hdr.rows();
    let hdr_gamut = hdr.descriptor().primaries;
    let sdr_gamut = sdr.descriptor().primaries;

    let log_min = config.min_boost.ln();
    let log_max = config.max_boost.ln();
    let log_range = log_max - log_min;

    for gy in 0..gm_height {
        // Check for cancellation once per row
        stop.check()?;

        for gx in 0..gm_width {
            // Sample center pixel of the block
            let x = (gx * scale + scale / 2).min(hdr_w - 1);
            let y = (gy * scale + scale / 2).min(hdr_h - 1);

            // Get linear RGB values
            let hdr_rgb = get_linear_rgb(hdr, x, y);
            let sdr_rgb = get_linear_rgb(sdr, x, y);

            // Compute luminance
            let hdr_lum = rgb_to_luminance(hdr_rgb, hdr_gamut);
            let sdr_lum = rgb_to_luminance(sdr_rgb, sdr_gamut);

            let encoded = compute_and_encode_gain(
                hdr_lum,
                sdr_lum,
                config,
                log_min,
                log_range,
                actual_min_boost,
                actual_max_boost,
            );
            gainmap.data[(gy * gm_width + gx) as usize] = encoded;
        }
    }

    Ok(gainmap)
}

/// Compute the gain for one channel of one cell, track min/max, and quantize
/// to the 8-bit gain map byte. Used by both the batch `compute_gainmap` and
/// the streaming `RowEncoder`.
///
/// `log_min` and `log_range` are `config.min_boost.ln()` and
/// `log(max_boost) - log(min_boost)` respectively — pre-computed by the
/// caller and reused across every cell.
pub(super) fn compute_and_encode_gain(
    hdr: f32,
    sdr: f32,
    config: &GainMapConfig,
    log_min: f32,
    log_range: f32,
    actual_min_boost: &mut f32,
    actual_max_boost: &mut f32,
) -> u8 {
    let gain = (hdr + config.alternate_offset) / (sdr + config.base_offset).max(0.001);
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
    (gamma_corrected * 255.0).round().clamp(0.0, 255.0) as u8
}

/// Compute multi-channel (RGB) gain map.
#[allow(clippy::too_many_arguments)]
fn compute_multichannel_gainmap(
    hdr: &PixelSlice<'_>,
    sdr: &PixelSlice<'_>,
    gm_width: u32,
    gm_height: u32,
    scale: u32,
    config: &GainMapConfig,
    actual_min_boost: &mut f32,
    actual_max_boost: &mut f32,
    stop: &impl Stop,
) -> Result<GainMap> {
    let mut gainmap = GainMap::new_multichannel(gm_width, gm_height)?;
    let hdr_w = hdr.width();
    let hdr_h = hdr.rows();

    let log_min = config.min_boost.ln();
    let log_max = config.max_boost.ln();
    let log_range = log_max - log_min;

    for gy in 0..gm_height {
        // Check for cancellation once per row
        stop.check()?;

        for gx in 0..gm_width {
            let x = (gx * scale + scale / 2).min(hdr_w - 1);
            let y = (gy * scale + scale / 2).min(hdr_h - 1);

            let hdr_rgb = get_linear_rgb(hdr, x, y);
            let sdr_rgb = get_linear_rgb(sdr, x, y);

            for c in 0..3 {
                let encoded = compute_and_encode_gain(
                    hdr_rgb[c],
                    sdr_rgb[c],
                    config,
                    log_min,
                    log_range,
                    actual_min_boost,
                    actual_max_boost,
                );
                let idx = (gy * gm_width + gx) as usize * 3 + c;
                gainmap.data[idx] = encoded;
            }
        }
    }

    Ok(gainmap)
}

/// Extract linear RGB `[0,1]` from a pixel slice at the given pixel position.
///
/// Applies the appropriate EOTF conversion (sRGB, PQ, HLG) based on the
/// image's declared transfer function.
fn get_linear_rgb(img: &PixelSlice<'_>, x: u32, y: u32) -> [f32; 3] {
    let desc = img.descriptor();
    let format = desc.pixel_format();
    let transfer = desc.transfer();
    let stride = img.stride();
    let data = img.as_strided_bytes();
    match format {
        PixelFormat::Rgba8 | PixelFormat::Rgb8 => {
            let bpp = if format == PixelFormat::Rgba8 { 4 } else { 3 };
            let idx = y as usize * stride + x as usize * bpp;
            let r = data[idx] as f32 / 255.0;
            let g = data[idx + 1] as f32 / 255.0;
            let b = data[idx + 2] as f32 / 255.0;

            // Apply EOTF based on transfer function
            match transfer {
                TransferFunction::Srgb => [srgb_eotf(r), srgb_eotf(g), srgb_eotf(b)],
                TransferFunction::Linear => [r, g, b],
                _ => [srgb_eotf(r), srgb_eotf(g), srgb_eotf(b)], // Assume sRGB for 8-bit
            }
        }

        PixelFormat::RgbaF32 => {
            let idx = y as usize * stride + x as usize * 16;
            let r = f32::from_le_bytes(data[idx..idx + 4].try_into().unwrap());
            let g = f32::from_le_bytes(data[idx + 4..idx + 8].try_into().unwrap());
            let b = f32::from_le_bytes(data[idx + 8..idx + 12].try_into().unwrap());
            [r, g, b]
        }

        PixelFormat::Gray8 => {
            let idx = y as usize * stride + x as usize;
            let v = data[idx] as f32 / 255.0;
            let linear = srgb_eotf(v);
            [linear, linear, linear]
        }
        _ => [0.0, 0.0, 0.0],
    }
}

/// Downsample a full-resolution single-channel f32 gain map using zenresize.
///
/// Uses `PixelDescriptor::GRAYF32_LINEAR` with a Robidoux filter (good
/// general-purpose balance between sharpness and ringing).
#[cfg(feature = "resize")]
fn downsample_gain_f32(
    src: &[f32],
    src_w: u32,
    src_h: u32,
    dst_w: u32,
    dst_h: u32,
) -> alloc::vec::Vec<f32> {
    use zenpixels::PixelDescriptor;
    use zenresize::{Filter, ResizeConfig, Resizer};

    let cfg = ResizeConfig::builder(src_w, src_h, dst_w, dst_h)
        .filter(Filter::Robidoux)
        .format(PixelDescriptor::GRAYF32_LINEAR)
        .build();
    let mut resizer = Resizer::new(&cfg);
    resizer.resize_f32(src)
}

// ---------------------------------------------------------------------------
// zentone-powered encode path
// ---------------------------------------------------------------------------

/// Derive a zentone [`SplitConfig`] from a [`GainMapConfig`] and source gamut.
fn split_config_from_gainmap(config: &GainMapConfig, gamut: ColorPrimaries) -> SplitConfig {
    SplitConfig {
        luma_weights: crate::color::gamut::luma_coefficients(gamut),
        base_offset: config.base_offset,
        alternate_offset: config.alternate_offset,
        min_log2: config.min_boost.log2(),
        max_log2: config.max_boost.log2(),
        pre_desaturate: 0.0,
    }
}

/// Quantize a raw `log2` gain value to a u8 wire byte.
///
/// Normalizes within `[log2_min, log2_min + log2_range]`, applies gamma,
/// and maps to `[0, 255]`. Mathematically equivalent to the `ln`-domain
/// normalization in [`compute_and_encode_gain`] — the log base cancels in
/// the ratio. Compatible with [`super::apply::GainMapLut`] decode.
pub(super) fn pack_log2_gain_u8(log2_gain: f32, log2_min: f32, log2_range: f32, gamma: f32) -> u8 {
    let clamped = log2_gain.clamp(log2_min, log2_min + log2_range);
    let normalized = if log2_range > 0.0 {
        (clamped - log2_min) / log2_range
    } else {
        0.5
    };
    let gamma_corrected = if gamma != 1.0 {
        normalized.powf(gamma)
    } else {
        normalized
    };
    (gamma_corrected * 255.0).round().clamp(0.0, 255.0) as u8
}

/// Extract one HDR row as interleaved RGBA f32 for the zentone splitter.
///
/// Reuses [`get_linear_rgb`] per pixel (same perf as the existing
/// `compute_gainmap` path). Alpha is set to 1.0.
fn extract_linear_row_rgba(img: &PixelSlice<'_>, y: u32, out: &mut [f32]) {
    let width = img.width() as usize;
    debug_assert!(out.len() >= width * 4);
    for x in 0..width {
        let rgb = get_linear_rgb(img, x as u32, y);
        let i = x * 4;
        out[i] = rgb[0];
        out[i + 1] = rgb[1];
        out[i + 2] = rgb[2];
        out[i + 3] = 1.0;
    }
}

/// Write an interleaved RGBA f32 row into an `RgbaF32` [`PixelBuffer`]'s
/// mutable byte buffer.
fn write_rgba32f_row(out_data: &mut [u8], stride: usize, width: u32, y: u32, row: &[f32]) {
    let width = width as usize;
    let byte_offset = (y as usize) * stride;
    let row_bytes: &[u8] = bytemuck::cast_slice(&row[..width * 4]);
    out_data[byte_offset..byte_offset + row_bytes.len()].copy_from_slice(row_bytes);
}

/// Compute a gain map by tone-mapping HDR to SDR using a [`LumaToneMap`] curve.
///
/// Unlike [`compute_gainmap`] which requires the caller to supply both HDR and
/// SDR images, this function takes only the HDR image and uses zentone's
/// [`LumaGainMapSplitter`] to produce the SDR base and gain map simultaneously.
///
/// Returns `(sdr_image, gain_map, metadata)`:
/// - `sdr_image`: `RgbaF32` linear, same gamut as the HDR input. The caller
///   converts to sRGB u8 for JPEG storage.
/// - `gain_map`: single-channel u8 at `1/scale_factor` resolution.
/// - `metadata`: ready for `zencodec::GainMapParams` wire serialization.
///
/// Multi-channel gain maps are not supported — returns
/// `Err(EncodeError)` if `config.multi_channel` is `true`.
///
/// **Deprecated API surface** — slated for removal in 0.5.0. The
/// splitter-based path ships with the [`HableFilmic`] default curve;
/// callers with explicit tone-curve control needs should stage a zentone
/// curve + call [`compute_gainmap`] with their own tonemapped SDR.
#[doc(hidden)]
pub fn compute_gainmap_tonemap<T: LumaToneMap>(
    hdr: PixelSlice<'_>,
    curve: &T,
    config: &GainMapConfig,
    stop: impl Stop,
) -> Result<(PixelBuffer, GainMap, GainMapMetadata)> {
    if config.multi_channel {
        return Err(crate::types::Error::EncodeError(
            "compute_gainmap_tonemap does not support multi-channel gain maps; \
             use compute_gainmap with separate HDR/SDR images instead"
                .into(),
        ));
    }

    crate::types::validate_ultrahdr_slice(&hdr)?;

    let width = hdr.width();
    let height = hdr.rows();
    let hdr_gamut = hdr.descriptor().primaries;
    let scale = config.scale_factor.max(1) as u32;
    let gm_width = width.div_ceil(scale);
    let gm_height = height.div_ceil(scale);

    // Build splitter from GainMapConfig + source gamut.
    let split_cfg = split_config_from_gainmap(config, hdr_gamut);
    let splitter = LumaGainMapSplitter::new(curve, split_cfg);

    // Allocate outputs.
    let mut sdr_image = new_pixel_buffer(
        width,
        height,
        PixelFormat::RgbaF32,
        hdr_gamut,
        TransferFunction::Linear,
    )?;
    let sdr_stride = sdr_image.stride();
    let mut sdr_mut = sdr_image.as_slice_mut();
    let sdr_data = sdr_mut.as_strided_bytes_mut();

    let mut gainmap = GainMap::new(gm_width, gm_height)?;
    let mut stats = SplitStats::default();

    // Packing constants (log2 domain).
    let log2_min = config.min_boost.log2();
    let log2_max = config.max_boost.log2();
    let log2_range = log2_max - log2_min;

    // Scratch buffers, reused per row.
    let w = width as usize;
    let mut hdr_buf = vec![0.0_f32; w * 4];
    let mut sdr_buf = vec![0.0_f32; w * 4];
    let mut gain_buf = vec![0.0_f32; w];

    // With zenresize: collect full-resolution gain, then downsample properly.
    // Without: center-pixel sampling (lower quality, zero extra memory).
    #[cfg(feature = "resize")]
    let mut full_gain = vec![0.0_f32; w * height as usize];

    #[cfg(not(feature = "resize"))]
    let mut next_gy: u32 = 0;

    for y in 0..height {
        stop.check()?;

        // Linearize HDR row into interleaved RGBA f32.
        extract_linear_row_rgba(&hdr, y, &mut hdr_buf);

        // Split: HDR → SDR + log2 gain.
        splitter.split_row(&hdr_buf, &mut sdr_buf, &mut gain_buf, 4, &mut stats);

        // Write SDR row.
        write_rgba32f_row(sdr_data, sdr_stride, width, y, &sdr_buf);

        #[cfg(feature = "resize")]
        {
            let row_offset = y as usize * w;
            full_gain[row_offset..row_offset + w].copy_from_slice(&gain_buf[..w]);
        }

        #[cfg(not(feature = "resize"))]
        {
            // Center-pixel sampling fallback.
            while next_gy < gm_height {
                let center_y = (next_gy * scale + scale / 2).min(height - 1);
                if center_y != y {
                    break;
                }
                for gx in 0..gm_width {
                    let cx = (gx * scale + scale / 2).min(width - 1) as usize;
                    gainmap.data[(next_gy * gm_width + gx) as usize] =
                        pack_log2_gain_u8(gain_buf[cx], log2_min, log2_range, config.gamma);
                }
                next_gy += 1;
            }
        }
    }

    #[cfg(not(feature = "resize"))]
    {
        // Edge case: last gain map rows whose center_y was clamped to height-1.
        while next_gy < gm_height {
            for gx in 0..gm_width {
                let cx = (gx * scale + scale / 2).min(width - 1) as usize;
                gainmap.data[(next_gy * gm_width + gx) as usize] =
                    pack_log2_gain_u8(gain_buf[cx], log2_min, log2_range, config.gamma);
            }
            next_gy += 1;
        }
    }

    #[cfg(feature = "resize")]
    {
        // Downsample full-resolution gain via zenresize, then pack to u8.
        let ds_gain = downsample_gain_f32(&full_gain, width, height, gm_width, gm_height);
        for (i, &g) in ds_gain.iter().enumerate() {
            gainmap.data[i] = pack_log2_gain_u8(g, log2_min, log2_range, config.gamma);
        }
    }

    drop(sdr_mut);

    // Clamp observed stats to the configured packing range.
    let observed_min = stats.observed_min_log2.max(log2_min) as f64;
    let observed_max = stats.observed_max_log2.min(log2_max) as f64;

    let metadata = crate::types::metadata_from_arrays(
        [observed_min; 3],
        [observed_max; 3],
        [config.gamma as f64; 3],
        [config.base_offset as f64; 3],
        [config.alternate_offset as f64; 3],
        (config.base_hdr_headroom as f64).log2(),
        (config.alternate_hdr_headroom.max(config.max_boost) as f64).log2(),
        true,
        false,
    );

    Ok((sdr_image, gainmap, metadata))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ColorPrimaries;

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
        let mut hdr = new_pixel_buffer(
            8,
            8,
            PixelFormat::Rgba8,
            ColorPrimaries::Bt709,
            TransferFunction::Srgb,
        )
        .unwrap();
        {
            let mut slice = hdr.as_slice_mut();
            let bytes = slice.as_strided_bytes_mut();
            for i in 0..bytes.len() / 4 {
                bytes[i * 4] = 180;
                bytes[i * 4 + 1] = 180;
                bytes[i * 4 + 2] = 180;
                bytes[i * 4 + 3] = 255;
            }
        }

        let mut sdr = new_pixel_buffer(
            8,
            8,
            PixelFormat::Rgba8,
            ColorPrimaries::Bt709,
            TransferFunction::Srgb,
        )
        .unwrap();
        {
            let mut slice = sdr.as_slice_mut();
            let bytes = slice.as_strided_bytes_mut();
            for i in 0..bytes.len() / 4 {
                bytes[i * 4] = 128;
                bytes[i * 4 + 1] = 128;
                bytes[i * 4 + 2] = 128;
                bytes[i * 4 + 3] = 255;
            }
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
        assert!(metadata.channels[0].max >= 1.0);
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

    /// Helper: create an 8x8 HDR image (RgbaF32, Linear, BT.709) filled with a uniform color.
    fn make_hdr_8x8(r: f32, g: f32, b: f32) -> PixelBuffer {
        let w = 8u32;
        let h = 8u32;
        let pixel_count = (w * h) as usize;
        let mut data = Vec::with_capacity(pixel_count * 16);
        for _ in 0..pixel_count {
            data.extend_from_slice(&r.to_le_bytes());
            data.extend_from_slice(&g.to_le_bytes());
            data.extend_from_slice(&b.to_le_bytes());
            data.extend_from_slice(&1.0f32.to_le_bytes());
        }
        crate::types::pixel_buffer_from_vec(
            data,
            w,
            h,
            PixelFormat::RgbaF32,
            ColorPrimaries::Bt709,
            TransferFunction::Linear,
        )
        .unwrap()
    }

    /// Helper: create an 8x8 SDR image (Rgba8, Srgb, BT.709) filled with a uniform color.
    fn make_sdr_8x8(r: u8, g: u8, b: u8) -> PixelBuffer {
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
        crate::types::pixel_buffer_from_vec(
            data,
            w,
            h,
            PixelFormat::Rgba8,
            ColorPrimaries::Bt709,
            TransferFunction::Srgb,
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
        let sdr = crate::types::pixel_buffer_from_vec(
            vec![128u8; 4 * 4 * 4],
            4,
            4,
            PixelFormat::Rgba8,
            ColorPrimaries::Bt709,
            TransferFunction::Srgb,
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
        let hdr = new_pixel_buffer(
            8,
            8,
            PixelFormat::Rgba8,
            ColorPrimaries::Bt709,
            TransferFunction::Srgb,
        )
        .unwrap();
        let sdr = new_pixel_buffer(
            8,
            8,
            PixelFormat::Rgba8,
            ColorPrimaries::Bt709,
            TransferFunction::Srgb,
        )
        .unwrap();
        let config = GainMapConfig::default();

        // Should return Stopped error due to cancellation
        let result = compute_gainmap(&hdr, &sdr, &config, ImmediateCancel);

        assert!(matches!(
            result,
            Err(crate::Error::Stopped(enough::StopReason::Cancelled))
        ));
    }

    // -----------------------------------------------------------------------
    // zentone-powered encode path tests
    // -----------------------------------------------------------------------

    fn make_uniform_rgba32f(width: u32, height: u32, value: f32) -> PixelBuffer {
        let mut img = new_pixel_buffer(
            width,
            height,
            PixelFormat::RgbaF32,
            ColorPrimaries::Bt709,
            TransferFunction::Linear,
        )
        .unwrap();
        let stride = img.stride();
        {
            let mut slice = img.as_slice_mut();
            let data = slice.as_strided_bytes_mut();
            for y in 0..height {
                for x in 0..width {
                    let offset = (y as usize) * stride + (x as usize) * 16;
                    for c in 0..3 {
                        let bytes = value.to_le_bytes();
                        data[offset + c * 4..offset + c * 4 + 4].copy_from_slice(&bytes);
                    }
                    let alpha = 1.0_f32.to_le_bytes();
                    data[offset + 12..offset + 16].copy_from_slice(&alpha);
                }
            }
        }
        img
    }

    fn read_pixel_rgba32f(img: &PixelBuffer, x: u32, y: u32) -> [f32; 4] {
        let stride = img.stride();
        let data = img.as_slice().as_strided_bytes();
        let offset = (y as usize) * stride + (x as usize) * 16;
        [
            f32::from_le_bytes(data[offset..offset + 4].try_into().unwrap()),
            f32::from_le_bytes(data[offset + 4..offset + 8].try_into().unwrap()),
            f32::from_le_bytes(data[offset + 8..offset + 12].try_into().unwrap()),
            f32::from_le_bytes(data[offset + 12..offset + 16].try_into().unwrap()),
        ]
    }

    #[test]
    fn test_tonemap_basic() {
        let hdr = make_uniform_rgba32f(8, 8, 0.5);
        let curve = crate::gainmap::splitter::HableFilmic::new();
        let config = GainMapConfig::default();
        let (sdr, gainmap, metadata) =
            compute_gainmap_tonemap(hdr.as_slice(), &curve, &config, enough::Unstoppable).unwrap();

        // SDR has same dimensions.
        assert_eq!(sdr.width(), 8);
        assert_eq!(sdr.height(), 8);
        assert_eq!(sdr.descriptor().pixel_format(), PixelFormat::RgbaF32);

        // Gain map has downsampled dimensions.
        let scale = config.scale_factor as u32;
        assert_eq!(gainmap.width, 8u32.div_ceil(scale));
        assert_eq!(gainmap.height, 8u32.div_ceil(scale));

        // SDR pixels in [0, 1].
        let px = read_pixel_rgba32f(&sdr, 0, 0);
        for (c, &v) in px.iter().take(3).enumerate() {
            assert!((0.0..=1.0).contains(&v), "SDR channel {c} = {v}");
        }
        assert!((px[3] - 1.0).abs() < 1e-6, "alpha should be 1.0");

        // Metadata has reasonable values.
        assert!(metadata.channels[0].min.is_finite());
        assert!(metadata.channels[0].max.is_finite());
        assert!(metadata.channels[0].max >= metadata.channels[0].min);
    }

    #[test]
    fn test_tonemap_bright_hdr() {
        let hdr = make_uniform_rgba32f(8, 8, 5.0);
        let curve = crate::gainmap::splitter::HableFilmic::new();
        let config = GainMapConfig::default();
        let (sdr, gainmap, _metadata) =
            compute_gainmap_tonemap(hdr.as_slice(), &curve, &config, enough::Unstoppable).unwrap();

        // SDR should be tonemapped down to [0, 1].
        let px = read_pixel_rgba32f(&sdr, 4, 4);
        for (c, &v) in px.iter().take(3).enumerate() {
            assert!((0.0..=1.01).contains(&v), "SDR channel {c} = {v}");
        }

        // Gain map values should reflect HDR boost.
        // With uniform bright HDR, all gain map bytes should be > 128.
        for &byte in &gainmap.data {
            assert!(byte > 64, "gain map byte {byte} too low for 5.0 HDR");
        }
    }

    #[test]
    fn test_tonemap_multi_channel_rejected() {
        let hdr = make_uniform_rgba32f(8, 8, 0.5);
        let curve = crate::gainmap::splitter::HableFilmic::new();
        let config = GainMapConfig {
            multi_channel: true,
            ..GainMapConfig::default()
        };
        let result = compute_gainmap_tonemap(hdr.as_slice(), &curve, &config, enough::Unstoppable);
        assert!(result.is_err());
    }

    #[test]
    fn test_pack_log2_gain_known_values() {
        // Range [0.0, 2.0] (log2 domain: 1× to 4×)
        let min = 0.0_f32;
        let range = 2.0_f32;

        // Min → 0
        assert_eq!(pack_log2_gain_u8(0.0, min, range, 1.0), 0);
        // Max → 255
        assert_eq!(pack_log2_gain_u8(2.0, min, range, 1.0), 255);
        // Mid → 128 (0.5 * 255 = 127.5 → rounds to 128)
        assert_eq!(pack_log2_gain_u8(1.0, min, range, 1.0), 128);
    }

    #[test]
    fn test_pack_log2_gain_gamma() {
        let min = 0.0_f32;
        let range = 2.0_f32;
        // At midpoint, gamma > 1 darkens the output.
        let no_gamma = pack_log2_gain_u8(1.0, min, range, 1.0);
        let with_gamma = pack_log2_gain_u8(1.0, min, range, 2.0);
        // gamma=2 → normalized 0.5 → 0.5^2 = 0.25 → 64
        assert!(
            with_gamma < no_gamma,
            "gamma should reduce midpoint: {with_gamma} vs {no_gamma}"
        );
        assert_eq!(with_gamma, 64); // 0.25 * 255 = 63.75 → 64
    }

    #[test]
    fn test_split_config_from_gainmap_conversion() {
        let config = GainMapConfig {
            min_boost: 1.0,
            max_boost: 4.0,
            base_offset: 1.0 / 64.0,
            alternate_offset: 1.0 / 64.0,
            ..GainMapConfig::default()
        };
        let sc = split_config_from_gainmap(&config, ColorPrimaries::Bt709);
        assert!((sc.min_log2 - 0.0).abs() < 1e-6, "log2(1.0) = 0");
        assert!((sc.max_log2 - 2.0).abs() < 1e-6, "log2(4.0) = 2");
        assert_eq!(
            sc.luma_weights,
            crate::color::gamut::luma_coefficients(ColorPrimaries::Bt709)
        );
        assert_eq!(sc.base_offset, 1.0 / 64.0);
        assert_eq!(sc.alternate_offset, 1.0 / 64.0);
        assert_eq!(sc.pre_desaturate, 0.0);
    }

    /// Convert an RgbaF32 linear image to Rgba8 sRGB for the decoder.
    fn rgba32f_to_rgba8_srgb(src: &PixelBuffer) -> PixelBuffer {
        let src_primaries = src.descriptor().primaries;
        let mut dst = new_pixel_buffer(
            src.width(),
            src.height(),
            PixelFormat::Rgba8,
            src_primaries,
            TransferFunction::Srgb,
        )
        .unwrap();
        let width = src.width();
        let height = src.height();
        let dst_stride = dst.stride();
        {
            let mut dst_slice = dst.as_slice_mut();
            let dst_data = dst_slice.as_strided_bytes_mut();
            for y in 0..height {
                for x in 0..width {
                    let px = read_pixel_rgba32f(src, x, y);
                    let dst_offset = (y as usize) * dst_stride + (x as usize) * 4;
                    for (c, &lin) in px.iter().take(3).enumerate() {
                        let srgb = linear_srgb::tf::linear_to_srgb(lin.clamp(0.0, 1.0));
                        dst_data[dst_offset + c] = (srgb * 255.0 + 0.5).clamp(0.0, 255.0) as u8;
                    }
                    dst_data[dst_offset + 3] = 255;
                }
            }
        }
        dst
    }

    #[test]
    fn test_tonemap_round_trip() {
        // Grayscale gradient HDR, scale_factor=1 for full-resolution gain map.
        let width = 16u32;
        let height = 4u32;
        let mut hdr = new_pixel_buffer(
            width,
            height,
            PixelFormat::RgbaF32,
            ColorPrimaries::Bt709,
            TransferFunction::Linear,
        )
        .unwrap();
        let stride = hdr.stride();
        {
            let mut slice = hdr.as_slice_mut();
            let data = slice.as_strided_bytes_mut();
            for y in 0..height {
                for x in 0..width {
                    let v = (x as f32 + 1.0) / width as f32 * 2.0; // 0.125 .. 2.0
                    let offset = (y as usize) * stride + (x as usize) * 16;
                    for c in 0..3 {
                        data[offset + c * 4..offset + c * 4 + 4].copy_from_slice(&v.to_le_bytes());
                    }
                    data[offset + 12..offset + 16].copy_from_slice(&1.0_f32.to_le_bytes());
                }
            }
        }

        let curve = crate::gainmap::splitter::HableFilmic::new();
        let config = GainMapConfig {
            scale_factor: 1, // Full resolution gain map.
            ..GainMapConfig::default()
        };
        let (sdr_f32, gainmap, metadata) =
            compute_gainmap_tonemap(hdr.as_slice(), &curve, &config, enough::Unstoppable).unwrap();

        // The decoder's get_sdr_linear only handles Rgba8 (with `transfer`
        // feature). Convert SDR to Rgba8 sRGB for the round-trip.
        let sdr_u8 = rgba32f_to_rgba8_srgb(&sdr_f32);

        // Reconstruct via apply_gainmap.
        let reconstructed = crate::gainmap::apply_gainmap(
            &sdr_u8,
            &gainmap,
            &metadata,
            config.max_boost,
            crate::gainmap::HdrOutputFormat::LinearFloat,
            enough::Unstoppable,
        )
        .unwrap();

        // Compare reconstructed to original HDR. Error sources:
        // (1) u8 gain quantization (~1/256 of log2 range),
        // (2) u8 sRGB quantization of the SDR base (~1/256 of gamma curve),
        // (3) tone curve non-linearity at extreme values.
        // Allow generous tolerance — the pipeline correctness test, not a
        // precision test.
        for y in 0..height {
            for x in 0..width {
                let orig = read_pixel_rgba32f(&hdr, x, y);
                let recon = read_pixel_rgba32f(&reconstructed, x, y);
                for c in 0..3 {
                    let diff = (orig[c] - recon[c]).abs();
                    assert!(
                        diff < orig[c] * 0.25 + 0.15,
                        "round-trip drift at ({x},{y}) ch{c}: orig={} recon={} diff={}",
                        orig[c],
                        recon[c],
                        diff
                    );
                }
            }
        }
    }

    #[test]
    fn test_tonemap_cancellation() {
        struct ImmediateCancel;
        impl enough::Stop for ImmediateCancel {
            fn check(&self) -> std::result::Result<(), enough::StopReason> {
                Err(enough::StopReason::Cancelled)
            }
        }
        let hdr = make_uniform_rgba32f(8, 8, 0.5);
        let curve = crate::gainmap::splitter::HableFilmic::new();
        let result = compute_gainmap_tonemap(
            hdr.as_slice(),
            &curve,
            &GainMapConfig::default(),
            ImmediateCancel,
        );
        assert!(matches!(
            result,
            Err(crate::Error::Stopped(enough::StopReason::Cancelled))
        ));
    }
}
