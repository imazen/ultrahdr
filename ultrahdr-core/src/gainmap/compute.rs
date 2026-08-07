//! Gain map computation from HDR and SDR images.

use alloc::vec;

use crate::color::gamut::rgb_to_luminance;

use crate::color::transfer::{hlg_eotf, pq_eotf, srgb_eotf};
use crate::types::TransferFunction;
use crate::types::{
    ColorPrimaries, Error, GainMap, GainMapMetadata, PixelBuffer, PixelFormat, PixelSlice, Result,
};
use enough::Stop;
use whereat::at;

/// Configuration for gain map computation.
///
/// **Every field here is in LINEAR domain** for ergonomics — humans think
/// "4× brighter" not "log2(4)=2". The log2 conversion happens exactly once,
/// when producing [`GainMapMetadata`] (whose `min`/`max`/headroom fields are
/// log2). Feeding a log2 value into any of these fields is a bug.
///
/// `min_boost ..= max_boost` is the **quantization grid**: gain-map bytes
/// are normalized over `[ln(min_boost), ln(max_boost)]`, and the produced
/// metadata declares exactly this range so readers dequantize on the grid
/// the bytes were written on (#33). Content gains outside the grid are
/// clamped. A narrower grid (e.g. `max_boost` = the content's true peak
/// ratio) spends the 8-bit code space more precisely; the default
/// HDR-encode grid of `[1, target_display_peak/203]` is 0.022 stops/step
/// at a 10,000-nit peak.
#[derive(Debug, Clone)]
pub struct GainMapConfig {
    /// Scale factor for gain map (1 = same size as image, 4 = 1/4 size).
    pub scale_factor: u8,
    /// Gamma to apply to the gain map encoding.
    pub gamma: f32,
    /// Use multi-channel (RGB) gain map instead of single-channel luminance.
    pub multi_channel: bool,
    /// Minimum gain (linear ratio). 1.0 = no darkening, 0.5 = allow 2×
    /// darker. Bottom of the quantization grid (byte 0).
    pub min_boost: f32,
    /// Maximum gain (linear ratio). HDR peak / SDR peak, e.g. 6.0 = ~2.5
    /// stops. Top of the quantization grid (byte 255).
    pub max_boost: f32,
    /// Offset for base (SDR) values to avoid division by zero. Linear domain.
    pub base_offset: f32,
    /// Offset for alternate (HDR) values. Linear domain.
    pub alternate_offset: f32,
    /// Display boost (linear ratio, NOT log2) below which the gain map is
    /// not applied at all. Stored in metadata as
    /// `base_hdr_headroom = log2(this)`; 1.0 ⇒ 0.0 in metadata.
    pub base_hdr_headroom: f32,
    /// Display boost (linear ratio, NOT log2) at which the gain map is
    /// applied at full weight. Stored in metadata as
    /// `alternate_hdr_headroom = log2(max(this, observed content max))`.
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
    compute_gainmap_slice_observed(hdr, sdr, config, &stop).map(|(gm, meta, _, _)| (gm, meta))
}

/// Content-fit [`compute_gainmap_slice`]: the quantization grid is SELECTED
/// FROM THE MEASURED CONTENT instead of taken verbatim from the config.
///
/// The configured `min_boost ..= max_boost` acts as the OUTER BOUND (the
/// caller's policy — e.g. "never encode boosts my target display can't
/// show"); within it the grid is narrowed to the content's actual observed
/// gain range, so the 8-bit code space is spent on gains that exist instead
/// of a configured range that is usually wrong about the content (a
/// 10,000-nit `target_display_peak` grid spends ~5.6 stops of code space on
/// content that may span 2). Narrowing is interop-safe by construction: the
/// produced metadata declares exactly the narrowed grid the bytes were
/// quantized on (#33 — declared grid == quantization grid, unchanged).
///
/// Mechanics: one measurement pass (the same subsampled gain scan the encode
/// uses — bit-identical math, shared code) observes the content gain range;
/// the grid is `[clamp(observed_min), clamp(observed_max)]` within the
/// config bounds, widened to at least [`CONTENT_FIT_MIN_SPAN_STOPS`] so uniform
/// content never degenerates to a zero-width grid; then the ordinary encode
/// runs on the narrowed grid. Cost: the gain scan runs twice (it is
/// subsampled by `scale_factor²` — small next to the JPEG encodes).
///
/// This is the campaign appendix-AA "measure, don't configure" rule applied
/// to the one encoder-side site where a config range was load-bearing: where
/// an encoder SELECTS its grid, it selects from measured content. Callers
/// that need the exact configured grid (byte-stable corpora, external grid
/// contracts) keep calling [`compute_gainmap_slice`].
pub fn compute_gainmap_content_fit(
    hdr: PixelSlice<'_>,
    sdr: PixelSlice<'_>,
    config: &GainMapConfig,
    stop: impl Stop,
) -> Result<(GainMap, GainMapMetadata)> {
    let (_, _, observed_min, observed_max) =
        compute_gainmap_slice_observed(hdr.clone(), sdr.clone(), config, &stop)?;
    let narrowed = content_fit_config(config, observed_min, observed_max);
    compute_gainmap_slice_observed(hdr, sdr, &narrowed, &stop).map(|(gm, meta, _, _)| (gm, meta))
}

/// Minimum log2 span of a content-fitted grid (1/16 stop). Uniform content
/// (every gain identical) would otherwise produce a zero-width grid whose
/// log-range normalization degenerates; 1/16 stop keeps the grid valid while
/// staying far finer than any visible step.
pub const CONTENT_FIT_MIN_SPAN_STOPS: f32 = 1.0 / 16.0;

/// `2^CONTENT_FIT_MIN_SPAN_STOPS` as a linear boost ratio (design-time
/// constant so the `no_std` build needs no runtime `exp2`; a test pins the
/// identity).
const CONTENT_FIT_MIN_SPAN_RATIO: f32 = 1.044_273_8;

/// Narrow `config`'s quantization grid to the observed content gain range
/// (see [`compute_gainmap_content_fit`]). Non-finite / empty observations
/// return the config unchanged.
fn content_fit_config(
    config: &GainMapConfig,
    observed_min: f32,
    observed_max: f32,
) -> GainMapConfig {
    let mut out = config.clone();
    if !(observed_min.is_finite() && observed_max.is_finite()) || observed_max <= 0.0 {
        return out;
    }
    // Top: the measured max gain, capped by the configured policy bound.
    // Bottom: the measured min gain, floored by the configured policy bound
    // (content darkening below `min_boost` stays clamped, as configured).
    let top = observed_max.clamp(config.min_boost, config.max_boost);
    let bottom = observed_min.clamp(config.min_boost, top);
    // Enforce the minimum span, preferring to widen upward (more headroom)
    // and falling back to lowering the bottom at the policy cap.
    // 2^(1/16) — the linear ratio of CONTENT_FIT_MIN_SPAN_STOPS (constant so
    // no libm/exp2 is needed in no_std; pinned by a test).
    let min_ratio = CONTENT_FIT_MIN_SPAN_RATIO;
    let (bottom, top) = if top < bottom * min_ratio {
        let widened_top = (bottom * min_ratio).min(config.max_boost);
        if widened_top >= bottom * min_ratio {
            (bottom, widened_top)
        } else {
            ((widened_top / min_ratio).max(config.min_boost), widened_top)
        }
    } else {
        (bottom, top)
    };
    out.min_boost = bottom;
    out.max_boost = top;
    out
}

/// Shared core of [`compute_gainmap_slice`] / [`compute_gainmap_content_fit`]:
/// the ordinary encode, additionally returning the observed
/// `(min_gain, max_gain)` accumulators.
fn compute_gainmap_slice_observed(
    hdr: PixelSlice<'_>,
    sdr: PixelSlice<'_>,
    config: &GainMapConfig,
    stop: &impl Stop,
) -> Result<(GainMap, GainMapMetadata, f32, f32)> {
    crate::types::validate_ultrahdr_slice(&hdr)?;
    crate::types::validate_ultrahdr_slice(&sdr)?;

    let hdr_w = hdr.width();
    let hdr_h = hdr.rows();
    let sdr_w = sdr.width();
    let sdr_h = sdr.rows();

    if hdr_w != sdr_w || hdr_h != sdr_h {
        return Err(at!(crate::types::Error::DimensionMismatch {
            hdr_w,
            hdr_h,
            sdr_w,
            sdr_h,
        }));
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
            stop,
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
            stop,
        )?
    };

    // The gain-map bytes were quantized on the CONFIG boost grid, so the
    // metadata must declare exactly that grid (#33). `actual_min_boost` is
    // tracked for the row-kernel accumulator contract but does not feed the
    // declared range; `actual_max_boost` only widens the alternate headroom.
    let metadata = metadata_for_config_grid(config, actual_max_boost);

    Ok((gainmap, metadata, actual_min_boost, actual_max_boost))
}

/// Build the [`GainMapMetadata`] that matches gain-map bytes quantized by
/// [`compute_and_encode_gain`] / [`compute_gain_row`] on `config`'s boost
/// grid.
///
/// **Contract (#33): the declared per-channel `min`/`max` ARE the
/// dequantization grid.** Readers reconstruct
/// `gain = 2^(min + byte/255 · (max − min))` (modulo gamma), so the metadata
/// must declare exactly the range the bytes were normalized over — the
/// config's `min_boost ..= max_boost`. Declaring anything else (e.g. the
/// observed content range) makes every conformant reader dequantize on the
/// wrong grid: a 2000-nit ramp encoded with the 10,000-nit default peak came
/// back at ~732 nits before this was pinned down.
///
/// `observed_max_boost` — the content's maximum gain as accumulated by the
/// encode kernel — does not affect the declared range; it only widens
/// `alternate_hdr_headroom` so that full gain application stays reachable
/// when the content exceeds the configured headroom. A non-finite or
/// non-positive value (no pixels observed) falls back to the grid top.
pub(crate) fn metadata_for_config_grid(
    config: &GainMapConfig,
    observed_max_boost: f32,
) -> GainMapMetadata {
    let observed = if observed_max_boost.is_finite() && observed_max_boost > 0.0 {
        observed_max_boost.clamp(config.min_boost, config.max_boost)
    } else {
        config.max_boost
    };
    crate::types::metadata_from_arrays(
        [(config.min_boost as f64).log2(); 3],
        [(config.max_boost as f64).log2(); 3],
        [config.gamma as f64; 3],
        [config.base_offset as f64; 3],
        [config.alternate_offset as f64; 3],
        (config.base_hdr_headroom as f64).log2(),
        (config.alternate_hdr_headroom.max(observed) as f64).log2(),
        true,
        false,
    )
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

    // Materialize one gain-map row's worth of subsampled HDR + SDR linear RGB,
    // then delegate quantization to `compute_gain_row`. Three channels —
    // luminance is gamut-derived from RGB inside the row kernel.
    let row_len = gm_width as usize * 3;
    let mut hdr_row_rgb = vec![0.0f32; row_len];
    let mut sdr_row_rgb = vec![0.0f32; row_len];
    let mut min_max = (*actual_min_boost, *actual_max_boost);

    for gy in 0..gm_height {
        // Check for cancellation once per row
        stop.check().map_err(|r| at!(Error::Stopped(r)))?;

        // Sample center pixel of each block on this row.
        let y = (gy * scale + scale / 2).min(hdr_h - 1);
        for gx in 0..gm_width {
            let x = (gx * scale + scale / 2).min(hdr_w - 1);
            let hdr_rgb = get_linear_rgb(hdr, x, y);
            let sdr_rgb = get_linear_rgb(sdr, x, y);
            let off = gx as usize * 3;
            hdr_row_rgb[off] = hdr_rgb[0];
            hdr_row_rgb[off + 1] = hdr_rgb[1];
            hdr_row_rgb[off + 2] = hdr_rgb[2];
            sdr_row_rgb[off] = sdr_rgb[0];
            sdr_row_rgb[off + 1] = sdr_rgb[1];
            sdr_row_rgb[off + 2] = sdr_rgb[2];
        }

        let row_start = (gy * gm_width) as usize;
        let row_end = row_start + gm_width as usize;
        compute_gain_row(
            &hdr_row_rgb,
            &sdr_row_rgb,
            3,
            hdr_gamut,
            sdr_gamut,
            &mut gainmap.data[row_start..row_end],
            config,
            &mut min_max,
        );
    }

    *actual_min_boost = min_max.0;
    *actual_max_boost = min_max.1;
    Ok(gainmap)
}

/// Compute and quantize gain-map bytes for one row, given paired HDR + SDR
/// linear-RGB rows.
///
/// Inputs:
/// - `hdr_row` and `sdr_row`: interleaved linear f32 RGB(A) of equal length.
///   `channels` is 3 or 4 (alpha is read but doesn't affect the gain
///   computation — only the first three channels are used).
/// - `hdr_primaries` / `sdr_primaries`: color primaries used to weight RGB
///   into luminance. Pass the same value for both when the inputs already
///   share a gamut.
/// - `gainmap_byte_out`: u8 output, one byte per pixel for single-channel
///   (luminance) gain maps; `len = hdr_row.len() / channels`.
/// - `config`: the gain-map configuration (offsets, min/max boost, gamma).
/// - `observed_min_max`: `(min, max)` f32 accumulator of the content's gain
///   range — updated in place across calls so callers can stitch row-level
///   invocations into a whole-image min/max.
///
/// **Metadata contract (#33):** the bytes this writes are quantized on the
/// CONFIG grid (`config.min_boost ..= config.max_boost`), so any
/// [`GainMapMetadata`] describing them must declare exactly that range —
/// build it from the config (see `metadata_for_config_grid`), never from the
/// `observed_min_max` accumulator. The accumulator is for headroom widening
/// and diagnostics only; declaring it as the range makes readers dequantize
/// on the wrong grid.
///
/// Used by `compute_gainmap` internally and by zenjpeg's encode flow to fuse
/// splitter + gain quantization in a single row pass. Bit-identical to the
/// per-cell math in `compute_and_encode_gain`.
#[allow(clippy::too_many_arguments)]
pub fn compute_gain_row(
    hdr_row: &[f32],
    sdr_row: &[f32],
    channels: u8,
    hdr_primaries: ColorPrimaries,
    sdr_primaries: ColorPrimaries,
    gainmap_byte_out: &mut [u8],
    config: &GainMapConfig,
    observed_min_max: &mut (f32, f32),
) {
    debug_assert!(channels == 3 || channels == 4);
    let chan = channels as usize;
    debug_assert_eq!(hdr_row.len(), sdr_row.len());
    debug_assert_eq!(gainmap_byte_out.len(), hdr_row.len() / chan);

    let log_min = config.min_boost.ln();
    let log_range = config.max_boost.ln() - log_min;

    for (i, byte_out) in gainmap_byte_out.iter_mut().enumerate() {
        let off = i * chan;
        let hdr_rgb = [hdr_row[off], hdr_row[off + 1], hdr_row[off + 2]];
        let sdr_rgb = [sdr_row[off], sdr_row[off + 1], sdr_row[off + 2]];
        let hdr_lum = rgb_to_luminance(hdr_rgb, hdr_primaries);
        let sdr_lum = rgb_to_luminance(sdr_rgb, sdr_primaries);
        *byte_out = compute_and_encode_gain(
            hdr_lum,
            sdr_lum,
            config,
            log_min,
            log_range,
            &mut observed_min_max.0,
            &mut observed_min_max.1,
        );
    }
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
        stop.check().map_err(|r| at!(Error::Stopped(r)))?;

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

/// Apply EOTF to a float-domain RGB triple based on the descriptor's transfer.
///
/// Used for f32 / f16 inputs where the pixel values aren't bytes — PQ and
/// HLG-encoded floats are common in HDR pipelines (Apple `kCGColorSpacePQ`,
/// EXR with non-linear transfer, GPU compositors). `Linear` passes through.
/// `Srgb` runs the EOTF directly on the float values.
#[inline]
fn apply_transfer_to_linear(rgb: [f32; 3], transfer: TransferFunction) -> [f32; 3] {
    match transfer {
        TransferFunction::Linear => rgb,
        TransferFunction::Srgb => [srgb_eotf(rgb[0]), srgb_eotf(rgb[1]), srgb_eotf(rgb[2])],
        TransferFunction::Pq => [pq_eotf(rgb[0]), pq_eotf(rgb[1]), pq_eotf(rgb[2])],
        TransferFunction::Hlg => [
            // hlg_eotf returns nits at 1000-nit peak; normalize to SDR-relative.
            hlg_eotf(rgb[0], 1000.0) / 1000.0,
            hlg_eotf(rgb[1], 1000.0) / 1000.0,
            hlg_eotf(rgb[2], 1000.0) / 1000.0,
        ],
        _ => rgb,
    }
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
            apply_transfer_to_linear([r, g, b], transfer)
        }

        #[cfg(feature = "f16")]
        PixelFormat::RgbaF16 | PixelFormat::RgbF16 => {
            let bpp = if format == PixelFormat::RgbaF16 { 8 } else { 6 };
            let idx = y as usize * stride + x as usize * bpp;
            let r = half::f16::from_le_bytes([data[idx], data[idx + 1]]).to_f32();
            let g = half::f16::from_le_bytes([data[idx + 2], data[idx + 3]]).to_f32();
            let b = half::f16::from_le_bytes([data[idx + 4], data[idx + 5]]).to_f32();
            apply_transfer_to_linear([r, g, b], transfer)
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ColorPrimaries;
    use crate::types::new_pixel_buffer;

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
            result.unwrap_err().error(),
            crate::types::Error::DimensionMismatch { .. }
        ));
    }

    #[test]
    fn compute_gain_row_matches_compute_gainmap() {
        // Regression gate: feeding `compute_gain_row` the same per-row inputs
        // that `compute_gainmap` would internally must produce the same bytes
        // and the same observed min/max. If this test diverges from
        // `compute_gainmap`'s output, the row kernel and the batch path are
        // out of sync — that's a contract break.
        let hdr = make_hdr_8x8(0.6, 0.4, 0.2);
        let sdr = make_sdr_8x8(160, 110, 70);
        let config = GainMapConfig {
            scale_factor: 1,
            ..Default::default()
        };
        let (gainmap_batch, _meta) =
            compute_gainmap(&hdr, &sdr, &config, enough::Unstoppable).unwrap();

        // Build the same linear RGB rows that compute_luminance_gainmap feeds
        // into compute_gain_row, and verify byte-for-byte parity.
        let hdr_slice = hdr.as_slice();
        let sdr_slice = sdr.as_slice();
        let w = hdr_slice.width() as usize;
        let h = hdr_slice.rows() as usize;
        let mut min_max = (f32::MAX, f32::MIN);
        let mut row_bytes = vec![0u8; w];
        let mut hdr_row_rgb = vec![0.0f32; w * 3];
        let mut sdr_row_rgb = vec![0.0f32; w * 3];
        for y in 0..h {
            for x in 0..w {
                let h_rgb = get_linear_rgb(&hdr_slice, x as u32, y as u32);
                let s_rgb = get_linear_rgb(&sdr_slice, x as u32, y as u32);
                hdr_row_rgb[x * 3..x * 3 + 3].copy_from_slice(&h_rgb);
                sdr_row_rgb[x * 3..x * 3 + 3].copy_from_slice(&s_rgb);
            }
            compute_gain_row(
                &hdr_row_rgb,
                &sdr_row_rgb,
                3,
                hdr_slice.descriptor().primaries,
                sdr_slice.descriptor().primaries,
                &mut row_bytes,
                &config,
                &mut min_max,
            );
            // The batch result is contiguous — compare row by row.
            let expected = &gainmap_batch.data[y * w..y * w + w];
            assert_eq!(row_bytes, expected, "row {y} bytes diverged");
        }
    }

    /// #33: the declared metadata range must be the range the bytes were
    /// quantized on — the CONFIG grid — even when the content's actual gain
    /// range is much narrower. Verified structurally (fields) and
    /// semantically (dequantizing a byte on the declared range at full
    /// weight recovers the true gain).
    #[test]
    fn metadata_declares_the_quantization_grid() {
        // Uniform pair with content boost ~2.23, well inside the default
        // [1.0, 6.0] grid.
        let hdr = make_hdr_8x8(0.5, 0.5, 0.5);
        let sdr = make_sdr_8x8(128, 128, 128);
        let config = GainMapConfig {
            scale_factor: 1,
            ..Default::default()
        };
        let (gainmap, metadata) =
            compute_gainmap(&hdr, &sdr, &config, enough::Unstoppable).unwrap();

        for ch in &metadata.channels {
            assert_eq!(
                ch.min,
                (config.min_boost as f64).log2(),
                "declared min must be the quantization grid bottom"
            );
            assert_eq!(
                ch.max,
                (config.max_boost as f64).log2(),
                "declared max must be the quantization grid top"
            );
        }
        assert_eq!(metadata.base_hdr_headroom, 0.0);
        assert_eq!(
            metadata.alternate_hdr_headroom,
            (config.alternate_hdr_headroom as f64).log2(),
            "content max (~2.23) below configured headroom must not shrink it"
        );

        // Semantic gate: byte -> declared-range LUT at weight 1.0 ≈ true gain.
        let sdr_lum = srgb_eotf(128.0 / 255.0);
        let true_gain = (0.5 + config.alternate_offset) / (sdr_lum + config.base_offset);
        let lut = crate::gainmap::apply::GainMapLut::new(&metadata, 1.0);
        let recovered = lut.lookup_luminance(gainmap.data[0])[0];
        let step = (config.max_boost.ln() - config.min_boost.ln()) / 255.0;
        assert!(
            (recovered.ln() - true_gain.ln()).abs() <= step * 0.75,
            "byte {} dequantized on the declared range gives {recovered:.4}, \
             true gain {true_gain:.4} (>{:.2}% off — declared range and \
             quantization basis disagree)",
            gainmap.data[0],
            step * 75.0
        );
    }

    #[test]
    fn content_fit_grid_is_measured_not_configured() {
        // Appendix AA: actual content boost (~4.06) is far below the
        // configured 10,000-nit-target grid top (10000/203 ≈ 49.26). The
        // content-fit grid must declare the MEASURED range, not the config.
        let hdr = make_hdr_8x8(2.0, 2.0, 2.0);
        let sdr = make_sdr_8x8(186, 186, 186); // ~0.5 linear
        let config = GainMapConfig {
            scale_factor: 1,
            max_boost: 10000.0 / 203.0,
            alternate_hdr_headroom: 10000.0 / 203.0,
            ..Default::default()
        };
        let (gainmap, metadata) = compute_gainmap_content_fit(
            hdr.as_slice(),
            sdr.as_slice(),
            &config,
            enough::Unstoppable,
        )
        .unwrap();

        let sdr_lum = srgb_eotf(186.0 / 255.0);
        let true_gain = ((2.0 + config.alternate_offset) / (sdr_lum + config.base_offset)) as f64;
        for ch in &metadata.channels {
            assert!(
                ch.max < (config.max_boost as f64).log2() - 1.0,
                "declared max {} must be the measured content range, not the \
                 configured {} (log2)",
                ch.max,
                (config.max_boost as f64).log2()
            );
            assert!(
                (ch.max - true_gain.log2()).abs() < 0.1,
                "declared max {} should sit at the measured max gain {}",
                ch.max,
                true_gain.log2()
            );
        }

        // #33 invariant: the declared (narrowed) grid IS the quantization
        // grid — dequantizing a byte on it recovers the true gain, at the
        // (much finer) narrowed step.
        let lut = crate::gainmap::apply::GainMapLut::new(&metadata, 1.0);
        let recovered = lut.lookup_luminance(gainmap.data[0])[0];
        let narrowed_step = ((metadata.channels[0].max - metadata.channels[0].min) / 255.0) as f32;
        assert!(
            (recovered.ln() - (true_gain as f32).ln()).abs()
                <= (narrowed_step * core::f32::consts::LN_2) * 0.75 + 1e-4,
            "byte {} on the content-fit grid gives {recovered:.4}, true gain {:.4}",
            gainmap.data[0],
            true_gain
        );
    }

    #[test]
    fn content_fit_beats_configured_grid_precision() {
        // Same content quantized both ways: the content-fit grid's
        // dequantization error must be strictly tighter than the configured
        // 49.26x grid's (that is the whole point of measuring).
        let hdr = make_hdr_8x8(1.7, 1.7, 1.7);
        let sdr = make_sdr_8x8(150, 150, 150);
        let config = GainMapConfig {
            scale_factor: 1,
            max_boost: 10000.0 / 203.0,
            alternate_hdr_headroom: 10000.0 / 203.0,
            ..Default::default()
        };
        let sdr_lum = srgb_eotf(150.0 / 255.0);
        let true_gain = (1.7 + config.alternate_offset) / (sdr_lum + config.base_offset);

        let (gm_cfg, meta_cfg) = compute_gainmap(&hdr, &sdr, &config, enough::Unstoppable).unwrap();
        let (gm_fit, meta_fit) = compute_gainmap_content_fit(
            hdr.as_slice(),
            sdr.as_slice(),
            &config,
            enough::Unstoppable,
        )
        .unwrap();
        let err = |gm: &GainMap, meta: &GainMapMetadata| -> f32 {
            let lut = crate::gainmap::apply::GainMapLut::new(meta, 1.0);
            (lut.lookup_luminance(gm.data[0])[0].ln() - true_gain.ln()).abs()
        };
        let (e_cfg, e_fit) = (err(&gm_cfg, &meta_cfg), err(&gm_fit, &meta_fit));
        assert!(
            e_fit <= e_cfg,
            "content-fit error {e_fit} must not exceed configured-grid error {e_cfg}"
        );
        // And the fit grid is dramatically finer: uniform content collapses
        // to the minimum span (1/16 stop over 255 codes) vs 5.62 stops.
        let span_fit = meta_fit.channels[0].max - meta_fit.channels[0].min;
        let span_cfg = meta_cfg.channels[0].max - meta_cfg.channels[0].min;
        assert!(
            span_fit < span_cfg / 8.0,
            "measured span {span_fit} should be far narrower than configured {span_cfg}"
        );
    }

    #[test]
    fn content_fit_uniform_content_never_degenerates() {
        // Identical HDR/SDR luminance -> every gain equal -> the min-span
        // guard must keep a valid (non-zero-width) grid and finite bytes.
        let hdr = make_hdr_8x8(0.5, 0.5, 0.5);
        let sdr = make_sdr_8x8(186, 186, 186); // ~0.5 linear -> gain ~1.0
        let config = GainMapConfig {
            scale_factor: 1,
            ..Default::default()
        };
        let (gainmap, metadata) = compute_gainmap_content_fit(
            hdr.as_slice(),
            sdr.as_slice(),
            &config,
            enough::Unstoppable,
        )
        .unwrap();
        for ch in &metadata.channels {
            assert!(
                (ch.max - ch.min) as f32 >= CONTENT_FIT_MIN_SPAN_STOPS * 0.99,
                "grid span {} collapsed below the minimum",
                ch.max - ch.min
            );
            assert!(ch.max.is_finite() && ch.min.is_finite());
        }
        assert!(gainmap.data.iter().all(|&b| b == gainmap.data[0]));
    }

    #[test]
    fn content_fit_min_span_ratio_matches_stops() {
        // Pin the design-time constant: 2^(1/16).
        assert!(
            ((CONTENT_FIT_MIN_SPAN_RATIO as f64).log2() - CONTENT_FIT_MIN_SPAN_STOPS as f64).abs()
                < 1e-6
        );
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
        let err = compute_gainmap(&hdr, &sdr, &config, ImmediateCancel).unwrap_err();

        assert!(matches!(
            err.error(),
            crate::Error::Stopped(enough::StopReason::Cancelled)
        ));
    }
}
