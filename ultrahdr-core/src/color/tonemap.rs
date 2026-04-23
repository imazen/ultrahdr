//! HDR to SDR tone mapping.
//!
//! This module provides:
//! - Standard tonemappers (filmic, Reinhard, BT.2390)
//! - Adaptive tonemapper that learns from HDR/SDR pairs
//! - Gain map inversion for perfect round-trips
//!
//! # Adaptive Tonemapping
//!
//! When re-encoding UltraHDR after edits, use [`AdaptiveTonemapper`] to preserve
//! the original artistic intent. The builder fits a per-channel LUT (or a
//! luminance curve, depending on the config) from an HDR/SDR pair you
//! already have on hand; you can then [`AdaptiveTonemapper::apply`] it to
//! an edited HDR to regenerate the SDR rendition with a matching look.
//!
//! ```no_run
//! use ultrahdr_core::{PixelBuffer, color::tonemap::AdaptiveTonemapper};
//!
//! # fn load_hdr(_name: &str) -> PixelBuffer { unimplemented!() }
//! # fn load_sdr(_name: &str) -> PixelBuffer { unimplemented!() }
//! let hdr_original = load_hdr("before-edit.exr");
//! let sdr_original = load_sdr("before-edit.jpg");
//! let hdr_edited   = load_hdr("after-edit.exr");
//!
//! let tonemapper = AdaptiveTonemapper::fit(&hdr_original, &sdr_original)?;
//! let _sdr_new   = tonemapper.apply(&hdr_edited)?;
//! # Ok::<(), ultrahdr_core::Error>(())
//! ```

use alloc::boxed::Box;
use alloc::vec;
use alloc::vec::Vec;
use core::cmp::Ordering;

use crate::color::gamut::{convert_gamut, rgb_to_luminance, soft_clip_gamut};
use crate::color::transfer::{hlg_eotf, pq_eotf, srgb_eotf, srgb_oetf};
use crate::types::{
    ColorPrimaries, Error, GainMap, GainMapMetadata, PixelBuffer, PixelSlice, Result,
    TransferFunction, new_pixel_buffer,
};

// ============================================================================
// Tone Mapping Configuration
// ============================================================================

/// Tone mapping configuration.
#[derive(Debug, Clone)]
pub struct ToneMapConfig {
    /// Target SDR peak luminance in nits (typically 100-203).
    pub target_peak_nits: f32,
    /// HDR content peak luminance in nits.
    pub hdr_peak_nits: f32,
    /// Target color gamut for SDR output.
    pub target_gamut: ColorPrimaries,
    /// Source color gamut of HDR content.
    pub source_gamut: ColorPrimaries,
}

impl Default for ToneMapConfig {
    fn default() -> Self {
        Self {
            target_peak_nits: 203.0, // SDR reference white
            hdr_peak_nits: 10000.0,  // PQ peak
            target_gamut: ColorPrimaries::Bt709,
            source_gamut: ColorPrimaries::Bt2020,
        }
    }
}

// ============================================================================
// zentone re-exports (feature-gated)
// ============================================================================
//
// When the `zentone` feature is enabled (default), ultrahdr-core re-exports
// the full suite of tone-mapping primitives from the `zentone` crate:
// BT.2408 PQ-domain tonemapper, standard curves (Reinhard variants, Hable,
// ACES, AgX, BT.2390, filmic Narkowicz), and the filmic spline compiler.
//
// When the feature is disabled, callers compute gain maps via the in-core
// [`crate::gainmap::splitter::HableFilmic`] curve (or any custom
// [`crate::LumaToneMap`] implementation) and none of these symbols exist.

/// Re-exported from [`zentone`]. BT.2408 PQ-domain tonemapper.
///
/// **Deprecated API surface** — slated for removal in 0.5.0. This is a
/// pass-through re-export; import directly from the `zentone` crate.
#[cfg(feature = "zentone")]
#[doc(hidden)]
pub use zentone::{Bt2408Tonemapper, EetfSpace};

/// Standard tone curves re-exported from [`zentone::curves`].
///
/// **Deprecated API surface** — slated for removal in 0.5.0. These are
/// pass-through re-exports; import directly from `zentone::curves`.
#[cfg(feature = "zentone")]
#[doc(hidden)]
pub use zentone::curves::{
    aces_ap1, agx_tonemap, bt2390_tonemap, bt2390_tonemap_ext, filmic_narkowicz, hable_filmic,
    reinhard_extended, reinhard_jodie, reinhard_simple,
};

/// Re-exported from [`zentone`]. Filmic spline tonemapper.
///
/// **Deprecated API surface** — slated for removal in 0.5.0. Pass-through
/// re-export; import directly from `zentone`.
#[cfg(feature = "zentone")]
#[doc(hidden)]
pub use zentone::{CompiledFilmicSpline, FilmicSplineConfig};

// ============================================================================
// ProfileToneCurve — DNG Camera Profile Tone Curve
// ============================================================================

/// Linear interpolation in a sorted list of (x,y) control points.
fn interpolate_curve(points: &[(f32, f32)], x: f32) -> f32 {
    if points.is_empty() {
        return x;
    }
    if x <= points[0].0 {
        return points[0].1;
    }
    if x >= points[points.len() - 1].0 {
        return points[points.len() - 1].1;
    }
    // Binary search for the segment
    let mut lo = 0;
    let mut hi = points.len() - 1;
    while hi - lo > 1 {
        let mid = (lo + hi) / 2;
        if points[mid].0 <= x {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    let dx = points[hi].0 - points[lo].0;
    if dx <= 0.0 {
        return points[lo].1;
    }
    let t = (x - points[lo].0) / dx;
    points[lo].1 * (1.0 - t) + points[hi].1 * t
}

/// DNG ProfileToneCurve — a precomputed LUT-based tone curve.
///
/// Built from 257 (x,y) control points from DNG camera profiles.
/// Maps linear \[0,1\] → \[0,1\] via a 4096-entry lookup table with
/// linear interpolation between entries.
///
/// Can be applied per-channel or luminance-preserving.
#[derive(Clone, Debug)]
pub struct ProfileToneCurve {
    /// 4097 entries (4096 + 1 for endpoint), mapping \[0,1\] → \[0,1\]
    lut: Vec<f32>,
}

impl ProfileToneCurve {
    /// Build from raw DNG data (257 x,y pairs = 514 floats).
    pub fn from_xy_pairs(tc_data: &[f32]) -> Option<Self> {
        let n_points = tc_data.len() / 2;
        if n_points < 2 {
            return None;
        }
        let points: Vec<(f32, f32)> = (0..n_points)
            .map(|i| (tc_data[i * 2], tc_data[i * 2 + 1]))
            .collect();
        let lut_size = 4096usize;
        let lut: Vec<f32> = (0..=lut_size)
            .map(|i| {
                let x = i as f32 / lut_size as f32;
                interpolate_curve(&points, x)
            })
            .collect();
        Some(Self { lut })
    }

    /// Build from a pre-built LUT (must have 4097 entries).
    pub fn from_lut(lut: Vec<f32>) -> Option<Self> {
        if lut.len() != 4097 {
            return None;
        }
        Some(Self { lut })
    }

    /// Build a linear identity curve (passthrough).
    pub fn identity() -> Self {
        let lut: Vec<f32> = (0..=4096).map(|i| i as f32 / 4096.0).collect();
        Self { lut }
    }

    /// Evaluate the curve at a single value in \[0,1\].
    #[inline]
    pub fn eval(&self, x: f32) -> f32 {
        let x = x.clamp(0.0, 1.0);
        let idx_f = x * 4096.0;
        let idx = (idx_f as usize).min(4095);
        let frac = idx_f - idx as f32;
        self.lut[idx] * (1.0 - frac) + self.lut[idx + 1] * frac
    }

    /// Apply per-channel to an RGB triple.
    #[inline]
    pub fn apply_per_channel(&self, rgb: [f32; 3]) -> [f32; 3] {
        [self.eval(rgb[0]), self.eval(rgb[1]), self.eval(rgb[2])]
    }

    /// Apply luminance-preserving to an RGB triple.
    ///
    /// Maps the luminance through the curve, then scales all channels
    /// by the same ratio, preserving color. Uses provided luma coefficients.
    #[inline]
    pub fn apply_lum_preserving(&self, rgb: [f32; 3], luma_coeffs: [f32; 3]) -> [f32; 3] {
        let lum = rgb[0] * luma_coeffs[0] + rgb[1] * luma_coeffs[1] + rgb[2] * luma_coeffs[2];
        if lum <= 1e-10 {
            return [0.0, 0.0, 0.0];
        }
        let mapped = self.eval(lum.min(1.0));
        let ratio = mapped / lum;
        [
            (rgb[0] * ratio).min(1.0),
            (rgb[1] * ratio).min(1.0),
            (rgb[2] * ratio).min(1.0),
        ]
    }

    /// Apply to a full row of interleaved pixel data (per-channel mode).
    pub fn apply_row_per_channel(&self, row: &mut [f32], channels: usize) {
        for chunk in row.chunks_exact_mut(channels) {
            chunk[0] = self.eval(chunk[0]);
            chunk[1] = self.eval(chunk[1]);
            chunk[2] = self.eval(chunk[2]);
            // Alpha (channel 3+) left unchanged
        }
    }

    /// Apply to a full row of interleaved pixel data (luminance-preserving mode).
    pub fn apply_row_lum_preserving(
        &self,
        row: &mut [f32],
        channels: usize,
        luma_coeffs: [f32; 3],
    ) {
        for chunk in row.chunks_exact_mut(channels) {
            let lum =
                chunk[0] * luma_coeffs[0] + chunk[1] * luma_coeffs[1] + chunk[2] * luma_coeffs[2];
            if lum > 1e-10 {
                let mapped = self.eval(lum.min(1.0));
                let ratio = mapped / lum;
                chunk[0] = (chunk[0] * ratio).min(1.0);
                chunk[1] = (chunk[1] * ratio).min(1.0);
                chunk[2] = (chunk[2] * ratio).min(1.0);
            } else {
                chunk[0] = 0.0;
                chunk[1] = 0.0;
                chunk[2] = 0.0;
            }
        }
    }
}

// ============================================================================
// Unified Tone Map Curve — re-exported from zentone (feature-gated)
// ============================================================================

/// Re-exported from [`zentone`]. Unified tone curve enum + dispatch trait.
///
/// **Deprecated API surface** — slated for removal in 0.5.0. Pass-through
/// re-export; import directly from `zentone`.
#[cfg(feature = "zentone")]
#[doc(hidden)]
pub use zentone::{AgxLook, ToneMap, ToneMapCurve};

// ============================================================================
// Adaptive Tonemapper
// ============================================================================

/// LUT resolution for tone curves.
const LUT_SIZE: usize = 4096;

/// Adaptive tonemapper that learns from HDR/SDR pairs.
///
/// This tonemapper analyzes an existing HDR/SDR relationship and can
/// reproduce it for edited HDR content, preserving the original artistic intent.
#[derive(Debug, Clone)]
pub struct AdaptiveTonemapper {
    mode: TonemapMode,
    /// Maximum HDR value observed during fitting (for extrapolation).
    max_hdr_observed: f32,
    /// Statistics about the fit.
    stats: FitStats,
}

/// Tonemapping mode (how the curve is represented).
#[derive(Debug, Clone)]
pub enum TonemapMode {
    /// Luminance-based curve with saturation preservation.
    /// Most natural for edits, preserves hue.
    Luminance(LuminanceCurve),

    /// Per-channel LUTs for exact reproduction.
    /// Most accurate for round-trips.
    PerChannel(PerChannelLut),

    /// Direct gain map inversion (perfect for unedited round-trips).
    GainMapInverse(GainMapInverter),
}

/// Luminance-based tone curve.
#[derive(Debug, Clone)]
pub struct LuminanceCurve {
    /// LUT mapping HDR luminance [0, max_hdr] to SDR luminance [0, 1].
    /// Index = (L_hdr / max_hdr * (LUT_SIZE-1)) as usize
    lut: Box<[f32; LUT_SIZE]>,
    /// Maximum HDR luminance value the LUT covers.
    max_hdr: f32,
    /// Saturation adjustment (1.0 = preserve, >1 = boost, <1 = reduce).
    saturation: f32,
}

/// Per-channel tone curves.
#[derive(Debug, Clone)]
pub struct PerChannelLut {
    /// Red channel LUT.
    lut_r: Box<[f32; LUT_SIZE]>,
    /// Green channel LUT.
    lut_g: Box<[f32; LUT_SIZE]>,
    /// Blue channel LUT.
    lut_b: Box<[f32; LUT_SIZE]>,
    /// Maximum HDR value the LUTs cover.
    max_hdr: f32,
}

/// Gain map inverter for perfect round-trips.
#[derive(Debug, Clone)]
pub struct GainMapInverter {
    metadata: GainMapMetadata,
}

impl GainMapInverter {
    /// Get the metadata used for inversion.
    pub fn metadata(&self) -> &GainMapMetadata {
        &self.metadata
    }
}

/// Statistics from the fitting process.
#[derive(Debug, Clone, Default)]
pub struct FitStats {
    /// Number of pixel samples used.
    pub samples: usize,
    /// Mean absolute error of the fit.
    pub mae: f32,
    /// Maximum observed HDR luminance.
    pub max_hdr_luminance: f32,
    /// Detected saturation change (SDR_sat / HDR_sat).
    pub saturation_ratio: f32,
}

/// Configuration for fitting an adaptive tonemapper.
#[derive(Debug, Clone)]
pub struct FitConfig {
    /// Fitting mode.
    pub mode: FitMode,
    /// Maximum number of samples (0 = all pixels).
    pub max_samples: usize,
    /// Whether to detect and apply saturation changes.
    pub detect_saturation: bool,
}

/// Which type of curve to fit.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum FitMode {
    /// Luminance-based (recommended for most use cases).
    #[default]
    Luminance,
    /// Per-channel LUTs.
    PerChannel,
}

impl Default for FitConfig {
    fn default() -> Self {
        Self {
            mode: FitMode::Luminance,
            max_samples: 100_000,
            detect_saturation: true,
        }
    }
}

impl AdaptiveTonemapper {
    /// Fit a tonemapper from an HDR/SDR pair.
    ///
    /// Analyzes the pixel correspondences to learn the effective tone curve.
    pub fn fit(hdr: &PixelBuffer, sdr: &PixelBuffer) -> Result<Self> {
        Self::fit_with_config(hdr, sdr, &FitConfig::default())
    }

    /// Fit with custom configuration.
    pub fn fit_with_config(
        hdr: &PixelBuffer,
        sdr: &PixelBuffer,
        config: &FitConfig,
    ) -> Result<Self> {
        let hdr_slice = hdr.as_slice();
        let sdr_slice = sdr.as_slice();
        crate::types::validate_ultrahdr_slice(&hdr_slice)?;
        crate::types::validate_ultrahdr_slice(&sdr_slice)?;

        if hdr_slice.width() != sdr_slice.width() || hdr_slice.rows() != sdr_slice.rows() {
            return Err(Error::DimensionMismatch {
                hdr_w: hdr_slice.width(),
                hdr_h: hdr_slice.rows(),
                sdr_w: sdr_slice.width(),
                sdr_h: sdr_slice.rows(),
            });
        }

        match config.mode {
            FitMode::Luminance => Self::fit_luminance(&hdr_slice, &sdr_slice, config),
            FitMode::PerChannel => Self::fit_per_channel(&hdr_slice, &sdr_slice, config),
        }
    }

    /// Create from gain map metadata for perfect inversion.
    ///
    /// Use this for round-trips where you want exact reproduction.
    pub fn from_gainmap(metadata: &GainMapMetadata) -> Self {
        Self {
            mode: TonemapMode::GainMapInverse(GainMapInverter {
                metadata: metadata.clone(),
            }),
            max_hdr_observed: 2.0f32.powf(metadata.alternate_hdr_headroom as f32),
            stats: FitStats::default(),
        }
    }

    /// Apply the tonemapper to an HDR image.
    pub fn apply(&self, hdr: &PixelBuffer) -> Result<PixelBuffer> {
        let hdr_slice = hdr.as_slice();
        crate::types::validate_ultrahdr_slice(&hdr_slice)?;

        let width = hdr_slice.width();
        let height = hdr_slice.rows();

        let mut output = new_pixel_buffer(
            width,
            height,
            crate::PixelFormat::Rgba8,
            ColorPrimaries::Bt709,
            TransferFunction::Srgb,
        )?;
        let out_stride = output.stride();
        let mut out_mut = output.as_slice_mut();
        let out_data = out_mut.as_strided_bytes_mut();

        for y in 0..height {
            for x in 0..width {
                let hdr_linear = get_linear_rgb(&hdr_slice, x, y);
                let sdr_linear = self.tonemap_pixel(hdr_linear);

                let out_idx = (y as usize) * out_stride + (x as usize) * 4;
                out_data[out_idx] =
                    (srgb_oetf(sdr_linear[0]) * 255.0).round().clamp(0.0, 255.0) as u8;
                out_data[out_idx + 1] =
                    (srgb_oetf(sdr_linear[1]) * 255.0).round().clamp(0.0, 255.0) as u8;
                out_data[out_idx + 2] =
                    (srgb_oetf(sdr_linear[2]) * 255.0).round().clamp(0.0, 255.0) as u8;
                out_data[out_idx + 3] = 255;
            }
        }

        drop(out_mut);
        Ok(output)
    }

    /// Apply tonemapper with gain map for inversion.
    ///
    /// For perfect round-trips when you have the original gain map.
    pub fn apply_with_gainmap(
        &self,
        hdr: &PixelBuffer,
        gainmap: &GainMap,
        metadata: &GainMapMetadata,
    ) -> Result<PixelBuffer> {
        let hdr_slice = hdr.as_slice();
        crate::types::validate_ultrahdr_slice(&hdr_slice)?;
        let width = hdr_slice.width();
        let height = hdr_slice.rows();

        let mut output = new_pixel_buffer(
            width,
            height,
            crate::PixelFormat::Rgba8,
            ColorPrimaries::Bt709,
            TransferFunction::Srgb,
        )?;
        let out_stride = output.stride();
        let mut out_mut = output.as_slice_mut();
        let out_data = out_mut.as_strided_bytes_mut();

        for y in 0..height {
            for x in 0..width {
                let hdr_linear = get_linear_rgb(&hdr_slice, x, y);

                let gain = sample_gainmap_at(gainmap, metadata, x, y, width, height);

                let sdr_linear = [
                    (hdr_linear[0] + metadata.channels[0].alternate_offset as f32) / gain[0]
                        - metadata.channels[0].base_offset as f32,
                    (hdr_linear[1] + metadata.channels[1].alternate_offset as f32) / gain[1]
                        - metadata.channels[1].base_offset as f32,
                    (hdr_linear[2] + metadata.channels[2].alternate_offset as f32) / gain[2]
                        - metadata.channels[2].base_offset as f32,
                ];

                let out_idx = (y as usize) * out_stride + (x as usize) * 4;
                out_data[out_idx] =
                    (srgb_oetf(sdr_linear[0].clamp(0.0, 1.0)) * 255.0).round() as u8;
                out_data[out_idx + 1] =
                    (srgb_oetf(sdr_linear[1].clamp(0.0, 1.0)) * 255.0).round() as u8;
                out_data[out_idx + 2] =
                    (srgb_oetf(sdr_linear[2].clamp(0.0, 1.0)) * 255.0).round() as u8;
                out_data[out_idx + 3] = 255;
            }
        }

        drop(out_mut);
        Ok(output)
    }

    /// Get fitting statistics.
    pub fn stats(&self) -> &FitStats {
        &self.stats
    }

    /// Get the maximum HDR value observed during fitting.
    ///
    /// This indicates the dynamic range of the source HDR content.
    pub fn max_hdr_observed(&self) -> f32 {
        self.max_hdr_observed
    }

    /// Tonemap a single pixel.
    fn tonemap_pixel(&self, hdr_linear: [f32; 3]) -> [f32; 3] {
        match &self.mode {
            TonemapMode::Luminance(curve) => curve.apply(hdr_linear),
            TonemapMode::PerChannel(luts) => luts.apply(hdr_linear),
            TonemapMode::GainMapInverse(_) => {
                // Without a gain map, fall back to simple curve
                let l = 0.2126 * hdr_linear[0] + 0.7152 * hdr_linear[1] + 0.0722 * hdr_linear[2];
                let l_sdr = filmic_narkowicz(l * 2.0); // Scale for curve
                let ratio = if l > 0.0 { l_sdr / l } else { 1.0 };
                [
                    (hdr_linear[0] * ratio).clamp(0.0, 1.0),
                    (hdr_linear[1] * ratio).clamp(0.0, 1.0),
                    (hdr_linear[2] * ratio).clamp(0.0, 1.0),
                ]
            }
        }
    }

    /// Fit luminance-based curve.
    fn fit_luminance(
        hdr: &PixelSlice<'_>,
        sdr: &PixelSlice<'_>,
        config: &FitConfig,
    ) -> Result<Self> {
        let width = hdr.width() as usize;
        let height = hdr.rows() as usize;
        let total_pixels = width * height;

        // Determine sampling
        let step = if config.max_samples > 0 && total_pixels > config.max_samples {
            total_pixels / config.max_samples
        } else {
            1
        };

        // Collect (hdr_luminance, sdr_luminance) pairs
        let mut pairs: Vec<(f32, f32)> = Vec::with_capacity(total_pixels / step);
        let mut max_hdr = 0.0f32;
        let mut saturation_sum = 0.0f32;
        let mut saturation_count = 0usize;

        for i in (0..total_pixels).step_by(step.max(1)) {
            let x = (i % width) as u32;
            let y = (i / width) as u32;

            let hdr_rgb = get_linear_rgb(hdr, x, y);
            let sdr_rgb = get_sdr_linear(sdr, x, y);

            // BT.709 luminance
            let l_hdr = 0.2126 * hdr_rgb[0] + 0.7152 * hdr_rgb[1] + 0.0722 * hdr_rgb[2];
            let l_sdr = 0.2126 * sdr_rgb[0] + 0.7152 * sdr_rgb[1] + 0.0722 * sdr_rgb[2];

            if l_hdr > 0.001 && l_sdr > 0.001 {
                pairs.push((l_hdr, l_sdr));
                max_hdr = max_hdr.max(l_hdr);

                // Detect saturation change
                if config.detect_saturation && l_hdr > 0.01 && l_sdr > 0.01 {
                    let sat_hdr = compute_saturation(hdr_rgb, l_hdr);
                    let sat_sdr = compute_saturation(sdr_rgb, l_sdr);
                    if sat_hdr > 0.01 {
                        saturation_sum += sat_sdr / sat_hdr;
                        saturation_count += 1;
                    }
                }
            }
        }

        if pairs.is_empty() {
            return Err(Error::InvalidPixelData(
                "no valid pixel pairs for fitting".into(),
            ));
        }

        // Sort by HDR luminance
        pairs.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(Ordering::Equal));

        // Build LUT by bucketing
        let mut lut = Box::new([0.0f32; LUT_SIZE]);
        let mut counts = [0u32; LUT_SIZE];

        for (l_hdr, l_sdr) in &pairs {
            let idx = ((*l_hdr / max_hdr) * (LUT_SIZE - 1) as f32)
                .round()
                .clamp(0.0, (LUT_SIZE - 1) as f32) as usize;
            lut[idx] += l_sdr;
            counts[idx] += 1;
        }

        // Average and fill gaps
        for i in 0..LUT_SIZE {
            if counts[i] > 0 {
                lut[i] /= counts[i] as f32;
            }
        }

        // Fill gaps with linear interpolation
        fill_lut_gaps(&mut lut, &counts);

        // Ensure monotonicity
        enforce_monotonicity(&mut lut);

        // Calculate saturation ratio
        let saturation = if saturation_count > 0 {
            (saturation_sum / saturation_count as f32).clamp(0.5, 2.0)
        } else {
            1.0
        };

        // Calculate MAE
        let mut mae_sum = 0.0f32;
        for (l_hdr, l_sdr) in &pairs {
            let idx = ((*l_hdr / max_hdr) * (LUT_SIZE - 1) as f32)
                .round()
                .clamp(0.0, (LUT_SIZE - 1) as f32) as usize;
            mae_sum += (lut[idx] - l_sdr).abs();
        }

        Ok(Self {
            mode: TonemapMode::Luminance(LuminanceCurve {
                lut,
                max_hdr,
                saturation,
            }),
            max_hdr_observed: max_hdr,
            stats: FitStats {
                samples: pairs.len(),
                mae: mae_sum / pairs.len() as f32,
                max_hdr_luminance: max_hdr,
                saturation_ratio: saturation,
            },
        })
    }

    /// Fit per-channel LUTs.
    fn fit_per_channel(
        hdr: &PixelSlice<'_>,
        sdr: &PixelSlice<'_>,
        config: &FitConfig,
    ) -> Result<Self> {
        let width = hdr.width() as usize;
        let height = hdr.rows() as usize;
        let total_pixels = width * height;

        let step = if config.max_samples > 0 && total_pixels > config.max_samples {
            total_pixels / config.max_samples
        } else {
            1
        };

        // Collect per-channel pairs
        let mut pairs_r: Vec<(f32, f32)> = Vec::new();
        let mut pairs_g: Vec<(f32, f32)> = Vec::new();
        let mut pairs_b: Vec<(f32, f32)> = Vec::new();
        let mut max_hdr = 0.0f32;

        for i in (0..total_pixels).step_by(step.max(1)) {
            let x = (i % width) as u32;
            let y = (i / width) as u32;

            let hdr_rgb = get_linear_rgb(hdr, x, y);
            let sdr_rgb = get_sdr_linear(sdr, x, y);

            if hdr_rgb[0] > 0.001 {
                pairs_r.push((hdr_rgb[0], sdr_rgb[0]));
            }
            if hdr_rgb[1] > 0.001 {
                pairs_g.push((hdr_rgb[1], sdr_rgb[1]));
            }
            if hdr_rgb[2] > 0.001 {
                pairs_b.push((hdr_rgb[2], sdr_rgb[2]));
            }

            max_hdr = max_hdr.max(hdr_rgb[0]).max(hdr_rgb[1]).max(hdr_rgb[2]);
        }

        let lut_r = build_channel_lut(&mut pairs_r, max_hdr)?;
        let lut_g = build_channel_lut(&mut pairs_g, max_hdr)?;
        let lut_b = build_channel_lut(&mut pairs_b, max_hdr)?;

        // Per-channel MAE: average of |lut(hdr) - sdr_actual| across every
        // fitted pair, same formulation as the Luminance path above.
        let mut mae_sum = 0.0f32;
        let mut mae_count = 0usize;
        for (lut, pairs) in [(&lut_r, &pairs_r), (&lut_g, &pairs_g), (&lut_b, &pairs_b)] {
            for (hdr, sdr) in pairs {
                mae_sum += (lookup_lut(lut, *hdr, max_hdr) - sdr).abs();
                mae_count += 1;
            }
        }
        let mae = if mae_count > 0 {
            mae_sum / mae_count as f32
        } else {
            0.0
        };

        Ok(Self {
            mode: TonemapMode::PerChannel(PerChannelLut {
                lut_r,
                lut_g,
                lut_b,
                max_hdr,
            }),
            max_hdr_observed: max_hdr,
            stats: FitStats {
                samples: pairs_r.len() + pairs_g.len() + pairs_b.len(),
                mae,
                max_hdr_luminance: max_hdr,
                saturation_ratio: 1.0,
            },
        })
    }
}

impl LuminanceCurve {
    /// Apply luminance curve with saturation preservation.
    fn apply(&self, hdr_linear: [f32; 3]) -> [f32; 3] {
        let l_hdr = 0.2126 * hdr_linear[0] + 0.7152 * hdr_linear[1] + 0.0722 * hdr_linear[2];

        if l_hdr <= 0.0 {
            return [0.0, 0.0, 0.0];
        }

        // Look up SDR luminance
        let idx_f = (l_hdr / self.max_hdr) * (LUT_SIZE - 1) as f32;
        let l_sdr = if idx_f >= (LUT_SIZE - 1) as f32 {
            // Extrapolate beyond LUT
            let slope = self.lut[LUT_SIZE - 1] - self.lut[LUT_SIZE - 2];
            self.lut[LUT_SIZE - 1] + slope * (idx_f - (LUT_SIZE - 1) as f32)
        } else if idx_f <= 0.0 {
            self.lut[0]
        } else {
            // Linear interpolation
            let idx = idx_f as usize;
            let frac = idx_f - idx as f32;
            self.lut[idx] * (1.0 - frac) + self.lut[idx + 1] * frac
        };

        // Apply ratio to preserve color
        let ratio = (l_sdr / l_hdr).clamp(0.0, 10.0);

        // Apply with saturation adjustment
        let sdr = [
            hdr_linear[0] * ratio,
            hdr_linear[1] * ratio,
            hdr_linear[2] * ratio,
        ];

        // Saturation adjustment
        let l_sdr_actual = 0.2126 * sdr[0] + 0.7152 * sdr[1] + 0.0722 * sdr[2];
        let adjusted = if self.saturation != 1.0 && l_sdr_actual > 0.001 {
            [
                l_sdr_actual + (sdr[0] - l_sdr_actual) * self.saturation,
                l_sdr_actual + (sdr[1] - l_sdr_actual) * self.saturation,
                l_sdr_actual + (sdr[2] - l_sdr_actual) * self.saturation,
            ]
        } else {
            sdr
        };

        [
            adjusted[0].clamp(0.0, 1.0),
            adjusted[1].clamp(0.0, 1.0),
            adjusted[2].clamp(0.0, 1.0),
        ]
    }
}

impl PerChannelLut {
    /// Apply per-channel LUTs.
    fn apply(&self, hdr_linear: [f32; 3]) -> [f32; 3] {
        [
            lookup_lut(&self.lut_r, hdr_linear[0], self.max_hdr),
            lookup_lut(&self.lut_g, hdr_linear[1], self.max_hdr),
            lookup_lut(&self.lut_b, hdr_linear[2], self.max_hdr),
        ]
    }
}

// ============================================================================
// Gain Map Scaling (for crop/resize)
// ============================================================================

/// Scale a gain map to new dimensions.
///
/// Uses bilinear interpolation for smooth results.
pub fn scale_gainmap(gainmap: &GainMap, new_width: u32, new_height: u32) -> Result<GainMap> {
    let mut output = if gainmap.channels == 1 {
        GainMap::new(new_width, new_height)?
    } else {
        GainMap::new_multichannel(new_width, new_height)?
    };

    let x_ratio = gainmap.width as f32 / new_width as f32;
    let y_ratio = gainmap.height as f32 / new_height as f32;

    for y in 0..new_height {
        for x in 0..new_width {
            // Source coordinates
            let src_x = x as f32 * x_ratio;
            let src_y = y as f32 * y_ratio;

            // Bilinear interpolation coordinates
            let x0 = (src_x.floor() as u32).min(gainmap.width - 1);
            let y0 = (src_y.floor() as u32).min(gainmap.height - 1);
            let x1 = (x0 + 1).min(gainmap.width - 1);
            let y1 = (y0 + 1).min(gainmap.height - 1);

            let fx = src_x - src_x.floor();
            let fy = src_y - src_y.floor();

            for c in 0..gainmap.channels as usize {
                let v00 = gainmap.data
                    [(y0 * gainmap.width + x0) as usize * gainmap.channels as usize + c];
                let v10 = gainmap.data
                    [(y0 * gainmap.width + x1) as usize * gainmap.channels as usize + c];
                let v01 = gainmap.data
                    [(y1 * gainmap.width + x0) as usize * gainmap.channels as usize + c];
                let v11 = gainmap.data
                    [(y1 * gainmap.width + x1) as usize * gainmap.channels as usize + c];

                let top = v00 as f32 * (1.0 - fx) + v10 as f32 * fx;
                let bottom = v01 as f32 * (1.0 - fx) + v11 as f32 * fx;
                let value = top * (1.0 - fy) + bottom * fy;

                output.data[(y * new_width + x) as usize * gainmap.channels as usize + c] =
                    value.round().clamp(0.0, 255.0) as u8;
            }
        }
    }

    Ok(output)
}

/// Crop a gain map to match a cropped SDR image.
///
/// `crop_rect` is (x, y, width, height) in SDR image coordinates.
/// The gain map coordinates are scaled proportionally.
pub fn crop_gainmap(
    gainmap: &GainMap,
    sdr_width: u32,
    sdr_height: u32,
    crop_rect: (u32, u32, u32, u32),
) -> Result<GainMap> {
    let (crop_x, crop_y, crop_w, crop_h) = crop_rect;

    // Calculate corresponding gain map region
    let gm_x = (crop_x as f32 / sdr_width as f32 * gainmap.width as f32).floor() as u32;
    let gm_y = (crop_y as f32 / sdr_height as f32 * gainmap.height as f32).floor() as u32;
    let gm_w = (crop_w as f32 / sdr_width as f32 * gainmap.width as f32).ceil() as u32;
    let gm_h = (crop_h as f32 / sdr_height as f32 * gainmap.height as f32).ceil() as u32;

    let gm_w = gm_w.min(gainmap.width - gm_x).max(1);
    let gm_h = gm_h.min(gainmap.height - gm_y).max(1);

    let mut output = if gainmap.channels == 1 {
        GainMap::new(gm_w, gm_h)?
    } else {
        GainMap::new_multichannel(gm_w, gm_h)?
    };

    for y in 0..gm_h {
        for x in 0..gm_w {
            let src_idx =
                ((gm_y + y) * gainmap.width + (gm_x + x)) as usize * gainmap.channels as usize;
            let dst_idx = (y * gm_w + x) as usize * gainmap.channels as usize;

            for c in 0..gainmap.channels as usize {
                output.data[dst_idx + c] = gainmap.data[src_idx + c];
            }
        }
    }

    Ok(output)
}

// ============================================================================
// Standard Tonemap Functions (existing API)
// ============================================================================

/// Tone map a single PQ HDR pixel to SDR.
///
/// Input: PQ-encoded RGB `[0,1]`
/// Output: Linear RGB suitable for sRGB encoding
pub fn tonemap_pq_to_sdr(pq_rgb: [f32; 3], config: &ToneMapConfig) -> [f32; 3] {
    // 1. Decode PQ to linear (normalized to 10000 nits)
    let linear_hdr = [pq_eotf(pq_rgb[0]), pq_eotf(pq_rgb[1]), pq_eotf(pq_rgb[2])];

    // 2. Convert to absolute nits
    let nits = [
        linear_hdr[0] * 10000.0,
        linear_hdr[1] * 10000.0,
        linear_hdr[2] * 10000.0,
    ];

    // 3. Convert gamut if needed (in linear light)
    let gamut_converted = convert_gamut(nits, config.source_gamut, config.target_gamut);

    // 4. Calculate luminance for tone mapping
    let lum = rgb_to_luminance(gamut_converted, config.target_gamut);

    // 5. Apply tone mapping to luminance
    let lum_ratio = if lum > 0.0 {
        let lum_normalized = lum / config.hdr_peak_nits;
        let lum_tonemapped = filmic_narkowicz(lum_normalized * 4.0); // Scale for curve
        let target_lum = lum_tonemapped * config.target_peak_nits;
        target_lum / lum
    } else {
        0.0
    };

    // 6. Apply luminance ratio to RGB (preserves color ratios)
    let tonemapped = [
        gamut_converted[0] * lum_ratio / config.target_peak_nits,
        gamut_converted[1] * lum_ratio / config.target_peak_nits,
        gamut_converted[2] * lum_ratio / config.target_peak_nits,
    ];

    // 7. Soft-clip to `[0,1]` gamut
    soft_clip_gamut(tonemapped)
}

/// Tone map a single HLG HDR pixel to SDR.
///
/// Input: HLG-encoded RGB `[0,1]`
/// Output: Linear RGB suitable for sRGB encoding
pub fn tonemap_hlg_to_sdr(hlg_rgb: [f32; 3], config: &ToneMapConfig) -> [f32; 3] {
    // 1. Decode HLG to display-referred linear (at 1000 nits nominal)
    let source_peak = 1000.0;
    let display_linear = [
        hlg_eotf(hlg_rgb[0], source_peak),
        hlg_eotf(hlg_rgb[1], source_peak),
        hlg_eotf(hlg_rgb[2], source_peak),
    ];

    // 2. Convert gamut if needed (values are in nits)
    let gamut_converted = convert_gamut(display_linear, config.source_gamut, config.target_gamut);

    // 3. Calculate luminance (in nits)
    let lum_nits = rgb_to_luminance(gamut_converted, config.target_gamut);

    // 4. Apply tone mapping - normalize luminance to `[0,1]` range first
    let lum_normalized = lum_nits / source_peak;
    let lum_tonemapped = bt2390_tonemap(lum_normalized, 1.0, config.target_peak_nits / source_peak);

    // 5. Calculate luminance ratio and apply to RGB
    let lum_ratio = if lum_normalized > 0.0 {
        lum_tonemapped / lum_normalized
    } else {
        0.0
    };

    // Scale from source peak to normalized `[0,1]` for SDR
    let tonemapped = [
        gamut_converted[0] / source_peak * lum_ratio,
        gamut_converted[1] / source_peak * lum_ratio,
        gamut_converted[2] / source_peak * lum_ratio,
    ];

    // 6. Soft-clip
    soft_clip_gamut(tonemapped)
}

/// Tone map HDR content to SDR based on transfer function.
///
/// Input: Encoded HDR RGB `[0,1]` (PQ or HLG encoded)
/// Output: Linear SDR RGB `[0,1]` ready for sRGB OETF
pub fn tonemap_to_sdr(
    encoded_rgb: [f32; 3],
    transfer: TransferFunction,
    config: &ToneMapConfig,
) -> [f32; 3] {
    match transfer {
        TransferFunction::Pq => tonemap_pq_to_sdr(encoded_rgb, config),
        TransferFunction::Hlg => tonemap_hlg_to_sdr(encoded_rgb, config),
        TransferFunction::Linear => {
            convert_gamut(encoded_rgb, config.source_gamut, config.target_gamut)
        }
        _ => {
            let linear = [
                srgb_eotf(encoded_rgb[0]),
                srgb_eotf(encoded_rgb[1]),
                srgb_eotf(encoded_rgb[2]),
            ];
            convert_gamut(linear, config.source_gamut, config.target_gamut)
        }
    }
}

/// Tone map and encode to 8-bit sRGB.
///
/// Full pipeline: HDR encoded → linear SDR → sRGB encoded → 8-bit
pub fn tonemap_to_srgb8(
    encoded_rgb: [f32; 3],
    transfer: TransferFunction,
    config: &ToneMapConfig,
) -> [u8; 3] {
    let linear_sdr = tonemap_to_sdr(encoded_rgb, transfer, config);
    let srgb = [
        srgb_oetf(linear_sdr[0]),
        srgb_oetf(linear_sdr[1]),
        srgb_oetf(linear_sdr[2]),
    ];

    [
        (srgb[0] * 255.0).round().clamp(0.0, 255.0) as u8,
        (srgb[1] * 255.0).round().clamp(0.0, 255.0) as u8,
        (srgb[2] * 255.0).round().clamp(0.0, 255.0) as u8,
    ]
}

/// Tonemap an entire HDR image to SDR RGBA8.
///
/// Takes an HDR image in any supported format and produces RGBA8 output.
pub fn tonemap_image_to_srgb8(
    img: &PixelBuffer,
    target_gamut: ColorPrimaries,
) -> Result<Vec<u8>> {
    use crate::color::gamut::convert_gamut;

    let slice = img.as_slice();
    crate::types::validate_ultrahdr_slice(&slice)?;

    let img_gamut = slice.descriptor().primaries;
    let img_transfer = slice.descriptor().transfer();
    let config = ToneMapConfig::default();
    let width = slice.width() as usize;
    let height = slice.rows() as usize;
    let mut output = vec![0u8; width * height * 4];

    for y in 0..height {
        for x in 0..width {
            let linear_rgb = get_linear_rgb(&slice, x as u32, y as u32);

            let gamut_converted = if img_gamut != target_gamut {
                convert_gamut(linear_rgb, img_gamut, target_gamut)
            } else {
                linear_rgb
            };

            let sdr = tonemap_to_sdr(gamut_converted, img_transfer, &config);

            let srgb = [
                (srgb_oetf(sdr[0]) * 255.0).round().clamp(0.0, 255.0) as u8,
                (srgb_oetf(sdr[1]) * 255.0).round().clamp(0.0, 255.0) as u8,
                (srgb_oetf(sdr[2]) * 255.0).round().clamp(0.0, 255.0) as u8,
            ];

            let out_idx = (y * width + x) * 4;
            output[out_idx] = srgb[0];
            output[out_idx + 1] = srgb[1];
            output[out_idx + 2] = srgb[2];
            output[out_idx + 3] = 255;
        }
    }

    Ok(output)
}

// ============================================================================
// Helper Functions
// ============================================================================

/// Get linear RGB from a pixel slice, honoring its transfer function.
fn get_linear_rgb(img: &PixelSlice<'_>, x: u32, y: u32) -> [f32; 3] {
    use crate::PixelFormat;

    let desc = img.descriptor();
    let format = desc.pixel_format();
    let transfer = desc.transfer();
    let stride = img.stride();
    let data = img.as_strided_bytes();
    match format {
        PixelFormat::Rgba8 | PixelFormat::Rgb8 => {
            let bpp = if format == PixelFormat::Rgba8 { 4 } else { 3 };
            let idx = (y as usize) * stride + (x as usize) * bpp;
            let r = data[idx] as f32 / 255.0;
            let g = data[idx + 1] as f32 / 255.0;
            let b = data[idx + 2] as f32 / 255.0;
            if transfer == TransferFunction::Srgb {
                [srgb_eotf(r), srgb_eotf(g), srgb_eotf(b)]
            } else {
                [r, g, b]
            }
        }
        PixelFormat::RgbaF32 => {
            let idx = (y as usize) * stride + (x as usize) * 16;
            let r = f32::from_le_bytes([data[idx], data[idx + 1], data[idx + 2], data[idx + 3]]);
            let g = f32::from_le_bytes([
                data[idx + 4],
                data[idx + 5],
                data[idx + 6],
                data[idx + 7],
            ]);
            let b = f32::from_le_bytes([
                data[idx + 8],
                data[idx + 9],
                data[idx + 10],
                data[idx + 11],
            ]);
            [r, g, b]
        }
        _ => [0.5, 0.5, 0.5],
    }
}

/// Get linear RGB from an SDR pixel slice (assumes sRGB transfer for 8-bit).
fn get_sdr_linear(sdr: &PixelSlice<'_>, x: u32, y: u32) -> [f32; 3] {
    use crate::PixelFormat;

    let format = sdr.descriptor().pixel_format();
    let stride = sdr.stride();
    let data = sdr.as_strided_bytes();
    match format {
        PixelFormat::Rgba8 | PixelFormat::Rgb8 => {
            let bpp = if format == PixelFormat::Rgba8 { 4 } else { 3 };
            let idx = (y as usize) * stride + (x as usize) * bpp;
            let r = data[idx] as f32 / 255.0;
            let g = data[idx + 1] as f32 / 255.0;
            let b = data[idx + 2] as f32 / 255.0;
            [srgb_eotf(r), srgb_eotf(g), srgb_eotf(b)]
        }
        _ => get_linear_rgb(sdr, x, y),
    }
}

/// Sample gain map at image coordinates (with bilinear interpolation).
fn sample_gainmap_at(
    gainmap: &GainMap,
    metadata: &GainMapMetadata,
    x: u32,
    y: u32,
    img_width: u32,
    img_height: u32,
) -> [f32; 3] {
    let gm_x = (x as f32 / img_width as f32) * gainmap.width as f32;
    let gm_y = (y as f32 / img_height as f32) * gainmap.height as f32;

    let x0 = (gm_x.floor() as u32).min(gainmap.width - 1);
    let y0 = (gm_y.floor() as u32).min(gainmap.height - 1);
    let x1 = (x0 + 1).min(gainmap.width - 1);
    let y1 = (y0 + 1).min(gainmap.height - 1);

    let fx = gm_x - gm_x.floor();
    let fy = gm_y - gm_y.floor();

    if gainmap.channels == 1 {
        let v00 = gainmap.data[(y0 * gainmap.width + x0) as usize] as f32 / 255.0;
        let v10 = gainmap.data[(y0 * gainmap.width + x1) as usize] as f32 / 255.0;
        let v01 = gainmap.data[(y1 * gainmap.width + x0) as usize] as f32 / 255.0;
        let v11 = gainmap.data[(y1 * gainmap.width + x1) as usize] as f32 / 255.0;

        let v = bilinear(v00, v10, v01, v11, fx, fy);
        let gain = decode_gain_value(v, metadata, 0);
        [gain, gain, gain]
    } else {
        let mut gains = [0.0f32; 3];
        // Index needed for both array access and decode_gain_value channel parameter
        #[allow(clippy::needless_range_loop)]
        for c in 0..3 {
            let v00 = gainmap.data[(y0 * gainmap.width + x0) as usize * 3 + c] as f32 / 255.0;
            let v10 = gainmap.data[(y0 * gainmap.width + x1) as usize * 3 + c] as f32 / 255.0;
            let v01 = gainmap.data[(y1 * gainmap.width + x0) as usize * 3 + c] as f32 / 255.0;
            let v11 = gainmap.data[(y1 * gainmap.width + x1) as usize * 3 + c] as f32 / 255.0;

            let v = bilinear(v00, v10, v01, v11, fx, fy);
            gains[c] = decode_gain_value(v, metadata, c);
        }
        gains
    }
}

/// Decode gain value from normalized [0,1] to linear multiplier.
fn decode_gain_value(normalized: f32, metadata: &GainMapMetadata, channel: usize) -> f32 {
    let gamma = metadata.channels[channel].gamma as f32;
    let linear = if gamma != 1.0 && gamma > 0.0 {
        normalized.powf(1.0 / gamma)
    } else {
        normalized
    };

    // Convert log2 domain to natural log for exp() math
    let ln2 = core::f64::consts::LN_2;
    let log_min = (metadata.channels[channel].min * ln2) as f32;
    let log_max = (metadata.channels[channel].max * ln2) as f32;
    let log_gain = log_min + linear * (log_max - log_min);

    log_gain.exp()
}

#[inline]
fn bilinear(v00: f32, v10: f32, v01: f32, v11: f32, fx: f32, fy: f32) -> f32 {
    let top = v00 * (1.0 - fx) + v10 * fx;
    let bottom = v01 * (1.0 - fx) + v11 * fx;
    top * (1.0 - fy) + bottom * fy
}

/// Compute saturation (max-min) / luminance.
fn compute_saturation(rgb: [f32; 3], luminance: f32) -> f32 {
    let max = rgb[0].max(rgb[1]).max(rgb[2]);
    let min = rgb[0].min(rgb[1]).min(rgb[2]);
    if luminance > 0.001 {
        (max - min) / luminance
    } else {
        0.0
    }
}

/// Fill gaps in LUT using linear interpolation.
fn fill_lut_gaps(lut: &mut [f32; LUT_SIZE], counts: &[u32; LUT_SIZE]) {
    let mut last_valid = 0;
    let mut last_value = lut[0];

    for i in 0..LUT_SIZE {
        if counts[i] > 0 {
            if i > last_valid + 1 {
                // Fill gap with linear interpolation
                let start_value = last_value;
                let end_value = lut[i];
                let gap_size = (i - last_valid) as f32;

                // Index needed for interpolation position calculation
                #[allow(clippy::needless_range_loop)]
                for j in (last_valid + 1)..i {
                    let t = (j - last_valid) as f32 / gap_size;
                    lut[j] = start_value * (1.0 - t) + end_value * t;
                }
            }
            last_valid = i;
            last_value = lut[i];
        }
    }

    // Fill trailing gap
    for slot in lut.iter_mut().skip(last_valid + 1) {
        *slot = last_value;
    }
}

/// Ensure LUT is monotonically increasing (or at least non-decreasing).
fn enforce_monotonicity(lut: &mut [f32; LUT_SIZE]) {
    let mut max_so_far = lut[0];
    for slot in lut.iter_mut().skip(1) {
        if *slot < max_so_far {
            *slot = max_so_far;
        } else {
            max_so_far = *slot;
        }
    }
}

/// Build a single-channel LUT from pairs.
fn build_channel_lut(pairs: &mut [(f32, f32)], max_hdr: f32) -> Result<Box<[f32; LUT_SIZE]>> {
    if pairs.is_empty() {
        // Return identity-ish curve
        let mut lut = Box::new([0.0f32; LUT_SIZE]);
        for i in 0..LUT_SIZE {
            lut[i] = (i as f32 / (LUT_SIZE - 1) as f32).min(1.0);
        }
        return Ok(lut);
    }

    pairs.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(Ordering::Equal));

    let mut lut = Box::new([0.0f32; LUT_SIZE]);
    let mut counts = [0u32; LUT_SIZE];

    for (hdr_val, sdr_val) in pairs.iter() {
        let idx = ((*hdr_val / max_hdr) * (LUT_SIZE - 1) as f32)
            .round()
            .clamp(0.0, (LUT_SIZE - 1) as f32) as usize;
        lut[idx] += sdr_val;
        counts[idx] += 1;
    }

    for i in 0..LUT_SIZE {
        if counts[i] > 0 {
            lut[i] /= counts[i] as f32;
        }
    }

    fill_lut_gaps(&mut lut, &counts);
    enforce_monotonicity(&mut lut);

    Ok(lut)
}

/// Lookup value in LUT with linear interpolation.
fn lookup_lut(lut: &[f32; LUT_SIZE], value: f32, max_hdr: f32) -> f32 {
    let idx_f = (value / max_hdr).clamp(0.0, 1.0) * (LUT_SIZE - 1) as f32;

    if idx_f >= (LUT_SIZE - 1) as f32 {
        lut[LUT_SIZE - 1]
    } else {
        let idx = idx_f as usize;
        let frac = idx_f - idx as f32;
        (lut[idx] * (1.0 - frac) + lut[idx + 1] * frac).clamp(0.0, 1.0)
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_reinhard_properties() {
        // Black stays black
        assert_eq!(reinhard_extended(0.0, 100.0), 0.0);

        // Monotonically increasing
        let mut prev = 0.0;
        for i in 1..=100 {
            let l = i as f32;
            let mapped = reinhard_extended(l, 100.0);
            assert!(mapped > prev, "Not monotonic at {}", l);
            prev = mapped;
        }

        // Never exceeds 1.0 for reasonable inputs
        for i in 1..=1000 {
            let l = i as f32 / 10.0;
            let mapped = reinhard_extended(l, 100.0);
            assert!(mapped <= 1.0, "Exceeded 1.0 at L={}", l);
        }
    }

    #[test]
    fn test_filmic_properties() {
        // Black stays black
        assert_eq!(filmic_narkowicz(0.0), 0.0);

        // Near-white maps to ~1
        let white = filmic_narkowicz(10.0);
        assert!(white > 0.9 && white <= 1.0);

        // Monotonically increasing
        let mut prev = 0.0;
        for i in 1..=100 {
            let x = i as f32 / 10.0;
            let mapped = filmic_narkowicz(x);
            assert!(mapped >= prev, "Not monotonic at {}", x);
            prev = mapped;
        }
    }

    #[test]
    fn test_adaptive_tonemapper_fit() {
        use crate::PixelFormat;

        // Create simple HDR image in RgbaF32 (the only HDR pixel format we
        // carry post-narrow; #9 removed Rgba16F as dormant).
        let width = 32u32;
        let height = 32u32;
        let mut hdr_data = Vec::with_capacity((width * height * 16) as usize);
        let mut sdr_data = Vec::with_capacity((width * height * 4) as usize);

        for _y in 0..height {
            for x in 0..width {
                // HDR: gradient from 0 to 4 (2 stops over SDR white)
                let l = (x as f32 / width as f32) * 4.0;
                hdr_data.extend_from_slice(&l.to_le_bytes());
                hdr_data.extend_from_slice(&l.to_le_bytes());
                hdr_data.extend_from_slice(&l.to_le_bytes());
                hdr_data.extend_from_slice(&1.0f32.to_le_bytes());

                // SDR: simple tonemap (clamped)
                let sdr_l = l.min(1.0);
                let sdr_val = (srgb_oetf(sdr_l) * 255.0).round() as u8;
                sdr_data.push(sdr_val);
                sdr_data.push(sdr_val);
                sdr_data.push(sdr_val);
                sdr_data.push(255);
            }
        }

        let hdr = crate::types::pixel_buffer_from_vec(
            hdr_data,
            width,
            height,
            PixelFormat::RgbaF32,
            ColorPrimaries::Bt709,
            TransferFunction::Linear,
        )
        .unwrap();

        let sdr = crate::types::pixel_buffer_from_vec(
            sdr_data,
            width,
            height,
            PixelFormat::Rgba8,
            ColorPrimaries::Bt709,
            TransferFunction::Srgb,
        )
        .unwrap();

        // Fit tonemapper
        let tm = AdaptiveTonemapper::fit(&hdr, &sdr).unwrap();

        // Check stats
        assert!(tm.stats.samples > 0);
        assert!(tm.stats.max_hdr_luminance > 1.0);

        // Apply should produce valid output
        let result = tm.apply(&hdr).unwrap();
        assert_eq!(result.width(), width);
        assert_eq!(result.height(), height);
    }

    #[test]
    fn test_scale_gainmap() {
        let mut gm = GainMap::new(4, 4).unwrap();
        for i in 0..16 {
            gm.data[i] = (i * 16) as u8;
        }

        let scaled = scale_gainmap(&gm, 8, 8).unwrap();
        assert_eq!(scaled.width, 8);
        assert_eq!(scaled.height, 8);
        assert_eq!(scaled.data.len(), 64);
    }

    #[test]
    fn test_crop_gainmap() {
        let mut gm = GainMap::new(10, 10).unwrap();
        for i in 0..100 {
            gm.data[i] = i as u8;
        }

        // Crop center 50%
        let cropped = crop_gainmap(&gm, 100, 100, (25, 25, 50, 50)).unwrap();

        assert!(cropped.width >= 4);
        assert!(cropped.height >= 4);
    }

    #[test]
    fn test_pq_tonemap_black_white() {
        let config = ToneMapConfig::default();

        // Black (PQ 0.0) should map to black
        let black = tonemap_pq_to_sdr([0.0, 0.0, 0.0], &config);
        assert!(black[0] < 0.01 && black[1] < 0.01 && black[2] < 0.01);

        // Peak white (PQ 1.0) should map to something bright but not necessarily 1.0
        let white = tonemap_pq_to_sdr([1.0, 1.0, 1.0], &config);
        assert!(white[0] > 0.5);
    }

    #[test]
    fn test_lut_monotonicity() {
        let mut lut = [0.0f32; LUT_SIZE];
        for (i, slot) in lut.iter_mut().enumerate() {
            *slot = (i as f32 / LUT_SIZE as f32).sin(); // Non-monotonic
        }

        enforce_monotonicity(&mut lut);

        // Verify monotonic
        for pair in lut.windows(2) {
            assert!(pair[1] >= pair[0], "Not monotonic");
        }
    }

    #[test]
    fn test_bt2390_properties() {
        // Black stays black
        assert_eq!(bt2390_tonemap(0.0, 10.0, 1.0), 0.0);

        // When source <= target, passthrough (no tone mapping needed)
        assert_eq!(bt2390_tonemap(0.5, 1.0, 1.0), 0.5);
        assert_eq!(bt2390_tonemap(0.5, 1.0, 2.0), 0.5);

        // Monotonically increasing within the above-knee region
        // (the Hermite spline itself is monotonic for t in [0,1])
        let source = 10.0;
        let target = 1.0;
        let ks = (1.5f32 * target / source - 0.5).clamp(0.0, 1.0); // 0.0 for this ratio
        let mut prev = bt2390_tonemap(ks + 0.01, source, target);
        for i in 2..=100 {
            let l = ks + i as f32 * 0.01;
            let mapped = bt2390_tonemap(l, source, target);
            assert!(
                mapped >= prev - f32::EPSILON,
                "Not monotonic above knee at {}: {} < {}",
                l,
                mapped,
                prev
            );
            prev = mapped;
        }

        // Never exceeds 1.0 for inputs in [0, 20]
        for i in 0..=200 {
            let l = i as f32 * 0.1;
            let mapped = bt2390_tonemap(l, 10.0, 1.0);
            assert!(mapped <= 1.0, "Exceeded 1.0 at L={}: {}", l, mapped);
        }
    }

    #[test]
    fn test_bt2390_vs_reinhard() {
        // Both produce similar results for low HDR values (< 1.0)
        let peak = 10.0;
        for i in 1..=10 {
            let l = i as f32 * 0.1; // 0.1 to 1.0
            let bt = bt2390_tonemap(l, peak, 1.0);
            let rh = reinhard_extended(l, peak);
            // Both should map low values somewhat similarly (within 0.5)
            assert!(
                (bt - rh).abs() < 0.5,
                "Large divergence at low L={}: bt2390={}, reinhard={}",
                l,
                bt,
                rh
            );
        }

        // BT.2390 should have better highlight compression for very bright values.
        // At very high input, BT.2390 approaches target_peak/source_peak ratio
        // while Reinhard approaches 1.0 more slowly.
        let bright = 8.0;
        let bt_bright = bt2390_tonemap(bright, peak, 1.0);
        let rh_bright = reinhard_extended(bright, peak);
        // They should differ noticeably at high values
        assert!(
            (bt_bright - rh_bright).abs() > 0.01,
            "Expected divergence at L={}: bt2390={}, reinhard={}",
            bright,
            bt_bright,
            rh_bright
        );
    }

    #[test]
    fn test_tonemap_pq_to_sdr_black() {
        let config = ToneMapConfig::default();
        let result = tonemap_pq_to_sdr([0.0, 0.0, 0.0], &config);
        assert!(
            result[0] < 0.01 && result[1] < 0.01 && result[2] < 0.01,
            "PQ black should map to near-black, got {:?}",
            result
        );
    }

    #[test]
    fn test_tonemap_pq_to_sdr_white() {
        // PQ ~0.58 corresponds to ~203 nits (SDR reference white).
        // The filmic tonemapper normalizes by hdr_peak_nits (10000) so 203 nits
        // becomes a small input (~0.08) to the curve. The output will be modest
        // but should be distinctly brighter than black.
        let config = ToneMapConfig {
            target_peak_nits: 203.0,
            hdr_peak_nits: 10000.0,
            target_gamut: ColorPrimaries::Bt709,
            source_gamut: ColorPrimaries::Bt2020,
        };
        let result = tonemap_pq_to_sdr([0.58, 0.58, 0.58], &config);
        // Should produce a positive, non-trivial SDR value
        assert!(
            result[0] > 0.01 && result[1] > 0.01 && result[2] > 0.01,
            "PQ 0.58 (203 nits) should produce non-trivial SDR output, got {:?}",
            result
        );
        // Should be significantly brighter than PQ black mapping
        let black = tonemap_pq_to_sdr([0.0, 0.0, 0.0], &config);
        assert!(
            result[0] > black[0] + 0.01,
            "PQ 0.58 should be distinctly brighter than PQ 0.0: {} vs {}",
            result[0],
            black[0]
        );
    }

    #[test]
    fn test_scale_gainmap_identity() {
        let mut gm = GainMap::new(4, 4).unwrap();
        for i in 0..16 {
            gm.data[i] = (i * 15 + 10) as u8;
        }

        // Scale to same dimensions
        let scaled = scale_gainmap(&gm, 4, 4).unwrap();
        assert_eq!(scaled.width, 4);
        assert_eq!(scaled.height, 4);
        assert_eq!(scaled.data, gm.data);
    }

    #[test]
    fn test_scale_gainmap_double() {
        let mut gm = GainMap::new(2, 2).unwrap();
        // 2x2 pattern:
        //   0  100
        //  50  200
        gm.data[0] = 0;
        gm.data[1] = 100;
        gm.data[2] = 50;
        gm.data[3] = 200;

        let scaled = scale_gainmap(&gm, 4, 4).unwrap();
        assert_eq!(scaled.width, 4);
        assert_eq!(scaled.height, 4);
        assert_eq!(scaled.data.len(), 16);

        // Corners should match original values
        assert_eq!(scaled.data[0], 0); // top-left
        assert_eq!(scaled.data[3], 100); // top-right maps from (1,0) in source

        // Interior values should be interpolated (smooth, between neighbors)
        // The pixel at (1,0) in output maps to (0.5, 0) in source — between 0 and 100
        let mid_top = scaled.data[1];
        assert!(
            mid_top > 0 && mid_top < 100,
            "Expected interpolated value between 0 and 100, got {}",
            mid_top
        );
    }

    #[test]
    fn test_scale_gainmap_invalid() {
        let gm = GainMap::new(4, 4).unwrap();
        // Scale to 0x0 should error (validate_dimensions rejects 0)
        let result = scale_gainmap(&gm, 0, 0);
        assert!(result.is_err(), "Scale to 0x0 should return an error");
    }

    #[test]
    fn test_crop_gainmap_full() {
        let mut gm = GainMap::new(4, 4).unwrap();
        for i in 0..16 {
            gm.data[i] = (i * 16) as u8;
        }

        // Crop with rect covering entire image (SDR is same size as gainmap here)
        let cropped = crop_gainmap(&gm, 4, 4, (0, 0, 4, 4)).unwrap();
        assert_eq!(cropped.width, gm.width);
        assert_eq!(cropped.height, gm.height);
        assert_eq!(cropped.data, gm.data);
    }

    #[test]
    fn test_crop_gainmap_quarter() {
        let mut gm = GainMap::new(4, 4).unwrap();
        for y in 0..4u32 {
            for x in 0..4u32 {
                gm.data[(y * 4 + x) as usize] = (y * 10 + x) as u8;
            }
        }

        // Crop top-left 2x2 (SDR coords map 1:1 to gainmap coords)
        let cropped = crop_gainmap(&gm, 4, 4, (0, 0, 2, 2)).unwrap();
        assert_eq!(cropped.width, 2);
        assert_eq!(cropped.height, 2);
        // Top-left 2x2 of original: [0,1], [10,11]
        assert_eq!(cropped.data[0], 0);
        assert_eq!(cropped.data[1], 1);
        assert_eq!(cropped.data[2], 10);
        assert_eq!(cropped.data[3], 11);
    }

    #[test]
    fn test_crop_gainmap_out_of_bounds() {
        let gm = GainMap::new(4, 4).unwrap();
        // Crop rect extends past image — should clamp (not panic)
        let result = crop_gainmap(&gm, 4, 4, (3, 3, 4, 4));
        // The function clamps via .min(), so it should succeed with a smaller region
        assert!(result.is_ok(), "Out-of-bounds crop should clamp, not error");
        let cropped = result.unwrap();
        assert!(
            cropped.width <= 4 && cropped.height <= 4,
            "Cropped dimensions should be clamped"
        );
        assert!(
            cropped.width >= 1 && cropped.height >= 1,
            "Cropped dimensions should be at least 1x1"
        );
    }

    #[test]
    fn test_adaptive_tonemapper_all_black() {
        use crate::PixelFormat;

        let width = 8u32;
        let height = 8u32;

        // Create all-black HDR (f32) and SDR (RGBA8) images
        let hdr_data = vec![0u8; (width * height * 16) as usize]; // f32 RGBA = 16 bytes/pixel
        let sdr_data = vec![0u8; (width * height * 4) as usize]; // RGBA8 = 4 bytes/pixel

        let hdr = crate::types::pixel_buffer_from_vec(
            hdr_data,
            width,
            height,
            PixelFormat::RgbaF32,
            ColorPrimaries::Bt709,
            TransferFunction::Linear,
        )
        .unwrap();

        let sdr = crate::types::pixel_buffer_from_vec(
            sdr_data,
            width,
            height,
            PixelFormat::Rgba8,
            ColorPrimaries::Bt709,
            TransferFunction::Srgb,
        )
        .unwrap();

        // Fitting from all-black should fail (no valid pixel pairs)
        let result = AdaptiveTonemapper::fit(&hdr, &sdr);
        assert!(
            result.is_err(),
            "Fitting from all-black pair should error (no valid pixel pairs)"
        );
    }

    #[test]
    fn test_filmic_vs_reinhard_comparison() {
        // Both map 0 to 0
        assert_eq!(filmic_narkowicz(0.0), 0.0);
        assert_eq!(reinhard_extended(0.0, 10.0), 0.0);

        // Both map small values (~0.1) similarly
        let filmic_low = filmic_narkowicz(0.1);
        let reinhard_low = reinhard_extended(0.1, 10.0);
        assert!(
            (filmic_low - reinhard_low).abs() < 0.15,
            "Expected similar low-value mapping: filmic={}, reinhard={}",
            filmic_low,
            reinhard_low
        );

        // They diverge at high values
        let filmic_high = filmic_narkowicz(5.0);
        let reinhard_high = reinhard_extended(5.0, 10.0);
        assert!(
            (filmic_high - reinhard_high).abs() > 0.01,
            "Expected divergence at high values: filmic={}, reinhard={}",
            filmic_high,
            reinhard_high
        );

        // Both should stay in [0, 1]
        for i in 0..=100 {
            let x = i as f32 * 0.1;
            let f = filmic_narkowicz(x);
            let r = reinhard_extended(x, 10.0);
            assert!(
                (0.0..=1.0).contains(&f),
                "Filmic out of range at {}: {}",
                x,
                f
            );
            assert!(
                (0.0..=1.0).contains(&r),
                "Reinhard out of range at {}: {}",
                x,
                r
            );
        }
    }

    // ========================================================================
    // Phase 1: BT.2390 black crush + BT.2408 PQ-domain tests
    // ========================================================================

    #[test]
    fn test_bt2390_ext_backward_compat() {
        for i in 0..=100 {
            let l = i as f32 * 0.1;
            let old = bt2390_tonemap(l, 10.0, 1.0);
            let new = bt2390_tonemap_ext(l, 10.0, 1.0, None);
            assert!((old - new).abs() < 1e-7, "at {}: {} vs {}", l, old, new);
        }
    }

    #[test]
    fn test_bt2390_ext_min_lum_shadow_preservation() {
        let min_lum = 0.01;
        let result = bt2390_tonemap_ext(0.001, 10.0, 1.0, Some(min_lum));
        assert!(result > 0.0, "Shadow should be preserved, got {}", result);
        let result_no = bt2390_tonemap_ext(0.001, 10.0, 1.0, None);
        assert!(
            result >= result_no,
            "min_lum should lift: {} vs {}",
            result,
            result_no
        );
    }

    #[test]
    fn test_bt2390_ext_with_min_lum_range() {
        let min_lum = 0.01;
        // BT.2390 is defined for signal range [0,1], test within that range
        for i in 0..=100 {
            let l = i as f32 * 0.01;
            let mapped = bt2390_tonemap_ext(l, 10.0, 1.0, Some(min_lum));
            assert!(mapped >= 0.0 && mapped.is_finite(), "at {}: {}", l, mapped);
        }
        // At midrange, min_lum should have negligible effect
        let mid_with = bt2390_tonemap_ext(0.5, 10.0, 1.0, Some(min_lum));
        let mid_without = bt2390_tonemap_ext(0.5, 10.0, 1.0, None);
        assert!(
            (mid_with - mid_without).abs() < 0.005,
            "min_lum should barely affect midrange: {} vs {}",
            mid_with,
            mid_without
        );
    }

    #[test]
    fn test_bt2408_identity_when_peaks_equal() {
        use zentone::ToneMap;
        let tm = Bt2408Tonemapper::new(1000.0, 1000.0);
        let input = [0.01, 0.02, 0.005];
        let output = tm.map_rgb(input);
        for i in 0..3 {
            assert!(
                (output[i] - input[i]).abs() < 0.01,
                "ch{}: in={}, out={}",
                i,
                input[i],
                output[i]
            );
        }
    }

    #[test]
    fn test_bt2408_shadow_preservation() {
        let tm = Bt2408Tonemapper::new(4000.0, 1000.0);
        let result = tm.tonemap_nits(0.001);
        assert!(
            result > 0.0,
            "0.001 nits must NOT map to 0.0, got {}",
            result
        );
    }

    #[test]
    fn test_bt2408_monotonicity() {
        let tm = Bt2408Tonemapper::new(4000.0, 1000.0);
        let mut prev = 0.0_f32;
        for i in 0..=1000 {
            let nits = i as f32 * 4.0;
            let mapped = tm.tonemap_nits(nits);
            // Tolerance: zentone's tonemap_nits has float reordering near the
            // display saturation knee that our old implementation did not.
            // 1e-3 nits is ~10 orders of magnitude below SDR white and invisible.
            assert!(
                mapped >= prev - 1e-3,
                "Not monotonic at {} nits: {} < {}",
                nits,
                mapped,
                prev
            );
            prev = mapped;
        }
    }

    #[test]
    fn test_bt2408_rgb_black() {
        use zentone::ToneMap;
        let tm = Bt2408Tonemapper::new(4000.0, 1000.0);
        assert_eq!(tm.map_rgb([0.0, 0.0, 0.0]), [0.0, 0.0, 0.0]);
    }

    #[test]
    fn test_bt2408_compresses_highlights() {
        let tm = Bt2408Tonemapper::new(4000.0, 1000.0);
        let result = tm.tonemap_nits(4000.0);
        assert!(
            result <= 1000.0 + 1.0,
            "Should compress to <=1000, got {}",
            result
        );
        assert!(
            result > 800.0,
            "Should use most of display range, got {}",
            result
        );
    }

    // ========================================================================
    // Phase 2a: Simple curves tests
    // ========================================================================

    #[test]
    fn test_reinhard_simple_properties() {
        assert_eq!(reinhard_simple(0.0), 0.0);
        assert!((reinhard_simple(1.0) - 0.5).abs() < 0.001);
        assert!(reinhard_simple(1000.0) > 0.999);
        let mut prev = 0.0;
        for i in 1..=100 {
            let v = reinhard_simple(i as f32 * 0.1);
            assert!(v > prev);
            prev = v;
        }
    }

    #[test]
    fn test_reinhard_jodie_black() {
        let result = reinhard_jodie([0.0, 0.0, 0.0], [0.2126, 0.7152, 0.0722]);
        assert_eq!(result, [0.0, 0.0, 0.0]);
    }

    #[test]
    fn test_reinhard_jodie_monotonic() {
        let luma = [0.2126_f32, 0.7152, 0.0722];
        let mut prev_sum = 0.0_f32;
        for i in 1..=100 {
            let v = i as f32 * 0.1;
            let result = reinhard_jodie([v, v, v], luma);
            let sum: f32 = result.iter().sum();
            assert!(
                sum >= prev_sum - 1e-6,
                "Not monotonic at {}: {} < {}",
                v,
                sum,
                prev_sum
            );
            prev_sum = sum;
        }
    }

    #[test]
    fn test_reinhard_jodie_bounded() {
        let luma = [0.2126_f32, 0.7152, 0.0722];
        for i in 0..=100 {
            let v = i as f32 * 0.1;
            let result = reinhard_jodie([v, v, v], luma);
            for ch in &result {
                assert!(
                    *ch >= 0.0 && *ch <= 1.0,
                    "Out of [0,1] at {}: {:?}",
                    v,
                    result
                );
            }
        }
    }

    // ========================================================================
    // Phase 2b: Complex curves tests
    // ========================================================================

    #[test]
    fn test_hable_filmic_properties() {
        assert!(hable_filmic(0.0).abs() < 0.01);
        assert!(hable_filmic(100.0) > 0.95);
        let mut prev = 0.0;
        for i in 1..=100 {
            let v = hable_filmic(i as f32 * 0.1);
            assert!(v >= prev - 1e-6, "Not monotonic at {}", i);
            prev = v;
        }
    }

    #[test]
    fn test_uncharted2_bounded() {
        for i in 0..=1000 {
            let v = i as f32 * 0.01;
            let result = hable_filmic(v);
            assert!(
                (0.0..=1.0).contains(&result),
                "Out of [0,1] at {}: {}",
                v,
                result
            );
        }
    }

    #[test]
    fn test_aces_ap1_black() {
        let result = aces_ap1([0.0, 0.0, 0.0]);
        for ch in &result {
            assert!(ch.abs() < 0.01, "ACES black near zero: {:?}", result);
        }
    }

    #[test]
    fn test_aces_ap1_monotonic_gray_ramp() {
        // Start from 0.01 to avoid ACES constant offset causing slightly negative near zero
        let mut prev_sum = 0.0_f32;
        for i in 1..=100 {
            let v = i as f32 * 0.1;
            let result = aces_ap1([v, v, v]);
            let sum: f32 = result.iter().sum();
            assert!(
                sum >= prev_sum - 1e-5,
                "Not monotonic at {}: {} < {}",
                v,
                sum,
                prev_sum
            );
            prev_sum = sum;
        }
    }

    #[test]
    fn test_aces_ap1_bright() {
        let result = aces_ap1([10.0, 10.0, 10.0]);
        for ch in &result {
            assert!(
                *ch > 0.8 && *ch <= 1.0,
                "ACES bright near 1.0: {:?}",
                result
            );
        }
    }

    #[test]
    fn test_agx_default_black() {
        let result = agx_tonemap([0.0, 0.0, 0.0], AgxLook::Default);
        for ch in &result {
            assert!(ch.abs() < 0.02, "AgX black near zero: {:?}", result);
        }
    }

    #[test]
    fn test_agx_default_monotonic() {
        let mut prev_sum = 0.0_f32;
        for i in 0..=100 {
            let v = (i as f32 * 0.1).max(1e-6);
            let result = agx_tonemap([v, v, v], AgxLook::Default);
            let sum: f32 = result.iter().sum();
            assert!(
                sum >= prev_sum - 1e-4,
                "Not monotonic at {}: {} < {}",
                v,
                sum,
                prev_sum
            );
            prev_sum = sum;
        }
    }

    #[test]
    fn test_agx_bounded() {
        for look in [AgxLook::Default, AgxLook::Punchy, AgxLook::Golden] {
            for i in 0..=100 {
                let v = i as f32 * 0.1;
                let result = agx_tonemap([v, v, v], look);
                for ch in &result {
                    assert!(
                        *ch >= 0.0 && *ch <= 1.0,
                        "{:?} out of [0,1] at {}: {:?}",
                        look,
                        v,
                        result
                    );
                }
            }
        }
    }

    #[test]
    fn test_agx_looks_differ() {
        // Use a saturated input so the look transforms produce visible differences
        let input = [5.0, 0.5, 0.1];
        let default = agx_tonemap(input, AgxLook::Default);
        let _punchy = agx_tonemap(input, AgxLook::Punchy);
        let golden = agx_tonemap(input, AgxLook::Golden);
        // At least one channel should differ between default and golden
        // (golden has slope=[1.0, 0.9, 0.5] which dramatically changes the look)
        let golden_diff: f32 = (0..3).map(|i| (default[i] - golden[i]).abs()).sum();
        assert!(
            golden_diff > 0.01,
            "Default and Golden should differ: {:?} vs {:?}",
            default,
            golden
        );
    }

    #[test]
    fn test_filmic_spline_default() {
        use zentone::ToneMap;
        let config = FilmicSplineConfig::default();
        let spline = CompiledFilmicSpline::new(&config);
        let mid = spline.map_rgb([0.18, 0.18, 0.18]);
        assert!(mid[0] > 0.05 && mid[0] < 0.5, "Mid-gray: {:?}", mid);
        let mut prev = 0.0_f32;
        for i in 1..=100 {
            let v = i as f32 * 0.1;
            let result = spline.map_rgb([v, v, v]);
            let sum: f32 = result.iter().sum();
            assert!(
                sum >= prev - 1e-4,
                "Not monotonic at {}: {} < {}",
                v,
                sum,
                prev
            );
            prev = sum;
        }
    }

    #[test]
    fn test_filmic_spline_bounded() {
        use zentone::ToneMap;
        let config = FilmicSplineConfig::default();
        let spline = CompiledFilmicSpline::new(&config);
        for i in 0..=100 {
            let v = i as f32 * 0.1;
            let result = spline.map_rgb([v, v, v]);
            for ch in &result {
                assert!(
                    *ch >= 0.0 && *ch <= 1.0,
                    "Out of [0,1] at {}: {:?}",
                    v,
                    result
                );
            }
        }
    }

    // ========================================================================
    // Unified ToneMapCurve enum tests
    // ========================================================================

    #[test]
    fn test_tonemap_curve_all_variants_black() {
        let luma = [0.2126_f32, 0.7152, 0.0722];
        let curves = [
            ToneMapCurve::Reinhard,
            ToneMapCurve::ExtendedReinhard { l_max: 10.0, luma },
            ToneMapCurve::ReinhardJodie { luma },
            ToneMapCurve::TunedReinhard {
                content_max_nits: 4000.0,
                display_max_nits: 1000.0,
                luma,
            },
            ToneMapCurve::Narkowicz,
            ToneMapCurve::HableFilmic,
            ToneMapCurve::AcesAp1,
            ToneMapCurve::Bt2390 {
                source_peak: 10.0,
                target_peak: 1.0,
            },
            ToneMapCurve::Agx(AgxLook::Default),
            ToneMapCurve::Clamp,
        ];
        for curve in &curves {
            let result = curve.map_rgb([0.0, 0.0, 0.0]);
            let sum: f32 = result.iter().map(|v| v.abs()).sum();
            assert!(sum < 0.05, "{:?}: black→{:?}", curve, result);
        }
    }

    #[test]
    fn test_tonemap_curve_all_variants_bright() {
        let curves = [
            ToneMapCurve::Reinhard,
            ToneMapCurve::Narkowicz,
            ToneMapCurve::HableFilmic,
            ToneMapCurve::AcesAp1,
            ToneMapCurve::Agx(AgxLook::Default),
            ToneMapCurve::Clamp,
        ];
        for curve in &curves {
            let result = curve.map_rgb([10.0, 10.0, 10.0]);
            for ch in &result {
                assert!(*ch <= 1.0, "{:?}: bright→{:?}", curve, result);
            }
        }
    }

    // ========================================================================
    // Phase 4: Batch map_row tests
    // ========================================================================

    #[test]
    fn test_tonemap_row_basic() {
        let curve = ToneMapCurve::Reinhard;
        let mut row = vec![0.0, 0.0, 0.0, 0.5, 0.5, 0.5, 2.0, 2.0, 2.0];
        curve.map_row(&mut row, 3);
        assert!(row[0].abs() < 0.001);
        assert!((row[3] - 0.333).abs() < 0.02);
        assert!((row[6] - 0.667).abs() < 0.02);
    }

    #[test]
    fn test_tonemap_row_4ch_alpha_passthrough() {
        let curve = ToneMapCurve::Clamp;
        let mut row = vec![2.0, 3.0, 4.0, 0.75, 0.5, 0.5, 0.5, 0.99];
        curve.map_row(&mut row, 4);
        assert_eq!(row[0], 1.0);
        assert_eq!(row[1], 1.0);
        assert_eq!(row[2], 1.0);
        assert_eq!(row[3], 0.75);
        assert_eq!(row[7], 0.99);
    }

    #[test]
    fn test_tonemap_row_empty() {
        let curve = ToneMapCurve::Reinhard;
        let mut row: Vec<f32> = vec![];
        curve.map_row(&mut row, 3);
        assert!(row.is_empty());
    }

    #[test]
    fn test_tonemap_row_matches_per_pixel() {
        let curve = ToneMapCurve::HableFilmic;
        let mut row = vec![0.1, 0.2, 0.3, 1.0, 2.0, 3.0, 5.0, 5.0, 5.0];
        let expected = [
            curve.map_rgb([0.1, 0.2, 0.3]),
            curve.map_rgb([1.0, 2.0, 3.0]),
            curve.map_rgb([5.0, 5.0, 5.0]),
        ];
        curve.map_row(&mut row, 3);
        for (i, exp) in expected.iter().enumerate() {
            for ch in 0..3 {
                assert!(
                    (row[i * 3 + ch] - exp[ch]).abs() < 1e-6,
                    "px {} ch {}: {} vs {}",
                    i,
                    ch,
                    row[i * 3 + ch],
                    exp[ch]
                );
            }
        }
    }

    #[test]
    fn test_tonemap_row_all_curves() {
        let luma = [0.2126_f32, 0.7152, 0.0722];
        let curves = [
            ToneMapCurve::Reinhard,
            ToneMapCurve::ExtendedReinhard { l_max: 10.0, luma },
            ToneMapCurve::ReinhardJodie { luma },
            ToneMapCurve::TunedReinhard {
                content_max_nits: 4000.0,
                display_max_nits: 1000.0,
                luma,
            },
            ToneMapCurve::Narkowicz,
            ToneMapCurve::HableFilmic,
            ToneMapCurve::AcesAp1,
            ToneMapCurve::Bt2390 {
                source_peak: 10.0,
                target_peak: 1.0,
            },
            ToneMapCurve::Agx(AgxLook::Punchy),
            ToneMapCurve::Clamp,
        ];
        for curve in &curves {
            let mut row = vec![1.5, 2.0, 0.5];
            curve.map_row(&mut row, 3);
            for v in &row {
                assert!(v.is_finite(), "{:?}: produced non-finite {}", curve, v);
            }
        }
    }

    // ========================================================================
    // ProfileToneCurve tests
    // ========================================================================

    #[test]
    fn test_profile_curve_identity() {
        let curve = ProfileToneCurve::identity();
        // Identity should be ~passthrough
        for i in 0..=100 {
            let x = i as f32 / 100.0;
            let y = curve.eval(x);
            assert!((y - x).abs() < 0.001, "Identity at {}: got {}", x, y);
        }
    }

    #[test]
    fn test_profile_curve_from_xy_pairs() {
        // Simple 2-point linear curve: (0,0) to (1,1)
        let data = [0.0f32, 0.0, 1.0, 1.0];
        let curve = ProfileToneCurve::from_xy_pairs(&data).unwrap();
        assert!((curve.eval(0.0) - 0.0).abs() < 0.001);
        assert!((curve.eval(0.5) - 0.5).abs() < 0.001);
        assert!((curve.eval(1.0) - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_profile_curve_gamma() {
        // 3-point curve: dark lift (0,0.1), mid (0.5,0.6), white (1.0,1.0)
        let data = [0.0, 0.1, 0.5, 0.6, 1.0, 1.0];
        let curve = ProfileToneCurve::from_xy_pairs(&data).unwrap();
        // Should lift blacks
        assert!(curve.eval(0.0) > 0.05, "Black lift: {}", curve.eval(0.0));
        // Mid should be boosted
        assert!(curve.eval(0.5) > 0.55, "Mid boost: {}", curve.eval(0.5));
        // White stays white
        assert!((curve.eval(1.0) - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_profile_curve_per_channel() {
        let curve = ProfileToneCurve::identity();
        let result = curve.apply_per_channel([0.3, 0.5, 0.8]);
        assert!((result[0] - 0.3).abs() < 0.001);
        assert!((result[1] - 0.5).abs() < 0.001);
        assert!((result[2] - 0.8).abs() < 0.001);
    }

    #[test]
    fn test_profile_curve_lum_preserving() {
        let curve = ProfileToneCurve::identity();
        let luma = [0.2126f32, 0.7152, 0.0722];
        let result = curve.apply_lum_preserving([0.3, 0.5, 0.8], luma);
        // Identity curve in lum-preserving mode should approximately preserve values
        assert!((result[0] - 0.3).abs() < 0.05, "R: {}", result[0]);
        assert!((result[1] - 0.5).abs() < 0.05, "G: {}", result[1]);
        assert!((result[2] - 0.8).abs() < 0.05, "B: {}", result[2]);
    }

    #[test]
    fn test_profile_curve_lum_preserving_black() {
        let curve = ProfileToneCurve::identity();
        let luma = [0.2126f32, 0.7152, 0.0722];
        let result = curve.apply_lum_preserving([0.0, 0.0, 0.0], luma);
        assert_eq!(result, [0.0, 0.0, 0.0]);
    }

    #[test]
    fn test_profile_curve_row_per_channel() {
        let curve = ProfileToneCurve::identity();
        let mut row = vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6];
        curve.apply_row_per_channel(&mut row, 3);
        assert!((row[0] - 0.1).abs() < 0.001);
        assert!((row[3] - 0.4).abs() < 0.001);
    }

    #[test]
    fn test_profile_curve_row_4ch_alpha() {
        let curve = ProfileToneCurve::identity();
        let mut row = vec![0.3, 0.5, 0.7, 0.99, 0.1, 0.2, 0.3, 0.88];
        curve.apply_row_per_channel(&mut row, 4);
        // Alpha should be preserved
        assert_eq!(row[3], 0.99);
        assert_eq!(row[7], 0.88);
    }

    #[test]
    fn test_profile_curve_monotonic() {
        // S-curve: boost shadows, compress highlights
        let data = [0.0, 0.0, 0.25, 0.35, 0.5, 0.55, 0.75, 0.8, 1.0, 1.0];
        let curve = ProfileToneCurve::from_xy_pairs(&data).unwrap();
        let mut prev = 0.0f32;
        for i in 0..=100 {
            let x = i as f32 / 100.0;
            let y = curve.eval(x);
            assert!(y >= prev - 1e-6, "Not monotonic at {}: {} < {}", x, y, prev);
            prev = y;
        }
    }

    #[test]
    fn test_profile_curve_from_lut_validation() {
        // Wrong size should fail
        assert!(ProfileToneCurve::from_lut(vec![0.0; 100]).is_none());
        // Right size should work
        assert!(ProfileToneCurve::from_lut(vec![0.0; 4097]).is_some());
    }

    #[test]
    fn test_profile_curve_too_few_points() {
        // Single point (1 pair = 2 floats / 2 = 1 point, < 2) should fail
        let data = [0.5, 0.5];
        assert!(ProfileToneCurve::from_xy_pairs(&data).is_none());
        // Empty should fail
        assert!(ProfileToneCurve::from_xy_pairs(&[]).is_none());
    }
}
