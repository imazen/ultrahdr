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
//! the original artistic intent:
//!
//! ```ignore
//! use ultrahdr_core::color::tonemap::AdaptiveTonemapper;
//!
//! // Fit from original HDR/SDR pair
//! let tonemapper = AdaptiveTonemapper::fit(&hdr_original, &sdr_original)?;
//!
//! // Apply to edited HDR (preserves original "look")
//! let sdr_new = tonemapper.apply(&hdr_edited)?;
//! ```

use crate::color::gamut::{convert_gamut, rgb_to_luminance, soft_clip_gamut};
use crate::color::transfer::{hlg_eotf, pq_eotf, srgb_eotf, srgb_oetf};
use crate::types::{ColorGamut, ColorTransfer, Error, GainMap, GainMapMetadata, Result};
use crate::RawImage;

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
    pub target_gamut: ColorGamut,
    /// Source color gamut of HDR content.
    pub source_gamut: ColorGamut,
}

impl Default for ToneMapConfig {
    fn default() -> Self {
        Self {
            target_peak_nits: 203.0, // SDR reference white
            hdr_peak_nits: 10000.0,  // PQ peak
            target_gamut: ColorGamut::Bt709,
            source_gamut: ColorGamut::Bt2100,
        }
    }
}

// ============================================================================
// Standard Tone Mapping Curves
// ============================================================================

/// Simple Reinhard tone mapping operator.
///
/// Maps HDR luminance to SDR range while preserving local contrast.
/// `L_in` is linear luminance (can exceed 1.0 for HDR).
/// `L_max` is the maximum expected luminance.
#[inline]
pub fn reinhard_tonemap(l_in: f32, l_max: f32) -> f32 {
    // Extended Reinhard: L_out = L_in * (1 + L_in/L_max²) / (1 + L_in)
    let l_max_sq = l_max * l_max;
    l_in * (1.0 + l_in / l_max_sq) / (1.0 + l_in)
}

/// ACES-inspired filmic tone mapping curve.
///
/// Attempt to match the ACES RRT + ODT look with a simpler curve.
/// Input and output are both in `[0, ~10]` range (HDR linear).
#[inline]
pub fn filmic_tonemap(x: f32) -> f32 {
    // Simple S-curve approximation
    let a = 2.51;
    let b = 0.03;
    let c = 2.43;
    let d = 0.59;
    let e = 0.14;

    let numerator = x * (a * x + b);
    let denominator = x * (c * x + d) + e;

    (numerator / denominator).clamp(0.0, 1.0)
}

/// BT.2390 EETF (EOTF-based tone mapping) for HLG.
///
/// Maps HLG content to a lower peak luminance display.
/// Based on ITU-R BT.2390 reference EETF.
#[inline]
pub fn bt2390_tonemap(scene_linear: f32, source_peak: f32, target_peak: f32) -> f32 {
    if source_peak <= target_peak {
        return scene_linear;
    }

    // Knee point and slope calculations per BT.2390
    let ks = 1.5 * target_peak / source_peak - 0.5;
    let ks = ks.clamp(0.0, 1.0);

    let e1 = scene_linear;
    if e1 < ks {
        // Below knee: linear pass-through
        e1
    } else {
        // Above knee: soft roll-off
        let t = (e1 - ks) / (1.0 - ks);
        let t2 = t * t;
        let t3 = t2 * t;

        // Hermite spline interpolation
        let p0 = ks;
        let p1 = 1.0;
        let m0 = 1.0 - ks;
        let m1 = 0.0;

        let a = 2.0 * t3 - 3.0 * t2 + 1.0;
        let b = t3 - 2.0 * t2 + t;
        let c = -2.0 * t3 + 3.0 * t2;
        let d = t3 - t2;

        let result = a * p0 + b * m0 + c * p1 + d * m1;
        result * target_peak / source_peak
    }
}

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
    pub fn fit(hdr: &RawImage, sdr: &RawImage) -> Result<Self> {
        Self::fit_with_config(hdr, sdr, &FitConfig::default())
    }

    /// Fit with custom configuration.
    pub fn fit_with_config(hdr: &RawImage, sdr: &RawImage, config: &FitConfig) -> Result<Self> {
        // Validate dimensions match
        if hdr.width != sdr.width || hdr.height != sdr.height {
            return Err(Error::DimensionMismatch {
                hdr_w: hdr.width,
                hdr_h: hdr.height,
                sdr_w: sdr.width,
                sdr_h: sdr.height,
            });
        }

        match config.mode {
            FitMode::Luminance => Self::fit_luminance(hdr, sdr, config),
            FitMode::PerChannel => Self::fit_per_channel(hdr, sdr, config),
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
            max_hdr_observed: metadata.hdr_capacity_max,
            stats: FitStats::default(),
        }
    }

    /// Apply the tonemapper to an HDR image.
    pub fn apply(&self, hdr: &RawImage) -> Result<RawImage> {
        let width = hdr.width;
        let height = hdr.height;

        let mut output = RawImage::new(width, height, crate::PixelFormat::Rgba8)?;
        output.gamut = ColorGamut::Bt709;
        output.transfer = ColorTransfer::Srgb;

        for y in 0..height {
            for x in 0..width {
                let hdr_linear = get_linear_rgb(hdr, x, y);
                let sdr_linear = self.tonemap_pixel(hdr_linear);

                // Apply sRGB OETF and write
                let out_idx = (y * output.stride + x * 4) as usize;
                output.data[out_idx] =
                    (srgb_oetf(sdr_linear[0]) * 255.0).round().clamp(0.0, 255.0) as u8;
                output.data[out_idx + 1] =
                    (srgb_oetf(sdr_linear[1]) * 255.0).round().clamp(0.0, 255.0) as u8;
                output.data[out_idx + 2] =
                    (srgb_oetf(sdr_linear[2]) * 255.0).round().clamp(0.0, 255.0) as u8;
                output.data[out_idx + 3] = 255;
            }
        }

        Ok(output)
    }

    /// Apply tonemapper with gain map for inversion.
    ///
    /// For perfect round-trips when you have the original gain map.
    pub fn apply_with_gainmap(
        &self,
        hdr: &RawImage,
        gainmap: &GainMap,
        metadata: &GainMapMetadata,
    ) -> Result<RawImage> {
        let width = hdr.width;
        let height = hdr.height;

        let mut output = RawImage::new(width, height, crate::PixelFormat::Rgba8)?;
        output.gamut = ColorGamut::Bt709;
        output.transfer = ColorTransfer::Srgb;

        for y in 0..height {
            for x in 0..width {
                let hdr_linear = get_linear_rgb(hdr, x, y);

                // Sample gain map (with interpolation for different resolutions)
                let gain = sample_gainmap_at(gainmap, metadata, x, y, width, height);

                // Invert: SDR = (HDR + offset_hdr) / gain - offset_sdr
                let sdr_linear = [
                    (hdr_linear[0] + metadata.offset_hdr[0]) / gain[0] - metadata.offset_sdr[0],
                    (hdr_linear[1] + metadata.offset_hdr[1]) / gain[1] - metadata.offset_sdr[1],
                    (hdr_linear[2] + metadata.offset_hdr[2]) / gain[2] - metadata.offset_sdr[2],
                ];

                // Clamp and apply sRGB OETF
                let out_idx = (y * output.stride + x * 4) as usize;
                output.data[out_idx] =
                    (srgb_oetf(sdr_linear[0].clamp(0.0, 1.0)) * 255.0).round() as u8;
                output.data[out_idx + 1] =
                    (srgb_oetf(sdr_linear[1].clamp(0.0, 1.0)) * 255.0).round() as u8;
                output.data[out_idx + 2] =
                    (srgb_oetf(sdr_linear[2].clamp(0.0, 1.0)) * 255.0).round() as u8;
                output.data[out_idx + 3] = 255;
            }
        }

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
                let l_sdr = filmic_tonemap(l * 2.0); // Scale for curve
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
    fn fit_luminance(hdr: &RawImage, sdr: &RawImage, config: &FitConfig) -> Result<Self> {
        let width = hdr.width as usize;
        let height = hdr.height as usize;
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
        pairs.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

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
    fn fit_per_channel(hdr: &RawImage, sdr: &RawImage, config: &FitConfig) -> Result<Self> {
        let width = hdr.width as usize;
        let height = hdr.height as usize;
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
                mae: 0.0, // TODO: calculate
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
        let lum_tonemapped = filmic_tonemap(lum_normalized * 4.0); // Scale for curve
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
    transfer: ColorTransfer,
    config: &ToneMapConfig,
) -> [f32; 3] {
    match transfer {
        ColorTransfer::Pq => tonemap_pq_to_sdr(encoded_rgb, config),
        ColorTransfer::Hlg => tonemap_hlg_to_sdr(encoded_rgb, config),
        ColorTransfer::Srgb | ColorTransfer::Linear => {
            // Already SDR, just convert gamut
            let linear = if transfer == ColorTransfer::Srgb {
                [
                    srgb_eotf(encoded_rgb[0]),
                    srgb_eotf(encoded_rgb[1]),
                    srgb_eotf(encoded_rgb[2]),
                ]
            } else {
                encoded_rgb
            };
            convert_gamut(linear, config.source_gamut, config.target_gamut)
        }
    }
}

/// Tone map and encode to 8-bit sRGB.
///
/// Full pipeline: HDR encoded → linear SDR → sRGB encoded → 8-bit
pub fn tonemap_to_srgb8(
    encoded_rgb: [f32; 3],
    transfer: ColorTransfer,
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
pub fn tonemap_image_to_srgb8(img: &RawImage, target_gamut: ColorGamut) -> Vec<u8> {
    use crate::color::gamut::convert_gamut;

    let config = ToneMapConfig::default();
    let width = img.width as usize;
    let height = img.height as usize;
    let mut output = vec![0u8; width * height * 4];

    for y in 0..height {
        for x in 0..width {
            // Extract pixel and convert to linear RGB
            let linear_rgb = get_linear_rgb(img, x as u32, y as u32);

            // Convert gamut if needed
            let gamut_converted = if img.gamut != target_gamut {
                convert_gamut(linear_rgb, img.gamut, target_gamut)
            } else {
                linear_rgb
            };

            // Tonemap
            let sdr = tonemap_to_sdr(gamut_converted, img.transfer, &config);

            // Apply sRGB OETF and quantize
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

    output
}

// ============================================================================
// Helper Functions
// ============================================================================

/// Get linear RGB from any image format.
fn get_linear_rgb(img: &RawImage, x: u32, y: u32) -> [f32; 3] {
    use crate::color::transfer::{hlg_oetf_inv, pq_eotf};
    use crate::PixelFormat;

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
            if img.transfer == ColorTransfer::Srgb {
                [srgb_eotf(r), srgb_eotf(g), srgb_eotf(b)]
            } else {
                [r, g, b]
            }
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
        PixelFormat::Rgba16F => {
            let idx = (y * img.stride + x * 8) as usize;
            let r = half::f16::from_le_bytes([img.data[idx], img.data[idx + 1]]).to_f32();
            let g = half::f16::from_le_bytes([img.data[idx + 2], img.data[idx + 3]]).to_f32();
            let b = half::f16::from_le_bytes([img.data[idx + 4], img.data[idx + 5]]).to_f32();
            [r, g, b]
        }
        PixelFormat::Rgba1010102Pq => {
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
            [pq_eotf(r), pq_eotf(g), pq_eotf(b)]
        }
        PixelFormat::Rgba1010102Hlg => {
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
            [hlg_oetf_inv(r), hlg_oetf_inv(g), hlg_oetf_inv(b)]
        }
        _ => [0.5, 0.5, 0.5],
    }
}

/// Get linear RGB from SDR image (assumes sRGB transfer).
fn get_sdr_linear(sdr: &RawImage, x: u32, y: u32) -> [f32; 3] {
    use crate::PixelFormat;

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
    let gamma = metadata.gamma[channel];
    let linear = if gamma != 1.0 && gamma > 0.0 {
        normalized.powf(1.0 / gamma)
    } else {
        normalized
    };

    let log_min = metadata.min_content_boost[channel].ln();
    let log_max = metadata.max_content_boost[channel].ln();
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

    pairs.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

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
        assert_eq!(reinhard_tonemap(0.0, 100.0), 0.0);

        // Monotonically increasing
        let mut prev = 0.0;
        for i in 1..=100 {
            let l = i as f32;
            let mapped = reinhard_tonemap(l, 100.0);
            assert!(mapped > prev, "Not monotonic at {}", l);
            prev = mapped;
        }

        // Never exceeds 1.0 for reasonable inputs
        for i in 1..=1000 {
            let l = i as f32 / 10.0;
            let mapped = reinhard_tonemap(l, 100.0);
            assert!(mapped <= 1.0, "Exceeded 1.0 at L={}", l);
        }
    }

    #[test]
    fn test_filmic_properties() {
        // Black stays black
        assert_eq!(filmic_tonemap(0.0), 0.0);

        // Near-white maps to ~1
        let white = filmic_tonemap(10.0);
        assert!(white > 0.9 && white <= 1.0);

        // Monotonically increasing
        let mut prev = 0.0;
        for i in 1..=100 {
            let x = i as f32 / 10.0;
            let mapped = filmic_tonemap(x);
            assert!(mapped >= prev, "Not monotonic at {}", x);
            prev = mapped;
        }
    }

    #[test]
    fn test_adaptive_tonemapper_fit() {
        use crate::PixelFormat;

        // Create simple HDR image
        let width = 32u32;
        let height = 32u32;
        let mut hdr_data = Vec::with_capacity((width * height * 8) as usize);
        let mut sdr_data = Vec::with_capacity((width * height * 4) as usize);

        for _y in 0..height {
            for x in 0..width {
                // HDR: gradient from 0 to 4 (2 stops over SDR white)
                let l = (x as f32 / width as f32) * 4.0;
                let hdr_r = half::f16::from_f32(l);
                let hdr_g = half::f16::from_f32(l);
                let hdr_b = half::f16::from_f32(l);
                let hdr_a = half::f16::from_f32(1.0);
                hdr_data.extend_from_slice(&hdr_r.to_le_bytes());
                hdr_data.extend_from_slice(&hdr_g.to_le_bytes());
                hdr_data.extend_from_slice(&hdr_b.to_le_bytes());
                hdr_data.extend_from_slice(&hdr_a.to_le_bytes());

                // SDR: simple tonemap (clamped)
                let sdr_l = l.min(1.0);
                let sdr_val = (srgb_oetf(sdr_l) * 255.0).round() as u8;
                sdr_data.push(sdr_val);
                sdr_data.push(sdr_val);
                sdr_data.push(sdr_val);
                sdr_data.push(255);
            }
        }

        let hdr = RawImage::from_data(
            width,
            height,
            PixelFormat::Rgba16F,
            ColorGamut::Bt709,
            ColorTransfer::Linear,
            hdr_data,
        )
        .unwrap();

        let sdr = RawImage::from_data(
            width,
            height,
            PixelFormat::Rgba8,
            ColorGamut::Bt709,
            ColorTransfer::Srgb,
            sdr_data,
        )
        .unwrap();

        // Fit tonemapper
        let tm = AdaptiveTonemapper::fit(&hdr, &sdr).unwrap();

        // Check stats
        assert!(tm.stats.samples > 0);
        assert!(tm.stats.max_hdr_luminance > 1.0);

        // Apply should produce valid output
        let result = tm.apply(&hdr).unwrap();
        assert_eq!(result.width, width);
        assert_eq!(result.height, height);
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
}
