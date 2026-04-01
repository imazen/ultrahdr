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

use alloc::boxed::Box;
use alloc::vec;
use alloc::vec::Vec;
use core::cmp::Ordering;

use crate::RawImage;
use crate::color::gamut::{convert_gamut, rgb_to_luminance, soft_clip_gamut};
use crate::color::transfer::{hlg_eotf, pq_eotf, pq_oetf, srgb_eotf, srgb_oetf};
use crate::types::{ColorGamut, ColorTransfer, Error, GainMap, GainMapMetadata, Result};

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
            source_gamut: ColorGamut::Bt2020,
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
    bt2390_tonemap_ext(scene_linear, source_peak, target_peak, None)
}

/// BT.2390 EETF with optional min_lum black crush correction.
///
/// The `min_lum` parameter adds the ITU-R BT.2390 black crush prevention term:
/// `e3 = min_lum * (1 - e2)^4 + e2`, which lifts near-black values slightly to
/// prevent shadow detail loss during tone mapping.
#[inline]
pub fn bt2390_tonemap_ext(
    scene_linear: f32,
    source_peak: f32,
    target_peak: f32,
    min_lum: Option<f32>,
) -> f32 {
    if source_peak <= target_peak {
        return scene_linear;
    }

    let ks = 1.5 * target_peak / source_peak - 0.5;
    let ks = ks.clamp(0.0, 1.0);

    let e1 = scene_linear;
    let e2 = if e1 < ks {
        e1
    } else {
        let t = (e1 - ks) / (1.0 - ks);
        let t2 = t * t;
        let t3 = t2 * t;
        let p0 = ks;
        let p1 = 1.0;
        let m0 = 1.0 - ks;
        let m1 = 0.0;
        let a = 2.0 * t3 - 3.0 * t2 + 1.0;
        let b = t3 - 2.0 * t2 + t;
        let c = -2.0 * t3 + 3.0 * t2;
        let d = t3 - t2;
        a * p0 + b * m0 + c * p1 + d * m1
    };

    let e3 = if let Some(ml) = min_lum {
        let one_minus_e2 = 1.0 - e2;
        let one_minus_e2_2 = one_minus_e2 * one_minus_e2;
        ml * (one_minus_e2_2 * one_minus_e2_2) + e2
    } else {
        e2
    };

    e3 * target_peak / source_peak
}

// ============================================================================
// BT.2408 PQ-domain Tonemapper (ITU-R BT.2408)
// ============================================================================

/// BT.2408 tone mapper operating in PQ perceptual domain.
///
/// Precomputes PQ-domain constants from content and display peak nits.
/// Operates in the perceptual PQ domain for better shadow/highlight handling
/// than scene-linear BT.2390.
pub struct Bt2408Tonemapper {
    content_min_pq: f32,
    content_range_pq: f32,
    inv_content_range_pq: f32,
    min_lum: f32,
    max_lum: f32,
    ks: f32,
    inv_one_minus_ks: f32,
    one_minus_ks: f32,
    normalizer: f32,
    inv_display_max: f32,
    content_max_nits: f32,
    display_max_nits: f32,
}

impl Bt2408Tonemapper {
    /// Create a new BT.2408 tonemapper.
    ///
    /// `content_max_nits`: Peak luminance of source content (e.g. 4000.0)
    /// `display_max_nits`: Peak luminance of target display (e.g. 1000.0)
    pub fn new(content_max_nits: f32, display_max_nits: f32) -> Self {
        let content_min_pq = pq_oetf(0.0);
        let content_max_pq = pq_oetf(content_max_nits / 10000.0);
        let content_range_pq = content_max_pq - content_min_pq;
        let inv_content_range_pq = if content_range_pq > 0.0 {
            1.0 / content_range_pq
        } else {
            1.0
        };
        let min_lum = (pq_oetf(0.0) - content_min_pq) * inv_content_range_pq;
        let max_lum = (pq_oetf(display_max_nits / 10000.0) - content_min_pq) * inv_content_range_pq;
        let ks = 1.5 * max_lum - 0.5;
        Self {
            content_min_pq,
            content_range_pq,
            inv_content_range_pq,
            min_lum,
            max_lum,
            ks,
            inv_one_minus_ks: 1.0 / (1.0 - ks).max(1e-6),
            one_minus_ks: 1.0 - ks,
            normalizer: content_max_nits / display_max_nits,
            inv_display_max: 1.0 / display_max_nits,
            content_max_nits,
            display_max_nits,
        }
    }

    /// Tone map a single luminance value (in nits).
    #[inline]
    pub fn tonemap_luminance(&self, nits: f32) -> f32 {
        if nits <= 0.0 {
            return 0.0;
        }
        let scale = self.make_luma_scale(nits);
        (nits * scale).min(self.display_max_nits).max(0.0)
    }

    /// Tone map an RGB triple (linear light, normalized to 10000 nits).
    #[inline]
    pub fn tonemap_rgb(&self, rgb: [f32; 3]) -> [f32; 3] {
        let luma = 0.2126 * rgb[0] + 0.7152 * rgb[1] + 0.0722 * rgb[2];
        let luma_nits = luma * self.content_max_nits;
        if luma_nits <= 0.0 {
            return [0.0, 0.0, 0.0];
        }
        let scale = self.make_luma_scale(luma_nits);
        [rgb[0] * scale, rgb[1] * scale, rgb[2] * scale]
    }

    #[inline(always)]
    fn t(&self, a: f32) -> f32 {
        (a - self.ks) * self.inv_one_minus_ks
    }

    #[inline]
    fn hermite_spline(&self, b: f32) -> f32 {
        let t_b = self.t(b);
        let t_b_2 = t_b * t_b;
        let t_b_3 = t_b_2 * t_b;
        (2.0 * t_b_3 - 3.0 * t_b_2 + 1.0) * self.ks
            + (t_b_3 - 2.0 * t_b_2 + t_b) * self.one_minus_ks
            + (-2.0 * t_b_3 + 3.0 * t_b_2) * self.max_lum
    }

    #[inline(always)]
    fn make_luma_scale(&self, luma_nits: f32) -> f32 {
        let s = pq_oetf(luma_nits / 10000.0);
        let normalized_pq = ((s - self.content_min_pq) * self.inv_content_range_pq).min(1.0);
        let e2 = if normalized_pq < self.ks {
            normalized_pq
        } else {
            self.hermite_spline(normalized_pq)
        };
        let one_minus_e2 = 1.0 - e2;
        let one_minus_e2_2 = one_minus_e2 * one_minus_e2;
        let e3 = self.min_lum * (one_minus_e2_2 * one_minus_e2_2) + e2;
        let e4 = e3 * self.content_range_pq + self.content_min_pq;
        let d4 = pq_eotf(e4) * 10000.0;
        let new_luminance = d4.min(self.display_max_nits).max(0.0);
        let min_luminance = 1e-6;
        if luma_nits <= min_luminance {
            new_luminance * self.inv_display_max
        } else {
            (new_luminance / luma_nits.max(min_luminance)) * self.normalizer
        }
    }
}

// ============================================================================
// Simple Tone Mapping Curves
// ============================================================================

/// Simple per-channel Reinhard: `x / (1 + x)`.
#[inline]
pub fn reinhard_simple(x: f32) -> f32 {
    x / (1.0 + x)
}

/// Clamp tone map: simply clamps to `[0, 1]`.
#[inline]
pub fn clamp_tonemap(x: f32) -> f32 {
    x.clamp(0.0, 1.0)
}

/// Reinhard-Jodie tone mapping.
///
/// Mixes per-channel Reinhard with luminance-based Reinhard using the
/// per-channel result as the blend factor.
pub fn reinhard_jodie(rgb: [f32; 3], luma_coeffs: [f32; 3]) -> [f32; 3] {
    let luma = rgb[0] * luma_coeffs[0] + rgb[1] * luma_coeffs[1] + rgb[2] * luma_coeffs[2];
    if luma <= 0.0 {
        return [0.0, 0.0, 0.0];
    }
    let luma_scale = 1.0 / (1.0 + luma);
    let mut out = [0.0f32; 3];
    for i in 0..3 {
        let tv = rgb[i] / (1.0 + rgb[i]);
        out[i] = ((1.0 - tv) * (rgb[i] * luma_scale) + tv * tv).min(1.0);
    }
    out
}

/// Tuned Reinhard with display-aware parameters.
pub fn tuned_reinhard(luma: f32, content_max: f32, display_max: f32) -> f32 {
    let white_point = 203.0;
    let ld = content_max / white_point;
    let w_a = (display_max / white_point) / (ld * ld);
    let w_b = 1.0 / (display_max / white_point);
    (1.0 + w_a * luma) / (1.0 + w_b * luma)
}

// ============================================================================
// Complex Tone Mapping Curves
// ============================================================================

/// Uncharted 2 filmic tone mapping (Hable).
pub fn uncharted2_filmic(v: f32) -> f32 {
    #[inline(always)]
    fn partial(x: f32) -> f32 {
        const A: f32 = 0.15;
        const B: f32 = 0.50;
        const C: f32 = 0.10;
        const D: f32 = 0.20;
        const E: f32 = 0.02;
        const F: f32 = 0.30;
        ((x * (A * x + C * B) + D * E) / (x * (A * x + B) + D * F)) - E / F
    }
    const EXPOSURE_BIAS: f32 = 2.0;
    const W: f32 = 11.2;
    const W_SCALE: f32 = 1.0 / partial_const(W);
    (partial(v * EXPOSURE_BIAS) * W_SCALE).min(1.0)
}

#[inline(always)]
const fn partial_const(x: f32) -> f32 {
    const A: f32 = 0.15;
    const B: f32 = 0.50;
    const C: f32 = 0.10;
    const D: f32 = 0.20;
    const E: f32 = 0.02;
    const F: f32 = 0.30;
    ((x * (A * x + C * B) + D * E) / (x * (A * x + B) + D * F)) - E / F
}

/// ACES AP1 filmic tone mapping.
#[allow(clippy::excessive_precision)]
pub fn aces_ap1(rgb: [f32; 3]) -> [f32; 3] {
    let a = 0.59719 * rgb[0] + 0.35458 * rgb[1] + 0.04823 * rgb[2];
    let b = 0.07600 * rgb[0] + 0.90834 * rgb[1] + 0.01566 * rgb[2];
    let c = 0.02840 * rgb[0] + 0.13383 * rgb[1] + 0.83777 * rgb[2];
    let ra = a * (a + 0.0245786) - 0.000090537;
    let rb = a * (a * 0.983729 + 0.4329510) + 0.238081;
    let ga = b * (b + 0.0245786) - 0.000090537;
    let gb = b * (b * 0.983729 + 0.4329510) + 0.238081;
    let ba = c * (c + 0.0245786) - 0.000090537;
    let bb = c * (c * 0.983729 + 0.4329510) + 0.238081;
    let mr = ra / rb;
    let mg = ga / gb;
    let mb = ba / bb;
    [
        (1.60475 * mr - 0.53108 * mg - 0.07367 * mb).min(1.0),
        (-0.10208 * mr + 1.10813 * mg - 0.00605 * mb).min(1.0),
        (-0.00327 * mr - 0.07276 * mg + 1.07602 * mb).min(1.0),
    ]
}

/// AgX look preset.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum AgxLook {
    /// Default AgX (no look applied).
    Default,
    /// Punchy: increased saturation (1.4x).
    Punchy,
    /// Golden: warm tone with reduced blue.
    Golden,
}

/// AgX tone mapping (Blender).
///
/// Operates in a log2 domain with a polynomial contrast curve.
#[allow(clippy::excessive_precision)]
pub fn agx_tonemap(rgb: [f32; 3], look: AgxLook) -> [f32; 3] {
    const AGX_MIN_EV: f32 = -12.47393;
    const AGX_MAX_EV: f32 = 4.026069;
    const RECIP_EV: f32 = 1.0 / (AGX_MAX_EV - AGX_MIN_EV);

    let z = [rgb[0].abs(), rgb[1].abs(), rgb[2].abs()];
    let z0 = [
        0.856627153315983 * z[0] + 0.137318972929847 * z[1] + 0.11189821299995 * z[2],
        0.0951212405381588 * z[0] + 0.761241990602591 * z[1] + 0.0767994186031903 * z[2],
        0.0482516061458583 * z[0] + 0.101439036467562 * z[1] + 0.811302368396859 * z[2],
    ];
    let z1 = [
        z0[0].max(1e-10).log2().clamp(AGX_MIN_EV, AGX_MAX_EV),
        z0[1].max(1e-10).log2().clamp(AGX_MIN_EV, AGX_MAX_EV),
        z0[2].max(1e-10).log2().clamp(AGX_MIN_EV, AGX_MAX_EV),
    ];
    let z2 = [
        (z1[0] - AGX_MIN_EV) * RECIP_EV,
        (z1[1] - AGX_MIN_EV) * RECIP_EV,
        (z1[2] - AGX_MIN_EV) * RECIP_EV,
    ];
    let z3 = [
        agx_contrast(z2[0]),
        agx_contrast(z2[1]),
        agx_contrast(z2[2]),
    ];
    let z4 = agx_apply_look(z3, look);
    [
        (1.19687900512017 * z4[0] - 0.0528968517574562 * z4[1] - 0.0529716355144438 * z4[2])
            .clamp(0.0, 1.0),
        (-0.0980208811401368 * z4[0] + 1.15190312990417 * z4[1] - 0.0505349770312032 * z4[2])
            .clamp(0.0, 1.0),
        (-0.0990297440797205 * z4[0] - 0.0989611768448433 * z4[1] + 1.15107367264116 * z4[2])
            .clamp(0.0, 1.0),
    ]
}

#[inline]
fn agx_contrast(x: f32) -> f32 {
    let x2 = x * x;
    let x4 = x2 * x2;
    let x6 = x4 * x2;
    let w0 = 0.002857 * x - 0.1718;
    let w1 = 4.361 * x - 28.72;
    let w2 = 92.06 * x - 126.7;
    let w3 = 78.01 * x - 17.86;
    let z0 = w0 * x2 + w1;
    let z1 = x4 * w2 * x6 + w3;
    z1 + z0
}

fn agx_apply_look(rgb: [f32; 3], look: AgxLook) -> [f32; 3] {
    let (slope, power, saturation) = match look {
        AgxLook::Default => return rgb,
        AgxLook::Punchy => ([1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.4, 1.4, 1.4]),
        AgxLook::Golden => ([1.0, 0.9, 0.5], [0.8, 0.8, 0.8], [1.2, 1.2, 1.2]),
    };
    let dot = [
        (slope[0] * rgb[0]).max(0.0),
        (slope[1] * rgb[1]).max(0.0),
        (slope[2] * rgb[2]).max(0.0),
    ];
    let z = [
        dot[0].powf(power[0]),
        dot[1].powf(power[1]),
        dot[2].powf(power[2]),
    ];
    let luma = 0.2126 * z[0] + 0.7152 * z[1] + 0.0722 * z[2];
    [
        saturation[0] * (z[0] - luma) + luma,
        saturation[1] * (z[1] - luma) + luma,
        saturation[2] * (z[2] - luma) + luma,
    ]
}

/// Filmic spline configuration parameters.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FilmicSplineConfig {
    /// Output power (gamma). Default: 1.0
    pub output_power: f32,
    /// Latitude percentage (0–100). Default: 33.0
    pub latitude: f32,
    /// White point in EV. Default: 3.0
    pub white_point_source: f32,
    /// Black point in EV. Default: -8.0
    pub black_point_source: f32,
    /// Contrast at middle gray. Default: 1.18
    pub contrast: f32,
    /// Target black luminance (%). Default: 0.01517634
    pub black_point_target: f32,
    /// Target middle gray (%). Default: 18.45
    pub grey_point_target: f32,
    /// Target white luminance (%). Default: 100.0
    pub white_point_target: f32,
    /// Balance (-50 to 50). Default: 0.0
    pub balance: f32,
    /// Extreme luminance saturation. Default: 0.0
    pub saturation: f32,
}

impl Default for FilmicSplineConfig {
    fn default() -> Self {
        Self {
            output_power: 1.0,
            latitude: 33.0,
            white_point_source: 3.0,
            black_point_source: -8.0,
            contrast: 1.18,
            black_point_target: 0.01517634,
            grey_point_target: 18.45,
            white_point_target: 100.0,
            balance: 0.0,
            saturation: 0.0,
        }
    }
}

/// Compiled filmic spline (precomputed from [`FilmicSplineConfig`]).
pub struct CompiledFilmicSpline {
    m1: [f32; 3],
    m2: [f32; 3],
    m3: [f32; 3],
    m4: [f32; 3],
    latitude_min: f32,
    latitude_max: f32,
    grey_source: f32,
    black_source: f32,
    dynamic_range: f32,
    sigma_toe: f32,
    sigma_shoulder: f32,
    saturation: f32,
}

impl CompiledFilmicSpline {
    /// Build a compiled spline from parameters.
    pub fn new(p: &FilmicSplineConfig) -> Self {
        let hardness = p.output_power;
        let grey_display = 0.1845_f32.powf(1.0 / hardness);
        let latitude = p.latitude.clamp(0.0, 100.0) / 100.0;
        let white_source = p.white_point_source;
        let black_source = p.black_point_source;
        let dynamic_range = white_source - black_source;
        let grey_log = black_source.abs() / dynamic_range;
        let white_log = 1.0_f32;
        let black_log = 0.0_f32;
        let black_display =
            (p.black_point_target.clamp(0.0, p.grey_point_target) / 100.0).powf(1.0 / hardness);
        let white_display =
            (p.white_point_target.max(p.grey_point_target) / 100.0).powf(1.0 / hardness);
        let balance = p.balance.clamp(-50.0, 50.0) / 100.0;
        let slope = p.contrast * dynamic_range / 8.0;
        let mut min_contrast = 1.0_f32;
        let mc2 = (white_display - grey_display) / (white_log - grey_log);
        if mc2.is_finite() {
            min_contrast = min_contrast.max(mc2);
        }
        const SAFETY_MARGIN: f32 = 0.01;
        min_contrast += SAFETY_MARGIN;
        let mut contrast = slope / (hardness * grey_display.powf(hardness - 1.0));
        contrast = contrast.clamp(min_contrast, 100.0);
        let linear_intercept = grey_display - contrast * grey_log;
        let xmin = (black_display + SAFETY_MARGIN * (white_display - black_display)
            - linear_intercept)
            / contrast;
        let xmax =
            (white_display - SAFETY_MARGIN * (white_display - black_display) - linear_intercept)
                / contrast;
        let mut toe_log = (1.0 - latitude) * grey_log + latitude * xmin;
        let mut shoulder_log = (1.0 - latitude) * grey_log + latitude * xmax;
        let balance_correction = if balance > 0.0 {
            2.0 * balance * (shoulder_log - grey_log)
        } else {
            2.0 * balance * (grey_log - toe_log)
        };
        toe_log -= balance_correction;
        shoulder_log -= balance_correction;
        toe_log = toe_log.max(xmin);
        shoulder_log = shoulder_log.min(xmax);
        let toe_display = toe_log * contrast + linear_intercept;
        let shoulder_display = shoulder_log * contrast + linear_intercept;
        let latitude_min = toe_log;
        let latitude_max = shoulder_log;
        let saturation = 2.0 * p.saturation / 100.0 + 1.0;
        let sigma_toe = (latitude_min / 3.0).powi(2);
        let sigma_shoulder = ((1.0 - latitude_max) / 3.0).powi(2);
        let m2_2 = contrast;
        let m1_2 = toe_display - m2_2 * toe_log;
        let (m1_0, m2_0, m3_0, m4_0) =
            Self::compute_rational([black_log, black_display], [toe_log, toe_display], contrast);
        let (m1_1, m2_1, m3_1, m4_1) = Self::compute_rational(
            [white_log, white_display],
            [shoulder_log, shoulder_display],
            contrast,
        );
        Self {
            m1: [m1_0, m1_1, m1_2],
            m2: [m2_0, m2_1, m2_2],
            m3: [m3_0, m3_1, 0.0],
            m4: [m4_0, m4_1, 0.0],
            latitude_min,
            latitude_max,
            grey_source: 0.1845,
            black_source,
            dynamic_range,
            sigma_toe,
            sigma_shoulder,
            saturation,
        }
    }

    fn compute_rational(p1: [f32; 2], p0: [f32; 2], g: f32) -> (f32, f32, f32, f32) {
        let x = p0[0] - p1[0];
        let y = p0[1] - p1[1];
        let jx = (x * g / y + 1.0).powi(2).max(4.0);
        let b = g / (2.0 * y) + ((jx - 4.0).sqrt() - 1.0) / (2.0 * x);
        let c = y / g * (b * x * x + x) / (b * x * x + x - y / g);
        let a = c * g;
        (a, b, c, p0[1])
    }

    /// Apply the spline to a single value in log-encoded domain.
    #[inline]
    pub fn apply_spline(&self, x: f32) -> f32 {
        if x < self.latitude_min {
            let xi = self.latitude_min - x;
            let rat = xi * (xi * self.m2[0] + 1.0);
            self.m4[0] - self.m1[0] * rat / (rat + self.m3[0])
        } else if x > self.latitude_max {
            let xi = x - self.latitude_max;
            let rat = xi * (xi * self.m2[1] + 1.0);
            self.m4[1] + self.m1[1] * rat / (rat + self.m3[1])
        } else {
            self.m1[2] + x * self.m2[2]
        }
    }

    #[inline]
    fn shaper(&self, x: f32) -> f32 {
        (((x.max(1.525879e-05) / self.grey_source).log2() - self.black_source) / self.dynamic_range)
            .clamp(0.0, 1.0)
    }

    #[inline]
    fn desaturate(&self, x: f32) -> f32 {
        let radius_toe = x;
        let radius_shoulder = 1.0 - x;
        let sat2 = 0.5 / self.saturation.sqrt();
        let key_toe = (-radius_toe * radius_toe / self.sigma_toe * sat2).exp();
        let key_shoulder = (-radius_shoulder * radius_shoulder / self.sigma_shoulder * sat2).exp();
        self.saturation - (key_toe + key_shoulder) * self.saturation
    }

    /// Tone map an RGB value through the filmic spline.
    pub fn tonemap_rgb(&self, rgb: [f32; 3], luma_coeffs: [f32; 3]) -> [f32; 3] {
        let mut norm =
            (rgb[0] * luma_coeffs[0] + rgb[1] * luma_coeffs[1] + rgb[2] * luma_coeffs[2])
                .max(1.525879e-05);
        let mut ratios = [rgb[0] / norm, rgb[1] / norm, rgb[2] / norm];
        let min_ratio = ratios[0].min(ratios[1]).min(ratios[2]);
        if min_ratio < 0.0 {
            ratios[0] -= min_ratio;
            ratios[1] -= min_ratio;
            ratios[2] -= min_ratio;
        }
        norm = self.shaper(norm);
        let desat = self.desaturate(norm);
        let mapped = self.apply_spline(norm).clamp(0.0, 1.0);
        [
            ((ratios[0] + (1.0 - ratios[0]) * (1.0 - desat)) * mapped).clamp(0.0, 1.0),
            ((ratios[1] + (1.0 - ratios[1]) * (1.0 - desat)) * mapped).clamp(0.0, 1.0),
            ((ratios[2] + (1.0 - ratios[2]) * (1.0 - desat)) * mapped).clamp(0.0, 1.0),
        ]
    }
}

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
// Unified Tone Map Curve Enum
// ============================================================================

/// Enumeration of all supported tone mapping curves.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ToneMapCurve {
    /// Simple per-channel Reinhard: `x / (1 + x)`
    Reinhard,
    /// Extended Reinhard with max luminance
    ExtendedReinhard {
        /// Maximum expected luminance
        l_max: f32,
    },
    /// Reinhard-Jodie (luminance-aware per-channel)
    ReinhardJodie,
    /// Tuned Reinhard with display-aware weights
    TunedReinhard {
        /// Content peak luminance in nits
        content_max: f32,
        /// Display peak luminance in nits
        display_max: f32,
    },
    /// ACES-inspired filmic (Narkowicz approximation)
    Narkowicz,
    /// Uncharted 2 filmic (Hable)
    Uncharted2,
    /// ACES AP1 RRT+ODT
    AcesAp1,
    /// BT.2390 EETF (scene-linear domain)
    Bt2390 {
        /// Source peak luminance (normalized)
        source_peak: f32,
        /// Target peak luminance (normalized)
        target_peak: f32,
    },
    /// AgX (Blender)
    Agx(AgxLook),
    /// Clamp to `[0, 1]`
    Clamp,
}

/// Apply a tone mapping curve to an RGB triple.
pub fn tonemap_rgb_curve(curve: &ToneMapCurve, rgb: [f32; 3], luma_coeffs: [f32; 3]) -> [f32; 3] {
    match *curve {
        ToneMapCurve::Reinhard => [
            reinhard_simple(rgb[0]).min(1.0),
            reinhard_simple(rgb[1]).min(1.0),
            reinhard_simple(rgb[2]).min(1.0),
        ],
        ToneMapCurve::ExtendedReinhard { l_max } => {
            let l = rgb[0] * luma_coeffs[0] + rgb[1] * luma_coeffs[1] + rgb[2] * luma_coeffs[2];
            if l <= 0.0 {
                return [0.0, 0.0, 0.0];
            }
            let new_l = reinhard_tonemap(l, l_max);
            let scale = new_l / l;
            [
                (rgb[0] * scale).min(1.0),
                (rgb[1] * scale).min(1.0),
                (rgb[2] * scale).min(1.0),
            ]
        }
        ToneMapCurve::ReinhardJodie => reinhard_jodie(rgb, luma_coeffs),
        ToneMapCurve::TunedReinhard {
            content_max,
            display_max,
        } => {
            let l = rgb[0] * luma_coeffs[0] + rgb[1] * luma_coeffs[1] + rgb[2] * luma_coeffs[2];
            if l <= 0.0 {
                return [0.0, 0.0, 0.0];
            }
            let scale = tuned_reinhard(l, content_max, display_max);
            [
                (rgb[0] * scale).min(1.0),
                (rgb[1] * scale).min(1.0),
                (rgb[2] * scale).min(1.0),
            ]
        }
        ToneMapCurve::Narkowicz => [
            filmic_tonemap(rgb[0]),
            filmic_tonemap(rgb[1]),
            filmic_tonemap(rgb[2]),
        ],
        ToneMapCurve::Uncharted2 => [
            uncharted2_filmic(rgb[0]),
            uncharted2_filmic(rgb[1]),
            uncharted2_filmic(rgb[2]),
        ],
        ToneMapCurve::AcesAp1 => aces_ap1(rgb),
        ToneMapCurve::Bt2390 {
            source_peak,
            target_peak,
        } => [
            bt2390_tonemap(rgb[0], source_peak, target_peak),
            bt2390_tonemap(rgb[1], source_peak, target_peak),
            bt2390_tonemap(rgb[2], source_peak, target_peak),
        ],
        ToneMapCurve::Agx(look) => agx_tonemap(rgb, look),
        ToneMapCurve::Clamp => [
            clamp_tonemap(rgb[0]),
            clamp_tonemap(rgb[1]),
            clamp_tonemap(rgb[2]),
        ],
    }
}

// ============================================================================
// Batch Tone Map API
// ============================================================================

/// Apply a tone mapping curve to a row of interleaved float pixel data.
///
/// Processes pixels in-place using `chunks_exact_mut` for bounds-check-free loops.
/// `channels` must be 3 or 4 (alpha is passed through unchanged for 4-channel).
pub fn tonemap_row(curve: &ToneMapCurve, row: &mut [f32], channels: usize, luma_coeffs: [f32; 3]) {
    debug_assert!(channels == 3 || channels == 4, "channels must be 3 or 4");
    for chunk in row.chunks_exact_mut(channels) {
        let rgb = [chunk[0], chunk[1], chunk[2]];
        let mapped = tonemap_rgb_curve(curve, rgb, luma_coeffs);
        chunk[0] = mapped[0];
        chunk[1] = mapped[1];
        chunk[2] = mapped[2];
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

        // Validate pixel data is large enough for declared dimensions
        hdr.validate_data_bounds()?;
        sdr.validate_data_bounds()?;

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
            max_hdr_observed: 2.0f32.powf(metadata.alternate_hdr_headroom as f32),
            stats: FitStats::default(),
        }
    }

    /// Apply the tonemapper to an HDR image.
    pub fn apply(&self, hdr: &RawImage) -> Result<RawImage> {
        // Validate pixel data is large enough for declared dimensions
        hdr.validate_data_bounds()?;

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

                // Invert: SDR = (HDR + alternate_offset) / gain - base_offset
                let sdr_linear = [
                    (hdr_linear[0] + metadata.alternate_offset[0] as f32) / gain[0]
                        - metadata.base_offset[0] as f32,
                    (hdr_linear[1] + metadata.alternate_offset[1] as f32) / gain[1]
                        - metadata.base_offset[1] as f32,
                    (hdr_linear[2] + metadata.alternate_offset[2] as f32) / gain[2]
                        - metadata.base_offset[2] as f32,
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
pub fn tonemap_image_to_srgb8(img: &RawImage, target_gamut: ColorGamut) -> Result<Vec<u8>> {
    use crate::color::gamut::convert_gamut;

    // Validate pixel data is large enough for declared dimensions
    img.validate_data_bounds()?;

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

    Ok(output)
}

// ============================================================================
// Helper Functions
// ============================================================================

/// Get linear RGB from any image format.
fn get_linear_rgb(img: &RawImage, x: u32, y: u32) -> [f32; 3] {
    use crate::PixelFormat;
    use crate::color::transfer::{hlg_oetf_inv, pq_eotf};

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
    let gamma = metadata.gamma[channel] as f32;
    let linear = if gamma != 1.0 && gamma > 0.0 {
        normalized.powf(1.0 / gamma)
    } else {
        normalized
    };

    // Convert log2 domain to natural log for exp() math
    let ln2 = core::f64::consts::LN_2;
    let log_min = (metadata.gain_map_min[channel] * ln2) as f32;
    let log_max = (metadata.gain_map_max[channel] * ln2) as f32;
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
            let rh = reinhard_tonemap(l, peak);
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
        let rh_bright = reinhard_tonemap(bright, peak);
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
            target_gamut: ColorGamut::Bt709,
            source_gamut: ColorGamut::Bt2020,
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

        // Create all-black HDR (16F) and SDR (RGBA8) images
        let hdr_data = vec![0u8; (width * height * 8) as usize]; // f16 RGBA = 8 bytes/pixel
        let sdr_data = vec![0u8; (width * height * 4) as usize]; // RGBA8 = 4 bytes/pixel

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
        assert_eq!(filmic_tonemap(0.0), 0.0);
        assert_eq!(reinhard_tonemap(0.0, 10.0), 0.0);

        // Both map small values (~0.1) similarly
        let filmic_low = filmic_tonemap(0.1);
        let reinhard_low = reinhard_tonemap(0.1, 10.0);
        assert!(
            (filmic_low - reinhard_low).abs() < 0.15,
            "Expected similar low-value mapping: filmic={}, reinhard={}",
            filmic_low,
            reinhard_low
        );

        // They diverge at high values
        let filmic_high = filmic_tonemap(5.0);
        let reinhard_high = reinhard_tonemap(5.0, 10.0);
        assert!(
            (filmic_high - reinhard_high).abs() > 0.01,
            "Expected divergence at high values: filmic={}, reinhard={}",
            filmic_high,
            reinhard_high
        );

        // Both should stay in [0, 1]
        for i in 0..=100 {
            let x = i as f32 * 0.1;
            let f = filmic_tonemap(x);
            let r = reinhard_tonemap(x, 10.0);
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
        let tm = Bt2408Tonemapper::new(1000.0, 1000.0);
        let input = [0.01, 0.02, 0.005];
        let output = tm.tonemap_rgb(input);
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
        let result = tm.tonemap_luminance(0.001);
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
            let mapped = tm.tonemap_luminance(nits);
            assert!(
                mapped >= prev - 1e-6,
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
        let tm = Bt2408Tonemapper::new(4000.0, 1000.0);
        assert_eq!(tm.tonemap_rgb([0.0, 0.0, 0.0]), [0.0, 0.0, 0.0]);
    }

    #[test]
    fn test_bt2408_compresses_highlights() {
        let tm = Bt2408Tonemapper::new(4000.0, 1000.0);
        let result = tm.tonemap_luminance(4000.0);
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
    fn test_clamp_tonemap_values() {
        assert_eq!(clamp_tonemap(0.0), 0.0);
        assert_eq!(clamp_tonemap(0.5), 0.5);
        assert_eq!(clamp_tonemap(1.0), 1.0);
        assert_eq!(clamp_tonemap(2.0), 1.0);
        assert_eq!(clamp_tonemap(-0.5), 0.0);
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

    #[test]
    fn test_tuned_reinhard_positive() {
        // tuned_reinhard returns a scale factor, not a mapped value
        // It should always be positive and finite
        for i in 0..=100 {
            let v = i as f32 * 0.1;
            let mapped = tuned_reinhard(v, 4000.0, 1000.0);
            assert!(mapped > 0.0 && mapped.is_finite(), "at {}: {}", v, mapped);
        }
    }

    // ========================================================================
    // Phase 2b: Complex curves tests
    // ========================================================================

    #[test]
    fn test_uncharted2_filmic_properties() {
        assert!(uncharted2_filmic(0.0).abs() < 0.01);
        assert!(uncharted2_filmic(100.0) > 0.95);
        let mut prev = 0.0;
        for i in 1..=100 {
            let v = uncharted2_filmic(i as f32 * 0.1);
            assert!(v >= prev - 1e-6, "Not monotonic at {}", i);
            prev = v;
        }
    }

    #[test]
    fn test_uncharted2_bounded() {
        for i in 0..=1000 {
            let v = i as f32 * 0.01;
            let result = uncharted2_filmic(v);
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
        let config = FilmicSplineConfig::default();
        let spline = CompiledFilmicSpline::new(&config);
        let luma = [0.2126_f32, 0.7152, 0.0722];
        let mid = spline.tonemap_rgb([0.18, 0.18, 0.18], luma);
        assert!(mid[0] > 0.05 && mid[0] < 0.5, "Mid-gray: {:?}", mid);
        let mut prev = 0.0_f32;
        for i in 1..=100 {
            let v = i as f32 * 0.1;
            let result = spline.tonemap_rgb([v, v, v], luma);
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
        let config = FilmicSplineConfig::default();
        let spline = CompiledFilmicSpline::new(&config);
        let luma = [0.2126_f32, 0.7152, 0.0722];
        for i in 0..=100 {
            let v = i as f32 * 0.1;
            let result = spline.tonemap_rgb([v, v, v], luma);
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
            ToneMapCurve::ExtendedReinhard { l_max: 10.0 },
            ToneMapCurve::ReinhardJodie,
            ToneMapCurve::TunedReinhard {
                content_max: 4000.0,
                display_max: 1000.0,
            },
            ToneMapCurve::Narkowicz,
            ToneMapCurve::Uncharted2,
            ToneMapCurve::AcesAp1,
            ToneMapCurve::Bt2390 {
                source_peak: 10.0,
                target_peak: 1.0,
            },
            ToneMapCurve::Agx(AgxLook::Default),
            ToneMapCurve::Clamp,
        ];
        for curve in &curves {
            let result = tonemap_rgb_curve(curve, [0.0, 0.0, 0.0], luma);
            let sum: f32 = result.iter().map(|v| v.abs()).sum();
            assert!(sum < 0.05, "{:?}: black→{:?}", curve, result);
        }
    }

    #[test]
    fn test_tonemap_curve_all_variants_bright() {
        let luma = [0.2126_f32, 0.7152, 0.0722];
        let curves = [
            ToneMapCurve::Reinhard,
            ToneMapCurve::Narkowicz,
            ToneMapCurve::Uncharted2,
            ToneMapCurve::AcesAp1,
            ToneMapCurve::Agx(AgxLook::Default),
            ToneMapCurve::Clamp,
        ];
        for curve in &curves {
            let result = tonemap_rgb_curve(curve, [10.0, 10.0, 10.0], luma);
            for ch in &result {
                assert!(*ch <= 1.0, "{:?}: bright→{:?}", curve, result);
            }
        }
    }

    // ========================================================================
    // Phase 4: Batch tonemap_row tests
    // ========================================================================

    #[test]
    fn test_tonemap_row_basic() {
        let luma = [0.2126_f32, 0.7152, 0.0722];
        let curve = ToneMapCurve::Reinhard;
        let mut row = vec![0.0, 0.0, 0.0, 0.5, 0.5, 0.5, 2.0, 2.0, 2.0];
        tonemap_row(&curve, &mut row, 3, luma);
        assert!(row[0].abs() < 0.001);
        assert!((row[3] - 0.333).abs() < 0.02);
        assert!((row[6] - 0.667).abs() < 0.02);
    }

    #[test]
    fn test_tonemap_row_4ch_alpha_passthrough() {
        let luma = [0.2126_f32, 0.7152, 0.0722];
        let curve = ToneMapCurve::Clamp;
        let mut row = vec![2.0, 3.0, 4.0, 0.75, 0.5, 0.5, 0.5, 0.99];
        tonemap_row(&curve, &mut row, 4, luma);
        assert_eq!(row[0], 1.0);
        assert_eq!(row[1], 1.0);
        assert_eq!(row[2], 1.0);
        assert_eq!(row[3], 0.75);
        assert_eq!(row[7], 0.99);
    }

    #[test]
    fn test_tonemap_row_empty() {
        let luma = [0.2126_f32, 0.7152, 0.0722];
        let curve = ToneMapCurve::Reinhard;
        let mut row: Vec<f32> = vec![];
        tonemap_row(&curve, &mut row, 3, luma);
        assert!(row.is_empty());
    }

    #[test]
    fn test_tonemap_row_matches_per_pixel() {
        let luma = [0.2126_f32, 0.7152, 0.0722];
        let curve = ToneMapCurve::Uncharted2;
        let mut row = vec![0.1, 0.2, 0.3, 1.0, 2.0, 3.0, 5.0, 5.0, 5.0];
        let expected = [
            tonemap_rgb_curve(&curve, [0.1, 0.2, 0.3], luma),
            tonemap_rgb_curve(&curve, [1.0, 2.0, 3.0], luma),
            tonemap_rgb_curve(&curve, [5.0, 5.0, 5.0], luma),
        ];
        tonemap_row(&curve, &mut row, 3, luma);
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
            ToneMapCurve::ExtendedReinhard { l_max: 10.0 },
            ToneMapCurve::ReinhardJodie,
            ToneMapCurve::TunedReinhard {
                content_max: 4000.0,
                display_max: 1000.0,
            },
            ToneMapCurve::Narkowicz,
            ToneMapCurve::Uncharted2,
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
            tonemap_row(curve, &mut row, 3, luma);
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
