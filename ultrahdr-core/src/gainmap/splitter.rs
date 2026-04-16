//! Luma gain map splitter — round-trippable HDR ↔ (SDR, log2 gain).
//!
//! Splits a linear-light HDR frame into an SDR base frame plus a per-pixel
//! `log2` gain map, using the ISO 21496-1 decode form:
//!
//! ```text
//! HDR_i = (SDR_i + base_offset_i) · 2^g − alternate_offset_i      (per channel i)
//! ```
//!
//! Mirrors the gain form consumed by [`super::apply`]. Round-trip is exact
//! within float precision when (a) the curve is strictly monotonic, (b) the
//! observed gain fits in `[min_log2, max_log2]`, and (c) the SDR rescale
//! stays in `[0, 1]`. Out-of-gamut highlights clamp on the SDR side and
//! become lossy — the splitter reports the count via [`SplitStats`].
//!
//! ## Built-in curve
//!
//! [`HableFilmic`] (John Hable, GDC 2010 — "Uncharted 2") is provided as a
//! broadly-safe default. It's parameter-free, monotonic on `[0, ∞)`, has
//! a gentle shoulder, and needs no luminance calibration. It's a reasonable
//! choice when the caller does not have scene metadata to pick a principled
//! curve (PQ/HLG peak, etc.). For those cases, enable the `zentone` feature
//! to pick up ITU-R BT.2408 / BT.2446 and filmic-spline implementations.
//!
//! ## Required input descriptor
//!
//! Inputs are interleaved `&[f32]` rows with channels = 3 or 4.
//!
//! - **Transfer**: linear light. Linearize PQ/HLG/sRGB beforehand.
//! - **Signal range**: full range.
//! - **Primaries**: arbitrary, but [`SplitConfig::luma_weights`] MUST match.
//! - **Alpha**: passed through unchanged on RGBA rows.
//!
//! ## Wire-format alignment with [`zencodec::GainMapParams`]
//!
//! After splitting, build the gain map metadata as:
//!
//! | `GainMapChannel` field | source |
//! |---|---|
//! | `base_offset` | [`SplitConfig::base_offset`] (typ. `1.0/64.0`) |
//! | `alternate_offset` | [`SplitConfig::alternate_offset`] (typ. `1.0/64.0`) |
//! | `min` | [`SplitConfig::min_log2`] (or [`SplitStats::observed_min_log2`] if tighter) |
//! | `max` | [`SplitConfig::max_log2`] (or [`SplitStats::observed_max_log2`] if tighter) |
//! | `gamma` | encoder choice (typ. `1.0`); the splitter emits raw `log2` |
//!
//! The splitter intentionally emits **raw f32 log2 gain**. u8 quantization
//! and gamma encoding are the caller's responsibility (see
//! `pack_log2_gain_u8` in the internal compute module for the canonical
//! wire quantization).

use alloc::boxed::Box;

use crate::types::ColorGamut;

/// A scalar luma tone curve: `Y_HDR` (linear, ≥0) → `Y_SDR` (linear, `[0, 1]`).
///
/// Implementors must be **strictly monotonic** on the operating range and
/// produce output in `[0, 1]`. To wrap an ad-hoc closure, use [`LumaFn`].
pub trait LumaToneMap {
    /// Map a single linear-light luminance sample.
    fn map_luma(&self, y_hdr: f32) -> f32;
}

/// Adapt a closure as a [`LumaToneMap`]. Caller is responsible for
/// monotonicity and `[0, 1]` output range.
pub struct LumaFn<F: Fn(f32) -> f32>(pub F);

impl<F: Fn(f32) -> f32> LumaToneMap for LumaFn<F> {
    #[inline]
    fn map_luma(&self, y: f32) -> f32 {
        (self.0)(y)
    }
}

impl<T: LumaToneMap + ?Sized> LumaToneMap for &T {
    #[inline]
    fn map_luma(&self, y: f32) -> f32 {
        (**self).map_luma(y)
    }
}

impl<T: LumaToneMap + ?Sized> LumaToneMap for Box<T> {
    #[inline]
    fn map_luma(&self, y: f32) -> f32 {
        (**self).map_luma(y)
    }
}

// ---------------------------------------------------------------------------
// Built-in broadly-safe curve: Hable filmic
// ---------------------------------------------------------------------------

/// John Hable's "Uncharted 2" filmic tone curve (GDC 2010).
///
/// Parameter-free. Maps `[0, ∞)` to `[0, 1)` with a gentle shoulder and
/// slight shadow lift. Useful as a default when the caller lacks scene
/// metadata. For content-aware curves, enable the `zentone` feature and
/// use BT.2408/BT.2446 or the filmic spline instead.
#[derive(Debug, Clone, Copy, Default)]
pub struct HableFilmic;

impl HableFilmic {
    /// Construct a new instance. Equivalent to `HableFilmic::default()`.
    pub const fn new() -> Self {
        Self
    }
}

#[inline(always)]
const fn hable_partial(x: f32) -> f32 {
    const A: f32 = 0.15;
    const B: f32 = 0.50;
    const C: f32 = 0.10;
    const D: f32 = 0.20;
    const E: f32 = 0.02;
    const F: f32 = 0.30;
    ((x * (A * x + C * B) + D * E) / (x * (A * x + B) + D * F)) - E / F
}

impl LumaToneMap for HableFilmic {
    #[inline]
    fn map_luma(&self, y: f32) -> f32 {
        const EXPOSURE_BIAS: f32 = 2.0;
        const W: f32 = 11.2;
        const W_SCALE: f32 = 1.0 / hable_partial(W);
        let y = y.max(0.0);
        (hable_partial(y * EXPOSURE_BIAS) * W_SCALE).min(1.0)
    }
}

// ---------------------------------------------------------------------------
// Zentone adapter impls (when feature = "zentone")
// ---------------------------------------------------------------------------

#[cfg(feature = "zentone")]
mod zentone_adapters {
    use super::LumaToneMap;
    use zentone::{Bt2408Tonemapper, Bt2446A, Bt2446B, Bt2446C, CompiledFilmicSpline, ToneMap};

    impl LumaToneMap for Bt2408Tonemapper {
        #[inline]
        fn map_luma(&self, y: f32) -> f32 {
            self.map_rgb([y, y, y])[0]
        }
    }

    impl LumaToneMap for Bt2446A {
        #[inline]
        fn map_luma(&self, y: f32) -> f32 {
            self.map_rgb([y, y, y])[0]
        }
    }

    impl LumaToneMap for Bt2446B {
        #[inline]
        fn map_luma(&self, y: f32) -> f32 {
            self.map_rgb([y, y, y])[0]
        }
    }

    impl LumaToneMap for Bt2446C {
        #[inline]
        fn map_luma(&self, y: f32) -> f32 {
            self.map_rgb([y, y, y])[0]
        }
    }

    impl LumaToneMap for CompiledFilmicSpline {
        #[inline]
        fn map_luma(&self, y: f32) -> f32 {
            self.map_rgb([y, y, y])[0]
        }
    }
}

// ---------------------------------------------------------------------------
// Splitter
// ---------------------------------------------------------------------------

/// Splitter configuration.
///
/// Defaults are chosen so the splitter "just works" with any qualifying
/// curve, including shadow-lifters. The `min_log2` and `max_log2` fields
/// are **safety clamps** that prevent infinite gain at black or
/// unreasonable values from numerical edge cases — they are not quality
/// knobs. The gain range to store in [`zencodec::GainMapChannel::min`] /
/// `max` should come from [`SplitStats::observed_min_log2`] /
/// [`SplitStats::observed_max_log2`] after a pass over the image.
#[derive(Debug, Clone, Copy)]
pub struct SplitConfig {
    /// RGB → Y weights. Must match the input primaries.
    pub luma_weights: [f32; 3],
    /// Offset on the base (SDR) image. Maps to `GainMapChannel.base_offset`.
    pub base_offset: f32,
    /// Offset on the alternate (HDR) image. Maps to `GainMapChannel.alternate_offset`.
    pub alternate_offset: f32,
    /// Sanity floor on `log2` gain. `-4.0` (1/16×) tolerates shadow-lifting curves.
    pub min_log2: f32,
    /// Sanity ceiling on `log2` gain. `6.0` (64×) covers typical HDR headroom plus margin.
    pub max_log2: f32,
    /// Pre-desaturation (crosstalk) parameter in `[0.0, 0.33)`.
    ///
    /// Before the chromaticity-preserving RGB rescale, each HDR channel
    /// is blended toward the pixel's mean:
    /// ```text
    /// R' = (1 − 2α)·R + α·G + α·B
    /// ```
    /// Pulls saturated primaries toward gray, reducing the chance that
    /// the SDR rescale pushes a channel above 1.0 (out-of-gamut). The
    /// inverse matrix is applied after the gain is computed, so the
    /// desaturation is transparent to the round-trip.
    pub pre_desaturate: f32,
}

impl SplitConfig {
    /// Construct a default config for the given gamut (populates
    /// [`luma_weights`](Self::luma_weights) from the gamut's BT-standard
    /// coefficients).
    pub fn for_gamut(gamut: ColorGamut) -> Self {
        Self {
            luma_weights: crate::color::gamut::luma_coefficients(gamut),
            ..Self::default()
        }
    }
}

impl Default for SplitConfig {
    fn default() -> Self {
        Self {
            // Default to BT.2020 luma — caller should override via
            // [`Self::for_gamut`] or manual construction when working in
            // BT.709 or DisplayP3.
            luma_weights: crate::color::gamut::BT2100_LUMA,
            base_offset: 1.0 / 64.0,
            alternate_offset: 1.0 / 64.0,
            min_log2: -4.0,
            max_log2: 6.0,
            pre_desaturate: 0.0,
        }
    }
}

/// Per-row splitter statistics. Accumulate across rows; pass into
/// [`zencodec::GainMapParams`] metadata or use to tighten `min_log2` /
/// `max_log2` on a second pass.
#[derive(Debug, Clone, Copy)]
pub struct SplitStats {
    /// Smallest pre-clamp `log2` gain seen. Initialize to `f32::INFINITY`.
    pub observed_min_log2: f32,
    /// Largest pre-clamp `log2` gain seen. Initialize to `f32::NEG_INFINITY`.
    pub observed_max_log2: f32,
    /// Pixels where any SDR channel had to be clamped to `[0, 1]`. These
    /// pixels are NOT exactly invertible (HDR roundtrip will differ).
    pub clipped_sdr_pixels: u32,
}

impl Default for SplitStats {
    fn default() -> Self {
        Self {
            observed_min_log2: f32::INFINITY,
            observed_max_log2: f32::NEG_INFINITY,
            clipped_sdr_pixels: 0,
        }
    }
}

/// Splits HDR rows into (SDR, log2-gain) pairs around a [`LumaToneMap`].
///
/// Stateless after construction. Safe to share across threads (`Sync`)
/// when the inner curve is `Sync`.
pub struct LumaGainMapSplitter<T: LumaToneMap> {
    curve: T,
    cfg: SplitConfig,
}

impl<T: LumaToneMap> LumaGainMapSplitter<T> {
    /// Construct from a curve and config.
    pub fn new(curve: T, cfg: SplitConfig) -> Self {
        Self { curve, cfg }
    }

    /// Borrow the configuration.
    pub fn config(&self) -> &SplitConfig {
        &self.cfg
    }

    /// Borrow the curve.
    pub fn curve(&self) -> &T {
        &self.curve
    }

    /// Encode one row.
    ///
    /// `hdr`: linear-light HDR (length = pixels · `channels`).
    /// `sdr_out`: receives linear SDR; must match `hdr.len()`.
    /// `gain_out`: receives one `log2` gain per pixel; length = pixels.
    /// `stats`: accumulator; updated in place.
    ///
    /// Panics if `channels` is not 3 or 4, or if lengths mismatch.
    pub fn split_row(
        &self,
        hdr: &[f32],
        sdr_out: &mut [f32],
        gain_out: &mut [f32],
        channels: u8,
        stats: &mut SplitStats,
    ) {
        match channels {
            3 => self.split_cn::<3>(hdr, sdr_out, gain_out, stats),
            4 => self.split_cn::<4>(hdr, sdr_out, gain_out, stats),
            _ => panic!("channels must be 3 or 4, got {channels}"),
        }
    }

    /// Decode one row: SDR + log2 gain → HDR.
    ///
    /// Inverse of [`Self::split_row`] (modulo SDR clipping recorded in
    /// [`SplitStats::clipped_sdr_pixels`] and any external u8 quantization
    /// of the gain map).
    pub fn apply_row(&self, sdr: &[f32], gain: &[f32], hdr_out: &mut [f32], channels: u8) {
        match channels {
            3 => self.apply_cn::<3>(sdr, gain, hdr_out),
            4 => self.apply_cn::<4>(sdr, gain, hdr_out),
            _ => panic!("channels must be 3 or 4, got {channels}"),
        }
    }

    #[inline]
    fn split_cn<const CN: usize>(
        &self,
        hdr: &[f32],
        sdr: &mut [f32],
        gain: &mut [f32],
        st: &mut SplitStats,
    ) {
        debug_assert!(CN == 3 || CN == 4);
        assert_eq!(hdr.len(), sdr.len());
        assert_eq!(hdr.len() / CN, gain.len());
        let [wr, wg, wb] = self.cfg.luma_weights;
        let (b, a) = (self.cfg.base_offset, self.cfg.alternate_offset);
        let (lo, hi) = (self.cfg.min_log2, self.cfg.max_log2);
        let alpha = self.cfg.pre_desaturate;
        let has_ct = alpha > 0.0;

        for ((h, s), gp) in hdr
            .chunks_exact(CN)
            .zip(sdr.chunks_exact_mut(CN))
            .zip(gain.iter_mut())
        {
            let r = h[0].max(0.0);
            let gc = h[1].max(0.0);
            let bl = h[2].max(0.0);

            // Optional pre-desaturation (crosstalk matrix).
            let (cr, cg, cb) = if has_ct {
                let d = 1.0 - 2.0 * alpha;
                (
                    d * r + alpha * gc + alpha * bl,
                    alpha * r + d * gc + alpha * bl,
                    alpha * r + alpha * gc + d * bl,
                )
            } else {
                (r, gc, bl)
            };

            let y_hdr = wr * cr + wg * cg + wb * cb;
            let y_sdr = self.curve.map_luma(y_hdr).clamp(0.0, 1.0);

            // Choose gain from the luma ratio. Both offsets prevent 0/0 at black.
            let raw_log2 = ((y_hdr + a) / (y_sdr + b)).log2();
            if raw_log2 < st.observed_min_log2 {
                st.observed_min_log2 = raw_log2;
            }
            if raw_log2 > st.observed_max_log2 {
                st.observed_max_log2 = raw_log2;
            }
            let g_log2 = raw_log2.clamp(lo, hi);
            let m = g_log2.exp2();

            // Per-channel SDR from the (possibly desaturated) HDR channels.
            //   HDR_i = (SDR_i + b) · 2^g − a   ⇒   SDR_i = (HDR_i + a) / 2^g − b
            let d0 = (cr + a) / m - b;
            let d1 = (cg + a) / m - b;
            let d2 = (cb + a) / m - b;

            // Inverse crosstalk to recover original chromaticity.
            let (s0, s1, s2) = if has_ct {
                let inv_a = -alpha / (1.0 - 3.0 * alpha);
                let id = 1.0 - 2.0 * inv_a;
                (
                    id * d0 + inv_a * d1 + inv_a * d2,
                    inv_a * d0 + id * d1 + inv_a * d2,
                    inv_a * d0 + inv_a * d1 + id * d2,
                )
            } else {
                (d0, d1, d2)
            };

            // Clip detection uses a small tolerance so float roundoff in the
            // log2/exp2 round of `m` doesn't get flagged as a real out-of-gamut
            // event. Real out-of-gamut highlights overshoot 1.0 by far more.
            const CLIP_EPS: f32 = 1.0e-4;
            let clipped = s0 < -CLIP_EPS
                || s1 < -CLIP_EPS
                || s2 < -CLIP_EPS
                || s0 > 1.0 + CLIP_EPS
                || s1 > 1.0 + CLIP_EPS
                || s2 > 1.0 + CLIP_EPS;
            if clipped {
                st.clipped_sdr_pixels = st.clipped_sdr_pixels.saturating_add(1);
            }
            s[0] = s0.clamp(0.0, 1.0);
            s[1] = s1.clamp(0.0, 1.0);
            s[2] = s2.clamp(0.0, 1.0);
            if CN == 4 {
                s[3] = h[3];
            }
            *gp = g_log2;
        }
    }

    #[inline]
    fn apply_cn<const CN: usize>(&self, sdr: &[f32], gain: &[f32], hdr: &mut [f32]) {
        debug_assert!(CN == 3 || CN == 4);
        assert_eq!(sdr.len(), hdr.len());
        assert_eq!(sdr.len() / CN, gain.len());
        let (b, a) = (self.cfg.base_offset, self.cfg.alternate_offset);

        for ((s, &g), h) in sdr
            .chunks_exact(CN)
            .zip(gain.iter())
            .zip(hdr.chunks_exact_mut(CN))
        {
            let m = g.exp2();
            h[0] = ((s[0] + b) * m - a).max(0.0);
            h[1] = ((s[1] + b) * m - a).max(0.0);
            h[2] = ((s[2] + b) * m - a).max(0.0);
            if CN == 4 {
                h[3] = s[3];
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;
    use alloc::vec::Vec;

    fn synth_grayscale_hdr_row(pixels: usize, channels: usize, max: f32) -> Vec<f32> {
        let mut row = Vec::with_capacity(pixels * channels);
        for i in 0..pixels {
            let y = (i as f32 / pixels.max(1) as f32) * max;
            row.push(y);
            row.push(y);
            row.push(y);
            if channels == 4 {
                row.push(0.25 + (i as f32 / pixels.max(1) as f32) * 0.5);
            }
        }
        row
    }

    #[test]
    fn hable_filmic_is_monotonic_and_bounded() {
        let c = HableFilmic::new();
        let mut prev = c.map_luma(0.0);
        assert!(prev.abs() < 1e-4, "T(0) = {prev} should be ~0");
        for i in 1..=512 {
            let y = i as f32 / 32.0;
            let s = c.map_luma(y);
            assert!(s.is_finite() && (0.0..=1.0).contains(&s), "T({y}) = {s}");
            assert!(s >= prev - 1.0e-5, "not monotonic at y={y}: {prev} -> {s}");
            prev = s;
        }
    }

    #[test]
    fn round_trip_grayscale_exact() {
        let split = LumaGainMapSplitter::new(
            HableFilmic::new(),
            SplitConfig {
                luma_weights: crate::color::gamut::BT709_LUMA,
                max_log2: 10.0,
                ..Default::default()
            },
        );
        let hdr = synth_grayscale_hdr_row(16, 3, 4.0);
        let mut sdr = vec![0.0; hdr.len()];
        let mut gain = vec![0.0; hdr.len() / 3];
        let mut rec = vec![0.0; hdr.len()];
        let mut stats = SplitStats::default();
        split.split_row(&hdr, &mut sdr, &mut gain, 3, &mut stats);
        split.apply_row(&sdr, &gain, &mut rec, 3);
        assert_eq!(stats.clipped_sdr_pixels, 0, "grayscale should never clip");
        for (a, b) in hdr.iter().zip(&rec) {
            assert!((a - b).abs() < 1e-4, "round-trip drift: {a} vs {b}");
        }
    }

    #[test]
    fn round_trip_rgba4_grayscale_exact() {
        let split = LumaGainMapSplitter::new(
            HableFilmic::new(),
            SplitConfig {
                luma_weights: crate::color::gamut::BT709_LUMA,
                max_log2: 10.0,
                ..Default::default()
            },
        );
        let hdr = synth_grayscale_hdr_row(8, 4, 2.0);
        let mut sdr = vec![0.0; hdr.len()];
        let mut gain = vec![0.0; hdr.len() / 4];
        let mut rec = vec![0.0; hdr.len()];
        let mut stats = SplitStats::default();
        split.split_row(&hdr, &mut sdr, &mut gain, 4, &mut stats);
        split.apply_row(&sdr, &gain, &mut rec, 4);
        assert_eq!(stats.clipped_sdr_pixels, 0);
        for (a, b) in hdr.iter().zip(&rec) {
            assert!((a - b).abs() < 1e-4, "RGBA round-trip drift: {a} vs {b}");
        }
    }

    #[test]
    fn stats_track_gain_extremes() {
        let split = LumaGainMapSplitter::new(HableFilmic::new(), SplitConfig::default());
        let hdr = synth_grayscale_hdr_row(4, 3, 10.0);
        let mut sdr = vec![0.0; hdr.len()];
        let mut gain = vec![0.0; hdr.len() / 3];
        let mut stats = SplitStats::default();
        split.split_row(&hdr, &mut sdr, &mut gain, 3, &mut stats);
        assert!(stats.observed_min_log2.is_finite());
        assert!(stats.observed_max_log2.is_finite());
        assert!(stats.observed_max_log2 >= stats.observed_min_log2);
    }

    #[test]
    fn luma_fn_closure_works() {
        let identity = LumaFn(|y: f32| y.clamp(0.0, 1.0));
        let split = LumaGainMapSplitter::new(identity, SplitConfig::default());
        let hdr: Vec<f32> = vec![0.3, 0.3, 0.3, 0.6, 0.6, 0.6];
        let mut sdr = vec![0.0; 6];
        let mut gain = vec![0.0; 2];
        let mut stats = SplitStats::default();
        split.split_row(&hdr, &mut sdr, &mut gain, 3, &mut stats);
        assert!(stats.observed_max_log2.is_finite());
    }
}
