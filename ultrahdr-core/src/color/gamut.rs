//! Luma coefficients, gamut conversion wrapper, and hue-preserving soft-clip
//! for ultrahdr-core.
//!
//! Gamut conversion (RGB ↔ RGB across BT.709, DisplayP3, BT.2020/2100)
//! previously lived here as hand-rolled 3×3 matrices plus an `apply_matrix`
//! helper. The matrices themselves now come from zenpixels'
//! [`ColorPrimaries::gamut_matrix_to`], which derives them from chromaticity
//! coordinates with Bradford chromatic adaptation.
//!
//! What stays here:
//! - **Luma coefficients** (BT.709, DisplayP3, BT.2020/2100). `zenpixels`'
//!   [`LumaCoefficients`](zenpixels::LumaCoefficients) enum covers only
//!   BT.601 / BT.709 today and exposes no `[f32; 3]` accessor. Until that's
//!   upstreamed we keep the triples needed by the gain map splitter.
//! - **`convert_gamut`** — thin wrapper over
//!   `ColorPrimaries::gamut_matrix_to` so the existing call sites in
//!   `color/tonemap.rs` don't churn.
//! - **`soft_clip_gamut`** — hue-preserving soft clip for out-of-gamut
//!   highlights used by the ultrahdr-core tone mapper. Not a zenpixels
//!   primitive.

use crate::types::ColorPrimaries;

// ============================================================================
// Luma coefficients
// ============================================================================

/// Luminance coefficients for BT.709 (`Y = 0.2126R + 0.7152G + 0.0722B`).
pub const BT709_LUMA: [f32; 3] = [0.2126, 0.7152, 0.0722];

/// Luminance coefficients for Display P3.
pub const P3_LUMA: [f32; 3] = [0.2289746, 0.6917385, 0.0792869];

/// Luminance coefficients for BT.2100 / BT.2020 (same primaries).
pub const BT2100_LUMA: [f32; 3] = [0.2627, 0.6780, 0.0593];

/// Luma coefficients for a color gamut.
///
/// Unknown/unhandled primaries fall through to BT.709 — matches historical
/// ultrahdr-core behavior where only three gamuts were representable.
pub fn luma_coefficients(gamut: ColorPrimaries) -> [f32; 3] {
    match gamut {
        ColorPrimaries::Bt709 => BT709_LUMA,
        ColorPrimaries::DisplayP3 => P3_LUMA,
        ColorPrimaries::Bt2020 => BT2100_LUMA,
        _ => BT709_LUMA,
    }
}

/// Linear RGB → Y luminance for the given gamut.
#[inline]
pub fn rgb_to_luminance(rgb: [f32; 3], gamut: ColorPrimaries) -> f32 {
    let coeffs = luma_coefficients(gamut);
    coeffs[0] * rgb[0] + coeffs[1] * rgb[1] + coeffs[2] * rgb[2]
}

// ============================================================================
// Gamut conversion (thin wrapper over zenpixels)
// ============================================================================

/// Convert linear RGB from one gamut to another.
///
/// Thin wrapper over [`zenpixels::ColorPrimaries::gamut_matrix_to`] for the
/// callers in ultrahdr-core's tone mapper. Returns the input unchanged when
/// `from == to` or when the primaries are not mutually convertible (e.g., one
/// is `Unknown`).
#[inline]
pub fn convert_gamut(rgb: [f32; 3], from: ColorPrimaries, to: ColorPrimaries) -> [f32; 3] {
    if from == to {
        return rgb;
    }
    match from.gamut_matrix_to(to) {
        Some(m) => [
            m[0][0] * rgb[0] + m[0][1] * rgb[1] + m[0][2] * rgb[2],
            m[1][0] * rgb[0] + m[1][1] * rgb[1] + m[1][2] * rgb[2],
            m[2][0] * rgb[0] + m[2][1] * rgb[1] + m[2][2] * rgb[2],
        ],
        None => rgb,
    }
}

// ============================================================================
// Hue-preserving soft clip
// ============================================================================

/// Hue-preserving soft clip for out-of-gamut highlights.
///
/// Negatives clamp to 0 (handles BT.2020 → BT.709 on saturated colors).
/// For positive over-range, sorts channels by magnitude, clamps the max
/// to 1.0, and linearly interpolates the mid channel to preserve the
/// ratio `(mid - min) / (max - min)` — this keeps hue constant while
/// pulling over-range values back into `[0, 1]`.
#[inline]
pub fn soft_clip_gamut(rgb: [f32; 3]) -> [f32; 3] {
    let [mut r, mut g, mut b] = rgb;

    r = r.max(0.0);
    g = g.max(0.0);
    b = b.max(0.0);

    if r <= 1.0 && g <= 1.0 && b <= 1.0 {
        return [r, g, b];
    }

    if r >= g {
        if g > b {
            clip_sorted(&mut r, &mut g, &mut b);
        } else if b > r {
            clip_sorted(&mut b, &mut r, &mut g);
        } else if b > g {
            clip_sorted(&mut r, &mut b, &mut g);
        } else {
            r = r.min(1.0);
            g = g.min(1.0);
        }
    } else if r >= b {
        clip_sorted(&mut g, &mut r, &mut b);
    } else if b > g {
        clip_sorted(&mut b, &mut g, &mut r);
    } else {
        clip_sorted(&mut g, &mut b, &mut r);
    }

    [r, g, b]
}

/// Helper for `soft_clip_gamut` — clamp `max` to 1.0 and rescale `mid` to
/// preserve `(mid - min) / (max - min)`.
#[inline]
fn clip_sorted(max: &mut f32, mid: &mut f32, min: &mut f32) {
    if *max <= 1.0 {
        return;
    }
    let span = *max - *min;
    if span > 0.0 {
        let t = (*mid - *min) / span;
        *mid = *min + t * (1.0 - *min);
    }
    *max = 1.0;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn luma_coefficients_known() {
        assert_eq!(luma_coefficients(ColorPrimaries::Bt709), BT709_LUMA);
        assert_eq!(luma_coefficients(ColorPrimaries::DisplayP3), P3_LUMA);
        assert_eq!(luma_coefficients(ColorPrimaries::Bt2020), BT2100_LUMA);
    }

    #[test]
    fn rgb_to_luminance_white() {
        let l = rgb_to_luminance([1.0, 1.0, 1.0], ColorPrimaries::Bt709);
        assert!((l - 1.0).abs() < 1e-5, "BT.709 white = 1.0, got {l}");
        let l = rgb_to_luminance([1.0, 1.0, 1.0], ColorPrimaries::Bt2020);
        assert!((l - 1.0).abs() < 1e-5, "BT.2020 white = 1.0, got {l}");
    }

    #[test]
    fn convert_gamut_identity() {
        let rgb = [0.3, 0.6, 0.9];
        assert_eq!(
            convert_gamut(rgb, ColorPrimaries::Bt709, ColorPrimaries::Bt709),
            rgb
        );
    }

    #[test]
    fn convert_gamut_bt709_to_bt2020_to_bt709_roundtrip() {
        let rgb = [0.3, 0.6, 0.9];
        let wide = convert_gamut(rgb, ColorPrimaries::Bt709, ColorPrimaries::Bt2020);
        let back = convert_gamut(wide, ColorPrimaries::Bt2020, ColorPrimaries::Bt709);
        for i in 0..3 {
            assert!(
                (rgb[i] - back[i]).abs() < 0.01,
                "BT.709↔BT.2020 roundtrip drift at {i}: {} vs {}",
                rgb[i],
                back[i]
            );
        }
    }

    #[test]
    fn soft_clip_passes_in_gamut() {
        let rgb = [0.3, 0.6, 0.9];
        assert_eq!(soft_clip_gamut(rgb), rgb);
    }

    #[test]
    fn soft_clip_clamps_negative() {
        let rgb = [-0.1, 0.5, 0.8];
        let out = soft_clip_gamut(rgb);
        assert_eq!(out[0], 0.0);
        assert_eq!(out[1], 0.5);
        assert_eq!(out[2], 0.8);
    }

    #[test]
    fn soft_clip_caps_overrange() {
        let rgb = [2.0, 1.0, 0.0];
        let out = soft_clip_gamut(rgb);
        assert!(out[0] <= 1.0 && out[0] > 0.0);
        assert!(out[1] <= 1.0 && out[1] >= 0.0);
        assert_eq!(out[2], 0.0);
    }
}
