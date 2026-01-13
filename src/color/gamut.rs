//! Color gamut definitions and conversion matrices.
//!
//! Reference primaries and matrices for:
//! - BT.709 / sRGB
//! - Display P3
//! - BT.2100 / BT.2020

// Allow full precision for color matrices - these values come from standards
#![allow(clippy::excessive_precision)]

use crate::types::ColorGamut;

/// 3x3 matrix for color transformations.
#[derive(Debug, Clone, Copy)]
pub struct Matrix3x3(pub [[f32; 3]; 3]);

impl Matrix3x3 {
    /// Identity matrix.
    pub const IDENTITY: Self = Self([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]);

    /// Multiply matrix by RGB vector: [R', G', B'] = M * [R, G, B]
    #[inline]
    pub fn transform(&self, rgb: [f32; 3]) -> [f32; 3] {
        let m = &self.0;
        [
            m[0][0] * rgb[0] + m[0][1] * rgb[1] + m[0][2] * rgb[2],
            m[1][0] * rgb[0] + m[1][1] * rgb[1] + m[1][2] * rgb[2],
            m[2][0] * rgb[0] + m[2][1] * rgb[1] + m[2][2] * rgb[2],
        ]
    }

    /// Matrix multiplication: self * other
    pub fn multiply(&self, other: &Self) -> Self {
        let a = &self.0;
        let b = &other.0;
        let mut result = [[0.0f32; 3]; 3];

        for i in 0..3 {
            for j in 0..3 {
                result[i][j] = a[i][0] * b[0][j] + a[i][1] * b[1][j] + a[i][2] * b[2][j];
            }
        }

        Self(result)
    }
}

// ============================================================================
// RGB to XYZ matrices (D65 illuminant)
// ============================================================================

/// BT.709 / sRGB RGB to XYZ (D65) - IEC 61966-2-1
pub const BT709_TO_XYZ: Matrix3x3 = Matrix3x3([
    [0.4124564, 0.3575761, 0.1804375],
    [0.2126729, 0.7151522, 0.0721750],
    [0.0193339, 0.1191920, 0.9503041],
]);

/// XYZ to BT.709 / sRGB RGB (D65) - computed inverse of BT709_TO_XYZ
pub const XYZ_TO_BT709: Matrix3x3 = Matrix3x3([
    [3.2404542, -1.5371385, -0.4985314],
    [-0.9692660, 1.8760108, 0.0415560],
    [0.0556434, -0.2040259, 1.0572252],
]);

// Note: These matrices may have small numerical errors. For precise conversions,
// consider computing the inverse at runtime or using higher precision constants.

/// Display P3 RGB to XYZ (D65)
pub const P3_TO_XYZ: Matrix3x3 = Matrix3x3([
    [0.4865709, 0.2656677, 0.1982173],
    [0.2289746, 0.6917385, 0.0792869],
    [0.0000000, 0.0451134, 1.0439444],
]);

/// XYZ to Display P3 RGB (D65)
pub const XYZ_TO_P3: Matrix3x3 = Matrix3x3([
    [2.4934969, -0.9313836, -0.4027108],
    [-0.8294890, 1.7626641, 0.0236247],
    [0.0358458, -0.0761724, 0.9568845],
]);

/// BT.2100 / BT.2020 RGB to XYZ (D65)
pub const BT2100_TO_XYZ: Matrix3x3 = Matrix3x3([
    [0.6369580, 0.1446169, 0.1688810],
    [0.2627002, 0.6779981, 0.0593017],
    [0.0000000, 0.0280727, 1.0609851],
]);

/// XYZ to BT.2100 / BT.2020 RGB (D65)
pub const XYZ_TO_BT2100: Matrix3x3 = Matrix3x3([
    [1.7166512, -0.3556708, -0.2533663],
    [-0.6666844, 1.6164812, 0.0157685],
    [0.0176399, -0.0427706, 0.9421031],
]);

// ============================================================================
// Direct gamut-to-gamut conversion matrices
// ============================================================================

/// BT.709 to Display P3
pub const BT709_TO_P3: Matrix3x3 = Matrix3x3([
    [0.8224622, 0.1775378, 0.0000000],
    [0.0331942, 0.9668058, 0.0000000],
    [0.0170826, 0.0723974, 0.9105200],
]);

/// Display P3 to BT.709
pub const P3_TO_BT709: Matrix3x3 = Matrix3x3([
    [1.2249401, -0.2249401, 0.0000000],
    [-0.0420569, 1.0420569, 0.0000000],
    [-0.0196376, -0.0786361, 1.0982737],
]);

/// BT.709 to BT.2100
pub const BT709_TO_BT2100: Matrix3x3 = Matrix3x3([
    [0.6274039, 0.3292831, 0.0433130],
    [0.0690973, 0.9195404, 0.0113623],
    [0.0163914, 0.0880133, 0.8955953],
]);

/// BT.2100 to BT.709
pub const BT2100_TO_BT709: Matrix3x3 = Matrix3x3([
    [1.6604910, -0.5876411, -0.0728499],
    [-0.1245505, 1.1328999, -0.0083494],
    [-0.0181508, -0.1005789, 1.1187297],
]);

/// Display P3 to BT.2100
pub const P3_TO_BT2100: Matrix3x3 = Matrix3x3([
    [0.7530407, 0.1986764, 0.0482829],
    [0.0457456, 0.9419067, 0.0123477],
    [-0.0012122, 0.0176044, 0.9836078],
]);

/// BT.2100 to Display P3
pub const BT2100_TO_P3: Matrix3x3 = Matrix3x3([
    [1.3434102, -0.2821438, -0.0612664],
    [-0.0652531, 1.0757414, -0.0104884],
    [0.0028020, -0.0195966, 1.0167946],
]);

// ============================================================================
// Luminance coefficients
// ============================================================================

/// Luminance coefficients for BT.709 (Y = 0.2126R + 0.7152G + 0.0722B)
pub const BT709_LUMA: [f32; 3] = [0.2126, 0.7152, 0.0722];

/// Luminance coefficients for Display P3 (sum to 1.0)
pub const P3_LUMA: [f32; 3] = [0.2289746, 0.6917385, 0.0792869];

/// Luminance coefficients for BT.2100 / BT.2020
pub const BT2100_LUMA: [f32; 3] = [0.2627, 0.6780, 0.0593];

/// Get luminance coefficients for a color gamut.
pub fn luma_coefficients(gamut: ColorGamut) -> [f32; 3] {
    match gamut {
        ColorGamut::Bt709 => BT709_LUMA,
        ColorGamut::DisplayP3 => P3_LUMA,
        ColorGamut::Bt2100 => BT2100_LUMA,
    }
}

/// Calculate luminance from linear RGB.
#[inline]
pub fn rgb_to_luminance(rgb: [f32; 3], gamut: ColorGamut) -> f32 {
    let coeffs = luma_coefficients(gamut);
    coeffs[0] * rgb[0] + coeffs[1] * rgb[1] + coeffs[2] * rgb[2]
}

// ============================================================================
// Gamut conversion functions
// ============================================================================

/// Get the matrix to convert from source gamut to target gamut.
pub fn gamut_conversion_matrix(from: ColorGamut, to: ColorGamut) -> Matrix3x3 {
    match (from, to) {
        (ColorGamut::Bt709, ColorGamut::Bt709) => Matrix3x3::IDENTITY,
        (ColorGamut::DisplayP3, ColorGamut::DisplayP3) => Matrix3x3::IDENTITY,
        (ColorGamut::Bt2100, ColorGamut::Bt2100) => Matrix3x3::IDENTITY,

        (ColorGamut::Bt709, ColorGamut::DisplayP3) => BT709_TO_P3,
        (ColorGamut::DisplayP3, ColorGamut::Bt709) => P3_TO_BT709,

        (ColorGamut::Bt709, ColorGamut::Bt2100) => BT709_TO_BT2100,
        (ColorGamut::Bt2100, ColorGamut::Bt709) => BT2100_TO_BT709,

        (ColorGamut::DisplayP3, ColorGamut::Bt2100) => P3_TO_BT2100,
        (ColorGamut::Bt2100, ColorGamut::DisplayP3) => BT2100_TO_P3,
    }
}

/// Convert linear RGB from one gamut to another.
#[inline]
pub fn convert_gamut(rgb: [f32; 3], from: ColorGamut, to: ColorGamut) -> [f32; 3] {
    if from == to {
        return rgb;
    }
    gamut_conversion_matrix(from, to).transform(rgb)
}

/// Soft-clip out-of-gamut colors to preserve hue.
///
/// When converting from a wider gamut to a narrower one, some colors may
/// fall outside [0,1]. This function clips them while preserving hue.
#[inline]
pub fn soft_clip_gamut(rgb: [f32; 3]) -> [f32; 3] {
    let max_channel = rgb[0].max(rgb[1]).max(rgb[2]);
    let min_channel = rgb[0].min(rgb[1]).min(rgb[2]);

    // If already in gamut, return as-is
    if max_channel <= 1.0 && min_channel >= 0.0 {
        return rgb;
    }

    // Scale down if any channel exceeds 1.0
    let mut result = rgb;
    if max_channel > 1.0 {
        let scale = 1.0 / max_channel;
        result = [result[0] * scale, result[1] * scale, result[2] * scale];
    }

    // Clip negative values (this can shift hue slightly)
    [result[0].max(0.0), result[1].max(0.0), result[2].max(0.0)]
}

#[cfg(test)]
mod tests {
    use super::*;

    const EPSILON: f32 = 0.02; // Allow 2% error for matrix precision

    fn approx_eq(a: f32, b: f32) -> bool {
        (a - b).abs() < EPSILON
    }

    fn rgb_approx_eq(a: [f32; 3], b: [f32; 3]) -> bool {
        approx_eq(a[0], b[0]) && approx_eq(a[1], b[1]) && approx_eq(a[2], b[2])
    }

    #[test]
    fn test_bt709_roundtrip_via_xyz() {
        let test_colors = [
            [1.0, 0.0, 0.0], // Red
            [0.0, 1.0, 0.0], // Green
            [0.0, 0.0, 1.0], // Blue
            [1.0, 1.0, 1.0], // White
            [0.5, 0.5, 0.5], // Gray
        ];

        for rgb in test_colors {
            let xyz = BT709_TO_XYZ.transform(rgb);
            let back = XYZ_TO_BT709.transform(xyz);
            assert!(
                rgb_approx_eq(rgb, back),
                "BT.709 roundtrip failed: {:?} -> {:?}",
                rgb,
                back
            );
        }
    }

    #[test]
    fn test_p3_roundtrip_via_xyz() {
        let test_colors = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];

        for rgb in test_colors {
            let xyz = P3_TO_XYZ.transform(rgb);
            let back = XYZ_TO_P3.transform(xyz);
            assert!(
                rgb_approx_eq(rgb, back),
                "P3 roundtrip failed: {:?} -> {:?}",
                rgb,
                back
            );
        }
    }

    #[test]
    fn test_bt2100_roundtrip_via_xyz() {
        let test_colors = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];

        for rgb in test_colors {
            let xyz = BT2100_TO_XYZ.transform(rgb);
            let back = XYZ_TO_BT2100.transform(xyz);
            assert!(
                rgb_approx_eq(rgb, back),
                "BT.2100 roundtrip failed: {:?} -> {:?}",
                rgb,
                back
            );
        }
    }

    #[test]
    fn test_gamut_conversion_roundtrip() {
        let rgb = [0.5, 0.3, 0.8];

        // BT.709 -> P3 -> BT.709
        let p3 = convert_gamut(rgb, ColorGamut::Bt709, ColorGamut::DisplayP3);
        let back = convert_gamut(p3, ColorGamut::DisplayP3, ColorGamut::Bt709);
        assert!(
            rgb_approx_eq(rgb, back),
            "709->P3->709 failed: {:?} -> {:?}",
            rgb,
            back
        );

        // BT.709 -> BT.2100 -> BT.709
        let bt2100 = convert_gamut(rgb, ColorGamut::Bt709, ColorGamut::Bt2100);
        let back = convert_gamut(bt2100, ColorGamut::Bt2100, ColorGamut::Bt709);
        assert!(
            rgb_approx_eq(rgb, back),
            "709->2100->709 failed: {:?} -> {:?}",
            rgb,
            back
        );
    }

    #[test]
    fn test_white_preserves_across_gamuts() {
        let white = [1.0, 1.0, 1.0];

        // White should map to white across all gamuts (same D65 white point)
        let p3 = convert_gamut(white, ColorGamut::Bt709, ColorGamut::DisplayP3);
        assert!(
            rgb_approx_eq(white, p3),
            "White not preserved 709->P3: {:?}",
            p3
        );

        let bt2100 = convert_gamut(white, ColorGamut::Bt709, ColorGamut::Bt2100);
        assert!(
            rgb_approx_eq(white, bt2100),
            "White not preserved 709->2100: {:?}",
            bt2100
        );
    }

    #[test]
    fn test_luminance_calculation() {
        // White should have luminance 1.0
        let white = [1.0, 1.0, 1.0];
        assert!(approx_eq(rgb_to_luminance(white, ColorGamut::Bt709), 1.0));
        assert!(approx_eq(
            rgb_to_luminance(white, ColorGamut::DisplayP3),
            1.0
        ));
        assert!(approx_eq(rgb_to_luminance(white, ColorGamut::Bt2100), 1.0));

        // Black should have luminance 0.0
        let black = [0.0, 0.0, 0.0];
        assert!(approx_eq(rgb_to_luminance(black, ColorGamut::Bt709), 0.0));

        // Pure green has different luminance in different gamuts
        let green = [0.0, 1.0, 0.0];
        let lum_709 = rgb_to_luminance(green, ColorGamut::Bt709);
        let lum_2100 = rgb_to_luminance(green, ColorGamut::Bt2100);
        assert!(approx_eq(lum_709, 0.7152)); // BT.709 green coefficient
        assert!(approx_eq(lum_2100, 0.6780)); // BT.2100 green coefficient
    }

    #[test]
    fn test_soft_clip() {
        // In-gamut colors unchanged
        let in_gamut = [0.5, 0.3, 0.8];
        assert!(rgb_approx_eq(in_gamut, soft_clip_gamut(in_gamut)));

        // Over-saturated scaled down
        let over = [1.5, 0.75, 0.3];
        let clipped = soft_clip_gamut(over);
        assert!(clipped[0] <= 1.0);
        assert!(clipped[1] <= 1.0);
        assert!(clipped[2] <= 1.0);
        // Should preserve ratios
        assert!(approx_eq(clipped[1] / clipped[0], over[1] / over[0]));

        // Negative values clipped
        let negative = [-0.1, 0.5, 0.5];
        let clipped = soft_clip_gamut(negative);
        assert!(clipped[0] >= 0.0);
    }
}
