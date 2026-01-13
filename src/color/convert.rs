//! RGB ↔ YUV color space conversions.
//!
//! Supports different YUV coefficient sets:
//! - BT.601 (legacy SD)
//! - BT.709 (HD)
//! - BT.2020 (UHD/HDR)

use crate::types::ColorGamut;

/// YUV coefficients for different standards.
#[derive(Debug, Clone, Copy)]
pub struct YuvCoefficients {
    /// Kr coefficient (red contribution to Y)
    pub kr: f32,
    /// Kb coefficient (blue contribution to Y)
    pub kb: f32,
}

impl YuvCoefficients {
    /// BT.601 coefficients (legacy SD video)
    pub const BT601: Self = Self {
        kr: 0.299,
        kb: 0.114,
    };

    /// BT.709 coefficients (HD video, matches BT.709 primaries)
    pub const BT709: Self = Self {
        kr: 0.2126,
        kb: 0.0722,
    };

    /// BT.2020 coefficients (UHD/HDR video, matches BT.2100 primaries)
    pub const BT2020: Self = Self {
        kr: 0.2627,
        kb: 0.0593,
    };

    /// Get Kg (green contribution to Y = 1 - Kr - Kb)
    #[inline]
    pub fn kg(&self) -> f32 {
        1.0 - self.kr - self.kb
    }

    /// Get coefficients for a color gamut.
    pub fn for_gamut(gamut: ColorGamut) -> Self {
        match gamut {
            ColorGamut::Bt709 => Self::BT709,
            ColorGamut::DisplayP3 => Self::BT709, // P3 typically uses BT.709 matrix
            ColorGamut::Bt2100 => Self::BT2020,
        }
    }
}

// ============================================================================
// Full-range RGB ↔ YCbCr conversions (float)
// ============================================================================

/// Convert linear RGB [0,1] to YCbCr.
///
/// Y is in [0, 1], Cb and Cr are in [-0.5, 0.5].
/// Uses full-range encoding (not limited range 16-235).
#[inline]
pub fn rgb_to_ycbcr(rgb: [f32; 3], coeffs: YuvCoefficients) -> [f32; 3] {
    let [r, g, b] = rgb;

    // Y = Kr*R + Kg*G + Kb*B
    let y = coeffs.kr * r + coeffs.kg() * g + coeffs.kb * b;

    // Cb = (B - Y) / (2 * (1 - Kb))
    let cb = (b - y) / (2.0 * (1.0 - coeffs.kb));

    // Cr = (R - Y) / (2 * (1 - Kr))
    let cr = (r - y) / (2.0 * (1.0 - coeffs.kr));

    [y, cb, cr]
}

/// Convert YCbCr to linear RGB [0,1].
#[inline]
pub fn ycbcr_to_rgb(ycbcr: [f32; 3], coeffs: YuvCoefficients) -> [f32; 3] {
    let [y, cb, cr] = ycbcr;

    // R = Y + Cr * 2 * (1 - Kr)
    let r = y + cr * 2.0 * (1.0 - coeffs.kr);

    // B = Y + Cb * 2 * (1 - Kb)
    let b = y + cb * 2.0 * (1.0 - coeffs.kb);

    // G = (Y - Kr*R - Kb*B) / Kg
    let g = (y - coeffs.kr * r - coeffs.kb * b) / coeffs.kg();

    [r, g, b]
}

// ============================================================================
// 8-bit integer conversions (full range)
// ============================================================================

/// Convert 8-bit sRGB [0,255] to YCbCr with 8-bit output.
///
/// Y is in [0, 255], Cb and Cr are in [0, 255] (128 = neutral).
#[inline]
pub fn rgb8_to_ycbcr8(rgb: [u8; 3], coeffs: YuvCoefficients) -> [u8; 3] {
    let r = rgb[0] as f32 / 255.0;
    let g = rgb[1] as f32 / 255.0;
    let b = rgb[2] as f32 / 255.0;

    let [y, cb, cr] = rgb_to_ycbcr([r, g, b], coeffs);

    // Convert to 8-bit: Y stays [0,255], Cb/Cr shifted to [0,255]
    let y_out = (y * 255.0).round().clamp(0.0, 255.0) as u8;
    let cb_out = ((cb + 0.5) * 255.0).round().clamp(0.0, 255.0) as u8;
    let cr_out = ((cr + 0.5) * 255.0).round().clamp(0.0, 255.0) as u8;

    [y_out, cb_out, cr_out]
}

/// Convert 8-bit YCbCr [0,255] to RGB.
#[inline]
pub fn ycbcr8_to_rgb8(ycbcr: [u8; 3], coeffs: YuvCoefficients) -> [u8; 3] {
    let y = ycbcr[0] as f32 / 255.0;
    let cb = ycbcr[1] as f32 / 255.0 - 0.5;
    let cr = ycbcr[2] as f32 / 255.0 - 0.5;

    let [r, g, b] = ycbcr_to_rgb([y, cb, cr], coeffs);

    [
        (r * 255.0).round().clamp(0.0, 255.0) as u8,
        (g * 255.0).round().clamp(0.0, 255.0) as u8,
        (b * 255.0).round().clamp(0.0, 255.0) as u8,
    ]
}

// ============================================================================
// 10-bit integer conversions (for P010/HDR)
// ============================================================================

/// Convert 10-bit YCbCr [0,1023] to linear RGB [0,1].
///
/// Assumes full-range encoding (not limited range).
#[inline]
pub fn ycbcr10_to_rgb(ycbcr: [u16; 3], coeffs: YuvCoefficients) -> [f32; 3] {
    let y = ycbcr[0] as f32 / 1023.0;
    let cb = ycbcr[1] as f32 / 1023.0 - 0.5;
    let cr = ycbcr[2] as f32 / 1023.0 - 0.5;

    ycbcr_to_rgb([y, cb, cr], coeffs)
}

/// Convert linear RGB [0,1] to 10-bit YCbCr [0,1023].
#[inline]
pub fn rgb_to_ycbcr10(rgb: [f32; 3], coeffs: YuvCoefficients) -> [u16; 3] {
    let [y, cb, cr] = rgb_to_ycbcr(rgb, coeffs);

    [
        (y * 1023.0).round().clamp(0.0, 1023.0) as u16,
        ((cb + 0.5) * 1023.0).round().clamp(0.0, 1023.0) as u16,
        ((cr + 0.5) * 1023.0).round().clamp(0.0, 1023.0) as u16,
    ]
}

// ============================================================================
// Limited range conversions (ITU-R BT.601/709/2020)
// ============================================================================

/// Convert full-range Y [0,1] to limited-range 8-bit [16,235].
#[inline]
pub fn y_full_to_limited_8(y: f32) -> u8 {
    let limited = 16.0 + y * (235.0 - 16.0);
    limited.round().clamp(16.0, 235.0) as u8
}

/// Convert limited-range 8-bit Y [16,235] to full-range [0,1].
#[inline]
pub fn y_limited_to_full_8(y: u8) -> f32 {
    ((y as f32) - 16.0) / (235.0 - 16.0)
}

/// Convert full-range CbCr [-0.5,0.5] to limited-range 8-bit [16,240].
#[inline]
pub fn cbcr_full_to_limited_8(c: f32) -> u8 {
    let limited = 128.0 + c * (240.0 - 16.0);
    limited.round().clamp(16.0, 240.0) as u8
}

/// Convert limited-range 8-bit CbCr [16,240] to full-range [-0.5,0.5].
#[inline]
pub fn cbcr_limited_to_full_8(c: u8) -> f32 {
    ((c as f32) - 128.0) / (240.0 - 16.0)
}

// ============================================================================
// Pixel format specific conversions
// ============================================================================

/// Unpack a 10-bit RGBA1010102 pixel to linear RGB + alpha.
///
/// Format: R(10) G(10) B(10) A(2), packed in little-endian u32.
#[inline]
pub fn unpack_rgba1010102(packed: u32) -> [f32; 4] {
    let r = (packed & 0x3FF) as f32 / 1023.0;
    let g = ((packed >> 10) & 0x3FF) as f32 / 1023.0;
    let b = ((packed >> 20) & 0x3FF) as f32 / 1023.0;
    let a = ((packed >> 30) & 0x3) as f32 / 3.0;
    [r, g, b, a]
}

/// Pack linear RGB + alpha to 10-bit RGBA1010102.
#[inline]
pub fn pack_rgba1010102(rgba: [f32; 4]) -> u32 {
    let r = (rgba[0] * 1023.0).round().clamp(0.0, 1023.0) as u32;
    let g = (rgba[1] * 1023.0).round().clamp(0.0, 1023.0) as u32;
    let b = (rgba[2] * 1023.0).round().clamp(0.0, 1023.0) as u32;
    let a = (rgba[3] * 3.0).round().clamp(0.0, 3.0) as u32;
    r | (g << 10) | (b << 20) | (a << 30)
}

/// Unpack P010 Y sample (16-bit with 10 bits of data in upper bits).
#[inline]
pub fn unpack_p010_y(y16: u16) -> f32 {
    // P010 stores 10-bit data in upper 10 bits, lower 6 bits are 0 or padding
    (y16 >> 6) as f32 / 1023.0
}

/// Unpack P010 UV sample pair (16-bit each with 10 bits of data).
#[inline]
pub fn unpack_p010_uv(u16_val: u16, v16_val: u16) -> (f32, f32) {
    let u = (u16_val >> 6) as f32 / 1023.0 - 0.5;
    let v = (v16_val >> 6) as f32 / 1023.0 - 0.5;
    (u, v)
}

#[cfg(test)]
mod tests {
    use super::*;

    const EPSILON: f32 = 0.01; // Allow 1% error for integer roundtrip

    fn approx_eq(a: f32, b: f32) -> bool {
        (a - b).abs() < EPSILON
    }

    #[test]
    fn test_rgb_ycbcr_roundtrip_bt709() {
        let test_colors = [
            [1.0, 0.0, 0.0], // Red
            [0.0, 1.0, 0.0], // Green
            [0.0, 0.0, 1.0], // Blue
            [1.0, 1.0, 1.0], // White
            [0.0, 0.0, 0.0], // Black
            [0.5, 0.5, 0.5], // Gray
            [0.2, 0.4, 0.8], // Random
        ];

        for rgb in test_colors {
            let ycbcr = rgb_to_ycbcr(rgb, YuvCoefficients::BT709);
            let back = ycbcr_to_rgb(ycbcr, YuvCoefficients::BT709);

            assert!(
                approx_eq(rgb[0], back[0])
                    && approx_eq(rgb[1], back[1])
                    && approx_eq(rgb[2], back[2]),
                "BT.709 roundtrip failed: {:?} -> {:?} -> {:?}",
                rgb,
                ycbcr,
                back
            );
        }
    }

    #[test]
    fn test_rgb_ycbcr_roundtrip_bt2020() {
        let test_colors = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];

        for rgb in test_colors {
            let ycbcr = rgb_to_ycbcr(rgb, YuvCoefficients::BT2020);
            let back = ycbcr_to_rgb(ycbcr, YuvCoefficients::BT2020);

            assert!(
                approx_eq(rgb[0], back[0])
                    && approx_eq(rgb[1], back[1])
                    && approx_eq(rgb[2], back[2]),
                "BT.2020 roundtrip failed: {:?} -> {:?}",
                rgb,
                back
            );
        }
    }

    #[test]
    fn test_ycbcr_known_values() {
        // White: Y=1, Cb=Cr=0
        let white = [1.0, 1.0, 1.0];
        let ycbcr = rgb_to_ycbcr(white, YuvCoefficients::BT709);
        assert!(approx_eq(ycbcr[0], 1.0)); // Y
        assert!(approx_eq(ycbcr[1], 0.0)); // Cb
        assert!(approx_eq(ycbcr[2], 0.0)); // Cr

        // Black: Y=0, Cb=Cr=0
        let black = [0.0, 0.0, 0.0];
        let ycbcr = rgb_to_ycbcr(black, YuvCoefficients::BT709);
        assert!(approx_eq(ycbcr[0], 0.0));
        assert!(approx_eq(ycbcr[1], 0.0));
        assert!(approx_eq(ycbcr[2], 0.0));

        // Pure blue: high Cb, low Cr
        let blue = [0.0, 0.0, 1.0];
        let ycbcr = rgb_to_ycbcr(blue, YuvCoefficients::BT709);
        assert!(ycbcr[1] > 0.0); // Cb should be positive
        assert!(ycbcr[2] < 0.0); // Cr should be negative
    }

    #[test]
    fn test_8bit_roundtrip() {
        let test_colors: [[u8; 3]; 5] = [
            [255, 0, 0],
            [0, 255, 0],
            [0, 0, 255],
            [128, 128, 128],
            [64, 192, 128],
        ];

        for rgb in test_colors {
            let ycbcr = rgb8_to_ycbcr8(rgb, YuvCoefficients::BT709);
            let back = ycbcr8_to_rgb8(ycbcr, YuvCoefficients::BT709);

            // Allow ±2 for rounding errors
            assert!(
                (rgb[0] as i16 - back[0] as i16).abs() <= 2
                    && (rgb[1] as i16 - back[1] as i16).abs() <= 2
                    && (rgb[2] as i16 - back[2] as i16).abs() <= 2,
                "8-bit roundtrip failed: {:?} -> {:?} -> {:?}",
                rgb,
                ycbcr,
                back
            );
        }
    }

    #[test]
    fn test_rgba1010102_roundtrip() {
        let test_values = [
            [0.0, 0.0, 0.0, 0.0],
            [1.0, 1.0, 1.0, 1.0],
            [0.5, 0.25, 0.75, 0.333],
            [1.0, 0.0, 0.0, 0.0],
        ];

        for rgba in test_values {
            let packed = pack_rgba1010102(rgba);
            let unpacked = unpack_rgba1010102(packed);

            // 10-bit precision for RGB, 2-bit for alpha
            assert!(
                approx_eq(rgba[0], unpacked[0])
                    && approx_eq(rgba[1], unpacked[1])
                    && approx_eq(rgba[2], unpacked[2]),
                "RGBA1010102 roundtrip failed: {:?} -> {:?}",
                rgba,
                unpacked
            );
        }
    }

    #[test]
    fn test_limited_range() {
        // Black: Y=0 -> limited 16
        assert_eq!(y_full_to_limited_8(0.0), 16);
        // White: Y=1 -> limited 235
        assert_eq!(y_full_to_limited_8(1.0), 235);

        // Roundtrip
        for y in [0.0f32, 0.25, 0.5, 0.75, 1.0] {
            let limited = y_full_to_limited_8(y);
            let back = y_limited_to_full_8(limited);
            assert!(
                approx_eq(y, back),
                "Limited range roundtrip failed: {} -> {} -> {}",
                y,
                limited,
                back
            );
        }
    }
}
