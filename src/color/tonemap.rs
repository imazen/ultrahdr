//! HDR to SDR tone mapping.
//!
//! Converts HDR content (PQ or HLG) to SDR (sRGB) while preserving
//! as much detail as possible. Based on libultrahdr's tone mapping approach.

use crate::color::gamut::{convert_gamut, rgb_to_luminance, soft_clip_gamut};
use crate::color::transfer::{hlg_eotf, pq_eotf, srgb_oetf};
use crate::types::{ColorGamut, ColorTransfer};

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

/// Simple Reinhard tone mapping operator.
///
/// Maps HDR luminance to SDR range while preserving local contrast.
/// `L_in` is linear luminance (can exceed 1.0 for HDR).
/// `L_max` is the maximum expected luminance.
#[inline]
#[allow(dead_code)]
fn reinhard_tonemap(l_in: f32, l_max: f32) -> f32 {
    // Extended Reinhard: L_out = L_in * (1 + L_in/L_max²) / (1 + L_in)
    let l_max_sq = l_max * l_max;
    l_in * (1.0 + l_in / l_max_sq) / (1.0 + l_in)
}

/// ACES-inspired filmic tone mapping curve.
///
/// Attempt to match the ACES RRT + ODT look with a simpler curve.
/// Input and output are both in `[0, ~10]` range (HDR linear).
#[inline]
fn filmic_tonemap(x: f32) -> f32 {
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
fn bt2390_tonemap(scene_linear: f32, source_peak: f32, target_peak: f32) -> f32 {
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
                    crate::color::transfer::srgb_eotf(encoded_rgb[0]),
                    crate::color::transfer::srgb_eotf(encoded_rgb[1]),
                    crate::color::transfer::srgb_eotf(encoded_rgb[2]),
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
pub fn tonemap_image_to_srgb8(img: &crate::RawImage, target_gamut: crate::ColorGamut) -> Vec<u8> {
    use crate::color::gamut::convert_gamut;
    use crate::color::transfer::{hlg_oetf_inv, pq_eotf, srgb_eotf};
    use crate::PixelFormat;

    let config = ToneMapConfig::default();
    let width = img.width as usize;
    let height = img.height as usize;
    let mut output = vec![0u8; width * height * 4];

    for y in 0..height {
        for x in 0..width {
            // Extract pixel and convert to linear RGB
            let linear_rgb = match img.format {
                PixelFormat::Rgba8 | PixelFormat::Rgb8 => {
                    let bpp = if img.format == PixelFormat::Rgba8 {
                        4
                    } else {
                        3
                    };
                    let idx = y * img.stride as usize + x * bpp;
                    let r = img.data[idx] as f32 / 255.0;
                    let g = img.data[idx + 1] as f32 / 255.0;
                    let b = img.data[idx + 2] as f32 / 255.0;
                    [srgb_eotf(r), srgb_eotf(g), srgb_eotf(b)]
                }
                PixelFormat::Rgba32F => {
                    let idx = y * img.stride as usize + x * 16;
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
                    [r, g, b] // Already linear
                }
                PixelFormat::Rgba16F => {
                    let idx = y * img.stride as usize + x * 8;
                    let r = half::f16::from_le_bytes([img.data[idx], img.data[idx + 1]]).to_f32();
                    let g =
                        half::f16::from_le_bytes([img.data[idx + 2], img.data[idx + 3]]).to_f32();
                    let b =
                        half::f16::from_le_bytes([img.data[idx + 4], img.data[idx + 5]]).to_f32();
                    [r, g, b] // Already linear
                }
                PixelFormat::Rgba1010102Pq => {
                    let idx = y * img.stride as usize + x * 4;
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
                    let idx = y * img.stride as usize + x * 4;
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
                _ => [0.5, 0.5, 0.5], // Fallback for unsupported formats
            };

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
    fn test_pq_tonemap_black_white() {
        let config = ToneMapConfig::default();

        // Black (PQ 0.0) should map to black
        let black = tonemap_pq_to_sdr([0.0, 0.0, 0.0], &config);
        assert!(black[0] < 0.01 && black[1] < 0.01 && black[2] < 0.01);

        // Peak white (PQ 1.0) should map to something bright but not necessarily 1.0
        let white = tonemap_pq_to_sdr([1.0, 1.0, 1.0], &config);
        assert!(white[0] > 0.5); // Should be reasonably bright
    }

    #[test]
    fn test_hlg_tonemap_black_white() {
        let config = ToneMapConfig {
            target_peak_nits: 203.0,
            hdr_peak_nits: 1000.0,
            target_gamut: ColorGamut::Bt709,
            source_gamut: ColorGamut::Bt2100,
        };

        // Black
        let black = tonemap_hlg_to_sdr([0.0, 0.0, 0.0], &config);
        assert!(black[0] < 0.01 && black[1] < 0.01 && black[2] < 0.01);

        // 75% HLG (reference white) - the result depends on tone mapping curve
        // Just verify it's non-zero and not clipped
        let ref_white = tonemap_hlg_to_sdr([0.75, 0.75, 0.75], &config);
        assert!(
            ref_white[0] > 0.0 && ref_white[0] <= 1.0,
            "HLG 75% should produce valid output, got {:?}",
            ref_white
        );
    }

    #[test]
    fn test_tonemap_preserves_neutral() {
        let config = ToneMapConfig::default();

        // Neutral gray should stay neutral (equal RGB)
        let gray_pq = [0.5, 0.5, 0.5];
        let result = tonemap_pq_to_sdr(gray_pq, &config);

        // Allow small differences due to gamut conversion
        let max_diff = (result[0] - result[1])
            .abs()
            .max((result[1] - result[2]).abs())
            .max((result[0] - result[2]).abs());
        assert!(
            max_diff < 0.05,
            "Neutral not preserved: {:?} -> {:?}",
            gray_pq,
            result
        );
    }

    #[test]
    fn test_bt2390_tonemap() {
        // Same peak: pass-through
        let result = bt2390_tonemap(0.5, 1000.0, 1000.0);
        assert!((result - 0.5).abs() < 0.001);

        // Lower target: reduces brightness
        let result = bt2390_tonemap(0.8, 1000.0, 400.0);
        assert!(result < 0.8);

        // Below knee point: linear
        let result = bt2390_tonemap(0.1, 1000.0, 400.0);
        assert!((result - 0.1).abs() < 0.1); // Approximately preserved
    }
}
