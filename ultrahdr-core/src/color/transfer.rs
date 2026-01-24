//! Electro-optical transfer functions (EOTF/OETF).
//!
//! - OETF: Opto-Electronic Transfer Function (scene linear → encoded)
//! - EOTF: Electro-Optical Transfer Function (encoded → display linear)
//!
//! For camera-captured content, the full chain is:
//! Scene → OETF → Storage → EOTF → Display
//!
//! Reference standards:
//! - sRGB: IEC 61966-2-1
//! - PQ: SMPTE ST 2084, ITU-R BT.2100
//! - HLG: ITU-R BT.2100, ARIB STD-B67

#![allow(clippy::excessive_precision)]

use alloc::boxed::Box;

use crate::types::ColorTransfer;

// ============================================================================
// sRGB Transfer Function (IEC 61966-2-1)
// ============================================================================

/// sRGB OETF: Linear `[0,1]` → sRGB encoded `[0,1]`
#[inline]
pub fn srgb_oetf(linear: f32) -> f32 {
    if linear <= 0.0031308 {
        linear * 12.92
    } else {
        1.055 * linear.powf(1.0 / 2.4) - 0.055
    }
}

/// sRGB EOTF (inverse OETF): sRGB encoded `[0,1]` → Linear `[0,1]`
#[inline]
pub fn srgb_eotf(encoded: f32) -> f32 {
    if encoded <= 0.04045 {
        encoded / 12.92
    } else {
        ((encoded + 0.055) / 1.055).powf(2.4)
    }
}

// ============================================================================
// PQ Transfer Function (SMPTE ST 2084 / ITU-R BT.2100)
// ============================================================================

// PQ constants
const PQ_M1: f32 = 2610.0 / 16384.0; // 0.1593017578125
const PQ_M2: f32 = 2523.0 / 4096.0 * 128.0; // 78.84375
const PQ_C1: f32 = 3424.0 / 4096.0; // 0.8359375
const PQ_C2: f32 = 2413.0 / 4096.0 * 32.0; // 18.8515625
const PQ_C3: f32 = 2392.0 / 4096.0 * 32.0; // 18.6875

/// PQ OETF: Linear `[0,1]` (normalized to 10000 nits) → PQ encoded `[0,1]`
///
/// Input is linear light normalized so that 1.0 = 10000 nits.
#[inline]
pub fn pq_oetf(linear: f32) -> f32 {
    if linear <= 0.0 {
        return 0.0;
    }

    let y = linear.max(0.0);
    let y_m1 = y.powf(PQ_M1);
    let numerator = PQ_C1 + PQ_C2 * y_m1;
    let denominator = 1.0 + PQ_C3 * y_m1;
    (numerator / denominator).powf(PQ_M2)
}

/// PQ EOTF: PQ encoded `[0,1]` → Linear `[0,1]` (normalized to 10000 nits)
///
/// Output is linear light normalized so that 1.0 = 10000 nits.
#[inline]
pub fn pq_eotf(encoded: f32) -> f32 {
    if encoded <= 0.0 {
        return 0.0;
    }

    let e = encoded.max(0.0);
    let e_inv_m2 = e.powf(1.0 / PQ_M2);
    let numerator = (e_inv_m2 - PQ_C1).max(0.0);
    let denominator = PQ_C2 - PQ_C3 * e_inv_m2;

    if denominator <= 0.0 {
        return 0.0;
    }

    (numerator / denominator).powf(1.0 / PQ_M1)
}

/// Convert PQ-normalized linear to absolute nits.
#[inline]
pub fn pq_to_nits(pq_linear: f32) -> f32 {
    pq_linear * 10000.0
}

/// Convert absolute nits to PQ-normalized linear.
#[inline]
pub fn nits_to_pq(nits: f32) -> f32 {
    nits / 10000.0
}

// ============================================================================
// HLG Transfer Function (ITU-R BT.2100 / ARIB STD-B67)
// ============================================================================

// HLG constants
const HLG_A: f32 = 0.17883277;
const HLG_B: f32 = 0.28466892; // 1 - 4*a
const HLG_C: f32 = 0.55991073; // 0.5 - a*ln(4*a)

/// HLG OETF: Scene linear `[0,1]` → HLG encoded `[0,1]`
///
/// Input is scene-referred linear light (not display-referred).
#[inline]
pub fn hlg_oetf(linear: f32) -> f32 {
    if linear <= 0.0 {
        return 0.0;
    }

    let e = linear.max(0.0);
    if e <= 1.0 / 12.0 {
        (3.0 * e).sqrt()
    } else {
        HLG_A * (12.0 * e - HLG_B).ln() + HLG_C
    }
}

/// HLG inverse OETF: HLG encoded `[0,1]` → Scene linear `[0,1]`
#[inline]
pub fn hlg_oetf_inv(encoded: f32) -> f32 {
    if encoded <= 0.0 {
        return 0.0;
    }

    let e = encoded.max(0.0);
    if e <= 0.5 {
        e * e / 3.0
    } else {
        ((e - HLG_C) / HLG_A).exp() / 12.0 + HLG_B / 12.0
    }
}

/// HLG OOTF: Scene linear → Display linear
///
/// The OOTF applies a system gamma that depends on the display peak luminance.
/// For a 1000 nit display, gamma ≈ 1.2.
///
/// `scene_linear`: Scene-referred linear `[0,1]`
/// `display_peak_nits`: Peak luminance of the display (typically 1000 for HLG)
/// Returns: Display-referred linear, scaled to display_peak_nits
#[inline]
pub fn hlg_ootf(scene_linear: f32, display_peak_nits: f32) -> f32 {
    // System gamma calculation per ITU-R BT.2100
    let gamma = 1.2 + 0.42 * (display_peak_nits / 1000.0).log10();
    let gamma = gamma.clamp(1.0, 1.5);

    // Y_s is the scene luminance (we use the input directly for single-channel)
    // For RGB, you'd compute Y_s = 0.2627*R + 0.6780*G + 0.0593*B first
    let y_s = scene_linear;

    // Apply OOTF: L_d = Y_s^(gamma-1) * E * L_w
    // where E is the scene linear value and L_w is the display peak
    y_s.powf(gamma - 1.0) * scene_linear * display_peak_nits
}

/// HLG inverse OOTF: Display linear → Scene linear
#[inline]
pub fn hlg_ootf_inv(display_linear: f32, display_peak_nits: f32) -> f32 {
    if display_linear <= 0.0 || display_peak_nits <= 0.0 {
        return 0.0;
    }

    let gamma = 1.2 + 0.42 * (display_peak_nits / 1000.0).log10();
    let gamma = gamma.clamp(1.0, 1.5);

    // Inverse: E = (L_d / L_w)^(1/gamma)
    (display_linear / display_peak_nits).powf(1.0 / gamma)
}

/// HLG EOTF: HLG encoded `[0,1]` → Display linear (nits)
///
/// This is the combination of inverse OETF and OOTF.
#[inline]
pub fn hlg_eotf(encoded: f32, display_peak_nits: f32) -> f32 {
    let scene_linear = hlg_oetf_inv(encoded);
    hlg_ootf(scene_linear, display_peak_nits)
}

// ============================================================================
// Generic Transfer Function Interface
// ============================================================================

/// Apply OETF (linear → encoded) for the given transfer function.
#[inline]
pub fn apply_oetf(linear: f32, transfer: ColorTransfer) -> f32 {
    match transfer {
        ColorTransfer::Srgb => srgb_oetf(linear),
        ColorTransfer::Linear => linear,
        ColorTransfer::Pq => pq_oetf(linear),
        ColorTransfer::Hlg => hlg_oetf(linear),
    }
}

/// Apply EOTF (encoded → linear) for the given transfer function.
///
/// For HLG, assumes a 1000 nit display and returns normalized linear `[0,1]`.
#[inline]
pub fn apply_eotf(encoded: f32, transfer: ColorTransfer) -> f32 {
    match transfer {
        ColorTransfer::Srgb => srgb_eotf(encoded),
        ColorTransfer::Linear => encoded,
        ColorTransfer::Pq => pq_eotf(encoded),
        ColorTransfer::Hlg => {
            // Return normalized to SDR white (203 nits)
            hlg_eotf(encoded, 1000.0) / 1000.0
        }
    }
}

// ============================================================================
// LUT-accelerated batch operations
// ============================================================================

/// Precomputed LUT for sRGB EOTF (8-bit input).
pub struct SrgbEotfLut {
    table: [f32; 256],
}

impl SrgbEotfLut {
    /// Create a new sRGB EOTF lookup table.
    pub fn new() -> Self {
        let mut table = [0.0f32; 256];
        for (i, entry) in table.iter_mut().enumerate() {
            *entry = srgb_eotf(i as f32 / 255.0);
        }
        Self { table }
    }

    /// Look up the linear value for an 8-bit sRGB encoded value.
    #[inline]
    pub fn lookup(&self, encoded_u8: u8) -> f32 {
        self.table[encoded_u8 as usize]
    }
}

impl Default for SrgbEotfLut {
    fn default() -> Self {
        Self::new()
    }
}

/// Precomputed LUT for PQ EOTF (10-bit input, 1024 entries).
pub struct PqEotfLut {
    table: Box<[f32; 1024]>,
}

impl PqEotfLut {
    /// Create a new PQ EOTF lookup table.
    pub fn new() -> Self {
        let mut table = Box::new([0.0f32; 1024]);
        for (i, entry) in table.iter_mut().enumerate() {
            *entry = pq_eotf(i as f32 / 1023.0);
        }
        Self { table }
    }

    /// Look up the linear value for a 10-bit PQ encoded value.
    #[inline]
    pub fn lookup(&self, encoded_10bit: u16) -> f32 {
        self.table[(encoded_10bit as usize).min(1023)]
    }
}

impl Default for PqEotfLut {
    fn default() -> Self {
        Self::new()
    }
}

/// Precomputed LUT for HLG inverse OETF (10-bit input).
pub struct HlgOetfInvLut {
    table: Box<[f32; 1024]>,
}

impl HlgOetfInvLut {
    /// Create a new HLG inverse OETF lookup table.
    pub fn new() -> Self {
        let mut table = Box::new([0.0f32; 1024]);
        for (i, entry) in table.iter_mut().enumerate() {
            *entry = hlg_oetf_inv(i as f32 / 1023.0);
        }
        Self { table }
    }

    /// Look up the scene linear value for a 10-bit HLG encoded value.
    #[inline]
    pub fn lookup(&self, encoded_10bit: u16) -> f32 {
        self.table[(encoded_10bit as usize).min(1023)]
    }
}

impl Default for HlgOetfInvLut {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const EPSILON: f32 = 1e-4;

    fn approx_eq(a: f32, b: f32) -> bool {
        (a - b).abs() < EPSILON || (a - b).abs() / a.abs().max(b.abs()).max(1e-10) < EPSILON
    }

    #[test]
    fn test_srgb_roundtrip() {
        for i in 0..=100 {
            let linear = i as f32 / 100.0;
            let encoded = srgb_oetf(linear);
            let decoded = srgb_eotf(encoded);
            assert!(
                approx_eq(linear, decoded),
                "sRGB roundtrip failed for {}: got {}",
                linear,
                decoded
            );
        }
    }

    #[test]
    fn test_srgb_known_values() {
        // Black
        assert!(approx_eq(srgb_oetf(0.0), 0.0));
        assert!(approx_eq(srgb_eotf(0.0), 0.0));

        // White
        assert!(approx_eq(srgb_oetf(1.0), 1.0));
        assert!(approx_eq(srgb_eotf(1.0), 1.0));

        // Mid-gray (linear 0.18 → encoded ~0.46)
        let mid_gray_encoded = srgb_oetf(0.18);
        assert!(
            mid_gray_encoded > 0.4 && mid_gray_encoded < 0.5,
            "Mid-gray should encode to ~0.46, got {}",
            mid_gray_encoded
        );
    }

    #[test]
    fn test_pq_roundtrip() {
        for i in 0..=100 {
            let linear = i as f32 / 100.0;
            let encoded = pq_oetf(linear);
            let decoded = pq_eotf(encoded);
            assert!(
                approx_eq(linear, decoded),
                "PQ roundtrip failed for {}: got {}",
                linear,
                decoded
            );
        }
    }

    #[test]
    fn test_pq_known_values() {
        // Black
        assert!(approx_eq(pq_oetf(0.0), 0.0));
        assert!(approx_eq(pq_eotf(0.0), 0.0));

        // Peak white (10000 nits normalized to 1.0)
        assert!(approx_eq(pq_oetf(1.0), 1.0));
        assert!(approx_eq(pq_eotf(1.0), 1.0));

        // SDR white (203 nits = 0.0203 normalized)
        let sdr_white_linear = 203.0 / 10000.0;
        let sdr_white_encoded = pq_oetf(sdr_white_linear);
        // PQ encodes 203 nits to approximately 0.58
        assert!(
            sdr_white_encoded > 0.5 && sdr_white_encoded < 0.65,
            "SDR white should encode to ~0.58, got {}",
            sdr_white_encoded
        );
    }

    #[test]
    fn test_hlg_roundtrip() {
        for i in 0..=100 {
            let linear = i as f32 / 100.0;
            let encoded = hlg_oetf(linear);
            let decoded = hlg_oetf_inv(encoded);
            assert!(
                approx_eq(linear, decoded),
                "HLG OETF roundtrip failed for {}: got {}",
                linear,
                decoded
            );
        }
    }

    #[test]
    fn test_hlg_known_values() {
        // Black
        assert!(approx_eq(hlg_oetf(0.0), 0.0));
        assert!(approx_eq(hlg_oetf_inv(0.0), 0.0));

        // White (scene linear 1.0)
        assert!(approx_eq(hlg_oetf(1.0), 1.0));
        assert!(approx_eq(hlg_oetf_inv(1.0), 1.0));

        // 75% HLG signal corresponds to reference white
        // HLG 0.75 → scene linear ~0.265
        let scene_linear = hlg_oetf_inv(0.75);
        assert!(
            scene_linear > 0.2 && scene_linear < 0.3,
            "HLG 75% should decode to ~0.265, got {}",
            scene_linear
        );
    }

    #[test]
    fn test_lut_matches_direct() {
        let srgb_lut = SrgbEotfLut::new();
        for i in 0..=255u8 {
            let direct = srgb_eotf(i as f32 / 255.0);
            let lut = srgb_lut.lookup(i);
            assert!(
                approx_eq(direct, lut),
                "sRGB LUT mismatch at {}: direct={}, lut={}",
                i,
                direct,
                lut
            );
        }

        let pq_lut = PqEotfLut::new();
        for i in (0..=1023u16).step_by(10) {
            let direct = pq_eotf(i as f32 / 1023.0);
            let lut = pq_lut.lookup(i);
            assert!(
                approx_eq(direct, lut),
                "PQ LUT mismatch at {}: direct={}, lut={}",
                i,
                direct,
                lut
            );
        }
    }
}
