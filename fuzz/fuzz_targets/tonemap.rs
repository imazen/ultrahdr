#![no_main]

use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    // Structured fuzzer for tone mapping and color math.
    // Exercises: reinhard/filmic/bt2390/agx/clamp tonemappers,
    // ToneMapCurve, tonemap_image_to_srgb8, transfer functions, gamut conversions.
    if data.len() < 16 {
        return;
    }

    let op = data[0];
    let remaining = &data[1..];

    match op % 6 {
        0 => {
            // Per-pixel tonemappers
            if remaining.len() < 16 {
                return;
            }
            let x = f32::from_le_bytes([remaining[0], remaining[1], remaining[2], remaining[3]]);
            let y = f32::from_le_bytes([remaining[4], remaining[5], remaining[6], remaining[7]]);
            if !x.is_finite() || !y.is_finite() {
                return;
            }
            let _ = ultrahdr_core::color::tonemap::reinhard_extended(x, y.abs().max(0.001));
            let _ = ultrahdr_core::color::tonemap::filmic_narkowicz(x);
            let _ = ultrahdr_core::color::tonemap::bt2390_tonemap(x, y.abs().max(1.0), 100.0);
            let _ = x.clamp(0.0, 1.0);
        }
        1 => {
            // AGX tonemapper
            if remaining.len() < 12 {
                return;
            }
            let r = f32::from_le_bytes([remaining[0], remaining[1], remaining[2], remaining[3]]);
            let g = f32::from_le_bytes([remaining[4], remaining[5], remaining[6], remaining[7]]);
            let b = f32::from_le_bytes([remaining[8], remaining[9], remaining[10], remaining[11]]);
            if !r.is_finite() || !g.is_finite() || !b.is_finite() {
                return;
            }
            let look = match remaining.get(12).unwrap_or(&0) % 3 {
                0 => ultrahdr_core::color::tonemap::AgxLook::Default,
                1 => ultrahdr_core::color::tonemap::AgxLook::Punchy,
                _ => ultrahdr_core::color::tonemap::AgxLook::Golden,
            };
            let _ = ultrahdr_core::color::tonemap::agx_tonemap([r, g, b], look);
        }
        2 => {
            // Bt2408Tonemapper
            if remaining.len() < 20 {
                return;
            }
            let source_peak = f32::from_le_bytes([remaining[0], remaining[1], remaining[2], remaining[3]]);
            let target_peak = f32::from_le_bytes([remaining[4], remaining[5], remaining[6], remaining[7]]);
            if !source_peak.is_finite() || !target_peak.is_finite()
                || source_peak <= 0.0 || target_peak <= 0.0 {
                return;
            }
            let tm = ultrahdr_core::color::tonemap::Bt2408Tonemapper::new(
                source_peak.min(10000.0),
                target_peak.min(10000.0),
            );
            let r = f32::from_le_bytes([remaining[8], remaining[9], remaining[10], remaining[11]]);
            let g = f32::from_le_bytes([remaining[12], remaining[13], remaining[14], remaining[15]]);
            let b = f32::from_le_bytes([remaining[16], remaining[17], remaining[18], remaining[19]]);
            if r.is_finite() && g.is_finite() && b.is_finite() {
                // Clamp to reasonable range to avoid upstream linear-srgb panic
                let r = r.clamp(0.0, 100.0);
                let g = g.clamp(0.0, 100.0);
                let b = b.clamp(0.0, 100.0);
                use zentone::ToneMap;
                let _ = tm.map_rgb([r, g, b]);
            }
        }
        3 => {
            // tonemap_pq_to_sdr / tonemap_hlg_to_sdr
            if remaining.len() < 12 {
                return;
            }
            let r = f32::from_le_bytes([remaining[0], remaining[1], remaining[2], remaining[3]]);
            let g = f32::from_le_bytes([remaining[4], remaining[5], remaining[6], remaining[7]]);
            let b = f32::from_le_bytes([remaining[8], remaining[9], remaining[10], remaining[11]]);
            if !r.is_finite() || !g.is_finite() || !b.is_finite() {
                return;
            }
            // Clamp to [0, 1] — upstream linear-srgb 0.6.7 panics on extreme values
            let r = r.clamp(0.0, 1.0);
            let g = g.clamp(0.0, 1.0);
            let b = b.clamp(0.0, 1.0);
            let config = ultrahdr_core::color::tonemap::ToneMapConfig {
                target_peak_nits: 203.0,
                hdr_peak_nits: 10000.0,
                target_gamut: ultrahdr_core::ColorPrimaries::Bt709,
                source_gamut: ultrahdr_core::ColorPrimaries::Bt2020,
            };
            let _ = ultrahdr_core::color::tonemap::tonemap_pq_to_sdr([r, g, b], &config);
            let _ = ultrahdr_core::color::tonemap::tonemap_hlg_to_sdr([r, g, b], &config);
        }
        4 => {
            // tonemap_image_to_srgb8 with a small image
            if remaining.len() < 10 {
                return;
            }
            let width = remaining[0].max(1).min(16) as u32;
            let height = remaining[1].max(1).min(16) as u32;
            let fmt_idx = remaining[2] % 2;
            let gamut_idx = remaining[3] % 3;
            let transfer_idx = remaining[4] % 4;

            let format = if fmt_idx == 0 {
                ultrahdr_core::PixelFormat::RgbaF32
            } else {
                ultrahdr_core::PixelFormat::Rgba8
            };
            let gamut = match gamut_idx {
                0 => ultrahdr_core::ColorPrimaries::Bt709,
                1 => ultrahdr_core::ColorPrimaries::DisplayP3,
                _ => ultrahdr_core::ColorPrimaries::Bt2020,
            };
            let transfer = match transfer_idx {
                0 => ultrahdr_core::TransferFunction::Srgb,
                1 => ultrahdr_core::TransferFunction::Linear,
                2 => ultrahdr_core::TransferFunction::Pq,
                _ => ultrahdr_core::TransferFunction::Hlg,
            };

            let pixel_start = 5;
            let bpp = format.bytes_per_pixel();
            let needed = (width as usize) * (height as usize) * bpp;
            if remaining.len() < pixel_start + needed {
                return;
            }

            let mut pixel_data = remaining[pixel_start..pixel_start + needed].to_vec();
            // For RgbaF32, clamp f32 values to avoid upstream linear-srgb panic
            if format == ultrahdr_core::PixelFormat::RgbaF32 {
                for chunk in pixel_data.chunks_exact_mut(4) {
                    let val = f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
                    let clamped = if val.is_finite() { val.clamp(0.0, 10.0) } else { 0.5 };
                    chunk.copy_from_slice(&clamped.to_le_bytes());
                }
            }
            let img = match ultrahdr_core::RawImage::from_data(
                width, height, format, gamut, transfer, pixel_data,
            ) {
                Ok(img) => img,
                Err(_) => return,
            };

            let _ = ultrahdr_core::color::tonemap::tonemap_image_to_srgb8(
                &img,
                ultrahdr_core::ColorPrimaries::Bt709,
            );
        }
        _ => {
            // Transfer function round-trips
            if remaining.len() < 4 {
                return;
            }
            let x = f32::from_le_bytes([remaining[0], remaining[1], remaining[2], remaining[3]]);
            if !x.is_finite() {
                return;
            }
            // Clamp to [0, 1] range — upstream linear-srgb 0.6.7 panics on
            // extreme values due to i32 overflow in fast_pow2f. Fixed locally
            // but not yet published.
            let x = x.clamp(0.0, 1.0);
            let _ = ultrahdr_core::color::transfer::srgb_eotf(x);
            let _ = ultrahdr_core::color::transfer::srgb_oetf(x);
            let _ = ultrahdr_core::color::transfer::pq_eotf(x);
            let _ = ultrahdr_core::color::transfer::pq_oetf(x);
            let _ = ultrahdr_core::color::transfer::hlg_eotf(x, 1000.0);
            let _ = ultrahdr_core::color::transfer::hlg_oetf(x);
        }
    }
});
