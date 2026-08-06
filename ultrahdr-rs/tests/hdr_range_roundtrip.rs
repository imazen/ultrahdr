//! Round-trip reconstruction gates for the gain-map range contract (#33).
//!
//! The encode kernel (`compute_and_encode_gain`) quantizes gain-map bytes on
//! the CONFIG boost grid (`config.min_boost ..= config.max_boost`). The
//! stored per-channel metadata min/max is the grid every conformant reader
//! dequantizes on, so it must declare exactly the range the bytes were
//! quantized on. Before the #33 fix the metadata declared the content's
//! observed (actual) range instead, so whenever the content range was
//! narrower than the config range — almost always with the 10,000-nit
//! default `target_display_peak` — every reader under-reconstructed: a
//! 2000-nit ramp decoded at ~732 nits.
//!
//! These tests encode known-luminance HDR content through both encode paths
//! (HDR-only and SDR+HDR), decode with the crate's own decoder at full gain
//! weight, and assert peak + mid-tone luminance land near the source values.

mod common;

use ultrahdr_rs::{
    ColorPrimaries, Decoder, Encoder, PixelBuffer, PixelFormat, TransferFunction,
    pixel_buffer_from_vec,
};

/// BT.2408 SDR reference white: linear 1.0 in the crate's `LinearFloat`
/// convention (both encode input and decode output).
const SDR_WHITE_NITS: f32 = 203.0;

/// Read the linear-light RGBA f32 pixel at (x, y) from a decoded
/// `PixelFormat::RgbaF32` buffer.
fn sample_rgba_f32(buf: &PixelBuffer, x: u32, y: u32) -> [f32; 4] {
    let slice = buf.as_slice();
    assert_eq!(slice.descriptor().pixel_format(), PixelFormat::RgbaF32);
    let stride = slice.stride();
    let bytes = slice.as_strided_bytes();
    let idx = y as usize * stride + x as usize * 16;
    let mut px = [0.0f32; 4];
    for (c, v) in px.iter_mut().enumerate() {
        let o = idx + c * 4;
        *v = f32::from_le_bytes(bytes[o..o + 4].try_into().unwrap());
    }
    px
}

/// Max linear R value over the whole buffer (peak luminance for the
/// gray-scale fixtures used here).
fn peak_linear(buf: &PixelBuffer) -> f32 {
    let slice = buf.as_slice();
    let mut peak = f32::MIN;
    for y in 0..slice.rows() {
        for x in 0..slice.width() {
            let px = sample_rgba_f32(buf, x, y);
            peak = peak.max(px[0]);
        }
    }
    peak
}

/// Decode `bytes` at full gain-map weight (display boost = 2^alternate
/// headroom, the point where the map is applied at weight 1.0) and return
/// the linear-light HDR buffer (1.0 = SDR white).
fn decode_full_boost(bytes: &[u8]) -> PixelBuffer {
    let decoder = Decoder::new(bytes).expect("decode Ultra HDR container");
    let metadata = decoder.metadata().expect("gain map metadata present");
    let display_boost = 2.0f32
        .powf((metadata.alternate_hdr_headroom as f32).max(0.0))
        .max(1.0);
    decoder.decode_hdr(display_boost).expect("decode HDR")
}

/// A 2000-nit horizontal ramp encoded through the HDR-only path
/// (`set_hdr_image` + `encode()`, all defaults) must reconstruct its peak
/// and mid-tones. Pre-#33-fix this decoded at ~732 nits peak.
#[test]
fn hdr_only_ramp_reconstructs_peak_and_midtone() {
    const PEAK_NITS: f32 = 2000.0;
    let width = 256u32;
    let hdr = common::create_hdr_gradient(width, 64, PEAK_NITS / SDR_WHITE_NITS);

    let mut encoder = Encoder::new();
    encoder.set_hdr_image(hdr);
    let bytes = encoder.encode().expect("HDR-only encode");

    let out = decode_full_boost(&bytes);

    let peak_nits = peak_linear(&out) * SDR_WHITE_NITS;
    eprintln!("HDR-only ramp: decoded peak {peak_nits:.1} nits (source {PEAK_NITS})");
    assert!(
        (1750.0..=2150.0).contains(&peak_nits),
        "HDR-only ramp peak: expected ~{PEAK_NITS} nits, decoded {peak_nits:.1} nits \
         (pre-#33-fix this was ~732: bytes quantized on the config boost range but \
         metadata declared the content's actual range)"
    );

    // Mid-tone: the source pixel at x=128 is t=128/255 -> ~1004 nits.
    let x = 128u32;
    let t = x as f32 / (width - 1) as f32;
    let src_nits = t * PEAK_NITS;
    let mid_nits = sample_rgba_f32(&out, x, 32)[0] * SDR_WHITE_NITS;
    let rel = (mid_nits - src_nits) / src_nits;
    assert!(
        rel.abs() <= 0.15,
        "HDR-only ramp mid-tone at x={x}: source {src_nits:.1} nits, decoded \
         {mid_nits:.1} nits ({:+.1}%)",
        rel * 100.0
    );
}

/// The same ramp through the SDR+HDR path (`set_sdr_image` + `set_hdr_image`)
/// funnels into the same gain-map computation and must reconstruct too. The
/// base is the crate's own tonemap derivation (what the HDR-only path would
/// build internally) so gains stay in the realistic range — the point here
/// is the explicit-SDR code path, not adversarial base content.
#[test]
fn sdr_plus_hdr_ramp_reconstructs_peak() {
    use ultrahdr_core::color::tonemap::tonemap_image_to_srgb8;

    const PEAK_NITS: f32 = 2000.0;
    let width = 256u32;
    let hdr = common::create_hdr_gradient(width, 64, PEAK_NITS / SDR_WHITE_NITS);
    let sdr_pixels = tonemap_image_to_srgb8(&hdr, ColorPrimaries::Bt709).expect("tonemap");
    let sdr = pixel_buffer_from_vec(
        sdr_pixels,
        width,
        64,
        PixelFormat::Rgba8,
        ColorPrimaries::Bt709,
        TransferFunction::Srgb,
    )
    .unwrap();

    let mut encoder = Encoder::new();
    encoder.set_hdr_image(hdr).set_sdr_image(sdr);
    let bytes = encoder.encode().expect("SDR+HDR encode");

    let out = decode_full_boost(&bytes);

    let peak_nits = peak_linear(&out) * SDR_WHITE_NITS;
    eprintln!("SDR+HDR ramp: decoded peak {peak_nits:.1} nits (source {PEAK_NITS})");
    assert!(
        (1750.0..=2150.0).contains(&peak_nits),
        "SDR+HDR ramp peak: expected ~{PEAK_NITS} nits, decoded {peak_nits:.1} nits"
    );

    // Mid-tone at x=128 (~1004 nits source).
    let x = 128u32;
    let src_nits = x as f32 / (width - 1) as f32 * PEAK_NITS;
    let mid_nits = sample_rgba_f32(&out, x, 32)[0] * SDR_WHITE_NITS;
    let rel = (mid_nits - src_nits) / src_nits;
    assert!(
        rel.abs() <= 0.15,
        "SDR+HDR ramp mid-tone at x={x}: source {src_nits:.1} nits, decoded \
         {mid_nits:.1} nits ({:+.1}%)",
        rel * 100.0
    );
}

/// Photo-like HDR scene (shadow texture, SDR-white plateau, sky gradient,
/// and a 2000-nit specular highlight) — a structured image rather than a
/// ramp, through the HDR-only path. Peak, mid-tone, and shadow gates.
#[test]
fn hdr_only_scene_reconstructs_highlight_and_midtone() {
    const SIZE: u32 = 128;
    const HIGHLIGHT_NITS: f32 = 2000.0;
    const SURROUND_NITS: f32 = 600.0;

    // Quadrants: top-left shadow texture (5-50 nits), top-right SDR-white
    // plateau (203 nits), bottom-left sky gradient (203 -> 800 nits),
    // bottom-right 600-nit surround with a 2000-nit disc (radius 14px).
    let mut data = Vec::with_capacity((SIZE * SIZE * 16) as usize);
    for y in 0..SIZE {
        for x in 0..SIZE {
            let half = SIZE / 2;
            let nits = if y < half && x < half {
                // Shadow texture: deterministic sine "grain".
                let g = ((x as f32 * 0.7).sin() * (y as f32 * 0.9).cos()).mul_add(0.5, 0.5);
                5.0 + 45.0 * g
            } else if y < half {
                SDR_WHITE_NITS
            } else if x < half {
                let t = (y - half) as f32 / (half - 1) as f32;
                203.0 + t * (800.0 - 203.0)
            } else {
                let cx = (half + half / 2) as f32;
                let cy = (half + half / 2) as f32;
                let dx = x as f32 - cx;
                let dy = y as f32 - cy;
                if (dx * dx + dy * dy).sqrt() < 14.0 {
                    HIGHLIGHT_NITS
                } else {
                    SURROUND_NITS
                }
            };
            let v = nits / SDR_WHITE_NITS;
            data.extend_from_slice(&v.to_le_bytes());
            data.extend_from_slice(&v.to_le_bytes());
            data.extend_from_slice(&v.to_le_bytes());
            data.extend_from_slice(&1.0f32.to_le_bytes());
        }
    }
    let hdr = pixel_buffer_from_vec(
        data,
        SIZE,
        SIZE,
        PixelFormat::RgbaF32,
        ColorPrimaries::Bt709,
        TransferFunction::Linear,
    )
    .unwrap();

    let mut encoder = Encoder::new();
    encoder.set_hdr_image(hdr);
    let bytes = encoder.encode().expect("HDR-only encode");

    let out = decode_full_boost(&bytes);

    // Highlight disc center (96, 96).
    let disc_nits = sample_rgba_f32(&out, 96, 96)[0] * SDR_WHITE_NITS;
    eprintln!("scene: decoded highlight {disc_nits:.1} nits (source {HIGHLIGHT_NITS})");
    assert!(
        (1700.0..=2250.0).contains(&disc_nits),
        "scene highlight: expected ~{HIGHLIGHT_NITS} nits, decoded {disc_nits:.1} nits"
    );

    // SDR-white plateau center (96, 32): should stay ~203 nits.
    let mid_nits = sample_rgba_f32(&out, 96, 32)[0] * SDR_WHITE_NITS;
    assert!(
        (170.0..=240.0).contains(&mid_nits),
        "scene SDR-white plateau: expected ~203 nits, decoded {mid_nits:.1} nits"
    );

    // Shadow quadrant sample (32, 32): must not be boosted into mid-tones.
    let shadow_nits = sample_rgba_f32(&out, 32, 32)[0] * SDR_WHITE_NITS;
    assert!(
        shadow_nits < 100.0,
        "scene shadow: expected <100 nits, decoded {shadow_nits:.1} nits"
    );
}
