//! Fuzz crash regression suite.
//!
//! Replays every seed in the repo-root `fuzz/regression/` through every fuzz
//! target's entry point, as an ordinary `cargo test` — no nightly toolchain and
//! no cargo-fuzz needed. Each seed is a previously-found crash that has been
//! fixed; a panic here means one of them came back.
//!
//! Until 2026-08-29 this file did not exist, while `.github/workflows/fuzz.yml`
//! ran `cargo test --test fuzz_regression 2>/dev/null || echo "No regression
//! test found…"` — so the job reported green over a target that was never
//! there, and the two committed seeds were never replayed by CI at all.
//!
//! # Coverage
//!
//! All seven targets under `fuzz/fuzz_targets/` are mirrored:
//!
//! | target | entry point |
//! |---|---|
//! | `decode` | `ultrahdr_rs::Decoder` |
//! | `parse_jpeg_segments` | `ultrahdr_rs::jpeg::parse_jpeg_segments`, `ultrahdr_rs::container::scan_segments` |
//! | `parse_xmp` | `zenjpeg::container::xmp::{parse_xmp, parse_xmp_full}` |
//! | `parse_mpf` | `zenjpeg::container::mpf::{parse_mpf, parse_mpf_segment}` |
//! | `parse_iso21496` | `zencodec::gainmap::parse_iso21496_fmt` (all three formats) |
//! | `apply_gainmap` | `ultrahdr_core::gainmap::apply::apply_gainmap` |
//! | `tonemap` | `ultrahdr_core::color::{tonemap, transfer}` |
//!
//! The two structured targets (`apply_gainmap`, `tonemap`) decode their
//! parameters out of the leading bytes. Their decoding is transcribed here from
//! the fuzz targets so a seed lands on the same cell it originally crashed;
//! keep the two in sync when either changes.
//!
//! To add a seed: drop the (preferably minimized) crash file into
//! `fuzz/regression/` and raise `MIN_SEEDS`.

use std::path::{Path, PathBuf};
use zenutils_fuzz::RegressionSuite;

/// Lower bound on the replayable seed corpus committed under `fuzz/regression/`.
///
/// `RegressionSuite` treats a missing or empty seed directory as a clean no-op,
/// so an emptied, renamed, or never-checked-out corpus would let this test pass
/// without replaying a single seed — the same "green over nothing" failure this
/// file was added to remove. The corpus also lives outside this crate (it
/// belongs to the root `fuzz/` cargo-fuzz workspace), so a layout change would
/// strand the path with no compile error.
///
/// Raise this when seeds are added; only lower it when deleting seeds on purpose.
const MIN_SEEDS: usize = 2;

/// The repo-root `fuzz/regression/`, which is one level above this crate.
///
/// `.parent()` rather than `.join("..")`: the latter leaves a literal `..`
/// component that sandboxed path resolution (WASI) refuses to traverse.
fn regression_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("crate directory always has a parent")
        .join("fuzz")
        .join("regression")
}

/// Count the files `RegressionSuite::run` will actually replay, using its own
/// filters: recurse into subdirectories, skip dotfiles, `*.md` and `*.txt`.
fn replayable_seeds(dir: &Path) -> usize {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return 0;
    };
    let mut found = 0;
    for entry in entries.flatten() {
        let path = entry.path();
        let Some(name) = path.file_name().and_then(|s| s.to_str()) else {
            continue;
        };
        if name.starts_with('.') {
            continue;
        }
        if path.is_dir() {
            found += replayable_seeds(&path);
        } else if path.is_file() {
            let lower = name.to_ascii_lowercase();
            if !lower.ends_with(".md") && !lower.ends_with(".txt") {
                found += 1;
            }
        }
    }
    found
}

/// Mirror of `fuzz/fuzz_targets/decode.rs`.
fn run_decode(data: &[u8]) {
    if let Ok(decoder) = ultrahdr_rs::Decoder::new(data) {
        let _ = decoder.is_ultrahdr();
        let _ = decoder.metadata();
        let _ = decoder.primary_jpeg();
        let _ = decoder.gainmap_jpeg();
        let _ = decoder.icc_profile();
    }
}

/// Mirror of `fuzz/fuzz_targets/parse_jpeg_segments.rs`.
fn run_parse_jpeg_segments(data: &[u8]) {
    let _ = ultrahdr_rs::jpeg::parse_jpeg_segments(data);
    let _ = ultrahdr_rs::container::scan_segments(data);
}

/// Mirror of `fuzz/fuzz_targets/parse_xmp.rs`.
fn run_parse_xmp(data: &[u8]) {
    if let Ok(xmp_str) = std::str::from_utf8(data) {
        let _ = zenjpeg::container::xmp::parse_xmp(xmp_str);
        let _ = zenjpeg::container::xmp::parse_xmp_full(xmp_str);
    }
}

/// Mirror of `fuzz/fuzz_targets/parse_mpf.rs`.
fn run_parse_mpf(data: &[u8]) {
    let _ = zenjpeg::container::mpf::parse_mpf(data);
    let _ = zenjpeg::container::mpf::parse_mpf_segment(data, 0);
}

/// Mirror of `fuzz/fuzz_targets/parse_iso21496.rs`.
fn run_parse_iso21496(data: &[u8]) {
    let _ = zencodec::gainmap::parse_iso21496_fmt(data, zencodec::Iso21496Format::AvifTmap);
    let _ = zencodec::gainmap::parse_iso21496_fmt(data, zencodec::Iso21496Format::JxlJhgm);
    let _ =
        zencodec::gainmap::parse_iso21496_fmt(data, zencodec::Iso21496Format::JpegApp2BodyWithUrn);
}

/// Mirror of `fuzz/fuzz_targets/apply_gainmap.rs` — parameter decoding included,
/// so a seed reaches the same gain-map cell it originally crashed on.
fn run_apply_gainmap(data: &[u8]) {
    if data.len() < 48 {
        return;
    }

    let width = u16::from_le_bytes([data[0], data[1]]).clamp(1, 64) as u32;
    let height = u16::from_le_bytes([data[2], data[3]]).clamp(1, 64) as u32;
    let gm_width = u16::from_le_bytes([data[4], data[5]]).clamp(1, 64) as u32;
    let gm_height = u16::from_le_bytes([data[6], data[7]]).clamp(1, 64) as u32;

    let display_boost = f32::from_bits(u32::from_le_bytes([data[8], data[9], data[10], data[11]]));
    if !display_boost.is_finite() || display_boost < 0.0 {
        return;
    }

    let ch = ultrahdr_core::GainMapChannel {
        max: (data[12] as f64 - 128.0) / 32.0,
        min: (data[13] as f64 - 128.0) / 32.0,
        gamma: (data[14] as f64) / 128.0 + 0.01,
        base_offset: (data[15] as f64) / 255.0,
        alternate_offset: (data[16] as f64) / 255.0,
    };
    // `GainMapMetadata` is `#[non_exhaustive]`, so it cannot be built with a
    // struct expression from outside its crate — default-then-assign, exactly
    // as `fuzz/fuzz_targets/apply_gainmap.rs` does.
    let mut metadata = ultrahdr_core::GainMapMetadata::default();
    metadata.channels = [ch; 3];
    metadata.base_hdr_headroom = (data[17] as f64 - 128.0) / 32.0;
    metadata.alternate_hdr_headroom = (data[18] as f64 - 128.0) / 32.0;
    let channels = if data[19] & 1 == 0 { 1u8 } else { 3u8 };
    let output_format_idx = data[20] & 1;
    let sdr_format_idx = data[21] % 2;
    metadata.use_base_color_space = data[22] & 1 != 0;
    metadata.backward_direction = data[22] & 2 != 0;

    let output_format = if output_format_idx == 0 {
        ultrahdr_core::gainmap::apply::HdrOutputFormat::LinearFloat
    } else {
        ultrahdr_core::gainmap::apply::HdrOutputFormat::Srgb8
    };

    let sdr_format = if sdr_format_idx == 0 {
        ultrahdr_core::PixelFormat::Rgba8
    } else {
        ultrahdr_core::PixelFormat::Rgb8
    };

    let pixel_data_start = 24;
    let bpp = sdr_format.bytes_per_pixel();
    let sdr_size = (width as usize) * (height as usize) * bpp;
    let gm_size = (gm_width as usize) * (gm_height as usize) * (channels as usize);

    if data.len() < pixel_data_start + sdr_size + gm_size {
        return;
    }

    let sdr_data = data[pixel_data_start..pixel_data_start + sdr_size].to_vec();
    let gm_data = data[pixel_data_start + sdr_size..pixel_data_start + sdr_size + gm_size].to_vec();

    let Ok(sdr) = ultrahdr_core::pixel_buffer_from_vec(
        sdr_data,
        width,
        height,
        sdr_format,
        ultrahdr_core::ColorPrimaries::Bt709,
        ultrahdr_core::TransferFunction::Srgb,
    ) else {
        return;
    };

    let gainmap = ultrahdr_core::GainMap {
        width: gm_width,
        height: gm_height,
        channels,
        data: gm_data,
    };

    let _ = ultrahdr_core::gainmap::apply::apply_gainmap(
        &sdr,
        &gainmap,
        &metadata,
        display_boost,
        output_format,
        ultrahdr_core::Unstoppable,
    );
}

/// Mirror of `fuzz/fuzz_targets/tonemap.rs`.
///
/// The target dispatches on `data[0] % 6`; a replayed seed only exercises its
/// own arm, so this runs every arm with the same remaining bytes to keep the
/// replay independent of which arm the seed's first byte happens to select.
fn run_tonemap(data: &[u8]) {
    if data.len() < 16 {
        return;
    }
    let remaining = &data[1..];
    for op in 0u8..6 {
        run_tonemap_arm(op, remaining);
    }
}

fn run_tonemap_arm(op: u8, remaining: &[u8]) {
    match op {
        0 => {
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
        }
        1 => {
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
            if remaining.len() < 20 {
                return;
            }
            let source_peak =
                f32::from_le_bytes([remaining[0], remaining[1], remaining[2], remaining[3]]);
            let target_peak =
                f32::from_le_bytes([remaining[4], remaining[5], remaining[6], remaining[7]]);
            if !source_peak.is_finite()
                || !target_peak.is_finite()
                || source_peak <= 0.0
                || target_peak <= 0.0
            {
                return;
            }
            let tm = ultrahdr_core::color::tonemap::Bt2408Tonemapper::new(
                source_peak.min(10000.0),
                target_peak.min(10000.0),
            );
            let r = f32::from_le_bytes([remaining[8], remaining[9], remaining[10], remaining[11]]);
            let g =
                f32::from_le_bytes([remaining[12], remaining[13], remaining[14], remaining[15]]);
            let b =
                f32::from_le_bytes([remaining[16], remaining[17], remaining[18], remaining[19]]);
            if r.is_finite() && g.is_finite() && b.is_finite() {
                // Trait imported through ultrahdr-core's re-export so it is the
                // SAME `zentone` that implemented it for Bt2408Tonemapper (#32).
                use ultrahdr_core::color::tonemap::ToneMap;
                let _ = tm.map_rgb([
                    r.clamp(0.0, 100.0),
                    g.clamp(0.0, 100.0),
                    b.clamp(0.0, 100.0),
                ]);
            }
        }
        3 => {
            if remaining.len() < 12 {
                return;
            }
            let r = f32::from_le_bytes([remaining[0], remaining[1], remaining[2], remaining[3]]);
            let g = f32::from_le_bytes([remaining[4], remaining[5], remaining[6], remaining[7]]);
            let b = f32::from_le_bytes([remaining[8], remaining[9], remaining[10], remaining[11]]);
            if !r.is_finite() || !g.is_finite() || !b.is_finite() {
                return;
            }
            let rgb = [r.clamp(0.0, 1.0), g.clamp(0.0, 1.0), b.clamp(0.0, 1.0)];
            let config = ultrahdr_core::color::tonemap::ToneMapConfig {
                target_peak_nits: 203.0,
                hdr_peak_nits: 10000.0,
                target_gamut: ultrahdr_core::ColorPrimaries::Bt709,
                source_gamut: ultrahdr_core::ColorPrimaries::Bt2020,
            };
            let _ = ultrahdr_core::color::tonemap::tonemap_pq_to_sdr(rgb, &config);
            let _ = ultrahdr_core::color::tonemap::tonemap_hlg_to_sdr(rgb, &config);
        }
        4 => {
            if remaining.len() < 10 {
                return;
            }
            let width = remaining[0].clamp(1, 16) as u32;
            let height = remaining[1].clamp(1, 16) as u32;
            // `fmt_idx == 0` in the fuzz target, where `fmt_idx = remaining[2] % 2`.
            let format = if remaining[2].is_multiple_of(2) {
                ultrahdr_core::PixelFormat::RgbaF32
            } else {
                ultrahdr_core::PixelFormat::Rgba8
            };
            let gamut = match remaining[3] % 3 {
                0 => ultrahdr_core::ColorPrimaries::Bt709,
                1 => ultrahdr_core::ColorPrimaries::DisplayP3,
                _ => ultrahdr_core::ColorPrimaries::Bt2020,
            };
            let transfer = match remaining[4] % 4 {
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
            if format == ultrahdr_core::PixelFormat::RgbaF32 {
                for chunk in pixel_data.as_chunks_mut::<4>().0 {
                    let val = f32::from_le_bytes(*chunk);
                    let clamped = if val.is_finite() {
                        val.clamp(0.0, 10.0)
                    } else {
                        0.5
                    };
                    *chunk = clamped.to_le_bytes();
                }
            }
            let Ok(img) = ultrahdr_core::pixel_buffer_from_vec(
                pixel_data, width, height, format, gamut, transfer,
            ) else {
                return;
            };

            let _ = ultrahdr_core::color::tonemap::tonemap_image_to_srgb8(
                &img,
                ultrahdr_core::ColorPrimaries::Bt709,
            );
        }
        _ => {
            if remaining.len() < 4 {
                return;
            }
            let x = f32::from_le_bytes([remaining[0], remaining[1], remaining[2], remaining[3]]);
            if !x.is_finite() {
                return;
            }
            let x = x.clamp(0.0, 1.0);
            let _ = ultrahdr_core::color::transfer::srgb_eotf(x);
            let _ = ultrahdr_core::color::transfer::srgb_oetf(x);
            let _ = ultrahdr_core::color::transfer::pq_eotf(x);
            let _ = ultrahdr_core::color::transfer::pq_oetf(x);
            let _ = ultrahdr_core::color::transfer::hlg_eotf(x, 1000.0);
            let _ = ultrahdr_core::color::transfer::hlg_oetf(x);
        }
    }
}

#[test]
fn fuzz_regression() {
    let dir = regression_dir();

    // Fail loudly when the corpus this suite exists to replay is not there.
    let found = replayable_seeds(&dir);
    assert!(
        found >= MIN_SEEDS,
        "{} holds {found} replayable seeds, expected at least {MIN_SEEDS} — \
         the committed regression corpus is missing or was renamed, which would \
         otherwise let this test pass without replaying anything",
        dir.display()
    );

    RegressionSuite::new(dir)
        .target("decode", run_decode)
        .target("parse_jpeg_segments", run_parse_jpeg_segments)
        .target("parse_xmp", run_parse_xmp)
        .target("parse_mpf", run_parse_mpf)
        .target("parse_iso21496", run_parse_iso21496)
        .target("apply_gainmap", run_apply_gainmap)
        .target("tonemap", run_tonemap)
        .run();
}
