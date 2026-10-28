//! Integration tests for zencodec trait implementation fixes.
//!
//! Verifies:
//! - `output_info()` returns non-zero dimensions
//! - `DecodeCapabilities` accurately reports supported features
//! - `probe()` detects gain maps and attaches metadata
//! - `decode()` ImageInfo includes has_alpha, CICP, and bit_depth

#![cfg(feature = "zencodec")]

use std::borrow::Cow;

use zencodec::Cicp;
use zencodec::decode::{Decode, DecodeJob, DecoderConfig};
use zenpixels::PixelDescriptor;

use ultrahdr_rs::codec::UltraHdrDecoderConfig;

const TEST_ULTRAHDR: &[u8] = include_bytes!("../../test_ultrahdr.jpg");

// =========================================================================
// Fix 1: output_info() returns real dimensions
// =========================================================================

#[test]
fn output_info_returns_nonzero_dimensions() {
    let config = UltraHdrDecoderConfig;
    let job = config.job();
    let info = job.output_info(TEST_ULTRAHDR).unwrap();
    assert!(info.width > 0, "output_info width should be non-zero");
    assert!(info.height > 0, "output_info height should be non-zero");
}

#[test]
fn output_info_matches_probe_dimensions() {
    let config = UltraHdrDecoderConfig;
    let probe_info = config.clone().job().probe(TEST_ULTRAHDR).unwrap();
    let output_info = config.job().output_info(TEST_ULTRAHDR).unwrap();
    assert_eq!(
        output_info.width, probe_info.width,
        "output_info width should match probe width"
    );
    assert_eq!(
        output_info.height, probe_info.height,
        "output_info height should match probe height"
    );
}

#[test]
fn output_info_native_format_is_rgb8() {
    let config = UltraHdrDecoderConfig;
    let info = config.job().output_info(TEST_ULTRAHDR).unwrap();
    assert_eq!(
        info.native_format,
        PixelDescriptor::RGB8_SRGB,
        "default native format should be RGB8"
    );
}

// =========================================================================
// Fix 2: DecodeCapabilities accurately reported
// =========================================================================

#[test]
fn capabilities_cancel_is_true() {
    let caps = UltraHdrDecoderConfig::capabilities();
    assert!(
        caps.stop(),
        "stop should be true (passes stop tokens to zenjpeg)"
    );
}

#[test]
fn capabilities_enforces_max_pixels() {
    let caps = UltraHdrDecoderConfig::capabilities();
    assert!(
        caps.enforces_max_pixels(),
        "enforces_max_pixels should be true (via check_dimensions)"
    );
}

#[test]
fn capabilities_enforces_max_memory() {
    let caps = UltraHdrDecoderConfig::capabilities();
    assert!(
        caps.enforces_max_memory(),
        "enforces_max_memory should be true (via check_memory)"
    );
}

#[test]
fn capabilities_enforces_max_input_bytes() {
    let caps = UltraHdrDecoderConfig::capabilities();
    assert!(
        caps.enforces_max_input_bytes(),
        "enforces_max_input_bytes should be true (via check_input_size)"
    );
}

#[test]
fn capabilities_cheap_probe() {
    let caps = UltraHdrDecoderConfig::capabilities();
    assert!(
        caps.cheap_probe(),
        "cheap_probe should be true (probe only parses JPEG headers)"
    );
}

#[test]
fn capabilities_hdr_support() {
    let caps = UltraHdrDecoderConfig::capabilities();
    assert!(
        caps.hdr(),
        "hdr should be true (Ultra HDR provides HDR via gain maps)"
    );
}

#[test]
fn capabilities_xmp_support() {
    let caps = UltraHdrDecoderConfig::capabilities();
    assert!(
        caps.xmp(),
        "xmp should be true (Ultra HDR metadata is in XMP)"
    );
}

#[test]
fn capabilities_no_animation() {
    let caps = UltraHdrDecoderConfig::capabilities();
    assert!(
        !caps.animation(),
        "animation should be false (Ultra HDR is single-frame)"
    );
}

#[test]
fn capabilities_no_row_level() {
    let caps = UltraHdrDecoderConfig::capabilities();
    assert!(
        !caps.streaming(),
        "streaming should be false (streaming decode is unsupported)"
    );
}

// =========================================================================
// Fix 3: probe() detects gain maps
// =========================================================================

#[test]
fn probe_detects_gain_map() {
    let config = UltraHdrDecoderConfig;
    let info = config.job().probe(TEST_ULTRAHDR).unwrap();
    assert!(
        info.gain_map.is_present(),
        "probe should detect gain map in Ultra HDR image"
    );
}

#[test]
fn probe_attaches_gain_map_metadata() {
    let config = UltraHdrDecoderConfig;
    let info = config.job().probe(TEST_ULTRAHDR).unwrap();
    assert!(
        info.gain_map.info().is_some(),
        "probe should attach gain map metadata when available"
    );
}

#[test]
fn probe_gain_map_metadata_has_valid_values() {
    let config = UltraHdrDecoderConfig;
    let info = config.job().probe(TEST_ULTRAHDR).unwrap();
    let gm_info = info
        .gain_map
        .info()
        .expect("gain map info should be present");
    // channel max values should be positive (log2 domain of max_content_boost > 1.0)
    for ch in &gm_info.params.channels {
        assert!(
            ch.max > 0.0,
            "channel max should be positive (log2 of boost > 1.0), got {}",
            ch.max
        );
    }
}

#[test]
fn probe_plain_jpeg_has_no_gain_map() {
    // Minimal JPEG (SOI + EOI) — not an Ultra HDR image
    let plain_jpeg = &[0xFF, 0xD8, 0xFF, 0xD9];
    let config = UltraHdrDecoderConfig;
    let info = config.job().probe(plain_jpeg).unwrap();
    assert!(
        !info.gain_map.is_present(),
        "plain JPEG should not have gain map"
    );
    assert!(
        info.gain_map.info().is_none(),
        "plain JPEG should not have gain map metadata"
    );
}

// =========================================================================
// Fix 4: decode() ImageInfo fields
// =========================================================================

#[test]
fn decode_imageinfo_has_cicp_srgb() {
    let config = UltraHdrDecoderConfig;
    let output = config
        .job()
        .decoder(Cow::Borrowed(TEST_ULTRAHDR), &[])
        .unwrap()
        .decode()
        .unwrap();
    let cicp = output.info().source_color.cicp;
    assert_eq!(
        cicp,
        Some(Cicp::SRGB),
        "decode output should have CICP set to sRGB"
    );
}

#[test]
fn decode_imageinfo_has_bit_depth_8() {
    let config = UltraHdrDecoderConfig;
    let output = config
        .job()
        .decoder(Cow::Borrowed(TEST_ULTRAHDR), &[])
        .unwrap()
        .decode()
        .unwrap();
    assert_eq!(
        output.info().source_color.bit_depth,
        Some(8),
        "decode output should report 8-bit depth"
    );
}

#[test]
fn decode_rgb8_has_no_alpha() {
    let config = UltraHdrDecoderConfig;
    let output = config
        .job()
        .decoder(Cow::Borrowed(TEST_ULTRAHDR), &[])
        .unwrap()
        .decode()
        .unwrap();
    assert!(
        !output.info().has_alpha,
        "RGB8 decode should not report has_alpha"
    );
}

#[test]
fn decode_rgba8_has_alpha() {
    let config = UltraHdrDecoderConfig;
    let output = config
        .job()
        .decoder(Cow::Borrowed(TEST_ULTRAHDR), &[PixelDescriptor::RGBA8_SRGB])
        .unwrap()
        .decode()
        .unwrap();
    assert!(
        output.info().has_alpha,
        "RGBA8 decode should report has_alpha"
    );
}

#[test]
fn decode_imageinfo_dimensions_are_nonzero() {
    let config = UltraHdrDecoderConfig;
    let output = config
        .job()
        .decoder(Cow::Borrowed(TEST_ULTRAHDR), &[])
        .unwrap()
        .decode()
        .unwrap();
    let info = output.info();
    assert!(info.width > 0, "decoded info width should be non-zero");
    assert!(info.height > 0, "decoded info height should be non-zero");
    // Verify dimensions match the probe result
    let probe_info = UltraHdrDecoderConfig.job().probe(TEST_ULTRAHDR).unwrap();
    assert_eq!(info.width, probe_info.width);
    assert_eq!(info.height, probe_info.height);
}

// =========================================================================
// Pulse: the job retains the operation pulse; the native decode polls it
// =========================================================================

/// Reports cancellation on every poll.
struct Cancelled;

impl zencodec::enough::Stop for Cancelled {
    fn check(&self) -> Result<(), zencodec::enough::StopReason> {
        Err(zencodec::enough::StopReason::Cancelled)
    }
}

fn decode_rgb8(
    job: ultrahdr_rs::codec::UltraHdrDecodeJob,
) -> Result<zencodec::decode::DecodeOutput, ultrahdr_rs::codec::ZenDecodeError> {
    job.decoder(Cow::Borrowed(TEST_ULTRAHDR), &[PixelDescriptor::RGB8_SRGB])?
        .decode()
}

fn assert_cancelled(
    result: Result<zencodec::decode::DecodeOutput, ultrahdr_rs::codec::ZenDecodeError>,
    what: &str,
) {
    match result {
        Err(ultrahdr_rs::codec::ZenDecodeError::Stopped(reason)) => {
            assert_eq!(reason, zencodec::enough::StopReason::Cancelled, "{what}")
        }
        Err(other) => panic!("{what}: another error: {other}"),
        Ok(_) => panic!("{what}: decoded through a cancelled pulse"),
    }
}

#[test]
fn a_cancelled_pulse_stops_the_decode() {
    let pulse = zencodec::SharedPulse::new(zencodec::how_far::StopOnly::new(
        zencodec::StopToken::new(Cancelled),
    ));
    assert_cancelled(
        decode_rgb8(UltraHdrDecoderConfig.job().with_pulse(pulse)),
        "with_pulse",
    );
}

#[test]
#[allow(deprecated)] // The legacy cancellation-only setter must keep working.
fn the_legacy_stop_setter_still_cancels() {
    let job = UltraHdrDecoderConfig
        .job()
        .with_stop(zencodec::StopToken::new(Cancelled));
    assert_cancelled(decode_rgb8(job), "with_stop");
}

#[test]
fn an_unstopped_pulse_decodes_what_no_pulse_decodes() {
    // A live pulse that never stops, unlike NoPulse.
    let plain = decode_rgb8(UltraHdrDecoderConfig.job()).unwrap();
    let pulsed = decode_rgb8(
        UltraHdrDecoderConfig
            .job()
            .with_pulse(zencodec::SharedPulse::new(
                zencodec::how_far::StopOnly::new(zencodec::Unstoppable),
            )),
    )
    .unwrap();
    assert_eq!(
        (plain.info().width, plain.info().height),
        (pulsed.info().width, pulsed.info().height)
    );
    assert_eq!(
        plain.pixels().contiguous_bytes(),
        pulsed.pixels().contiguous_bytes()
    );
}
