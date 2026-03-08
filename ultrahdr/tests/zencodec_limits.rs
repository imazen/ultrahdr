//! Regression tests for ResourceLimits enforcement in the zencodec adapter.
//!
//! These tests verify that `UltraHdrDecodeJob::with_limits()` actually
//! enforces the configured limits during probe and decode operations.

#![cfg(feature = "zencodec")]

use std::borrow::Cow;

use zc::ResourceLimits;
use zc::decode::{Decode, DecodeJob, DecoderConfig};

use ultrahdr_rs::zencodec::UltraHdrDecoderConfig;

const TEST_ULTRAHDR: &[u8] = include_bytes!("../../test_ultrahdr.jpg");

/// Decode with max_width=10 should fail for a wider image.
#[test]
fn decode_rejects_width_over_limit() {
    let config = UltraHdrDecoderConfig;
    let limits = ResourceLimits::none().with_max_width(10);
    let result = config
        .job()
        .with_limits(limits)
        .decoder(Cow::Borrowed(TEST_ULTRAHDR), &[])
        .unwrap()
        .decode();
    let err = match result {
        Err(e) => e,
        Ok(_) => panic!("decode should fail when image width exceeds max_width=10"),
    };
    let msg = err.to_string();
    assert!(
        msg.contains("width") || msg.contains("limit"),
        "error should mention width limit: {msg}"
    );
}

/// Decode with max_height=10 should fail for a taller image.
#[test]
fn decode_rejects_height_over_limit() {
    let config = UltraHdrDecoderConfig;
    let limits = ResourceLimits::none().with_max_height(10);
    let result = config
        .job()
        .with_limits(limits)
        .decoder(Cow::Borrowed(TEST_ULTRAHDR), &[])
        .unwrap()
        .decode();
    let err = match result {
        Err(e) => e,
        Ok(_) => panic!("decode should fail when image height exceeds max_height=10"),
    };
    let msg = err.to_string();
    assert!(
        msg.contains("height") || msg.contains("limit"),
        "error should mention height limit: {msg}"
    );
}

/// Decode with max_input_bytes=100 should fail for larger input.
#[test]
fn decode_rejects_input_over_limit() {
    let config = UltraHdrDecoderConfig;
    let limits = ResourceLimits::none().with_max_input_bytes(100);
    let result = config
        .job()
        .with_limits(limits)
        .decoder(Cow::Borrowed(TEST_ULTRAHDR), &[]);
    let err = match result {
        Err(e) => e,
        Ok(_) => panic!("decoder() should fail when input exceeds max_input_bytes=100"),
    };
    let msg = err.to_string();
    assert!(
        msg.contains("input") || msg.contains("limit"),
        "error should mention input size limit: {msg}"
    );
}

/// Decode with max_memory_bytes very small should fail.
#[test]
fn decode_rejects_memory_over_limit() {
    let config = UltraHdrDecoderConfig;
    // Set memory limit too small for the pixel buffer allocation
    let limits = ResourceLimits::none().with_max_memory(100);
    let result = config
        .job()
        .with_limits(limits)
        .decoder(Cow::Borrowed(TEST_ULTRAHDR), &[])
        .unwrap()
        .decode();
    let err = match result {
        Err(e) => e,
        Ok(_) => panic!("decode should fail when pixel buffer exceeds max_memory_bytes=100"),
    };
    let msg = err.to_string();
    assert!(
        msg.contains("memory") || msg.contains("limit"),
        "error should mention memory limit: {msg}"
    );
}

/// Probe with dimension limits should reject wide images.
#[test]
fn probe_rejects_dimensions_over_limit() {
    let config = UltraHdrDecoderConfig;
    let limits = ResourceLimits::none().with_max_width(10);
    let result = config.job().with_limits(limits).probe(TEST_ULTRAHDR);
    assert!(
        result.is_err(),
        "probe should fail when image width exceeds max_width=10"
    );
}

/// Generous limits should not reject the test image.
#[test]
fn decode_accepts_generous_limits() {
    let config = UltraHdrDecoderConfig;
    let limits = ResourceLimits::none()
        .with_max_width(10000)
        .with_max_height(10000)
        .with_max_input_bytes(100_000_000)
        .with_max_memory(100_000_000);
    let result = config
        .job()
        .with_limits(limits)
        .decoder(Cow::Borrowed(TEST_ULTRAHDR), &[])
        .unwrap()
        .decode();
    assert!(
        result.is_ok(),
        "generous limits should not reject test image: {:?}",
        result.err()
    );
}

/// No limits (default) should always succeed.
#[test]
fn decode_succeeds_with_no_limits() {
    let config = UltraHdrDecoderConfig;
    let result = config
        .job()
        .decoder(Cow::Borrowed(TEST_ULTRAHDR), &[])
        .unwrap()
        .decode();
    assert!(
        result.is_ok(),
        "no limits should not reject test image: {:?}",
        result.err()
    );
}
