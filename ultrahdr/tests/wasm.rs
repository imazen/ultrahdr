//! WASM-specific tests.
//!
//! These tests verify that the ultrahdr crate works correctly in WebAssembly.
//! Run with: wasm-pack test --headless --chrome
//! Or: wasm-pack test --node

#![cfg(target_arch = "wasm32")]

use wasm_bindgen_test::*;

// Note: run_in_browser is commented out to allow --node testing
// wasm_bindgen_test_configure!(run_in_browser);

/// Set up panic hook for better error messages in WASM.
fn setup() {
    console_error_panic_hook::set_once();
}

// ============================================================================
// Basic Decode Tests
// ============================================================================

const TEST_ULTRAHDR: &[u8] = include_bytes!("../../test_ultrahdr.jpg");

#[wasm_bindgen_test]
fn test_wasm_decoder_creation() {
    setup();
    let decoder = ultrahdr::Decoder::new(TEST_ULTRAHDR).expect("create decoder");
    assert!(decoder.is_ultrahdr(), "should recognize Ultra HDR");
}

#[wasm_bindgen_test]
fn test_wasm_decode_sdr() {
    setup();
    let decoder = ultrahdr::Decoder::new(TEST_ULTRAHDR).expect("create decoder");
    let sdr = decoder.decode_sdr().expect("decode SDR");
    assert!(sdr.width > 0, "width should be positive");
    assert!(sdr.height > 0, "height should be positive");
    assert!(!sdr.data.is_empty(), "data should not be empty");
}

#[wasm_bindgen_test]
fn test_wasm_decode_gainmap() {
    setup();
    let decoder = ultrahdr::Decoder::new(TEST_ULTRAHDR).expect("create decoder");
    let gainmap = decoder.decode_gainmap().expect("decode gainmap");
    assert!(gainmap.width > 0, "gainmap width should be positive");
    assert!(gainmap.height > 0, "gainmap height should be positive");
}

#[wasm_bindgen_test]
fn test_wasm_decode_hdr() {
    setup();
    let decoder = ultrahdr::Decoder::new(TEST_ULTRAHDR).expect("create decoder");
    let hdr = decoder.decode_hdr(4.0).expect("decode HDR");
    assert!(hdr.width > 0, "HDR width should be positive");
    assert!(hdr.height > 0, "HDR height should be positive");
    assert!(!hdr.data.is_empty(), "HDR data should not be empty");
}

#[wasm_bindgen_test]
fn test_wasm_metadata() {
    setup();
    let decoder = ultrahdr::Decoder::new(TEST_ULTRAHDR).expect("create decoder");
    let metadata = decoder.metadata().expect("get metadata");
    assert!(
        metadata.hdr_capacity_max > 1.0,
        "HDR capacity should be > 1"
    );
}

// ============================================================================
// Basic Encode Tests
// ============================================================================

/// Create a small test HDR image.
fn create_small_hdr(width: u32, height: u32) -> ultrahdr::RawImage {
    let pixels: Vec<[f32; 4]> = (0..width * height)
        .map(|i| {
            let x = (i % width) as f32 / width as f32;
            let y = (i / width) as f32 / height as f32;
            // Create gradient with HDR values
            let r = x * 2.0;
            let g = y * 2.0;
            let b = (x + y) * 1.5;
            [r, g, b, 1.0]
        })
        .collect();

    let data: Vec<u8> = pixels
        .iter()
        .flat_map(|p| p.iter().flat_map(|f| f.to_ne_bytes()))
        .collect();

    ultrahdr::RawImage {
        width,
        height,
        stride: width * 16,
        format: ultrahdr::PixelFormat::Rgba32F,
        gamut: ultrahdr::ColorGamut::Bt2100,
        transfer: ultrahdr::ColorTransfer::Linear,
        data,
    }
}

/// Create a small test SDR image.
fn create_small_sdr(width: u32, height: u32) -> ultrahdr::RawImage {
    let pixels: Vec<u8> = (0..width * height)
        .flat_map(|i| {
            let x = (i % width) as f32 / width as f32;
            let y = (i / width) as f32 / height as f32;
            let r = (x * 255.0) as u8;
            let g = (y * 255.0) as u8;
            let b = ((x + y) * 127.0) as u8;
            [r, g, b, 255u8]
        })
        .collect();

    ultrahdr::RawImage {
        width,
        height,
        stride: width * 4,
        format: ultrahdr::PixelFormat::Rgba8,
        gamut: ultrahdr::ColorGamut::Bt709,
        transfer: ultrahdr::ColorTransfer::Srgb,
        data: pixels,
    }
}

#[wasm_bindgen_test]
fn test_wasm_encode_hdr_only() {
    setup();
    let hdr = create_small_hdr(64, 64);

    let result = ultrahdr::Encoder::new()
        .set_hdr_image(hdr)
        .set_quality(80, 70)
        .encode();

    assert!(result.is_ok(), "encode should succeed: {:?}", result.err());
    let jpeg = result.unwrap();
    assert!(&jpeg[0..2] == &[0xFF, 0xD8], "should be valid JPEG");
}

#[wasm_bindgen_test]
fn test_wasm_encode_hdr_and_sdr() {
    setup();
    let hdr = create_small_hdr(64, 64);
    let sdr = create_small_sdr(64, 64);

    let result = ultrahdr::Encoder::new()
        .set_hdr_image(hdr)
        .set_sdr_image(sdr)
        .set_quality(80, 70)
        .encode();

    assert!(result.is_ok(), "encode should succeed: {:?}", result.err());
}

#[wasm_bindgen_test]
fn test_wasm_roundtrip() {
    setup();
    let hdr = create_small_hdr(64, 64);

    // Encode
    let jpeg = ultrahdr::Encoder::new()
        .set_hdr_image(hdr)
        .set_quality(85, 75)
        .encode()
        .expect("encode");

    // Decode
    let decoder = ultrahdr::Decoder::new(&jpeg).expect("create decoder");
    assert!(decoder.is_ultrahdr(), "should be Ultra HDR");

    let decoded_hdr = decoder.decode_hdr(4.0).expect("decode HDR");
    assert_eq!(decoded_hdr.width, 64);
    assert_eq!(decoded_hdr.height, 64);
}
