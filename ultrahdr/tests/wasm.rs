//! WASM-specific tests.
//!
//! These tests verify that the ultrahdr crate works correctly in WebAssembly.
//! Run with: wasm-pack test --headless --chrome
//! Or: wasm-pack test --node

#![cfg(target_arch = "wasm32")]

use wasm_bindgen_test::*;

// Enable browser testing to match zenimage-web environment
wasm_bindgen_test_configure!(run_in_browser);

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

// ============================================================================
// Jpegli Grayscale Decode Tests (reproducing browser WASM crash)
// ============================================================================

/// Create a grayscale JPEG programmatically for testing.
fn create_grayscale_jpeg(width: u32, height: u32) -> Vec<u8> {
    use jpegli::encoder::{EncoderConfig, PixelLayout};

    // Create simple grayscale gradient
    let mut gray_data = vec![0u8; (width * height) as usize];
    for y in 0..height {
        for x in 0..width {
            gray_data[(y * width + x) as usize] = ((x + y) % 256) as u8;
        }
    }

    // Encode as grayscale
    let config = EncoderConfig::grayscale(90.0);
    let mut enc = config
        .encode_from_bytes(width, height, PixelLayout::Gray8Srgb)
        .expect("encoder setup");
    enc.push_packed(&gray_data, enough::Unstoppable)
        .expect("push");
    enc.finish().expect("finish encode")
}

/// Test direct jpegli grayscale decode - THIS CRASHES IN BROWSER WASM.
///
/// This test isolates the jpegli grayscale decode issue from ultrahdr.
/// - Node.js WASM: PASS
/// - Browser WASM: CRASH ("RuntimeError: unreachable")
///
/// The crash occurs in jpegli's decoder when decoding grayscale JPEGs.
/// RGB decode works fine; only grayscale decode crashes.
#[wasm_bindgen_test]
fn test_jpegli_grayscale_decode_direct() {
    setup();

    use jpegli::decoder::Decoder;

    // Create a 64x64 grayscale JPEG (similar size to gain maps)
    let jpeg_data = create_grayscale_jpeg(64, 64);

    let decoder = Decoder::new();

    // This crashes in browser WASM with "RuntimeError: unreachable"
    let result = decoder.decode(&jpeg_data);

    match result {
        Ok(decoded) => {
            assert_eq!(decoded.width, 64, "Expected 64x64 image");
            assert_eq!(decoded.height, 64, "Expected 64x64 image");
        }
        Err(e) => {
            panic!("Jpegli grayscale decode failed: {:?}", e);
        }
    }
}

/// Test jpegli RGB decode for comparison - this should always work.
#[wasm_bindgen_test]
fn test_jpegli_rgb_decode_direct() {
    setup();

    use jpegli::decoder::Decoder;
    use jpegli::encoder::{ChromaSubsampling, EncoderConfig, PixelLayout};

    // Create a simple RGB JPEG
    let config = EncoderConfig::ycbcr(90.0, ChromaSubsampling::Quarter);
    let mut enc = config
        .encode_from_bytes(8, 8, PixelLayout::Rgb8Srgb)
        .expect("encoder setup");

    let pixels = vec![255u8, 0, 0].repeat(64); // 8x8 red image
    enc.push_packed(&pixels, enough::Unstoppable).expect("push");
    let jpeg_data = enc.finish().expect("finish");

    // Decode RGB - this works in both Node.js and browser WASM
    let decoder = Decoder::new();
    let decoded = decoder
        .decode(&jpeg_data)
        .expect("RGB decode should work in WASM");

    assert_eq!(decoded.width, 8);
    assert_eq!(decoded.height, 8);
}

// ============================================================================
// Raw JPEG Passthrough Workaround for Browser WASM
// ============================================================================

/// Test extracting raw gain map JPEG without decoding.
///
/// This demonstrates the workaround for browser WASM grayscale decode crash:
/// 1. Use decoder.gainmap_jpeg() to get raw bytes (no decode)
/// 2. Use encoder.set_existing_gainmap_jpeg() to pass through (no re-encode)
#[wasm_bindgen_test]
fn test_gainmap_jpeg_extraction_workaround() {
    setup();

    let decoder = ultrahdr::Decoder::new(TEST_ULTRAHDR).expect("create decoder");
    assert!(decoder.is_ultrahdr());

    // This DOES NOT decode - just extracts raw JPEG bytes
    let gainmap_jpeg = decoder.gainmap_jpeg().expect("extract gainmap JPEG");

    // Verify it's a valid JPEG
    assert!(!gainmap_jpeg.is_empty(), "gainmap JPEG should not be empty");
    assert_eq!(
        &gainmap_jpeg[0..2],
        &[0xFF, 0xD8],
        "gainmap should start with JPEG SOI"
    );

    // Get metadata (this also doesn't decode pixels)
    let metadata = decoder.metadata().expect("get metadata");

    // Now we could re-encode using set_existing_gainmap_jpeg() to bypass decode/re-encode
    // This is the workaround used by zenimage-web for browser WASM
}

/// Test full roundtrip using raw JPEG passthrough (bypasses grayscale decode).
///
/// This is the recommended pattern for browser WASM where grayscale decode crashes.
#[wasm_bindgen_test]
fn test_roundtrip_with_raw_jpeg_passthrough() {
    setup();

    // Decode original UltraHDR
    let decoder = ultrahdr::Decoder::new(TEST_ULTRAHDR).expect("create decoder");
    assert!(decoder.is_ultrahdr());

    // Extract raw gain map JPEG (no decode, avoids crash)
    let gainmap_jpeg = decoder.gainmap_jpeg().expect("extract gainmap JPEG").to_vec();
    let metadata = decoder.metadata().expect("get metadata").clone();

    // Decode only SDR (RGB decode works fine in browser WASM)
    let sdr = decoder.decode_sdr().expect("decode SDR");

    // Create HDR image from SDR (simple: just scale up for test)
    let hdr_data: Vec<u8> = sdr
        .data
        .chunks(4)
        .flat_map(|rgba| {
            let r = (rgba[0] as f32 / 255.0 * 1.5).to_ne_bytes();
            let g = (rgba[1] as f32 / 255.0 * 1.5).to_ne_bytes();
            let b = (rgba[2] as f32 / 255.0 * 1.5).to_ne_bytes();
            let a = 1.0f32.to_ne_bytes();
            [r, g, b, a].concat()
        })
        .collect();

    let hdr = ultrahdr::RawImage {
        width: sdr.width,
        height: sdr.height,
        stride: sdr.width * 16,
        format: ultrahdr::PixelFormat::Rgba32F,
        gamut: ultrahdr::ColorGamut::Bt709,
        transfer: ultrahdr::ColorTransfer::Linear,
        data: hdr_data,
    };

    // Re-encode using raw JPEG passthrough (bypasses grayscale encode too)
    let result = ultrahdr::Encoder::new()
        .set_hdr_image(hdr)
        .set_sdr_image(sdr)
        .set_existing_gainmap_jpeg(gainmap_jpeg, metadata)
        .set_quality(90, 85)
        .encode()
        .expect("encode with raw JPEG passthrough");

    // Verify result is valid UltraHDR
    assert!(!result.is_empty());
    assert_eq!(&result[0..2], &[0xFF, 0xD8], "should be valid JPEG");

    let new_decoder = ultrahdr::Decoder::new(&result).expect("decode result");
    assert!(new_decoder.is_ultrahdr(), "result should be UltraHDR");
}
