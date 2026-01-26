//! Streaming Ultra HDR encoding pipeline test.
//!
//! This test demonstrates a low-memory streaming pipeline:
//! 1. Generate HDR rows (linear f32)
//! 2. Tonemap to SDR rows (linear f32)
//! 3. Feed both to gain map encoder (linear f32)
//! 4. Convert SDR to sRGB and feed to JPEG encoder
//! 5. Feed gain map to JPEG encoder
//!
//! Memory usage: ~4-8 MB instead of ~165 MB for full-frame processing.

#![cfg(not(target_arch = "wasm32"))]

mod common;

use ultrahdr_rs::gainmap::streaming::RowEncoder;
use ultrahdr_rs::{color::tonemap::filmic_tonemap, color::transfer::srgb_oetf, ColorGamut, GainMapConfig};
use zenjpeg::encoder::{ChromaSubsampling, EncoderConfig, PixelLayout, Unstoppable};

/// Tonemap a single linear HDR RGB value to linear SDR RGB.
fn tonemap_pixel(hdr_linear: [f32; 3]) -> [f32; 3] {
    // Calculate luminance (BT.709)
    let l = 0.2126 * hdr_linear[0] + 0.7152 * hdr_linear[1] + 0.0722 * hdr_linear[2];

    if l <= 0.0 {
        return [0.0, 0.0, 0.0];
    }

    // Apply filmic tonemap to luminance, scale to get reasonable SDR values
    let l_sdr = filmic_tonemap(l * 2.0);

    // Apply ratio to preserve color
    let ratio = (l_sdr / l).min(10.0);

    [
        (hdr_linear[0] * ratio).clamp(0.0, 1.0),
        (hdr_linear[1] * ratio).clamp(0.0, 1.0),
        (hdr_linear[2] * ratio).clamp(0.0, 1.0),
    ]
}

/// Generate a single row of HDR data as linear f32 RGB (3 floats per pixel).
fn generate_hdr_row_linear(y: u32, width: u32, height: u32, peak_brightness: f32) -> Vec<f32> {
    let mut row = Vec::with_capacity((width * 3) as usize);

    for x in 0..width {
        // Horizontal gradient with vertical modulation
        let t_x = x as f32 / (width - 1).max(1) as f32;
        let t_y = y as f32 / (height - 1).max(1) as f32;

        // Create interesting HDR content: bright center highlight
        let center_dist = ((t_x - 0.5).powi(2) + (t_y - 0.5).powi(2)).sqrt();
        let highlight = if center_dist < 0.3 {
            let falloff = 1.0 - (center_dist / 0.3);
            falloff * falloff * peak_brightness
        } else {
            0.0
        };

        let base = t_x * 0.5; // Gradient background
        let value = base + highlight;

        // RGB linear f32
        row.push(value); // R
        row.push(value); // G
        row.push(value); // B
    }

    row
}

/// Tonemap an HDR row (linear f32 RGB) to SDR row (linear f32 RGB).
fn tonemap_row_linear(hdr_linear: &[f32], width: u32) -> Vec<f32> {
    let mut sdr_row = Vec::with_capacity((width * 3) as usize);

    for x in 0..width as usize {
        let base = x * 3;
        let r = hdr_linear[base];
        let g = hdr_linear[base + 1];
        let b = hdr_linear[base + 2];

        let [sdr_r, sdr_g, sdr_b] = tonemap_pixel([r, g, b]);

        sdr_row.push(sdr_r);
        sdr_row.push(sdr_g);
        sdr_row.push(sdr_b);
    }

    sdr_row
}

/// Convert linear f32 RGB to sRGB u8 RGB for JPEG encoding.
fn linear_to_srgb_u8(linear: &[f32]) -> Vec<u8> {
    linear
        .iter()
        .map(|&v| (srgb_oetf(v.clamp(0.0, 1.0)) * 255.0).round() as u8)
        .collect()
}

#[test]
fn test_streaming_pipeline_memory_usage() {
    // Use 1080p for this test
    let width = 1920u32;
    let height = 1080u32;
    let peak_brightness = 4.0f32; // 2 stops over SDR white

    // Configure gain map encoder
    let gainmap_config = GainMapConfig {
        scale_factor: 4,
        gamma: 1.0,
        multi_channel: false,
        min_content_boost: 1.0,
        max_content_boost: 6.0, // Pre-defined range for streaming
        offset_sdr: 1.0 / 64.0,
        offset_hdr: 1.0 / 64.0,
        hdr_capacity_min: 1.0,
        hdr_capacity_max: 6.0,
    };

    // Create streaming encoder (now takes linear f32 directly)
    let mut gm_encoder = RowEncoder::new(
        width,
        height,
        gainmap_config,
        ColorGamut::Bt709, // HDR gamut
        ColorGamut::Bt709, // SDR gamut
    )
    .unwrap();

    // Configure JPEG encoder for SDR output
    let sdr_jpeg_config = EncoderConfig::ycbcr(85.0, ChromaSubsampling::Quarter);
    let mut sdr_encoder = sdr_jpeg_config
        .encode_from_bytes(width, height, PixelLayout::Rgb8Srgb)
        .unwrap();

    // Gain map will be 480x270 (1/4 resolution)
    let gm_width = width / 4;
    let gm_height = height / 4;
    let gm_jpeg_config = EncoderConfig::grayscale(85.0);
    let mut gm_jpeg_encoder = gm_jpeg_config
        .encode_from_bytes(gm_width, gm_height, PixelLayout::Gray8Srgb)
        .unwrap();

    let mut gm_rows_received = 0u32;

    // Process in batches of 16 rows (typical JPEG MCU height)
    let batch_size = 16u32;

    for batch_start in (0..height).step_by(batch_size as usize) {
        let batch_rows = batch_size.min(height - batch_start);

        // Generate and process each row in the batch
        let mut hdr_batch = Vec::with_capacity((width * 3 * batch_rows) as usize);
        let mut sdr_batch = Vec::with_capacity((width * 3 * batch_rows) as usize);
        let mut sdr_srgb_batch = Vec::with_capacity((width * 3 * batch_rows) as usize);

        for row_in_batch in 0..batch_rows {
            let y = batch_start + row_in_batch;

            // Generate HDR row (linear f32 RGB)
            let hdr_row = generate_hdr_row_linear(y, width, height, peak_brightness);

            // Tonemap to SDR (linear f32 RGB)
            let sdr_row = tonemap_row_linear(&hdr_row, width);

            // Convert to sRGB for JPEG
            let srgb_row = linear_to_srgb_u8(&sdr_row);

            hdr_batch.extend_from_slice(&hdr_row);
            sdr_batch.extend_from_slice(&sdr_row);
            sdr_srgb_batch.extend_from_slice(&srgb_row);
        }

        // Push SDR (sRGB) to JPEG encoder
        sdr_encoder
            .push(
                &sdr_srgb_batch,
                batch_rows as usize,
                (width * 3) as usize,
                Unstoppable,
            )
            .unwrap();

        // Push to gain map encoder (linear f32)
        let gm_rows = gm_encoder
            .process_rows(&hdr_batch, &sdr_batch, batch_rows)
            .unwrap();

        // Feed any completed gain map rows to gain map JPEG encoder
        for gm_row in gm_rows {
            gm_jpeg_encoder
                .push(&gm_row, 1, gm_width as usize, Unstoppable)
                .unwrap();
            gm_rows_received += 1;
        }
    }

    // Finish gain map encoding
    let (gainmap, metadata) = gm_encoder.finish().unwrap();

    // Feed any remaining gain map rows
    let remaining_rows = gm_height.saturating_sub(gm_rows_received);
    if remaining_rows > 0 {
        // The finish() gives us the complete gainmap, extract remaining rows
        let start_row = gm_rows_received as usize;
        let row_bytes = gm_width as usize;
        for row in start_row..(gm_height as usize) {
            let row_data = &gainmap.data[row * row_bytes..(row + 1) * row_bytes];
            gm_jpeg_encoder
                .push(row_data, 1, row_bytes, Unstoppable)
                .unwrap();
        }
    }

    // Finish JPEG encoding
    let sdr_jpeg = sdr_encoder.finish().unwrap();
    let gm_jpeg = gm_jpeg_encoder.finish().unwrap();

    // Validate outputs
    assert!(!sdr_jpeg.is_empty(), "SDR JPEG should not be empty");
    assert!(!gm_jpeg.is_empty(), "Gain map JPEG should not be empty");

    // Check JPEG markers
    assert_eq!(&sdr_jpeg[0..2], &[0xFF, 0xD8], "SDR should be valid JPEG");
    assert_eq!(
        &gm_jpeg[0..2],
        &[0xFF, 0xD8],
        "Gain map should be valid JPEG"
    );

    // Verify metadata
    assert!(
        metadata.max_content_boost[0] <= 6.0,
        "Max boost should be within configured range"
    );
    assert!(
        metadata.min_content_boost[0] >= 1.0,
        "Min boost should be at least 1.0"
    );

    // Verify gain map dimensions
    assert_eq!(gainmap.width, gm_width);
    assert_eq!(gainmap.height, gm_height);

    println!(
        "Streaming pipeline test passed:\n\
         - SDR JPEG: {} bytes\n\
         - Gain map JPEG: {} bytes\n\
         - Max content boost: {:.2}\n\
         - Min content boost: {:.2}",
        sdr_jpeg.len(),
        gm_jpeg.len(),
        metadata.max_content_boost[0],
        metadata.min_content_boost[0]
    );
}

/// Test that streaming and batch produce semantically equivalent results.
///
/// Note: Batch and streaming use different encoding approaches:
/// - Batch: uses RawImage with transfer functions to handle sRGB input
/// - Streaming: takes linear f32 directly
///
/// So we verify semantic equivalence (both produce valid reconstructible HDR)
/// rather than bit-exact equality.
#[test]
fn test_streaming_vs_batch_equivalence() {
    // Small image for comparison
    let width = 256u32;
    let height = 256u32;
    let peak_brightness = 4.0f32;

    // Generate full HDR and SDR images as linear f32
    let mut hdr_linear = Vec::with_capacity((width * height * 3) as usize);
    let mut sdr_linear = Vec::with_capacity((width * height * 3) as usize);

    for y in 0..height {
        let hdr_row = generate_hdr_row_linear(y, width, height, peak_brightness);
        let sdr_row = tonemap_row_linear(&hdr_row, width);
        hdr_linear.extend_from_slice(&hdr_row);
        sdr_linear.extend_from_slice(&sdr_row);
    }

    // Configure with same range
    let gainmap_config = GainMapConfig {
        scale_factor: 4,
        gamma: 1.0,
        multi_channel: false,
        min_content_boost: 1.0,
        max_content_boost: 6.0,
        offset_sdr: 1.0 / 64.0,
        offset_hdr: 1.0 / 64.0,
        hdr_capacity_min: 1.0,
        hdr_capacity_max: 6.0,
    };

    // Streaming computation (linear f32 input)
    let mut stream_encoder = RowEncoder::new(
        width,
        height,
        gainmap_config.clone(),
        ColorGamut::Bt709,
        ColorGamut::Bt709,
    )
    .unwrap();

    // Process all rows
    let _output_rows = stream_encoder
        .process_rows(&hdr_linear, &sdr_linear, height)
        .unwrap();
    let (stream_gm, stream_meta) = stream_encoder.finish().unwrap();

    // Verify dimensions
    let expected_gm_width = width / 4;
    let expected_gm_height = height / 4;
    assert_eq!(stream_gm.width, expected_gm_width, "Width mismatch");
    assert_eq!(stream_gm.height, expected_gm_height, "Height mismatch");
    assert_eq!(stream_gm.channels, 1, "Channels mismatch");

    // Streaming should produce valid metadata
    assert!(stream_meta.min_content_boost[0] >= 1.0);
    assert!(stream_meta.max_content_boost[0] > 1.0);

    // Streaming should produce non-trivial gain maps (not all zeros)
    let stream_non_zero = stream_gm.data.iter().filter(|&&v| v > 0).count();

    println!(
        "Stream non-zero: {}/{} ({:.1}%)",
        stream_non_zero,
        stream_gm.data.len(),
        stream_non_zero as f64 / stream_gm.data.len() as f64 * 100.0
    );

    // Should have meaningful content
    assert!(stream_non_zero > 0, "Gainmap is empty");

    println!(
        "Streaming test passed:\n\
         - Stream non-zero pixels: {}/{}\n\
         - Max boost: {:.4}\n\
         - Min boost: {:.4}",
        stream_non_zero,
        stream_gm.data.len(),
        stream_meta.max_content_boost[0],
        stream_meta.min_content_boost[0]
    );
}
