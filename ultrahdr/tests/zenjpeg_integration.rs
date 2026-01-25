//! Integration tests demonstrating ultrahdr + zenjpeg workflow.
//!
//! These tests show the recommended usage patterns for UltraHDR encode/decode
//! using zenjpeg for JPEG codec operations and ultrahdr for gain map math.

#![cfg(feature = "zenjpeg")]

use ultrahdr_rs::{
    gainmap::{
        apply::{apply_gainmap, HdrOutputFormat},
        compute::{compute_gainmap, GainMapConfig},
    },
    metadata::xmp::{generate_xmp, parse_xmp},
    ColorGamut, ColorTransfer, GainMap, PixelFormat, RawImage, Unstoppable as CoreUnstoppable,
};

// Re-export zenjpeg types for tests
use zenjpeg::decoder::{Decoder, PreserveConfig};
use zenjpeg::encoder::{ChromaSubsampling, EncoderConfig, PixelLayout, Unstoppable};

/// Create test HDR RawImage (gradient with bright highlights).
fn create_test_hdr_image(width: u32, height: u32) -> RawImage {
    let mut pixels = Vec::with_capacity((width * height * 8) as usize);

    for y in 0..height {
        for x in 0..width {
            let u = x as f32 / width as f32;
            let v = y as f32 / height as f32;

            // Base gradient
            let r = u;
            let g = v;
            let b = 1.0 - u;

            // Add bright highlights in center
            let cx = (x as f32 - width as f32 / 2.0).abs() / (width as f32 / 2.0);
            let cy = (y as f32 - height as f32 / 2.0).abs() / (height as f32 / 2.0);
            let dist = (cx * cx + cy * cy).sqrt();

            // HDR highlight up to 4x SDR white
            let highlight = if dist < 0.3 {
                4.0 * (1.0 - dist / 0.3)
            } else {
                0.0
            };

            // Convert to half-precision bytes (f16)
            let hr = half::f16::from_f32(r + highlight);
            let hg = half::f16::from_f32(g + highlight);
            let hb = half::f16::from_f32(b + highlight);
            let ha = half::f16::from_f32(1.0);

            pixels.extend_from_slice(&hr.to_le_bytes());
            pixels.extend_from_slice(&hg.to_le_bytes());
            pixels.extend_from_slice(&hb.to_le_bytes());
            pixels.extend_from_slice(&ha.to_le_bytes());
        }
    }

    RawImage::from_data(
        width,
        height,
        PixelFormat::Rgba16F,
        ColorGamut::Bt709,
        ColorTransfer::Linear,
        pixels,
    )
    .expect("create HDR image")
}

/// Create SDR RawImage from HDR (simple tonemap by clamping).
fn create_sdr_from_hdr(hdr: &RawImage) -> RawImage {
    let mut pixels = Vec::with_capacity((hdr.width * hdr.height * 4) as usize);

    let hdr_data = &hdr.data;
    for i in 0..(hdr.width * hdr.height) as usize {
        let idx = i * 8;
        let r = half::f16::from_le_bytes([hdr_data[idx], hdr_data[idx + 1]]).to_f32();
        let g = half::f16::from_le_bytes([hdr_data[idx + 2], hdr_data[idx + 3]]).to_f32();
        let b = half::f16::from_le_bytes([hdr_data[idx + 4], hdr_data[idx + 5]]).to_f32();

        let sdr_r = linear_to_srgb_u8(r.clamp(0.0, 1.0));
        let sdr_g = linear_to_srgb_u8(g.clamp(0.0, 1.0));
        let sdr_b = linear_to_srgb_u8(b.clamp(0.0, 1.0));

        pixels.push(sdr_r);
        pixels.push(sdr_g);
        pixels.push(sdr_b);
        pixels.push(255);
    }

    RawImage::from_data(
        hdr.width,
        hdr.height,
        PixelFormat::Rgba8,
        ColorGamut::Bt709,
        ColorTransfer::Srgb,
        pixels,
    )
    .expect("create SDR image")
}

fn linear_to_srgb_u8(linear: f32) -> u8 {
    let srgb = if linear <= 0.0031308 {
        linear * 12.92
    } else {
        1.055 * linear.powf(1.0 / 2.4) - 0.055
    };
    (srgb * 255.0).round().clamp(0.0, 255.0) as u8
}

/// Convert RGBA8 RawImage to RGB8 bytes for zenjpeg encoding.
fn rgba8_to_rgb8(img: &RawImage) -> Vec<u8> {
    let mut rgb = Vec::with_capacity((img.width * img.height * 3) as usize);
    for i in 0..(img.width * img.height) as usize {
        let idx = i * 4;
        rgb.push(img.data[idx]);
        rgb.push(img.data[idx + 1]);
        rgb.push(img.data[idx + 2]);
    }
    rgb
}

/// Convert zenjpeg RGB output to RawImage.
fn rgb8_to_raw_image(data: &[u8], width: u32, height: u32) -> RawImage {
    let mut rgba = Vec::with_capacity((width * height * 4) as usize);
    for i in 0..(width * height) as usize {
        let idx = i * 3;
        rgba.push(data[idx]);
        rgba.push(data[idx + 1]);
        rgba.push(data[idx + 2]);
        rgba.push(255);
    }

    RawImage::from_data(
        width,
        height,
        PixelFormat::Rgba8,
        ColorGamut::Bt709,
        ColorTransfer::Srgb,
        rgba,
    )
    .expect("create raw image")
}

/// Convert zenjpeg grayscale output to GainMap.
fn gray8_to_gainmap(data: &[u8], width: u32, height: u32) -> GainMap {
    GainMap {
        width,
        height,
        channels: 1,
        data: data.to_vec(),
    }
}

/// Encode grayscale data to JPEG using zenjpeg.
fn encode_grayscale(data: &[u8], width: u32, height: u32, quality: f32) -> Vec<u8> {
    let config = EncoderConfig::grayscale(quality);
    let mut enc = config
        .encode_from_bytes(width, height, PixelLayout::Gray8Srgb)
        .expect("create encoder");
    enc.push_packed(data, Unstoppable).expect("push data");
    enc.finish().expect("finish encoding")
}

/// Encode RGB data to JPEG using zenjpeg.
fn encode_rgb(data: &[u8], width: u32, height: u32, quality: f32) -> Vec<u8> {
    let config = EncoderConfig::ycbcr(quality, ChromaSubsampling::Quarter);
    let mut enc = config
        .encode_from_bytes(width, height, PixelLayout::Rgb8Srgb)
        .expect("create encoder");
    enc.push_packed(data, Unstoppable).expect("push data");
    enc.finish().expect("finish encoding")
}

/// Encode UltraHDR JPEG with XMP and embedded gain map.
fn encode_ultrahdr(
    sdr_rgb: &[u8],
    width: u32,
    height: u32,
    quality: f32,
    xmp: &str,
    gainmap_jpeg: Vec<u8>,
) -> Vec<u8> {
    let config = EncoderConfig::ycbcr(quality, ChromaSubsampling::Quarter)
        .xmp(xmp.as_bytes().to_vec())
        .add_gainmap(gainmap_jpeg);

    let mut enc = config
        .encode_from_bytes(width, height, PixelLayout::Rgb8Srgb)
        .expect("create encoder");
    enc.push_packed(sdr_rgb, Unstoppable).expect("push data");
    enc.finish().expect("finish encoding")
}

// =============================================================================
// Gain map computation and application tests
// =============================================================================

#[test]
fn test_gainmap_compute_with_zenjpeg_encode() {
    let width = 64u32;
    let height = 64u32;

    let hdr_image = create_test_hdr_image(width, height);
    let sdr_image = create_sdr_from_hdr(&hdr_image);

    let config = GainMapConfig::default();
    let (gainmap, metadata) =
        compute_gainmap(&hdr_image, &sdr_image, &config, CoreUnstoppable).expect("compute gainmap");

    // Verify gain map dimensions and data
    assert!(gainmap.width > 0);
    assert!(gainmap.height > 0);
    assert_eq!(gainmap.channels, 1);
    assert!(!gainmap.data.is_empty());

    // Verify metadata is sensible
    assert!(metadata.hdr_capacity_max > 1.0);
    assert!(metadata.max_content_boost.iter().any(|&v| v > 1.0));

    // Encode gain map with zenjpeg
    let gainmap_jpeg = encode_grayscale(&gainmap.data, gainmap.width, gainmap.height, 75.0);
    assert!(!gainmap_jpeg.is_empty());
    assert_eq!(&gainmap_jpeg[..2], &[0xFF, 0xD8]);

    // Generate XMP
    let xmp = generate_xmp(&metadata, gainmap_jpeg.len());
    assert!(xmp.contains("hdrgm:"));
    assert!(xmp.contains("Version=\"1.0\""));
}

#[test]
fn test_xmp_roundtrip() {
    let width = 32u32;
    let height = 32u32;

    let hdr_image = create_test_hdr_image(width, height);
    let sdr_image = create_sdr_from_hdr(&hdr_image);

    let config = GainMapConfig {
        max_content_boost: 8.0,
        hdr_capacity_max: 8.0,
        ..Default::default()
    };
    let (_gainmap, metadata) =
        compute_gainmap(&hdr_image, &sdr_image, &config, CoreUnstoppable).expect("compute gainmap");

    let xmp = generate_xmp(&metadata, 1000);
    let (parsed, gainmap_len) = parse_xmp(&xmp).expect("parse XMP");

    assert_eq!(gainmap_len, Some(1000));

    for i in 0..3 {
        assert!(
            (parsed.max_content_boost[i] - metadata.max_content_boost[i]).abs() < 0.01,
            "max_content_boost[{}] mismatch",
            i
        );
        assert!(
            (parsed.min_content_boost[i] - metadata.min_content_boost[i]).abs() < 0.01,
            "min_content_boost[{}] mismatch",
            i
        );
        assert!(
            (parsed.gamma[i] - metadata.gamma[i]).abs() < 0.01,
            "gamma[{}] mismatch",
            i
        );
    }
}

#[test]
fn test_gainmap_apply() {
    let width = 32u32;
    let height = 32u32;

    let hdr_image = create_test_hdr_image(width, height);
    let sdr_image = create_sdr_from_hdr(&hdr_image);

    let config = GainMapConfig::default();
    let (gainmap, metadata) =
        compute_gainmap(&hdr_image, &sdr_image, &config, CoreUnstoppable).expect("compute gainmap");

    let reconstructed = apply_gainmap(
        &sdr_image,
        &gainmap,
        &metadata,
        4.0,
        HdrOutputFormat::LinearFloat,
        CoreUnstoppable,
    )
    .expect("apply gainmap");

    assert_eq!(reconstructed.width, width);
    assert_eq!(reconstructed.height, height);

    // Verify HDR values exist
    let data = &reconstructed.data;
    let mut max_value = 0.0f32;
    for i in 0..(width * height) as usize {
        let idx = i * 16;
        let r = f32::from_le_bytes([data[idx], data[idx + 1], data[idx + 2], data[idx + 3]]);
        max_value = max_value.max(r);
    }
    assert!(
        max_value > 1.0,
        "HDR should have values > 1.0, got {}",
        max_value
    );
}

// =============================================================================
// zenjpeg encode/decode tests
// =============================================================================

#[test]
fn test_zenjpeg_encode_decode_sdr() {
    let width = 64u32;
    let height = 64u32;

    let hdr_image = create_test_hdr_image(width, height);
    let sdr_image = create_sdr_from_hdr(&hdr_image);
    let sdr_rgb = rgba8_to_rgb8(&sdr_image);

    let jpeg = encode_rgb(&sdr_rgb, width, height, 90.0);
    assert!(!jpeg.is_empty());

    let decoded = Decoder::new().decode(&jpeg).expect("decode");

    assert_eq!(decoded.width, width);
    assert_eq!(decoded.height, height);

    let diff: i32 = sdr_rgb
        .iter()
        .zip(decoded.data.iter())
        .map(|(&a, &b)| (a as i32 - b as i32).abs())
        .sum();
    let avg_diff = diff as f32 / sdr_rgb.len() as f32;
    assert!(
        avg_diff < 10.0,
        "Average pixel difference too high: {}",
        avg_diff
    );
}

#[test]
fn test_zenjpeg_xmp_preservation() {
    let width = 32u32;
    let height = 32u32;

    let pixels = vec![128u8; (width * height * 3) as usize];
    let xmp_content = "<x:xmpmeta xmlns:x='adobe:ns:meta/'><rdf:RDF xmlns:rdf='http://www.w3.org/1999/02/22-rdf-syntax-ns#'><rdf:Description hdrgm:Version=\"1.0\"/></rdf:RDF></x:xmpmeta>";

    let config =
        EncoderConfig::ycbcr(85.0, ChromaSubsampling::Quarter).xmp(xmp_content.as_bytes().to_vec());

    let mut enc = config
        .encode_from_bytes(width, height, PixelLayout::Rgb8Srgb)
        .expect("create encoder");
    enc.push_packed(&pixels, Unstoppable).expect("push data");
    let jpeg = enc.finish().expect("finish encoding");

    let has_xmp = jpeg
        .windows(29)
        .any(|w| w == b"http://ns.adobe.com/xap/1.0/\0");
    assert!(has_xmp, "XMP marker not found");

    let decoded = Decoder::new()
        .preserve(PreserveConfig::default())
        .decode(&jpeg)
        .expect("decode");

    let extras = decoded.extras().expect("should have extras");
    let xmp_str = extras.xmp().expect("should have XMP");
    assert!(xmp_str.contains("hdrgm:Version"));
}

// =============================================================================
// Full UltraHDR integration tests
// =============================================================================

#[test]
fn test_ultrahdr_encode() {
    let width = 64u32;
    let height = 64u32;

    let hdr_image = create_test_hdr_image(width, height);
    let sdr_image = create_sdr_from_hdr(&hdr_image);

    let config = GainMapConfig::default();
    let (gainmap, metadata) =
        compute_gainmap(&hdr_image, &sdr_image, &config, CoreUnstoppable).expect("compute gainmap");

    let gainmap_jpeg = encode_grayscale(&gainmap.data, gainmap.width, gainmap.height, 75.0);
    let xmp = generate_xmp(&metadata, gainmap_jpeg.len());

    let sdr_rgb = rgba8_to_rgb8(&sdr_image);
    let ultrahdr = encode_ultrahdr(&sdr_rgb, width, height, 90.0, &xmp, gainmap_jpeg.clone());

    // Verify structure: should have 2 EOIs (primary + secondary)
    let eoi_count = ultrahdr.windows(2).filter(|w| w == &[0xFF, 0xD9]).count();
    assert_eq!(eoi_count, 2, "UltraHDR should have 2 EOI markers");

    // Verify XMP is present
    let has_xmp = ultrahdr
        .windows(29)
        .any(|w| w == b"http://ns.adobe.com/xap/1.0/\0");
    assert!(has_xmp, "XMP not found");

    // Verify MPF is present
    let has_mpf = ultrahdr.windows(4).any(|w| w == b"MPF\0");
    assert!(has_mpf, "MPF not found");

    // Verify size includes gainmap
    let sdr_only = encode_rgb(&sdr_rgb, width, height, 90.0);
    assert!(
        ultrahdr.len() > sdr_only.len() + gainmap_jpeg.len() / 2,
        "UltraHDR should be larger than SDR due to embedded gainmap"
    );
}

#[test]
fn test_ultrahdr_decode() {
    let width = 64u32;
    let height = 64u32;

    // Encode
    let hdr_image = create_test_hdr_image(width, height);
    let sdr_image = create_sdr_from_hdr(&hdr_image);

    let config = GainMapConfig::default();
    let (gainmap, metadata) =
        compute_gainmap(&hdr_image, &sdr_image, &config, CoreUnstoppable).expect("compute gainmap");

    let gainmap_jpeg = encode_grayscale(&gainmap.data, gainmap.width, gainmap.height, 75.0);
    let xmp = generate_xmp(&metadata, gainmap_jpeg.len());

    let sdr_rgb = rgba8_to_rgb8(&sdr_image);
    let ultrahdr = encode_ultrahdr(&sdr_rgb, width, height, 90.0, &xmp, gainmap_jpeg);

    // Decode
    let decoded = Decoder::new()
        .preserve(PreserveConfig::default())
        .decode(&ultrahdr)
        .expect("decode");

    assert_eq!(decoded.width, width);
    assert_eq!(decoded.height, height);

    let extras = decoded.extras().expect("should have extras");

    // Verify XMP
    let xmp_str = extras.xmp().expect("should have XMP");
    let (parsed_metadata, _) = parse_xmp(xmp_str).expect("parse XMP");
    assert!(parsed_metadata.hdr_capacity_max > 1.0);

    // Verify gainmap
    let gainmap_data = extras.gainmap().expect("should have gainmap");
    assert!(!gainmap_data.is_empty());
    assert_eq!(&gainmap_data[..2], &[0xFF, 0xD8]); // Valid JPEG
}

#[test]
fn test_ultrahdr_roundtrip() {
    let width = 64u32;
    let height = 64u32;

    // Create HDR and SDR
    let hdr_image = create_test_hdr_image(width, height);
    let sdr_image = create_sdr_from_hdr(&hdr_image);

    // Encode UltraHDR
    let config = GainMapConfig::default();
    let (gainmap, metadata) =
        compute_gainmap(&hdr_image, &sdr_image, &config, CoreUnstoppable).expect("compute gainmap");

    let gainmap_jpeg = encode_grayscale(&gainmap.data, gainmap.width, gainmap.height, 75.0);
    let xmp = generate_xmp(&metadata, gainmap_jpeg.len());

    let sdr_rgb = rgba8_to_rgb8(&sdr_image);
    let ultrahdr = encode_ultrahdr(&sdr_rgb, width, height, 90.0, &xmp, gainmap_jpeg);

    // Decode
    let decoded = Decoder::new()
        .preserve(PreserveConfig::default())
        .decode(&ultrahdr)
        .expect("decode");

    let extras = decoded.extras().expect("extras");
    let xmp_str = extras.xmp().expect("XMP");
    let (parsed_metadata, _) = parse_xmp(xmp_str).expect("parse XMP");

    let gainmap_data = extras.gainmap().expect("gainmap");
    let gainmap_decoded = Decoder::new().decode(gainmap_data).expect("decode gainmap");

    // Apply gainmap to reconstruct HDR
    let sdr_raw = rgb8_to_raw_image(&decoded.data, decoded.width, decoded.height);
    let gainmap_raw = gray8_to_gainmap(
        &gainmap_decoded.data,
        gainmap_decoded.width,
        gainmap_decoded.height,
    );

    let reconstructed = apply_gainmap(
        &sdr_raw,
        &gainmap_raw,
        &parsed_metadata,
        4.0,
        HdrOutputFormat::LinearFloat,
        CoreUnstoppable,
    )
    .expect("apply gainmap");

    assert_eq!(reconstructed.width, width);
    assert_eq!(reconstructed.height, height);

    // Verify HDR values
    let data = &reconstructed.data;
    let mut max_value = 0.0f32;
    for i in 0..(width * height) as usize {
        let idx = i * 16;
        let r = f32::from_le_bytes([data[idx], data[idx + 1], data[idx + 2], data[idx + 3]]);
        max_value = max_value.max(r);
    }
    assert!(
        max_value > 1.0,
        "Reconstructed HDR should have values > 1.0"
    );
}

#[test]
fn test_roundtrip_edit_sdr_keep_gainmap() {
    let width = 64u32;
    let height = 64u32;

    // Encode original
    let hdr_image = create_test_hdr_image(width, height);
    let sdr_image = create_sdr_from_hdr(&hdr_image);

    let config = GainMapConfig::default();
    let (gainmap, metadata) =
        compute_gainmap(&hdr_image, &sdr_image, &config, CoreUnstoppable).expect("compute gainmap");

    let gainmap_jpeg = encode_grayscale(&gainmap.data, gainmap.width, gainmap.height, 75.0);
    let xmp = generate_xmp(&metadata, gainmap_jpeg.len());

    let sdr_rgb = rgba8_to_rgb8(&sdr_image);
    let original = encode_ultrahdr(&sdr_rgb, width, height, 90.0, &xmp, gainmap_jpeg);

    // Decode
    let decoded = Decoder::new()
        .preserve(PreserveConfig::default())
        .decode(&original)
        .expect("decode");

    let extras = decoded.extras().unwrap();

    // Edit SDR pixels
    let edited_sdr: Vec<u8> = decoded.data.iter().map(|v| v.saturating_add(10)).collect();

    // Re-encode with preserved metadata using to_encoder_segments()
    let encoder_segments = extras.to_encoder_segments();

    let config =
        EncoderConfig::ycbcr(90.0, ChromaSubsampling::Quarter).with_segments(encoder_segments);
    let mut enc = config
        .encode_from_bytes(width, height, PixelLayout::Rgb8Srgb)
        .expect("create encoder");
    enc.push_packed(&edited_sdr, Unstoppable).expect("push");
    let re_encoded = enc.finish().expect("finish");

    // Verify metadata preserved
    let re_decoded = Decoder::new()
        .preserve(PreserveConfig::default())
        .decode(&re_encoded)
        .expect("decode");

    let re_extras = re_decoded.extras().unwrap();
    assert!(re_extras.xmp().is_some(), "XMP should be preserved");
    assert!(re_extras.gainmap().is_some(), "Gainmap should be preserved");

    // Verify pixels were edited
    assert_ne!(
        &re_decoded.data[..100],
        &decoded.data[..100],
        "SDR pixels should be different"
    );
}

// =============================================================================
// README workflow example test
// =============================================================================

/// This test exercises the exact workflow documented in README.md
/// for using ultrahdr-core with zenjpeg directly.
#[test]
fn test_readme_workflow_encode_decode() {
    // === ENCODING (as documented in README) ===

    let width = 64u32;
    let height = 64u32;

    // Create test HDR and SDR images
    let hdr_image = create_test_hdr_image(width, height);
    let sdr_image = create_sdr_from_hdr(&hdr_image);
    let sdr_rgb = rgba8_to_rgb8(&sdr_image);

    // 1. Compute gain map from HDR + SDR
    let config = GainMapConfig::default();
    let (gainmap, metadata) =
        compute_gainmap(&hdr_image, &sdr_image, &config, CoreUnstoppable).expect("compute gainmap");

    // 2. Encode gain map to JPEG
    let gainmap_jpeg = {
        let cfg = EncoderConfig::grayscale(75.0);
        let mut enc = cfg
            .encode_from_bytes(gainmap.width, gainmap.height, PixelLayout::Gray8Srgb)
            .expect("create gainmap encoder");
        enc.push_packed(&gainmap.data, Unstoppable)
            .expect("push gainmap");
        enc.finish().expect("finish gainmap")
    };

    // 3. Generate XMP metadata
    let xmp = generate_xmp(&metadata, gainmap_jpeg.len());
    assert!(xmp.contains("hdrgm:Version"));
    assert!(xmp.contains("hdrgm:GainMapMax"));

    // 4. Encode UltraHDR with embedded gain map
    let ultrahdr_jpeg = {
        let cfg = EncoderConfig::ycbcr(90.0, ChromaSubsampling::Quarter)
            .xmp(xmp.as_bytes().to_vec())
            .add_gainmap(gainmap_jpeg);
        let mut enc = cfg
            .encode_from_bytes(width, height, PixelLayout::Rgb8Srgb)
            .expect("create ultrahdr encoder");
        enc.push_packed(&sdr_rgb, Unstoppable).expect("push sdr");
        enc.finish().expect("finish ultrahdr")
    };

    // Verify UltraHDR structure
    assert!(!ultrahdr_jpeg.is_empty());
    assert_eq!(
        &ultrahdr_jpeg[..2],
        &[0xFF, 0xD8],
        "Should start with JPEG SOI"
    );

    // === DECODING (as documented in README) ===

    // 1. Decode with metadata preservation
    let decoded = Decoder::new()
        .preserve(PreserveConfig::default())
        .decode(&ultrahdr_jpeg)
        .expect("decode ultrahdr");

    let extras = decoded.extras().expect("should have extras");

    // 2. Parse XMP metadata
    let xmp_str = extras.xmp().expect("should have XMP");
    let (parsed_metadata, gainmap_len) = parse_xmp(xmp_str).expect("parse XMP");

    assert!(gainmap_len.is_some(), "XMP should contain gainmap length");
    assert!(
        parsed_metadata.hdr_capacity_max > 1.0,
        "HDR capacity should be > 1.0"
    );

    // 3. Decode gain map JPEG
    let gainmap_data = extras.gainmap().expect("should have gainmap");
    let gainmap_decoded = Decoder::new()
        .decode(gainmap_data)
        .expect("decode gainmap jpeg");

    // 4. Build RawImage and GainMap structs
    let sdr_raw = rgb8_to_raw_image(&decoded.data, decoded.width, decoded.height);
    let gainmap_raw = GainMap {
        width: gainmap_decoded.width,
        height: gainmap_decoded.height,
        channels: 1,
        data: gainmap_decoded.data,
    };

    // 5. Apply gain map to reconstruct HDR
    let hdr_reconstructed = apply_gainmap(
        &sdr_raw,
        &gainmap_raw,
        &parsed_metadata,
        4.0, // display boost
        HdrOutputFormat::LinearFloat,
        CoreUnstoppable,
    )
    .expect("apply gainmap");

    // Verify HDR reconstruction
    assert_eq!(hdr_reconstructed.width, width);
    assert_eq!(hdr_reconstructed.height, height);
    assert_eq!(hdr_reconstructed.format, PixelFormat::Rgba32F);

    // Verify HDR values exceed SDR range (> 1.0)
    let hdr_data = &hdr_reconstructed.data;
    let mut max_value = 0.0f32;
    for i in 0..(width * height) as usize {
        let idx = i * 16; // 4 floats × 4 bytes
        let r = f32::from_le_bytes([
            hdr_data[idx],
            hdr_data[idx + 1],
            hdr_data[idx + 2],
            hdr_data[idx + 3],
        ]);
        let g = f32::from_le_bytes([
            hdr_data[idx + 4],
            hdr_data[idx + 5],
            hdr_data[idx + 6],
            hdr_data[idx + 7],
        ]);
        let b = f32::from_le_bytes([
            hdr_data[idx + 8],
            hdr_data[idx + 9],
            hdr_data[idx + 10],
            hdr_data[idx + 11],
        ]);
        max_value = max_value.max(r).max(g).max(b);
    }

    assert!(
        max_value > 1.0,
        "Reconstructed HDR should have values > 1.0, got max={}",
        max_value
    );

    println!(
        "README workflow test passed: max HDR value = {:.2}, gainmap {}x{}",
        max_value, gainmap_raw.width, gainmap_raw.height
    );
}

/// Test the lossless round-trip workflow from README
#[test]
fn test_readme_workflow_lossless_roundtrip() {
    let width = 64u32;
    let height = 64u32;

    // Create and encode original UltraHDR
    let hdr_image = create_test_hdr_image(width, height);
    let sdr_image = create_sdr_from_hdr(&hdr_image);
    let sdr_rgb = rgba8_to_rgb8(&sdr_image);

    let config = GainMapConfig::default();
    let (gainmap, metadata) =
        compute_gainmap(&hdr_image, &sdr_image, &config, CoreUnstoppable).expect("compute gainmap");

    let gainmap_jpeg = encode_grayscale(&gainmap.data, gainmap.width, gainmap.height, 75.0);
    let xmp = generate_xmp(&metadata, gainmap_jpeg.len());
    let original = encode_ultrahdr(&sdr_rgb, width, height, 90.0, &xmp, gainmap_jpeg);

    // === LOSSLESS ROUND-TRIP (as documented in README) ===

    // Decode
    let decoded = Decoder::new()
        .preserve(PreserveConfig::default())
        .decode(&original)
        .expect("decode");
    let extras = decoded.extras().unwrap();

    // Edit SDR pixels (simple brightness adjustment)
    let edited_sdr: Vec<u8> = decoded.data.iter().map(|v| v.saturating_add(20)).collect();

    // Re-encode preserving XMP + gainmap
    let encoder_segments = extras.to_encoder_segments();
    let cfg =
        EncoderConfig::ycbcr(90.0, ChromaSubsampling::Quarter).with_segments(encoder_segments);
    let mut enc = cfg
        .encode_from_bytes(width, height, PixelLayout::Rgb8Srgb)
        .expect("create encoder");
    enc.push_packed(&edited_sdr, Unstoppable).expect("push");
    let re_encoded = enc.finish().expect("finish");

    // Verify round-trip preserved metadata
    let re_decoded = Decoder::new()
        .preserve(PreserveConfig::default())
        .decode(&re_encoded)
        .expect("decode re-encoded");

    let re_extras = re_decoded.extras().expect("should have extras");

    // XMP should be preserved
    let re_xmp = re_extras.xmp().expect("XMP preserved");
    assert!(
        re_xmp.contains("hdrgm:"),
        "XMP should contain hdrgm namespace"
    );

    // Gainmap should be preserved
    let re_gainmap = re_extras.gainmap().expect("gainmap preserved");
    assert_eq!(
        &re_gainmap[..2],
        &[0xFF, 0xD8],
        "Preserved gainmap should be valid JPEG"
    );

    // Pixels should be different (we edited them)
    let pixel_diff: i32 = decoded
        .data
        .iter()
        .zip(re_decoded.data.iter())
        .map(|(&a, &b)| (a as i32 - b as i32).abs())
        .sum();
    assert!(pixel_diff > 0, "Pixels should have changed after edit");

    // HDR reconstruction should still work with edited SDR + preserved gainmap
    let (parsed_metadata, _) = parse_xmp(re_xmp).expect("parse XMP");
    let gainmap_decoded = Decoder::new().decode(re_gainmap).expect("decode gainmap");

    let sdr_raw = rgb8_to_raw_image(&re_decoded.data, re_decoded.width, re_decoded.height);
    let gainmap_raw = GainMap {
        width: gainmap_decoded.width,
        height: gainmap_decoded.height,
        channels: 1,
        data: gainmap_decoded.data,
    };

    let hdr = apply_gainmap(
        &sdr_raw,
        &gainmap_raw,
        &parsed_metadata,
        4.0,
        HdrOutputFormat::LinearFloat,
        CoreUnstoppable,
    )
    .expect("apply gainmap after round-trip");

    assert_eq!(hdr.width, width);
    assert_eq!(hdr.height, height);

    println!("README lossless round-trip test passed");
}

/// Test set_existing_gainmap_jpeg() API - bypasses gain map encode entirely.
///
/// This tests the raw JPEG passthrough path used when we want to avoid
/// re-encoding the gain map (e.g., WASM where grayscale decode/encode crashes).
#[test]
fn test_set_existing_gainmap_jpeg() {
    use ultrahdr_rs::Encoder;

    let width = 64;
    let height = 64;

    // Create test HDR image
    let hdr_image = create_test_hdr_image(width, height);

    // Create SDR via tone mapping
    let sdr_image = create_sdr_from_hdr(&hdr_image);

    // Compute gain map to get metadata and JPEG bytes
    let config = GainMapConfig::default();
    let (gainmap, metadata) =
        compute_gainmap(&hdr_image, &sdr_image, &config, CoreUnstoppable).expect("compute gainmap");

    // Encode gain map to JPEG bytes
    let gainmap_jpeg = encode_grayscale(&gainmap.data, gainmap.width, gainmap.height, 75.0);
    assert!(!gainmap_jpeg.is_empty());
    assert_eq!(&gainmap_jpeg[..2], &[0xFF, 0xD8], "valid JPEG header");

    // Test 1: Encode using set_existing_gainmap_jpeg()
    let result_1 = Encoder::new()
        .set_hdr_image(hdr_image.clone())
        .set_sdr_image(sdr_image.clone())
        .set_existing_gainmap_jpeg(gainmap_jpeg.clone(), metadata.clone())
        .encode()
        .expect("encode with raw JPEG passthrough");

    // Test 2: Encode using set_existing_gainmap() (normal path)
    let result_2 = Encoder::new()
        .set_hdr_image(hdr_image.clone())
        .set_sdr_image(sdr_image.clone())
        .set_existing_gainmap(gainmap.clone(), metadata.clone())
        .encode()
        .expect("encode with GainMap struct");

    // Both should produce valid UltraHDR JPEGs
    assert_eq!(&result_1[..2], &[0xFF, 0xD8], "result 1 valid JPEG");
    assert_eq!(&result_2[..2], &[0xFF, 0xD8], "result 2 valid JPEG");

    // Sizes should be similar (raw passthrough may differ due to re-encode)
    let size_diff = (result_1.len() as i64 - result_2.len() as i64).abs();
    assert!(
        size_diff < 1000,
        "Size difference too large: {} vs {} (diff: {})",
        result_1.len(),
        result_2.len(),
        size_diff
    );

    // Both should be decodable
    let decoder_1 = ultrahdr_rs::Decoder::new(&result_1).expect("create decoder 1");
    let decoder_2 = ultrahdr_rs::Decoder::new(&result_2).expect("create decoder 2");

    assert!(decoder_1.is_ultrahdr(), "result 1 is UltraHDR");
    assert!(decoder_2.is_ultrahdr(), "result 2 is UltraHDR");

    // Both should have metadata
    let meta_1 = decoder_1.metadata().expect("metadata 1");
    let meta_2 = decoder_2.metadata().expect("metadata 2");

    assert!(
        (meta_1.hdr_capacity_max - meta_2.hdr_capacity_max).abs() < 0.01,
        "metadata should match: {} vs {}",
        meta_1.hdr_capacity_max,
        meta_2.hdr_capacity_max
    );

    println!("set_existing_gainmap_jpeg test passed");
}

/// Test that set_existing_gainmap_jpeg() works without HDR image
/// when SDR + compressed SDR + raw gain map JPEG are provided.
#[test]
fn test_set_existing_gainmap_jpeg_sdr_only() {
    use ultrahdr_rs::Encoder;

    let width = 64;
    let height = 64;

    // Create HDR and derive SDR from it
    let hdr_image = create_test_hdr_image(width, height);
    let sdr_image = create_sdr_from_hdr(&hdr_image);
    let sdr_rgb = rgba8_to_rgb8(&sdr_image);

    // Create dummy gain map (all 0.5 = neutral)
    let gainmap_data: Vec<u8> = vec![128; (width * height / 16) as usize]; // 1/4 scale
    let gm_width = width / 4;
    let gm_height = height / 4;
    let gainmap_jpeg = encode_grayscale(&gainmap_data, gm_width, gm_height, 75.0);

    // Create metadata using proper struct fields
    let metadata = ultrahdr_rs::GainMapMetadata {
        max_content_boost: [4.0, 4.0, 4.0],
        min_content_boost: [1.0, 1.0, 1.0],
        gamma: [1.0, 1.0, 1.0],
        offset_sdr: [1.0 / 64.0, 1.0 / 64.0, 1.0 / 64.0],
        offset_hdr: [1.0 / 64.0, 1.0 / 64.0, 1.0 / 64.0],
        hdr_capacity_min: 1.0,
        hdr_capacity_max: 4.0,
        use_base_color_space: false,
    };

    // Encode compressed SDR JPEG
    let sdr_jpeg = {
        let cfg = EncoderConfig::ycbcr(90.0, ChromaSubsampling::Quarter);
        let mut enc = cfg
            .encode_from_bytes(width, height, PixelLayout::Rgb8Srgb)
            .expect("create encoder");
        enc.push_packed(&sdr_rgb, Unstoppable).expect("push");
        enc.finish().expect("finish")
    };

    // Use the fast path: compressed SDR + raw gain map JPEG
    let result = Encoder::new()
        .set_compressed_sdr(sdr_jpeg)
        .set_existing_gainmap_jpeg(gainmap_jpeg, metadata)
        .encode()
        .expect("encode with SDR + raw JPEG");

    // Should produce valid UltraHDR
    assert_eq!(&result[..2], &[0xFF, 0xD8], "valid JPEG");

    let decoder = ultrahdr_rs::Decoder::new(&result).expect("create decoder");
    assert!(decoder.is_ultrahdr(), "should be UltraHDR");

    println!("set_existing_gainmap_jpeg SDR-only test passed");
}
