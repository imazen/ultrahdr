//! Compatibility tests with zune-image/zune-jpeg.
//!
//! Tests that our Ultra HDR JPEGs can be read by zune-jpeg as regular JPEGs
//! (backwards compatibility) and verifies the SDR base image is correct.

mod common;

use common::{create_hdr_gradient, create_hdr_solid, create_sdr_gradient, create_sdr_solid};
use std::io::Cursor;

/// Test that zune-jpeg can decode the SDR base of our Ultra HDR.
#[test]
fn test_zune_decodes_ultrahdr_base() {
    use zune_jpeg::JpegDecoder;

    let hdr = create_hdr_gradient(128, 128, 4.0);
    let sdr = create_sdr_gradient(128, 128);

    let mut encoder = ultrahdr::Encoder::new();
    encoder
        .set_hdr_image(hdr)
        .set_sdr_image(sdr)
        .set_quality(90, 85);

    let encoded = encoder.encode().unwrap();

    // zune-jpeg should decode the SDR base image
    let cursor = Cursor::new(&encoded);
    let mut decoder = JpegDecoder::new(cursor);

    let result = decoder.decode();
    assert!(
        result.is_ok(),
        "zune-jpeg should decode Ultra HDR: {:?}",
        result.err()
    );

    let pixels = result.unwrap();
    assert!(!pixels.is_empty(), "Decoded pixels should not be empty");

    // Check dimensions
    let info = decoder.info().expect("Should have image info");
    assert_eq!(info.width, 128);
    assert_eq!(info.height, 128);
}

/// Test that zune-jpeg preserves SDR pixel values.
#[test]
fn test_zune_sdr_pixel_preservation() {
    use zune_jpeg::JpegDecoder;

    // Use solid color for easier comparison
    let hdr = create_hdr_solid(64, 64, 0.5, 0.5, 0.5);
    let sdr = create_sdr_solid(64, 64, 186, 186, 186); // ~0.5 in sRGB

    let mut encoder = ultrahdr::Encoder::new();
    encoder
        .set_hdr_image(hdr)
        .set_sdr_image(sdr.clone())
        .set_quality(95, 90); // High quality

    let encoded = encoder.encode().unwrap();

    // Decode with zune-jpeg
    let cursor = Cursor::new(&encoded);
    let mut decoder = JpegDecoder::new(cursor);
    let pixels = decoder.decode().unwrap();

    // Check that pixel values are close to original
    // JPEG is lossy, so allow some tolerance
    let info = decoder.info().unwrap();
    let channels = info.components as usize;

    for i in 0..64 {
        let offset = i * channels;
        if offset + 2 < pixels.len() {
            let r = pixels[offset];
            let g = pixels[offset + 1];
            let b = pixels[offset + 2];

            // Should be close to 186 (gray)
            let max_diff = (r as i16 - 186)
                .abs()
                .max((g as i16 - 186).abs())
                .max((b as i16 - 186).abs());

            assert!(
                max_diff < 20,
                "Pixel {} color diff {} too high: RGB({},{},{})",
                i,
                max_diff,
                r,
                g,
                b
            );
        }
    }
}

/// Test zune-image integration (full image pipeline).
#[test]
fn test_zune_image_integration() {
    use zune_core::options::DecoderOptions;
    use zune_image::image::Image;

    let hdr = create_hdr_gradient(100, 100, 3.0);
    let sdr = create_sdr_gradient(100, 100);

    let mut encoder = ultrahdr::Encoder::new();
    encoder.set_hdr_image(hdr).set_sdr_image(sdr);

    let encoded = encoder.encode().unwrap();

    // Use zune-image's Image type
    let cursor = Cursor::new(&encoded);
    let result = Image::read(cursor, DecoderOptions::new_fast());

    assert!(
        result.is_ok(),
        "zune-image should read Ultra HDR: {:?}",
        result.err()
    );

    let image = result.unwrap();
    let (width, height) = image.dimensions();

    assert_eq!(width, 100);
    assert_eq!(height, 100);
}

/// Test that various image sizes work with zune-jpeg.
#[test]
fn test_zune_various_sizes() {
    use zune_jpeg::JpegDecoder;

    let sizes = [(32, 32), (64, 48), (100, 75), (128, 128), (200, 150)];

    for (w, h) in sizes {
        let hdr = create_hdr_gradient(w, h, 2.0);
        let sdr = create_sdr_gradient(w, h);

        let mut encoder = ultrahdr::Encoder::new();
        encoder.set_hdr_image(hdr).set_sdr_image(sdr);

        let encoded = encoder.encode().unwrap();

        let cursor = Cursor::new(&encoded);
        let mut decoder = JpegDecoder::new(cursor);

        let result = decoder.decode();
        assert!(
            result.is_ok(),
            "zune-jpeg should decode {}x{}: {:?}",
            w,
            h,
            result.err()
        );

        let info = decoder.info().unwrap();
        assert_eq!(info.width as u32, w, "Width mismatch for {}x{}", w, h);
        assert_eq!(info.height as u32, h, "Height mismatch for {}x{}", w, h);
    }
}

/// Test that zune handles the XMP/MPF metadata gracefully.
#[test]
fn test_zune_handles_metadata_markers() {
    use zune_jpeg::JpegDecoder;

    let hdr = create_hdr_gradient(64, 64, 4.0);
    let sdr = create_sdr_gradient(64, 64);

    let mut encoder = ultrahdr::Encoder::new();
    encoder
        .set_hdr_image(hdr)
        .set_sdr_image(sdr)
        .set_use_iso_metadata(true); // Include ISO metadata

    let encoded = encoder.encode().unwrap();

    // Verify the encoded data has the expected markers
    let has_xmp = encoded.windows(4).any(|w| w == b"http");
    let has_mpf = encoded.windows(4).any(|w| w == b"MPF\0");

    assert!(has_xmp, "Should contain XMP marker");
    assert!(has_mpf, "Should contain MPF marker");

    // zune-jpeg should still decode despite extra markers
    let cursor = Cursor::new(&encoded);
    let mut decoder = JpegDecoder::new(cursor);

    let result = decoder.decode();
    assert!(
        result.is_ok(),
        "zune-jpeg should handle XMP/MPF markers: {:?}",
        result.err()
    );
}

/// Test round-trip: encode with us, decode with zune, compare.
#[test]
fn test_roundtrip_with_zune() {
    use zune_jpeg::JpegDecoder;

    let hdr = create_hdr_gradient(80, 80, 2.0);
    let sdr = create_sdr_gradient(80, 80);

    let mut encoder = ultrahdr::Encoder::new();
    encoder
        .set_hdr_image(hdr)
        .set_sdr_image(sdr.clone())
        .set_quality(95, 90);

    let encoded = encoder.encode().unwrap();

    // Decode with zune-jpeg
    let cursor = Cursor::new(&encoded);
    let mut zune_decoder = JpegDecoder::new(cursor);
    let zune_pixels = zune_decoder.decode().unwrap();

    // Decode with our decoder
    let our_decoder = ultrahdr::Decoder::new(&encoded).unwrap();
    let our_sdr = our_decoder.decode_sdr().unwrap();

    // Both should have same dimensions
    let zune_info = zune_decoder.info().unwrap();
    assert_eq!(zune_info.width as u32, our_sdr.width);
    assert_eq!(zune_info.height as u32, our_sdr.height);

    // Pixel values should be very similar
    // Note: our decoder outputs RGBA, zune might output RGB
    let zune_channels = zune_info.components as usize;

    let mut max_diff = 0i16;
    let pixel_count = (80 * 80) as usize;

    for i in 0..pixel_count.min(zune_pixels.len() / zune_channels) {
        let zune_r = zune_pixels[i * zune_channels] as i16;
        let our_r = our_sdr.data[i * 4] as i16;

        let diff = (zune_r - our_r).abs();
        max_diff = max_diff.max(diff);
    }

    // Different JPEG decoders (jpegli vs zune-jpeg) may produce slightly different results
    // due to different IDCT implementations and rounding
    assert!(
        max_diff < 60,
        "Pixel difference between zune and our decoder: {}",
        max_diff
    );
}

/// Test that the gain map JPEG is also valid for zune.
#[test]
fn test_zune_decodes_gainmap() {
    use zune_jpeg::JpegDecoder;

    let hdr = create_hdr_gradient(128, 128, 4.0);
    let sdr = create_sdr_gradient(128, 128);

    let mut encoder = ultrahdr::Encoder::new();
    encoder
        .set_hdr_image(hdr)
        .set_sdr_image(sdr)
        .set_gainmap_scale(4);

    let encoded = encoder.encode().unwrap();

    // Extract the gain map JPEG using our decoder
    let our_decoder = ultrahdr::Decoder::new(&encoded).unwrap();
    let gm_jpeg = our_decoder.gainmap_jpeg().expect("Should have gain map");

    // Verify the gain map is a valid JPEG that zune can decode
    let cursor = Cursor::new(gm_jpeg);
    let mut gm_decoder = JpegDecoder::new(cursor);

    let result = gm_decoder.decode();
    assert!(
        result.is_ok(),
        "zune-jpeg should decode gain map JPEG: {:?}",
        result.err()
    );

    // Gain map should be 1/4 size
    let info = gm_decoder.info().unwrap();
    assert_eq!(info.width, 32, "Gain map width");
    assert_eq!(info.height, 32, "Gain map height");

    // Should be grayscale (1 component)
    assert_eq!(info.components, 1, "Gain map should be grayscale");
}
