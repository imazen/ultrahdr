//! Compatibility tests with external JPEG decoders.
//!
//! Tests that our Ultra HDR JPEGs can be read by standard JPEG decoders as
//! regular JPEGs (backwards compatibility) and verifies the SDR base image
//! is correct.

#![cfg(not(target_arch = "wasm32"))]

mod common;

use common::{create_hdr_gradient, create_hdr_solid, create_sdr_gradient, create_sdr_solid};

/// Decode a JPEG with zenjpeg, returning (width, height, components, pixels).
fn decode_jpeg(data: &[u8]) -> (u32, u32, usize, Vec<u8>) {
    let result = zenjpeg::decoder::Decoder::new()
        .decode(data, enough::Unstoppable)
        .expect("zenjpeg should decode JPEG");
    let w = result.width();
    let h = result.height();
    let pixels = result.pixels_u8().expect("expected u8 output").to_vec();
    // zenjpeg decodes to RGB (3 components) by default
    (w, h, 3, pixels)
}

/// Test that zenjpeg can decode the SDR base of our Ultra HDR.
#[test]
fn test_decoder_decodes_ultrahdr_base() {
    let hdr = create_hdr_gradient(128, 128, 4.0);
    let sdr = create_sdr_gradient(128, 128);

    let mut encoder = ultrahdr_rs::Encoder::new();
    encoder
        .set_hdr_image(hdr)
        .set_sdr_image(sdr)
        .set_quality(90, 85);

    let encoded = encoder.encode().unwrap();

    let (w, h, _components, pixels) = decode_jpeg(&encoded);
    assert!(!pixels.is_empty(), "Decoded pixels should not be empty");
    assert_eq!(w, 128);
    assert_eq!(h, 128);
}

/// Test that the decoder preserves SDR pixel values.
#[test]
fn test_decoder_sdr_pixel_preservation() {
    // Use solid color for easier comparison
    let hdr = create_hdr_solid(64, 64, 0.5, 0.5, 0.5);
    let sdr = create_sdr_solid(64, 64, 186, 186, 186); // ~0.5 in sRGB

    let mut encoder = ultrahdr_rs::Encoder::new();
    encoder
        .set_hdr_image(hdr)
        .set_sdr_image(sdr)
        .set_quality(95, 90); // High quality

    let encoded = encoder.encode().unwrap();

    let (_w, _h, components, pixels) = decode_jpeg(&encoded);

    // Check that pixel values are close to original
    // JPEG is lossy, so allow some tolerance
    for i in 0..64 {
        let offset = i * components;
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

/// Test that various image sizes work with the decoder.
#[test]
fn test_decoder_various_sizes() {
    let sizes = [(32, 32), (64, 48), (100, 75), (128, 128), (200, 150)];

    for (w, h) in sizes {
        let hdr = create_hdr_gradient(w, h, 2.0);
        let sdr = create_sdr_gradient(w, h);

        let mut encoder = ultrahdr_rs::Encoder::new();
        encoder.set_hdr_image(hdr).set_sdr_image(sdr);

        let encoded = encoder.encode().unwrap();

        let (dw, dh, _components, _pixels) = decode_jpeg(&encoded);
        assert_eq!(dw, w, "Width mismatch for {}x{}", w, h);
        assert_eq!(dh, h, "Height mismatch for {}x{}", w, h);
    }
}

/// Test that the decoder handles the XMP/MPF metadata gracefully.
#[test]
fn test_decoder_handles_metadata_markers() {
    let hdr = create_hdr_gradient(64, 64, 4.0);
    let sdr = create_sdr_gradient(64, 64);

    let mut encoder = ultrahdr_rs::Encoder::new();
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

    // Decoder should still decode despite extra markers
    let (_w, _h, _components, pixels) = decode_jpeg(&encoded);
    assert!(!pixels.is_empty());
}

/// Test round-trip: encode with us, decode with zenjpeg, compare to our decoder.
#[test]
fn test_roundtrip_with_decoder() {
    let hdr = create_hdr_gradient(80, 80, 2.0);
    let sdr = create_sdr_gradient(80, 80);

    let mut encoder = ultrahdr_rs::Encoder::new();
    encoder
        .set_hdr_image(hdr)
        .set_sdr_image(sdr)
        .set_quality(95, 90);

    let encoded = encoder.encode().unwrap();

    // Decode with zenjpeg
    let (_w, _h, zen_channels, zen_pixels) = decode_jpeg(&encoded);

    // Decode with our decoder
    let our_decoder = ultrahdr_rs::Decoder::new(&encoded).unwrap();
    let our_sdr = our_decoder.decode_sdr().unwrap();

    // Both should have same dimensions
    assert_eq!(_w, our_sdr.width);
    assert_eq!(_h, our_sdr.height);

    // Pixel values should be very similar
    // Note: our decoder outputs RGBA, zenjpeg outputs RGB
    let mut max_diff = 0i16;
    let pixel_count = (80 * 80) as usize;

    for i in 0..pixel_count.min(zen_pixels.len() / zen_channels) {
        let zen_r = zen_pixels[i * zen_channels] as i16;
        let our_r = our_sdr.data[i * 4] as i16;

        let diff = (zen_r - our_r).abs();
        max_diff = max_diff.max(diff);
    }

    // Different decoders may produce slightly different results
    // due to different IDCT implementations and rounding
    assert!(
        max_diff < 60,
        "Pixel difference between zenjpeg and our decoder: {}",
        max_diff
    );
}

/// Test that the gain map JPEG is also valid.
#[test]
fn test_decoder_decodes_gainmap() {
    let hdr = create_hdr_gradient(128, 128, 4.0);
    let sdr = create_sdr_gradient(128, 128);

    let mut encoder = ultrahdr_rs::Encoder::new();
    encoder
        .set_hdr_image(hdr)
        .set_sdr_image(sdr)
        .set_gainmap_scale(4);

    let encoded = encoder.encode().unwrap();

    // Extract the gain map JPEG using our decoder
    let our_decoder = ultrahdr_rs::Decoder::new(&encoded).unwrap();
    let gm_jpeg = our_decoder.gainmap_jpeg().expect("Should have gain map");

    // Verify the gain map is a valid JPEG that can be decoded
    let (gw, gh, _components, pixels) = decode_jpeg(gm_jpeg);
    assert!(!pixels.is_empty());

    // Gain map should be 1/4 size
    assert_eq!(gw, 32, "Gain map width");
    assert_eq!(gh, 32, "Gain map height");
}
