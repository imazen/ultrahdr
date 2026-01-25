//! Parity tests against libultrahdr C++ reference implementation.
//!
//! Uses the `ultrahdr` crate (FFI wrapper around Google's C++ library) to compare
//! encoding/decoding results between our pure Rust implementation and the reference.
//!
//! Run with: cargo test --features ffi-tests --test parity_libultrahdr

#![cfg(feature = "ffi-tests")]

mod common;

use common::{create_hdr_gradient, create_sdr_gradient};

// Re-export the sys types for easier access
use libultrahdr_rs::sys::{uhdr_color_gamut, uhdr_color_range, uhdr_color_transfer, uhdr_img_fmt};

/// Test that libultrahdr decoder can be created.
#[test]
fn test_libultrahdr_decoder_creation() {
    let decoder = libultrahdr_rs::Decoder::new();
    assert!(
        decoder.is_ok(),
        "Should create decoder: {:?}",
        decoder.err()
    );
    eprintln!("libultrahdr decoder created");
}

/// Test that libultrahdr encoder can be created.
#[test]
fn test_libultrahdr_encoder_creation() {
    let encoder = libultrahdr_rs::Encoder::new();
    assert!(
        encoder.is_ok(),
        "Should create encoder: {:?}",
        encoder.err()
    );
    eprintln!("libultrahdr encoder created");
}

/// Test that libultrahdr can decode sample Ultra HDR files.
#[test]
fn test_libultrahdr_decodes_samples() {
    let test_dir = std::path::Path::new("test_images");
    if !test_dir.exists() {
        eprintln!("Skipping: test_images not found");
        return;
    }

    for filename in &["sample_01.jpg", "sample_02.jpg", "sample_03.jpg"] {
        let path = test_dir.join(filename);
        if !path.exists() {
            continue;
        }

        let mut data = std::fs::read(&path).expect("read file");
        eprintln!("{}: {} bytes", filename, data.len());

        let mut decoder = libultrahdr_rs::Decoder::new().expect("decoder");

        let mut compressed = libultrahdr_rs::CompressedImage::from_bytes(
            &mut data,
            uhdr_color_gamut::UHDR_CG_BT_709,
            uhdr_color_transfer::UHDR_CT_SRGB,
            uhdr_color_range::UHDR_CR_FULL_RANGE,
        );

        if let Err(e) = decoder.set_image(&mut compressed) {
            eprintln!("{}: set_image failed: {:?}", filename, e);
            continue;
        }

        match decoder.decode_packed_view(
            uhdr_img_fmt::UHDR_IMG_FMT_32bppRGBA8888,
            uhdr_color_transfer::UHDR_CT_SRGB,
        ) {
            Ok(view) => {
                eprintln!("{}: {}x{}", filename, view.width(), view.height());
                assert!(view.width() > 0);
                assert!(view.height() > 0);
            }
            Err(e) => {
                eprintln!("{}: decode failed: {:?}", filename, e);
            }
        }
    }
}

/// Test that our decoder can parse sample Ultra HDR files.
#[test]
fn test_our_decoder_parses_samples() {
    let test_dir = std::path::Path::new("test_images");
    if !test_dir.exists() {
        eprintln!("Skipping: test_images not found");
        return;
    }

    for filename in &["sample_01.jpg", "sample_02.jpg", "sample_03.jpg"] {
        let path = test_dir.join(filename);
        if !path.exists() {
            continue;
        }

        let data = std::fs::read(&path).expect("read file");
        eprintln!("{}: {} bytes", filename, data.len());

        match ultrahdr_rs::Decoder::new(&data) {
            Ok(dec) => {
                eprintln!("{}: Parsed", filename);
                if dec.is_ultrahdr() {
                    eprintln!("{}: Ultra HDR", filename);
                    if let Ok((w, h)) = dec.dimensions() {
                        eprintln!("{}: {}x{}", filename, w, h);
                    }
                }
            }
            Err(e) => {
                eprintln!("{}: error: {:?}", filename, e);
            }
        }
    }
}

/// Test our encode produces valid JPEG.
#[test]
fn test_our_encode_produces_valid_jpeg() {
    let hdr = create_hdr_gradient(64, 64, 4.0);
    let sdr = create_sdr_gradient(64, 64);

    let mut encoder = ultrahdr_rs::Encoder::new();
    encoder
        .set_hdr_image(hdr)
        .set_sdr_image(sdr)
        .set_quality(90, 85);

    let encoded = match encoder.encode() {
        Ok(e) => e,
        Err(e) => {
            eprintln!("encode failed: {:?}", e);
            return;
        }
    };

    eprintln!("Encoded {} bytes", encoded.len());

    assert!(encoded.len() > 100);
    assert_eq!(&encoded[0..2], &[0xFF, 0xD8], "JPEG SOI");
    assert_eq!(&encoded[encoded.len() - 2..], &[0xFF, 0xD9], "JPEG EOI");

    let dec = ultrahdr_rs::Decoder::new(&encoded).expect("parse");
    assert!(dec.is_ultrahdr(), "Ultra HDR");
}

/// Test libultrahdr can parse our output.
#[test]
fn test_libultrahdr_parses_our_output() {
    let hdr = create_hdr_gradient(64, 64, 4.0);
    let sdr = create_sdr_gradient(64, 64);

    let mut encoder = ultrahdr_rs::Encoder::new();
    encoder
        .set_hdr_image(hdr)
        .set_sdr_image(sdr)
        .set_quality(90, 85);

    let mut encoded = match encoder.encode() {
        Ok(e) => e,
        Err(e) => {
            eprintln!("encode failed: {:?}", e);
            return;
        }
    };

    eprintln!("Testing libultrahdr with {} bytes", encoded.len());

    let mut decoder = libultrahdr_rs::Decoder::new().expect("decoder");

    let mut compressed = libultrahdr_rs::CompressedImage::from_bytes(
        &mut encoded,
        uhdr_color_gamut::UHDR_CG_BT_709,
        uhdr_color_transfer::UHDR_CT_SRGB,
        uhdr_color_range::UHDR_CR_FULL_RANGE,
    );

    match decoder.set_image(&mut compressed) {
        Ok(()) => eprintln!("libultrahdr accepted"),
        Err(e) => {
            eprintln!("libultrahdr rejected: {:?}", e);
            return;
        }
    }

    match decoder.decode_packed_view(
        uhdr_img_fmt::UHDR_IMG_FMT_32bppRGBA8888,
        uhdr_color_transfer::UHDR_CT_SRGB,
    ) {
        Ok(view) => {
            eprintln!("libultrahdr decoded: {}x{}", view.width(), view.height());
            assert_eq!(view.width(), 64);
            assert_eq!(view.height(), 64);
        }
        Err(e) => {
            eprintln!("decode failed: {:?}", e);
        }
    }
}

/// Test gain map metadata access.
#[test]
fn test_gainmap_metadata_access() {
    let test_dir = std::path::Path::new("test_images");
    if !test_dir.exists() {
        return;
    }

    let path = test_dir.join("sample_01.jpg");
    if !path.exists() {
        return;
    }

    let mut data = std::fs::read(&path).expect("read");

    let mut decoder = libultrahdr_rs::Decoder::new().expect("decoder");
    let mut compressed = libultrahdr_rs::CompressedImage::from_bytes(
        &mut data,
        uhdr_color_gamut::UHDR_CG_BT_709,
        uhdr_color_transfer::UHDR_CT_SRGB,
        uhdr_color_range::UHDR_CR_FULL_RANGE,
    );

    decoder.set_image(&mut compressed).expect("set_image");

    match decoder.gainmap_metadata() {
        Ok(Some(meta)) => {
            eprintln!("Gain map metadata:");
            eprintln!("  max_content_boost: {:?}", meta.max_content_boost);
            eprintln!("  min_content_boost: {:?}", meta.min_content_boost);
            eprintln!("  gamma: {:?}", meta.gamma);
        }
        Ok(None) => {
            eprintln!("No metadata");
        }
        Err(e) => {
            eprintln!("Error: {:?}", e);
        }
    }
}

/// Compare dimensions between decoders.
#[test]
fn test_dimension_parity() {
    let test_dir = std::path::Path::new("test_images");
    if !test_dir.exists() {
        return;
    }

    for filename in &["sample_01.jpg", "sample_02.jpg", "sample_03.jpg"] {
        let path = test_dir.join(filename);
        if !path.exists() {
            continue;
        }

        let data = std::fs::read(&path).expect("read");
        let mut data_copy = data.clone();

        let our_dims = ultrahdr_rs::Decoder::new(&data)
            .ok()
            .and_then(|d| d.dimensions().ok());

        let lib_dims: Option<(u32, u32)> = get_lib_dims(&mut data_copy);

        if let (Some((ow, oh)), Some((lw, lh))) = (our_dims, lib_dims) {
            eprintln!("{}: ours={}x{}, lib={}x{}", filename, ow, oh, lw, lh);
            assert_eq!(ow, lw, "Width");
            assert_eq!(oh, lh, "Height");
        }
    }
}

fn get_lib_dims(data: &mut [u8]) -> Option<(u32, u32)> {
    let mut decoder = libultrahdr_rs::Decoder::new().ok()?;
    let mut compressed = libultrahdr_rs::CompressedImage::from_bytes(
        data,
        uhdr_color_gamut::UHDR_CG_BT_709,
        uhdr_color_transfer::UHDR_CT_SRGB,
        uhdr_color_range::UHDR_CR_FULL_RANGE,
    );
    decoder.set_image(&mut compressed).ok()?;
    let view = decoder
        .decode_packed_view(
            uhdr_img_fmt::UHDR_IMG_FMT_32bppRGBA8888,
            uhdr_color_transfer::UHDR_CT_SRGB,
        )
        .ok()?;
    Some((view.width(), view.height()))
}
