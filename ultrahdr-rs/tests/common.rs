//! Common test utilities for synthetic image generation.
//!
//! These helpers create test images programmatically, avoiding the need
//! to include large binary test files in the repository.

#![allow(dead_code)]

use ultrahdr_rs::{
    ColorPrimaries, GainMapMetadata, PixelBuffer, PixelFormat, TransferFunction,
    pixel_buffer_from_vec,
};

/// Create an HDR gradient image for testing.
///
/// Creates a horizontal gradient from black to the specified peak brightness.
/// Output is in linear RGB float format.
pub fn create_hdr_gradient(width: u32, height: u32, peak_brightness: f32) -> PixelBuffer {
    let mut data = Vec::with_capacity((width * height * 16) as usize);

    for y in 0..height {
        for x in 0..width {
            let t = x as f32 / (width - 1).max(1) as f32;
            let value = t * peak_brightness;

            data.extend_from_slice(&value.to_le_bytes());
            data.extend_from_slice(&value.to_le_bytes());
            data.extend_from_slice(&value.to_le_bytes());
            data.extend_from_slice(&1.0f32.to_le_bytes());
        }
        let _ = y;
    }

    pixel_buffer_from_vec(
        data,
        width,
        height,
        PixelFormat::RgbaF32,
        ColorPrimaries::Bt709,
        TransferFunction::Linear,
    )
    .unwrap()
}

/// Create an HDR gradient image in RGBA half-float format with chosen transfer.
pub fn create_hdr_f16_gradient_with_transfer(
    width: u32,
    height: u32,
    peak_brightness: f32,
    transfer: TransferFunction,
) -> PixelBuffer {
    let mut data = Vec::with_capacity((width * height * 8) as usize);

    for _y in 0..height {
        for x in 0..width {
            let t = x as f32 / (width - 1).max(1) as f32;
            let value = half::f16::from_f32(t * peak_brightness).to_le_bytes();
            let alpha = half::f16::ONE.to_le_bytes();

            data.extend_from_slice(&value);
            data.extend_from_slice(&value);
            data.extend_from_slice(&value);
            data.extend_from_slice(&alpha);
        }
    }

    pixel_buffer_from_vec(
        data,
        width,
        height,
        PixelFormat::RgbaF16,
        ColorPrimaries::Bt709,
        transfer,
    )
    .unwrap()
}

/// Create an SDR gradient image for testing.
pub fn create_sdr_gradient(width: u32, height: u32) -> PixelBuffer {
    let mut data = Vec::with_capacity((width * height * 4) as usize);

    for _y in 0..height {
        for x in 0..width {
            let t = x as f32 / (width - 1).max(1) as f32;
            let value = (t * 255.0) as u8;

            data.push(value);
            data.push(value);
            data.push(value);
            data.push(255);
        }
    }

    pixel_buffer_from_vec(
        data,
        width,
        height,
        PixelFormat::Rgba8,
        ColorPrimaries::Bt709,
        TransferFunction::Srgb,
    )
    .unwrap()
}

/// Create a solid color HDR image.
pub fn create_hdr_solid(width: u32, height: u32, r: f32, g: f32, b: f32) -> PixelBuffer {
    let mut data = Vec::with_capacity((width * height * 16) as usize);

    for _y in 0..height {
        for _x in 0..width {
            data.extend_from_slice(&r.to_le_bytes());
            data.extend_from_slice(&g.to_le_bytes());
            data.extend_from_slice(&b.to_le_bytes());
            data.extend_from_slice(&1.0f32.to_le_bytes());
        }
    }

    pixel_buffer_from_vec(
        data,
        width,
        height,
        PixelFormat::RgbaF32,
        ColorPrimaries::Bt709,
        TransferFunction::Linear,
    )
    .unwrap()
}

/// Create a solid color SDR image.
pub fn create_sdr_solid(width: u32, height: u32, r: u8, g: u8, b: u8) -> PixelBuffer {
    let mut data = Vec::with_capacity((width * height * 4) as usize);

    for _y in 0..height {
        for _x in 0..width {
            data.push(r);
            data.push(g);
            data.push(b);
            data.push(255);
        }
    }

    pixel_buffer_from_vec(
        data,
        width,
        height,
        PixelFormat::Rgba8,
        ColorPrimaries::Bt709,
        TransferFunction::Srgb,
    )
    .unwrap()
}

/// Create a checkerboard pattern HDR image.
pub fn create_hdr_checkerboard(width: u32, height: u32, low: f32, high: f32) -> PixelBuffer {
    let mut data = Vec::with_capacity((width * height * 16) as usize);
    let block_size = 8u32;

    for y in 0..height {
        for x in 0..width {
            let checker = ((x / block_size) + (y / block_size)).is_multiple_of(2);
            let value = if checker { high } else { low };

            data.extend_from_slice(&value.to_le_bytes());
            data.extend_from_slice(&value.to_le_bytes());
            data.extend_from_slice(&value.to_le_bytes());
            data.extend_from_slice(&1.0f32.to_le_bytes());
        }
    }

    pixel_buffer_from_vec(
        data,
        width,
        height,
        PixelFormat::RgbaF32,
        ColorPrimaries::Bt709,
        TransferFunction::Linear,
    )
    .unwrap()
}

/// Create a checkerboard pattern SDR image.
pub fn create_sdr_checkerboard(width: u32, height: u32, low: u8, high: u8) -> PixelBuffer {
    let mut data = Vec::with_capacity((width * height * 4) as usize);
    let block_size = 8u32;

    for y in 0..height {
        for x in 0..width {
            let checker = ((x / block_size) + (y / block_size)).is_multiple_of(2);
            let value = if checker { high } else { low };

            data.push(value);
            data.push(value);
            data.push(value);
            data.push(255);
        }
    }

    pixel_buffer_from_vec(
        data,
        width,
        height,
        PixelFormat::Rgba8,
        ColorPrimaries::Bt709,
        TransferFunction::Srgb,
    )
    .unwrap()
}

/// Create HDR image with bright highlights (for testing specular regions).
pub fn create_hdr_highlights(
    width: u32,
    height: u32,
    background: f32,
    highlight: f32,
) -> PixelBuffer {
    let mut data = Vec::with_capacity((width * height * 16) as usize);
    let center_x = width / 2;
    let center_y = height / 2;
    let radius = (width.min(height) / 4) as f32;

    for y in 0..height {
        for x in 0..width {
            let dx = x as f32 - center_x as f32;
            let dy = y as f32 - center_y as f32;
            let dist = (dx * dx + dy * dy).sqrt();

            let value = if dist < radius {
                let t = 1.0 - (dist / radius);
                background + (highlight - background) * t * t
            } else {
                background
            };

            data.extend_from_slice(&value.to_le_bytes());
            data.extend_from_slice(&value.to_le_bytes());
            data.extend_from_slice(&value.to_le_bytes());
            data.extend_from_slice(&1.0f32.to_le_bytes());
        }
    }

    pixel_buffer_from_vec(
        data,
        width,
        height,
        PixelFormat::RgbaF32,
        ColorPrimaries::Bt709,
        TransferFunction::Linear,
    )
    .unwrap()
}

/// Create test metadata with specified max boost (linear domain, converted to log2).
pub fn create_test_metadata(max_boost: f32) -> GainMapMetadata {
    let log2_max = (max_boost as f64).log2();
    let mut m = GainMapMetadata::default();
    for ch in &mut m.channels {
        ch.max = log2_max;
        ch.min = 0.0;
        ch.gamma = 1.0;
        ch.base_offset = 1.0 / 64.0;
        ch.alternate_offset = 1.0 / 64.0;
    }
    m.base_hdr_headroom = 0.0;
    m.alternate_hdr_headroom = log2_max;
    m.use_base_color_space = true;
    m
}

/// Linear to sRGB transfer function for reference calculations.
#[allow(dead_code)]
pub fn linear_to_srgb(v: f32) -> f32 {
    if v <= 0.0031308 {
        v * 12.92
    } else {
        1.055 * v.powf(1.0 / 2.4) - 0.055
    }
}

/// sRGB to linear transfer function for reference calculations.
#[allow(dead_code)]
pub fn srgb_to_linear(v: f32) -> f32 {
    if v <= 0.04045 {
        v / 12.92
    } else {
        ((v + 0.055) / 1.055).powf(2.4)
    }
}
