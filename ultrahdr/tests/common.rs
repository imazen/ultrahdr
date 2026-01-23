//! Common test utilities for synthetic image generation.
//!
//! These helpers create test images programmatically, avoiding the need
//! to include large binary test files in the repository.

#![allow(dead_code)]

use ultrahdr::{ColorGamut, ColorTransfer, GainMapMetadata, PixelFormat, RawImage};

/// Create an HDR gradient image for testing.
///
/// Creates a horizontal gradient from black to the specified peak brightness.
/// Output is in linear RGB float format.
pub fn create_hdr_gradient(width: u32, height: u32, peak_brightness: f32) -> RawImage {
    let mut data = Vec::with_capacity((width * height * 16) as usize);

    for y in 0..height {
        for x in 0..width {
            // Horizontal gradient 0.0 to peak_brightness
            let t = x as f32 / (width - 1).max(1) as f32;
            let value = t * peak_brightness;

            // RGBA32F - 4 floats per pixel
            data.extend_from_slice(&value.to_le_bytes()); // R
            data.extend_from_slice(&value.to_le_bytes()); // G
            data.extend_from_slice(&value.to_le_bytes()); // B
            data.extend_from_slice(&1.0f32.to_le_bytes()); // A
        }
        // Add slight vertical variation
        let _ = y;
    }

    RawImage::from_data(
        width,
        height,
        PixelFormat::Rgba32F,
        ColorGamut::Bt709,
        ColorTransfer::Linear,
        data,
    )
    .unwrap()
}

/// Create an SDR gradient image for testing.
///
/// Creates a horizontal gradient from black to white in sRGB.
pub fn create_sdr_gradient(width: u32, height: u32) -> RawImage {
    let mut data = Vec::with_capacity((width * height * 4) as usize);

    for _y in 0..height {
        for x in 0..width {
            let t = x as f32 / (width - 1).max(1) as f32;
            let value = (t * 255.0) as u8;

            data.push(value); // R
            data.push(value); // G
            data.push(value); // B
            data.push(255); // A
        }
    }

    RawImage::from_data(
        width,
        height,
        PixelFormat::Rgba8,
        ColorGamut::Bt709,
        ColorTransfer::Srgb,
        data,
    )
    .unwrap()
}

/// Create a solid color HDR image.
pub fn create_hdr_solid(width: u32, height: u32, r: f32, g: f32, b: f32) -> RawImage {
    let mut data = Vec::with_capacity((width * height * 16) as usize);

    for _y in 0..height {
        for _x in 0..width {
            data.extend_from_slice(&r.to_le_bytes());
            data.extend_from_slice(&g.to_le_bytes());
            data.extend_from_slice(&b.to_le_bytes());
            data.extend_from_slice(&1.0f32.to_le_bytes());
        }
    }

    RawImage::from_data(
        width,
        height,
        PixelFormat::Rgba32F,
        ColorGamut::Bt709,
        ColorTransfer::Linear,
        data,
    )
    .unwrap()
}

/// Create a solid color SDR image.
pub fn create_sdr_solid(width: u32, height: u32, r: u8, g: u8, b: u8) -> RawImage {
    let mut data = Vec::with_capacity((width * height * 4) as usize);

    for _y in 0..height {
        for _x in 0..width {
            data.push(r);
            data.push(g);
            data.push(b);
            data.push(255);
        }
    }

    RawImage::from_data(
        width,
        height,
        PixelFormat::Rgba8,
        ColorGamut::Bt709,
        ColorTransfer::Srgb,
        data,
    )
    .unwrap()
}

/// Create a checkerboard pattern HDR image.
pub fn create_hdr_checkerboard(width: u32, height: u32, low: f32, high: f32) -> RawImage {
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

    RawImage::from_data(
        width,
        height,
        PixelFormat::Rgba32F,
        ColorGamut::Bt709,
        ColorTransfer::Linear,
        data,
    )
    .unwrap()
}

/// Create a checkerboard pattern SDR image.
pub fn create_sdr_checkerboard(width: u32, height: u32, low: u8, high: u8) -> RawImage {
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

    RawImage::from_data(
        width,
        height,
        PixelFormat::Rgba8,
        ColorGamut::Bt709,
        ColorTransfer::Srgb,
        data,
    )
    .unwrap()
}

/// Create HDR image with bright highlights (for testing specular regions).
pub fn create_hdr_highlights(width: u32, height: u32, background: f32, highlight: f32) -> RawImage {
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
                // Smooth falloff from center
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

    RawImage::from_data(
        width,
        height,
        PixelFormat::Rgba32F,
        ColorGamut::Bt709,
        ColorTransfer::Linear,
        data,
    )
    .unwrap()
}

/// Create test metadata with specified max boost.
pub fn create_test_metadata(max_boost: f32) -> GainMapMetadata {
    GainMapMetadata {
        max_content_boost: [max_boost; 3],
        min_content_boost: [1.0; 3],
        gamma: [1.0; 3],
        offset_sdr: [1.0 / 64.0; 3],
        offset_hdr: [1.0 / 64.0; 3],
        hdr_capacity_min: 1.0,
        hdr_capacity_max: max_boost,
        use_base_color_space: true,
    }
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
