//! Example: Encode a synthetic HDR image to Ultra HDR JPEG.
//!
//! This creates a test HDR image and encodes it to Ultra HDR format.
//!
//! Run with: cargo run --example encode

use ultrahdr::{ColorGamut, ColorTransfer, Encoder, PixelFormat, RawImage};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Ultra HDR Encoder Example");
    println!("=========================");

    // Create a synthetic HDR test image (gradient with HDR values)
    let width = 256;
    let height = 256;

    // Create linear float RGBA data (4 channels * 4 bytes each = 16 bytes per pixel)
    let mut hdr_data = vec![0u8; width * height * 16];

    for y in 0..height {
        for x in 0..width {
            // Create a gradient that exceeds SDR range
            let u = x as f32 / (width - 1) as f32;
            let v = y as f32 / (height - 1) as f32;

            // Linear light values - can exceed 1.0 for HDR
            let r = u * 4.0; // Up to 4x SDR (~ 800 nits if SDR = 200 nits)
            let g = v * 3.0; // Up to 3x SDR
            let b = ((u + v) / 2.0) * 2.0; // Up to 2x SDR
            let a: f32 = 1.0;

            let idx = (y * width + x) * 16;
            hdr_data[idx..idx + 4].copy_from_slice(&r.to_le_bytes());
            hdr_data[idx + 4..idx + 8].copy_from_slice(&g.to_le_bytes());
            hdr_data[idx + 8..idx + 12].copy_from_slice(&b.to_le_bytes());
            hdr_data[idx + 12..idx + 16].copy_from_slice(&a.to_le_bytes());
        }
    }

    let hdr_image = RawImage::from_data(
        width as u32,
        height as u32,
        PixelFormat::Rgba32F,
        ColorGamut::Bt709,
        ColorTransfer::Linear,
        hdr_data,
    )?;

    println!("Created {}x{} HDR test image", width, height);

    // Encode to Ultra HDR JPEG
    let mut encoder = Encoder::new();
    encoder
        .set_hdr_image(hdr_image)
        .set_quality(90, 85)
        .set_gainmap_scale(4)
        .set_target_display_peak(1000.0); // 1000 nits

    println!("Encoding to Ultra HDR JPEG...");
    let ultrahdr_jpeg = encoder.encode()?;

    println!(
        "Successfully encoded to {} bytes ({:.1} KB)",
        ultrahdr_jpeg.len(),
        ultrahdr_jpeg.len() as f64 / 1024.0
    );

    // Save to file
    let output_path = "test_ultrahdr.jpg";
    std::fs::write(output_path, &ultrahdr_jpeg)?;
    println!("Saved to {}", output_path);

    // Verify it's a valid Ultra HDR by decoding
    let decoder = ultrahdr::Decoder::new(&ultrahdr_jpeg)?;
    println!("\nVerification:");
    println!("  Is Ultra HDR: {}", decoder.is_ultrahdr());

    if let Some(metadata) = decoder.metadata() {
        println!("  Max content boost: {:.2}x", metadata.max_content_boost[0]);
        println!("  HDR capacity max: {:.2}x", metadata.hdr_capacity_max);
    }

    if let Ok((w, h)) = decoder.dimensions() {
        println!("  Dimensions: {}x{}", w, h);
    }

    println!("\nDone!");
    Ok(())
}
