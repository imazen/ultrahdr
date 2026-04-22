//! Example: Encode a synthetic HDR image to Ultra HDR JPEG.
//!
//! Creates a test HDR image and encodes it to Ultra HDR format.
//!
//! Run with: cargo run --example encode --package ultrahdr-rs

use ultrahdr_rs::{
    ColorPrimaries, PixelFormat, TransferFunction, pixel_buffer_from_vec,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Ultra HDR Encoder Example");
    println!("=========================");

    let width = 256;
    let height = 256;

    // Create linear float RGBA data (4 channels * 4 bytes each = 16 bytes per pixel)
    let mut hdr_data = vec![0u8; width * height * 16];

    for y in 0..height {
        for x in 0..width {
            let u = x as f32 / (width - 1) as f32;
            let v = y as f32 / (height - 1) as f32;

            let r = u * 4.0;
            let g = v * 3.0;
            let b = ((u + v) / 2.0) * 2.0;
            let a: f32 = 1.0;

            let idx = (y * width + x) * 16;
            hdr_data[idx..idx + 4].copy_from_slice(&r.to_le_bytes());
            hdr_data[idx + 4..idx + 8].copy_from_slice(&g.to_le_bytes());
            hdr_data[idx + 8..idx + 12].copy_from_slice(&b.to_le_bytes());
            hdr_data[idx + 12..idx + 16].copy_from_slice(&a.to_le_bytes());
        }
    }

    let hdr_image = pixel_buffer_from_vec(
        hdr_data,
        width as u32,
        height as u32,
        PixelFormat::RgbaF32,
        ColorPrimaries::Bt709,
        TransferFunction::Linear,
    )?;

    println!("Created {}x{} HDR test image", width, height);

    let mut encoder = ultrahdr_rs::Encoder::new();
    encoder
        .set_hdr_image(hdr_image)
        .set_quality(90, 85)
        .set_gainmap_scale(4)
        .set_target_display_peak(1000.0);

    println!("Encoding to Ultra HDR JPEG...");
    let ultrahdr_jpeg = encoder.encode()?;

    println!(
        "Successfully encoded to {} bytes ({:.1} KB)",
        ultrahdr_jpeg.len(),
        ultrahdr_jpeg.len() as f64 / 1024.0
    );

    let output_path = "test_ultrahdr.jpg";
    std::fs::write(output_path, &ultrahdr_jpeg)?;
    println!("Saved to {}", output_path);

    let decoder = ultrahdr_rs::Decoder::new(&ultrahdr_jpeg)?;
    println!("\nVerification:");
    println!("  Is Ultra HDR: {}", decoder.is_ultrahdr());

    if let Some(metadata) = decoder.metadata() {
        let linear_max = 2f64.powf(metadata.channels[0].max);
        let linear_headroom = 2f64.powf(metadata.alternate_hdr_headroom);
        println!("  Max content boost: {:.2}x linear", linear_max);
        println!("  HDR capacity max: {:.2}x linear", linear_headroom);
    }

    if let Ok((w, h)) = decoder.dimensions() {
        println!("  Dimensions: {}x{}", w, h);
    }

    println!("\nDone!");
    Ok(())
}
