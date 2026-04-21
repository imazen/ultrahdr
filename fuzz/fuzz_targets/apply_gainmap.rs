#![no_main]

use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    // Structured fuzzer for gain map application.
    // Exercises: GainMapLut, apply_gainmap, color conversions, transfer functions.
    // Requires at least 24 bytes of control data + some pixel data.
    if data.len() < 48 {
        return;
    }

    // Extract structured parameters from first bytes
    let width = u16::from_le_bytes([data[0], data[1]]).max(1).min(64) as u32;
    let height = u16::from_le_bytes([data[2], data[3]]).max(1).min(64) as u32;
    let gm_width = u16::from_le_bytes([data[4], data[5]]).max(1).min(64) as u32;
    let gm_height = u16::from_le_bytes([data[6], data[7]]).max(1).min(64) as u32;

    let display_boost = f32::from_bits(u32::from_le_bytes([data[8], data[9], data[10], data[11]]));
    if !display_boost.is_finite() || display_boost < 0.0 {
        return;
    }

    // Metadata fields from fuzzer data. GainMapParams uses log2 domain and
    // per-channel structs; values chosen to exercise positive, negative, and
    // near-zero gains via the 128-centered byte-to-float map.
    let ch = ultrahdr_core::GainMapChannel {
        max: (data[12] as f64 - 128.0) / 32.0,
        min: (data[13] as f64 - 128.0) / 32.0,
        gamma: (data[14] as f64) / 128.0 + 0.01, // > 0
        base_offset: (data[15] as f64) / 255.0,
        alternate_offset: (data[16] as f64) / 255.0,
    };
    let mut metadata = ultrahdr_core::GainMapMetadata::default();
    metadata.channels = [ch; 3];
    metadata.base_hdr_headroom = (data[17] as f64 - 128.0) / 32.0;
    metadata.alternate_hdr_headroom = (data[18] as f64 - 128.0) / 32.0;
    let channels = if data[19] & 1 == 0 { 1u8 } else { 3u8 };
    let output_format_idx = data[20] % 3;
    let sdr_format_idx = data[21] % 2;
    metadata.use_base_color_space = data[22] & 1 != 0;
    metadata.backward_direction = data[22] & 2 != 0;

    let output_format = match output_format_idx {
        0 => ultrahdr_core::gainmap::apply::HdrOutputFormat::LinearFloat,
        1 => ultrahdr_core::gainmap::apply::HdrOutputFormat::Pq1010102,
        _ => ultrahdr_core::gainmap::apply::HdrOutputFormat::Srgb8,
    };

    let sdr_format = if sdr_format_idx == 0 {
        ultrahdr_core::PixelFormat::Rgba8
    } else {
        ultrahdr_core::PixelFormat::Rgb8
    };

    let pixel_data_start = 24;
    let bpp = sdr_format.bytes_per_pixel().unwrap();
    let sdr_size = (width as usize) * (height as usize) * bpp;
    let gm_size = (gm_width as usize) * (gm_height as usize) * (channels as usize);

    if data.len() < pixel_data_start + sdr_size + gm_size {
        return;
    }

    let sdr_data = data[pixel_data_start..pixel_data_start + sdr_size].to_vec();
    let gm_data = data[pixel_data_start + sdr_size..pixel_data_start + sdr_size + gm_size].to_vec();

    let sdr = match ultrahdr_core::RawImage::from_data(
        width,
        height,
        sdr_format,
        ultrahdr_core::ColorPrimaries::Bt709,
        ultrahdr_core::TransferFunction::Srgb,
        sdr_data,
    ) {
        Ok(img) => img,
        Err(_) => return,
    };

    let gainmap = ultrahdr_core::GainMap {
        width: gm_width,
        height: gm_height,
        channels,
        data: gm_data,
    };

    let _ = ultrahdr_core::gainmap::apply::apply_gainmap(
        &sdr,
        &gainmap,
        &metadata,
        display_boost,
        output_format,
        ultrahdr_core::Unstoppable,
    );
});
