//! Simple WASM benchmark for ultrahdr-core.
//!
//! This can be compiled to WASI and run under wasmtime or wasmer.

use ultrahdr_core::{
    gainmap::{
        apply::{apply_gainmap, HdrOutputFormat},
        compute::{compute_gainmap, GainMapConfig},
    },
    ColorGamut, ColorTransfer, GainMap, GainMapMetadata, PixelFormat, RawImage, Unstoppable,
};

/// Simple timer using WASI clock.
fn now_ns() -> u64 {
    #[cfg(target_arch = "wasm32")]
    {
        // WASI provides clock_time_get
        let mut time: u64 = 0;
        unsafe {
            // clockid 1 = CLOCK_MONOTONIC
            wasi::clock_time_get(wasi::CLOCKID_MONOTONIC, 1, &mut time);
        }
        time
    }
    #[cfg(not(target_arch = "wasm32"))]
    {
        use std::time::Instant;
        static START: std::sync::OnceLock<Instant> = std::sync::OnceLock::new();
        let start = START.get_or_init(Instant::now);
        start.elapsed().as_nanos() as u64
    }
}

/// Create a test SDR image.
fn create_sdr_image(width: u32, height: u32) -> RawImage {
    let mut img = RawImage::new(width, height, PixelFormat::Rgba8).unwrap();
    img.gamut = ColorGamut::Bt709;
    img.transfer = ColorTransfer::Srgb;

    for y in 0..height {
        for x in 0..width {
            let idx = ((y * img.stride + x * 4) as usize).min(img.data.len() - 4);
            img.data[idx] = ((x * 255) / width.max(1)) as u8;
            img.data[idx + 1] = ((y * 255) / height.max(1)) as u8;
            img.data[idx + 2] = 128;
            img.data[idx + 3] = 255;
        }
    }
    img
}

/// Create a test HDR image.
fn create_hdr_image(width: u32, height: u32) -> RawImage {
    let mut img = RawImage::new(width, height, PixelFormat::Rgba8).unwrap();
    img.gamut = ColorGamut::Bt709;
    img.transfer = ColorTransfer::Srgb;

    for y in 0..height {
        for x in 0..width {
            let idx = ((y * img.stride + x * 4) as usize).min(img.data.len() - 4);
            img.data[idx] = (((x * 255) / width.max(1)) as u16).min(255) as u8;
            img.data[idx + 1] = (((y * 255) / height.max(1)) as u16 + 50).min(255) as u8;
            img.data[idx + 2] = 200;
            img.data[idx + 3] = 255;
        }
    }
    img
}

/// Create a test gain map.
fn create_gainmap(width: u32, height: u32) -> GainMap {
    let mut gm = GainMap::new(width, height).unwrap();
    for v in &mut gm.data {
        *v = 180;
    }
    gm
}

fn bench_apply_gainmap(width: u32, height: u32, iterations: u32) -> f64 {
    let sdr = create_sdr_image(width, height);
    let gainmap = create_gainmap(width / 4, height / 4);
    let metadata = GainMapMetadata {
        min_content_boost: [1.0; 3],
        max_content_boost: [4.0; 3],
        gamma: [1.0; 3],
        offset_sdr: [0.015625; 3],
        offset_hdr: [0.015625; 3],
        hdr_capacity_min: 1.0,
        hdr_capacity_max: 4.0,
        use_base_color_space: true,
    };

    // Warmup
    let _ = apply_gainmap(
        &sdr,
        &gainmap,
        &metadata,
        4.0,
        HdrOutputFormat::Srgb8,
        Unstoppable,
    );

    let start = now_ns();
    for _ in 0..iterations {
        let _ = apply_gainmap(
            &sdr,
            &gainmap,
            &metadata,
            4.0,
            HdrOutputFormat::Srgb8,
            Unstoppable,
        );
    }
    let elapsed = now_ns() - start;

    (elapsed as f64) / (iterations as f64) / 1_000_000.0 // ms per iteration
}

fn bench_compute_gainmap(width: u32, height: u32, iterations: u32) -> f64 {
    let hdr = create_hdr_image(width, height);
    let sdr = create_sdr_image(width, height);
    let config = GainMapConfig {
        scale_factor: 4,
        ..Default::default()
    };

    // Warmup
    let _ = compute_gainmap(&hdr, &sdr, &config, Unstoppable);

    let start = now_ns();
    for _ in 0..iterations {
        let _ = compute_gainmap(&hdr, &sdr, &config, Unstoppable);
    }
    let elapsed = now_ns() - start;

    (elapsed as f64) / (iterations as f64) / 1_000_000.0 // ms per iteration
}

fn main() {
    println!("=== ultrahdr-core WASM Benchmark ===");
    println!();

    let sizes = [(512, 512), (1024, 1024), (1920, 1080)];

    println!("apply_gainmap (srgb8 output):");
    for (width, height) in sizes {
        let iterations = if width * height > 1_000_000 { 5 } else { 20 };
        let ms = bench_apply_gainmap(width, height, iterations);
        let mpix_per_sec = (width * height) as f64 / ms / 1000.0;
        println!(
            "  {}x{}: {:.2} ms ({:.2} Mpix/s)",
            width, height, ms, mpix_per_sec
        );
    }
    println!();

    println!("compute_gainmap (luminance):");
    for (width, height) in sizes {
        let iterations = if width * height > 1_000_000 { 10 } else { 50 };
        let ms = bench_compute_gainmap(width, height, iterations);
        let mpix_per_sec = (width * height) as f64 / ms / 1000.0;
        println!(
            "  {}x{}: {:.2} ms ({:.2} Mpix/s)",
            width, height, ms, mpix_per_sec
        );
    }
}

#[cfg(target_arch = "wasm32")]
mod wasi {
    #[allow(non_camel_case_types)]
    pub type __wasi_clockid_t = u32;
    #[allow(non_camel_case_types)]
    pub type __wasi_timestamp_t = u64;

    pub const CLOCKID_MONOTONIC: __wasi_clockid_t = 1;

    #[link(wasm_import_module = "wasi_snapshot_preview1")]
    extern "C" {
        #[link_name = "clock_time_get"]
        fn __wasi_clock_time_get(
            id: __wasi_clockid_t,
            precision: __wasi_timestamp_t,
            time: *mut __wasi_timestamp_t,
        ) -> u16;
    }

    pub unsafe fn clock_time_get(
        id: __wasi_clockid_t,
        precision: __wasi_timestamp_t,
        time: &mut __wasi_timestamp_t,
    ) -> u16 {
        __wasi_clock_time_get(id, precision, time)
    }
}
