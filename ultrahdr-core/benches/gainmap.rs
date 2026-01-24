//! Benchmarks for gain map operations.

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use std::hint::black_box;
use ultrahdr_core::{
    gainmap::{
        apply::{apply_gainmap, HdrOutputFormat},
        compute::{compute_gainmap, GainMapConfig},
    },
    ColorGamut, ColorTransfer, GainMap, GainMapMetadata, PixelFormat, RawImage,
};

/// Create a test SDR image of given dimensions.
fn create_sdr_image(width: u32, height: u32) -> RawImage {
    let mut img = RawImage::new(width, height, PixelFormat::Rgba8).unwrap();
    img.gamut = ColorGamut::Bt709;
    img.transfer = ColorTransfer::Srgb;

    // Fill with a gradient pattern
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

/// Create a test HDR image (brighter version of SDR).
fn create_hdr_image(width: u32, height: u32) -> RawImage {
    let mut img = RawImage::new(width, height, PixelFormat::Rgba8).unwrap();
    img.gamut = ColorGamut::Bt709;
    img.transfer = ColorTransfer::Srgb;

    // Fill with brighter gradient
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
    // Fill with mid-gain values
    for v in &mut gm.data {
        *v = 180;
    }
    gm
}

fn bench_apply_gainmap(c: &mut Criterion) {
    let mut group = c.benchmark_group("apply_gainmap");

    // Test different image sizes
    let sizes = [(256, 256), (512, 512), (1024, 1024), (1920, 1080)];

    for (width, height) in sizes {
        let pixels = (width * height) as u64;
        group.throughput(Throughput::Elements(pixels));

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

        group.bench_with_input(
            BenchmarkId::new("linear_float", format!("{}x{}", width, height)),
            &(width, height),
            |b, _| {
                b.iter(|| {
                    apply_gainmap(
                        black_box(&sdr),
                        black_box(&gainmap),
                        black_box(&metadata),
                        black_box(4.0),
                        HdrOutputFormat::LinearFloat,
                        enough::Unstoppable,
                    )
                    .unwrap()
                });
            },
        );

        group.bench_with_input(
            BenchmarkId::new("srgb8", format!("{}x{}", width, height)),
            &(width, height),
            |b, _| {
                b.iter(|| {
                    apply_gainmap(
                        black_box(&sdr),
                        black_box(&gainmap),
                        black_box(&metadata),
                        black_box(4.0),
                        HdrOutputFormat::Srgb8,
                        enough::Unstoppable,
                    )
                    .unwrap()
                });
            },
        );

        group.bench_with_input(
            BenchmarkId::new("pq1010102", format!("{}x{}", width, height)),
            &(width, height),
            |b, _| {
                b.iter(|| {
                    apply_gainmap(
                        black_box(&sdr),
                        black_box(&gainmap),
                        black_box(&metadata),
                        black_box(4.0),
                        HdrOutputFormat::Pq1010102,
                        enough::Unstoppable,
                    )
                    .unwrap()
                });
            },
        );
    }

    group.finish();
}

fn bench_compute_gainmap(c: &mut Criterion) {
    let mut group = c.benchmark_group("compute_gainmap");

    let sizes = [(256, 256), (512, 512), (1024, 1024), (1920, 1080)];

    for (width, height) in sizes {
        let pixels = (width * height) as u64;
        group.throughput(Throughput::Elements(pixels));

        let hdr = create_hdr_image(width, height);
        let sdr = create_sdr_image(width, height);
        let config = GainMapConfig {
            scale_factor: 4,
            ..Default::default()
        };

        group.bench_with_input(
            BenchmarkId::new("luminance", format!("{}x{}", width, height)),
            &(width, height),
            |b, _| {
                b.iter(|| {
                    compute_gainmap(
                        black_box(&hdr),
                        black_box(&sdr),
                        black_box(&config),
                        enough::Unstoppable,
                    )
                    .unwrap()
                });
            },
        );

        let config_multi = GainMapConfig {
            scale_factor: 4,
            multi_channel: true,
            ..Default::default()
        };

        group.bench_with_input(
            BenchmarkId::new("multichannel", format!("{}x{}", width, height)),
            &(width, height),
            |b, _| {
                b.iter(|| {
                    compute_gainmap(
                        black_box(&hdr),
                        black_box(&sdr),
                        black_box(&config_multi),
                        enough::Unstoppable,
                    )
                    .unwrap()
                });
            },
        );
    }

    group.finish();
}

criterion_group!(benches, bench_apply_gainmap, bench_compute_gainmap);
criterion_main!(benches);
