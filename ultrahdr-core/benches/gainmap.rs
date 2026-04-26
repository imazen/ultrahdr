//! Benchmarks for gain map operations.

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use std::hint::black_box;
use ultrahdr_core::{
    ColorPrimaries, GainMap, GainMapChannel, GainMapMetadata, PixelBuffer, PixelFormat,
    TransferFunction,
    gainmap::{
        apply::{HdrOutputFormat, apply_gainmap},
        compute::{GainMapConfig, compute_gainmap},
    },
    new_pixel_buffer,
};

/// Create a test SDR image of given dimensions.
fn create_sdr_image(width: u32, height: u32) -> PixelBuffer {
    let mut img = new_pixel_buffer(
        width,
        height,
        PixelFormat::Rgba8,
        ColorPrimaries::Bt709,
        TransferFunction::Srgb,
    )
    .unwrap();
    let stride = img.stride();
    let mut slice = img.as_slice_mut();
    let data = slice.as_strided_bytes_mut();
    for y in 0..height {
        for x in 0..width {
            let idx = ((y as usize) * stride + (x as usize) * 4).min(data.len() - 4);
            data[idx] = ((x * 255) / width.max(1)) as u8;
            data[idx + 1] = ((y * 255) / height.max(1)) as u8;
            data[idx + 2] = 128;
            data[idx + 3] = 255;
        }
    }
    drop(slice);
    img
}

/// Create a test HDR image (brighter version of SDR).
fn create_hdr_image(width: u32, height: u32) -> PixelBuffer {
    let mut img = new_pixel_buffer(
        width,
        height,
        PixelFormat::Rgba8,
        ColorPrimaries::Bt709,
        TransferFunction::Srgb,
    )
    .unwrap();
    let stride = img.stride();
    let mut slice = img.as_slice_mut();
    let data = slice.as_strided_bytes_mut();
    for y in 0..height {
        for x in 0..width {
            let idx = ((y as usize) * stride + (x as usize) * 4).min(data.len() - 4);
            data[idx] = (((x * 255) / width.max(1)) as u16).min(255) as u8;
            data[idx + 1] = (((y * 255) / height.max(1)) as u16 + 50).min(255) as u8;
            data[idx + 2] = 200;
            data[idx + 3] = 255;
        }
    }
    drop(slice);
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
        // log2 domain: 2× linear max → log2(2) = 1.0.
        let ch = GainMapChannel {
            min: 0.0,
            max: 1.0,
            gamma: 1.0,
            base_offset: 1.0 / 64.0,
            alternate_offset: 1.0 / 64.0,
        };
        let mut metadata = GainMapMetadata::default();
        metadata.channels = [ch; 3];
        metadata.base_hdr_headroom = 0.0;
        metadata.alternate_hdr_headroom = 1.0;
        metadata.use_base_color_space = true;

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
