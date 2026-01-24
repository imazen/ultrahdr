//! Cross-platform SIMD benchmarks for gain map operations.
//!
//! Compares scalar LUT vs SIMD implementations on x86_64, aarch64, and wasm32.
//!
//! Run with:
//!   cargo bench --package ultrahdr-core --bench simd_xplat --features simd
//!
//! Cross-compile check:
//!   cargo build --package ultrahdr-core --bench simd_xplat --features simd --target aarch64-unknown-linux-gnu
//!   cargo build --package ultrahdr-core --bench simd_xplat --features simd --target wasm32-unknown-unknown

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};

// ============================================================================
// Scalar baseline - current LUT implementation
// ============================================================================

/// Precompute gain LUT (256 entries) - same as GainMapLut
fn build_scalar_lut(gamma: f32, log_min: f32, log_max: f32, weight: f32) -> [f32; 256] {
    let mut table = [0.0f32; 256];
    let log_range = log_max - log_min;

    for (i, entry) in table.iter_mut().enumerate() {
        let normalized = i as f32 / 255.0;
        let linear = if gamma != 1.0 {
            normalized.powf(1.0 / gamma)
        } else {
            normalized
        };
        let log_gain = log_min + linear * log_range;
        *entry = (log_gain * weight).exp();
    }
    table
}

/// Scalar LUT lookup with bilinear interpolation
fn apply_gain_scalar_lut(
    sdr: &[[f32; 3]],
    gainmap: &[u8],
    lut: &[f32; 256],
    output: &mut [[f32; 3]],
) {
    for (i, (sdr_px, out_px)) in sdr.iter().zip(output.iter_mut()).enumerate() {
        let g = lut[gainmap[i] as usize];
        out_px[0] = sdr_px[0] * g;
        out_px[1] = sdr_px[1] * g;
        out_px[2] = sdr_px[2] * g;
    }
}

// ============================================================================
// SIMD implementation using archmage/magetypes
// ============================================================================

#[cfg(feature = "simd")]
mod simd_impl {
    use super::*;

    #[cfg(target_arch = "x86_64")]
    pub fn apply_gain_simd(
        sdr: &[[f32; 3]],
        gainmap: &[u8],
        lut: &[f32; 256],
        output: &mut [[f32; 3]],
    ) {
        use archmage::SimdToken;

        if let Some(token) = archmage::Avx2FmaToken::try_new() {
            apply_gain_avx2(token, sdr, gainmap, lut, output);
        } else if let Some(token) = archmage::Sse41Token::try_new() {
            apply_gain_sse(token, sdr, gainmap, lut, output);
        } else {
            apply_gain_scalar_lut(sdr, gainmap, lut, output);
        }
    }

    #[cfg(target_arch = "x86_64")]
    #[archmage::arcane]
    fn apply_gain_avx2(
        token: archmage::Avx2FmaToken,
        sdr: &[[f32; 3]],
        gainmap: &[u8],
        lut: &[f32; 256],
        output: &mut [[f32; 3]],
    ) {
        use magetypes::f32x8;

        // Process 8 pixels at a time
        let chunks = sdr.len() / 8;

        for chunk_idx in 0..chunks {
            let base = chunk_idx * 8;

            // Gather gains from LUT (8 lookups)
            let gains: [f32; 8] = std::array::from_fn(|i| lut[gainmap[base + i] as usize]);
            let g = f32x8::from_array(token, gains);

            // Load R channel (strided - every 3rd element starting at 0)
            let r: [f32; 8] = std::array::from_fn(|i| sdr[base + i][0]);
            let r_v = f32x8::from_array(token, r);

            // Load G channel
            let g_ch: [f32; 8] = std::array::from_fn(|i| sdr[base + i][1]);
            let g_v = f32x8::from_array(token, g_ch);

            // Load B channel
            let b: [f32; 8] = std::array::from_fn(|i| sdr[base + i][2]);
            let b_v = f32x8::from_array(token, b);

            // Apply gain
            let r_out = r_v * g;
            let g_out = g_v * g;
            let b_out = b_v * g;

            // Store back (strided)
            let r_arr = r_out.to_array();
            let g_arr = g_out.to_array();
            let b_arr = b_out.to_array();
            for i in 0..8 {
                output[base + i] = [r_arr[i], g_arr[i], b_arr[i]];
            }
        }

        // Handle remainder
        let remainder_start = chunks * 8;
        for i in remainder_start..sdr.len() {
            let g = lut[gainmap[i] as usize];
            output[i][0] = sdr[i][0] * g;
            output[i][1] = sdr[i][1] * g;
            output[i][2] = sdr[i][2] * g;
        }
    }

    #[cfg(target_arch = "x86_64")]
    #[archmage::arcane]
    fn apply_gain_sse(
        token: archmage::Sse41Token,
        sdr: &[[f32; 3]],
        gainmap: &[u8],
        lut: &[f32; 256],
        output: &mut [[f32; 3]],
    ) {
        use magetypes::f32x4;

        // Process 4 pixels at a time
        let chunks = sdr.len() / 4;

        for chunk_idx in 0..chunks {
            let base = chunk_idx * 4;

            let gains: [f32; 4] = std::array::from_fn(|i| lut[gainmap[base + i] as usize]);
            let g = f32x4::from_array(token, gains);

            let r: [f32; 4] = std::array::from_fn(|i| sdr[base + i][0]);
            let r_v = f32x4::from_array(token, r);

            let g_ch: [f32; 4] = std::array::from_fn(|i| sdr[base + i][1]);
            let g_v = f32x4::from_array(token, g_ch);

            let b: [f32; 4] = std::array::from_fn(|i| sdr[base + i][2]);
            let b_v = f32x4::from_array(token, b);

            let r_out = r_v * g;
            let g_out = g_v * g;
            let b_out = b_v * g;

            let r_arr = r_out.to_array();
            let g_arr = g_out.to_array();
            let b_arr = b_out.to_array();
            for i in 0..4 {
                output[base + i] = [r_arr[i], g_arr[i], b_arr[i]];
            }
        }

        let remainder_start = chunks * 4;
        for i in remainder_start..sdr.len() {
            let g = lut[gainmap[i] as usize];
            output[i][0] = sdr[i][0] * g;
            output[i][1] = sdr[i][1] * g;
            output[i][2] = sdr[i][2] * g;
        }
    }

    #[cfg(target_arch = "aarch64")]
    pub fn apply_gain_simd(
        sdr: &[[f32; 3]],
        gainmap: &[u8],
        lut: &[f32; 256],
        output: &mut [[f32; 3]],
    ) {
        use archmage::SimdToken;

        if let Some(token) = archmage::NeonToken::try_new() {
            apply_gain_neon(token, sdr, gainmap, lut, output);
        } else {
            apply_gain_scalar_lut(sdr, gainmap, lut, output);
        }
    }

    #[cfg(target_arch = "aarch64")]
    fn apply_gain_neon(
        _token: archmage::NeonToken,
        sdr: &[[f32; 3]],
        gainmap: &[u8],
        lut: &[f32; 256],
        output: &mut [[f32; 3]],
    ) {
        // TODO: Implement NEON version when archmage ARM support is complete
        apply_gain_scalar_lut(sdr, gainmap, lut, output);
    }

    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    pub fn apply_gain_simd(
        sdr: &[[f32; 3]],
        gainmap: &[u8],
        lut: &[f32; 256],
        output: &mut [[f32; 3]],
    ) {
        apply_gain_scalar_lut(sdr, gainmap, lut, output);
    }

    /// SIMD exp using archmage transcendentals (no LUT)
    #[cfg(target_arch = "x86_64")]
    pub fn apply_gain_simd_exp(
        sdr: &[[f32; 3]],
        gainmap: &[u8],
        gamma: f32,
        log_min: f32,
        log_max: f32,
        weight: f32,
        output: &mut [[f32; 3]],
    ) {
        use archmage::SimdToken;

        if let Some(token) = archmage::Avx2FmaToken::try_new() {
            apply_gain_avx2_exp(token, sdr, gainmap, gamma, log_min, log_max, weight, output);
        } else {
            // Fallback to scalar with exp
            let log_range = log_max - log_min;
            for (i, (sdr_px, out_px)) in sdr.iter().zip(output.iter_mut()).enumerate() {
                let normalized = gainmap[i] as f32 / 255.0;
                let linear = if gamma != 1.0 {
                    normalized.powf(1.0 / gamma)
                } else {
                    normalized
                };
                let log_gain = log_min + linear * log_range;
                let g = (log_gain * weight).exp();
                out_px[0] = sdr_px[0] * g;
                out_px[1] = sdr_px[1] * g;
                out_px[2] = sdr_px[2] * g;
            }
        }
    }

    #[cfg(target_arch = "x86_64")]
    #[archmage::arcane]
    fn apply_gain_avx2_exp(
        token: archmage::Avx2FmaToken,
        sdr: &[[f32; 3]],
        gainmap: &[u8],
        gamma: f32,
        log_min: f32,
        log_max: f32,
        weight: f32,
        output: &mut [[f32; 3]],
    ) {
        use magetypes::f32x8;

        let log_range = log_max - log_min;
        let inv_gamma = 1.0 / gamma;
        let log_min_v = f32x8::splat(token, log_min);
        let log_range_v = f32x8::splat(token, log_range);
        let weight_v = f32x8::splat(token, weight);
        let scale = f32x8::splat(token, 1.0 / 255.0);

        let chunks = sdr.len() / 8;

        for chunk_idx in 0..chunks {
            let base = chunk_idx * 8;

            // Convert gainmap bytes to normalized floats
            let gm: [f32; 8] = std::array::from_fn(|i| gainmap[base + i] as f32);
            let normalized = f32x8::from_array(token, gm) * scale;

            // Apply gamma (use pow_midp for better accuracy)
            let linear = if gamma != 1.0 {
                normalized.pow_midp(inv_gamma)
            } else {
                normalized
            };

            // Compute log_gain = log_min + linear * log_range
            let log_gain = linear.mul_add(log_range_v, log_min_v);

            // Compute gain = exp(log_gain * weight)
            let g = (log_gain * weight_v).exp_midp();

            // Apply to RGB
            let r: [f32; 8] = std::array::from_fn(|i| sdr[base + i][0]);
            let g_ch: [f32; 8] = std::array::from_fn(|i| sdr[base + i][1]);
            let b: [f32; 8] = std::array::from_fn(|i| sdr[base + i][2]);

            let r_out = f32x8::from_array(token, r) * g;
            let g_out = f32x8::from_array(token, g_ch) * g;
            let b_out = f32x8::from_array(token, b) * g;

            let r_arr = r_out.to_array();
            let g_arr = g_out.to_array();
            let b_arr = b_out.to_array();
            for i in 0..8 {
                output[base + i] = [r_arr[i], g_arr[i], b_arr[i]];
            }
        }

        // Remainder
        let remainder_start = chunks * 8;
        for i in remainder_start..sdr.len() {
            let normalized = gainmap[i] as f32 / 255.0;
            let linear = if gamma != 1.0 {
                normalized.powf(inv_gamma)
            } else {
                normalized
            };
            let log_gain = log_min + linear * log_range;
            let g = (log_gain * weight).exp();
            output[i][0] = sdr[i][0] * g;
            output[i][1] = sdr[i][1] * g;
            output[i][2] = sdr[i][2] * g;
        }
    }

    #[cfg(not(target_arch = "x86_64"))]
    pub fn apply_gain_simd_exp(
        sdr: &[[f32; 3]],
        gainmap: &[u8],
        gamma: f32,
        log_min: f32,
        log_max: f32,
        weight: f32,
        output: &mut [[f32; 3]],
    ) {
        let log_range = log_max - log_min;
        for (i, (sdr_px, out_px)) in sdr.iter().zip(output.iter_mut()).enumerate() {
            let normalized = gainmap[i] as f32 / 255.0;
            let linear = if gamma != 1.0 {
                normalized.powf(1.0 / gamma)
            } else {
                normalized
            };
            let log_gain = log_min + linear * log_range;
            let g = (log_gain * weight).exp();
            out_px[0] = sdr_px[0] * g;
            out_px[1] = sdr_px[1] * g;
            out_px[2] = sdr_px[2] * g;
        }
    }
}

// ============================================================================
// Benchmarks
// ============================================================================

fn bench_gain_apply(c: &mut Criterion) {
    let mut group = c.benchmark_group("gain_apply");

    // Test parameters
    let gamma = 1.0f32;
    let weight = 1.0f32;
    let log_min = 1.0f32.ln(); // 0.0
    let log_max = 4.0f32.ln(); // ~1.386

    for size in [(512, 512), (1920, 1080)] {
        let (width, height) = size;
        let pixel_count = width * height;

        // Generate test data
        let sdr: Vec<[f32; 3]> = (0..pixel_count)
            .map(|i| {
                let v = (i % 256) as f32 / 255.0;
                [v, v, v]
            })
            .collect();

        let gainmap: Vec<u8> = (0..pixel_count).map(|i| ((i * 7) % 256) as u8).collect();

        let lut = build_scalar_lut(gamma, log_min, log_max, weight);

        let mut output: Vec<[f32; 3]> = vec![[0.0; 3]; pixel_count];

        group.throughput(Throughput::Elements(pixel_count as u64));

        // Scalar LUT benchmark
        group.bench_with_input(
            BenchmarkId::new("scalar_lut", format!("{}x{}", width, height)),
            &pixel_count,
            |b, _| {
                b.iter(|| {
                    apply_gain_scalar_lut(
                        black_box(&sdr),
                        black_box(&gainmap),
                        black_box(&lut),
                        black_box(&mut output),
                    );
                });
            },
        );

        // SIMD LUT benchmark
        #[cfg(feature = "simd")]
        group.bench_with_input(
            BenchmarkId::new("simd_lut", format!("{}x{}", width, height)),
            &pixel_count,
            |b, _| {
                b.iter(|| {
                    simd_impl::apply_gain_simd(
                        black_box(&sdr),
                        black_box(&gainmap),
                        black_box(&lut),
                        black_box(&mut output),
                    );
                });
            },
        );

        // SIMD with transcendentals (no LUT)
        #[cfg(feature = "simd")]
        group.bench_with_input(
            BenchmarkId::new("simd_exp", format!("{}x{}", width, height)),
            &pixel_count,
            |b, _| {
                b.iter(|| {
                    simd_impl::apply_gain_simd_exp(
                        black_box(&sdr),
                        black_box(&gainmap),
                        black_box(gamma),
                        black_box(log_min),
                        black_box(log_max),
                        black_box(weight),
                        black_box(&mut output),
                    );
                });
            },
        );
    }

    group.finish();
}

fn bench_platform_info(c: &mut Criterion) {
    let mut group = c.benchmark_group("platform_info");
    group.sample_size(10);

    group.bench_function("detect_features", |b| {
        b.iter(|| {
            #[cfg(target_arch = "x86_64")]
            {
                use archmage::SimdToken;
                black_box(archmage::Avx2FmaToken::try_new().is_some());
            }
            #[cfg(target_arch = "aarch64")]
            {
                use archmage::SimdToken;
                black_box(archmage::NeonToken::try_new().is_some());
            }
        });
    });

    group.finish();
}

criterion_group!(benches, bench_gain_apply, bench_platform_info);
criterion_main!(benches);
