# Context Handoff: WASM SIMD & Performance Testing

## Session Summary

This session completed WASM SIMD support, performance testing, and cross-platform SIMD investigation.

## Completed

### 1. no_std Support
- Added `#![cfg_attr(not(feature = "std"), no_std)]` to ultrahdr-core
- All tests pass in both std and no_std configurations

### 2. GainMapLut Performance Optimization
Precomputed LUT eliminates per-pixel `powf()` and `exp()` calls:

| Target | Before | After | Speedup |
|--------|--------|-------|---------|
| Native sRGB | 175ms | 99ms | **32%** |
| Native PQ | 200ms | 153ms | **25%** |
| WASM sRGB | 193ms | 121ms | **37%** |

### 3. WASM Runtime Testing (wasmtime)
| Benchmark | Native | WASM | % of Native |
|-----------|--------|------|-------------|
| compute_gainmap | 8.0ms | 9.7ms | 83% |
| apply_gainmap | 99ms | 121ms | 87% |

Binary size: **127KB**

### 4. MSRV Bump to 1.91
Enables use of archmage/magetypes crates for explicit SIMD.

### 5. Cross-Platform SIMD Investigation
Added `simd` feature with archmage/magetypes dependencies.
Created benchmark comparing:

| Approach | 1920x1080 | Throughput |
|----------|-----------|------------|
| scalar_lut | 3.0ms | 677 Melem/s |
| simd_lut (AVX2) | 3.2ms | 645 Melem/s |
| simd_exp (AVX2) | 3.0ms | 690 Melem/s |

**Key Finding**: AoS (Array of Structs) memory layout hurts SIMD due to gather/scatter overhead. Would need SoA layout to benefit from explicit SIMD.

### 6. CI Updates
- WASI benchmark testing with wasmtime
- MSRV check updated to 1.91

## Cross-Platform Build Verification
All targets build successfully with `--features simd`:
- x86_64-unknown-linux-gnu ✓
- aarch64-unknown-linux-gnu ✓
- wasm32-unknown-unknown ✓

## Commits Pushed
1. feat: add no_std support to ultrahdr-core
2. ci: add WASI benchmark testing with wasmtime
3. perf: add GainMapLut for 32-37% faster apply_gainmap
4. chore: bump MSRV to 1.91
5. feat: add cross-platform SIMD benchmark with archmage/magetypes

## Future Optimization Path
To benefit from explicit SIMD:
1. Convert hot paths from AoS to SoA layout
2. Process pixels in planar format (separate R, G, B arrays)
3. Use magetypes `f32x8` directly on contiguous data

The current scalar LUT is already very fast (677 Melem/s) because:
- LUT eliminates transcendentals
- Memory access pattern is sequential
- Auto-vectorization handles simple multiply-accumulate

## Commands
```bash
# Run SIMD benchmark
cargo bench --package ultrahdr-core --bench simd_xplat --features simd

# Run WASM benchmark
cargo build --package wasm-bench --target wasm32-wasip1 --release
wasmtime target/wasm32-wasip1/release/wasm-bench.wasm

# Cross-compile for ARM
cargo build --package ultrahdr-core --features simd --target aarch64-unknown-linux-gnu
```
