# Context Handoff: WASM SIMD & Performance Testing

## Completed This Session

### 1. no_std Support for ultrahdr-core
- Added `#![cfg_attr(not(feature = "std"), no_std)]` to lib.rs
- Replaced std imports with alloc/core equivalents in all modules
- Fixed `#[from]` attribute on StopReason (requires std::error::Error)
- All tests pass in both std and no_std configurations

### 2. WASM SIMD128 Build Support
- `.cargo/config.toml` enables `+simd128` for all wasm32 targets
- Builds successfully for wasm32-unknown-unknown and wasm32-wasip1

### 3. WASM Runtime Testing with wasmtime
Successfully tested with wasmtime. Results (1920x1080):

| Benchmark | Native (x86_64) | WASM (wasmtime) | % of Native |
|-----------|-----------------|-----------------|-------------|
| compute_gainmap/luminance | 8.0ms (257 Melem/s) | 9.7ms (213 Melem/s) | 83% |
| compute_gainmap/multichannel | 9.6ms (215 Melem/s) | ~10ms (213 Melem/s) | ~99% |
| apply_gainmap/srgb8 | **99ms (21 Melem/s)** | **121ms (17 Melem/s)** | 87% |

**Excellent WASM performance - 83-99% of native!**

### 4. GainMapLut Performance Optimization
Added precomputed lookup table that eliminates per-pixel `powf()` and `exp()` calls:

| Target | Before | After | Speedup |
|--------|--------|-------|---------|
| Native sRGB | 175ms (11.8 Mp/s) | 99ms (21 Mp/s) | **32%** |
| Native PQ | 200ms (10.2 Mp/s) | 153ms (13 Mp/s) | **25%** |
| WASM sRGB | 193ms (10.7 Mp/s) | 121ms (17 Mp/s) | **37%** |

### 5. WASM Binary Size
- wasm-bench.wasm: **127KB** (release, LTO enabled)
- Size is identical with/without SIMD128 flag (no explicit SIMD intrinsics yet)

### 6. CI Updates
- Added wasm32-wasip1 target to WASM build job
- Added wasmtime smoke test for wasm-bench
- Reports binary size in CI output

### 7. ARM Cross-Compilation
- Verified aarch64-unknown-linux-gnu builds successfully
- Native ARM testing already in CI (ubuntu-24.04-arm)

## Investigation: magetypes Crate

Found promising SIMD crate for hot path optimization:
- **magetypes** v0.1.0: Token-gated SIMD types with natural operators
- Cross-platform: AVX2, AVX-512, NEON, WASM SIMD128
- no_std compatible
- **Requires Rust 1.89+** (ultrahdr-core MSRV is 1.75)

### Recommended Approach
Add optional `simd` feature that:
1. Requires Rust 1.89+
2. Depends on `magetypes` and `archmage`
3. Provides optimized implementations of hot paths:
   - `apply_gainmap` (currently 11 Melem/s - slow due to pow/exp)
   - Transfer functions (sRGB, PQ, HLG EOTF/OETF)
   - Color space conversions

### Hot Paths (by impact)

1. **apply_gainmap** - bottleneck is `powf()` in gain application
   - Current: pixel-by-pixel scalar f32 with powf
   - Needs: LUT-based pow approximation or polynomial SIMD

2. **Transfer functions** - already have LUT implementations
   - `SrgbLut`, `SrgbInverseLut` exist but aren't used in hot paths
   - Could SIMD-ify the interpolation

3. **compute_gainmap** - already fast (200+ Melem/s)
   - Lower priority for optimization

## Remaining Tasks

1. **Investigate wasmer** - network issues prevented installation
2. **wasm-opt optimization** - measure size reduction with binaryen
3. **Explicit SIMD** - magetypes requires Rust 1.89+ (current MSRV is 1.75)
4. **Streaming API LUT** - streaming.rs still uses per-pixel decode_gain

## Key Files

- `ultrahdr-core/src/lib.rs` - no_std gate
- `ultrahdr-core/src/types.rs` - Error type without #[from]
- `.cargo/config.toml` - WASM SIMD rustflags
- `.github/workflows/ci.yml` - WASI testing
- `wasm-bench/` - WASI benchmark binary
- `ultrahdr-core/benches/gainmap.rs` - Criterion benchmarks

## Commands

```bash
# Build WASM with SIMD
cargo build --package wasm-bench --target wasm32-wasip1 --release

# Run WASM benchmark
wasmtime target/wasm32-wasip1/release/wasm-bench.wasm

# Run native benchmarks
cargo bench --package ultrahdr-core --bench gainmap

# Test no_std build
cargo build --package ultrahdr-core --target wasm32-wasip1 --no-default-features

# Run all CI checks
just ci
```
