# Context Handoff: WASM SIMD & Performance Testing

## Current State

Working on adding WASM SIMD128 support, benchmarking, and multi-runtime testing.

## Completed

1. **WASM SIMD128 build support** - Added `.cargo/config.toml` with `target-feature=+simd128` for wasm32 targets
2. **Justfile** - Common commands for development (wasm-build, wasm-test, ci, etc.)
3. **Criterion benchmarks** - `ultrahdr-core/benches/gainmap.rs` benchmarks apply_gainmap and compute_gainmap

### Baseline Native Performance (x86_64)
```
apply_gainmap/srgb8/1920x1080: ~175ms (11.8 Melem/s)
apply_gainmap/pq1010102/1920x1080: ~200ms (10.2 Melem/s)
compute_gainmap/luminance/1920x1080: ~8ms (265 Melem/s)
compute_gainmap/multichannel/1920x1080: ~10ms (212 Melem/s)
```

## In Progress

### Task 3: WASM Runtime Testing (wasmtime + wasmer)
- Created `wasm-bench/` crate - standalone WASI benchmark binary
- **NOT YET TESTED** - needs to be compiled to wasm32-wasip1 and run

To test:
```bash
rustup target add wasm32-wasip1
cargo build --package wasm-bench --target wasm32-wasip1 --release
wasmtime target/wasm32-wasip1/release/wasm-bench.wasm
wasmer target/wasm32-wasip1/release/wasm-bench.wasm
```

## Remaining Tasks

1. **Task 3** - Complete WASM runtime testing (wasmtime + wasmer)
2. **Task 4** - Measure WASM binary size (with/without SIMD, with/without wasm-opt)
3. **Task 5** - Add WASM SIMD CI testing (update .github/workflows/ci.yml)
4. **Task 6** - Test ARM (NEON) performance

## New Task: Investigate magetypes crate

The `magetypes` crate may provide accelerated color/pixel operations that could improve performance of the hot paths:
- `ultrahdr-core/src/gainmap/apply.rs` - apply_gainmap (11 Melem/s currently)
- `ultrahdr-core/src/gainmap/compute.rs` - compute_gainmap (265 Melem/s currently)
- `ultrahdr-core/src/color/convert.rs` - RGB↔YUV conversions
- `ultrahdr-core/src/color/transfer.rs` - sRGB EOTF/OETF, PQ EOTF/OETF

Research needed:
- Check if magetypes supports no_std
- Check WASM compatibility
- Evaluate API fit for ultrahdr-core use cases
- Compare performance with current scalar/wide implementations

## Key Files

- `.cargo/config.toml` - WASM SIMD rustflags
- `justfile` - Common commands
- `ultrahdr-core/benches/gainmap.rs` - Criterion benchmarks
- `wasm-bench/` - WASI benchmark binary (WIP)
- `.github/workflows/ci.yml` - CI config (needs WASM SIMD updates)

## Hot Paths for Optimization

The `wide` crate (v1.1.1) is declared as a dependency but **not currently used** in computation. The hot paths use scalar f32 operations:

1. `apply_gainmap` in `ultrahdr-core/src/gainmap/apply.rs:64-81` - pixel-by-pixel loop
2. `compute_gainmap` in `ultrahdr-core/src/gainmap/compute.rs:143-184` - gain computation loop
3. `srgb_eotf`/`srgb_oetf` in `ultrahdr-core/src/color/transfer.rs` - transfer functions
4. `pq_oetf`/`pq_eotf` in `ultrahdr-core/src/color/transfer.rs` - PQ functions

All these could benefit from SIMD vectorization using `wide` types like `f32x4` or `f32x8`.

## Commands

```bash
# Build WASM with SIMD
just wasm-build

# Run native benchmarks
cargo bench --package ultrahdr-core

# Test WASM build
cargo build --package wasm-bench --target wasm32-wasip1 --release

# Run all CI checks locally
just ci
```
