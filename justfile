# Justfile for ultrahdr project

# Default target
default:
    @just --list

# Build everything
build:
    cargo build --workspace --all-targets

# Run all tests
test:
    cargo test --workspace --all-targets

# Run clippy
clippy:
    cargo clippy --workspace --all-targets -- -D warnings

# Check formatting
fmt-check:
    cargo fmt --all --check

# Format code
fmt:
    cargo fmt --all

# Build for WASM (with SIMD enabled via .cargo/config.toml)
wasm-build:
    cargo build --package ultrahdr-core --target wasm32-unknown-unknown --release

# Build WASM without SIMD (for comparison)
wasm-build-no-simd:
    RUSTFLAGS="" cargo build --package ultrahdr-core --target wasm32-unknown-unknown --release

# Run WASM tests via wasm-pack
wasm-test:
    wasm-pack test --node ultrahdr

# Check WASM binary size
wasm-size:
    @echo "=== WASM binary sizes (with SIMD) ==="
    @ls -lh target/wasm32-unknown-unknown/release/*.rlib 2>/dev/null || echo "No .rlib files"
    @ls -lh target/wasm32-unknown-unknown/release/*.wasm 2>/dev/null || echo "No .wasm files"

# Build for ARM (native on ARM, cross-compile otherwise)
arm-build:
    cargo build --workspace --all-targets --release

# Run benchmarks
bench:
    cargo bench --workspace

# Run benchmarks for a specific benchmark
bench-name NAME:
    cargo bench --workspace -- {{NAME}}

# Generate documentation
doc:
    cargo doc --workspace --no-deps

# Check MSRV
msrv:
    cargo +1.75 check --workspace

# CI check (runs all CI steps locally)
ci: fmt-check clippy test doc

# Clean build artifacts
clean:
    cargo clean

# Download test images
download-test-images:
    mkdir -p test-images
    cd test-images && curl -sLO https://github.com/user-attachments/files/17968428/ultrahdr_test_images.zip
    cd test-images && unzip -o ultrahdr_test_images.zip

# Run wasmtime tests (requires wasmtime installed)
wasm-wasmtime:
    @echo "Building WASM..."
    cargo build --package ultrahdr-core --target wasm32-wasip1 --release
    @echo "Note: wasmtime module execution requires a .wasm binary, not .rlib"

# Run wasmer tests (requires wasmer installed)
wasm-wasmer:
    @echo "Building WASM..."
    cargo build --package ultrahdr-core --target wasm32-wasip1 --release
    @echo "Note: wasmer module execution requires a .wasm binary, not .rlib"
