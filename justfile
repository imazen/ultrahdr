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

# Format code (also regenerates the public-API surface snapshots).
# The snapshot runner lives in the workspace-excluded apidoc/ package, so it
# is never built or run by plain `cargo test` or any CI job.
fmt:
    cargo fmt --all
    cargo test --manifest-path apidoc/Cargo.toml

# Regenerate the public-API surface snapshots (docs/public-api/) only
api-doc:
    cargo test --manifest-path apidoc/Cargo.toml

# Verify the committed snapshots are current
api-doc-check:
    ZEN_API_DOC=check cargo test --manifest-path apidoc/Cargo.toml

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

# Test feature permutations
feature-check:
    cargo test --workspace --all-targets --features simd
    cargo build --package ultrahdr-core --no-default-features

# CI check (runs all CI steps locally)
ci: fmt-check clippy test feature-check doc

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

# Shared metadata wrappers must retain the published ultrahdr-owned API.
metadata-check:
    cargo test -p ultrahdr-core --lib metadata
    cargo check -p ultrahdr-core --no-default-features
    cargo clippy -p ultrahdr-core --lib -- -D warnings

metadata-semver:
    cargo semver-checks check-release --manifest-path ultrahdr-core/Cargo.toml -p ultrahdr-core --baseline-version 0.6.0 --all-features

metadata-consumer-check:
    cargo test -p ultrahdr-core --lib
    cargo check -p ultrahdr-core --no-default-features --target wasm32-wasip1
    cargo check --manifest-path fuzz/Cargo.toml --bins

metadata-fuzz-check:
    cargo update --manifest-path fuzz/Cargo.toml
    cargo check --manifest-path fuzz/Cargo.toml --bins

metadata-regression-check:
    cargo test -p ultrahdr-rs --test fuzz_regression
    cargo check --manifest-path fuzz/Cargo.toml --bins

metadata-simd-ci:
    cargo test -p ultrahdr-core --all-targets --features simd
    cargo clippy -p ultrahdr-core --all-targets --features simd -- -D warnings

# Exercise the coordinated component filter before its replacement PR is landed.
# The explicit override keeps this local integration check reviewable.
metadata-policy-integration:
    ULTRAHDR_METADATA_ORACLE=exiftool cargo test -p ultrahdr-rs --test encoder_production
    cargo clippy -p ultrahdr-rs --all-targets -- -D warnings

metadata-policy-api:
    #!/usr/bin/env bash
    set -euo pipefail
    baseline=$(cargo read ultrahdr-rs==0.3.5 --path-only)
    RUSTC_BOOTSTRAP=1 cargo rustdoc -p ultrahdr-rs --lib --features zencodec -- -Zunstable-options --output-format json
    RUSTC_BOOTSTRAP=1 cargo rustdoc --manifest-path "$baseline/Cargo.toml" --target-dir "$PWD/target/published-api/ultrahdr-rs-0.3.5" --lib --features zencodec -- -Zunstable-options --output-format json
    cargo semver-checks check-release -p ultrahdr-rs --current-rustdoc target/doc/ultrahdr_rs.json --baseline-rustdoc target/published-api/ultrahdr-rs-0.3.5/doc/ultrahdr_rs.json --release-type patch

metadata-policy-targets:
    CARGO_TARGET_I686_UNKNOWN_LINUX_GNU_RUNNER=env cargo test -p ultrahdr-rs --target i686-unknown-linux-gnu --test encoder_production
    cargo test -p ultrahdr-rs --doc
    just api-doc

metadata-directory-dependency-check:
    just metadata-check
    cargo check --workspace

# Check one shared metadata type identity through HDR and JPEG consumers.
shared-contract-check:
    cargo test -p ultrahdr-core --lib
    cargo check -p ultrahdr-rs --features zencodec
    cargo clippy -p ultrahdr-core --lib -- -D warnings

# One gain-map type identity in core, JPEG consumers and standalone fuzz builds.
contract-pin-check:
    cargo fmt -p ultrahdr-core -p ultrahdr-rs
    just shared-contract-check
    just metadata-simd-ci
    cargo check --manifest-path fuzz/Cargo.toml --bins
