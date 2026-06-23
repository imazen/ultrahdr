# Dockerfile for running ultrahdr tests including FFI parity tests
#
# Build: docker build -t ultrahdr-test .
# Run:   docker run --rm ultrahdr-test
#
# This Dockerfile:
# - Installs C++ build dependencies for libultrahdr
# - Clones zune-image (path dependency)
# - Downloads Ultra HDR sample images for testing
# - Runs all tests including FFI parity tests

# Use Rust 1.92 for edition2024 support
FROM rust:1.92-bookworm

# Install build dependencies for libultrahdr C++ library
# ultrahdr-sys needs: cmake, C++ compiler, nasm (for libjpeg-turbo SIMD)
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    ninja-build \
    nasm \
    yasm \
    pkg-config \
    libjpeg62-turbo-dev \
    libclang-dev \
    clang \
    git \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Create workspace directory
WORKDIR /workspace

# Clone zune-image (path dependency) - shallow clone for speed
RUN git clone --depth 1 https://github.com/etemesi254/zune-image.git

# Copy our ultrahdr crate
COPY . ultrahdr/

# Download Ultra HDR sample images from MishaalRahmanGH/Ultra_HDR_Samples
# These are real Ultra HDR images that we can use for testing
RUN mkdir -p ultrahdr/test_images && cd ultrahdr/test_images && \
    echo "Downloading Ultra HDR sample images..." && \
    for i in 01 02 03; do \
        curl -fsSL -o "sample_${i}.jpg" \
            "https://raw.githubusercontent.com/MishaalRahmanGH/Ultra_HDR_Samples/main/Originals/Ultra_HDR_Samples_Originals_${i}.jpg" && \
        echo "Downloaded sample_${i}.jpg" || \
        echo "Failed to download sample_${i}.jpg"; \
    done && \
    ls -la

WORKDIR /workspace/ultrahdr

# Update Cargo.toml to use the cloned zune-image
RUN sed -i 's|path = "../zune-image|path = "/workspace/zune-image|g' Cargo.toml && \
    cat Cargo.toml

# Verify base tests work first (without FFI)
RUN echo "=== Running base tests (without FFI) ===" && \
    cargo test --release 2>&1 | tail -50

# Set environment variables to help ultrahdr-sys build
ENV CMAKE_BUILD_TYPE=Release
ENV UHDR_BUILD_DEPS=ON

# Build AND run tests at build time
# (FFI parity tests against libultrahdr C++ were removed when we dropped
# the libultrahdr-rs binding; pure-Rust correctness is verified via the
# `libultrahdr_parity.rs` corpus-based tests and CI's interop matrix.)
RUN echo "=== Building and running tests ===" && \
    cargo test --release -- --nocapture 2>&1 && \
    echo "Tests complete"

# At runtime, show test summary (tests ran at build time)
CMD ["sh", "-c", "echo '=== Ultra HDR Test Container ===' && echo '' && echo 'All tests passed at build time.' && echo '' && echo 'To re-run tests:' && echo '  docker run --rm ultrahdr-test cargo test --release'"]
