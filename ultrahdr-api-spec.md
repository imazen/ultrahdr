> **Partially outdated.** This spec was written before implementation. As of March 2026, dependency versions, feature flags, struct field names, and MPF ownership have changed. Corrections are inline below; see `ultrahdr-core/Cargo.toml` and `ultrahdr-core/src/types.rs` for ground truth.

# ultrahdr API Specification

## Overview

Pure Rust implementation of Ultra HDR (gain map HDR) computations. This crate handles **only**:
1. Gain map metadata parsing/generation (XMP, ISO 21496-1)
2. Pixel math for applying/computing gain maps
3. Tone mapping

**Not handled** (delegated to JPEG codec like zenjpeg):
- JPEG decode/encode
- APP segment extraction/injection
- ICC profile handling

**Also handled by ultrahdr-core** (generalized container primitives):
- MPF parsing/assembly (`metadata/mpf.rs`)
- GContainer XMP parsing/generation (`metadata/xmp.rs`)

## Dependencies

```toml
[dependencies]
# No JPEG codec dependency - user provides pixels and metadata

# Cooperative cancellation
enough = "0.4"

# Math
wide = "1.1"           # SIMD
half = "2.7"           # f16 support

# SIMD (optional)
archmage = { version = "0.9.4", optional = true }
magetypes = { version = "0.9.4", optional = true }

# Types
bytemuck = "1.14"

# Error handling
thiserror = "2.0"

# Transfer functions (optional)
linear-srgb = { version = "0.6.5", optional = true }

[dev-dependencies]
proptest = "1.10"
criterion = "0.8"
```

## Core Types

```rust
/// Gain map metadata (from XMP hdrgm namespace or ISO 21496-1).
/// All gain/headroom values are in log2 domain, f64 precision, per-channel [R, G, B].
#[derive(Clone, Debug, PartialEq)]
pub struct GainMapMetadata {
    /// Log2 of maximum gain per channel. 0.0 = no boost, 2.0 = 4x brightness.
    pub gain_map_max: [f64; 3],

    /// Log2 of minimum gain per channel. Can be negative (darkening).
    pub gain_map_min: [f64; 3],

    /// Gamma applied to the gain map encoding. Linear domain, must be > 0.
    pub gamma: [f64; 3],

    /// Offset added to base (SDR) values before gain application. Linear domain.
    pub base_offset: [f64; 3],

    /// Offset added to alternate (HDR) values before gain application. Linear domain.
    pub alternate_offset: [f64; 3],

    /// Log2 of base image HDR headroom. 0.0 = SDR (peak luminance ratio 1:1).
    pub base_hdr_headroom: f64,

    /// Log2 of alternate image HDR headroom.
    pub alternate_hdr_headroom: f64,

    /// Whether the gain map uses the base image color space.
    pub use_base_color_space: bool,
}

/// Configuration for gain map computation
#[derive(Clone, Debug)]
pub struct GainMapConfig {
    /// Target display peak luminance (nits)
    /// Default: 1000.0
    pub target_peak_nits: f32,

    /// Gain map scale factor (1 = full resolution, 4 = 1/4 resolution)
    /// Default: 4
    pub scale: u32,

    /// Whether to use per-channel gain (more accurate) or single channel (smaller)
    /// Default: false (single channel)
    pub per_channel: bool,

    /// Gamma for gain map encoding
    /// Default: 1.0
    pub gamma: f32,
}

/// Configuration for tone mapping
#[derive(Clone, Debug)]
pub struct TonemapConfig {
    /// Target peak luminance (nits)
    /// Default: 203.0 (SDR reference white)
    pub target_peak_nits: f32,

    /// Tone mapping operator
    /// Default: Reinhard
    pub operator: TonemapOperator,

    /// Output color space
    /// Default: Srgb
    pub output_space: ColorSpace,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TonemapOperator {
    /// Simple Reinhard (x / (1 + x))
    Reinhard,
    /// Filmic (ACES-like)
    Filmic,
    /// BT.2390 EETF (broadcast standard)
    Bt2390,
    /// Clip only (no tone mapping)
    Clip,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ColorSpace {
    Srgb,
    DisplayP3,
    Bt2020,
}
```

## Metadata Parsing/Generation

```rust
// === XMP (hdrgm namespace) ===

/// Parse gain map metadata from XMP string
///
/// Extracts hdrgm:* attributes from XMP. Works with both
/// standalone XMP and XMP embedded in larger documents.
pub fn parse_xmp(xmp: &str) -> Result<GainMapMetadata, ParseError>;

/// Generate XMP string with gain map metadata
///
/// Creates minimal XMP with hdrgm namespace. Can be merged
/// with existing XMP by the caller.
pub fn generate_xmp(metadata: &GainMapMetadata) -> String;

/// Generate XMP Container directory for MPF
///
/// Creates the Container:Directory element pointing to gain map.
/// `gainmap_size` is the byte size of the gain map JPEG.
pub fn generate_xmp_container(metadata: &GainMapMetadata, gainmap_size: usize) -> String;

// === ISO 21496-1 (binary format) ===

/// Parse gain map metadata from ISO 21496-1 binary format
pub fn parse_iso21496(data: &[u8]) -> Result<GainMapMetadata, ParseError>;

/// Generate ISO 21496-1 binary metadata
pub fn generate_iso21496(metadata: &GainMapMetadata) -> Vec<u8>;

// === Utilities ===

/// Check if XMP contains gain map metadata
pub fn xmp_has_gainmap(xmp: &str) -> bool;

/// Extract hdrgm namespace from XMP, leaving other namespaces intact
pub fn extract_gainmap_xmp(xmp: &str) -> Option<String>;

/// Merge gain map XMP into existing XMP document
pub fn merge_xmp(existing: &str, gainmap_xmp: &str) -> String;

/// Remove gain map metadata from XMP
pub fn strip_gainmap_xmp(xmp: &str) -> String;
```

## Streaming APIs

For memory-constrained environments, streaming APIs process images row-by-row:

```rust
// === Streaming Decode (SDR + Gain Map → HDR) ===

/// Row-based decoder with full gainmap in memory.
/// Best when gainmap is small (e.g., 1/4 resolution).
pub struct RowDecoder { /* ... */ }

/// Streaming decoder with ring buffer for gainmap.
/// Best for parallel JPEG decode with minimal memory.
pub struct StreamDecoder { /* ... */ }

/// Input configuration for decoders.
pub struct DecodeInput {
    pub format: PixelFormat,
    pub stride: u32,
    pub y_only: bool,
}

// === Streaming Encode (HDR + SDR → Gain Map) ===

/// Row-based encoder for synchronized HDR+SDR input.
/// Best when both inputs come from the same decode loop.
pub struct RowEncoder { /* ... */ }

/// Streaming encoder for independent HDR/SDR streams.
/// Best for parallel decode of separate sources.
pub struct StreamEncoder { /* ... */ }

/// Input configuration for encoders.
pub struct EncodeInput {
    pub hdr_format: PixelFormat,
    pub hdr_stride: u32,
    pub hdr_transfer: ColorTransfer,
    pub hdr_gamut: ColorGamut,
    pub sdr_format: PixelFormat,
    pub sdr_stride: u32,
    pub sdr_gamut: ColorGamut,
    pub y_only: bool,
}
```

### Memory Comparison (4K image, 3840×2160)

| API | Peak Memory |
|-----|-------------|
| Full decode | ~166 MB |
| Streaming decode (16 rows) | ~2 MB |
| Full encode | ~170 MB |
| Streaming encode (16 rows) | ~4 MB |

## Pixel Math

```rust
// === Apply Gain Map (SDR + Gain Map → HDR) ===

/// Apply gain map to reconstruct HDR
///
/// # Arguments
/// * `sdr` - SDR pixels (u8 RGB/RGBA, assumed sRGB)
/// * `sdr_width`, `sdr_height` - SDR dimensions
/// * `gainmap` - Gain map pixels (u8 grayscale or RGB)
/// * `gm_width`, `gm_height` - Gain map dimensions (may differ from SDR)
/// * `metadata` - Gain map parameters
/// * `boost` - Display boost factor (1.0 = SDR, 4.0 = 4x brighter)
///
/// # Returns
/// Linear float RGB pixels (0.0 - boost range)
pub fn apply_gainmap(
    sdr: &[u8],
    sdr_width: u32,
    sdr_height: u32,
    gainmap: &[u8],
    gm_width: u32,
    gm_height: u32,
    metadata: &GainMapMetadata,
    boost: f32,
) -> Vec<f32>;

/// Apply gain map with output format control
pub fn apply_gainmap_to<O: OutputFormat>(
    sdr: &[u8],
    sdr_width: u32,
    sdr_height: u32,
    gainmap: &[u8],
    gm_width: u32,
    gm_height: u32,
    metadata: &GainMapMetadata,
    boost: f32,
) -> O::Output;

/// Output format trait for apply_gainmap_to
pub trait OutputFormat {
    type Output;
    fn convert(linear_rgb: &[f32], width: u32, height: u32) -> Self::Output;
}

/// Linear float RGB output
pub struct LinearF32;
/// PQ-encoded 10-bit output
pub struct Pq10Bit;
/// HLG-encoded output
pub struct Hlg;

// === Compute Gain Map (HDR + SDR → Gain Map) ===

/// Compute gain map from HDR and SDR images
///
/// # Arguments
/// * `hdr` - HDR pixels (linear float RGB, may exceed 1.0)
/// * `sdr` - SDR pixels (u8 RGB, sRGB)
/// * `width`, `height` - Image dimensions (must match)
/// * `config` - Gain map computation settings
///
/// # Returns
/// * Gain map pixels (u8 grayscale)
/// * Gain map dimensions (may be scaled down)
/// * Computed metadata
pub fn compute_gainmap(
    hdr: &[f32],
    sdr: &[u8],
    width: u32,
    height: u32,
    config: &GainMapConfig,
) -> (Vec<u8>, u32, u32, GainMapMetadata);

/// Compute gain map from HDR only (uses internal tonemapping for SDR)
pub fn compute_gainmap_from_hdr(
    hdr: &[f32],
    width: u32,
    height: u32,
    config: &GainMapConfig,
    tonemap_config: &TonemapConfig,
) -> (Vec<u8>, Vec<u8>, u32, u32, GainMapMetadata);
// Returns: (sdr_pixels, gainmap_pixels, gm_width, gm_height, metadata)

// === Tone Mapping (HDR → SDR) ===

/// Tone map HDR to SDR
///
/// # Arguments
/// * `hdr` - Linear float RGB pixels
/// * `width`, `height` - Dimensions
/// * `config` - Tone mapping settings
///
/// # Returns
/// sRGB u8 pixels
pub fn tonemap(
    hdr: &[f32],
    width: u32,
    height: u32,
    config: &TonemapConfig,
) -> Vec<u8>;

/// Inverse tone map SDR to pseudo-HDR
/// (Approximation - cannot recover clipped highlights)
pub fn inverse_tonemap(
    sdr: &[u8],
    width: u32,
    height: u32,
    config: &TonemapConfig,
) -> Vec<f32>;
```

## Usage Examples

### Decode UltraHDR (with zenjpeg)

```rust
use zenjpeg::decoder::{Decoder, PixelFormat};

// zenjpeg decodes JPEG and preserves metadata
let decoded = Decoder::new()
    .output_format(PixelFormat::Rgb)
    .decode(&ultrahdr_bytes)?;

let extras = decoded.extras().unwrap();

// ultrahdr parses XMP
let metadata = ultrahdr::parse_xmp(extras.xmp().unwrap())?;

// zenjpeg decodes gain map
let gm_jpeg = extras.gainmap().unwrap();
let gainmap = Decoder::new()
    .output_format(PixelFormat::Gray)
    .decode(gm_jpeg)?;

// ultrahdr does the math
let hdr = ultrahdr::apply_gainmap(
    &decoded.data, decoded.width, decoded.height,
    &gainmap.data, gainmap.width, gainmap.height,
    &metadata,
    4.0,  // 4x boost
);
```

### Encode UltraHDR (with zenjpeg)

```rust
use zenjpeg::encoder::{EncoderConfig, ChromaSubsampling, PixelLayout};

// ultrahdr computes gain map
let (gm_pixels, gm_w, gm_h, metadata) = ultrahdr::compute_gainmap(
    &hdr_pixels, &sdr_pixels, width, height,
    &ultrahdr::GainMapConfig::default(),
);

// ultrahdr generates XMP
let xmp = ultrahdr::generate_xmp_container(&metadata, estimated_gm_size);

// zenjpeg encodes gain map
let gm_jpeg = EncoderConfig::grayscale(75.0)
    .encode_oneshot(&gm_pixels, gm_w, gm_h, PixelLayout::Gray8Srgb)?;

// zenjpeg encodes primary with metadata and MPF
let ultrahdr_jpeg = EncoderConfig::ycbcr(90.0, ChromaSubsampling::Quarter)
    .with_xmp(&xmp)
    .with_icc(srgb_icc)
    .add_gainmap(gm_jpeg)
    .encode_oneshot(&sdr_pixels, width, height, PixelLayout::Rgb8Srgb)?;
```

### HDR-only Input (Auto SDR)

```rust
// Just have HDR, need full UltraHDR output
let (sdr_pixels, gm_pixels, gm_w, gm_h, metadata) = ultrahdr::compute_gainmap_from_hdr(
    &hdr_pixels, width, height,
    &ultrahdr::GainMapConfig::default(),
    &ultrahdr::TonemapConfig::default(),
);

// Now encode as above...
```

### Round-Trip Edit SDR Only

```rust
// Decode
let decoded = Decoder::new().decode(&original)?;
let extras = decoded.extras().unwrap();

// Edit SDR
let edited_sdr = adjust_exposure(&decoded.data);

// Re-encode with same metadata and gain map
let mut segments = extras.to_encoder_segments();

let output = EncoderConfig::ycbcr(90.0, ChromaSubsampling::Quarter)
    .with_segments(segments)
    .encode_oneshot(&edited_sdr, w, h, PixelLayout::Rgb8Srgb)?;
```

### Edit Gain Map Directly

```rust
// Decode
let decoded = Decoder::new().decode(&original)?;
let extras = decoded.extras().unwrap();
let gm_jpeg = extras.gainmap().unwrap();

// Decode and edit gain map
let mut gainmap = Decoder::new()
    .output_format(PixelFormat::Gray)
    .decode(gm_jpeg)?;

// Paint highlights onto gain map
paint_highlights(&mut gainmap.data);

// Re-encode gain map
let new_gm_jpeg = EncoderConfig::grayscale(75.0)
    .encode_oneshot(&gainmap.data, gainmap.width, gainmap.height, PixelLayout::Gray8Srgb)?;

// Re-encode primary with new gain map
let mut segments = extras.to_encoder_segments();
segments.clear_mpf_images();
segments.add_gainmap(new_gm_jpeg);

let output = EncoderConfig::ycbcr(90.0, ChromaSubsampling::Quarter)
    .with_segments(segments)
    .encode_oneshot(&decoded.data, w, h, PixelLayout::Rgb8Srgb)?;
```

## Error Types

```rust
#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("XMP parse error: {0}")]
    XmpParse(String),

    #[error("ISO 21496-1 parse error: {0}")]
    IsoParse(String),

    #[error("Invalid metadata: {0}")]
    InvalidMetadata(String),

    #[error("Dimension mismatch: expected {expected}, got {got}")]
    DimensionMismatch { expected: String, got: String },

    #[error("Invalid pixel data: {0}")]
    InvalidPixelData(String),
}
```

## Feature Flags (ultrahdr-core)

```toml
[features]
default = ["std", "transfer"]
std = ["enough/std", "archmage?/std", "magetypes?/std", "linear-srgb?/std"]
simd = ["archmage", "magetypes"]        # Enable explicit SIMD optimizations
transfer = ["dep:linear-srgb"]          # Transfer functions (sRGB, PQ, HLG)
zencodec = ["dep:zenpixels", "dep:zencodec"]  # zen* ecosystem interop
```

## Feature Flags (ultrahdr-rs)

```toml
[features]
default = []
simd = ["ultrahdr-core/simd"]           # Enable explicit SIMD optimizations
ffi-tests = ["libultrahdr"]             # Enable FFI parity tests with libultrahdr C++
zencodec = ["ultrahdr-core/zencodec", "dep:zencodec", "dep:zenpixels", "dep:zenjpeg"]
```

## What ultrahdr Does NOT Do

| Responsibility | Handled By |
|---------------|------------|
| JPEG decode | zenjpeg or user's codec |
| JPEG encode | zenjpeg or user's codec |
| APP segment extraction | zenjpeg `DecodedExtras` |
| APP segment injection | zenjpeg `EncoderSegments` |
| MPF parsing | ultrahdr-core (`metadata/mpf.rs`) |
| MPF assembly | ultrahdr-core (`metadata/mpf.rs`) |
| GContainer XMP | ultrahdr-core (`metadata/xmp.rs`) |
| ICC profile handling | zenjpeg or user |
| XMP beyond hdrgm | User's XMP library |
| EXIF handling | User's EXIF library |

ultrahdr is deliberately minimal: **metadata + math, nothing else**.
