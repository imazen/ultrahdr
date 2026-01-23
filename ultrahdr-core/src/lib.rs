//! Core gain map math and metadata for Ultra HDR.
//!
//! This crate provides the pure computational components for Ultra HDR:
//! - Gain map metadata parsing/generation (XMP, ISO 21496-1)
//! - Pixel math for applying/computing gain maps
//! - Tone mapping (HDR → SDR)
//! - Color space conversions and transfer functions
//!
//! This crate has **no JPEG codec dependency**. For full Ultra HDR encode/decode,
//! use the `ultrahdr` crate which provides codec integration.
//!
//! # Cooperative Cancellation
//!
//! Long-running operations accept an `impl Stop` parameter from the `enough` crate
//! for cooperative cancellation. Use `Unstoppable` when cancellation is not needed.
//!
//! # Example
//!
//! ```ignore
//! use ultrahdr_core::{
//!     gainmap::{apply_gainmap, compute_gainmap, GainMapConfig, HdrOutputFormat},
//!     metadata::xmp::{parse_xmp, generate_xmp},
//!     GainMap, GainMapMetadata, RawImage,
//! };
//! use enough::Unstoppable;
//!
//! // Compute gain map from HDR and SDR images
//! let config = GainMapConfig::default();
//! let (gainmap, metadata) = compute_gainmap(&hdr, &sdr, &config, Unstoppable)?;
//!
//! // Generate XMP metadata
//! let xmp = generate_xmp(&metadata, gainmap_jpeg_size);
//!
//! // Apply gain map to reconstruct HDR
//! let hdr_output = apply_gainmap(&sdr, &gainmap, &metadata, 4.0, HdrOutputFormat::LinearFloat, Unstoppable)?;
//! ```

#![warn(missing_docs)]
#![warn(clippy::all)]

pub mod color;
pub mod gainmap;
pub mod metadata;
mod types;

// Re-export core types
pub use types::{
    luminance, ColorGamut, ColorTransfer, Error, Fraction, GainMap, GainMapMetadata, PixelFormat,
    RawImage, Result,
};

// Re-export enough for convenience
pub use enough::{Stop, StopReason, Unstoppable};

// Re-export gain map types
pub use gainmap::{apply::HdrOutputFormat, compute::GainMapConfig};

/// Safety limits for parsing and allocation.
pub mod limits {
    /// Maximum XMP string length to parse (16 MB).
    pub const MAX_XMP_LENGTH: usize = 16 * 1024 * 1024;

    /// Maximum image dimension (width or height).
    pub const MAX_IMAGE_DIMENSION: u32 = 65535;

    /// Maximum total pixels (width * height).
    pub const MAX_TOTAL_PIXELS: u64 = 500_000_000; // 500 megapixels

    /// Maximum gain map metadata array length.
    pub const MAX_METADATA_ARRAY_LENGTH: usize = 1024;
}
