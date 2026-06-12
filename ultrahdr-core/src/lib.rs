//! Core gain map math for Ultra HDR.
//!
//! This crate provides the pure computational components for Ultra HDR:
//! - Pixel math for applying/computing gain maps
//! - Tone mapping (HDR → SDR)
//! - Color space conversions and transfer functions
//!
//! JPEG-container metadata (MPF, XMP, ISO 21496-1 APP2 envelope) previously
//! lived here in `ultrahdr_core::metadata`; it has moved to
//! `zenjpeg::container` (JPEG-specific parsing) and `zencodec::gainmap`
//! (codec-agnostic payload). See issue #8.
//!
//! This crate has **no JPEG codec dependency**. For full Ultra HDR encode/decode,
//! use the `ultrahdr` crate which provides codec integration.
//!
//! # no_std Support
//!
//! This crate is `no_std` compatible with alloc. Disable default features:
//! ```toml
//! ultrahdr-core = { version = "0.1", default-features = false }
//! ```
//!
//! # Cooperative Cancellation
//!
//! Long-running operations accept an `impl Stop` parameter from the `enough` crate
//! for cooperative cancellation. Use `Unstoppable` when cancellation is not needed.
//!
//! # Example — compute a gain map from an HDR/SDR pair
//!
//! ```
//! use ultrahdr_core::{
//!     ColorPrimaries, TransferFunction, PixelFormat, new_pixel_buffer, Unstoppable,
//!     gainmap::{apply_gainmap, compute_gainmap, GainMapConfig, HdrOutputFormat},
//! };
//!
//! // Minimal 8x8 matching HDR + SDR surfaces. In practice these come from
//! // your image decoder — this example just shows the call shape.
//! let hdr = new_pixel_buffer(
//!     8, 8, PixelFormat::Rgba8, ColorPrimaries::Bt709, TransferFunction::Srgb,
//! )?;
//! let sdr = new_pixel_buffer(
//!     8, 8, PixelFormat::Rgba8, ColorPrimaries::Bt709, TransferFunction::Srgb,
//! )?;
//!
//! // Derive gain map + metadata.
//! let config = GainMapConfig::default();
//! let (gainmap, metadata) = compute_gainmap(&hdr, &sdr, &config, Unstoppable)?;
//!
//! // For XMP / ISO 21496-1 APP2 serialization, use `zenjpeg::container::xmp`
//! // and `zencodec::gainmap`.
//!
//! // Reconstruct HDR at 4× boost.
//! let _hdr_out = apply_gainmap(
//!     &sdr, &gainmap, &metadata,
//!     4.0, HdrOutputFormat::LinearFloat, Unstoppable,
//! )?;
//! # Ok::<(), ultrahdr_core::Error>(())
//! ```

#![cfg_attr(not(feature = "std"), no_std)]
#![forbid(unsafe_code)]
#![warn(missing_docs)]
#![warn(clippy::all)]

extern crate alloc;

pub mod color;
pub mod gainmap;
pub mod metadata;
mod types;

// Re-export core types (local)
pub use types::{
    ColorPrimaries, Error, GainMap, GainMapEncodingFormat, PixelBuffer, PixelFormat, PixelSlice,
    PixelSliceMut, Result, TransferFunction, clone_pixel_buffer, descriptor_for, luminance,
    new_pixel_buffer, pixel_buffer_from_vec, require_supported_format, validate_gainmap_magnitude,
    validate_gainmap_metadata, validate_ultrahdr_dimensions, validate_ultrahdr_image,
    validate_ultrahdr_slice,
};

// Re-export from zencodec (canonical gain map metadata types)
pub use types::{
    Fraction, GainMapChannel, GainMapMetadata, Iso21496Format, UnsignedFraction,
    full_reconstruction_boost,
};
pub use zencodec::GainMapParams;
pub use zencodec::gainmap::{parse_iso21496_fmt, serialize_iso21496_fmt};

// Re-export enough for convenience
pub use enough::{Stop, StopReason, Unstoppable};

// Re-export gain map types
pub use gainmap::{apply::HdrOutputFormat, compute::GainMapConfig, compute::compute_gain_row};

// Re-export Apple MakerNote HDR extraction (HEIC/JPEG gain-map headroom)
pub use metadata::apple::{
    AppleHdrInfo, from_apple_headroom, parse_apple_makernote, parse_exif_for_apple_hdr,
};

// Splitter API lives in zentone — re-export at crate root for back-compat
// with `ultrahdr_core::LumaToneMap` etc. Gated behind the `tonemap` feature
// (default-on) so decoder-only consumers can build without pulling in zentone.
#[cfg(feature = "tonemap")]
pub use zentone::{
    Bt2408Yrgb, ExtendedReinhardLuma, HableFilmic, LumaFn, LumaGainMapSplitter, LumaToneMap,
    SplitConfig, SplitStats,
};

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

    /// Maximum absolute magnitude for `min` / `max` gain values
    /// (log2 domain). Real Ultra HDR metadata never approaches ±30; the
    /// cap exists to keep `(value * ln2) as f32` finite and the resulting
    /// `exp()` inside `[2^-30, 2^30]` ≈ `[1e-9, 1e9]` — wider than any
    /// HDR display will ever support, narrower than `f32` overflow.
    pub const MAX_LOG_GAIN_MAGNITUDE: f64 = 30.0;

    /// Maximum absolute magnitude for `alternate_hdr_headroom` /
    /// `base_hdr_headroom` (log2 domain). Same rationale as
    /// [`MAX_LOG_GAIN_MAGNITUDE`].
    pub const MAX_HEADROOM_MAGNITUDE: f64 = 30.0;

    /// Maximum absolute magnitude for `base_offset` / `alternate_offset`
    /// (linear domain). Spec values are in `[0, 1]`; the cap is generous
    /// to allow for non-spec-compliant metadata while still rejecting
    /// values that would push `(sdr + offset) / gain` to `±inf`.
    pub const MAX_OFFSET_MAGNITUDE: f64 = 16.0;

    /// Maximum gamma value for gain map decoding. Spec uses values
    /// near 1.0; `gamma > 100` produces a near-step function and risks
    /// `powf` precision loss.
    pub const MAX_GAMMA: f64 = 100.0;

    /// Minimum gamma value (must be > 0; cap below for the same
    /// precision-loss reason).
    pub const MIN_GAMMA: f64 = 0.01;

    /// Maximum precomputed Shepard's IDW table entries
    /// (`scale_x * scale_y`). Practical Ultra HDR ratios never exceed
    /// 1:16 (so `scale ≤ 16`, `scale² ≤ 256`); the cap of 4096 is
    /// generous and keeps the four LUTs (`full`, `no_right`,
    /// `no_bottom`, `corner`) under 256 KB combined.
    pub const MAX_SHEPARDS_TABLE_ENTRIES: u32 = 4096;
}
