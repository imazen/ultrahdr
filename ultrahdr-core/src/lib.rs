//! Core gain map math for Ultra HDR.
//!
//! This crate provides the pure computational components for Ultra HDR:
//! - Pixel math for applying/computing gain maps
//! - Tone mapping (HDR → SDR)
//! - Color space conversions and transfer functions
//!
//! JPEG-container metadata (MPF, XMP, ISO 21496-1 APP2 envelope) previously
//! lived here in `ultrahdr_core::metadata`; it has moved to
//! [`zenjpeg::container`] (JPEG-specific parsing) and
//! [`zencodec::gainmap`] (codec-agnostic payload). See issue #8.
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
//!     ColorPrimaries, TransferFunction, PixelFormat, RawImage, Unstoppable,
//!     gainmap::{apply_gainmap, compute_gainmap, GainMapConfig, HdrOutputFormat},
//! };
//!
//! // Minimal 8x8 matching HDR + SDR surfaces. In practice these come from
//! // your image decoder — this example just shows the call shape.
//! let mut hdr = RawImage::new(8, 8, PixelFormat::Rgba8)?;
//! hdr.gamut = ColorPrimaries::Bt709;
//! hdr.transfer = TransferFunction::Srgb;
//! let mut sdr = RawImage::new(8, 8, PixelFormat::Rgba8)?;
//! sdr.gamut = ColorPrimaries::Bt709;
//! sdr.transfer = TransferFunction::Srgb;
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
mod types;

// Re-export core types (local)
pub use types::{
    ColorPrimaries, TransferFunction, Error, GainMap, GainMapEncodingFormat, PixelFormat, RawImage,
    RawImageRef, RawImageRefMut, Result, luminance, validate_gainmap_metadata,
};

// Re-export from zencodec (canonical gain map metadata types)
pub use types::{Fraction, GainMapChannel, GainMapMetadata, Iso21496Format, UnsignedFraction};
pub use zencodec::GainMapParams;
pub use zencodec::gainmap::{parse_iso21496_fmt, serialize_iso21496_fmt};

// Re-export enough for convenience
pub use enough::{Stop, StopReason, Unstoppable};

// Re-export gain map types
pub use gainmap::{
    apply::HdrOutputFormat,
    compute::{GainMapConfig, compute_gainmap_tonemap},
    splitter::{HableFilmic, LumaFn, LumaGainMapSplitter, LumaToneMap, SplitConfig, SplitStats},
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
}
