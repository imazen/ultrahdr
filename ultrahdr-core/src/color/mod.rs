//! Color space handling: transfer functions, gamut matrices, HDR→SDR tone mapping.
//!
//! YCbCr conversions live in `zenyuv` / `yuv` / `zenjpeg::color`; RGBA1010102
//! pack/unpack lives in `garb`; f16 pixel storage is handled by `zenpixels`.
//! This module owns only what's codec-agnostic and Ultra-HDR-specific.

pub mod gamut;
pub mod transfer;

/// The full `color::tonemap` surface (AdaptiveTonemapper, ProfileToneCurve,
/// tonemap_image_to_srgb8, and the zentone curve re-exports) is gated on
/// the `zentone` feature (default-on).
///
/// When `zentone` is off, callers produce SDR base images via
/// [`crate::gainmap::splitter::LumaGainMapSplitter`] (with the in-core
/// [`crate::gainmap::splitter::HableFilmic`] or a custom curve) instead.
#[cfg(feature = "zentone")]
pub mod tonemap;

pub use gamut::*;
pub use transfer::*;

#[cfg(feature = "zentone")]
pub use tonemap::*;

/// **Deprecated** — slated for removal in 0.5.0. Pass-through re-export
/// of `zentone::experimental::{StreamingTonemapConfig, StreamingTonemapper}`;
/// no ultrahdr-core-specific logic here. Import directly from `zentone`.
#[cfg(feature = "zentone")]
#[doc(hidden)]
pub mod streaming_tonemap {
    pub use zentone::experimental::{StreamingTonemapConfig, StreamingTonemapper};
}
#[cfg(feature = "zentone")]
#[doc(hidden)]
pub use streaming_tonemap::*;
