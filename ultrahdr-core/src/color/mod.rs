//! Color space handling: transfer functions, gamut matrices, HDR→SDR tone mapping.
//!
//! YCbCr conversions live in `zenyuv` / `yuv` / `zenjpeg::color`; RGBA1010102
//! pack/unpack lives in `garb`; f16 pixel storage is handled by `zenpixels`.
//! This module owns only what's codec-agnostic and Ultra-HDR-specific.

pub mod gamut;
pub mod transfer;

/// The full `color::tonemap` surface (AdaptiveTonemapper, ProfileToneCurve,
/// tonemap_image_to_srgb8, and the zentone curve re-exports). Callers
/// computing the SDR base for a gain map can also drive the splitter
/// directly via [`zentone::LumaGainMapSplitter`] (re-exported at the
/// crate root) with [`zentone::HableFilmic`] or any other
/// [`zentone::LumaToneMap`] implementation.
///
/// Gated behind the `tonemap` feature (default-on); the entire module is
/// zentone-backed so decoder-only consumers (`--no-default-features
/// --features std`) opt out of it along with the transitive zentone dep.
#[cfg(feature = "tonemap")]
pub mod tonemap;

pub use gamut::*;
pub use transfer::*;

#[cfg(feature = "tonemap")]
pub use tonemap::*;

/// **Deprecated** — slated for removal in 0.5.0. Pass-through re-export
/// of `zentone::experimental::{StreamingTonemapConfig, StreamingTonemapper}`;
/// no ultrahdr-core-specific logic here. Import directly from `zentone`.
#[cfg(feature = "tonemap")]
#[doc(hidden)]
pub mod streaming_tonemap {
    pub use zentone::experimental::{StreamingTonemapConfig, StreamingTonemapper};
}
#[cfg(feature = "tonemap")]
#[doc(hidden)]
pub use streaming_tonemap::*;
