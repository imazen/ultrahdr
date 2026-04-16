//! Color space handling: transfer functions, gamut matrices, conversions.

pub mod convert;
pub mod gamut;

/// Tonemapping modules require the `transfer` feature for EOTF/OETF
/// functions. The full `color::tonemap` surface (AdaptiveTonemapper,
/// ProfileToneCurve, tonemap_image_to_srgb8, and the zentone curve
/// re-exports) additionally requires the `zentone` feature.
///
/// When `zentone` is off, callers produce SDR base images via
/// [`crate::gainmap::splitter::LumaGainMapSplitter`] (with the in-core
/// [`crate::gainmap::splitter::HableFilmic`] or a custom curve) instead.
#[cfg(all(feature = "transfer", feature = "zentone"))]
pub mod tonemap;
#[cfg(feature = "transfer")]
pub mod transfer;

pub use convert::*;
pub use gamut::*;

#[cfg(all(feature = "transfer", feature = "zentone"))]
pub use tonemap::*;
#[cfg(feature = "transfer")]
pub use transfer::*;

/// Streaming (row-based, bounded-memory) HDR→SDR tonemapper.
///
/// Re-exported from [`zentone::experimental::streaming`] when the `zentone`
/// feature is enabled (default). Pull-based API:
/// [`push_row`](StreamingTonemapper::push_row), [`finish`](StreamingTonemapper::finish),
/// [`pull_row`](StreamingTonemapper::pull_row). Channel count (3 or 4) is
/// passed to [`StreamingTonemapper::new`], not stored in the config.
#[cfg(all(feature = "transfer", feature = "zentone"))]
pub mod streaming_tonemap {
    pub use zentone::experimental::{StreamingTonemapConfig, StreamingTonemapper};
}
#[cfg(all(feature = "transfer", feature = "zentone"))]
pub use streaming_tonemap::*;
