//! Color space handling: transfer functions, gamut matrices, conversions.

pub mod convert;
pub mod gamut;

/// Tonemapping modules require the `transfer` feature for EOTF/OETF functions.
#[cfg(feature = "transfer")]
pub mod tonemap;
#[cfg(feature = "transfer")]
pub mod transfer;

pub use convert::*;
pub use gamut::*;

#[cfg(feature = "transfer")]
pub use tonemap::*;
#[cfg(feature = "transfer")]
pub use transfer::*;

/// Streaming (row-based, bounded-memory) HDR→SDR tonemapper.
///
/// Re-exported from [`zentone::experimental::streaming`]. Pull-based API:
/// [`push_row`](StreamingTonemapper::push_row), [`finish`](StreamingTonemapper::finish),
/// [`pull_row`](StreamingTonemapper::pull_row). Channel count (3 or 4) is
/// passed to [`StreamingTonemapper::new`], not stored in the config.
#[cfg(feature = "transfer")]
pub mod streaming_tonemap {
    pub use zentone::experimental::{StreamingTonemapConfig, StreamingTonemapper};
}
#[cfg(feature = "transfer")]
pub use streaming_tonemap::*;
