//! Color space handling: transfer functions, gamut matrices, conversions.

pub mod convert;
pub mod gamut;

/// Tonemapping modules require the `transfer` feature for EOTF/OETF functions.
#[cfg(feature = "transfer")]
pub mod streaming_tonemap;
#[cfg(feature = "transfer")]
pub mod tonemap;
#[cfg(feature = "transfer")]
pub mod transfer;

pub use convert::*;
pub use gamut::*;

#[cfg(feature = "transfer")]
pub use streaming_tonemap::*;
#[cfg(feature = "transfer")]
pub use tonemap::*;
#[cfg(feature = "transfer")]
pub use transfer::*;
