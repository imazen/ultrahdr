//! Gain map computation and application.
//!
//! The gain map stores the ratio between HDR and SDR luminance values,
//! allowing reconstruction of HDR content from the SDR base image.
//!
//! The luma gain map splitter (HDR ↔ (SDR, log2 gain) round-trip) lives
//! in the `zentone` crate; with the `tonemap` feature (default-on)
//! ultrahdr-core re-exports `LumaGainMapSplitter`, `SplitConfig`,
//! `SplitStats`, `LumaToneMap`, `LumaFn`, `HableFilmic`, `Bt2408Yrgb`,
//! and `ExtendedReinhardLuma` at the crate root for back-compat.

pub mod apply;
pub mod apply_simd;
pub mod compute;
/// **Deprecated** — slated for removal in 0.5.0. See the individual
/// struct docs for rationale. The per-row kernels in
/// [`apply`](self::apply) + [`apply_simd`](self::apply_simd) are the
/// reusable surface.
#[doc(hidden)]
pub mod streaming;

pub use apply::*;
pub use apply_simd::*;
pub use compute::*;
#[doc(hidden)]
pub use streaming::*;
