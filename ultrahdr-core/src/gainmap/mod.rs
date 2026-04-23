//! Gain map computation and application.
//!
//! The gain map stores the ratio between HDR and SDR luminance values,
//! allowing reconstruction of HDR content from the SDR base image.

pub mod apply;
pub mod apply_simd;
pub mod compute;
/// **Deprecated** — slated for removal in 0.5.0. The splitter trait +
/// Hable curve are used only by [`compute::compute_gainmap_tonemap`]
/// (also deprecated). Hidden from docs; types are still accessible for
/// internal and any stray external use until 0.5.0.
#[doc(hidden)]
pub mod splitter;
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
pub use splitter::{
    HableFilmic, LumaFn, LumaGainMapSplitter, LumaToneMap, SplitConfig, SplitStats,
};
#[doc(hidden)]
pub use streaming::*;
