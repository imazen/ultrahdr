//! Gain map computation and application.
//!
//! The gain map stores the ratio between HDR and SDR luminance values,
//! allowing reconstruction of HDR content from the SDR base image.

pub mod apply;
pub mod apply_simd;
pub mod compute;
pub mod splitter;
pub mod streaming;

pub use apply::*;
pub use apply_simd::*;
pub use compute::*;
pub use splitter::{
    HableFilmic, LumaFn, LumaGainMapSplitter, LumaToneMap, SplitConfig, SplitStats,
};
pub use streaming::*;
