//! Gain map computation and application.
//!
//! The gain map stores the ratio between HDR and SDR luminance values,
//! allowing reconstruction of HDR content from the SDR base image.

pub mod apply;
pub mod compute;

pub use apply::*;
pub use compute::*;
