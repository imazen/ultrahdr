//! Ultra HDR - Pure Rust encoder/decoder for HDR images with gain maps.
//!
//! Ultra HDR is an image format that stores HDR (High Dynamic Range) content
//! in a backwards-compatible JPEG file. Legacy viewers see the SDR (Standard
//! Dynamic Range) base image, while HDR-capable displays can reconstruct the
//! full HDR content using an embedded gain map.
//!
//! # Crate Structure
//!
//! - [`ultrahdr_core`] - Core gain map math and metadata (no codec dependency)
//! - `ultrahdr` (this crate) - Full encoder/decoder with jpegli integration
//!
//! # Format Overview
//!
//! An Ultra HDR JPEG contains:
//! - Primary JPEG: SDR base image (8-bit, sRGB)
//! - Gain map JPEG: Compressed ratio of HDR/SDR luminance
//! - XMP metadata: Describes how to apply the gain map
//! - MPF header: Multi-Picture Format container
//!
//! # Example
//!
//! ```ignore
//! use ultrahdr::{Encoder, Decoder, RawImage, PixelFormat, ColorTransfer};
//!
//! // Encoding HDR to Ultra HDR JPEG
//! let hdr_image = RawImage::from_data(
//!     1920, 1080,
//!     PixelFormat::Rgba16F,
//!     ColorGamut::Bt2100,
//!     ColorTransfer::Pq,
//!     hdr_pixels,
//! )?;
//!
//! let ultrahdr_jpeg = Encoder::new()
//!     .set_hdr_image(hdr_image)
//!     .set_quality(90, 85)
//!     .encode()?;
//!
//! // Decoding Ultra HDR JPEG
//! let decoder = Decoder::new(&ultrahdr_jpeg)?;
//! let hdr_output = decoder.decode_hdr(4.0)?; // 4x SDR brightness
//! ```
//!
//! # Standards
//!
//! This implementation follows:
//! - [Ultra HDR Image Format v1.1](https://developer.android.com/media/platform/hdr-image-format)
//! - ISO 21496-1 (gain map metadata)
//! - Adobe XMP (hdrgm namespace)

#![warn(missing_docs)]
#![warn(clippy::all)]

// Re-export everything from ultrahdr-core
pub use ultrahdr_core::color;
pub use ultrahdr_core::gainmap;
pub use ultrahdr_core::metadata;

// Re-export core types at crate root
pub use ultrahdr_core::{
    limits, luminance, ColorGamut, ColorTransfer, Error, Fraction, GainMap, GainMapMetadata,
    GainMapConfig, HdrOutputFormat, PixelFormat, RawImage, Result, Stop, StopReason, Unstoppable,
};

// This crate's additional modules
pub mod container;
pub mod jpeg;

mod decode;
mod encode;

// Re-export encoder/decoder
pub use decode::Decoder;
pub use encode::Encoder;
