//! Ultra HDR - Pure Rust encoder/decoder for HDR images with gain maps.
//!
//! Ultra HDR is an image format that stores HDR (High Dynamic Range) content
//! in a backwards-compatible JPEG file. Legacy viewers see the SDR (Standard
//! Dynamic Range) base image, while HDR-capable displays can reconstruct the
//! full HDR content using an embedded gain map.
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

pub mod color;
pub mod gainmap;
pub mod jpeg;
pub mod metadata;
pub mod types;

mod decode;
mod encode;

// Re-export main types
pub use types::{
    ColorGamut, ColorTransfer, Error, Fraction, GainMap, GainMapMetadata, PixelFormat, RawImage,
    Result,
};

// Re-export encoder/decoder
pub use decode::Decoder;
pub use encode::Encoder;

// Re-export useful gain map types
pub use gainmap::{apply::HdrOutputFormat, compute::GainMapConfig};
