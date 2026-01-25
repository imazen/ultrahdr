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
//! - `ultrahdr` (this crate) - Full encoder/decoder (bring your own JPEG codec)
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
//! use ultrahdr::{encode_ultrahdr, Decoder, GainMapMetadata, ColorGamut};
//! use ultrahdr::gainmap::compute::{compute_gainmap, GainMapConfig};
//!
//! // 1. Prepare your images (using your own JPEG codec)
//! let sdr_jpeg = my_encoder.encode_rgb(&sdr_pixels)?;
//!
//! // 2. Compute gain map from HDR and SDR
//! let config = GainMapConfig::default();
//! let (gainmap, metadata) = compute_gainmap(&hdr, &sdr, &config, Unstoppable)?;
//! let gainmap_jpeg = my_encoder.encode_grayscale(&gainmap.data)?;
//!
//! // 3. Assemble Ultra HDR file
//! let ultrahdr = encode_ultrahdr(&sdr_jpeg, &gainmap_jpeg, &metadata, ColorGamut::Bt709)?;
//!
//! // 4. Decode: get raw JPEG bytes and decode with your codec
//! let decoder = Decoder::new(&ultrahdr)?;
//! let sdr_jpeg = decoder.primary_jpeg().unwrap();
//! let gainmap_jpeg = decoder.gainmap_jpeg().unwrap();
//! let metadata = decoder.metadata().unwrap();
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
    limits, luminance, ColorGamut, ColorTransfer, Error, Fraction, GainMap, GainMapConfig,
    GainMapMetadata, HdrOutputFormat, PixelFormat, RawImage, Result, Stop, StopReason, Unstoppable,
};

// This crate's additional modules
pub mod container;
pub mod jpeg;

mod decode;
mod encode;

// Re-export encoder/decoder
pub use decode::Decoder;
pub use encode::{encode_ultrahdr, Encoder};
