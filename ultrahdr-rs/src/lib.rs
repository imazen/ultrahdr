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
//! Decode with the bundled zenjpeg codec, with resource limits for
//! untrusted input:
//!
//! ```no_run
//! use ultrahdr_rs::{Decoder, ResourceLimits};
//!
//! # fn main() -> ultrahdr_rs::Result<()> {
//! let data = std::fs::read("photo_ultrahdr.jpg").expect("read");
//! let decoder = Decoder::new_with_limits(&data, ResourceLimits::default())?;
//! if decoder.is_ultrahdr() {
//!     let sdr = decoder.decode_sdr()?;    // Rgba8 SDR base
//!     let hdr = decoder.decode_hdr(4.0)?; // linear-float HDR at 4x boost
//!     let metadata = decoder.metadata();  // gain-map metadata (log2 domain)
//! }
//! # Ok(()) }
//! ```
//!
//! Bring-your-own-codec: [`Decoder::primary_jpeg`] and
//! [`Decoder::gainmap_jpeg`] return the raw JPEG codestreams for your own
//! decoder, and [`encode_ultrahdr`] assembles an Ultra HDR file from
//! pre-encoded JPEGs.
//!
//! # Resource limits & cancellation
//!
//! [`Decoder::new`] decodes uncapped — appropriate for trusted input. For
//! untrusted input, [`Decoder::new_with_limits`] validates JPEG header
//! dimensions against a pixel/memory budget ([`ResourceLimits`]) *before*
//! any pixel allocation; over-budget input yields
//! [`Error::LimitExceeded`]. Decode and
//! encode entry points also have `*_with_stop` variants taking a [`Stop`]
//! token for cooperative cancellation
//! ([`Error::Stopped`]).
//!
//! # Standards
//!
//! This implementation follows:
//! - [Ultra HDR Image Format v1.1](https://developer.android.com/media/platform/hdr-image-format)
//! - ISO 21496-1 (gain map metadata)
//! - Adobe XMP (hdrgm namespace)

#![forbid(unsafe_code)]
#![warn(missing_docs)]
#![warn(clippy::all)]

// Crate-info getter required by `whereat::at!()` for server-side error stack
// traces, so this crate's `Err(Error::…)` origins capture their precise call
// site (matching ultrahdr-core's instrumentation across the boundary).
whereat::define_at_crate_info!();

// Re-export everything from ultrahdr-core
pub use ultrahdr_core::color;
pub use ultrahdr_core::gainmap;
// NOTE: `ultrahdr_core::metadata` is being retired as part of issue #8
// — the canonical XMP / MPF / ISO-21496-1 APP2 container parsers live
// in `zenjpeg::container::*` and `zencodec::gainmap::*`. Consumers of
// `ultrahdr_rs::metadata::xmp::parse_xmp` should migrate to
// `zenjpeg::container::xmp::parse_xmp`; `ultrahdr_rs::metadata::iso_jpeg::*`
// helpers are superseded by `zencodec::gainmap::ISO_21496_1_PRIMARY_APP2_BODY`
// + a trivial `[FF E2, len_hi, len_lo]` wrap.

// Re-export core types at crate root
pub use ultrahdr_core::{
    ColorPrimaries, Error, Fraction, GainMap, GainMapConfig, GainMapEncodingFormat,
    GainMapMetadata, HdrOutputFormat, Iso21496Format, PixelBuffer, PixelFormat, PixelSlice,
    PixelSliceMut, Result, Stop, StopReason, TransferFunction, Unstoppable, clone_pixel_buffer,
    descriptor_for, limits, luminance, new_pixel_buffer, pixel_buffer_from_vec,
};

// This crate's additional modules
pub mod container;
pub mod jpeg;

/// zencodec trait integration (requires `zencodec` feature).
#[cfg(feature = "zencodec")]
pub mod codec;

mod decode;
mod encode;

// Re-export encoder/decoder
pub use decode::{Decoder, ResourceLimits};
pub use encode::{Encoder, encode_ultrahdr, encode_ultrahdr_with_format};
