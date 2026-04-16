//! Metadata handling for Ultra HDR and multi-image JPEG formats.
//!
//! - [`container`] — GContainer and MPF types shared across gain maps, depth maps, etc.
//! - [`xmp`] — XMP serialization (Adobe hdrgm namespace, GContainer directory)
//! - [`mpf`] — Multi-Picture Format (CIPA DC-007) parse/serialize
//! - [`iso_jpeg`] — JPEG-specific ISO 21496-1 APP2 marker helpers
//!
//! ISO 21496-1 binary parse/serialize is provided by
//! [`zencodec::gainmap::parse_iso21496_fmt`] / [`zencodec::gainmap::serialize_iso21496_fmt`],
//! re-exported from the crate root.

pub mod container;
pub mod iso_jpeg;
pub mod mpf;
pub mod xmp;

pub use container::*;
pub use iso_jpeg::*;
pub use mpf::*;
pub use xmp::*;
