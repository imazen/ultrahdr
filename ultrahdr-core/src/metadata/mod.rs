//! Metadata handling for Ultra HDR and multi-image JPEG formats.
//!
//! - [`container`] — GContainer and MPF types shared across gain maps, depth maps, etc.
//! - [`xmp`] — XMP serialization (Adobe hdrgm namespace, GContainer directory)
//! - [`mpf`] — Multi-Picture Format (CIPA DC-007) parse/serialize
//! - [`iso21496`] — ISO 21496-1 binary gain map metadata

pub mod container;
pub mod iso21496;
pub mod mpf;
pub mod xmp;

pub use container::*;
pub use iso21496::*;
pub use mpf::*;
pub use xmp::*;
