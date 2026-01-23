//! Metadata handling for Ultra HDR images.
//!
//! Supports both XMP (Adobe hdrgm namespace) and ISO 21496-1 binary format.

pub mod iso21496;
pub mod mpf;
pub mod xmp;

pub use iso21496::*;
pub use mpf::*;
pub use xmp::*;
