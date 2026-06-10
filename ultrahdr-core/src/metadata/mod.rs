//! Metadata container & sidecar parsing for HDR / multi-image captures.
//!
//! Houses parsers for the out-of-band metadata that carries HDR gain-map
//! information but lives outside the gain-map image itself:
//!
//! - [`apple`] — Apple iOS MakerNote HDR headroom (HEIC/JPEG).
//! - [`bplist`] — the minimal binary property-list reader [`apple`] depends on.
//!
//! (MPF / GContainer XMP container parsing is slated to consolidate here per
//! the crate roadmap; today those primitives still live with their callers.)

pub mod apple;
pub mod bplist;
