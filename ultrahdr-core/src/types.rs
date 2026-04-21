//! Core types for Ultra HDR encoding/decoding.

use alloc::format;
use alloc::string::String;
use alloc::vec;
use alloc::vec::Vec;
use enough::StopReason;
use thiserror::Error;

use crate::limits;

/// Errors that can occur during Ultra HDR operations.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum Error {
    /// Operation was stopped via cooperative cancellation.
    #[error("operation stopped: {0}")]
    Stopped(StopReason),

    /// Image dimensions are invalid (zero or too large).
    #[error("invalid image dimensions: {0}x{1}")]
    InvalidDimensions(u32, u32),

    /// HDR and SDR images have different dimensions.
    #[error("dimension mismatch: HDR is {hdr_w}x{hdr_h}, SDR is {sdr_w}x{sdr_h}")]
    DimensionMismatch {
        /// HDR image width.
        hdr_w: u32,
        /// HDR image height.
        hdr_h: u32,
        /// SDR image width.
        sdr_w: u32,
        /// SDR image height.
        sdr_h: u32,
    },

    /// The pixel format is not supported for this operation.
    #[error("unsupported pixel format: {0:?}")]
    UnsupportedFormat(PixelFormat),

    /// A required input (HDR image, SDR image, etc.) was not provided.
    #[error("missing required input: {0}")]
    MissingInput(&'static str),

    /// Gain map metadata is invalid or malformed.
    #[error("invalid metadata: {0}")]
    InvalidMetadata(String),

    /// The input is not an Ultra HDR image.
    #[error("not an Ultra HDR image")]
    NotUltraHdr,

    /// XMP metadata parsing failed.
    #[error("XMP parsing error: {0}")]
    XmpParse(String),

    /// ISO 21496-1 metadata parsing failed.
    #[error("ISO 21496-1 parsing error: {0}")]
    IsoParse(String),

    /// Multi-Picture Format parsing failed.
    #[error("MPF parsing error: {0}")]
    MpfParse(String),

    /// Input exceeds safety limits.
    #[error("input exceeds safety limit: {0}")]
    LimitExceeded(String),

    /// Pixel data is invalid or corrupted.
    #[error("invalid pixel data: {0}")]
    InvalidPixelData(String),

    /// Allocation failed.
    #[error("allocation failed: requested {0} bytes")]
    AllocationFailed(usize),

    /// JPEG encoding failed.
    #[error("JPEG encoding error: {0}")]
    JpegEncode(String),

    /// JPEG decoding failed.
    #[error("JPEG decoding error: {0}")]
    JpegDecode(String),

    /// General encoding error.
    #[error("encoding error: {0}")]
    EncodeError(String),

    /// General decoding error.
    #[error("decoding error: {0}")]
    DecodeError(String),
}

/// Result type for Ultra HDR operations.
pub type Result<T> = core::result::Result<T, Error>;

impl From<StopReason> for Error {
    fn from(reason: StopReason) -> Self {
        Error::Stopped(reason)
    }
}

/// Color primaries. Re-exported from [`zenpixels::ColorPrimaries`].
///
/// ultrahdr-core used to define its own `ColorGamut` enum; #9 folded it
/// into zenpixels so every `zen*` crate speaks the same color-metadata
/// vocabulary.
pub use zenpixels::ColorPrimaries;

/// Electro-optical transfer function. Re-exported from
/// [`zenpixels::TransferFunction`].
pub use zenpixels::TransferFunction;

/// Pixel format for raw images. Re-exported from [`zenpixels::PixelFormat`].
///
/// ultrahdr-core uses a subset (`Rgba8`, `Rgb8`, `RgbaF32`, `Gray8`); other
/// zenpixels variants are accepted at the type level but will fail validation
/// in the gain map / tone mapping kernels that only accept SDR 8-bit and
/// linear float HDR.
pub use zenpixels::PixelFormat;

/// Controls which metadata format(s) to embed in Ultra HDR output.
///
/// For maximum cross-platform compatibility, use [`Both`](Self::Both) (the default).
/// XMP is universally supported; ISO 21496-1 binary is preferred by Android 15+
/// and iOS 18+ when present.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[non_exhaustive]
pub enum GainMapEncodingFormat {
    /// XMP only (`hdrgm:` namespace in APP1). Universally supported.
    Xmp,
    /// ISO 21496-1 binary only (APP2 with `urn:iso:std:iso:ts:21496:-1`).
    /// Newer apps prefer this when present. Some older tools may not read it.
    Iso21496,
    /// Both XMP and ISO 21496-1 binary. Maximum compatibility.
    /// XMP goes in the gain map JPEG's APP1; ISO binary goes in APP2.
    /// Cost: ~60 bytes extra for the ISO binary block.
    #[default]
    Both,
}

/// Wire format variant for ISO 21496-1 gain map metadata serialization.
///
/// Re-exported from [`zencodec::gainmap::Iso21496Format`].
pub use zencodec::Iso21496Format;

/// A raw (uncompressed) image — ultrahdr-core's owned pixel container.
///
/// This is a thin structure over a `Vec<u8>` plus zenpixels-compatible metadata
/// (`PixelFormat`, `ColorPrimaries`, `TransferFunction`). Use it when you need
/// public-field access to width/height/stride/data for hand-written pixel loops
/// (as the gain map and tone mapping kernels do).
///
/// For interop, use [`From<&zenpixels::PixelBuffer>`] to lift a zenpixels buffer
/// in, and [`to_pixel_buffer`](Self::to_pixel_buffer) to produce a zenpixels
/// buffer that carries the same pixel data and descriptor.
#[derive(Debug, Clone)]
pub struct RawImage {
    /// Image width in pixels.
    pub width: u32,
    /// Image height in pixels.
    pub height: u32,
    /// Pixel format.
    pub format: PixelFormat,
    /// Color gamut.
    pub gamut: ColorPrimaries,
    /// Transfer function.
    pub transfer: TransferFunction,
    /// Pixel data (layout depends on format).
    pub data: Vec<u8>,
    /// Row stride in bytes (for packed formats).
    /// For planar formats, this is the Y plane stride.
    pub stride: u32,
}

impl RawImage {
    /// Create a new raw image with the given dimensions and format.
    ///
    /// Returns an error if dimensions exceed safety limits.
    pub fn new(width: u32, height: u32, format: PixelFormat) -> Result<Self> {
        Self::validate_dimensions(width, height)?;

        let bpp = format.bytes_per_pixel();
        let stride = width
            .checked_mul(bpp as u32)
            .ok_or_else(|| Error::LimitExceeded(format!("stride overflow: {}x{}", width, bpp)))?;

        let data_size = Self::calculate_data_size(width, height, stride, format)?;

        Ok(Self {
            width,
            height,
            format,
            gamut: ColorPrimaries::default(),
            // zenpixels::TransferFunction has no Default impl.
            transfer: TransferFunction::Srgb,
            data: vec![0u8; data_size],
            stride,
        })
    }

    /// Create a raw image from existing data.
    pub fn from_data(
        width: u32,
        height: u32,
        format: PixelFormat,
        gamut: ColorPrimaries,
        transfer: TransferFunction,
        data: Vec<u8>,
    ) -> Result<Self> {
        Self::validate_dimensions(width, height)?;

        let bpp = format.bytes_per_pixel();
        let stride = width
            .checked_mul(bpp as u32)
            .ok_or_else(|| Error::LimitExceeded(format!("stride overflow: {}x{}", width, bpp)))?;

        let expected_size = Self::calculate_data_size(width, height, stride, format)?;
        if data.len() < expected_size {
            return Err(Error::InvalidPixelData(format!(
                "data too small: expected at least {} bytes, got {}",
                expected_size,
                data.len()
            )));
        }

        Ok(Self {
            width,
            height,
            format,
            gamut,
            transfer,
            data,
            stride,
        })
    }

    /// Validate dimensions against safety limits.
    fn validate_dimensions(width: u32, height: u32) -> Result<()> {
        if width == 0 || height == 0 {
            return Err(Error::InvalidDimensions(width, height));
        }

        if width > limits::MAX_IMAGE_DIMENSION || height > limits::MAX_IMAGE_DIMENSION {
            return Err(Error::LimitExceeded(format!(
                "dimension {} exceeds maximum {}",
                width.max(height),
                limits::MAX_IMAGE_DIMENSION
            )));
        }

        let total_pixels = width as u64 * height as u64;
        if total_pixels > limits::MAX_TOTAL_PIXELS {
            return Err(Error::LimitExceeded(format!(
                "total pixels {} exceeds maximum {}",
                total_pixels,
                limits::MAX_TOTAL_PIXELS
            )));
        }

        Ok(())
    }

    /// Borrow this image as a [`RawImageRef`].
    ///
    /// Cheap — no allocation, no validation repeated (assumes `self` was
    /// already validated at construction).
    pub fn as_ref(&self) -> RawImageRef<'_> {
        RawImageRef {
            width: self.width,
            height: self.height,
            stride: self.stride as usize,
            format: self.format,
            gamut: self.gamut,
            transfer: self.transfer,
            data: &self.data,
        }
    }

    /// Mutably borrow this image as a [`RawImageRefMut`].
    pub fn as_mut(&mut self) -> RawImageRefMut<'_> {
        RawImageRefMut {
            width: self.width,
            height: self.height,
            stride: self.stride as usize,
            format: self.format,
            gamut: self.gamut,
            transfer: self.transfer,
            data: &mut self.data,
        }
    }

    /// Validate that pixel data is large enough for declared dimensions.
    ///
    /// This should be called at entry points that iterate over pixel data
    /// (e.g., gain map apply/compute, tonemapping) to prevent OOB panics
    /// when stride or dimensions are inconsistent with data length.
    pub fn validate_data_bounds(&self) -> Result<()> {
        let required =
            Self::calculate_data_size(self.width, self.height, self.stride, self.format)?;
        if self.data.len() < required {
            return Err(Error::InvalidPixelData(format!(
                "data too small for {}x{} {:?}: need {} bytes, have {}",
                self.width,
                self.height,
                self.format,
                required,
                self.data.len()
            )));
        }
        Ok(())
    }

    /// Calculate required data size with overflow checking.
    fn calculate_data_size(
        width: u32,
        height: u32,
        stride: u32,
        format: PixelFormat,
    ) -> Result<usize> {
        calculate_data_size(width, height, stride as usize, format)
    }
}

/// Shared data-size calculator used by both owned and borrowed image types.
pub(crate) fn calculate_data_size(
    _width: u32,
    height: u32,
    stride: usize,
    format: PixelFormat,
) -> Result<usize> {
    let _ = format;
    let h = height as u64;
    let s = stride as u64;
    let size = h
        .checked_mul(s)
        .ok_or_else(|| Error::LimitExceeded("data size overflow".into()))?;

    if size > usize::MAX as u64 {
        return Err(Error::LimitExceeded(format!(
            "data size {} exceeds address space",
            size
        )));
    }

    Ok(size as usize)
}

/// Shared dimension validator used by both owned and borrowed image types.
pub(crate) fn validate_dimensions(width: u32, height: u32) -> Result<()> {
    if width == 0 || height == 0 {
        return Err(Error::InvalidDimensions(width, height));
    }

    if width > limits::MAX_IMAGE_DIMENSION || height > limits::MAX_IMAGE_DIMENSION {
        return Err(Error::LimitExceeded(format!(
            "dimension {} exceeds maximum {}",
            width.max(height),
            limits::MAX_IMAGE_DIMENSION
        )));
    }

    let total_pixels = width as u64 * height as u64;
    if total_pixels > limits::MAX_TOTAL_PIXELS {
        return Err(Error::LimitExceeded(format!(
            "total pixels {} exceeds maximum {}",
            total_pixels,
            limits::MAX_TOTAL_PIXELS
        )));
    }

    Ok(())
}

/// Minimum stride in bytes required for one row of the given format at `width`.
pub(crate) fn min_stride_bytes(width: u32, format: PixelFormat) -> Result<usize> {
    let bpp = format.bytes_per_pixel();
    (width as usize)
        .checked_mul(bpp)
        .ok_or_else(|| Error::LimitExceeded(format!("stride overflow: {}x{}", width, bpp)))
}

// ============================================================================
// RawImageRef / RawImageRefMut — borrow-only views over pixel data
// ============================================================================

/// Borrowed view over raw pixel data.
///
/// Cheap, `Copy`, carries no allocation. Use this as the primary input type for
/// gain map kernels and tone mapping — callers can construct one over any
/// `&[u8]`, including data owned by [`RawImage`], a [`zenpixels::PixelSlice`],
/// or any external buffer.
///
/// Fields are public for read access but construction goes through [`new`](Self::new)
/// so bounds are validated once. The `#[non_exhaustive]` attribute means new
/// metadata fields can be added without breaking callers.
#[derive(Copy, Clone, Debug)]
#[non_exhaustive]
pub struct RawImageRef<'a> {
    /// Image width in pixels.
    pub width: u32,
    /// Image height in pixels.
    pub height: u32,
    /// Row stride in bytes. For planar formats, Y-plane stride.
    pub stride: usize,
    /// Pixel format.
    pub format: PixelFormat,
    /// Color gamut.
    pub gamut: ColorPrimaries,
    /// Transfer function.
    pub transfer: TransferFunction,
    /// Borrowed pixel data (layout depends on format).
    pub data: &'a [u8],
}

impl<'a> RawImageRef<'a> {
    /// Construct a validated borrowed image view.
    ///
    /// Validates dimensions against [`limits`], checks stride is large enough
    /// for one row of `format` at `width`, and checks `data` is large enough
    /// for the declared image.
    pub fn new(
        data: &'a [u8],
        width: u32,
        height: u32,
        stride: usize,
        format: PixelFormat,
        gamut: ColorPrimaries,
        transfer: TransferFunction,
    ) -> Result<Self> {
        validate_dimensions(width, height)?;
        let min_row = min_stride_bytes(width, format)?;
        if stride < min_row {
            return Err(Error::InvalidPixelData(format!(
                "stride {} too small for {}x{:?} (need at least {})",
                stride, width, format, min_row
            )));
        }
        let required = calculate_data_size(width, height, stride, format)?;
        if data.len() < required {
            return Err(Error::InvalidPixelData(format!(
                "data too small: need {} bytes, got {}",
                required,
                data.len()
            )));
        }
        Ok(Self {
            width,
            height,
            stride,
            format,
            gamut,
            transfer,
            data,
        })
    }
}

/// Mutably borrowed view over raw pixel data.
///
/// Like [`RawImageRef`] but with `&mut [u8]` storage, for kernels that write
/// output in place. Not `Copy` — the mutable borrow is exclusive.
#[derive(Debug)]
#[non_exhaustive]
pub struct RawImageRefMut<'a> {
    /// Image width in pixels.
    pub width: u32,
    /// Image height in pixels.
    pub height: u32,
    /// Row stride in bytes. For planar formats, Y-plane stride.
    pub stride: usize,
    /// Pixel format.
    pub format: PixelFormat,
    /// Color gamut.
    pub gamut: ColorPrimaries,
    /// Transfer function.
    pub transfer: TransferFunction,
    /// Mutably borrowed pixel data (layout depends on format).
    pub data: &'a mut [u8],
}

impl<'a> RawImageRefMut<'a> {
    /// Construct a validated mutably borrowed image view.
    ///
    /// Same validation as [`RawImageRef::new`].
    pub fn new(
        data: &'a mut [u8],
        width: u32,
        height: u32,
        stride: usize,
        format: PixelFormat,
        gamut: ColorPrimaries,
        transfer: TransferFunction,
    ) -> Result<Self> {
        validate_dimensions(width, height)?;
        let min_row = min_stride_bytes(width, format)?;
        if stride < min_row {
            return Err(Error::InvalidPixelData(format!(
                "stride {} too small for {}x{:?} (need at least {})",
                stride, width, format, min_row
            )));
        }
        let required = calculate_data_size(width, height, stride, format)?;
        if data.len() < required {
            return Err(Error::InvalidPixelData(format!(
                "data too small: need {} bytes, got {}",
                required,
                data.len()
            )));
        }
        Ok(Self {
            width,
            height,
            stride,
            format,
            gamut,
            transfer,
            data,
        })
    }

    /// Downgrade to a shared borrow.
    pub fn as_ref(&self) -> RawImageRef<'_> {
        RawImageRef {
            width: self.width,
            height: self.height,
            stride: self.stride,
            format: self.format,
            gamut: self.gamut,
            transfer: self.transfer,
            data: self.data,
        }
    }
}

/// A gain map image (8-bit grayscale or per-channel).
#[derive(Debug, Clone)]
pub struct GainMap {
    /// Width of the gain map (may be smaller than base image).
    pub width: u32,
    /// Height of the gain map.
    pub height: u32,
    /// Number of channels (1 for luminance-only, 3 for per-channel RGB).
    pub channels: u8,
    /// Pixel data (u8 values 0-255).
    pub data: Vec<u8>,
}

impl GainMap {
    /// Create a new single-channel gain map.
    ///
    /// Returns an error if dimensions exceed safety limits.
    pub fn new(width: u32, height: u32) -> Result<Self> {
        RawImage::validate_dimensions(width, height)?;

        let size = (width as usize)
            .checked_mul(height as usize)
            .ok_or_else(|| Error::LimitExceeded("gain map size overflow".into()))?;

        Ok(Self {
            width,
            height,
            channels: 1,
            data: vec![0u8; size],
        })
    }

    /// Create a new multi-channel (RGB) gain map.
    ///
    /// Returns an error if dimensions exceed safety limits.
    pub fn new_multichannel(width: u32, height: u32) -> Result<Self> {
        RawImage::validate_dimensions(width, height)?;

        let size = (width as usize)
            .checked_mul(height as usize)
            .and_then(|s| s.checked_mul(3))
            .ok_or_else(|| Error::LimitExceeded("gain map size overflow".into()))?;

        Ok(Self {
            width,
            height,
            channels: 3,
            data: vec![0u8; size],
        })
    }
}

/// ISO 21496-1 gain map metadata.
///
/// This is the canonical [`zencodec::GainMapParams`] type. All gains and headroom
/// values are stored in **log2 domain** to match the ISO 21496-1 wire format.
///
/// # Field access (migrated from flat arrays to per-channel structs)
///
/// | Old API | New API |
/// |---------|---------|
/// | `metadata.gain_map_min[i]` | `metadata.channels[i].min` |
/// | `metadata.gain_map_max[i]` | `metadata.channels[i].max` |
/// | `metadata.gamma[i]` | `metadata.channels[i].gamma` |
/// | `metadata.base_offset[i]` | `metadata.channels[i].base_offset` |
/// | `metadata.alternate_offset[i]` | `metadata.channels[i].alternate_offset` |
/// | `metadata.base_hdr_headroom` | `metadata.base_hdr_headroom` |
/// | `metadata.alternate_hdr_headroom` | `metadata.alternate_hdr_headroom` |
/// | `metadata.use_base_color_space` | `metadata.use_base_color_space` |
/// | `metadata.backward_direction` | `metadata.backward_direction` |
pub type GainMapMetadata = zencodec::GainMapParams;

/// Per-channel gain map parameters.
///
/// Re-exported from [`zencodec::GainMapChannel`].
pub use zencodec::GainMapChannel;

/// Validate gain map metadata with ultrahdr-core's stricter checks.
///
/// Delegates to [`zencodec::GainMapParams::validate()`] and additionally rejects
/// negative `alternate_hdr_headroom` (log2 domain), which the base
/// zencodec validation does not enforce.
pub fn validate_gainmap_metadata(metadata: &GainMapMetadata) -> Result<()> {
    metadata
        .validate()
        .map_err(|e| Error::InvalidMetadata(alloc::string::ToString::to_string(&e)))?;
    if metadata.alternate_hdr_headroom < 0.0 {
        return Err(Error::InvalidMetadata(
            "alternate_hdr_headroom must be >= 0.0 (log2 domain)".into(),
        ));
    }
    Ok(())
}

// ============================================================================
// zenpixels interop: From conversions for ColorPrimaries/TransferFunction
// ============================================================================

mod zenpixels_interop {
    use super::*;
    use zenpixels::{PixelBuffer, PixelDescriptor, PixelSlice};

    fn accepts_format(format: PixelFormat) -> bool {
        matches!(
            format,
            PixelFormat::Rgba8 | PixelFormat::Rgb8 | PixelFormat::RgbaF32 | PixelFormat::Gray8
        )
    }

    /// Accept any [`zenpixels::PixelSlice`] that uses one of the pixel formats
    /// ultrahdr-core's gain map / tone mapping kernels understand (`Rgba8`,
    /// `Rgb8`, `RgbaF32`, `Gray8`). ICC color context on the slice is dropped —
    /// the caller is responsible for pre-converting to a primaries/transfer
    /// combination that the kernels accept.
    impl<'a> TryFrom<&PixelSlice<'a>> for RawImageRef<'a> {
        type Error = Error;

        fn try_from(ps: &PixelSlice<'a>) -> Result<Self> {
            let desc = ps.descriptor();
            let format = desc.pixel_format();
            if !accepts_format(format) {
                return Err(Error::InvalidPixelData(format!(
                    "zenpixels format {:?} not supported by ultrahdr-core kernels",
                    format
                )));
            }
            RawImageRef::new(
                ps.as_strided_bytes(),
                ps.width(),
                ps.rows(),
                ps.stride(),
                format,
                desc.primaries,
                desc.transfer(),
            )
        }
    }

    /// Lift a zenpixels [`PixelBuffer`] into a [`RawImage`] by cloning its
    /// pixel bytes. Returns `Err` if the buffer's pixel format isn't one of the
    /// four ultrahdr-core kernels operate on.
    impl TryFrom<&PixelBuffer> for RawImage {
        type Error = Error;

        fn try_from(pb: &PixelBuffer) -> Result<Self> {
            let desc = pb.descriptor();
            let format = desc.pixel_format();
            if !accepts_format(format) {
                return Err(Error::InvalidPixelData(format!(
                    "zenpixels format {:?} not supported by ultrahdr-core kernels",
                    format
                )));
            }
            let slice = pb.as_slice();
            Ok(RawImage {
                width: slice.width(),
                height: slice.rows(),
                format,
                gamut: desc.primaries,
                transfer: desc.transfer(),
                data: slice.as_strided_bytes().to_vec(),
                stride: slice.stride() as u32,
            })
        }
    }

    impl RawImage {
        fn descriptor(&self) -> PixelDescriptor {
            PixelDescriptor::from_pixel_format(self.format)
                .with_transfer(self.transfer)
                .with_primaries(self.gamut)
        }

        /// Borrow this image as a [`zenpixels::PixelSlice`] so it can flow into
        /// zenpixels-native pipelines (resize, color conversion) without
        /// copying pixel bytes.
        pub fn as_pixel_slice(&self) -> PixelSlice<'_> {
            PixelSlice::new(
                &self.data,
                self.width,
                self.height,
                self.stride as usize,
                self.descriptor(),
            )
            .expect("RawImage invariants imply a valid PixelSlice")
        }

        /// Produce an owned [`zenpixels::PixelBuffer`] carrying a copy of this
        /// image's pixels and descriptor.
        pub fn to_pixel_buffer(&self) -> PixelBuffer {
            PixelBuffer::from_vec(self.data.clone(), self.width, self.height, self.descriptor())
                .expect("RawImage invariants imply a valid PixelBuffer")
        }
    }
}

/// Signed fraction for ISO 21496-1 metadata encoding.
///
/// Re-exported from [`zencodec::gainmap::Fraction`]. Use `from_f64_cf()` for
/// continued-fraction encoding and `to_f64()` for conversion to float.
pub type Fraction = zencodec::gainmap::Fraction;

/// Unsigned fraction for ISO 21496-1 metadata encoding.
///
/// Re-exported from [`zencodec::gainmap::UFraction`]. Use `from_f64_cf()` for
/// continued-fraction encoding and `to_f64()` for conversion to float.
pub type UnsignedFraction = zencodec::gainmap::UFraction;

/// Construct a [`GainMapMetadata`] from per-channel flat arrays.
///
/// Convenience constructor that maps `[f64; 3]` fields into the per-channel
/// [`GainMapChannel`] records that [`GainMapMetadata`] uses internally.
#[allow(clippy::too_many_arguments)]
pub(crate) fn metadata_from_arrays(
    min: [f64; 3],
    max: [f64; 3],
    gamma: [f64; 3],
    base_offset: [f64; 3],
    alt_offset: [f64; 3],
    base_hdr_headroom: f64,
    alt_hdr_headroom: f64,
    use_base_cs: bool,
    backward: bool,
) -> GainMapMetadata {
    let mut m = GainMapMetadata::default();
    for i in 0..3 {
        m.channels[i].min = min[i];
        m.channels[i].max = max[i];
        m.channels[i].gamma = gamma[i];
        m.channels[i].base_offset = base_offset[i];
        m.channels[i].alternate_offset = alt_offset[i];
    }
    m.base_hdr_headroom = base_hdr_headroom;
    m.alternate_hdr_headroom = alt_hdr_headroom;
    m.use_base_color_space = use_base_cs;
    m.backward_direction = backward;
    m
}

/// Reference display luminance values (in nits).
pub mod luminance {
    /// SDR reference white (diffuse white)
    pub const SDR_WHITE_NITS: f32 = 203.0;

    /// HLG reference white (75% signal level)
    pub const HLG_WHITE_NITS: f32 = 1000.0;

    /// PQ peak luminance
    pub const PQ_PEAK_NITS: f32 = 10000.0;

    /// PQ reference white (58% signal level, ~203 nits)
    pub const PQ_WHITE_NITS: f32 = 203.0;
}

#[cfg(test)]
mod tests {
    use super::*;
    use zenpixels::{ColorPrimaries, TransferFunction};

    /// Helper to construct GainMapMetadata (GainMapParams) from per-channel arrays.
    #[allow(clippy::too_many_arguments)]
    fn make_metadata(
        min: [f64; 3],
        max: [f64; 3],
        gamma: [f64; 3],
        base_offset: [f64; 3],
        alt_offset: [f64; 3],
        base_hdr_headroom: f64,
        alt_hdr_headroom: f64,
        use_base_cs: bool,
        backward: bool,
    ) -> GainMapMetadata {
        let mut m = GainMapMetadata::default();
        for i in 0..3 {
            m.channels[i].min = min[i];
            m.channels[i].max = max[i];
            m.channels[i].gamma = gamma[i];
            m.channels[i].base_offset = base_offset[i];
            m.channels[i].alternate_offset = alt_offset[i];
        }
        m.base_hdr_headroom = base_hdr_headroom;
        m.alternate_hdr_headroom = alt_hdr_headroom;
        m.use_base_color_space = use_base_cs;
        m.backward_direction = backward;
        m
    }

    #[test]
    fn test_error_from_stop_reason() {
        let err: Error = StopReason::Cancelled.into();
        assert!(matches!(err, Error::Stopped(StopReason::Cancelled)));
    }

    #[test]
    fn test_raw_image_dimension_limits() {
        assert!(RawImage::new(1920, 1080, PixelFormat::Rgba8).is_ok());
        assert!(RawImage::new(0, 100, PixelFormat::Rgba8).is_err());
        assert!(RawImage::new(100, 0, PixelFormat::Rgba8).is_err());
        assert!(RawImage::new(100000, 100, PixelFormat::Rgba8).is_err());
    }

    #[test]
    fn test_gain_map_metadata_validation() {
        let mut metadata = GainMapMetadata::default();
        assert!(metadata.validate().is_ok());

        metadata.channels[0].gamma = f64::NAN;
        assert!(metadata.validate().is_err());

        metadata.channels[0].gamma = 1.0;
        metadata.channels[1].max = -1.0; // min(0.0) > max(-1.0)
        assert!(metadata.validate().is_err());
    }

    // ========================================================================
    // Metadata validation tests (C++ libultrahdr parity)
    // ========================================================================

    #[test]
    fn test_validate_rejects_min_gt_max_boost() {
        let metadata = make_metadata(
            [5.0; 3],
            [2.0; 3],
            [1.0; 3],
            [1.0 / 64.0; 3],
            [1.0 / 64.0; 3],
            0.0,
            5.0,
            true,
            false,
        );
        let err = metadata.validate().unwrap_err();
        let msg = err.to_string();
        assert!(
            msg.contains("min") || msg.contains("max"),
            "Error should mention min/max: {msg}"
        );
    }

    #[test]
    fn test_validate_rejects_negative_gamma() {
        let metadata = make_metadata(
            [0.0; 3],
            [2.0; 3],
            [-1.0, 1.0, 1.0],
            [1.0 / 64.0; 3],
            [1.0 / 64.0; 3],
            0.0,
            2.0,
            true,
            false,
        );
        let err = metadata.validate().unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("gamma"), "Error should mention gamma: {msg}");

        // Zero gamma
        let mut metadata_zero = metadata.clone();
        metadata_zero.channels[0].gamma = 0.0;
        assert!(metadata_zero.validate().is_err());
    }

    /// Negative alternate_hdr_headroom (log2 domain) should be rejected by
    /// our stricter validate_gainmap_metadata.
    #[test]
    fn test_validate_rejects_negative_headroom() {
        let metadata = make_metadata(
            [0.0; 3],
            [2.0; 3],
            [1.0; 3],
            [1.0 / 64.0; 3],
            [1.0 / 64.0; 3],
            0.0,
            -0.5,
            true,
            false,
        );
        let err = validate_gainmap_metadata(&metadata).unwrap_err();
        let msg = err.to_string();
        assert!(
            msg.contains("alternate_hdr_headroom"),
            "Error should mention alternate_hdr_headroom: {msg}",
        );
    }

    #[test]
    fn test_validate_per_channel_independent() {
        let metadata = make_metadata(
            [1.0, 1.0, 5.0],
            [4.0, 4.0, 2.0],
            [1.0; 3],
            [1.0 / 64.0; 3],
            [1.0 / 64.0; 3],
            0.0,
            2.0,
            true,
            false,
        );
        assert!(metadata.validate().is_err());
    }

    #[test]
    fn test_validate_rejects_nan_infinity() {
        let base = make_metadata(
            [0.0; 3],
            [2.0; 3],
            [1.0; 3],
            [1.0 / 64.0; 3],
            [1.0 / 64.0; 3],
            0.0,
            2.0,
            true,
            false,
        );
        assert!(base.validate().is_ok());

        let mut m = base.clone();
        m.channels[0].max = f64::NAN;
        assert!(m.validate().is_err());

        let mut m = base.clone();
        m.channels[1].min = f64::NAN;
        assert!(m.validate().is_err());

        let mut m = base.clone();
        m.channels[2].base_offset = f64::NAN;
        assert!(m.validate().is_err());

        let mut m = base.clone();
        m.channels[0].alternate_offset = f64::INFINITY;
        assert!(m.validate().is_err());

        let mut m = base.clone();
        m.base_hdr_headroom = f64::NAN;
        assert!(m.validate().is_err());

        let mut m = base;
        m.alternate_hdr_headroom = f64::INFINITY;
        assert!(m.validate().is_err());
    }

    #[test]
    fn test_fraction_roundtrip() {
        let values = [0.0, 1.0, -1.0, 0.5, 3.5, -2.5];
        for &v in &values {
            let f = Fraction::from_f64_cf(v);
            let roundtrip = f.to_f64();
            assert!(
                (roundtrip - v).abs() < 0.000001,
                "roundtrip failed for {}: got {}",
                v,
                roundtrip
            );
        }
    }

    // ========================================================================
    // zenpixels interop tests (always available, no feature gate)
    // ========================================================================

    #[test]
    fn test_iso21496_format_identity() {
        // Iso21496Format is now a re-export — no conversion needed
        assert_eq!(Iso21496Format::AvifTmap, zencodec::Iso21496Format::AvifTmap);
        assert_eq!(Iso21496Format::JxlJhgm, zencodec::Iso21496Format::JxlJhgm);
        assert_eq!(
            Iso21496Format::JpegApp2BodyWithUrn,
            zencodec::Iso21496Format::JpegApp2BodyWithUrn
        );
    }

    #[test]
    fn raw_image_roundtrips_through_pixel_buffer() {
        let mut pixels = Vec::with_capacity(4 * 4 * 4);
        for i in 0..(4 * 4) {
            pixels.extend_from_slice(&[(i * 4) as u8, (i * 4 + 1) as u8, (i * 4 + 2) as u8, 255]);
        }
        let img = RawImage::from_data(
            4,
            4,
            PixelFormat::Rgba8,
            ColorPrimaries::DisplayP3,
            TransferFunction::Srgb,
            pixels,
        )
        .expect("raw image");

        let pb = img.to_pixel_buffer();
        assert_eq!(pb.width(), 4);
        assert_eq!(pb.height(), 4);
        assert_eq!(pb.descriptor().pixel_format(), PixelFormat::Rgba8);
        assert_eq!(pb.descriptor().primaries, ColorPrimaries::DisplayP3);

        let back = RawImage::try_from(&pb).expect("convert back");
        assert_eq!(back.width, 4);
        assert_eq!(back.height, 4);
        assert_eq!(back.format, PixelFormat::Rgba8);
        assert_eq!(back.gamut, ColorPrimaries::DisplayP3);
        assert_eq!(back.transfer, TransferFunction::Srgb);
        // Pixel bytes preserved on the addressable region of each row.
        for y in 0..4 {
            let orig_row = &img.data[(y * img.stride as usize)..(y * img.stride as usize + 16)];
            let back_row = &back.data[(y * back.stride as usize)..(y * back.stride as usize + 16)];
            assert_eq!(orig_row, back_row, "row {y} pixels differ");
        }
    }

    #[test]
    fn raw_image_as_pixel_slice_exposes_descriptor() {
        let img = RawImage::new(8, 8, PixelFormat::RgbaF32).expect("raw image");
        let slice = img.as_pixel_slice();
        assert_eq!(slice.width(), 8);
        assert_eq!(slice.rows(), 8);
        assert_eq!(slice.descriptor().pixel_format(), PixelFormat::RgbaF32);
    }

    // XMP roundtrip parity now lives alongside the builders in
    // crate::metadata::xmp::tests.

    #[test]
    fn raw_image_ref_new_validates_stride_and_data_size() {
        // 4x4 Rgba8 = 16 pixels * 4 bytes = 64 bytes, stride 16
        let data = vec![0u8; 64];
        let r = RawImageRef::new(
            &data,
            4,
            4,
            16,
            PixelFormat::Rgba8,
            ColorPrimaries::Bt709,
            TransferFunction::Srgb,
        )
        .expect("valid image should construct");
        assert_eq!(r.width, 4);
        assert_eq!(r.stride, 16);
        assert_eq!(r.data.len(), 64);

        // Stride too small — should fail.
        assert!(
            RawImageRef::new(
                &data,
                4,
                4,
                8,
                PixelFormat::Rgba8,
                ColorPrimaries::Bt709,
                TransferFunction::Srgb,
            )
            .is_err()
        );

        // Data too small — should fail.
        let short = vec![0u8; 32];
        assert!(
            RawImageRef::new(
                &short,
                4,
                4,
                16,
                PixelFormat::Rgba8,
                ColorPrimaries::Bt709,
                TransferFunction::Srgb,
            )
            .is_err()
        );
    }

    #[test]
    fn raw_image_as_ref_round_trips() {
        let img = RawImage::new(8, 4, PixelFormat::Rgba8).unwrap();
        let r = img.as_ref();
        assert_eq!(r.width, img.width);
        assert_eq!(r.height, img.height);
        assert_eq!(r.stride, img.stride as usize);
        assert_eq!(r.format, img.format);
        assert_eq!(r.data.len(), img.data.len());
    }

    #[test]
    fn raw_image_ref_mut_downgrades_to_ref() {
        let mut img = RawImage::new(8, 4, PixelFormat::Rgba8).unwrap();
        let m = img.as_mut();
        m.data[0] = 42;
        let shared = m.as_ref();
        assert_eq!(shared.data[0], 42);
    }

    #[test]
    fn zenpixels_pixel_slice_adapts_to_raw_image_ref() {
        use zenpixels::{PixelDescriptor, PixelSlice};

        let data = vec![0u8; 4 * 4 * 4]; // 4x4 Rgba8
        let desc = PixelDescriptor::from_pixel_format(zenpixels::PixelFormat::Rgba8);
        let ps = PixelSlice::new(&data, 4, 4, 16, desc).expect("valid PixelSlice");
        let r: RawImageRef<'_> = (&ps).try_into().expect("adapter should succeed");
        assert_eq!(r.width, 4);
        assert_eq!(r.height, 4);
        assert_eq!(r.stride, 16);
        assert_eq!(r.format, PixelFormat::Rgba8);
        assert_eq!(r.data.len(), ps.as_strided_bytes().len());
    }

    #[test]
    fn iso21496_roundtrip_preserves_metadata() {
        let original = make_metadata(
            [0.5, 0.25, 1.0],
            [4.0, 8.0, 2.0],
            [1.0, 0.75, 1.5],
            [1.0 / 64.0, 1.0 / 32.0, 1.0 / 128.0],
            [1.0 / 64.0; 3],
            1.0,
            8.0,
            true,
            false,
        );

        let bytes = zencodec::gainmap::serialize_iso21496_fmt(&original, Iso21496Format::AvifTmap);
        let parsed = zencodec::gainmap::parse_iso21496_fmt(&bytes, Iso21496Format::AvifTmap)
            .expect("ISO parse failed");

        for ch in 0..3 {
            assert!((original.channels[ch].max - parsed.channels[ch].max).abs() < 1e-3);
            assert!((original.channels[ch].gamma - parsed.channels[ch].gamma).abs() < 1e-4);
        }
        assert!((original.alternate_hdr_headroom - parsed.alternate_hdr_headroom).abs() < 1e-3);
    }
}
