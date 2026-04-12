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

/// Color gamut / color space primaries.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ColorGamut {
    /// BT.709 / sRGB primaries
    #[default]
    Bt709,
    /// Display P3 primaries
    DisplayP3,
    /// BT.2020 primaries (also used by BT.2100 for HDR)
    Bt2020,
}

/// Electro-optical transfer function (EOTF/OETF).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ColorTransfer {
    /// sRGB transfer function (gamma ~2.2)
    #[default]
    Srgb,
    /// Linear (gamma 1.0)
    Linear,
    /// Perceptual Quantizer (SMPTE ST 2084) - HDR
    Pq,
    /// Hybrid Log-Gamma (ITU-R BT.2100) - HDR
    Hlg,
}

/// Pixel format for raw images.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PixelFormat {
    /// 8-bit RGBA (SDR)
    Rgba8,
    /// 8-bit RGB (SDR)
    Rgb8,
    /// 16-bit float RGBA (HDR linear)
    Rgba16F,
    /// 32-bit float RGBA (HDR linear)
    Rgba32F,
    /// 10-bit YCbCr 4:2:0 P010 format (HDR)
    P010,
    /// 8-bit YCbCr 4:2:0 (SDR)
    Yuv420,
    /// 10-bit packed RGBA (1010102) with PQ transfer
    Rgba1010102Pq,
    /// 10-bit packed RGBA (1010102) with HLG transfer
    Rgba1010102Hlg,
    /// 8-bit grayscale (for gain maps)
    Gray8,
}

impl PixelFormat {
    /// Returns the number of bytes per pixel for packed formats.
    /// Returns None for planar formats like P010 and Yuv420.
    pub fn bytes_per_pixel(&self) -> Option<usize> {
        match self {
            Self::Rgba8 => Some(4),
            Self::Rgb8 => Some(3),
            Self::Rgba16F => Some(8),
            Self::Rgba32F => Some(16),
            Self::Rgba1010102Pq | Self::Rgba1010102Hlg => Some(4),
            Self::Gray8 => Some(1),
            Self::P010 | Self::Yuv420 => None, // Planar
        }
    }

    /// Returns true if this is an HDR format.
    pub fn is_hdr(&self) -> bool {
        matches!(
            self,
            Self::Rgba16F | Self::Rgba32F | Self::P010 | Self::Rgba1010102Pq | Self::Rgba1010102Hlg
        )
    }
}

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

/// A raw (uncompressed) image.
#[derive(Debug, Clone)]
pub struct RawImage {
    /// Image width in pixels.
    pub width: u32,
    /// Image height in pixels.
    pub height: u32,
    /// Pixel format.
    pub format: PixelFormat,
    /// Color gamut.
    pub gamut: ColorGamut,
    /// Transfer function.
    pub transfer: ColorTransfer,
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

        let stride = match format.bytes_per_pixel() {
            Some(bpp) => width.checked_mul(bpp as u32).ok_or_else(|| {
                Error::LimitExceeded(format!("stride overflow: {}x{}", width, bpp))
            })?,
            None => width, // For planar, stride is width
        };

        let data_size = Self::calculate_data_size(width, height, stride, format)?;

        Ok(Self {
            width,
            height,
            format,
            gamut: ColorGamut::default(),
            transfer: ColorTransfer::default(),
            data: vec![0u8; data_size],
            stride,
        })
    }

    /// Create a raw image from existing data.
    pub fn from_data(
        width: u32,
        height: u32,
        format: PixelFormat,
        gamut: ColorGamut,
        transfer: ColorTransfer,
        data: Vec<u8>,
    ) -> Result<Self> {
        Self::validate_dimensions(width, height)?;

        let stride = match format.bytes_per_pixel() {
            Some(bpp) => width.checked_mul(bpp as u32).ok_or_else(|| {
                Error::LimitExceeded(format!("stride overflow: {}x{}", width, bpp))
            })?,
            None => width,
        };

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
        _width: u32,
        height: u32,
        stride: u32,
        format: PixelFormat,
    ) -> Result<usize> {
        let size = match format {
            PixelFormat::Yuv420 => {
                // Y plane + U plane (1/4) + V plane (1/4)
                let y_size = (height as u64) * (stride as u64);
                let uv_size = 2 * ((height as u64 / 2) * (stride as u64 / 2));
                y_size.checked_add(uv_size)
            }
            PixelFormat::P010 => {
                // Y plane (16-bit) + UV interleaved plane (16-bit, half height)
                let y_size = (height as u64) * (stride as u64) * 2;
                let uv_size = (height as u64 / 2) * (stride as u64) * 2;
                y_size.checked_add(uv_size)
            }
            _ => Some((height as u64) * (stride as u64)),
        };

        let size = size.ok_or_else(|| Error::LimitExceeded("data size overflow".into()))?;

        if size > usize::MAX as u64 {
            return Err(Error::LimitExceeded(format!(
                "data size {} exceeds address space",
                size
            )));
        }

        Ok(size as usize)
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
/// Delegates to [`GainMapParams::validate()`] and additionally rejects
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
// zenpixels interop: From conversions for ColorGamut/ColorTransfer
// ============================================================================

mod zenpixels_interop {
    use super::*;
    use zenpixels::{ColorPrimaries, TransferFunction};

    // --- ColorGamut ↔ ColorPrimaries ---

    impl From<ColorGamut> for ColorPrimaries {
        fn from(gamut: ColorGamut) -> Self {
            match gamut {
                ColorGamut::Bt709 => ColorPrimaries::Bt709,
                ColorGamut::DisplayP3 => ColorPrimaries::DisplayP3,
                ColorGamut::Bt2020 => ColorPrimaries::Bt2020,
            }
        }
    }

    impl From<ColorPrimaries> for ColorGamut {
        fn from(primaries: ColorPrimaries) -> Self {
            match primaries {
                ColorPrimaries::Bt709 => ColorGamut::Bt709,
                ColorPrimaries::DisplayP3 => ColorGamut::DisplayP3,
                ColorPrimaries::Bt2020 => ColorGamut::Bt2020,
                _ => ColorGamut::Bt709, // fallback
            }
        }
    }

    // --- ColorTransfer ↔ TransferFunction ---

    impl From<ColorTransfer> for TransferFunction {
        fn from(transfer: ColorTransfer) -> Self {
            match transfer {
                ColorTransfer::Srgb => TransferFunction::Srgb,
                ColorTransfer::Linear => TransferFunction::Linear,
                ColorTransfer::Pq => TransferFunction::Pq,
                ColorTransfer::Hlg => TransferFunction::Hlg,
            }
        }
    }

    impl From<TransferFunction> for ColorTransfer {
        fn from(tf: TransferFunction) -> Self {
            match tf {
                TransferFunction::Srgb => ColorTransfer::Srgb,
                TransferFunction::Linear => ColorTransfer::Linear,
                TransferFunction::Pq => ColorTransfer::Pq,
                TransferFunction::Hlg => ColorTransfer::Hlg,
                _ => ColorTransfer::Srgb, // fallback
            }
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

/// Construct a [`GainMapMetadata`] from per-channel arrays.
///
/// This is the migration bridge for code that previously constructed `GainMapMetadata`
/// with flat `[f64; 3]` array fields. Now that `GainMapMetadata` is `GainMapParams`
/// (which uses `channels: [GainMapChannel; 3]`), this helper does the mapping.
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
    fn test_color_gamut_to_primaries() {
        assert_eq!(
            ColorPrimaries::from(ColorGamut::Bt709),
            ColorPrimaries::Bt709
        );
        assert_eq!(
            ColorPrimaries::from(ColorGamut::DisplayP3),
            ColorPrimaries::DisplayP3
        );
        assert_eq!(
            ColorPrimaries::from(ColorGamut::Bt2020),
            ColorPrimaries::Bt2020
        );
    }

    #[test]
    fn test_primaries_to_color_gamut() {
        assert_eq!(ColorGamut::from(ColorPrimaries::Bt709), ColorGamut::Bt709);
        assert_eq!(
            ColorGamut::from(ColorPrimaries::DisplayP3),
            ColorGamut::DisplayP3
        );
        assert_eq!(ColorGamut::from(ColorPrimaries::Bt2020), ColorGamut::Bt2020);
        assert_eq!(ColorGamut::from(ColorPrimaries::Unknown), ColorGamut::Bt709);
    }

    #[test]
    fn test_color_transfer_to_transfer_function() {
        assert_eq!(
            TransferFunction::from(ColorTransfer::Srgb),
            TransferFunction::Srgb
        );
        assert_eq!(
            TransferFunction::from(ColorTransfer::Linear),
            TransferFunction::Linear
        );
        assert_eq!(
            TransferFunction::from(ColorTransfer::Pq),
            TransferFunction::Pq
        );
        assert_eq!(
            TransferFunction::from(ColorTransfer::Hlg),
            TransferFunction::Hlg
        );
    }

    #[test]
    fn test_transfer_function_to_color_transfer() {
        assert_eq!(
            ColorTransfer::from(TransferFunction::Srgb),
            ColorTransfer::Srgb
        );
        assert_eq!(
            ColorTransfer::from(TransferFunction::Linear),
            ColorTransfer::Linear
        );
        assert_eq!(ColorTransfer::from(TransferFunction::Pq), ColorTransfer::Pq);
        assert_eq!(
            ColorTransfer::from(TransferFunction::Hlg),
            ColorTransfer::Hlg
        );
        assert_eq!(
            ColorTransfer::from(TransferFunction::Unknown),
            ColorTransfer::Srgb
        );
        assert_eq!(
            ColorTransfer::from(TransferFunction::Bt709),
            ColorTransfer::Srgb
        );
    }

    #[test]
    fn test_iso21496_format_identity() {
        // Iso21496Format is now a re-export — no conversion needed
        assert_eq!(Iso21496Format::AvifTmap, zencodec::Iso21496Format::AvifTmap);
        assert_eq!(Iso21496Format::JpegApp2, zencodec::Iso21496Format::JpegApp2);
    }

    // =========================================================================
    // XMP roundtrip parity test
    // =========================================================================

    fn reference_metadata() -> GainMapMetadata {
        make_metadata(
            [0.5, 0.25, 1.0],
            [4.0, 8.0, 2.0],
            [1.0, 0.75, 1.5],
            [1.0 / 64.0, 1.0 / 32.0, 1.0 / 128.0],
            [1.0 / 64.0; 3],
            1.0,
            8.0,
            true,
            false,
        )
    }

    #[test]
    fn xmp_roundtrip_preserves_metadata() {
        use crate::metadata::xmp::{generate_gainmap_xmp, parse_xmp};

        let original = reference_metadata();
        let xmp_str = generate_gainmap_xmp(&original);
        let (parsed, _) = parse_xmp(&xmp_str).expect("XMP parse failed");

        for ch in 0..3 {
            assert!((original.channels[ch].min - parsed.channels[ch].min).abs() < 1e-4);
            assert!((original.channels[ch].max - parsed.channels[ch].max).abs() < 1e-4);
            assert!((original.channels[ch].gamma - parsed.channels[ch].gamma).abs() < 1e-4);
            assert!(
                (original.channels[ch].base_offset - parsed.channels[ch].base_offset).abs() < 1e-6
            );
            assert!(
                (original.channels[ch].alternate_offset - parsed.channels[ch].alternate_offset)
                    .abs()
                    < 1e-6
            );
        }
        assert!((original.base_hdr_headroom - parsed.base_hdr_headroom).abs() < 1e-4);
        assert!((original.alternate_hdr_headroom - parsed.alternate_hdr_headroom).abs() < 1e-4);
        assert_eq!(original.use_base_color_space, parsed.use_base_color_space);
    }

    #[test]
    fn iso21496_roundtrip_preserves_metadata() {
        let original = reference_metadata();

        // AVIF tmap variant
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
