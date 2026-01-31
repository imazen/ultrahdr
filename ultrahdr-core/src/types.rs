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
    /// BT.2100 / BT.2020 primaries (wide gamut for HDR)
    Bt2100,
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

/// Gain map metadata (linear scale values).
/// These values describe how to interpret the gain map.
#[derive(Debug, Clone, Default)]
pub struct GainMapMetadata {
    /// Maximum content boost per channel (HDR/SDR ratio).
    /// log2 of this value gives the maximum gain map value.
    pub max_content_boost: [f32; 3],

    /// Minimum content boost per channel.
    /// Allows for darkening as well as brightening.
    pub min_content_boost: [f32; 3],

    /// Gamma applied to the gain map encoding.
    pub gamma: [f32; 3],

    /// Offset added to SDR values before gain computation.
    pub offset_sdr: [f32; 3],

    /// Offset added to HDR values before gain computation.
    pub offset_hdr: [f32; 3],

    /// Minimum display boost for full gain map effect.
    pub hdr_capacity_min: f32,

    /// Maximum display boost for full gain map effect.
    pub hdr_capacity_max: f32,

    /// Whether the gain map uses the base image color space.
    pub use_base_color_space: bool,
}

impl GainMapMetadata {
    /// Create metadata with default values per Ultra HDR spec.
    pub fn new() -> Self {
        Self {
            max_content_boost: [1.0; 3],
            min_content_boost: [1.0; 3],
            gamma: [1.0; 3],
            offset_sdr: [1.0 / 64.0; 3], // 0.015625
            offset_hdr: [1.0 / 64.0; 3],
            hdr_capacity_min: 1.0,
            hdr_capacity_max: 1.0,
            use_base_color_space: true,
        }
    }

    /// Check if this is a single-channel (luminance-only) gain map.
    pub fn is_single_channel(&self) -> bool {
        self.max_content_boost[0] == self.max_content_boost[1]
            && self.max_content_boost[1] == self.max_content_boost[2]
            && self.min_content_boost[0] == self.min_content_boost[1]
            && self.min_content_boost[1] == self.min_content_boost[2]
            && self.gamma[0] == self.gamma[1]
            && self.gamma[1] == self.gamma[2]
    }

    /// Validate metadata values are within reasonable bounds.
    pub fn validate(&self) -> Result<()> {
        for i in 0..3 {
            if !self.max_content_boost[i].is_finite() || self.max_content_boost[i] <= 0.0 {
                return Err(Error::InvalidMetadata(format!(
                    "max_content_boost[{}] must be positive finite",
                    i
                )));
            }
            if !self.min_content_boost[i].is_finite() || self.min_content_boost[i] <= 0.0 {
                return Err(Error::InvalidMetadata(format!(
                    "min_content_boost[{}] must be positive finite",
                    i
                )));
            }
            if !self.gamma[i].is_finite() || self.gamma[i] <= 0.0 {
                return Err(Error::InvalidMetadata(format!(
                    "gamma[{}] must be positive finite",
                    i
                )));
            }
            if !self.offset_sdr[i].is_finite() {
                return Err(Error::InvalidMetadata(format!(
                    "offset_sdr[{}] must be finite",
                    i
                )));
            }
            if !self.offset_hdr[i].is_finite() {
                return Err(Error::InvalidMetadata(format!(
                    "offset_hdr[{}] must be finite",
                    i
                )));
            }
        }

        if !self.hdr_capacity_min.is_finite() || self.hdr_capacity_min < 0.0 {
            return Err(Error::InvalidMetadata(
                "hdr_capacity_min must be non-negative finite".into(),
            ));
        }
        if !self.hdr_capacity_max.is_finite() || self.hdr_capacity_max < 1.0 {
            return Err(Error::InvalidMetadata(
                "hdr_capacity_max must be >= 1.0".into(),
            ));
        }

        for i in 0..3 {
            if self.min_content_boost[i] > self.max_content_boost[i] {
                return Err(Error::InvalidMetadata(format!(
                    "min_content_boost[{}] ({}) > max_content_boost[{}] ({})",
                    i, self.min_content_boost[i], i, self.max_content_boost[i]
                )));
            }
        }

        Ok(())
    }
}

/// A fraction for ISO 21496-1 metadata encoding.
///
/// ISO 21496-1 uses fractional representation for gain map metadata
/// to preserve precision without floating-point ambiguity.
#[derive(Debug, Clone, Copy, Default)]
pub struct Fraction {
    /// The numerator of the fraction.
    pub numerator: i32,
    /// The denominator of the fraction (must be non-zero for valid fractions).
    pub denominator: u32,
}

impl Fraction {
    /// Create a new fraction with the given numerator and denominator.
    pub fn new(numerator: i32, denominator: u32) -> Self {
        Self {
            numerator,
            denominator,
        }
    }

    /// Convert a floating-point value to a fraction.
    ///
    /// Uses a fixed denominator of 1,000,000 for reasonable precision.
    pub fn from_f32(value: f32) -> Self {
        // Use a reasonable denominator for precision
        let denominator = 1_000_000u32;
        let numerator = (value * denominator as f32).round() as i32;
        Self {
            numerator,
            denominator,
        }
    }

    /// Convert the fraction to a floating-point value.
    ///
    /// Returns 0.0 if the denominator is zero.
    pub fn to_f32(self) -> f32 {
        if self.denominator == 0 {
            0.0
        } else {
            self.numerator as f32 / self.denominator as f32
        }
    }
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

    #[test]
    fn test_error_from_stop_reason() {
        let err: Error = StopReason::Cancelled.into();
        assert!(matches!(err, Error::Stopped(StopReason::Cancelled)));
    }

    #[test]
    fn test_raw_image_dimension_limits() {
        // Valid dimensions
        assert!(RawImage::new(1920, 1080, PixelFormat::Rgba8).is_ok());

        // Zero dimensions
        assert!(RawImage::new(0, 100, PixelFormat::Rgba8).is_err());
        assert!(RawImage::new(100, 0, PixelFormat::Rgba8).is_err());

        // Exceeds max dimension
        assert!(RawImage::new(100000, 100, PixelFormat::Rgba8).is_err());
    }

    #[test]
    fn test_gain_map_metadata_validation() {
        let mut metadata = GainMapMetadata::new();
        assert!(metadata.validate().is_ok());

        metadata.gamma[0] = f32::NAN;
        assert!(metadata.validate().is_err());

        metadata.gamma[0] = 1.0;
        metadata.max_content_boost[1] = -1.0;
        assert!(metadata.validate().is_err());
    }

    // ========================================================================
    // Metadata validation tests (C++ libultrahdr parity)
    // ========================================================================

    /// min_content_boost > max_content_boost should be rejected.
    #[test]
    fn test_validate_rejects_min_gt_max_boost() {
        let metadata = GainMapMetadata {
            min_content_boost: [5.0; 3],
            max_content_boost: [2.0; 3],
            gamma: [1.0; 3],
            offset_sdr: [1.0 / 64.0; 3],
            offset_hdr: [1.0 / 64.0; 3],
            hdr_capacity_min: 1.0,
            hdr_capacity_max: 5.0,
            use_base_color_space: true,
        };
        let err = metadata.validate().unwrap_err();
        let msg = err.to_string();
        assert!(
            msg.contains("min_content_boost"),
            "Error should mention min_content_boost: {}",
            msg
        );
    }

    /// gamma < 0 should be rejected (also covers gamma = 0).
    #[test]
    fn test_validate_rejects_negative_gamma() {
        let metadata = GainMapMetadata {
            min_content_boost: [1.0; 3],
            max_content_boost: [4.0; 3],
            gamma: [-1.0, 1.0, 1.0],
            offset_sdr: [1.0 / 64.0; 3],
            offset_hdr: [1.0 / 64.0; 3],
            hdr_capacity_min: 1.0,
            hdr_capacity_max: 4.0,
            use_base_color_space: true,
        };
        let err = metadata.validate().unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("gamma"), "Error should mention gamma: {}", msg);

        // Zero gamma
        let metadata_zero = GainMapMetadata {
            gamma: [0.0, 1.0, 1.0],
            ..metadata.clone()
        };
        assert!(metadata_zero.validate().is_err());
    }

    /// hdr_capacity_max < 1.0 should be rejected.
    #[test]
    fn test_validate_rejects_capacity_below_one() {
        let metadata = GainMapMetadata {
            min_content_boost: [1.0; 3],
            max_content_boost: [4.0; 3],
            gamma: [1.0; 3],
            offset_sdr: [1.0 / 64.0; 3],
            offset_hdr: [1.0 / 64.0; 3],
            hdr_capacity_min: 1.0,
            hdr_capacity_max: 0.5, // Invalid: < 1.0
            use_base_color_space: true,
        };
        let err = metadata.validate().unwrap_err();
        let msg = err.to_string();
        assert!(
            msg.contains("hdr_capacity_max"),
            "Error should mention hdr_capacity_max: {}",
            msg
        );
    }

    /// Per-channel validation: only one channel invalid should still fail.
    #[test]
    fn test_validate_per_channel_independent() {
        // Channel 2 has min > max, others are fine
        let metadata = GainMapMetadata {
            min_content_boost: [1.0, 1.0, 5.0],
            max_content_boost: [4.0, 4.0, 2.0],
            gamma: [1.0; 3],
            offset_sdr: [1.0 / 64.0; 3],
            offset_hdr: [1.0 / 64.0; 3],
            hdr_capacity_min: 1.0,
            hdr_capacity_max: 4.0,
            use_base_color_space: true,
        };
        assert!(metadata.validate().is_err());
    }

    /// NaN and infinity should be rejected in all numeric fields.
    #[test]
    fn test_validate_rejects_nan_infinity() {
        let base = GainMapMetadata {
            min_content_boost: [1.0; 3],
            max_content_boost: [4.0; 3],
            gamma: [1.0; 3],
            offset_sdr: [1.0 / 64.0; 3],
            offset_hdr: [1.0 / 64.0; 3],
            hdr_capacity_min: 1.0,
            hdr_capacity_max: 4.0,
            use_base_color_space: true,
        };
        assert!(base.validate().is_ok());

        // NaN in each field
        let mut m = base.clone();
        m.max_content_boost[0] = f32::NAN;
        assert!(m.validate().is_err());

        let mut m = base.clone();
        m.min_content_boost[1] = f32::NAN;
        assert!(m.validate().is_err());

        let mut m = base.clone();
        m.offset_sdr[2] = f32::NAN;
        assert!(m.validate().is_err());

        let mut m = base.clone();
        m.offset_hdr[0] = f32::INFINITY;
        assert!(m.validate().is_err());

        let mut m = base.clone();
        m.hdr_capacity_min = f32::NAN;
        assert!(m.validate().is_err());

        let mut m = base;
        m.hdr_capacity_max = f32::INFINITY;
        assert!(m.validate().is_err());
    }

    #[test]
    fn test_fraction_roundtrip() {
        let values = [0.0, 1.0, -1.0, 0.5, 3.5, -2.5];
        for &v in &values {
            let f = Fraction::from_f32(v);
            let roundtrip = f.to_f32();
            assert!(
                (roundtrip - v).abs() < 0.000001,
                "roundtrip failed for {}: got {}",
                v,
                roundtrip
            );
        }
    }
}
