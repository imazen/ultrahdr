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

/// ISO 21496-1 gain map metadata.
///
/// All gains and headroom values are stored in **log2 domain** to match the
/// ISO 21496-1 wire format and avoid lossy domain conversions. Gamma and
/// offsets are in **linear domain**.
///
/// # Domain conventions
///
/// | Field | Domain | Example |
/// |-------|--------|---------|
/// | `gain_map_min[i]` | log2 | −1.0 means ½× brightness |
/// | `gain_map_max[i]` | log2 | 2.0 means 4× brightness |
/// | `gamma[i]` | linear | 1.0 = linear gain map encoding |
/// | `base_offset[i]` | linear | 1/64 default |
/// | `alternate_offset[i]` | linear | 1/64 default |
/// | `base_hdr_headroom` | log2 | 0.0 = SDR (1:1) |
/// | `alternate_hdr_headroom` | log2 | 1.3 ≈ 2.46× peak brightness |
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct GainMapMetadata {
    /// Log2 of maximum gain per channel. 0.0 = no boost, 2.0 = 4× brightness.
    pub gain_map_max: [f64; 3],

    /// Log2 of minimum gain per channel. Can be negative (darkening).
    pub gain_map_min: [f64; 3],

    /// Gamma applied to the gain map encoding. Linear domain, must be > 0.
    pub gamma: [f64; 3],

    /// Offset added to base (SDR) values before gain application. Linear domain.
    pub base_offset: [f64; 3],

    /// Offset added to alternate (HDR) values before gain application. Linear domain.
    pub alternate_offset: [f64; 3],

    /// Log2 of base image HDR headroom. 0.0 = SDR (peak luminance ratio 1:1).
    pub base_hdr_headroom: f64,

    /// Log2 of alternate image HDR headroom.
    pub alternate_hdr_headroom: f64,

    /// Whether the gain map uses the base image color space.
    pub use_base_color_space: bool,
}

impl Default for GainMapMetadata {
    fn default() -> Self {
        Self {
            gain_map_max: [0.0; 3], // log2(1.0) = 0
            gain_map_min: [0.0; 3], // log2(1.0) = 0
            gamma: [1.0; 3],
            base_offset: [1.0 / 64.0; 3],
            alternate_offset: [1.0 / 64.0; 3],
            base_hdr_headroom: 0.0, // log2(1.0) = 0 = SDR
            alternate_hdr_headroom: 0.0,
            use_base_color_space: true,
        }
    }
}

impl GainMapMetadata {
    /// Create metadata with default values per ISO 21496-1.
    pub fn new() -> Self {
        Self::default()
    }

    /// Whether all three channels have identical parameters.
    pub fn is_single_channel(&self) -> bool {
        self.gain_map_max[0] == self.gain_map_max[1]
            && self.gain_map_max[1] == self.gain_map_max[2]
            && self.gain_map_min[0] == self.gain_map_min[1]
            && self.gain_map_min[1] == self.gain_map_min[2]
            && self.gamma[0] == self.gamma[1]
            && self.gamma[1] == self.gamma[2]
            && self.base_offset[0] == self.base_offset[1]
            && self.base_offset[1] == self.base_offset[2]
            && self.alternate_offset[0] == self.alternate_offset[1]
            && self.alternate_offset[1] == self.alternate_offset[2]
    }

    /// Validate metadata values are within reasonable bounds.
    pub fn validate(&self) -> Result<()> {
        for i in 0..3 {
            if !self.gain_map_max[i].is_finite() {
                return Err(Error::InvalidMetadata(format!(
                    "gain_map_max[{}] must be finite",
                    i
                )));
            }
            if !self.gain_map_min[i].is_finite() {
                return Err(Error::InvalidMetadata(format!(
                    "gain_map_min[{}] must be finite",
                    i
                )));
            }
            if !self.gamma[i].is_finite() || self.gamma[i] <= 0.0 {
                return Err(Error::InvalidMetadata(format!(
                    "gamma[{}] must be positive finite",
                    i
                )));
            }
            if !self.base_offset[i].is_finite() {
                return Err(Error::InvalidMetadata(format!(
                    "base_offset[{}] must be finite",
                    i
                )));
            }
            if !self.alternate_offset[i].is_finite() {
                return Err(Error::InvalidMetadata(format!(
                    "alternate_offset[{}] must be finite",
                    i
                )));
            }
            if self.gain_map_min[i] > self.gain_map_max[i] {
                return Err(Error::InvalidMetadata(format!(
                    "gain_map_min[{}] ({}) > gain_map_max[{}] ({})",
                    i, self.gain_map_min[i], i, self.gain_map_max[i]
                )));
            }
        }

        if !self.base_hdr_headroom.is_finite() {
            return Err(Error::InvalidMetadata(
                "base_hdr_headroom must be finite".into(),
            ));
        }
        if !self.alternate_hdr_headroom.is_finite() {
            return Err(Error::InvalidMetadata(
                "alternate_hdr_headroom must be finite".into(),
            ));
        }
        if self.alternate_hdr_headroom < 0.0 {
            return Err(Error::InvalidMetadata(
                "alternate_hdr_headroom must be >= 0.0 (log2 domain)".into(),
            ));
        }

        Ok(())
    }
}

// ============================================================================
// zencodec interop: From conversions for zenpixels / zencodec
// ============================================================================

#[cfg(feature = "zencodec")]
mod zencodec_interop {
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

    // --- GainMapMetadata ↔ zencodec::GainMapParams ---
    //
    // Both types now use log2 domain and f64 precision, so conversions are
    // trivial field copies with no domain transforms.

    impl From<&zencodec::GainMapParams> for GainMapMetadata {
        fn from(p: &zencodec::GainMapParams) -> Self {
            let mut meta = Self::new();
            for i in 0..3 {
                meta.gain_map_min[i] = p.channels[i].min;
                meta.gain_map_max[i] = p.channels[i].max;
                meta.gamma[i] = p.channels[i].gamma;
                meta.base_offset[i] = p.channels[i].base_offset;
                meta.alternate_offset[i] = p.channels[i].alternate_offset;
            }
            meta.base_hdr_headroom = p.base_hdr_headroom;
            meta.alternate_hdr_headroom = p.alternate_hdr_headroom;
            meta.use_base_color_space = p.use_base_color_space;
            meta
        }
    }

    impl From<&GainMapMetadata> for zencodec::GainMapParams {
        fn from(m: &GainMapMetadata) -> Self {
            let mut channels = [zencodec::GainMapChannel::default(); 3];
            for i in 0..3 {
                channels[i].min = m.gain_map_min[i];
                channels[i].max = m.gain_map_max[i];
                channels[i].gamma = m.gamma[i];
                channels[i].base_offset = m.base_offset[i];
                channels[i].alternate_offset = m.alternate_offset[i];
            }
            let mut params = Self::default();
            params.channels = channels;
            params.base_hdr_headroom = m.base_hdr_headroom;
            params.alternate_hdr_headroom = m.alternate_hdr_headroom;
            params.use_base_color_space = m.use_base_color_space;
            params
        }
    }
}

/// A fraction for ISO 21496-1 metadata encoding.
///
/// ISO 21496-1 uses fractional representation for gain map metadata
/// to preserve precision without floating-point ambiguity.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
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

/// An unsigned fraction for ISO 21496-1 metadata encoding.
///
/// Used for fields that are always non-negative (gamma, HDR headroom).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct UnsignedFraction {
    /// The numerator of the fraction.
    pub numerator: u32,
    /// The denominator of the fraction (must be non-zero for valid fractions).
    pub denominator: u32,
}

impl UnsignedFraction {
    /// Create a new unsigned fraction with the given numerator and denominator.
    pub fn new(numerator: u32, denominator: u32) -> Self {
        Self {
            numerator,
            denominator,
        }
    }

    /// Convert a non-negative floating-point value to an unsigned fraction.
    ///
    /// Uses a fixed denominator of 1,000,000 for reasonable precision.
    /// Negative values are clamped to zero.
    pub fn from_f32(value: f32) -> Self {
        let denominator = 1_000_000u32;
        let numerator = (value.max(0.0) * denominator as f32).round() as u32;
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

        metadata.gamma[0] = f64::NAN;
        assert!(metadata.validate().is_err());

        metadata.gamma[0] = 1.0;
        metadata.gain_map_max[1] = -1.0; // min(0.0) > max(-1.0)
        assert!(metadata.validate().is_err());
    }

    // ========================================================================
    // Metadata validation tests (C++ libultrahdr parity)
    // ========================================================================

    /// min_content_boost > max_content_boost should be rejected.
    #[test]
    fn test_validate_rejects_min_gt_max_boost() {
        let metadata = GainMapMetadata {
            gain_map_min: [5.0; 3],
            gain_map_max: [2.0; 3],
            gamma: [1.0; 3],
            base_offset: [1.0 / 64.0; 3],
            alternate_offset: [1.0 / 64.0; 3],
            base_hdr_headroom: 0.0,
            alternate_hdr_headroom: 5.0,
            use_base_color_space: true,
        };
        let err = metadata.validate().unwrap_err();
        let msg = err.to_string();
        assert!(
            msg.contains("gain_map_min"),
            "Error should mention gain_map_min: {}",
            msg
        );
    }

    /// gamma < 0 should be rejected (also covers gamma = 0).
    #[test]
    fn test_validate_rejects_negative_gamma() {
        let metadata = GainMapMetadata {
            gain_map_min: [0.0; 3],
            gain_map_max: [2.0; 3],
            gamma: [-1.0, 1.0, 1.0],
            base_offset: [1.0 / 64.0; 3],
            alternate_offset: [1.0 / 64.0; 3],
            base_hdr_headroom: 0.0,
            alternate_hdr_headroom: 2.0,
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

    /// Negative alternate_hdr_headroom (log2 domain) should be rejected.
    #[test]
    fn test_validate_rejects_negative_headroom() {
        let metadata = GainMapMetadata {
            gain_map_min: [0.0; 3],
            gain_map_max: [2.0; 3],
            gamma: [1.0; 3],
            base_offset: [1.0 / 64.0; 3],
            alternate_offset: [1.0 / 64.0; 3],
            base_hdr_headroom: 0.0,
            alternate_hdr_headroom: -0.5, // Invalid: negative in log2 domain
            use_base_color_space: true,
        };
        let err = metadata.validate().unwrap_err();
        let msg = err.to_string();
        assert!(
            msg.contains("alternate_hdr_headroom"),
            "Error should mention alternate_hdr_headroom: {}",
            msg
        );
    }

    /// Per-channel validation: only one channel invalid should still fail.
    #[test]
    fn test_validate_per_channel_independent() {
        // Channel 2 has min > max, others are fine
        let metadata = GainMapMetadata {
            gain_map_min: [1.0, 1.0, 5.0],
            gain_map_max: [4.0, 4.0, 2.0],
            gamma: [1.0; 3],
            base_offset: [1.0 / 64.0; 3],
            alternate_offset: [1.0 / 64.0; 3],
            base_hdr_headroom: 0.0,
            alternate_hdr_headroom: 2.0,
            use_base_color_space: true,
        };
        assert!(metadata.validate().is_err());
    }

    /// NaN and infinity should be rejected in all numeric fields.
    #[test]
    fn test_validate_rejects_nan_infinity() {
        let base = GainMapMetadata {
            gain_map_min: [0.0; 3],
            gain_map_max: [2.0; 3],
            gamma: [1.0; 3],
            base_offset: [1.0 / 64.0; 3],
            alternate_offset: [1.0 / 64.0; 3],
            base_hdr_headroom: 0.0,
            alternate_hdr_headroom: 2.0,
            use_base_color_space: true,
        };
        assert!(base.validate().is_ok());

        // NaN in each field
        let mut m = base.clone();
        m.gain_map_max[0] = f64::NAN;
        assert!(m.validate().is_err());

        let mut m = base.clone();
        m.gain_map_min[1] = f64::NAN;
        assert!(m.validate().is_err());

        let mut m = base.clone();
        m.base_offset[2] = f64::NAN;
        assert!(m.validate().is_err());

        let mut m = base.clone();
        m.alternate_offset[0] = f64::INFINITY;
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

#[cfg(all(test, feature = "zencodec"))]
mod zencodec_tests {
    use super::*;
    use zenpixels::{ColorPrimaries, TransferFunction};

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
        // Unknown falls back to Bt709
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
        // Unknown/Bt709 fall back to Srgb
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
    fn gainmap_params_to_metadata_roundtrip() {
        let mut params = zencodec::GainMapParams::default();
        params.channels = [zencodec::GainMapChannel {
            min: 0.0, // log2(1.0)
            max: 2.0, // log2(4.0)
            gamma: 1.0,
            base_offset: 1.0 / 64.0,
            alternate_offset: 1.0 / 64.0,
        }; 3];
        params.base_hdr_headroom = 0.0; // log2(1.0)
        params.alternate_hdr_headroom = 2.0; // log2(4.0)
        params.use_base_color_space = true;

        let meta = GainMapMetadata::from(&params);
        assert!((meta.gain_map_min[0] - 1.0).abs() < 1e-5); // 2^0 = 1
        assert!((meta.gain_map_max[0] - 4.0).abs() < 1e-5); // 2^2 = 4
        assert!((meta.gamma[0] - 1.0).abs() < 1e-5);
        assert!((meta.base_offset[0] - 1.0 / 64.0).abs() < 1e-5);
        assert!((meta.base_hdr_headroom - 1.0).abs() < 1e-5); // 2^0 = 1
        assert!((meta.alternate_hdr_headroom - 4.0).abs() < 1e-5); // 2^2 = 4
        assert!(meta.use_base_color_space);

        // Round-trip back to GainMapParams
        let back = zencodec::GainMapParams::from(&meta);
        assert!((back.channels[0].min - 0.0).abs() < 1e-4);
        assert!((back.channels[0].max - 2.0).abs() < 1e-4);
        assert!((back.base_hdr_headroom - 0.0).abs() < 1e-4);
        assert!((back.alternate_hdr_headroom - 2.0).abs() < 1e-4);
    }

    #[test]
    fn gainmap_metadata_to_params() {
        let meta = GainMapMetadata {
            gain_map_min: [0.0; 3],
            gain_map_max: [2.0; 3],
            gamma: [1.0; 3],
            base_offset: [1.0 / 64.0; 3],
            alternate_offset: [1.0 / 64.0; 3],
            base_hdr_headroom: 0.0,
            alternate_hdr_headroom: 2.0,
            use_base_color_space: true,
        };

        let params = zencodec::GainMapParams::from(&meta);
        assert!((params.channels[0].min - 0.0).abs() < 1e-5); // log2(1) = 0
        assert!((params.channels[0].max - 2.0).abs() < 1e-5); // log2(4) = 2
        assert!((params.base_hdr_headroom - 0.0).abs() < 1e-5); // log2(1) = 0
        assert!((params.alternate_hdr_headroom - 2.0).abs() < 1e-5); // log2(4) = 2
    }

    /// AVIF regression: headroom n=13,d=10 must produce linear 2^1.3 ≈ 2.46, NOT 1.3
    #[test]
    fn avif_headroom_regression() {
        // Simulate what the AVIF parser produces: log2 value 1.3 (from 13/10)
        let mut params = zencodec::GainMapParams::default();
        params.alternate_hdr_headroom = 1.3; // log2 domain

        let meta = GainMapMetadata::from(&params);
        // hdr_capacity_max should be 2^1.3 ≈ 2.462, NOT 1.3
        let expected = 2.0f32.powf(1.3);
        assert!(
            (meta.alternate_hdr_headroom - expected).abs() < 0.01,
            "hdr_capacity_max should be {expected}, got {}",
            meta.alternate_hdr_headroom,
        );
    }

    #[test]
    fn gainmap_params_multichannel() {
        let meta = GainMapMetadata {
            gain_map_min: [1.0, 0.5, 2.0],
            gain_map_max: [4.0, 8.0, 16.0],
            gamma: [1.0, 0.8, 1.2],
            base_offset: [0.01, 0.02, 0.03],
            alternate_offset: [0.04, 0.05, 0.06],
            base_hdr_headroom: 0.0,
            alternate_hdr_headroom: 10.0,
            use_base_color_space: false,
        };

        let params = zencodec::GainMapParams::from(&meta);
        assert!(!params.is_single_channel());
        assert!(!params.use_base_color_space);

        // Verify per-channel log2 conversion
        assert!((params.channels[0].max - 2.0).abs() < 1e-4); // log2(4) = 2
        assert!((params.channels[1].max - 3.0).abs() < 1e-4); // log2(8) = 3
        assert!((params.channels[2].max - 4.0).abs() < 1e-4); // log2(16) = 4
    }
}
