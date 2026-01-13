//! Core types for Ultra HDR encoding/decoding.

use thiserror::Error;

/// Errors that can occur during Ultra HDR operations.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum Error {
    /// Image dimensions are invalid (zero or too large).
    #[error("Invalid image dimensions: {0}x{1}")]
    InvalidDimensions(u32, u32),

    /// HDR and SDR images have different dimensions.
    #[error("Dimension mismatch: HDR is {hdr_w}x{hdr_h}, SDR is {sdr_w}x{sdr_h}")]
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
    #[error("Unsupported pixel format: {0:?}")]
    UnsupportedFormat(PixelFormat),

    /// A required input (HDR image, SDR image, etc.) was not provided.
    #[error("Missing required input: {0}")]
    MissingInput(&'static str),

    /// JPEG encoding failed.
    #[error("JPEG encoding error: {0}")]
    JpegEncode(String),

    /// JPEG decoding failed.
    #[error("JPEG decoding error: {0}")]
    JpegDecode(String),

    /// Gain map metadata is invalid or malformed.
    #[error("Invalid metadata: {0}")]
    InvalidMetadata(String),

    /// The input is not an Ultra HDR image.
    #[error("Not an Ultra HDR image")]
    NotUltraHdr,

    /// Multi-Picture Format parsing failed.
    #[error("MPF parsing error: {0}")]
    MpfParse(String),

    /// XMP metadata parsing failed.
    #[error("XMP parsing error: {0}")]
    XmpParse(String),

    /// ICC profile error.
    #[error("ICC profile error: {0}")]
    IccError(String),

    /// General encoding error.
    #[error("Encoding error: {0}")]
    EncodeError(String),

    /// General decoding error.
    #[error("Decoding error: {0}")]
    DecodeError(String),
}

/// Result type for Ultra HDR operations.
pub type Result<T> = std::result::Result<T, Error>;

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
    pub fn new(width: u32, height: u32, format: PixelFormat) -> Self {
        let stride = match format.bytes_per_pixel() {
            Some(bpp) => width * bpp as u32,
            None => width, // For planar, stride is width
        };

        let data_size = match format {
            PixelFormat::Yuv420 => {
                // Y plane + U plane (1/4) + V plane (1/4)
                (height * stride) + 2 * ((height / 2) * (stride / 2))
            }
            PixelFormat::P010 => {
                // Y plane (16-bit) + UV interleaved plane (16-bit, half height)
                (height * stride * 2) + (height / 2) * stride * 2
            }
            _ => height * stride,
        };

        Self {
            width,
            height,
            format,
            gamut: ColorGamut::default(),
            transfer: ColorTransfer::default(),
            data: vec![0u8; data_size as usize],
            stride,
        }
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
        if width == 0 || height == 0 {
            return Err(Error::InvalidDimensions(width, height));
        }

        let stride = match format.bytes_per_pixel() {
            Some(bpp) => width * bpp as u32,
            None => width,
        };

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
    pub fn new(width: u32, height: u32) -> Self {
        Self {
            width,
            height,
            channels: 1,
            data: vec![0u8; (width * height) as usize],
        }
    }

    /// Create a new multi-channel (RGB) gain map.
    pub fn new_multichannel(width: u32, height: u32) -> Self {
        Self {
            width,
            height,
            channels: 3,
            data: vec![0u8; (width * height * 3) as usize],
        }
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
