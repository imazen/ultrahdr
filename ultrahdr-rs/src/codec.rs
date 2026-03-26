//! zencodec trait implementation for Ultra HDR.
//!
//! Implements the zencodec decode traits so Ultra HDR images can be decoded
//! through the generic zencodec pipeline. The primary JPEG is decoded to
//! pixels via zenjpeg, and the gain map JPEG + metadata are provided as
//! type-erased extras on the [`DecodeOutput`].
//!
//! # Extras
//!
//! After decoding, call `output.extras::<UltraHdrExtras>()` to retrieve:
//! - `gainmap_jpeg`: raw gain map JPEG bytes
//! - `metadata`: gain map metadata (linear domain)
//!
//! # Example
//!
//! ```ignore
//! use ultrahdr::codec::UltraHdrDecoderConfig;
//! use zencodec::decode::DecoderConfig;
//!
//! let config = UltraHdrDecoderConfig;
//! let job = config.job();
//! let dec = job.decoder(data.into(), &[])?;
//! let output = dec.decode()?;
//!
//! // SDR primary image as pixels
//! let rgb8 = output.pixels();
//!
//! // Gain map data
//! if let Some(extras) = output.extras::<ultrahdr::codec::UltraHdrExtras>() {
//!     let gainmap_jpeg = &extras.gainmap_jpeg;
//!     let metadata = &extras.metadata;
//! }
//! ```

use alloc::borrow::Cow;

use zencodec::decode::{
    Decode, DecodeCapabilities, DecodeJob, DecodeOutput, DecoderConfig, OutputInfo,
};
use zencodec::{
    ImageFormat as ZenImageFormat, ImageInfo as ZenImageInfo, LimitExceeded, ResourceLimits,
    Unsupported, UnsupportedOperation,
};
use zenpixels::{PixelBuffer, PixelDescriptor};

use crate::Decoder;

extern crate alloc;

use enough::Stop as _;

/// Extra data from Ultra HDR decode.
///
/// Attached to [`DecodeOutput`] via `with_extras()`. Retrieve with
/// `output.extras::<UltraHdrExtras>()`.
pub struct UltraHdrExtras {
    /// Raw gain map JPEG bytes.
    pub gainmap_jpeg: Vec<u8>,
    /// Gain map metadata (linear domain, ultrahdr-core types).
    pub metadata: ultrahdr_core::GainMapMetadata,
}

/// Error type for zencodec Ultra HDR operations.
#[derive(Debug, thiserror::Error)]
pub enum ZenDecodeError {
    /// Error from ultrahdr-core.
    #[error("{0}")]
    Core(#[from] ultrahdr_core::Error),
    /// Unsupported operation.
    #[error("unsupported: {0}")]
    Unsupported(#[from] UnsupportedOperation),
    /// JPEG decode error.
    #[error("JPEG decode: {0}")]
    Jpeg(String),
    /// Decode row sink error.
    #[error("sink error: {0}")]
    Sink(zencodec::decode::SinkError),
    /// Resource limit exceeded.
    #[error("{0}")]
    LimitExceeded(#[from] LimitExceeded),
    /// Operation stopped by cooperative cancellation.
    #[error("stopped: {0:?}")]
    Stopped(enough::StopReason),
}

/// Reusable Ultra HDR decoder configuration.
///
/// Implements [`DecoderConfig`] for the zencodec trait system.
#[derive(Clone)]
pub struct UltraHdrDecoderConfig;

impl DecoderConfig for UltraHdrDecoderConfig {
    type Error = ZenDecodeError;
    type Job<'a> = UltraHdrDecodeJob;

    fn formats() -> &'static [ZenImageFormat] {
        &[ZenImageFormat::Jpeg] // Ultra HDR is JPEG-based
    }

    fn supported_descriptors() -> &'static [PixelDescriptor] {
        &[PixelDescriptor::RGB8_SRGB, PixelDescriptor::RGBA8_SRGB]
    }

    fn capabilities() -> &'static DecodeCapabilities {
        static CAPS: DecodeCapabilities = DecodeCapabilities::new()
            .with_stop(true)
            .with_enforces_max_pixels(true)
            .with_enforces_max_memory(true)
            .with_enforces_max_input_bytes(true)
            .with_cheap_probe(true)
            .with_hdr(true)
            .with_xmp(true);
        &CAPS
    }

    fn job<'a>(self) -> Self::Job<'a> {
        UltraHdrDecodeJob {
            _config: self,
            limits: None,
            stop: None,
        }
    }
}

/// Per-operation Ultra HDR decode job.
pub struct UltraHdrDecodeJob {
    _config: UltraHdrDecoderConfig,
    limits: Option<ResourceLimits>,
    stop: Option<zencodec::StopToken>,
}

impl<'a> DecodeJob<'a> for UltraHdrDecodeJob {
    type Error = ZenDecodeError;
    type Dec = UltraHdrDecoder<'a>;
    type StreamDec = Unsupported<ZenDecodeError>;
    type AnimationFrameDec = Unsupported<ZenDecodeError>;

    fn with_stop(mut self, stop: zencodec::StopToken) -> Self {
        self.stop = Some(stop);
        self
    }

    fn with_limits(mut self, limits: ResourceLimits) -> Self {
        self.limits = Some(limits);
        self
    }

    fn probe(&self, data: &[u8]) -> Result<ZenImageInfo, Self::Error> {
        if let Some(ref limits) = self.limits {
            limits.check_input_size(data.len() as u64)?;
        }

        let decoder = Decoder::new(data)?;
        let mut info = ZenImageInfo::new(0, 0, ZenImageFormat::Jpeg);
        if decoder.is_ultrahdr() {
            // Try to read dimensions from the primary JPEG header for limit checks
            if let Some(primary) = decoder.primary_jpeg()
                && let Ok(jpeg_info) = zenjpeg::decoder::Decoder::new().read_info(primary)
            {
                let w = jpeg_info.dimensions.width;
                let h = jpeg_info.dimensions.height;
                info = ZenImageInfo::new(w, h, ZenImageFormat::Jpeg);
                if let Some(ref limits) = self.limits {
                    limits.check_dimensions(w, h)?;
                }
            }

            // Ultra HDR always has a gain map (detected via is_ultrahdr above)
            // Gain map metadata is available via UltraHdrExtras after decode
        }
        Ok(info)
    }

    fn output_info(&self, data: &[u8]) -> Result<OutputInfo, Self::Error> {
        let decoder = Decoder::new(data)?;
        let (w, h) = if let Some(primary) = decoder.primary_jpeg()
            && let Ok(jpeg_info) = zenjpeg::decoder::Decoder::new().read_info(primary)
        {
            (jpeg_info.dimensions.width, jpeg_info.dimensions.height)
        } else {
            (0, 0)
        };
        Ok(OutputInfo::full_decode(w, h, PixelDescriptor::RGB8_SRGB))
    }

    fn decoder(
        self,
        data: Cow<'a, [u8]>,
        preferred: &[PixelDescriptor],
    ) -> Result<Self::Dec, Self::Error> {
        let limits = self.limits.unwrap_or(ResourceLimits::none());

        // Check input size before proceeding
        limits.check_input_size(data.len() as u64)?;

        let want_rgba = preferred.iter().any(|d| d.layout().has_alpha());

        Ok(UltraHdrDecoder {
            data,
            want_rgba,
            limits,
            stop: self.stop,
        })
    }

    fn push_decoder(
        self,
        data: Cow<'a, [u8]>,
        sink: &mut dyn zencodec::decode::DecodeRowSink,
        preferred: &[PixelDescriptor],
    ) -> Result<OutputInfo, Self::Error> {
        zencodec::helpers::copy_decode_to_sink(self, data, sink, preferred, ZenDecodeError::Sink)
    }

    fn streaming_decoder(
        self,
        _data: Cow<'a, [u8]>,
        _preferred: &[PixelDescriptor],
    ) -> Result<Self::StreamDec, Self::Error> {
        Err(UnsupportedOperation::RowLevelDecode.into())
    }

    fn animation_frame_decoder(
        self,
        _data: Cow<'a, [u8]>,
        _preferred: &[PixelDescriptor],
    ) -> Result<Self::AnimationFrameDec, Self::Error> {
        Err(UnsupportedOperation::AnimationDecode.into())
    }
}

/// One-shot Ultra HDR decoder.
pub struct UltraHdrDecoder<'a> {
    data: Cow<'a, [u8]>,
    want_rgba: bool,
    limits: ResourceLimits,
    stop: Option<zencodec::StopToken>,
}

impl<'a> Decode for UltraHdrDecoder<'a> {
    type Error = ZenDecodeError;

    fn decode(self) -> Result<DecodeOutput, Self::Error> {
        use zenjpeg::decoder::{Decoder as JpegDecoder, PixelFormat as JpegPixelFormat};

        // Input size was already checked in decoder(), but check again for
        // callers that construct UltraHdrDecoder directly.
        self.limits.check_input_size(self.data.len() as u64)?;

        let uhdr = Decoder::new(&self.data)?;
        if !uhdr.is_ultrahdr() {
            return Err(ultrahdr_core::Error::NotUltraHdr.into());
        }

        // Decode primary JPEG
        let primary_jpeg = uhdr
            .primary_jpeg()
            .ok_or_else(|| ultrahdr_core::Error::DecodeError("no primary image".into()))?;

        let output_fmt = if self.want_rgba {
            JpegPixelFormat::Rgba
        } else {
            JpegPixelFormat::Rgb
        };

        let stop = self.stop.unwrap_or_else(|| zencodec::StopToken::new(enough::Unstoppable));
        stop.check().map_err(ZenDecodeError::Stopped)?;

        let decoded = JpegDecoder::new()
            .output_format(output_fmt)
            .decode(primary_jpeg, &stop)
            .map_err(|e| ZenDecodeError::Jpeg(e.to_string()))?;

        let width = decoded.width();
        let height = decoded.height();

        // Check dimensions after parsing JPEG headers
        self.limits.check_dimensions(width, height)?;

        // Check memory before pixel buffer allocation
        let bpp = if self.want_rgba { 4u64 } else { 3u64 };
        self.limits
            .check_memory(width as u64 * height as u64 * bpp)?;

        let pixels_u8 = decoded
            .pixels_u8()
            .ok_or_else(|| ZenDecodeError::Jpeg("no pixel data".into()))?;

        let descriptor = if self.want_rgba {
            PixelDescriptor::RGBA8_SRGB
        } else {
            PixelDescriptor::RGB8_SRGB
        };

        let pixel_buf = PixelBuffer::from_vec(pixels_u8.to_vec(), width, height, descriptor)
            .map_err(|e| ZenDecodeError::Jpeg(format!("pixel buffer: {e}")))?;

        let zen_info = ZenImageInfo::new(width, height, ZenImageFormat::Jpeg)
            .with_alpha(self.want_rgba)
            .with_cicp(zencodec::Cicp::SRGB)
            .with_bit_depth(8);

        let mut output = DecodeOutput::new(pixel_buf, zen_info);

        // Attach gain map data as extras
        if let (Some(gm_jpeg), Some(metadata)) = (uhdr.gainmap_jpeg(), uhdr.metadata()) {
            output = output.with_extras(UltraHdrExtras {
                gainmap_jpeg: gm_jpeg.to_vec(),
                metadata: metadata.clone(),
            });
        }

        Ok(output)
    }
}
