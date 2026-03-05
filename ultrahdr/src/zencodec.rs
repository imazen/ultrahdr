//! zencodec-types trait implementation for Ultra HDR.
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
//! - `metadata`: gain map metadata in zencodec-types log2 domain
//!
//! # Example
//!
//! ```ignore
//! use ultrahdr::zencodec::UltraHdrDecoderConfig;
//! use zencodec_types::DecoderConfig;
//!
//! let config = UltraHdrDecoderConfig;
//! let job = config.job();
//! let dec = job.decoder(ultrahdr_bytes, &[])?;
//! let output = dec.decode()?;
//!
//! // SDR primary image as pixels
//! let rgb8 = output.as_rgb8().unwrap();
//!
//! // Gain map data
//! if let Some(extras) = output.extras::<ultrahdr::zencodec::UltraHdrExtras>() {
//!     let gainmap_jpeg = &extras.gainmap_jpeg;
//!     let metadata = &extras.metadata;
//! }
//! ```

use zencodec_types::{
    CodecCapabilities, Decode, DecodeJob, DecodeOutput, DecoderConfig, FrameDecode,
    ImageFormat as ZenImageFormat, ImageInfo as ZenImageInfo, OutputInfo, ResourceLimits,
    StreamingDecode, UnsupportedOperation,
};
use zenpixels::{PixelBuffer, PixelDescriptor, PixelSlice};

use crate::Decoder;

/// Extra data from Ultra HDR decode.
///
/// Attached to [`DecodeOutput`] via `with_extras()`. Retrieve with
/// `output.extras::<UltraHdrExtras>()`.
pub struct UltraHdrExtras {
    /// Raw gain map JPEG bytes.
    pub gainmap_jpeg: Vec<u8>,
    /// Gain map metadata in zencodec-types log2 domain.
    pub metadata: zencodec_types::GainMapMetadata,
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
}

/// Reusable Ultra HDR decoder configuration.
///
/// Implements [`DecoderConfig`] for the zencodec trait system.
#[derive(Clone)]
pub struct UltraHdrDecoderConfig;

impl DecoderConfig for UltraHdrDecoderConfig {
    type Error = ZenDecodeError;
    type Job<'a> = UltraHdrDecodeJob<'a>;

    fn format() -> ZenImageFormat {
        ZenImageFormat::Jpeg // Ultra HDR is JPEG-based
    }

    fn supported_descriptors() -> &'static [PixelDescriptor] {
        &[PixelDescriptor::RGB8, PixelDescriptor::RGBA8]
    }

    fn capabilities() -> &'static CodecCapabilities {
        static CAPS: CodecCapabilities = CodecCapabilities::EMPTY;
        &CAPS
    }

    fn job(&self) -> Self::Job<'_> {
        UltraHdrDecodeJob {
            _config: self,
            limits: None,
        }
    }
}

/// Per-operation Ultra HDR decode job.
pub struct UltraHdrDecodeJob<'a> {
    _config: &'a UltraHdrDecoderConfig,
    limits: Option<ResourceLimits>,
}

impl<'a> DecodeJob<'a> for UltraHdrDecodeJob<'a> {
    type Error = ZenDecodeError;
    type Dec = UltraHdrDecoder<'a>;
    type StreamDec = NoStreamingDecode;
    type FrameDec = NoFrameDecode;

    fn with_stop(self, _stop: &'a dyn zencodec_types::Stop) -> Self {
        self // cancellation not yet wired
    }

    fn with_limits(mut self, limits: ResourceLimits) -> Self {
        self.limits = Some(limits);
        self
    }

    fn probe(&self, data: &[u8]) -> Result<ZenImageInfo, Self::Error> {
        let decoder = Decoder::new(data)?;
        let mut info = ZenImageInfo::new(0, 0, ZenImageFormat::Jpeg);
        if decoder.is_ultrahdr() {
            info = info.with_frame_count(1);
        }
        Ok(info)
    }

    fn output_info(&self, data: &[u8]) -> Result<OutputInfo, Self::Error> {
        let _decoder = Decoder::new(data)?;
        // We'd need to parse JPEG headers for dimensions without full decode.
        // For now return a placeholder — callers use probe() or just decode.
        Ok(OutputInfo::full_decode(0, 0, PixelDescriptor::RGB8))
    }

    fn decoder(
        self,
        data: &'a [u8],
        preferred: &[PixelDescriptor],
    ) -> Result<Self::Dec, Self::Error> {
        let want_rgba = preferred.iter().any(|d| d.layout().has_alpha());

        Ok(UltraHdrDecoder { data, want_rgba })
    }

    fn streaming_decoder(
        self,
        _data: &'a [u8],
        _preferred: &[PixelDescriptor],
    ) -> Result<Self::StreamDec, Self::Error> {
        Err(UnsupportedOperation::RowLevelDecode.into())
    }

    fn frame_decoder(
        self,
        _data: &'a [u8],
        _preferred: &[PixelDescriptor],
    ) -> Result<Self::FrameDec, Self::Error> {
        Err(UnsupportedOperation::AnimationDecode.into())
    }
}

/// One-shot Ultra HDR decoder.
pub struct UltraHdrDecoder<'a> {
    data: &'a [u8],
    want_rgba: bool,
}

impl<'a> Decode for UltraHdrDecoder<'a> {
    type Error = ZenDecodeError;

    fn decode(self) -> Result<DecodeOutput, Self::Error> {
        use zenjpeg::decoder::{Decoder as JpegDecoder, PixelFormat as JpegPixelFormat};

        let uhdr = Decoder::new(self.data)?;
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

        let decoded = JpegDecoder::new()
            .output_format(output_fmt)
            .decode(primary_jpeg, enough::Unstoppable)
            .map_err(|e| ZenDecodeError::Jpeg(e.to_string()))?;

        let width = decoded.width();
        let height = decoded.height();
        let pixels_u8 = decoded
            .pixels_u8()
            .ok_or_else(|| ZenDecodeError::Jpeg("no pixel data".into()))?;

        let descriptor = if self.want_rgba {
            PixelDescriptor::RGBA8
        } else {
            PixelDescriptor::RGB8
        };

        let pixel_buf = PixelBuffer::from_vec(pixels_u8.to_vec(), width, height, descriptor)
            .map_err(|e| ZenDecodeError::Jpeg(format!("pixel buffer: {e}")))?;

        let zen_info = ZenImageInfo::new(width, height, ZenImageFormat::Jpeg).with_frame_count(1);

        let mut output = DecodeOutput::new(pixel_buf, zen_info);

        // Attach gain map data as extras
        if let (Some(gm_jpeg), Some(metadata)) = (uhdr.gainmap_jpeg(), uhdr.metadata()) {
            let zen_metadata: zencodec_types::GainMapMetadata = metadata.clone().into();
            output = output.with_extras(UltraHdrExtras {
                gainmap_jpeg: gm_jpeg.to_vec(),
                metadata: zen_metadata,
            });
        }

        Ok(output)
    }
}

// --- Rejection types for unsupported decode modes ---

/// Rejection stub: Ultra HDR does not support streaming decode.
pub struct NoStreamingDecode;

impl StreamingDecode for NoStreamingDecode {
    type Error = ZenDecodeError;

    fn next_batch(&mut self) -> Result<Option<(u32, PixelSlice<'_>)>, Self::Error> {
        Err(UnsupportedOperation::RowLevelDecode.into())
    }

    fn info(&self) -> &ZenImageInfo {
        panic!("streaming decode not supported for Ultra HDR")
    }
}

/// Rejection stub: Ultra HDR does not support animation decode.
pub struct NoFrameDecode;

impl FrameDecode for NoFrameDecode {
    type Error = ZenDecodeError;

    fn next_frame(&mut self) -> Result<Option<zencodec_types::DecodeFrame>, Self::Error> {
        Err(UnsupportedOperation::AnimationDecode.into())
    }
}
