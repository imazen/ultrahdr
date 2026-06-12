//! Core types for Ultra HDR encoding/decoding.
//!
//! The pixel container is [`zenpixels::PixelBuffer`] (owning) /
//! [`zenpixels::PixelSlice`] (borrowed). ultrahdr-core used to ship its own
//! `RawImage` / `RawImageRef` / `RawImageRefMut` triplet; those were
//! eliminated so every `zen*` crate speaks the same vocabulary.
//!
//! The gain map byte buffer is [`GainMap`], kept bespoke because its bytes
//! are log2-quantized gain, not color samples with a transfer function.

use alloc::format;
use alloc::string::String;
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

impl<E: core::fmt::Display> From<zenpixels::At<E>> for Error {
    fn from(err: zenpixels::At<E>) -> Self {
        Error::InvalidPixelData(alloc::string::ToString::to_string(&err))
    }
}

/// Color primaries. Re-exported from [`zenpixels::ColorPrimaries`].
pub use zenpixels::ColorPrimaries;

/// Electro-optical transfer function. Re-exported from
/// [`zenpixels::TransferFunction`].
pub use zenpixels::TransferFunction;

/// Pixel format for raw images. Re-exported from [`zenpixels::PixelFormat`].
///
/// ultrahdr-core's kernels accept a subset (`Rgba8`, `Rgb8`, `RgbaF32`,
/// `RgbaF16`, `RgbF16`, `Gray8`). Other formats are rejected by
/// [`require_supported_format`].
pub use zenpixels::PixelFormat;

/// Owning pixel container. Re-exported from [`zenpixels::PixelBuffer`].
///
/// Replaces the former `ultrahdr_core::RawImage` type. Construct with
/// `PixelBuffer::try_new` / `PixelBuffer::from_vec`; attach color metadata
/// via the `with_transfer`/`with_primaries` builders.
pub use zenpixels::PixelBuffer;

/// Borrowed pixel view. Re-exported from [`zenpixels::PixelSlice`].
pub use zenpixels::PixelSlice;

/// Mutably borrowed pixel view. Re-exported from [`zenpixels::PixelSliceMut`].
pub use zenpixels::PixelSliceMut;

/// Build a [`zenpixels::PixelDescriptor`] with the given format, primaries,
/// and transfer function. Convenience for the common pattern used by
/// ultrahdr-core's tests and examples.
pub fn descriptor_for(
    format: PixelFormat,
    primaries: ColorPrimaries,
    transfer: TransferFunction,
) -> zenpixels::PixelDescriptor {
    zenpixels::PixelDescriptor::from_pixel_format(format)
        .with_primaries(primaries)
        .with_transfer(transfer)
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

// ============================================================================
// ultrahdr-core-specific validators
// ============================================================================

/// Validate image dimensions against ultrahdr-core's stricter caps.
///
/// `PixelBuffer::try_new` only checks for arithmetic overflow. ultrahdr-core
/// additionally rejects dimensions above [`limits::MAX_IMAGE_DIMENSION`] and
/// total pixel counts above [`limits::MAX_TOTAL_PIXELS`] to bound the work
/// the gain map / tone mapping kernels are willing to do.
pub fn validate_ultrahdr_dimensions(width: u32, height: u32) -> Result<()> {
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

/// Reject pixel formats the kernels don't understand.
///
/// The gain map / tone mapping kernels accept `Rgba8`, `Rgb8`, `RgbaF32`,
/// and `Gray8`. Everything else is rejected with [`Error::UnsupportedFormat`].
pub fn require_supported_format(format: PixelFormat) -> Result<()> {
    match format {
        PixelFormat::Rgba8
        | PixelFormat::Rgb8
        | PixelFormat::RgbaF32
        | PixelFormat::RgbaF16
        | PixelFormat::RgbF16
        | PixelFormat::Gray8 => Ok(()),
        _ => Err(Error::UnsupportedFormat(format)),
    }
}

/// Validate a [`PixelBuffer`] against ultrahdr-core's dimension caps and
/// the kernels' format subset. Used at public entry points.
pub fn validate_ultrahdr_image(buffer: &PixelBuffer) -> Result<()> {
    validate_ultrahdr_dimensions(buffer.width(), buffer.height())?;
    require_supported_format(buffer.descriptor().pixel_format())?;
    Ok(())
}

/// Validate a [`PixelSlice`] against ultrahdr-core's dimension caps and
/// the kernels' format subset.
pub fn validate_ultrahdr_slice(slice: &PixelSlice<'_>) -> Result<()> {
    validate_ultrahdr_dimensions(slice.width(), slice.rows())?;
    require_supported_format(slice.descriptor().pixel_format())?;
    Ok(())
}

/// Allocate a [`PixelBuffer`] for ultrahdr-core's kernels.
///
/// Convenience over [`PixelBuffer::try_new`] that also enforces the stricter
/// ultrahdr-core dimension caps. The descriptor is built from `format`,
/// `primaries`, and `transfer`.
pub fn new_pixel_buffer(
    width: u32,
    height: u32,
    format: PixelFormat,
    primaries: ColorPrimaries,
    transfer: TransferFunction,
) -> Result<PixelBuffer> {
    validate_ultrahdr_dimensions(width, height)?;
    require_supported_format(format)?;
    let desc = descriptor_for(format, primaries, transfer);
    let buf = PixelBuffer::try_new(width, height, desc)
        .map_err(|e| Error::AllocationFailed(error_size_hint(&e)))?;
    Ok(buf)
}

/// Deep-copy a [`PixelBuffer`]. [`zenpixels::PixelBuffer`] intentionally does
/// not implement [`Clone`] to discourage silent large-pixel copies; this helper
/// gives ultrahdr-core callers a single owned duplicate (tightly packed, same
/// descriptor) when they genuinely need one.
pub fn clone_pixel_buffer(src: &PixelBuffer) -> PixelBuffer {
    src.crop_copy(0, 0, src.width(), src.height())
}

/// Wrap an existing `Vec<u8>` as a [`PixelBuffer`], validating ultrahdr-core's
/// dimension caps and format subset.
pub fn pixel_buffer_from_vec(
    data: Vec<u8>,
    width: u32,
    height: u32,
    format: PixelFormat,
    primaries: ColorPrimaries,
    transfer: TransferFunction,
) -> Result<PixelBuffer> {
    validate_ultrahdr_dimensions(width, height)?;
    require_supported_format(format)?;
    let desc = descriptor_for(format, primaries, transfer);
    PixelBuffer::from_vec(data, width, height, desc).map_err(Error::from)
}

fn error_size_hint<E>(_: &E) -> usize {
    0
}

/// A gain map image (8-bit grayscale or per-channel).
///
/// Kept bespoke: gain map bytes are log2-quantized gain, not color samples,
/// so a [`PixelBuffer`] descriptor would misrepresent them.
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
    /// Validate a [`GainMap`]'s field invariants.
    ///
    /// Constructors ([`GainMap::new`], [`GainMap::new_multichannel`])
    /// preserve these invariants, but the fields are `pub` for serde /
    /// codec interop, so a struct-literal `GainMap { ... }` can produce
    /// values that violate them. Every consumer that derefs `data`
    /// based on `width`/`height`/`channels` must call this first.
    ///
    /// Rejects:
    /// * `width == 0` / `height == 0` (every sampler does
    ///   `width - 1` arithmetic that would underflow);
    /// * `width` / `height` above [`limits::MAX_IMAGE_DIMENSION`];
    /// * `channels` outside `{1, 3}`;
    /// * `data.len() < width * height * channels`.
    pub fn validate(&self) -> Result<()> {
        validate_ultrahdr_dimensions(self.width, self.height)?;
        if self.channels != 1 && self.channels != 3 {
            return Err(Error::InvalidPixelData(alloc::format!(
                "gainmap.channels must be 1 or 3, got {}",
                self.channels
            )));
        }
        let expected = (self.width as usize)
            .checked_mul(self.height as usize)
            .and_then(|p| p.checked_mul(self.channels as usize))
            .ok_or_else(|| {
                Error::LimitExceeded(alloc::string::ToString::to_string(
                    &"gainmap dimensions overflow usize",
                ))
            })?;
        if self.data.len() < expected {
            return Err(Error::InvalidPixelData(alloc::format!(
                "gainmap data buffer too small: {} < {}",
                self.data.len(),
                expected
            )));
        }
        Ok(())
    }

    /// Create a new single-channel gain map.
    pub fn new(width: u32, height: u32) -> Result<Self> {
        validate_ultrahdr_dimensions(width, height)?;

        let size = (width as usize)
            .checked_mul(height as usize)
            .ok_or_else(|| Error::LimitExceeded("gain map size overflow".into()))?;

        Ok(Self {
            width,
            height,
            channels: 1,
            data: alloc::vec![0u8; size],
        })
    }

    /// Create a new multi-channel (RGB) gain map.
    pub fn new_multichannel(width: u32, height: u32) -> Result<Self> {
        validate_ultrahdr_dimensions(width, height)?;

        let size = (width as usize)
            .checked_mul(height as usize)
            .and_then(|s| s.checked_mul(3))
            .ok_or_else(|| Error::LimitExceeded("gain map size overflow".into()))?;

        Ok(Self {
            width,
            height,
            channels: 3,
            data: alloc::vec![0u8; size],
        })
    }
}

/// ISO 21496-1 gain map metadata.
///
/// Canonical [`zencodec::GainMapParams`] type. All gains and headroom values
/// are stored in **log2 domain** to match the ISO 21496-1 wire format.
pub type GainMapMetadata = zencodec::GainMapParams;

/// The display boost for a full reconstruction at the gain map's encoded
/// maximum — the `GainMapRender::ReconstructHdr { target_headroom: None }`
/// semantics: the alternate image's linear headroom, clamped to ≥ 1.0.
///
/// This is the **canonical rounding route**: f64 `exp2` of the log2
/// headroom ([`linear_alternate_headroom`]), rounded to f32 once. Adapters
/// must use this (not `2f32.powf(stops as f32)`, which double-rounds and
/// can land 1 ULP away) so reconstructions of the same parameters are
/// bit-identical across codecs (imazen/heic#20: the corpus 1520 file's
/// stops value rounds to `0x40caa5da` via `powf` but `0x40caa5db` via this
/// route, flipping 0.32% of PQ16 samples by one LSB downstream).
///
/// [`linear_alternate_headroom`]: zencodec::GainMapParams::linear_alternate_headroom
#[must_use]
pub fn full_reconstruction_boost(params: &GainMapMetadata) -> f32 {
    let boost = params.linear_alternate_headroom() as f32;
    if boost > 1.0 { boost } else { 1.0 }
}

/// Per-channel gain map parameters.
///
/// Re-exported from [`zencodec::GainMapChannel`].
pub use zencodec::GainMapChannel;

/// Reject gain map metadata whose magnitude would push f32 math to
/// `±inf` / `NaN` even though every value is `is_finite()` in f64.
///
/// This is a strict subset of [`validate_gainmap_metadata`] — it does
/// NOT inherit zencodec's `min <= max` invariant, which can be
/// violated by a few ULPs after ISO 21496-1 fraction quantization
/// without indicating an attack. The decode-side LUT independently
/// clamps decoded gains to a finite range, so the magnitude bound is
/// enough on its own to prevent `+inf` / `NaN` in the output buffer.
pub fn validate_gainmap_magnitude(metadata: &GainMapMetadata) -> Result<()> {
    if !metadata.alternate_hdr_headroom.is_finite() || !metadata.base_hdr_headroom.is_finite() {
        return Err(Error::InvalidMetadata("hdr headroom must be finite".into()));
    }
    if metadata.alternate_hdr_headroom.abs() > limits::MAX_HEADROOM_MAGNITUDE
        || metadata.base_hdr_headroom.abs() > limits::MAX_HEADROOM_MAGNITUDE
    {
        return Err(Error::InvalidMetadata(alloc::format!(
            "hdr headroom magnitude exceeds {} (log2 domain)",
            limits::MAX_HEADROOM_MAGNITUDE
        )));
    }
    for (i, ch) in metadata.channels.iter().enumerate() {
        if !ch.min.is_finite()
            || !ch.max.is_finite()
            || !ch.gamma.is_finite()
            || !ch.base_offset.is_finite()
            || !ch.alternate_offset.is_finite()
        {
            return Err(Error::InvalidMetadata(alloc::format!(
                "channel {i} contains non-finite values"
            )));
        }
        if ch.min.abs() > limits::MAX_LOG_GAIN_MAGNITUDE
            || ch.max.abs() > limits::MAX_LOG_GAIN_MAGNITUDE
        {
            return Err(Error::InvalidMetadata(alloc::format!(
                "channel {} min/max magnitude exceeds {} (log2 domain)",
                i,
                limits::MAX_LOG_GAIN_MAGNITUDE
            )));
        }
        if ch.base_offset.abs() > limits::MAX_OFFSET_MAGNITUDE
            || ch.alternate_offset.abs() > limits::MAX_OFFSET_MAGNITUDE
        {
            return Err(Error::InvalidMetadata(alloc::format!(
                "channel {} offset magnitude exceeds {} (linear domain)",
                i,
                limits::MAX_OFFSET_MAGNITUDE
            )));
        }
        if !(limits::MIN_GAMMA..=limits::MAX_GAMMA).contains(&ch.gamma) {
            return Err(Error::InvalidMetadata(alloc::format!(
                "channel {} gamma must be within [{}, {}]",
                i,
                limits::MIN_GAMMA,
                limits::MAX_GAMMA
            )));
        }
    }
    Ok(())
}

/// Validate gain map metadata with ultrahdr-core's stricter checks.
///
/// Delegates to [`zencodec::GainMapParams::validate()`] for `is_finite`
/// / `min ≤ max` / `gamma > 0`, then additionally rejects:
///
/// * `alternate_hdr_headroom < 0` (log2 domain) — base zencodec
///   validation does not enforce sign.
/// * `|min|`, `|max|` greater than [`limits::MAX_LOG_GAIN_MAGNITUDE`].
///   Without this bound a finite f64 like `1e30` casts to `f32 +inf`
///   inside the LUT and silently propagates `+inf` into the HDR
///   output (`LinearFloat` / `LinearF16` arms write the raw float;
///   the `Srgb8` arm clamps but is still wrong).
/// * `|alternate_hdr_headroom|`, `|base_hdr_headroom|` greater than
///   [`limits::MAX_HEADROOM_MAGNITUDE`] — same `2^headroom` overflow
///   risk as `min`/`max`.
/// * `|base_offset|`, `|alternate_offset|` greater than
///   [`limits::MAX_OFFSET_MAGNITUDE`] — keeps
///   `(sdr + offset) / gain - other_offset` finite.
/// * `gamma` outside `[`[`limits::MIN_GAMMA`]`,` [`limits::MAX_GAMMA`]`]` —
///   above-spec values either produce a near-step function (`> 100`)
///   or saturate `powf` (`< 0.01`).
pub fn validate_gainmap_metadata(metadata: &GainMapMetadata) -> Result<()> {
    metadata
        .validate()
        .map_err(|e| Error::InvalidMetadata(alloc::string::ToString::to_string(&e)))?;
    if metadata.alternate_hdr_headroom < 0.0 {
        return Err(Error::InvalidMetadata(
            "alternate_hdr_headroom must be >= 0.0 (log2 domain)".into(),
        ));
    }
    if metadata.alternate_hdr_headroom.abs() > limits::MAX_HEADROOM_MAGNITUDE
        || metadata.base_hdr_headroom.abs() > limits::MAX_HEADROOM_MAGNITUDE
    {
        return Err(Error::InvalidMetadata(alloc::format!(
            "hdr headroom magnitude exceeds {} (log2 domain)",
            limits::MAX_HEADROOM_MAGNITUDE
        )));
    }
    for (i, ch) in metadata.channels.iter().enumerate() {
        if ch.min.abs() > limits::MAX_LOG_GAIN_MAGNITUDE
            || ch.max.abs() > limits::MAX_LOG_GAIN_MAGNITUDE
        {
            return Err(Error::InvalidMetadata(alloc::format!(
                "channel {} min/max magnitude exceeds {} (log2 domain)",
                i,
                limits::MAX_LOG_GAIN_MAGNITUDE
            )));
        }
        if ch.base_offset.abs() > limits::MAX_OFFSET_MAGNITUDE
            || ch.alternate_offset.abs() > limits::MAX_OFFSET_MAGNITUDE
        {
            return Err(Error::InvalidMetadata(alloc::format!(
                "channel {} offset magnitude exceeds {} (linear domain)",
                i,
                limits::MAX_OFFSET_MAGNITUDE
            )));
        }
        if !(limits::MIN_GAMMA..=limits::MAX_GAMMA).contains(&ch.gamma) {
            return Err(Error::InvalidMetadata(alloc::format!(
                "channel {} gamma must be within [{}, {}]",
                i,
                limits::MIN_GAMMA,
                limits::MAX_GAMMA
            )));
        }
    }
    Ok(())
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

    /// Helper to construct GainMapMetadata from per-channel arrays.
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
        metadata_from_arrays(
            min,
            max,
            gamma,
            base_offset,
            alt_offset,
            base_hdr_headroom,
            alt_hdr_headroom,
            use_base_cs,
            backward,
        )
    }

    #[test]
    fn test_error_from_stop_reason() {
        let err: Error = StopReason::Cancelled.into();
        assert!(matches!(err, Error::Stopped(StopReason::Cancelled)));
    }

    #[test]
    fn test_dimension_limits() {
        assert!(validate_ultrahdr_dimensions(1920, 1080).is_ok());
        assert!(validate_ultrahdr_dimensions(0, 100).is_err());
        assert!(validate_ultrahdr_dimensions(100, 0).is_err());
        assert!(validate_ultrahdr_dimensions(100_000, 100).is_err());
    }

    #[test]
    fn test_new_pixel_buffer_validates() {
        let buf = new_pixel_buffer(
            16,
            16,
            PixelFormat::Rgba8,
            ColorPrimaries::Bt709,
            TransferFunction::Srgb,
        )
        .unwrap();
        assert_eq!(buf.width(), 16);
        assert_eq!(buf.height(), 16);
        assert_eq!(buf.descriptor().pixel_format(), PixelFormat::Rgba8);
        assert_eq!(buf.descriptor().primaries, ColorPrimaries::Bt709);

        // Reject zero dimensions.
        assert!(
            new_pixel_buffer(
                0,
                16,
                PixelFormat::Rgba8,
                ColorPrimaries::Bt709,
                TransferFunction::Srgb
            )
            .is_err()
        );

        // Reject unsupported formats.
        assert!(matches!(
            new_pixel_buffer(
                16,
                16,
                PixelFormat::Cmyk8,
                ColorPrimaries::Bt709,
                TransferFunction::Srgb
            ),
            Err(Error::UnsupportedFormat(_))
        ));
    }

    #[test]
    fn test_gain_map_metadata_validation() {
        let mut metadata = GainMapMetadata::default();
        assert!(metadata.validate().is_ok());

        metadata.channels[0].gamma = f64::NAN;
        assert!(metadata.validate().is_err());

        metadata.channels[0].gamma = 1.0;
        metadata.channels[1].max = -1.0;
        assert!(metadata.validate().is_err());
    }

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
    }

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

    #[test]
    fn test_iso21496_format_identity() {
        assert_eq!(Iso21496Format::AvifTmap, zencodec::Iso21496Format::AvifTmap);
        assert_eq!(Iso21496Format::JxlJhgm, zencodec::Iso21496Format::JxlJhgm);
        assert_eq!(
            Iso21496Format::JpegApp2BodyWithUrn,
            zencodec::Iso21496Format::JpegApp2BodyWithUrn
        );
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

    // Regression: large finite metadata values used to cast `f64 * ln2` to
    // `f32 +inf`, propagating `+inf` into LinearFloat / LinearF16 output
    // (security audit 2026-05-06, finding C1).
    #[test]
    fn validate_gainmap_magnitude_rejects_huge_finite_min_max() {
        let metadata = make_metadata(
            [1.0e30; 3], // log2 domain — clearly way past 30
            [1.0e30; 3],
            [1.0; 3],
            [0.0; 3],
            [0.0; 3],
            0.0,
            2.0,
            true,
            false,
        );
        let err = validate_gainmap_magnitude(&metadata).unwrap_err();
        let msg = alloc::string::ToString::to_string(&err);
        assert!(
            msg.contains("magnitude") && msg.contains("log2"),
            "expected magnitude/log2 message: {msg}"
        );
    }

    #[test]
    fn validate_gainmap_magnitude_rejects_huge_headroom() {
        let metadata = make_metadata(
            [0.0; 3], [2.0; 3], [1.0; 3], [0.0; 3], [0.0; 3], 0.0,
            1.0e10, // alt headroom way past 30
            true, false,
        );
        assert!(validate_gainmap_magnitude(&metadata).is_err());
    }

    #[test]
    fn validate_gainmap_magnitude_rejects_huge_offsets() {
        let mut metadata = make_metadata(
            [0.0; 3], [2.0; 3], [1.0; 3],
            [100.0; 3], // base_offset way past MAX_OFFSET_MAGNITUDE
            [0.0; 3], 0.0, 2.0, true, false,
        );
        assert!(validate_gainmap_magnitude(&metadata).is_err());

        metadata.channels[0].base_offset = 0.0;
        metadata.channels[0].alternate_offset = 100.0;
        assert!(validate_gainmap_magnitude(&metadata).is_err());
    }

    #[test]
    fn validate_gainmap_magnitude_rejects_extreme_gamma() {
        let metadata = make_metadata(
            [0.0; 3],
            [2.0; 3],
            [1000.0, 1.0, 1.0], // gamma > MAX_GAMMA
            [0.0; 3],
            [0.0; 3],
            0.0,
            2.0,
            true,
            false,
        );
        assert!(validate_gainmap_magnitude(&metadata).is_err());
    }

    #[test]
    fn validate_gainmap_magnitude_accepts_realistic_values() {
        let metadata = make_metadata(
            [-2.0, -2.0, -2.0],
            [3.0, 3.0, 3.0],
            [1.0, 1.0, 1.0],
            [1.0 / 64.0; 3],
            [1.0 / 64.0; 3],
            0.0,
            3.0,
            true,
            false,
        );
        assert!(validate_gainmap_magnitude(&metadata).is_ok());
    }

    // Regression: zero-dim GainMap used to underflow `width - 1` in every
    // sampler (security audit 2026-05-06, finding C2).
    #[test]
    fn gainmap_validate_rejects_zero_dim() {
        let gm = GainMap {
            width: 0,
            height: 4,
            channels: 1,
            data: alloc::vec![0u8; 0],
        };
        assert!(gm.validate().is_err());

        let gm = GainMap {
            width: 4,
            height: 0,
            channels: 1,
            data: alloc::vec![0u8; 0],
        };
        assert!(gm.validate().is_err());
    }

    #[test]
    fn gainmap_validate_rejects_bad_channels() {
        for c in [0u8, 2, 4, 255] {
            let gm = GainMap {
                width: 2,
                height: 2,
                channels: c,
                data: alloc::vec![0u8; 16],
            };
            assert!(gm.validate().is_err(), "channels={c} should be rejected");
        }
    }

    #[test]
    fn gainmap_validate_rejects_short_data_buffer() {
        let gm = GainMap {
            width: 4,
            height: 4,
            channels: 1,
            data: alloc::vec![0u8; 4], // expected 16
        };
        assert!(gm.validate().is_err());
    }
}

#[cfg(test)]
mod boost_route_tests {
    use super::*;

    /// heic#20 pin: the canonical route differs from the double-rounding
    /// `powf` route by exactly 1 ULP for the corpus 1520 stops value — and
    /// the helper is the f64-exp2 single-rounding side.
    #[test]
    fn canonical_boost_is_single_rounded() {
        let mut params = GainMapMetadata::default();
        params.alternate_hdr_headroom = 2.662831178973982_f64;
        let canonical = full_reconstruction_boost(&params);
        assert_eq!(canonical.to_bits(), 0x40ca_a5db);
        let double_rounded = 2.0f32.powf(params.alternate_hdr_headroom as f32);
        assert_eq!(double_rounded.to_bits(), 0x40ca_a5da);
        assert_ne!(canonical.to_bits(), double_rounded.to_bits());
    }

    /// Negative/zero headroom clamps to 1.0 (never darken below the base).
    #[test]
    fn boost_clamps_to_one() {
        let mut params = GainMapMetadata::default();
        params.alternate_hdr_headroom = -0.5;
        assert_eq!(full_reconstruction_boost(&params), 1.0);
        params.alternate_hdr_headroom = 0.0;
        assert_eq!(full_reconstruction_boost(&params), 1.0);
    }
}
