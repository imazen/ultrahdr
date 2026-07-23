//! Ultra HDR encoder.

use ultrahdr_core::color::tonemap::tonemap_image_to_srgb8;

use ultrahdr_core::gainmap::compute::{GainMapConfig, compute_gainmap};
use ultrahdr_core::{ColorPrimaries, Error, GainMapEncodingFormat, GainMapMetadata, Result};

use ultrahdr_core::{PixelFormat, Stop, TransferFunction, Unstoppable};

use ultrahdr_core::{GainMap, PixelBuffer, clone_pixel_buffer, pixel_buffer_from_vec};

use whereat::at;
use zencodec::Iso21496Format;
use zencodec::gainmap::{ISO_21496_1_PRIMARY_APP2_BODY, serialize_iso21496_fmt};
use zenjpeg::container::mpf::create_mpf_header;
use zenjpeg::container::xmp::{create_xmp_app1_marker, generate_gainmap_xmp, generate_primary_xmp};

use crate::jpeg::{
    JpegSegment, create_icc_markers, get_icc_profile_for_gamut, insert_segment_after_soi,
};

/// Wrap a body in a JPEG APP2 marker (`FF E2` + big-endian length including
/// the length bytes themselves + body).
///
/// Inlined equivalent of the retired
/// `ultrahdr_core::metadata::iso_jpeg::create_iso_app2_marker` helper.
fn wrap_app2(body: &[u8]) -> Vec<u8> {
    let total_length = 2 + body.len();
    let mut marker = Vec::with_capacity(4 + body.len());
    marker.extend_from_slice(&[
        0xFF,
        0xE2,
        ((total_length >> 8) & 0xFF) as u8,
        (total_length & 0xFF) as u8,
    ]);
    marker.extend_from_slice(body);
    marker
}

/// Assemble the gain-map secondary's metadata APP markers (XMP APP1 and/or
/// ISO 21496-1 APP2) in canonical order. Inlined equivalent of the retired
/// `ultrahdr_core::metadata::xmp::build_gainmap_metadata_markers`.
fn build_gainmap_metadata_markers(
    metadata: &GainMapMetadata,
    format: GainMapEncodingFormat,
) -> Vec<Vec<u8>> {
    let mut markers = Vec::with_capacity(2);
    if matches!(
        format,
        GainMapEncodingFormat::Xmp | GainMapEncodingFormat::Both
    ) {
        let xmp = generate_gainmap_xmp(metadata);
        markers.push(create_xmp_app1_marker(&xmp));
    }
    if matches!(
        format,
        GainMapEncodingFormat::Iso21496 | GainMapEncodingFormat::Both
    ) {
        // Preserves prior behavior: `JxlJhgm` and `JpegApp2` serialize the
        // same payload (zencodec::gainmap::serialize_iso21496_fmt_into).
        let iso_body = serialize_iso21496_fmt(metadata, Iso21496Format::JpegApp2BodyWithUrn);
        markers.push(wrap_app2(&iso_body));
    }
    markers
}

/// Retired helper: version-only ISO 21496-1 APP2 for the primary JPEG.
///
/// Thin wrapper around `zencodec::gainmap::ISO_21496_1_PRIMARY_APP2_BODY`.
fn create_version_only_iso_app2() -> Vec<u8> {
    wrap_app2(ISO_21496_1_PRIMARY_APP2_BODY)
}

/// Assemble an Ultra HDR JPEG from pre-encoded components.
///
/// Uses [`GainMapEncodingFormat::Both`] (XMP + ISO 21496-1) for maximum compatibility.
/// For format control, use [`encode_ultrahdr_with_format`].
pub fn encode_ultrahdr(
    base_jpeg: &[u8],
    gainmap_jpeg: &[u8],
    metadata: &GainMapMetadata,
    gamut: ColorPrimaries,
) -> Result<Vec<u8>> {
    encode_ultrahdr_with_format(
        base_jpeg,
        gainmap_jpeg,
        metadata,
        gamut,
        GainMapEncodingFormat::Both,
    )
}

/// Assemble an Ultra HDR JPEG from pre-encoded components with format control.
///
/// - `base_jpeg`: Pre-encoded SDR JPEG (the backwards-compatible base image)
/// - `gainmap_jpeg`: Pre-encoded gain map JPEG (typically grayscale)
/// - `metadata`: Gain map metadata describing how to apply the gain map
/// - `gamut`: Color gamut of the base image (for ICC profile selection)
/// - `format`: Which metadata format(s) to embed in the gain map JPEG
pub fn encode_ultrahdr_with_format(
    base_jpeg: &[u8],
    gainmap_jpeg: &[u8],
    metadata: &GainMapMetadata,
    gamut: ColorPrimaries,
    format: GainMapEncodingFormat,
) -> Result<Vec<u8>> {
    // Build metadata markers for the gain map JPEG (XMP and/or ISO 21496-1).
    let metadata_markers = build_gainmap_metadata_markers(metadata, format);

    // Inject metadata markers into the gain map JPEG after SOI.
    let mut gainmap_final = gainmap_jpeg.to_vec();
    // Insert in reverse order so each goes right after SOI
    for marker in metadata_markers.iter().rev() {
        let segment = JpegSegment {
            marker: marker[1], // FF xx — take the marker byte
            data: marker[4..].to_vec(),
            offset: 0,
        };
        gainmap_final = insert_segment_after_soi(&gainmap_final, &segment)?;
    }

    // Generate primary XMP with container directory (points to gain map by size).
    let primary_xmp = generate_primary_xmp(gainmap_final.len());
    let primary_xmp_marker = create_xmp_app1_marker(&primary_xmp);

    // Generate ICC profile
    let icc_profile = get_icc_profile_for_gamut(gamut);
    let icc_markers = create_icc_markers(&icc_profile);

    // Insert primary XMP after SOI
    let xmp_segment = JpegSegment {
        marker: 0xE1,
        data: primary_xmp_marker[4..].to_vec(),
        offset: 0,
    };
    let mut primary = insert_segment_after_soi(base_jpeg, &xmp_segment)?;

    // Insert version-only ISO 21496-1 APP2 into primary JPEG when ISO is enabled.
    // This 4-byte block (min_version=0, writer_version=0) signals ISO 21496-1
    // awareness. The actual gain map metadata lives in the secondary JPEG's APP2.
    let include_iso = matches!(
        format,
        GainMapEncodingFormat::Iso21496 | GainMapEncodingFormat::Both
    );
    if include_iso {
        let version_marker = create_version_only_iso_app2();
        let iso_segment = JpegSegment {
            marker: 0xE2,
            data: version_marker[4..].to_vec(),
            offset: 0,
        };
        primary = insert_segment_after_soi(&primary, &iso_segment)?;
    }

    // Insert ICC markers
    for icc_marker in &icc_markers {
        let icc_segment = JpegSegment {
            marker: 0xE2,
            data: icc_marker[4..].to_vec(),
            offset: 0,
        };
        primary = insert_segment_after_soi(&primary, &icc_segment)?;
    }

    // Calculate sizes for MPF
    let mpf_insert_pos = 2;
    let mpf_estimate = create_mpf_header(0, 0, Some(mpf_insert_pos)).len();
    let primary_with_mpf_len = primary.len() + mpf_estimate;

    // Create MPF header
    let mpf_header = create_mpf_header(
        primary_with_mpf_len,
        gainmap_final.len(),
        Some(mpf_insert_pos),
    );

    // Insert MPF header
    let mpf_segment = JpegSegment {
        marker: 0xE2,
        data: mpf_header[4..].to_vec(),
        offset: 0,
    };
    let primary_final = insert_segment_after_soi(&primary, &mpf_segment)?;

    // Concatenate primary and gain map
    let mut result = primary_final;
    result.extend_from_slice(&gainmap_final);

    Ok(result)
}

/// Ultra HDR encoder.
///
/// For production use without a bundled JPEG codec, use [`encode_ultrahdr`] directly.
///
/// The builder methods that require a JPEG codec (`set_hdr_image`, `set_sdr_image`,
/// `encode`) are only available in tests where zenjpeg is a dev-dependency.
#[derive(Default)]
pub struct Encoder {
    hdr_image: Option<PixelBuffer>,

    sdr_image: Option<PixelBuffer>,
    compressed_sdr: Option<Vec<u8>>,

    existing_gainmap: Option<GainMap>,
    existing_metadata: Option<GainMapMetadata>,
    existing_gainmap_jpeg: Option<Vec<u8>>,
    base_quality: u8,
    gainmap_quality: u8,
    gainmap_scale: u8,
    target_display_peak: f32,
    gain_map_min: f32,

    use_iso_metadata: bool,
}

impl Encoder {
    /// Create a new encoder with default settings.
    pub fn new() -> Self {
        Self {
            hdr_image: None,

            sdr_image: None,
            compressed_sdr: None,

            existing_gainmap: None,
            existing_metadata: None,
            existing_gainmap_jpeg: None,
            base_quality: 90,
            gainmap_quality: 85,
            gainmap_scale: 4,
            target_display_peak: 10000.0,
            gain_map_min: 1.0,

            use_iso_metadata: true,
        }
    }

    /// Set the HDR input image.
    pub fn set_hdr_image(&mut self, image: PixelBuffer) -> &mut Self {
        self.hdr_image = Some(image);
        self
    }

    /// Set the SDR input image.
    pub fn set_sdr_image(&mut self, image: PixelBuffer) -> &mut Self {
        self.sdr_image = Some(image);
        self
    }

    /// Set a pre-compressed SDR JPEG.
    pub fn set_compressed_sdr(&mut self, jpeg: Vec<u8>) -> &mut Self {
        self.compressed_sdr = Some(jpeg);
        self
    }

    /// Alias for set_compressed_sdr.
    pub fn set_base_jpeg(&mut self, jpeg: Vec<u8>) -> &mut Self {
        self.set_compressed_sdr(jpeg)
    }

    /// Set an existing gain map and metadata.
    pub fn set_existing_gainmap(
        &mut self,
        gainmap: GainMap,
        metadata: GainMapMetadata,
    ) -> &mut Self {
        self.existing_gainmap = Some(gainmap);
        self.existing_metadata = Some(metadata);
        self
    }

    /// Clear any existing gain map.
    pub fn clear_existing_gainmap(&mut self) -> &mut Self {
        self.existing_gainmap = None;
        self.existing_metadata = None;
        self.existing_gainmap_jpeg = None;
        self
    }

    /// Set an existing gain map as raw JPEG bytes and metadata.
    pub fn set_existing_gainmap_jpeg(
        &mut self,
        jpeg: Vec<u8>,
        metadata: GainMapMetadata,
    ) -> &mut Self {
        self.existing_gainmap_jpeg = Some(jpeg);
        self.existing_metadata = Some(metadata);
        self
    }

    /// Alias for set_existing_gainmap_jpeg.
    pub fn set_gainmap_jpeg(&mut self, jpeg: Vec<u8>, metadata: GainMapMetadata) -> &mut Self {
        self.set_existing_gainmap_jpeg(jpeg, metadata)
    }

    /// Check if an existing gain map is set.
    pub fn has_existing_gainmap(&self) -> bool {
        self.existing_gainmap.is_some() && self.existing_metadata.is_some()
    }

    /// Set JPEG quality for base and gain map images.
    pub fn set_quality(&mut self, base: u8, gainmap: u8) -> &mut Self {
        self.base_quality = base.clamp(1, 100);
        self.gainmap_quality = gainmap.clamp(1, 100);
        self
    }

    /// Set gain map downscale factor.
    pub fn set_gainmap_scale(&mut self, scale: u8) -> &mut Self {
        self.gainmap_scale = scale.clamp(1, 128);
        self
    }

    /// Set target display peak brightness in nits.
    pub fn set_target_display_peak(&mut self, nits: f32) -> &mut Self {
        self.target_display_peak = nits.max(100.0);
        self
    }

    /// Set minimum content boost.
    pub fn set_min_content_boost(&mut self, boost: f32) -> &mut Self {
        self.gain_map_min = boost.max(1.0);
        self
    }

    /// Enable or disable ISO 21496-1 metadata.
    pub fn set_use_iso_metadata(&mut self, use_iso: bool) -> &mut Self {
        self.use_iso_metadata = use_iso;
        self
    }

    /// Encode to Ultra HDR JPEG.
    pub fn encode(&self) -> Result<Vec<u8>> {
        self.encode_with_stop(Unstoppable)
    }

    /// [`encode`](Self::encode) with cooperative cancellation.
    ///
    /// The `stop` token is checked throughout gain-map computation and both
    /// JPEG encodes. Cancellation surfaces as
    /// [`Error::Stopped`](ultrahdr_core::Error::Stopped).
    pub fn encode_with_stop(&self, stop: impl Stop) -> Result<Vec<u8>> {
        stop.check().map_err(|r| at!(Error::Stopped(r)))?;
        // Fast path: if we have raw gain map JPEG bytes, skip gain map processing
        if let (Some(gainmap_jpeg), Some(metadata)) =
            (&self.existing_gainmap_jpeg, &self.existing_metadata)
        {
            let (base_jpeg, gamut) = if let Some(ref compressed) = self.compressed_sdr {
                (compressed.clone(), ColorPrimaries::Bt709)
            } else if let Some(ref sdr_img) = self.sdr_image {
                let gamut = sdr_img.descriptor().primaries;
                (self.encode_base_jpeg(sdr_img, &stop)?, gamut)
            } else if let Some(ref hdr) = self.hdr_image {
                let sdr_pixels = tonemap_image_to_srgb8(hdr, ColorPrimaries::Bt709)?;
                let sdr = pixel_buffer_from_vec(
                    sdr_pixels,
                    hdr.width(),
                    hdr.height(),
                    PixelFormat::Rgba8,
                    ColorPrimaries::Bt709,
                    TransferFunction::Srgb,
                )?;
                let gamut = sdr.descriptor().primaries;
                (self.encode_base_jpeg(&sdr, &stop)?, gamut)
            } else {
                return Err(at!(Error::EncodeError(
                    "Either HDR image, SDR image, or compressed SDR is required".into(),
                )));
            };

            return encode_ultrahdr(&base_jpeg, gainmap_jpeg, metadata, gamut);
        }

        // Validate inputs
        let hdr = self
            .hdr_image
            .as_ref()
            .ok_or_else(|| at!(Error::EncodeError("HDR image is required".into())))?;

        // Generate or use provided SDR
        let sdr: PixelBuffer = if let Some(ref sdr_img) = self.sdr_image {
            clone_pixel_buffer(sdr_img)
        } else {
            let sdr_pixels = tonemap_image_to_srgb8(hdr, ColorPrimaries::Bt709)?;
            pixel_buffer_from_vec(
                sdr_pixels,
                hdr.width(),
                hdr.height(),
                PixelFormat::Rgba8,
                ColorPrimaries::Bt709,
                TransferFunction::Srgb,
            )?
        };

        // Use existing gain map if provided, otherwise compute a new one
        let (gainmap, metadata) =
            if let (Some(gm), Some(meta)) = (&self.existing_gainmap, &self.existing_metadata) {
                let expected_scale = self.gainmap_scale.max(1) as u32;
                let expected_width = sdr.width().div_ceil(expected_scale);
                let expected_height = sdr.height().div_ceil(expected_scale);

                let width_ok =
                    gm.width >= expected_width.saturating_sub(1) && gm.width <= expected_width + 1;
                let height_ok = gm.height >= expected_height.saturating_sub(1)
                    && gm.height <= expected_height + 1;

                if width_ok && height_ok {
                    (gm.clone(), meta.clone())
                } else {
                    self.compute_new_gainmap(hdr, &sdr, &stop)?
                }
            } else {
                self.compute_new_gainmap(hdr, &sdr, &stop)?
            };

        // Encode base JPEG
        let base_jpeg = if let Some(ref compressed) = self.compressed_sdr {
            compressed.clone()
        } else {
            self.encode_base_jpeg(&sdr, &stop)?
        };

        // Encode gain map JPEG
        let gainmap_jpeg = self.encode_gainmap_jpeg(&gainmap, &stop)?;

        let gamut = sdr.descriptor().primaries;
        encode_ultrahdr(&base_jpeg, &gainmap_jpeg, &metadata, gamut)
    }

    /// Encode to Ultra HDR JPEG from pre-set JPEGs (production API).
    pub fn encode_from_jpegs(&self) -> Result<Vec<u8>> {
        let base_jpeg = self
            .compressed_sdr
            .as_ref()
            .ok_or_else(|| at!(Error::EncodeError("Base JPEG not set".into())))?;

        let gainmap_jpeg = self
            .existing_gainmap_jpeg
            .as_ref()
            .ok_or_else(|| at!(Error::EncodeError("Gainmap JPEG not set".into())))?;

        let metadata = self
            .existing_metadata
            .as_ref()
            .ok_or_else(|| at!(Error::EncodeError("Metadata not set".into())))?;

        encode_ultrahdr(base_jpeg, gainmap_jpeg, metadata, ColorPrimaries::Bt709)
    }

    /// Compute a new gain map.
    fn compute_new_gainmap(
        &self,
        hdr: &PixelBuffer,
        sdr: &PixelBuffer,
        stop: impl Stop,
    ) -> Result<(GainMap, GainMapMetadata)> {
        let config = GainMapConfig {
            scale_factor: self.gainmap_scale,
            gamma: 1.0,
            multi_channel: false,
            min_boost: self.gain_map_min,
            max_boost: self.target_display_peak / 203.0,
            base_offset: 1.0 / 64.0,
            alternate_offset: 1.0 / 64.0,
            base_hdr_headroom: 1.0, // linear: 1.0 = no boost → log2 = 0.0
            alternate_hdr_headroom: self.target_display_peak / 203.0,
        };

        compute_gainmap(hdr, sdr, &config, stop)
    }

    /// Encode base SDR image to JPEG.
    fn encode_base_jpeg(&self, sdr: &PixelBuffer, stop: impl Stop) -> Result<Vec<u8>> {
        use zenjpeg::encoder::{ChromaSubsampling, EncoderConfig, PixelLayout};

        let format = sdr.descriptor().pixel_format();
        let src_bytes = sdr.as_slice();
        let src_data = src_bytes.as_strided_bytes();
        let (pixel_layout, data): (PixelLayout, std::borrow::Cow<[u8]>) = match format {
            PixelFormat::Rgba8 => {
                let rgb: Vec<u8> = src_data
                    .chunks(4)
                    .flat_map(|rgba| [rgba[0], rgba[1], rgba[2]])
                    .collect();
                (PixelLayout::Rgb8Srgb, std::borrow::Cow::Owned(rgb))
            }
            PixelFormat::Rgb8 => (PixelLayout::Rgb8Srgb, std::borrow::Cow::Borrowed(src_data)),
            _ => {
                return Err(at!(Error::EncodeError(format!(
                    "Unsupported SDR pixel format: {:?}",
                    format
                ))));
            }
        };

        let config = EncoderConfig::ycbcr(self.base_quality as f32, ChromaSubsampling::Quarter);
        let mut enc = config
            .encode_from_bytes(sdr.width(), sdr.height(), pixel_layout)
            .map_err(map_jpeg_encode_error)?;
        enc.push_packed(&data, stop)
            .map_err(map_jpeg_encode_error)?;
        enc.finish().map_err(map_jpeg_encode_error)
    }

    /// Encode gain map to JPEG.
    fn encode_gainmap_jpeg(&self, gainmap: &GainMap, stop: impl Stop) -> Result<Vec<u8>> {
        use zenjpeg::encoder::{EncoderConfig, PixelLayout};

        let config = EncoderConfig::grayscale(self.gainmap_quality as f32);
        let mut enc = config
            .encode_from_bytes(gainmap.width, gainmap.height, PixelLayout::Gray8Srgb)
            .map_err(map_jpeg_encode_error)?;
        enc.push_packed(&gainmap.data, stop)
            .map_err(map_jpeg_encode_error)?;
        enc.finish().map_err(map_jpeg_encode_error)
    }
}

/// Map a zenjpeg encode error to a typed ultrahdr error.
///
/// Cancellation keeps its type
/// ([`Error::Stopped`](ultrahdr_core::Error::Stopped)) instead of
/// collapsing into a [`Error::JpegEncode`] string. Returns the `At<Error>`
/// wrapper directly — converting a built `At` back into a bare `Error`
/// would route through core's `From<zenpixels::At<E>>` impl and stringify
/// the variant.
fn map_jpeg_encode_error(e: zenjpeg::encoder::Error) -> whereat::At<Error> {
    use zenjpeg::encoder::ErrorKind;
    match e.kind() {
        ErrorKind::Cancelled(reason) => at!(Error::Stopped(*reason)),
        _ => at!(Error::JpegEncode(e.to_string())),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_encoder_creation() {
        let encoder = Encoder::new();
        assert_eq!(encoder.base_quality, 90);
        assert_eq!(encoder.gainmap_quality, 85);
        assert_eq!(encoder.gainmap_scale, 4);
    }

    #[test]
    fn test_encoder_builder() {
        let mut encoder = Encoder::new();
        encoder
            .set_quality(95, 90)
            .set_gainmap_scale(2)
            .set_target_display_peak(4000.0);

        assert_eq!(encoder.base_quality, 95);
        assert_eq!(encoder.gainmap_quality, 90);
        assert_eq!(encoder.gainmap_scale, 2);
        assert_eq!(encoder.target_display_peak, 4000.0);
    }

    #[test]
    fn test_encode_requires_hdr() {
        let encoder = Encoder::new();
        let result = encoder.encode();
        assert!(result.is_err());
    }

    #[test]
    fn test_existing_gainmap_methods() {
        let mut encoder = Encoder::new();

        assert!(!encoder.has_existing_gainmap());

        let gainmap = GainMap::new(100, 100).unwrap();
        let metadata = GainMapMetadata::default();
        encoder.set_existing_gainmap(gainmap, metadata);
        assert!(encoder.has_existing_gainmap());

        encoder.clear_existing_gainmap();
        assert!(!encoder.has_existing_gainmap());
    }
}
