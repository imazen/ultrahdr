//! Ultra HDR encoder.

#[cfg(feature = "_test-helpers")]
use ultrahdr_core::color::tonemap::tonemap_image_to_srgb8;
#[cfg(feature = "_test-helpers")]
use ultrahdr_core::gainmap::compute::{compute_gainmap, GainMapConfig};
use ultrahdr_core::metadata::{
    mpf::create_mpf_header,
    xmp::{create_xmp_app1_marker, generate_xmp},
};
use ultrahdr_core::{ColorGamut, Error, GainMapMetadata, Result};
#[cfg(feature = "_test-helpers")]
use ultrahdr_core::{ColorTransfer, PixelFormat, Unstoppable};
#[cfg(feature = "_test-helpers")]
use ultrahdr_core::{GainMap, RawImage};

use crate::jpeg::{
    create_icc_markers, get_icc_profile_for_gamut, insert_segment_after_soi, JpegSegment,
};

/// Assemble an Ultra HDR JPEG from pre-encoded components.
///
/// This is the primary encoding function. You provide:
/// - `base_jpeg`: Pre-encoded SDR JPEG (the backwards-compatible base image)
/// - `gainmap_jpeg`: Pre-encoded gain map JPEG (typically grayscale)
/// - `metadata`: Gain map metadata describing how to apply the gain map
/// - `gamut`: Color gamut of the base image (for ICC profile selection)
///
/// # Example
///
/// ```ignore
/// use ultrahdr::{encode_ultrahdr, GainMapMetadata, ColorGamut};
///
/// let ultrahdr = encode_ultrahdr(&base_jpeg, &gainmap_jpeg, &metadata, ColorGamut::Bt709)?;
/// ```
pub fn encode_ultrahdr(
    base_jpeg: &[u8],
    gainmap_jpeg: &[u8],
    metadata: &GainMapMetadata,
    gamut: ColorGamut,
) -> Result<Vec<u8>> {
    // Generate XMP
    let xmp = generate_xmp(metadata, gainmap_jpeg.len());
    let xmp_marker = create_xmp_app1_marker(&xmp);

    // Generate ICC profile
    let icc_profile = get_icc_profile_for_gamut(gamut);
    let icc_markers = create_icc_markers(&icc_profile);

    // Insert XMP after SOI
    let xmp_segment = JpegSegment {
        marker: 0xE1,
        data: xmp_marker[4..].to_vec(), // Skip FF E1 and length
        offset: 0,
    };
    let mut primary = insert_segment_after_soi(base_jpeg, &xmp_segment)?;

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
        gainmap_jpeg.len(),
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
    result.extend_from_slice(gainmap_jpeg);

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
    #[cfg(feature = "_test-helpers")]
    hdr_image: Option<RawImage>,
    #[cfg(feature = "_test-helpers")]
    sdr_image: Option<RawImage>,
    compressed_sdr: Option<Vec<u8>>,
    #[cfg(feature = "_test-helpers")]
    existing_gainmap: Option<GainMap>,
    existing_metadata: Option<GainMapMetadata>,
    existing_gainmap_jpeg: Option<Vec<u8>>,
    base_quality: u8,
    gainmap_quality: u8,
    gainmap_scale: u8,
    target_display_peak: f32,
    min_content_boost: f32,
    #[cfg(feature = "_test-helpers")]
    use_iso_metadata: bool,
}

impl Encoder {
    /// Create a new encoder with default settings.
    pub fn new() -> Self {
        Self {
            #[cfg(feature = "_test-helpers")]
            hdr_image: None,
            #[cfg(feature = "_test-helpers")]
            sdr_image: None,
            compressed_sdr: None,
            #[cfg(feature = "_test-helpers")]
            existing_gainmap: None,
            existing_metadata: None,
            existing_gainmap_jpeg: None,
            base_quality: 90,
            gainmap_quality: 85,
            gainmap_scale: 4,
            target_display_peak: 10000.0,
            min_content_boost: 1.0,
            #[cfg(feature = "_test-helpers")]
            use_iso_metadata: true,
        }
    }

    /// Set the HDR input image (test only - requires JPEG codec).
    #[cfg(feature = "_test-helpers")]
    pub fn set_hdr_image(&mut self, image: RawImage) -> &mut Self {
        self.hdr_image = Some(image);
        self
    }

    /// Set the SDR input image (test only - requires JPEG codec).
    #[cfg(feature = "_test-helpers")]
    pub fn set_sdr_image(&mut self, image: RawImage) -> &mut Self {
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

    /// Set an existing gain map and metadata (test only).
    #[cfg(feature = "_test-helpers")]
    pub fn set_existing_gainmap(
        &mut self,
        gainmap: GainMap,
        metadata: GainMapMetadata,
    ) -> &mut Self {
        self.existing_gainmap = Some(gainmap);
        self.existing_metadata = Some(metadata);
        self
    }

    /// Clear any existing gain map (test only).
    #[cfg(feature = "_test-helpers")]
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

    /// Check if an existing gain map is set (test only).
    #[cfg(feature = "_test-helpers")]
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
        self.min_content_boost = boost.max(1.0);
        self
    }

    /// Enable or disable ISO 21496-1 metadata (test only).
    #[cfg(feature = "_test-helpers")]
    pub fn set_use_iso_metadata(&mut self, use_iso: bool) -> &mut Self {
        self.use_iso_metadata = use_iso;
        self
    }

    /// Encode to Ultra HDR JPEG (test only - requires JPEG codec).
    #[cfg(feature = "_test-helpers")]
    pub fn encode(&self) -> Result<Vec<u8>> {
        // Fast path: if we have raw gain map JPEG bytes, skip gain map processing
        if let (Some(ref gainmap_jpeg), Some(ref metadata)) =
            (&self.existing_gainmap_jpeg, &self.existing_metadata)
        {
            let (base_jpeg, gamut) = if let Some(ref compressed) = self.compressed_sdr {
                (compressed.clone(), ColorGamut::Bt709)
            } else if let Some(ref sdr_img) = self.sdr_image {
                (self.encode_base_jpeg(sdr_img)?, sdr_img.gamut)
            } else if let Some(ref hdr) = self.hdr_image {
                let sdr_pixels = tonemap_image_to_srgb8(hdr, ColorGamut::Bt709);
                let sdr = RawImage {
                    width: hdr.width,
                    height: hdr.height,
                    stride: hdr.width * 4,
                    data: sdr_pixels,
                    format: PixelFormat::Rgba8,
                    gamut: ColorGamut::Bt709,
                    transfer: ColorTransfer::Srgb,
                };
                (self.encode_base_jpeg(&sdr)?, sdr.gamut)
            } else {
                return Err(Error::EncodeError(
                    "Either HDR image, SDR image, or compressed SDR is required".into(),
                ));
            };

            return encode_ultrahdr(&base_jpeg, gainmap_jpeg, metadata, gamut);
        }

        // Validate inputs
        let hdr = self
            .hdr_image
            .as_ref()
            .ok_or_else(|| Error::EncodeError("HDR image is required".into()))?;

        // Generate or use provided SDR
        let sdr = if let Some(ref sdr_img) = self.sdr_image {
            sdr_img.clone()
        } else {
            let sdr_pixels = tonemap_image_to_srgb8(hdr, ColorGamut::Bt709);
            RawImage {
                width: hdr.width,
                height: hdr.height,
                stride: hdr.width * 4,
                data: sdr_pixels,
                format: PixelFormat::Rgba8,
                gamut: ColorGamut::Bt709,
                transfer: ColorTransfer::Srgb,
            }
        };

        // Use existing gain map if provided, otherwise compute a new one
        let (gainmap, metadata) =
            if let (Some(gm), Some(meta)) = (&self.existing_gainmap, &self.existing_metadata) {
                let expected_scale = self.gainmap_scale.max(1) as u32;
                let expected_width = sdr.width.div_ceil(expected_scale);
                let expected_height = sdr.height.div_ceil(expected_scale);

                let width_ok =
                    gm.width >= expected_width.saturating_sub(1) && gm.width <= expected_width + 1;
                let height_ok = gm.height >= expected_height.saturating_sub(1)
                    && gm.height <= expected_height + 1;

                if width_ok && height_ok {
                    (gm.clone(), meta.clone())
                } else {
                    self.compute_new_gainmap(hdr, &sdr)?
                }
            } else {
                self.compute_new_gainmap(hdr, &sdr)?
            };

        // Encode base JPEG
        let base_jpeg = if let Some(ref compressed) = self.compressed_sdr {
            compressed.clone()
        } else {
            self.encode_base_jpeg(&sdr)?
        };

        // Encode gain map JPEG
        let gainmap_jpeg = self.encode_gainmap_jpeg(&gainmap)?;

        encode_ultrahdr(&base_jpeg, &gainmap_jpeg, &metadata, sdr.gamut)
    }

    /// Encode to Ultra HDR JPEG from pre-set JPEGs (production API).
    pub fn encode_from_jpegs(&self) -> Result<Vec<u8>> {
        let base_jpeg = self
            .compressed_sdr
            .as_ref()
            .ok_or_else(|| Error::EncodeError("Base JPEG not set".into()))?;

        let gainmap_jpeg = self
            .existing_gainmap_jpeg
            .as_ref()
            .ok_or_else(|| Error::EncodeError("Gainmap JPEG not set".into()))?;

        let metadata = self
            .existing_metadata
            .as_ref()
            .ok_or_else(|| Error::EncodeError("Metadata not set".into()))?;

        encode_ultrahdr(base_jpeg, gainmap_jpeg, metadata, ColorGamut::Bt709)
    }

    /// Compute a new gain map (test only).
    #[cfg(feature = "_test-helpers")]
    fn compute_new_gainmap(
        &self,
        hdr: &RawImage,
        sdr: &RawImage,
    ) -> Result<(GainMap, GainMapMetadata)> {
        let config = GainMapConfig {
            scale_factor: self.gainmap_scale,
            gamma: 1.0,
            multi_channel: false,
            min_content_boost: self.min_content_boost,
            max_content_boost: self.target_display_peak / 203.0,
            offset_sdr: 1.0 / 64.0,
            offset_hdr: 1.0 / 64.0,
            hdr_capacity_min: 1.0,
            hdr_capacity_max: self.target_display_peak / 203.0,
        };

        compute_gainmap(hdr, sdr, &config, Unstoppable)
    }

    /// Encode base SDR image to JPEG (test only).
    #[cfg(feature = "_test-helpers")]
    fn encode_base_jpeg(&self, sdr: &RawImage) -> Result<Vec<u8>> {
        use zenjpeg::encoder::{ChromaSubsampling, EncoderConfig, PixelLayout, Unstoppable};

        let (pixel_layout, data): (PixelLayout, std::borrow::Cow<[u8]>) = match sdr.format {
            PixelFormat::Rgba8 => {
                let rgb: Vec<u8> = sdr
                    .data
                    .chunks(4)
                    .flat_map(|rgba| [rgba[0], rgba[1], rgba[2]])
                    .collect();
                (PixelLayout::Rgb8Srgb, std::borrow::Cow::Owned(rgb))
            }
            PixelFormat::Rgb8 => (
                PixelLayout::Rgb8Srgb,
                std::borrow::Cow::Borrowed(&sdr.data[..]),
            ),
            _ => {
                return Err(Error::EncodeError(format!(
                    "Unsupported SDR pixel format: {:?}",
                    sdr.format
                )))
            }
        };

        let config = EncoderConfig::ycbcr(self.base_quality as f32, ChromaSubsampling::Quarter);
        let mut enc = config
            .encode_from_bytes(sdr.width, sdr.height, pixel_layout)
            .map_err(|e| Error::JpegEncode(e.to_string()))?;
        enc.push_packed(&data, Unstoppable)
            .map_err(|e| Error::JpegEncode(e.to_string()))?;
        enc.finish().map_err(|e| Error::JpegEncode(e.to_string()))
    }

    /// Encode gain map to JPEG (test only).
    #[cfg(feature = "_test-helpers")]
    fn encode_gainmap_jpeg(&self, gainmap: &GainMap) -> Result<Vec<u8>> {
        use zenjpeg::encoder::{EncoderConfig, PixelLayout, Unstoppable};

        let config = EncoderConfig::grayscale(self.gainmap_quality as f32);
        let mut enc = config
            .encode_from_bytes(gainmap.width, gainmap.height, PixelLayout::Gray8Srgb)
            .map_err(|e| Error::JpegEncode(e.to_string()))?;
        enc.push_packed(&gainmap.data, Unstoppable)
            .map_err(|e| Error::JpegEncode(e.to_string()))?;
        enc.finish().map_err(|e| Error::JpegEncode(e.to_string()))
    }
}

#[cfg(all(test, feature = "_test-helpers"))]
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
        let metadata = GainMapMetadata::new();
        encoder.set_existing_gainmap(gainmap, metadata);
        assert!(encoder.has_existing_gainmap());

        encoder.clear_existing_gainmap();
        assert!(!encoder.has_existing_gainmap());
    }
}
