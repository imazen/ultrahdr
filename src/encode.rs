//! Ultra HDR encoder.

use crate::color::tonemap::tonemap_image_to_srgb8;
use crate::gainmap::compute::{compute_gainmap, GainMapConfig};
use crate::jpeg::{
    create_icc_markers, get_icc_profile_for_gamut, insert_segment_after_soi, JpegSegment,
};
use crate::metadata::{
    mpf::create_mpf_header,
    xmp::{create_xmp_app1_marker, generate_xmp},
};
use crate::types::{
    ColorGamut, ColorTransfer, Error, GainMap, GainMapMetadata, PixelFormat, RawImage, Result,
};

/// Ultra HDR encoder.
///
/// Supports multiple input modes:
/// - HDR only: Automatically generates SDR via tone mapping
/// - HDR + SDR: Uses provided SDR image
/// - HDR + compressed SDR: Uses pre-encoded JPEG for base image
/// - HDR + SDR + existing gain map: Reuses pre-computed gain map (for lossless round-trips)
pub struct Encoder {
    hdr_image: Option<RawImage>,
    sdr_image: Option<RawImage>,
    compressed_sdr: Option<Vec<u8>>,
    existing_gainmap: Option<GainMap>,
    existing_metadata: Option<GainMapMetadata>,
    base_quality: u8,
    gainmap_quality: u8,
    gainmap_scale: u8,
    target_display_peak: f32,
    min_content_boost: f32,
    use_iso_metadata: bool,
}

impl Default for Encoder {
    fn default() -> Self {
        Self::new()
    }
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
            base_quality: 90,
            gainmap_quality: 85,
            gainmap_scale: 4,
            target_display_peak: 10000.0,
            min_content_boost: 1.0,
            use_iso_metadata: true,
        }
    }

    /// Set the HDR input image (required).
    pub fn set_hdr_image(&mut self, image: RawImage) -> &mut Self {
        self.hdr_image = Some(image);
        self
    }

    /// Set the SDR input image (optional).
    ///
    /// If not provided, SDR will be generated via tone mapping.
    pub fn set_sdr_image(&mut self, image: RawImage) -> &mut Self {
        self.sdr_image = Some(image);
        self
    }

    /// Set a pre-compressed SDR JPEG (optional).
    ///
    /// If provided, this JPEG will be used as the base image instead of
    /// compressing the SDR image.
    pub fn set_compressed_sdr(&mut self, jpeg: Vec<u8>) -> &mut Self {
        self.compressed_sdr = Some(jpeg);
        self
    }

    /// Set an existing gain map and metadata (optional).
    ///
    /// If provided, this gain map will be used instead of computing a new one.
    /// This enables lossless UltraHDR round-trips (decode → process → encode)
    /// when the SDR dimensions haven't changed.
    ///
    /// Note: The caller is responsible for ensuring the gain map is appropriate
    /// for the SDR image. If the SDR dimensions have changed (e.g., after resize),
    /// the gain map should be invalidated and recomputed.
    pub fn set_existing_gainmap(
        &mut self,
        gainmap: GainMap,
        metadata: GainMapMetadata,
    ) -> &mut Self {
        self.existing_gainmap = Some(gainmap);
        self.existing_metadata = Some(metadata);
        self
    }

    /// Clear any existing gain map, forcing recomputation.
    pub fn clear_existing_gainmap(&mut self) -> &mut Self {
        self.existing_gainmap = None;
        self.existing_metadata = None;
        self
    }

    /// Check if an existing gain map is set.
    pub fn has_existing_gainmap(&self) -> bool {
        self.existing_gainmap.is_some() && self.existing_metadata.is_some()
    }

    /// Set JPEG quality for base and gain map images.
    ///
    /// Quality ranges from 1-100. Default: base=90, gainmap=85.
    pub fn set_quality(&mut self, base: u8, gainmap: u8) -> &mut Self {
        self.base_quality = base.clamp(1, 100);
        self.gainmap_quality = gainmap.clamp(1, 100);
        self
    }

    /// Set gain map downscale factor.
    ///
    /// The gain map is typically smaller than the base image.
    /// Factor of 4 means gain map is 1/4 the width and height.
    /// Default: 4. Range: 1-128.
    pub fn set_gainmap_scale(&mut self, scale: u8) -> &mut Self {
        self.gainmap_scale = scale.clamp(1, 128);
        self
    }

    /// Set target display peak brightness in nits.
    ///
    /// Default: 10000.0 (HDR10 max).
    pub fn set_target_display_peak(&mut self, nits: f32) -> &mut Self {
        self.target_display_peak = nits.max(100.0);
        self
    }

    /// Set minimum content boost.
    ///
    /// Default: 1.0 (no boost at minimum).
    pub fn set_min_content_boost(&mut self, boost: f32) -> &mut Self {
        self.min_content_boost = boost.max(1.0);
        self
    }

    /// Enable or disable ISO 21496-1 metadata.
    ///
    /// Default: true (include both XMP and ISO metadata).
    pub fn set_use_iso_metadata(&mut self, use_iso: bool) -> &mut Self {
        self.use_iso_metadata = use_iso;
        self
    }

    /// Encode to Ultra HDR JPEG.
    pub fn encode(&self) -> Result<Vec<u8>> {
        // Validate inputs
        let hdr = self
            .hdr_image
            .as_ref()
            .ok_or_else(|| Error::EncodeError("HDR image is required".into()))?;

        // Generate or use provided SDR
        let sdr = if let Some(ref sdr_img) = self.sdr_image {
            sdr_img.clone()
        } else {
            // Generate SDR via tone mapping
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
                // Validate that the existing gain map dimensions are appropriate
                // The gain map should be roughly (sdr_width / scale) x (sdr_height / scale)
                let expected_scale = self.gainmap_scale.max(1) as u32;
                let expected_width = (sdr.width + expected_scale - 1) / expected_scale;
                let expected_height = (sdr.height + expected_scale - 1) / expected_scale;

                // Allow some tolerance for rounding differences
                let width_ok =
                    gm.width >= expected_width.saturating_sub(1) && gm.width <= expected_width + 1;
                let height_ok = gm.height >= expected_height.saturating_sub(1)
                    && gm.height <= expected_height + 1;

                if width_ok && height_ok {
                    // Reuse existing gain map
                    (gm.clone(), meta.clone())
                } else {
                    // Dimensions don't match - recompute
                    self.compute_new_gainmap(hdr, &sdr)?
                }
            } else {
                // No existing gain map - compute new one
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

        // Create Ultra HDR structure
        self.create_ultrahdr_jpeg(&base_jpeg, &gainmap_jpeg, &metadata, sdr.gamut)
    }

    /// Compute a new gain map from HDR and SDR images.
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
            max_content_boost: self.target_display_peak / 203.0, // SDR white = 203 nits
            offset_sdr: 1.0 / 64.0,
            offset_hdr: 1.0 / 64.0,
            hdr_capacity_min: 1.0,
            hdr_capacity_max: self.target_display_peak / 203.0,
        };

        compute_gainmap(hdr, sdr, &config)
    }

    /// Encode base SDR image to JPEG.
    fn encode_base_jpeg(&self, sdr: &RawImage) -> Result<Vec<u8>> {
        use jpegli::encoder::{ChromaSubsampling, EncoderConfig, PixelLayout, Unstoppable};

        let (pixel_layout, data): (PixelLayout, std::borrow::Cow<[u8]>) = match sdr.format {
            PixelFormat::Rgba8 => {
                // Convert RGBA to RGB for JPEG
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

    /// Encode gain map to JPEG.
    fn encode_gainmap_jpeg(&self, gainmap: &crate::GainMap) -> Result<Vec<u8>> {
        use jpegli::encoder::{EncoderConfig, PixelLayout, Unstoppable};

        let config = EncoderConfig::grayscale(self.gainmap_quality as f32);
        let mut enc = config
            .encode_from_bytes(gainmap.width, gainmap.height, PixelLayout::Gray8Srgb)
            .map_err(|e| Error::JpegEncode(e.to_string()))?;
        enc.push_packed(&gainmap.data, Unstoppable)
            .map_err(|e| Error::JpegEncode(e.to_string()))?;
        enc.finish().map_err(|e| Error::JpegEncode(e.to_string()))
    }

    /// Create final Ultra HDR JPEG structure.
    fn create_ultrahdr_jpeg(
        &self,
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
        // MPF header will be inserted, so we need to account for it
        let mpf_estimate = create_mpf_header(0, 0).len();
        let primary_with_mpf_len = primary.len() + mpf_estimate;

        // Create MPF header
        let mpf_header = create_mpf_header(primary_with_mpf_len, gainmap_jpeg.len());

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

        // Initially no existing gain map
        assert!(!encoder.has_existing_gainmap());

        // Set existing gain map
        let gainmap = GainMap::new(100, 100);
        let metadata = GainMapMetadata::new();
        encoder.set_existing_gainmap(gainmap, metadata);
        assert!(encoder.has_existing_gainmap());

        // Clear it
        encoder.clear_existing_gainmap();
        assert!(!encoder.has_existing_gainmap());
    }
}
