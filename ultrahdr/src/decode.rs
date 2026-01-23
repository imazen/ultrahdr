//! Ultra HDR decoder.

use ultrahdr_core::gainmap::apply::{apply_gainmap, HdrOutputFormat};
use ultrahdr_core::metadata::{
    mpf::{find_jpeg_boundaries, parse_mpf},
    xmp::parse_xmp,
};
use ultrahdr_core::{
    ColorGamut, ColorTransfer, Error, GainMap, GainMapMetadata, PixelFormat, RawImage, Result,
    Unstoppable,
};

use crate::jpeg::{extract_icc_profile, find_xmp_data};

/// Ultra HDR decoder.
///
/// Decodes Ultra HDR JPEGs, extracting the SDR base image, gain map,
/// and metadata. Can reconstruct HDR content at various display
/// brightness levels.
pub struct Decoder {
    data: Vec<u8>,
    metadata: Option<GainMapMetadata>,
    primary_jpeg: Option<(usize, usize)>,
    gainmap_jpeg: Option<(usize, usize)>,
    is_ultrahdr: bool,
}

impl Decoder {
    /// Create a new decoder from JPEG data.
    pub fn new(data: &[u8]) -> Result<Self> {
        let mut decoder = Self {
            data: data.to_vec(),
            metadata: None,
            primary_jpeg: None,
            gainmap_jpeg: None,
            is_ultrahdr: false,
        };

        decoder.parse()?;
        Ok(decoder)
    }

    /// Check if this is a valid Ultra HDR image.
    pub fn is_ultrahdr(&self) -> bool {
        self.is_ultrahdr
    }

    /// Get the gain map metadata.
    pub fn metadata(&self) -> Option<&GainMapMetadata> {
        self.metadata.as_ref()
    }

    /// Get the raw gain map JPEG data.
    pub fn gainmap_jpeg(&self) -> Option<&[u8]> {
        self.gainmap_jpeg.map(|(start, end)| &self.data[start..end])
    }

    /// Decode the SDR base image.
    pub fn decode_sdr(&self) -> Result<RawImage> {
        let (start, end) = self
            .primary_jpeg
            .ok_or_else(|| Error::DecodeError("No primary image found".into()))?;

        let primary_data = &self.data[start..end];
        decode_jpeg_to_rgb(primary_data)
    }

    /// Decode the gain map.
    pub fn decode_gainmap(&self) -> Result<GainMap> {
        let (start, end) = self
            .gainmap_jpeg
            .ok_or_else(|| Error::DecodeError("No gain map found".into()))?;

        let gainmap_data = &self.data[start..end];
        let decoded = decode_jpeg_to_grayscale(gainmap_data)?;

        Ok(GainMap {
            width: decoded.width,
            height: decoded.height,
            channels: 1,
            data: decoded.data,
        })
    }

    /// Decode to HDR at the specified display boost level.
    ///
    /// `display_boost` is the ratio of display peak brightness to SDR white.
    /// For example:
    /// - 1.0 = SDR display (no HDR enhancement)
    /// - 4.0 = Display capable of 4x SDR brightness
    /// - ~49.0 = Full HDR10 (10000 nits / 203 SDR nits)
    pub fn decode_hdr(&self, display_boost: f32) -> Result<RawImage> {
        self.decode_hdr_with_format(display_boost, HdrOutputFormat::LinearFloat)
    }

    /// Decode to HDR with a specific output format.
    pub fn decode_hdr_with_format(
        &self,
        display_boost: f32,
        format: HdrOutputFormat,
    ) -> Result<RawImage> {
        if !self.is_ultrahdr {
            return Err(Error::DecodeError("Not an Ultra HDR image".into()));
        }

        let metadata = self
            .metadata
            .as_ref()
            .ok_or_else(|| Error::DecodeError("No gain map metadata".into()))?;

        let sdr = self.decode_sdr()?;
        let gainmap = self.decode_gainmap()?;

        apply_gainmap(&sdr, &gainmap, metadata, display_boost, format, Unstoppable)
    }

    /// Parse the Ultra HDR structure.
    fn parse(&mut self) -> Result<()> {
        // Check for valid JPEG
        if self.data.len() < 4 || self.data[0] != 0xFF || self.data[1] != 0xD8 {
            return Err(Error::DecodeError("Not a valid JPEG".into()));
        }

        // Try to find XMP metadata with hdrgm namespace
        if let Some(xmp) = find_xmp_data(&self.data) {
            if xmp.contains("hdrgm:") || xmp.contains("http://ns.adobe.com/hdr-gain-map/") {
                if let Ok((metadata, _gainmap_len)) = parse_xmp(&xmp) {
                    self.metadata = Some(metadata);
                    self.is_ultrahdr = true;
                }
            }
        }

        // Try to parse MPF to find gain map
        if let Ok(images) = parse_mpf(&self.data) {
            if images.len() >= 2 {
                // First image is primary, second is gain map
                self.primary_jpeg = Some(images[0]);
                self.gainmap_jpeg = Some(images[1]);
                self.is_ultrahdr = true;
            }
        }

        // Fallback: look for multiple JPEGs in the file
        if self.gainmap_jpeg.is_none() {
            let boundaries = find_jpeg_boundaries(&self.data);
            if boundaries.len() >= 2 {
                self.primary_jpeg = Some(boundaries[0]);
                self.gainmap_jpeg = Some(boundaries[1]);
            }
        }

        // Set primary to full data if not found via MPF
        if self.primary_jpeg.is_none() {
            self.primary_jpeg = Some((0, self.data.len()));
        }

        Ok(())
    }

    /// Get the ICC profile from the primary image if present.
    pub fn icc_profile(&self) -> Option<Vec<u8>> {
        extract_icc_profile(&self.data)
    }

    /// Get information about the decoded image dimensions.
    pub fn dimensions(&self) -> Result<(u32, u32)> {
        let sdr = self.decode_sdr()?;
        Ok((sdr.width, sdr.height))
    }
}

/// Decode JPEG to RGB.
fn decode_jpeg_to_rgb(jpeg_data: &[u8]) -> Result<RawImage> {
    use jpegli::decoder::{Decoder as JpegDecoder, PixelFormat as JpegPixelFormat};
    let decoded = JpegDecoder::new()
        .output_format(JpegPixelFormat::Rgb)
        .decode(jpeg_data)
        .map_err(|e| Error::DecodeError(format!("JPEG decode failed: {}", e)))?;

    let width = decoded.width;
    let height = decoded.height;
    let pixels = &decoded.data;
    let bpp = decoded.bytes_per_pixel();

    // Convert to RGBA if needed
    let data = if bpp == 3 {
        // RGB -> RGBA
        let mut rgba = Vec::with_capacity((width * height * 4) as usize);
        for chunk in pixels.chunks(3) {
            rgba.push(chunk[0]);
            rgba.push(chunk[1]);
            rgba.push(chunk[2]);
            rgba.push(255);
        }
        rgba
    } else if bpp == 4 {
        pixels.to_vec()
    } else if bpp == 1 {
        // Grayscale -> RGBA
        let mut rgba = Vec::with_capacity((width * height * 4) as usize);
        for &g in pixels {
            rgba.push(g);
            rgba.push(g);
            rgba.push(g);
            rgba.push(255);
        }
        rgba
    } else {
        return Err(Error::DecodeError(format!(
            "Unsupported bytes per pixel: {}",
            bpp
        )));
    };

    Ok(RawImage {
        width,
        height,
        stride: width * 4,
        data,
        format: PixelFormat::Rgba8,
        gamut: ColorGamut::Bt709, // Assume sRGB for SDR
        transfer: ColorTransfer::Srgb,
    })
}

/// Decode JPEG to grayscale.
fn decode_jpeg_to_grayscale(jpeg_data: &[u8]) -> Result<RawImage> {
    use jpegli::decoder::{Decoder as JpegDecoder, PixelFormat as JpegPixelFormat};
    let decoded = JpegDecoder::new()
        .output_format(JpegPixelFormat::Gray)
        .decode(jpeg_data)
        .map_err(|e| Error::DecodeError(format!("JPEG decode failed: {}", e)))?;

    let width = decoded.width;
    let height = decoded.height;
    let pixels = &decoded.data;
    let bpp = decoded.bytes_per_pixel();

    // Convert to grayscale if needed
    let data = if bpp == 1 {
        pixels.to_vec()
    } else if bpp == 3 {
        // RGB -> Grayscale (using luminance)
        pixels
            .chunks(3)
            .map(|rgb| {
                let r = rgb[0] as f32;
                let g = rgb[1] as f32;
                let b = rgb[2] as f32;
                // BT.709 luminance
                (0.2126_f32 * r + 0.7152 * g + 0.0722 * b).clamp(0.0, 255.0) as u8
            })
            .collect()
    } else {
        return Err(Error::DecodeError(format!(
            "Unsupported bytes per pixel for grayscale: {}",
            bpp
        )));
    };

    Ok(RawImage {
        width,
        height,
        stride: width,
        data,
        format: PixelFormat::Gray8,
        gamut: ColorGamut::Bt709,
        transfer: ColorTransfer::Srgb,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_decoder_invalid_data() {
        let result = Decoder::new(&[0, 1, 2, 3]);
        assert!(result.is_err());
    }

    #[test]
    fn test_decoder_minimal_jpeg() {
        // Minimal JPEG (just SOI + EOI)
        let data = vec![0xFF, 0xD8, 0xFF, 0xD9];
        let decoder = Decoder::new(&data);
        // This will fail to parse as a valid image but won't error on construction
        assert!(decoder.is_ok());
        assert!(!decoder.unwrap().is_ultrahdr());
    }
}
