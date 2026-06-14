//! Ultra HDR decoder.

use ultrahdr_core::gainmap::apply::{HdrOutputFormat, apply_gainmap};
use ultrahdr_core::{
    ColorPrimaries, Error, GainMap, GainMapMetadata, PixelBuffer, PixelFormat, Result,
    TransferFunction, Unstoppable, pixel_buffer_from_vec,
};
use zenjpeg::container::marker::find_jpeg_boundaries;
use zenjpeg::container::xmp::parse_xmp;

use crate::container::{self, AppSegment};

/// Ultra HDR decoder.
///
/// Decodes Ultra HDR JPEGs, extracting the SDR base image, gain map,
/// and metadata. Can reconstruct HDR content at various display
/// brightness levels.
///
/// The decoder borrows the input data to avoid an unconditional copy.
pub struct Decoder<'a> {
    data: &'a [u8],
    metadata: Option<GainMapMetadata>,
    primary_jpeg: Option<(usize, usize)>,
    gainmap_jpeg: Option<(usize, usize)>,
    is_ultrahdr: bool,
}

impl<'a> Decoder<'a> {
    /// Create a new decoder from JPEG data.
    ///
    /// The decoder borrows the data — no copy is made.
    pub fn new(data: &'a [u8]) -> Result<Self> {
        let mut decoder = Self {
            data,
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

    /// Get the raw primary (SDR base) JPEG data.
    ///
    /// Use this to decode the base image with your own JPEG codec.
    pub fn primary_jpeg(&self) -> Option<&[u8]> {
        self.primary_jpeg
            .and_then(|(start, end)| self.data.get(start..end))
    }

    /// Get the raw gain map JPEG data.
    ///
    /// Use this to decode the gain map with your own JPEG codec.
    pub fn gainmap_jpeg(&self) -> Option<&[u8]> {
        self.gainmap_jpeg
            .and_then(|(start, end)| self.data.get(start..end))
    }

    /// Decode the SDR base image using the bundled zenjpeg codec.
    ///
    /// Returns a linear/sRGB `Rgba8` [`PixelBuffer`] reconstructed from the
    /// primary JPEG codestream. If you want to decode with a different
    /// JPEG codec, call [`Decoder::primary_jpeg`] for the raw bytes.
    pub fn decode_sdr(&self) -> Result<PixelBuffer> {
        let primary_data = self
            .primary_jpeg()
            .ok_or_else(|| Error::DecodeError("No primary image found".into()))?;
        decode_jpeg_to_rgb(primary_data)
    }

    /// Decode the gain map using the bundled zenjpeg codec.
    ///
    /// Returns a [`GainMap`] reconstructed from the gain-map JPEG
    /// codestream: single-channel for luma-only maps, 3-channel
    /// (interleaved RGB) for per-channel maps (Adobe exports, iOS 18 —
    /// issue #27).
    ///
    /// The channel count is driven by the ISO 21496-1 **metadata**
    /// (`is_single_channel`), not by pixel inspection: a single-channel
    /// map JPEG-coded as YCbCr picks up ±1 chroma noise from subsampling,
    /// and treating that noise as per-channel gain would change pixels.
    /// Only when no metadata is available does a full achromatic scan
    /// decide. For a different JPEG codec, see [`Decoder::gainmap_jpeg`].
    pub fn decode_gainmap(&self) -> Result<GainMap> {
        let gainmap_data = self
            .gainmap_jpeg()
            .ok_or_else(|| Error::DecodeError("No gain map found".into()))?;

        let single_channel = self.metadata.as_ref().map(|m| m.is_single_channel());

        if single_channel != Some(false) {
            // Single-channel per metadata (or unknown): the historical Gray
            // decode is the exact luma plane — keep it as the fast path.
            // Some color encodings can't produce Gray output ("unsupported
            // color conversion", #27); those fall through to the RGB path.
            if let Ok((width, height, data)) = decode_jpeg_to_grayscale_bytes(gainmap_data) {
                return Ok(GainMap {
                    width,
                    height,
                    channels: 1,
                    data,
                });
            }
        }

        let (width, height, rgb) = decode_jpeg_to_rgb_bytes(gainmap_data)?;
        let collapse = match single_channel {
            // Metadata says luma-only: collapse regardless of decode noise.
            Some(true) => true,
            // Metadata says per-channel: keep all three.
            Some(false) => false,
            // No metadata: collapse only when provably achromatic (the
            // zenpixels load-bearing predicate — full scan, no sampling).
            None => rgb
                .chunks_exact(3)
                .all(|px| px[0] == px[1] && px[1] == px[2]),
        };
        let (data, channels) = if collapse {
            // BT.709 luma — the same weighting the Gray decode applies.
            (
                rgb.chunks_exact(3)
                    .map(|px| {
                        (0.2126_f32 * f32::from(px[0])
                            + 0.7152 * f32::from(px[1])
                            + 0.0722 * f32::from(px[2]))
                        .clamp(0.0, 255.0) as u8
                    })
                    .collect(),
                1,
            )
        } else {
            (rgb, 3)
        };

        Ok(GainMap {
            width,
            height,
            channels,
            data,
        })
    }

    /// Decode to HDR at the specified display boost level.
    ///
    /// `display_boost` is the ratio of display peak brightness to SDR white.
    /// For example:
    /// - 1.0 = SDR display (no HDR enhancement)
    /// - 4.0 = Display capable of 4x SDR brightness
    /// - ~49.0 = Full HDR10 (10000 nits / 203 SDR nits)
    pub fn decode_hdr(&self, display_boost: f32) -> Result<PixelBuffer> {
        self.decode_hdr_with_format(display_boost, HdrOutputFormat::LinearFloat)
    }

    /// Decode to HDR with a specific output format.
    pub fn decode_hdr_with_format(
        &self,
        display_boost: f32,
        format: HdrOutputFormat,
    ) -> Result<PixelBuffer> {
        if !self.is_ultrahdr {
            return Err(Error::DecodeError("Not an Ultra HDR image".into()));
        }

        if !display_boost.is_finite() || display_boost < 1.0 {
            return Err(Error::DecodeError(format!(
                "display_boost must be >= 1.0, got {}",
                display_boost
            )));
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
    ///
    /// Uses `container::scan_segments` for efficient marker-to-marker scanning
    /// instead of byte-by-byte search.
    fn parse(&mut self) -> Result<()> {
        // Check for valid JPEG
        if self.data.len() < 4 || self.data[0] != 0xFF || self.data[1] != 0xD8 {
            return Err(Error::DecodeError("Not a valid JPEG".into()));
        }

        // Scan APP segments efficiently (walks marker-to-marker, not byte-by-byte)
        let segments = container::scan_segments(self.data);

        // Find XMP metadata with hdrgm namespace in primary
        if let Some(xmp_str) = find_xmp_in_segments(&segments)
            && (xmp_str.contains("hdrgm:") || xmp_str.contains("http://ns.adobe.com/hdr-gain-map/"))
        {
            self.is_ultrahdr = true;
            // Try parsing numeric metadata from primary XMP (legacy format)
            if let Ok((metadata, _gainmap_len)) = parse_xmp(&xmp_str)
                && (metadata.alternate_hdr_headroom != 0.0 || metadata.channels[0].max != 0.0)
            {
                self.metadata = Some(metadata);
            }
        }

        // Try to parse MPF to find the gain map. MPF is one of several
        // discovery routes — a malformed or unsupported MPF index (e.g.
        // zenjpeg#148: valid big-endian `MM` indexes misread as "zero
        // images") must degrade to the JPEG-boundary fallback below, never
        // abort detection that the XMP scan above already established.
        let mpf_entries = container::parse_mpf(self.data).unwrap_or_default();
        if mpf_entries.len() >= 2 {
            // Primary image — locate via JPEG marker scan, NOT MPF's declared
            // size. Some encoders (notably Pixel HDR+ 1.0.*) write a too-short
            // primary_size that cuts off the last MCU row's entropy-coded data.
            // libultrahdr's own decoder uses a JpegScanner (see jpegr.cpp
            // extractPrimaryImageAndGainMap) for exactly this reason.
            if let Some(bounds) = container::primary_bounds(self.data) {
                self.primary_jpeg = Some((bounds.start, bounds.end));
            } else {
                // Fallback to MPF's size if marker scan somehow fails.
                self.primary_jpeg = Some((0, mpf_entries[0].size));
            }

            // First secondary image = gain map.
            let gm_entry = &mpf_entries[1];
            let gm_start = gm_entry.offset;
            // `checked_add` defends against the 32-bit case where
            // `offset + size` could wrap past `usize::MAX` and pass the
            // `<= self.data.len()` bound check before slicing panics.
            let gm_end = match gm_start.checked_add(gm_entry.size) {
                Some(end) => end,
                None => {
                    return Err(Error::DecodeError("MPF entry offset+size overflows".into()));
                }
            };
            if gm_end <= self.data.len() {
                self.gainmap_jpeg = Some((gm_start, gm_end));
                self.is_ultrahdr = true;

                // Check gain map JPEG for metadata XMP (modern format:
                // libultrahdr puts metadata in the secondary JPEG's XMP).
                if self.metadata.is_none() {
                    let gm = &self.data[gm_start..gm_end];
                    let gm_segments = container::scan_segments(gm);
                    if let Some(gm_xmp) = find_xmp_in_segments(&gm_segments)
                        && gm_xmp.contains("hdrgm:")
                        && let Ok((gm_metadata, _)) = parse_xmp(&gm_xmp)
                    {
                        self.metadata = Some(gm_metadata);
                    }
                }
            }
        }

        // Fallback: look for multiple JPEGs in the file
        if self.gainmap_jpeg.is_none() {
            let boundaries = find_jpeg_boundaries(self.data);
            if boundaries.len() >= 2 {
                self.primary_jpeg = Some((boundaries[0].start, boundaries[0].end));
                self.gainmap_jpeg = Some((boundaries[1].start, boundaries[1].end));

                // Also try to find metadata in the gain map JPEG
                if self.metadata.is_none()
                    && let Some(gm_data) = self.data.get(boundaries[1].clone())
                {
                    let gm_segments = container::scan_segments(gm_data);
                    if let Some(gm_xmp) = find_xmp_in_segments(&gm_segments)
                        && gm_xmp.contains("hdrgm:")
                        && let Ok((gm_metadata, _)) = parse_xmp(&gm_xmp)
                    {
                        self.metadata = Some(gm_metadata);
                    }
                }
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
        crate::jpeg::extract_icc_profile(self.data)
    }

    /// Get the image dimensions by decoding the primary JPEG header.
    pub fn dimensions(&self) -> Result<(u32, u32)> {
        let sdr = self.decode_sdr()?;
        Ok((sdr.width(), sdr.height()))
    }
}

/// Find XMP data in pre-scanned APP segments.
///
/// This is O(segments) instead of O(bytes), since we use the already-scanned
/// segment list from `container::scan_segments`.
fn find_xmp_in_segments(segments: &[AppSegment]) -> Option<String> {
    let xmp_ns = b"http://ns.adobe.com/xap/1.0/\0";

    for seg in segments {
        if seg.is_xmp() && seg.data.len() > xmp_ns.len() {
            let xmp_bytes = &seg.data[xmp_ns.len()..];
            if let Ok(xmp) = std::str::from_utf8(xmp_bytes) {
                return Some(xmp.to_string());
            }
        }
    }

    None
}

/// Decode JPEG to RGB.
fn decode_jpeg_to_rgb(jpeg_data: &[u8]) -> Result<PixelBuffer> {
    use zenjpeg::decoder::{Decoder as JpegDecoder, PixelFormat as JpegPixelFormat};
    let decoded = JpegDecoder::new()
        .output_format(JpegPixelFormat::Rgb)
        .decode(jpeg_data, Unstoppable)
        .map_err(|e| Error::DecodeError(format!("JPEG decode failed: {}", e)))?;

    let width = decoded.width();
    let height = decoded.height();
    let pixels = decoded
        .pixels_u8()
        .ok_or_else(|| Error::DecodeError("No pixel data in decoded JPEG".into()))?;
    let bpp = decoded.bytes_per_pixel();

    // Convert to RGBA if needed
    let data = if bpp == 3 {
        // RGB -> RGBA
        let mut rgba = Vec::with_capacity(width as usize * height as usize * 4);
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
        let mut rgba = Vec::with_capacity(width as usize * height as usize * 4);
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

    pixel_buffer_from_vec(
        data,
        width,
        height,
        PixelFormat::Rgba8,
        ColorPrimaries::Bt709, // assume sRGB for SDR
        TransferFunction::Srgb,
    )
}

/// Decode a grayscale JPEG and return (width, height, packed bytes).
///
/// Used by [`Decoder::decode_gainmap`] to lift the decoded codestream into a
/// [`GainMap`] without wrapping the byte buffer in a [`PixelBuffer`] (gain
/// map bytes are log2-quantized gain, not color samples).
/// Decode a gain-map JPEG to its exact luma plane (Gray output).
///
/// The fast path for single-channel maps. Can fail for some color
/// encodings ("unsupported color conversion", #27) — callers fall back
/// to [`decode_jpeg_to_rgb_bytes`].
fn decode_jpeg_to_grayscale_bytes(jpeg_data: &[u8]) -> Result<(u32, u32, Vec<u8>)> {
    use zenjpeg::decoder::{Decoder as JpegDecoder, PixelFormat as JpegPixelFormat};
    let decoded = JpegDecoder::new()
        .output_format(JpegPixelFormat::Gray)
        .decode(jpeg_data, Unstoppable)
        .map_err(|e| Error::DecodeError(format!("JPEG decode failed: {}", e)))?;

    let width = decoded.width();
    let height = decoded.height();
    let pixels = decoded
        .pixels_u8()
        .ok_or_else(|| Error::DecodeError("No pixel data in decoded JPEG".into()))?;
    match decoded.bytes_per_pixel() {
        1 => Ok((width, height, pixels.to_vec())),
        bpp => Err(Error::DecodeError(format!(
            "Unsupported bytes per pixel for grayscale gain-map decode: {bpp}"
        ))),
    }
}

/// Decode a gain-map JPEG to tight interleaved RGB8 bytes.
///
/// Used for per-channel (multi-channel) maps and as the fallback when
/// Gray output is unavailable: RGB output is universally supported
/// (grayscale codestreams expand to identical channels), and per-channel
/// maps must not be flattened to luma (#27).
fn decode_jpeg_to_rgb_bytes(jpeg_data: &[u8]) -> Result<(u32, u32, Vec<u8>)> {
    use zenjpeg::decoder::{Decoder as JpegDecoder, PixelFormat as JpegPixelFormat};
    let decoded = JpegDecoder::new()
        .output_format(JpegPixelFormat::Rgb)
        .decode(jpeg_data, Unstoppable)
        .map_err(|e| Error::DecodeError(format!("JPEG decode failed: {}", e)))?;

    let width = decoded.width();
    let height = decoded.height();
    let pixels = decoded
        .pixels_u8()
        .ok_or_else(|| Error::DecodeError("No pixel data in decoded JPEG".into()))?;
    match decoded.bytes_per_pixel() {
        3 => Ok((width, height, pixels.to_vec())),
        bpp => Err(Error::DecodeError(format!(
            "Unsupported bytes per pixel for RGB gain-map decode: {bpp}"
        ))),
    }
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
        assert!(decoder.is_ok());
        assert!(!decoder.unwrap().is_ultrahdr());
    }

    #[test]
    fn test_decoder_not_ultrahdr() {
        // JPEG with APP0 but no UltraHDR content
        let data = vec![
            0xFF, 0xD8, // SOI
            0xFF, 0xE0, 0x00, 0x07, // APP0 length 7
            b'J', b'F', b'I', b'F', 0x00, // JFIF
            0xFF, 0xD9, // EOI
        ];
        let decoder = Decoder::new(&data).unwrap();
        assert!(!decoder.is_ultrahdr());
        assert!(decoder.metadata().is_none());
        assert!(decoder.gainmap_jpeg().is_none());
        // Primary should be the whole file
        assert!(decoder.primary_jpeg().is_some());
    }

    #[test]
    fn test_decoder_borrows_data() {
        let data = vec![0xFF, 0xD8, 0xFF, 0xD9];
        let decoder = Decoder::new(&data).unwrap();
        // The decoder borrows data, so primary_jpeg should be a subslice of our data
        let primary = decoder.primary_jpeg().unwrap();
        assert_eq!(primary.as_ptr(), data.as_ptr());
    }

    #[test]
    fn test_decoder_empty_too_short() {
        assert!(Decoder::new(&[]).is_err());
        assert!(Decoder::new(&[0xFF]).is_err());
        assert!(Decoder::new(&[0xFF, 0xD8]).is_err()); // Too short (< 4)
    }

    #[test]
    fn test_decoder_icc_profile_none() {
        let data = vec![0xFF, 0xD8, 0xFF, 0xD9];
        let decoder = Decoder::new(&data).unwrap();
        assert!(decoder.icc_profile().is_none());
    }

    #[test]
    fn test_decoder_two_jpeg_fallback() {
        // Two concatenated JPEGs — should find both via boundary scan
        let data = vec![
            0xFF, 0xD8, // SOI 1
            0xFF, 0xD9, // EOI 1
            0xFF, 0xD8, // SOI 2
            0xFF, 0xD9, // EOI 2
        ];
        // Need to be >= 4 bytes total
        let decoder = Decoder::new(&data).unwrap();
        assert!(decoder.primary_jpeg().is_some());
        assert!(decoder.gainmap_jpeg().is_some());
    }

    #[test]
    fn test_find_xmp_in_segments_none() {
        let segments: Vec<AppSegment> = vec![];
        assert!(find_xmp_in_segments(&segments).is_none());
    }

    #[test]
    fn test_decoder_xmp_without_hdrgm() {
        // Build a fake JPEG with XMP APP1 containing valid XML but no hdrgm namespace
        let xmp_ns = b"http://ns.adobe.com/xap/1.0/\0";
        let xmp_body = b"<x:xmpmeta xmlns:x=\"adobe:ns:meta/\"><rdf:RDF xmlns:rdf=\"http://www.w3.org/1999/02/22-rdf-syntax-ns#\"><rdf:Description rdf:about=\"\" xmlns:dc=\"http://purl.org/dc/elements/1.1/\"><dc:creator>test</dc:creator></rdf:Description></rdf:RDF></x:xmpmeta>";
        let segment_data_len = xmp_ns.len() + xmp_body.len();
        let segment_len = (segment_data_len + 2) as u16; // +2 for length field itself

        let mut data = Vec::new();
        data.extend_from_slice(&[0xFF, 0xD8]); // SOI
        data.push(0xFF);
        data.push(0xE1); // APP1
        data.extend_from_slice(&segment_len.to_be_bytes());
        data.extend_from_slice(xmp_ns);
        data.extend_from_slice(xmp_body);
        data.extend_from_slice(&[0xFF, 0xD9]); // EOI

        let decoder = Decoder::new(&data).unwrap();
        assert!(!decoder.is_ultrahdr());
        assert!(decoder.metadata().is_none());
    }

    #[test]
    fn test_decoder_primary_jpeg_is_full_data_when_no_mpf() {
        // Plain JPEG with no MPF — primary_jpeg() should return the entire data
        let data = vec![
            0xFF, 0xD8, // SOI
            0xFF, 0xE0, 0x00, 0x07, // APP0 length 7
            b'J', b'F', b'I', b'F', 0x00, // JFIF
            0xFF, 0xD9, // EOI
        ];
        let decoder = Decoder::new(&data).unwrap();
        let primary = decoder.primary_jpeg().unwrap();
        assert_eq!(primary.len(), data.len());
        assert_eq!(primary, &data[..]);
    }

    #[test]
    fn test_decoder_gainmap_none_on_plain_jpeg() {
        // Plain JPEG with no secondary images — gainmap_jpeg() should be None
        let data = vec![
            0xFF, 0xD8, // SOI
            0xFF, 0xE0, 0x00, 0x07, // APP0 length 7
            b'J', b'F', b'I', b'F', 0x00, // JFIF
            0xFF, 0xD9, // EOI
        ];
        let decoder = Decoder::new(&data).unwrap();
        assert!(decoder.gainmap_jpeg().is_none());
    }

    #[test]
    fn test_find_xmp_in_segments_with_non_xmp() {
        // APP1 segment that does NOT start with the XMP namespace (e.g., EXIF)
        let segments = vec![AppSegment {
            marker_num: 1,
            data: b"Exif\0\0some_exif_data_here".to_vec(),
            offset: 0,
        }];
        assert!(find_xmp_in_segments(&segments).is_none());

        // APP1 with arbitrary data (not XMP, not EXIF)
        let segments = vec![AppSegment {
            marker_num: 1,
            data: b"SomeRandomPrefix\0and_data".to_vec(),
            offset: 0,
        }];
        assert!(find_xmp_in_segments(&segments).is_none());
    }

    #[test]
    fn test_find_xmp_in_segments_with_xmp() {
        let xmp_ns = b"http://ns.adobe.com/xap/1.0/\0";
        let xmp_xml = b"<x:xmpmeta><rdf:RDF><rdf:Description/></rdf:RDF></x:xmpmeta>";

        let mut segment_data = Vec::new();
        segment_data.extend_from_slice(xmp_ns);
        segment_data.extend_from_slice(xmp_xml);

        let segments = vec![AppSegment {
            marker_num: 1,
            data: segment_data,
            offset: 10,
        }];

        let result = find_xmp_in_segments(&segments);
        assert!(result.is_some());
        let xmp_str = result.unwrap();
        assert!(xmp_str.contains("<x:xmpmeta>"));
        assert!(xmp_str.contains("<rdf:RDF>"));
    }
}
