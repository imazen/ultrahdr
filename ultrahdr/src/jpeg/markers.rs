//! JPEG marker handling utilities.

use ultrahdr_core::{Error, Result};

/// JPEG marker types.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum Marker {
    /// Start of Image
    Soi = 0xD8,
    /// End of Image
    Eoi = 0xD9,
    /// Start of Frame (baseline DCT)
    Sof0 = 0xC0,
    /// Start of Frame (progressive DCT)
    Sof2 = 0xC2,
    /// Define Huffman Table
    Dht = 0xC4,
    /// Define Quantization Table
    Dqt = 0xDB,
    /// Define Restart Interval
    Dri = 0xDD,
    /// Start of Scan
    Sos = 0xDA,
    /// APP0 (JFIF)
    App0 = 0xE0,
    /// APP1 (EXIF/XMP)
    App1 = 0xE1,
    /// APP2 (ICC/MPF)
    App2 = 0xE2,
    /// Comment
    Com = 0xFE,
}

impl Marker {
    /// Check if this marker type has a length field.
    pub fn has_length(&self) -> bool {
        !matches!(self, Marker::Soi | Marker::Eoi)
    }
}

/// A parsed JPEG segment.
#[derive(Debug, Clone)]
pub struct JpegSegment {
    /// Marker type.
    pub marker: u8,
    /// Segment data (excluding marker and length bytes).
    pub data: Vec<u8>,
    /// Offset in the original file.
    pub offset: usize,
}

/// Parse JPEG into segments.
pub fn parse_jpeg_segments(data: &[u8]) -> Result<Vec<JpegSegment>> {
    let mut segments = Vec::new();

    // Check for SOI
    if data.len() < 2 || data[0] != 0xFF || data[1] != 0xD8 {
        return Err(Error::JpegDecode("Not a valid JPEG (missing SOI)".into()));
    }

    segments.push(JpegSegment {
        marker: 0xD8,
        data: Vec::new(),
        offset: 0,
    });
    let mut pos = 2;

    while pos < data.len() - 1 {
        // Find next marker
        if data[pos] != 0xFF {
            pos += 1;
            continue;
        }

        // Skip padding FF bytes
        while pos < data.len() - 1 && data[pos + 1] == 0xFF {
            pos += 1;
        }

        if pos >= data.len() - 1 {
            break;
        }

        let marker = data[pos + 1];
        let offset = pos;
        pos += 2;

        // EOI - end of image
        if marker == 0xD9 {
            segments.push(JpegSegment {
                marker,
                data: Vec::new(),
                offset,
            });
            break;
        }

        // RST markers (D0-D7) - no length
        if (0xD0..=0xD7).contains(&marker) {
            segments.push(JpegSegment {
                marker,
                data: Vec::new(),
                offset,
            });
            continue;
        }

        // SOI - no length
        if marker == 0xD8 {
            segments.push(JpegSegment {
                marker,
                data: Vec::new(),
                offset,
            });
            continue;
        }

        // Read length for other markers
        if pos + 2 > data.len() {
            break;
        }

        let length = u16::from_be_bytes([data[pos], data[pos + 1]]) as usize;
        if length < 2 || pos + length > data.len() {
            return Err(Error::JpegDecode(format!(
                "Invalid segment length {} at offset {}",
                length, offset
            )));
        }

        let segment_data = data[pos + 2..pos + length].to_vec();
        segments.push(JpegSegment {
            marker,
            data: segment_data,
            offset,
        });

        pos += length;

        // SOS - scan data follows until next marker
        if marker == 0xDA {
            // Find the end of scan data (next marker that's not 0x00 or RST)
            let scan_start = pos;
            while pos < data.len() - 1 {
                if data[pos] == 0xFF
                    && data[pos + 1] != 0x00
                    && !(0xD0..=0xD7).contains(&data[pos + 1])
                {
                    break;
                }
                pos += 1;
            }

            // Add scan data as pseudo-segment
            if pos > scan_start {
                segments.push(JpegSegment {
                    marker: 0x00, // Pseudo-marker for scan data
                    data: data[scan_start..pos].to_vec(),
                    offset: scan_start,
                });
            }
        }
    }

    Ok(segments)
}

/// Reconstruct JPEG from segments.
pub fn reconstruct_jpeg(segments: &[JpegSegment]) -> Vec<u8> {
    let mut data = Vec::new();

    for segment in segments {
        if segment.marker == 0x00 {
            // Scan data - no marker
            data.extend_from_slice(&segment.data);
        } else if segment.marker == 0xD8 || segment.marker == 0xD9 {
            // SOI/EOI - no length
            data.push(0xFF);
            data.push(segment.marker);
        } else if (0xD0..=0xD7).contains(&segment.marker) {
            // RST - no length
            data.push(0xFF);
            data.push(segment.marker);
        } else {
            // Marker with length
            data.push(0xFF);
            data.push(segment.marker);
            let length = (segment.data.len() + 2) as u16;
            data.push((length >> 8) as u8);
            data.push((length & 0xFF) as u8);
            data.extend_from_slice(&segment.data);
        }
    }

    data
}

/// Insert a segment after the SOI marker.
pub fn insert_segment_after_soi(jpeg: &[u8], segment: &JpegSegment) -> Result<Vec<u8>> {
    if jpeg.len() < 2 || jpeg[0] != 0xFF || jpeg[1] != 0xD8 {
        return Err(Error::JpegDecode("Not a valid JPEG".into()));
    }

    let mut result = Vec::with_capacity(jpeg.len() + segment.data.len() + 4);

    // SOI
    result.push(0xFF);
    result.push(0xD8);

    // New segment
    result.push(0xFF);
    result.push(segment.marker);
    let length = (segment.data.len() + 2) as u16;
    result.push((length >> 8) as u8);
    result.push((length & 0xFF) as u8);
    result.extend_from_slice(&segment.data);

    // Rest of original JPEG
    result.extend_from_slice(&jpeg[2..]);

    Ok(result)
}

/// Find XMP data in JPEG segments.
pub fn find_xmp_data(data: &[u8]) -> Option<String> {
    let xmp_marker = b"http://ns.adobe.com/xap/1.0/\0";

    let mut pos = 0;
    while pos + 4 < data.len() {
        if data[pos] == 0xFF && data[pos + 1] == 0xE1 {
            let length = u16::from_be_bytes([data[pos + 2], data[pos + 3]]) as usize;
            if pos + 4 + xmp_marker.len() < data.len() {
                let marker_data = &data[pos + 4..];
                if marker_data.starts_with(xmp_marker) {
                    let xmp_start = xmp_marker.len();
                    let xmp_end = length - 2;
                    if xmp_start < xmp_end && pos + 4 + xmp_end <= data.len() {
                        if let Ok(xmp) = std::str::from_utf8(&marker_data[xmp_start..xmp_end]) {
                            return Some(xmp.to_string());
                        }
                    }
                }
            }
        }
        pos += 1;
    }

    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_minimal_jpeg() {
        // Minimal JPEG: SOI + EOI
        let data = vec![0xFF, 0xD8, 0xFF, 0xD9];
        let segments = parse_jpeg_segments(&data).unwrap();

        assert_eq!(segments.len(), 2);
        assert_eq!(segments[0].marker, 0xD8);
        assert_eq!(segments[1].marker, 0xD9);
    }

    #[test]
    fn test_reconstruct_jpeg() {
        let segments = vec![
            JpegSegment {
                marker: 0xD8,
                data: Vec::new(),
                offset: 0,
            },
            JpegSegment {
                marker: 0xE0,
                data: vec![0x4A, 0x46, 0x49, 0x46, 0x00],
                offset: 2,
            },
            JpegSegment {
                marker: 0xD9,
                data: Vec::new(),
                offset: 10,
            },
        ];

        let reconstructed = reconstruct_jpeg(&segments);

        // SOI
        assert_eq!(reconstructed[0], 0xFF);
        assert_eq!(reconstructed[1], 0xD8);
        // APP0
        assert_eq!(reconstructed[2], 0xFF);
        assert_eq!(reconstructed[3], 0xE0);
    }
}
