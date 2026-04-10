//! PixelBuffer/PixelSlice adapters for zenpixels interop.
//!
//! Provides zero-ceremony conversion between zenpixels pixel types and
//! ultrahdr-core's [`RawImage`], plus a one-call tone mapping function
//! that takes and returns zenpixels types.
//!
//! # Feature gate
//!
//! Requires the `zenpixels` feature (which implies `transfer`).

use alloc::format;
use alloc::vec::Vec;

use zenpixels::{Cicp, ColorPrimaries, PixelBuffer, PixelDescriptor, PixelFormat, PixelSlice};

use crate::color::tonemap::{ToneMapConfig, tonemap_to_sdr};
use crate::color::transfer::srgb_oetf;
use crate::types::{ColorGamut, ColorTransfer, Error, RawImage, Result};

// ============================================================================
// PixelBuffer / PixelSlice → RawImage
// ============================================================================

/// Build a [`RawImage`] from a [`PixelBuffer`] and [`Cicp`] color description.
///
/// Copies pixel data contiguously (removes stride padding if present).
/// The CICP provides the color gamut and transfer function — these are not
/// stored on `PixelBuffer` itself.
///
/// # Supported formats
///
/// | `PixelFormat` | Maps to |
/// |---|---|
/// | `Rgba8` | `PixelFormat::Rgba8` |
/// | `Rgb8` | `PixelFormat::Rgb8` |
/// | `RgbaF32` | `PixelFormat::Rgba32F` |
/// | `Rgba16` | `PixelFormat::Rgba32F` (u16 → f32 conversion) |
///
/// Other formats return `Error::InvalidPixelData`.
pub fn raw_image_from_buffer(buffer: &PixelBuffer, cicp: &Cicp) -> Result<RawImage> {
    let desc = buffer.descriptor();
    let (format, data) = convert_pixels(buffer, desc)?;
    let gamut = ColorGamut::from(
        ColorPrimaries::from_cicp(cicp.color_primaries).unwrap_or(ColorPrimaries::Bt709),
    );
    let transfer = ColorTransfer::from(
        zenpixels::TransferFunction::from_cicp(cicp.transfer_characteristics)
            .unwrap_or(zenpixels::TransferFunction::Srgb),
    );

    RawImage::from_data(
        buffer.width(),
        buffer.height(),
        format,
        gamut,
        transfer,
        data,
    )
}

/// Build a [`RawImage`] from a [`PixelSlice`] and [`Cicp`] color description.
///
/// Same as [`raw_image_from_buffer`] but works with borrowed pixel data.
pub fn raw_image_from_slice(
    slice: PixelSlice<'_>,
    cicp: &Cicp,
    width: u32,
    height: u32,
) -> Result<RawImage> {
    let desc = slice.descriptor();
    let format = pixel_format_to_ultrahdr(desc.format)?;
    let data = if desc.format == PixelFormat::Rgba16 || desc.format == PixelFormat::Rgb16 {
        convert_u16_to_f32(slice.contiguous_bytes().as_ref(), desc)
    } else {
        slice.contiguous_bytes().into_owned()
    };
    let gamut = ColorGamut::from(
        ColorPrimaries::from_cicp(cicp.color_primaries).unwrap_or(ColorPrimaries::Bt709),
    );
    let transfer = ColorTransfer::from(
        zenpixels::TransferFunction::from_cicp(cicp.transfer_characteristics)
            .unwrap_or(zenpixels::TransferFunction::Srgb),
    );

    RawImage::from_data(width, height, format, gamut, transfer, data)
}

// ============================================================================
// Tone mapping: HDR PixelBuffer → SDR PixelBuffer
// ============================================================================

/// Tone map HDR pixels to SDR. Returns `None` for SDR input (no work needed).
///
/// When tone mapping is performed, returns `Some((buffer, cicp))` with RGBA8
/// output and the corresponding `Cicp` describing the output color space.
///
/// # Smart defaults
///
/// When `target_primaries` is `None`:
/// - Display P3 source → Display P3 SDR (preserves wide gamut)
/// - BT.2020 source → BT.709 sRGB (broadest compatibility)
/// - BT.709 source → BT.709 sRGB (identity gamut)
/// - Unknown source → BT.709 sRGB (safe fallback)
///
/// # Supported input
///
/// Any HDR `PixelBuffer` with PQ (tc=16) or HLG (tc=18) transfer function
/// and known primaries. SDR input (any other transfer) is returned as-is
/// (cloned).
///
/// # Example
///
/// ```rust,ignore
/// use ultrahdr_core::zenpixels_adapter::tonemap_to_sdr_buffer;
/// use zenpixels::Cicp;
///
/// let (sdr_buffer, sdr_cicp) = tonemap_to_sdr_buffer(&hdr_buffer, &hdr_cicp, None)?;
/// // sdr_buffer is RGBA8 sRGB (or P3 SDR if source was P3)
/// // sdr_cicp is Cicp::SRGB or Cicp::DISPLAY_P3
/// ```
pub fn tonemap_to_sdr_buffer(
    buffer: &PixelBuffer,
    source_cicp: &Cicp,
    target_primaries: Option<ColorPrimaries>,
) -> Result<Option<(PixelBuffer, Cicp)>> {
    // SDR passthrough — not PQ or HLG, nothing to tone map
    if !matches!(source_cicp.transfer_characteristics, 16 | 18) {
        return Ok(None);
    }

    let raw = raw_image_from_buffer(buffer, source_cicp)?;

    let source_gamut = raw.gamut;
    let target_gamut = match target_primaries {
        Some(p) => ColorGamut::from(p),
        None => default_target_gamut(source_gamut),
    };

    let config = ToneMapConfig {
        target_gamut,
        source_gamut,
        ..ToneMapConfig::default()
    };

    let width = buffer.width() as usize;
    let height = buffer.height() as usize;
    let mut output = alloc::vec![0u8; width * height * 4];

    for y in 0..height {
        for x in 0..width {
            let linear_rgb = get_linear_rgb_from_raw(&raw, x as u32, y as u32);

            // Gamut convert in linear space
            let gamut_rgb = if source_gamut != target_gamut {
                crate::color::gamut::convert_gamut(linear_rgb, source_gamut, target_gamut)
            } else {
                linear_rgb
            };

            // Tone map
            let sdr = tonemap_to_sdr(gamut_rgb, raw.transfer, &config);

            // sRGB OETF + quantize (correct for both BT.709 and P3 SDR)
            let idx = (y * width + x) * 4;
            output[idx] = (srgb_oetf(sdr[0]) * 255.0).round().clamp(0.0, 255.0) as u8;
            output[idx + 1] = (srgb_oetf(sdr[1]) * 255.0).round().clamp(0.0, 255.0) as u8;
            output[idx + 2] = (srgb_oetf(sdr[2]) * 255.0).round().clamp(0.0, 255.0) as u8;
            output[idx + 3] = 255;
        }
    }

    let out_cicp = match target_gamut {
        ColorGamut::Bt709 => Cicp::SRGB,
        ColorGamut::DisplayP3 => Cicp::DISPLAY_P3,
        ColorGamut::Bt2020 => Cicp::new(9, 13, 0, true), // BT.2020 + sRGB TRC (approximation)
    };

    let sdr_buffer = PixelBuffer::from_vec(
        output,
        buffer.width(),
        buffer.height(),
        PixelDescriptor::RGBA8_SRGB,
    )
    .map_err(|e| Error::InvalidPixelData(format!("failed to create SDR buffer: {e}")))?;

    Ok(Some((sdr_buffer, out_cicp)))
}

// ============================================================================
// Internal helpers
// ============================================================================

/// Smart default: preserve gamut for P3, downgrade to BT.709 for BT.2020.
fn default_target_gamut(source: ColorGamut) -> ColorGamut {
    match source {
        ColorGamut::DisplayP3 => ColorGamut::DisplayP3,
        _ => ColorGamut::Bt709,
    }
}

/// Map zenpixels PixelFormat to ultrahdr-core PixelFormat.
fn pixel_format_to_ultrahdr(format: PixelFormat) -> Result<crate::PixelFormat> {
    match format {
        PixelFormat::Rgba8 => Ok(crate::PixelFormat::Rgba8),
        PixelFormat::Rgb8 => Ok(crate::PixelFormat::Rgb8),
        PixelFormat::RgbaF32 => Ok(crate::PixelFormat::Rgba32F),
        // u16 formats get promoted to f32 in convert_pixels
        PixelFormat::Rgba16 | PixelFormat::Rgb16 => Ok(crate::PixelFormat::Rgba32F),
        _ => Err(Error::InvalidPixelData(format!(
            "unsupported pixel format for tone mapping: {format:?}"
        ))),
    }
}

/// Extract pixels from PixelBuffer, converting u16 to f32 if needed.
fn convert_pixels(
    buffer: &PixelBuffer,
    desc: PixelDescriptor,
) -> Result<(crate::PixelFormat, Vec<u8>)> {
    let format = pixel_format_to_ultrahdr(desc.format)?;
    let slice = buffer.as_slice();

    if desc.format == PixelFormat::Rgba16 || desc.format == PixelFormat::Rgb16 {
        let data = convert_u16_to_f32(slice.contiguous_bytes().as_ref(), desc);
        Ok((crate::PixelFormat::Rgba32F, data))
    } else {
        Ok((format, slice.contiguous_bytes().into_owned()))
    }
}

/// Convert u16 pixel data to f32 RGBA (normalized 0-1).
fn convert_u16_to_f32(bytes: &[u8], desc: PixelDescriptor) -> Vec<u8> {
    let u16s: &[u16] = bytemuck::cast_slice(bytes);
    let channels = desc.format.layout().channels();
    let pixel_count = u16s.len() / channels;
    let mut out = alloc::vec![0u8; pixel_count * 16]; // 4 × f32 per pixel
    let f32s: &mut [f32] = bytemuck::cast_slice_mut(&mut out);

    for i in 0..pixel_count {
        let src = i * channels;
        let dst = i * 4;
        f32s[dst] = u16s[src] as f32 / 65535.0;
        f32s[dst + 1] = u16s[src + 1] as f32 / 65535.0;
        f32s[dst + 2] = u16s[src + 2] as f32 / 65535.0;
        f32s[dst + 3] = if channels >= 4 {
            u16s[src + 3] as f32 / 65535.0
        } else {
            1.0
        };
    }
    out
}

/// Get linear RGB from a RawImage at (x, y).
///
/// Handles PQ/HLG/sRGB/Linear transfers and Rgba8/Rgb8/Rgba32F formats.
fn get_linear_rgb_from_raw(img: &RawImage, x: u32, y: u32) -> [f32; 3] {
    use crate::color::transfer::{hlg_eotf, pq_eotf, srgb_eotf};

    match img.format {
        crate::PixelFormat::Rgba8 | crate::PixelFormat::Rgb8 => {
            let bpp = if img.format == crate::PixelFormat::Rgba8 {
                4
            } else {
                3
            };
            let idx = (y * img.stride + x * bpp as u32) as usize;
            let r = img.data[idx] as f32 / 255.0;
            let g = img.data[idx + 1] as f32 / 255.0;
            let b = img.data[idx + 2] as f32 / 255.0;
            match img.transfer {
                ColorTransfer::Srgb => [srgb_eotf(r), srgb_eotf(g), srgb_eotf(b)],
                ColorTransfer::Pq => {
                    let scale = 10000.0;
                    [pq_eotf(r) * scale, pq_eotf(g) * scale, pq_eotf(b) * scale]
                }
                ColorTransfer::Hlg => [
                    hlg_eotf(r, 1000.0),
                    hlg_eotf(g, 1000.0),
                    hlg_eotf(b, 1000.0),
                ],
                ColorTransfer::Linear => [r, g, b],
            }
        }
        crate::PixelFormat::Rgba32F => {
            let idx = (y * img.stride + x * 16) as usize;
            let r = f32::from_le_bytes([
                img.data[idx],
                img.data[idx + 1],
                img.data[idx + 2],
                img.data[idx + 3],
            ]);
            let g = f32::from_le_bytes([
                img.data[idx + 4],
                img.data[idx + 5],
                img.data[idx + 6],
                img.data[idx + 7],
            ]);
            let b = f32::from_le_bytes([
                img.data[idx + 8],
                img.data[idx + 9],
                img.data[idx + 10],
                img.data[idx + 11],
            ]);
            match img.transfer {
                ColorTransfer::Pq => {
                    let scale = 10000.0;
                    [pq_eotf(r) * scale, pq_eotf(g) * scale, pq_eotf(b) * scale]
                }
                ColorTransfer::Hlg => [
                    hlg_eotf(r, 1000.0),
                    hlg_eotf(g, 1000.0),
                    hlg_eotf(b, 1000.0),
                ],
                _ => [r, g, b], // Linear or sRGB f32
            }
        }
        _ => [0.0, 0.0, 0.0],
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_srgb_rgba8(w: u32, h: u32, pixel: [u8; 4]) -> PixelBuffer {
        let mut data = Vec::new();
        for _ in 0..(w * h) {
            data.extend_from_slice(&pixel);
        }
        PixelBuffer::from_vec(data, w, h, PixelDescriptor::RGBA8_SRGB).unwrap()
    }

    #[test]
    fn raw_image_from_srgb_buffer() {
        let buf = make_srgb_rgba8(2, 2, [128, 64, 32, 255]);
        let raw = raw_image_from_buffer(&buf, &Cicp::SRGB).unwrap();
        assert_eq!(raw.width, 2);
        assert_eq!(raw.height, 2);
        assert_eq!(raw.gamut, ColorGamut::Bt709);
        assert_eq!(raw.transfer, ColorTransfer::Srgb);
        assert_eq!(raw.format, crate::PixelFormat::Rgba8);
        assert_eq!(raw.data[0], 128);
        assert_eq!(raw.data[1], 64);
        assert_eq!(raw.data[2], 32);
        assert_eq!(raw.data[3], 255);
    }

    #[test]
    fn raw_image_from_pq_buffer() {
        let buf = make_srgb_rgba8(1, 1, [100, 100, 100, 255]);
        let pq_cicp = Cicp::BT2100_PQ;
        let raw = raw_image_from_buffer(&buf, &pq_cicp).unwrap();
        assert_eq!(raw.gamut, ColorGamut::Bt2020);
        assert_eq!(raw.transfer, ColorTransfer::Pq);
    }

    #[test]
    fn raw_image_from_p3_buffer() {
        let buf = make_srgb_rgba8(1, 1, [200, 150, 100, 255]);
        let p3_cicp = Cicp::DISPLAY_P3;
        let raw = raw_image_from_buffer(&buf, &p3_cicp).unwrap();
        assert_eq!(raw.gamut, ColorGamut::DisplayP3);
        assert_eq!(raw.transfer, ColorTransfer::Srgb);
    }

    #[test]
    fn tonemap_sdr_passthrough() {
        let buf = make_srgb_rgba8(2, 2, [128, 64, 32, 255]);
        let result = tonemap_to_sdr_buffer(&buf, &Cicp::SRGB, None).unwrap();
        // SDR passthrough — returns None (no work needed)
        assert!(result.is_none());
    }

    #[test]
    fn tonemap_pq_bt2020_to_srgb() {
        // Synthetic PQ RGBA8 — mid-gray in PQ is around value 127
        let buf = make_srgb_rgba8(4, 4, [127, 127, 127, 255]);
        let pq_cicp = Cicp::BT2100_PQ;
        let (result, cicp) = tonemap_to_sdr_buffer(&buf, &pq_cicp, None)
            .unwrap()
            .unwrap();
        assert_eq!(cicp, Cicp::SRGB);
        assert_eq!(result.width(), 4);
        assert_eq!(result.height(), 4);
        // Output should be valid sRGB — not all 0 or all 255 (would indicate clipping)
        let slice = result.as_slice();
        let row = slice.row(0);
        assert!(
            row[0] > 0 && row[0] < 255,
            "R should be in SDR range, got {}",
            row[0]
        );
    }

    #[test]
    fn tonemap_pq_p3_preserves_gamut() {
        let buf = make_srgb_rgba8(2, 2, [127, 127, 127, 255]);
        let p3_pq = Cicp::new(12, 16, 0, true); // P3 + PQ
        let (_, cicp) = tonemap_to_sdr_buffer(&buf, &p3_pq, None).unwrap().unwrap();
        // Smart default: P3 source → P3 SDR
        assert_eq!(cicp, Cicp::DISPLAY_P3);
    }

    #[test]
    fn tonemap_explicit_target_overrides_default() {
        let buf = make_srgb_rgba8(2, 2, [127, 127, 127, 255]);
        let pq_cicp = Cicp::BT2100_PQ;
        // Explicitly request P3 SDR instead of default BT.709
        let (_, cicp) = tonemap_to_sdr_buffer(&buf, &pq_cicp, Some(ColorPrimaries::DisplayP3))
            .unwrap()
            .unwrap();
        assert_eq!(cicp, Cicp::DISPLAY_P3);
    }

    #[test]
    fn tonemap_hlg_to_srgb() {
        let buf = make_srgb_rgba8(2, 2, [127, 127, 127, 255]);
        let hlg_cicp = Cicp::BT2100_HLG;
        let (result, cicp) = tonemap_to_sdr_buffer(&buf, &hlg_cicp, None)
            .unwrap()
            .unwrap();
        assert_eq!(cicp, Cicp::SRGB);
        let slice = result.as_slice();
        let row = slice.row(0);
        assert!(row[0] > 0 && row[0] < 255, "should produce valid SDR");
    }
}
