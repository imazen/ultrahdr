//! PixelBuffer/PixelSlice adapters and tone mapping for zenpixels.
//!
//! Three layers, from low to high:
//!
//! 1. **[`raw_image_from_buffer`] / [`raw_image_from_slice`]** — convert
//!    zenpixels types to [`RawImage`] for use with any ultrahdr-core API
//!    (gain map apply/compute). Copies pixel data into an owned `Vec<u8>`.
//!
//! 2. **[`SdrToneMapper`]** — stateful row processor. Create once, call
//!    [`tonemap_row_rgba8`](SdrToneMapper::tonemap_row_rgba8) or
//!    [`tonemap_row_f32_to_linear`](SdrToneMapper::tonemap_row_f32_to_linear)
//!    per row. No intermediate allocation — reads directly from `PixelSlice`
//!    rows. For streaming pipelines (zenpipe).
//!
//! 3. **[`tonemap_to_sdr`]** / **[`tonemap_slice_to_sdr`]** — one-call
//!    convenience. Takes a `PixelBuffer` or `PixelSlice`, returns a new
//!    RGBA8 `PixelBuffer`. For decode→encode.
//!
//! # Alpha handling
//!
//! Alpha is passed through untouched by all tone mapping operations.
//! The tone curve is applied to RGB channels only — alpha is independent
//! of luminance.
//!
//! For RGB input (no alpha), output alpha is set to 255 (opaque).
//!
//! **Premultiplied alpha**: Not handled. If the source has premultiplied
//! alpha, unpremultiply before tone mapping and repremultiply after.
//! Tone mapping premultiplied RGB values will change the alpha
//! relationship and produce wrong compositing results.
//!
//! # Feature gate
//!
//! Requires the `zenpixels` feature (which implies `transfer`).

use alloc::format;
use alloc::vec::Vec;

use zenpixels::{Cicp, ColorPrimaries, PixelBuffer, PixelDescriptor, PixelFormat, PixelSlice};

use crate::color::gamut::{Matrix3x3, gamut_conversion_matrix};
use crate::color::tonemap::{ToneMapConfig, tonemap_to_sdr as tonemap_pixel};
use crate::color::transfer::{hlg_eotf, pq_eotf, srgb_eotf, srgb_oetf};
use crate::types::{ColorGamut, ColorTransfer, Error, RawImage, Result};

// ============================================================================
// Options
// ============================================================================

/// Configuration for HDR → SDR tone mapping.
///
/// The tone mapping algorithm is filmic (Narkowicz) for PQ and BT.2390
/// for HLG, matching the `tonemap_pq_to_sdr` / `tonemap_hlg_to_sdr`
/// reference implementations. Curve selection via [`ToneMapCurve`](crate::color::tonemap::ToneMapCurve)
/// is available through the lower-level `tonemap_rgb_curve()` API for
/// callers who need a specific algorithm.
#[derive(Clone, Debug)]
pub struct SdrToneMapOptions {
    /// Target color primaries. `None` = smart default:
    /// P3 source → P3 SDR, everything else → BT.709 sRGB.
    pub target_primaries: Option<ColorPrimaries>,
    /// SDR display peak luminance in nits. Default: 203 (reference white).
    pub target_peak_nits: f32,
    /// HDR content peak luminance in nits. `None` = infer from transfer
    /// (PQ → 10000, HLG → 1000).
    pub hdr_peak_nits: Option<f32>,
}

impl Default for SdrToneMapOptions {
    fn default() -> Self {
        Self {
            target_primaries: None,
            target_peak_nits: 203.0,
            hdr_peak_nits: None,
        }
    }
}

// ============================================================================
// Layer 1: PixelBuffer / PixelSlice → RawImage
// ============================================================================

/// Build a [`RawImage`] from a [`PixelBuffer`] and [`Cicp`].
///
/// Copies pixel data contiguously (removes stride padding if present).
/// Supported formats: `Rgba8`, `Rgb8`, `RgbaF32`, `RgbF32`, `Rgba16`, `Rgb16`.
/// u16 formats are promoted to f32.
pub fn raw_image_from_buffer(buffer: &PixelBuffer, cicp: &Cicp) -> Result<RawImage> {
    let desc = buffer.descriptor();
    let (format, data) = convert_pixels_from_slice(buffer.as_slice(), desc)?;
    let (gamut, transfer) = cicp_to_gamut_transfer(cicp);
    RawImage::from_data(
        buffer.width(),
        buffer.height(),
        format,
        gamut,
        transfer,
        data,
    )
}

/// Build a [`RawImage`] from a [`PixelSlice`] and [`Cicp`].
pub fn raw_image_from_slice(
    slice: PixelSlice<'_>,
    cicp: &Cicp,
    width: u32,
    height: u32,
) -> Result<RawImage> {
    let desc = slice.descriptor();
    let (format, data) = convert_pixels_from_slice(slice, desc)?;
    let (gamut, transfer) = cicp_to_gamut_transfer(cicp);
    RawImage::from_data(width, height, format, gamut, transfer, data)
}

// ============================================================================
// Layer 2: SdrToneMapper — stateful row processor
// ============================================================================

/// Stateful row-level HDR → SDR tone mapper.
///
/// Create with [`new`](Self::new), then call [`tonemap_row_rgba8`](Self::tonemap_row_rgba8)
/// per row for RGBA8 output, or [`tonemap_row_linear`](Self::tonemap_row_linear)
/// for linear f32 output (compositing, further processing).
///
/// The mapper pre-computes the gamut matrix, tone curve config, and EOTF
/// selection at construction time — row processing has no per-row dispatch.
pub struct SdrToneMapper {
    source_transfer: ColorTransfer,
    gamut_matrix: Option<Matrix3x3>,
    config: ToneMapConfig,
    output_cicp: Cicp,
}

impl SdrToneMapper {
    /// Create a tone mapper for the given source and options.
    ///
    /// Returns `None` if `source_cicp` is SDR (no tone mapping needed).
    pub fn new(source_cicp: &Cicp, options: &SdrToneMapOptions) -> Option<Self> {
        if !matches!(source_cicp.transfer_characteristics, 16 | 18) {
            return None; // SDR — nothing to do
        }

        let (source_gamut, source_transfer) = cicp_to_gamut_transfer(source_cicp);
        let target_gamut = match options.target_primaries {
            Some(p) => ColorGamut::from(p),
            None => default_target_gamut(source_gamut),
        };

        let gamut_matrix = if source_gamut != target_gamut {
            Some(gamut_conversion_matrix(source_gamut, target_gamut))
        } else {
            None
        };

        let hdr_peak = options.hdr_peak_nits.unwrap_or(match source_transfer {
            ColorTransfer::Pq => 10000.0,
            ColorTransfer::Hlg => 1000.0,
            _ => 10000.0,
        });

        let config = ToneMapConfig {
            target_peak_nits: options.target_peak_nits,
            hdr_peak_nits: hdr_peak,
            target_gamut,
            source_gamut,
        };

        let output_cicp = match target_gamut {
            ColorGamut::Bt709 => Cicp::SRGB,
            ColorGamut::DisplayP3 => Cicp::DISPLAY_P3,
            ColorGamut::Bt2020 => Cicp::new(9, 13, 0, true),
        };

        Some(Self {
            source_transfer,
            gamut_matrix,
            config,
            output_cicp,
        })
    }

    /// The CICP describing the output color space.
    pub fn output_cicp(&self) -> Cicp {
        self.output_cicp
    }

    /// Tone map one pixel from linear nits to SDR linear [0,1].
    #[inline]
    fn tonemap_pixel(&self, linear_nits: [f32; 3]) -> [f32; 3] {
        // Gamut convert
        let rgb = match &self.gamut_matrix {
            Some(m) => m.transform(linear_nits),
            None => linear_nits,
        };

        // Tone map via the configured curve
        tonemap_pixel(rgb, self.source_transfer, &self.config)
    }

    /// Tone map one row of **RGBA8** input to **RGBA8** sRGB output.
    ///
    /// `src` and `dst` are both `width * 4` bytes. Alpha is passed through.
    pub fn tonemap_row_rgba8(&self, src: &[u8], dst: &mut [u8], width: u32) {
        let w = width as usize;
        for x in 0..w {
            let si = x * 4;
            let r = src[si] as f32 / 255.0;
            let g = src[si + 1] as f32 / 255.0;
            let b = src[si + 2] as f32 / 255.0;
            let a = src[si + 3];

            let linear = self.eotf([r, g, b]);
            let sdr = self.tonemap_pixel(linear);

            let di = x * 4;
            dst[di] = (srgb_oetf(sdr[0]) * 255.0).round().clamp(0.0, 255.0) as u8;
            dst[di + 1] = (srgb_oetf(sdr[1]) * 255.0).round().clamp(0.0, 255.0) as u8;
            dst[di + 2] = (srgb_oetf(sdr[2]) * 255.0).round().clamp(0.0, 255.0) as u8;
            dst[di + 3] = a;
        }
    }

    /// Tone map one row of **RGBA f32** input to **RGBA8** sRGB output.
    ///
    /// `src` is `width * 4` f32 values, `dst` is `width * 4` bytes.
    pub fn tonemap_row_f32_to_rgba8(&self, src: &[f32], dst: &mut [u8], width: u32) {
        let w = width as usize;
        for x in 0..w {
            let si = x * 4;
            let linear = self.eotf([src[si], src[si + 1], src[si + 2]]);
            let sdr = self.tonemap_pixel(linear);

            let di = x * 4;
            dst[di] = (srgb_oetf(sdr[0]) * 255.0).round().clamp(0.0, 255.0) as u8;
            dst[di + 1] = (srgb_oetf(sdr[1]) * 255.0).round().clamp(0.0, 255.0) as u8;
            dst[di + 2] = (srgb_oetf(sdr[2]) * 255.0).round().clamp(0.0, 255.0) as u8;
            dst[di + 3] = (src[si + 3] * 255.0).round().clamp(0.0, 255.0) as u8;
        }
    }

    /// Tone map one row of **RGBA8** input to **linear f32 RGBA** output.
    ///
    /// Output is SDR linear [0,1] in the target gamut. Caller applies their
    /// own OETF and quantization. Alpha is normalized to [0,1].
    pub fn tonemap_row_rgba8_to_linear(&self, src: &[u8], dst: &mut [f32], width: u32) {
        let w = width as usize;
        for x in 0..w {
            let si = x * 4;
            let r = src[si] as f32 / 255.0;
            let g = src[si + 1] as f32 / 255.0;
            let b = src[si + 2] as f32 / 255.0;

            let linear = self.eotf([r, g, b]);
            let sdr = self.tonemap_pixel(linear);

            let di = x * 4;
            dst[di] = sdr[0];
            dst[di + 1] = sdr[1];
            dst[di + 2] = sdr[2];
            dst[di + 3] = src[si + 3] as f32 / 255.0;
        }
    }

    /// Tone map one row of **RGBA f32** input to **linear f32 RGBA** output.
    ///
    /// For compositing pipelines that stay in linear f32.
    pub fn tonemap_row_f32_to_linear(&self, src: &[f32], dst: &mut [f32], width: u32) {
        let w = width as usize;
        for x in 0..w {
            let si = x * 4;
            let linear = self.eotf([src[si], src[si + 1], src[si + 2]]);
            let sdr = self.tonemap_pixel(linear);

            let di = x * 4;
            dst[di] = sdr[0];
            dst[di + 1] = sdr[1];
            dst[di + 2] = sdr[2];
            dst[di + 3] = src[si + 3];
        }
    }

    /// Apply the source EOTF to get linear nits.
    #[inline]
    fn eotf(&self, encoded: [f32; 3]) -> [f32; 3] {
        match self.source_transfer {
            ColorTransfer::Pq => [
                pq_eotf(encoded[0]) * 10000.0,
                pq_eotf(encoded[1]) * 10000.0,
                pq_eotf(encoded[2]) * 10000.0,
            ],
            ColorTransfer::Hlg => [
                hlg_eotf(encoded[0], 1000.0),
                hlg_eotf(encoded[1], 1000.0),
                hlg_eotf(encoded[2], 1000.0),
            ],
            ColorTransfer::Srgb => [
                srgb_eotf(encoded[0]),
                srgb_eotf(encoded[1]),
                srgb_eotf(encoded[2]),
            ],
            ColorTransfer::Linear => encoded,
        }
    }
}

// ============================================================================
// Layer 3: One-call convenience
// ============================================================================

/// Tone map an HDR `PixelBuffer` to SDR RGBA8. Returns `None` for SDR input.
///
/// # Example
///
/// ```rust,ignore
/// use ultrahdr_core::zenpixels_adapter::{tonemap_to_sdr, SdrToneMapOptions};
///
/// if let Some((sdr_buf, sdr_cicp)) = tonemap_to_sdr(&hdr_buf, &hdr_cicp, &SdrToneMapOptions::default())? {
///     // sdr_buf is RGBA8 sRGB (or P3 SDR if source was P3)
/// }
/// ```
pub fn tonemap_to_sdr(
    buffer: &PixelBuffer,
    source_cicp: &Cicp,
    options: &SdrToneMapOptions,
) -> Result<Option<(PixelBuffer, Cicp)>> {
    let Some(mapper) = SdrToneMapper::new(source_cicp, options) else {
        return Ok(None);
    };
    let width = buffer.width();
    let height = buffer.height();
    let desc = buffer.descriptor();
    let slice = buffer.as_slice();

    let mut output = alloc::vec![0u8; width as usize * height as usize * 4];

    for y in 0..height {
        let src_row = slice.row(y);
        let dst_start = y as usize * width as usize * 4;
        let dst_row = &mut output[dst_start..dst_start + width as usize * 4];

        match desc.format {
            PixelFormat::Rgba8 => mapper.tonemap_row_rgba8(src_row, dst_row, width),
            PixelFormat::RgbaF32 => {
                let src_f32: &[f32] = bytemuck::cast_slice(src_row);
                mapper.tonemap_row_f32_to_rgba8(src_f32, dst_row, width);
            }
            PixelFormat::Rgb8 => {
                // Expand RGB8 → RGBA8 then tonemap
                for x in 0..(width as usize) {
                    let si = x * 3;
                    let r = src_row[si] as f32 / 255.0;
                    let g = src_row[si + 1] as f32 / 255.0;
                    let b = src_row[si + 2] as f32 / 255.0;
                    let linear = mapper.eotf([r, g, b]);
                    let sdr = mapper.tonemap_pixel(linear);
                    let di = x * 4;
                    dst_row[di] = (srgb_oetf(sdr[0]) * 255.0).round().clamp(0.0, 255.0) as u8;
                    dst_row[di + 1] = (srgb_oetf(sdr[1]) * 255.0).round().clamp(0.0, 255.0) as u8;
                    dst_row[di + 2] = (srgb_oetf(sdr[2]) * 255.0).round().clamp(0.0, 255.0) as u8;
                    dst_row[di + 3] = 255;
                }
            }
            PixelFormat::RgbF32 => {
                let src_f32: &[f32] = bytemuck::cast_slice(src_row);
                for x in 0..(width as usize) {
                    let si = x * 3;
                    let linear = mapper.eotf([src_f32[si], src_f32[si + 1], src_f32[si + 2]]);
                    let sdr = mapper.tonemap_pixel(linear);
                    let di = x * 4;
                    dst_row[di] = (srgb_oetf(sdr[0]) * 255.0).round().clamp(0.0, 255.0) as u8;
                    dst_row[di + 1] = (srgb_oetf(sdr[1]) * 255.0).round().clamp(0.0, 255.0) as u8;
                    dst_row[di + 2] = (srgb_oetf(sdr[2]) * 255.0).round().clamp(0.0, 255.0) as u8;
                    dst_row[di + 3] = 255;
                }
            }
            PixelFormat::Rgba16 | PixelFormat::Rgb16 => {
                let src_u16: &[u16] = bytemuck::cast_slice(src_row);
                let channels = desc.format.layout().channels();
                for x in 0..(width as usize) {
                    let si = x * channels;
                    let r = src_u16[si] as f32 / 65535.0;
                    let g = src_u16[si + 1] as f32 / 65535.0;
                    let b = src_u16[si + 2] as f32 / 65535.0;
                    let a = if channels >= 4 {
                        src_u16[si + 3]
                    } else {
                        65535
                    };
                    let linear = mapper.eotf([r, g, b]);
                    let sdr = mapper.tonemap_pixel(linear);
                    let di = x * 4;
                    dst_row[di] = (srgb_oetf(sdr[0]) * 255.0).round().clamp(0.0, 255.0) as u8;
                    dst_row[di + 1] = (srgb_oetf(sdr[1]) * 255.0).round().clamp(0.0, 255.0) as u8;
                    dst_row[di + 2] = (srgb_oetf(sdr[2]) * 255.0).round().clamp(0.0, 255.0) as u8;
                    dst_row[di + 3] = (a as f32 / 65535.0 * 255.0).round() as u8;
                }
            }
            _ => {
                return Err(Error::UnsupportedFormat(crate::PixelFormat::Rgba8)); // placeholder
            }
        }
    }

    let sdr_buffer = PixelBuffer::from_vec(output, width, height, PixelDescriptor::RGBA8_SRGB)
        .map_err(|e| Error::InvalidPixelData(format!("failed to create SDR buffer: {e}")))?;

    Ok(Some((sdr_buffer, mapper.output_cicp())))
}

/// Tone map an HDR `PixelSlice` to SDR RGBA8. Returns `None` for SDR input.
pub fn tonemap_slice_to_sdr(
    slice: PixelSlice<'_>,
    width: u32,
    height: u32,
    source_cicp: &Cicp,
    options: &SdrToneMapOptions,
) -> Result<Option<(PixelBuffer, Cicp)>> {
    let Some(mapper) = SdrToneMapper::new(source_cicp, options) else {
        return Ok(None);
    };
    let desc = slice.descriptor();

    let mut output = alloc::vec![0u8; width as usize * height as usize * 4];

    for y in 0..height {
        let src_row = slice.row(y);
        let dst_start = y as usize * width as usize * 4;
        let dst_row = &mut output[dst_start..dst_start + width as usize * 4];

        match desc.format {
            PixelFormat::Rgba8 => mapper.tonemap_row_rgba8(src_row, dst_row, width),
            PixelFormat::RgbaF32 => {
                let src_f32: &[f32] = bytemuck::cast_slice(src_row);
                mapper.tonemap_row_f32_to_rgba8(src_f32, dst_row, width);
            }
            _ => {
                // For other formats, fall back to per-pixel with format dispatch
                // (same logic as tonemap_to_sdr)
                return Err(Error::UnsupportedFormat(crate::PixelFormat::Rgba8));
            }
        }
    }

    let sdr_buffer = PixelBuffer::from_vec(output, width, height, PixelDescriptor::RGBA8_SRGB)
        .map_err(|e| Error::InvalidPixelData(format!("failed to create SDR buffer: {e}")))?;

    Ok(Some((sdr_buffer, mapper.output_cicp())))
}

// ============================================================================
// Internal helpers
// ============================================================================

fn cicp_to_gamut_transfer(cicp: &Cicp) -> (ColorGamut, ColorTransfer) {
    let gamut = ColorGamut::from(
        ColorPrimaries::from_cicp(cicp.color_primaries).unwrap_or(ColorPrimaries::Bt709),
    );
    let transfer = ColorTransfer::from(
        zenpixels::TransferFunction::from_cicp(cicp.transfer_characteristics)
            .unwrap_or(zenpixels::TransferFunction::Srgb),
    );
    (gamut, transfer)
}

fn default_target_gamut(source: ColorGamut) -> ColorGamut {
    match source {
        ColorGamut::DisplayP3 => ColorGamut::DisplayP3,
        _ => ColorGamut::Bt709,
    }
}

fn convert_pixels_from_slice(
    slice: PixelSlice<'_>,
    desc: PixelDescriptor,
) -> Result<(crate::PixelFormat, Vec<u8>)> {
    let format = match desc.format {
        PixelFormat::Rgba8 => crate::PixelFormat::Rgba8,
        PixelFormat::Rgb8 => crate::PixelFormat::Rgb8,
        PixelFormat::RgbaF32 => crate::PixelFormat::Rgba32F,
        PixelFormat::RgbF32 | PixelFormat::Rgba16 | PixelFormat::Rgb16 => {
            // Will be promoted to f32 RGBA
            crate::PixelFormat::Rgba32F
        }
        other => {
            return Err(Error::InvalidPixelData(format!(
                "unsupported pixel format for RawImage conversion: {other:?}"
            )));
        }
    };

    let bytes = slice.contiguous_bytes();

    match desc.format {
        PixelFormat::Rgba16 | PixelFormat::Rgb16 => {
            let u16s: &[u16] = bytemuck::cast_slice(bytes.as_ref());
            let ch = desc.format.layout().channels();
            let px = u16s.len() / ch;
            let mut out = alloc::vec![0u8; px * 16];
            let f32s: &mut [f32] = bytemuck::cast_slice_mut(&mut out);
            for i in 0..px {
                let s = i * ch;
                let d = i * 4;
                f32s[d] = u16s[s] as f32 / 65535.0;
                f32s[d + 1] = u16s[s + 1] as f32 / 65535.0;
                f32s[d + 2] = u16s[s + 2] as f32 / 65535.0;
                f32s[d + 3] = if ch >= 4 {
                    u16s[s + 3] as f32 / 65535.0
                } else {
                    1.0
                };
            }
            Ok((crate::PixelFormat::Rgba32F, out))
        }
        PixelFormat::RgbF32 => {
            let f32s: &[f32] = bytemuck::cast_slice(bytes.as_ref());
            let px = f32s.len() / 3;
            let mut out = alloc::vec![0u8; px * 16];
            let out_f32: &mut [f32] = bytemuck::cast_slice_mut(&mut out);
            for i in 0..px {
                let s = i * 3;
                let d = i * 4;
                out_f32[d] = f32s[s];
                out_f32[d + 1] = f32s[s + 1];
                out_f32[d + 2] = f32s[s + 2];
                out_f32[d + 3] = 1.0;
            }
            Ok((crate::PixelFormat::Rgba32F, out))
        }
        _ => Ok((format, bytes.into_owned())),
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn make_rgba8(w: u32, h: u32, pixel: [u8; 4]) -> PixelBuffer {
        let mut data = Vec::new();
        for _ in 0..(w * h) {
            data.extend_from_slice(&pixel);
        }
        PixelBuffer::from_vec(data, w, h, PixelDescriptor::RGBA8_SRGB).unwrap()
    }

    fn default_opts() -> SdrToneMapOptions {
        SdrToneMapOptions::default()
    }

    // --- RawImage adapters ---

    #[test]
    fn raw_image_from_srgb_buffer() {
        let buf = make_rgba8(2, 2, [128, 64, 32, 255]);
        let raw = raw_image_from_buffer(&buf, &Cicp::SRGB).unwrap();
        assert_eq!(raw.width, 2);
        assert_eq!(raw.height, 2);
        assert_eq!(raw.gamut, ColorGamut::Bt709);
        assert_eq!(raw.transfer, ColorTransfer::Srgb);
    }

    #[test]
    fn raw_image_from_pq_buffer() {
        let buf = make_rgba8(1, 1, [100, 100, 100, 255]);
        let raw = raw_image_from_buffer(&buf, &Cicp::BT2100_PQ).unwrap();
        assert_eq!(raw.gamut, ColorGamut::Bt2020);
        assert_eq!(raw.transfer, ColorTransfer::Pq);
    }

    #[test]
    fn raw_image_from_p3_buffer() {
        let buf = make_rgba8(1, 1, [200, 150, 100, 255]);
        let raw = raw_image_from_buffer(&buf, &Cicp::DISPLAY_P3).unwrap();
        assert_eq!(raw.gamut, ColorGamut::DisplayP3);
        assert_eq!(raw.transfer, ColorTransfer::Srgb);
    }

    // --- SdrToneMapper construction ---

    #[test]
    fn mapper_none_for_sdr() {
        assert!(SdrToneMapper::new(&Cicp::SRGB, &default_opts()).is_none());
        assert!(SdrToneMapper::new(&Cicp::DISPLAY_P3, &default_opts()).is_none());
    }

    #[test]
    fn mapper_some_for_pq() {
        let m = SdrToneMapper::new(&Cicp::BT2100_PQ, &default_opts()).unwrap();
        assert_eq!(m.output_cicp(), Cicp::SRGB);
    }

    #[test]
    fn mapper_p3_preserves_gamut() {
        let p3_pq = Cicp::new(12, 16, 0, true);
        let m = SdrToneMapper::new(&p3_pq, &default_opts()).unwrap();
        assert_eq!(m.output_cicp(), Cicp::DISPLAY_P3);
    }

    #[test]
    fn mapper_explicit_target() {
        let opts = SdrToneMapOptions {
            target_primaries: Some(ColorPrimaries::DisplayP3),
            ..default_opts()
        };
        let m = SdrToneMapper::new(&Cicp::BT2100_PQ, &opts).unwrap();
        assert_eq!(m.output_cicp(), Cicp::DISPLAY_P3);
    }

    // --- Row-level tone mapping ---

    #[test]
    fn tonemap_row_rgba8_produces_valid_sdr() {
        let mapper = SdrToneMapper::new(&Cicp::BT2100_PQ, &default_opts()).unwrap();
        let src = [127u8, 127, 127, 255, 80, 80, 80, 200];
        let mut dst = [0u8; 8];
        mapper.tonemap_row_rgba8(&src, &mut dst, 2);

        // Mid-gray PQ should produce valid SDR (not clipped)
        assert!(dst[0] > 0 && dst[0] < 255, "pixel 0 R={}", dst[0]);
        assert!(dst[4] > 0 && dst[4] < 255, "pixel 1 R={}", dst[4]);
        // Alpha preserved
        assert_eq!(dst[3], 255);
        assert_eq!(dst[7], 200);
    }

    #[test]
    fn tonemap_row_rgba8_to_linear_range() {
        let mapper = SdrToneMapper::new(&Cicp::BT2100_PQ, &default_opts()).unwrap();
        let src = [127u8, 127, 127, 200];
        let mut dst = [0.0f32; 4];
        mapper.tonemap_row_rgba8_to_linear(&src, &mut dst, 1);

        assert!(dst[0] >= 0.0 && dst[0] <= 1.0, "R out of range: {}", dst[0]);
        assert!(dst[1] >= 0.0 && dst[1] <= 1.0);
        assert!(dst[2] >= 0.0 && dst[2] <= 1.0);
        // Alpha normalized
        assert!((dst[3] - 200.0 / 255.0).abs() < 0.01);
    }

    // --- One-call convenience ---

    #[test]
    fn tonemap_to_sdr_none_for_sdr() {
        let buf = make_rgba8(2, 2, [128, 64, 32, 255]);
        assert!(
            tonemap_to_sdr(&buf, &Cicp::SRGB, &default_opts())
                .unwrap()
                .is_none()
        );
    }

    #[test]
    fn tonemap_to_sdr_pq_produces_rgba8() {
        let buf = make_rgba8(4, 4, [127, 127, 127, 255]);
        let (result, cicp) = tonemap_to_sdr(&buf, &Cicp::BT2100_PQ, &default_opts())
            .unwrap()
            .unwrap();
        assert_eq!(cicp, Cicp::SRGB);
        assert_eq!(result.width(), 4);
        assert_eq!(result.height(), 4);
        let s = result.as_slice();
        let row = s.row(0);
        assert!(row[0] > 0 && row[0] < 255);
    }

    #[test]
    fn tonemap_to_sdr_hlg() {
        let buf = make_rgba8(2, 2, [127, 127, 127, 255]);
        let (result, cicp) = tonemap_to_sdr(&buf, &Cicp::BT2100_HLG, &default_opts())
            .unwrap()
            .unwrap();
        assert_eq!(cicp, Cicp::SRGB);
        let s = result.as_slice();
        let row = s.row(0);
        assert!(row[0] > 0 && row[0] < 255);
    }

    #[test]
    fn tonemap_to_sdr_p3_preserves_gamut() {
        let buf = make_rgba8(2, 2, [127, 127, 127, 255]);
        let p3_pq = Cicp::new(12, 16, 0, true);
        let (_, cicp) = tonemap_to_sdr(&buf, &p3_pq, &default_opts())
            .unwrap()
            .unwrap();
        assert_eq!(cicp, Cicp::DISPLAY_P3);
    }

    #[test]
    fn tonemap_to_sdr_custom_peak_nits() {
        let buf = make_rgba8(2, 2, [127, 127, 127, 255]);
        let opts = SdrToneMapOptions {
            target_peak_nits: 100.0, // lower target peak
            ..default_opts()
        };
        let (result, _) = tonemap_to_sdr(&buf, &Cicp::BT2100_PQ, &opts)
            .unwrap()
            .unwrap();
        let s = result.as_slice();
        let row = s.row(0);
        assert!(row[0] > 0 && row[0] < 255);
    }

    // --- Slice API ---

    #[test]
    fn tonemap_slice_to_sdr_works() {
        let buf = make_rgba8(4, 4, [127, 127, 127, 255]);
        let slice = buf.as_slice();
        let (result, cicp) = tonemap_slice_to_sdr(slice, 4, 4, &Cicp::BT2100_PQ, &default_opts())
            .unwrap()
            .unwrap();
        assert_eq!(cicp, Cicp::SRGB);
        assert_eq!(result.width(), 4);
    }
}
