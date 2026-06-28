//! Gain map application for HDR reconstruction.

use alloc::boxed::Box;
use alloc::vec;

use crate::color::transfer::{srgb_eotf, srgb_oetf};
use crate::types::{
    Error, GainMap, GainMapMetadata, PixelBuffer, PixelFormat, PixelSlice, Result,
    TransferFunction, new_pixel_buffer, validate_gainmap_magnitude,
};
use enough::Stop;
use whereat::at;

/// Defense-in-depth clamp for an LUT-decoded gain multiplier.
///
/// Mirrors the bound in [`crate::limits::MAX_LOG_GAIN_MAGNITUDE`]:
/// `exp(±30 * ln(2)) ≈ 2^±30 ≈ [9.3e-10, 1.07e9]`. Anything outside
/// this range or non-finite is replaced with a neutral `1.0` so the
/// [`HdrOutputFormat::LinearFloat`] / [`HdrOutputFormat::LinearF16`]
/// arms cannot emit `+inf` / `NaN` pixels even if a caller skipped
/// metadata validation.
#[inline]
fn clamp_gain_finite(gain: f32) -> f32 {
    const MAX_GAIN: f32 = 1.073_741_8e9; // 2^30
    const MIN_GAIN: f32 = 1.0 / MAX_GAIN;
    if !gain.is_finite() {
        1.0
    } else {
        gain.clamp(MIN_GAIN, MAX_GAIN)
    }
}

/// Precomputed lookup table for gain map decoding.
///
/// This LUT eliminates expensive `powf()` and `exp()` calls per pixel by
/// precomputing the mapping from 8-bit gain map values to linear gain multipliers.
/// Provides ~10x speedup for `apply_gainmap`.
#[derive(Debug)]
pub struct GainMapLut {
    /// 256 entries per channel (R, G, B), mapping byte value to linear gain.
    /// Layout: [R0..R255, G0..G255, B0..B255]
    table: Box<[f32; 256 * 3]>,
}

impl GainMapLut {
    /// Create a new gain map LUT for the given metadata and display boost.
    ///
    /// The `weight` parameter is typically calculated from `display_boost` and
    /// the metadata's `base_hdr_headroom`/`alternate_hdr_headroom`.
    pub fn new(metadata: &GainMapMetadata, weight: f32) -> Self {
        let mut table = Box::new([0.0f32; 256 * 3]);

        for channel in 0..3 {
            let gamma = metadata.channels[channel].gamma as f32;
            // Convert log2 domain to natural log for exp() math
            let ln2 = core::f64::consts::LN_2;
            let log_min = (metadata.channels[channel].min * ln2) as f32;
            let log_max = (metadata.channels[channel].max * ln2) as f32;
            let log_range = log_max - log_min;

            for i in 0..256 {
                // Convert byte to normalized [0,1]
                let normalized = i as f32 / 255.0;

                // Undo gamma
                let linear = if gamma != 1.0 && gamma > 0.0 {
                    normalized.powf(1.0 / gamma)
                } else {
                    normalized
                };

                // Convert from normalized to log gain, apply weight, convert to linear.
                // Clamp to a finite, sane range so non-validated metadata cannot
                // smuggle `+inf` / `NaN` into the HDR output buffer
                // (see `clamp_gain_finite` for the bound and rationale).
                let log_gain = log_min + linear * log_range;
                let gain = clamp_gain_finite((log_gain * weight).exp());

                table[channel * 256 + i] = gain;
            }
        }

        Self { table }
    }

    /// Look up the gain multiplier for a single channel.
    #[inline(always)]
    pub fn lookup(&self, byte_value: u8, channel: usize) -> f32 {
        // Safety: channel is always 0..3 and byte_value is u8 (0..255)
        debug_assert!(channel < 3);
        self.table[channel * 256 + byte_value as usize]
    }

    /// Look up gain multipliers for all 3 channels from a single byte (luminance mode).
    #[inline(always)]
    pub fn lookup_luminance(&self, byte_value: u8) -> [f32; 3] {
        let g = self.table[byte_value as usize]; // Channel 0
        [g, g, g]
    }

    /// Look up gain multipliers for RGB from 3 bytes.
    #[inline(always)]
    pub fn lookup_rgb(&self, r: u8, g: u8, b: u8) -> [f32; 3] {
        [
            self.table[r as usize],
            self.table[256 + g as usize],
            self.table[512 + b as usize],
        ]
    }
}

/// Output format for HDR reconstruction.
///
/// Mirrors libultrahdr's three supported decode outputs:
/// - [`LinearFloat`](Self::LinearFloat) ↔ `UHDR_IMG_FMT_64bppRGBAHalfFloat`
///   semantically (same linear-light content), but at f32 precision instead
///   of f16. Use when downstream wants float math.
/// - [`LinearF16`](Self::LinearF16) ↔ `UHDR_IMG_FMT_64bppRGBAHalfFloat` exactly.
///   Use for direct compositor / GPU-texture handoff.
/// - [`Srgb8`](Self::Srgb8) ↔ `UHDR_IMG_FMT_32bppRGBA8888` with sRGB transfer.
///   Use when downstream wants SDR (display_boost = 1.0 typical).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum HdrOutputFormat {
    /// Linear f32 RGBA where 1.0 = SDR white (203 nits). Range `[0, ~display_boost]`.
    /// 16 bytes/pixel (`RgbaF32`).
    LinearFloat,
    /// Linear f16 (IEEE 754 half-precision) RGBA where 1.0 = SDR white.
    /// 8 bytes/pixel (`RgbaF16`). Mirrors libultrahdr's
    /// `UHDR_IMG_FMT_64bppRGBAHalfFloat`.
    ///
    /// Requires the `f16` Cargo feature (default-off). Without it this
    /// variant is not compiled and f16 decode output is unavailable.
    #[cfg(feature = "f16")]
    LinearF16,
    /// sRGB 8-bit (SDR output, no HDR boost). 4 bytes/pixel (`Rgba8`).
    Srgb8,
}

/// Apply a gain map to an SDR image to reconstruct HDR.
///
/// The `display_boost` parameter controls how much HDR effect to apply:
/// - 1.0 = SDR output (no boost)
/// - 2.0 = 2x brightness capability
/// - 4.0 = 4x brightness capability (typical HDR display)
///
/// The `stop` parameter enables cooperative cancellation. Pass `Unstoppable`
/// when cancellation is not needed.
pub fn apply_gainmap(
    sdr: &PixelBuffer,
    gainmap: &GainMap,
    metadata: &GainMapMetadata,
    display_boost: f32,
    output_format: HdrOutputFormat,
    stop: impl Stop,
) -> Result<PixelBuffer> {
    let sdr_slice = sdr.as_slice();
    apply_gainmap_slice(
        sdr_slice,
        gainmap,
        metadata,
        display_boost,
        output_format,
        stop,
    )
}

/// [`apply_gainmap`] variant that takes a borrowed [`PixelSlice`] directly.
///
/// Useful when the caller already has a slice view over pixel bytes (e.g.
/// from a cropped region or a foreign allocation) and doesn't want to copy
/// into a [`PixelBuffer`] first.
pub fn apply_gainmap_slice(
    sdr: PixelSlice<'_>,
    gainmap: &GainMap,
    metadata: &GainMapMetadata,
    display_boost: f32,
    output_format: HdrOutputFormat,
    stop: impl Stop,
) -> Result<PixelBuffer> {
    crate::types::validate_ultrahdr_slice(&sdr)?;
    gainmap.validate()?;
    validate_gainmap_magnitude(metadata)?;

    let width = sdr.width();
    let height = sdr.rows();
    let sdr_primaries = sdr.descriptor().primaries;

    // Calculate weight factor based on display capability
    let weight = calculate_weight(display_boost, metadata);

    // Create precomputed LUT for fast gain decoding
    let lut = GainMapLut::new(metadata, weight);

    // Build the Shepard's weight table if image-to-gainmap ratio is integer
    // (the common case — ISO 21496-1 maps are typically 1/2, 1/4, 1/8, 1/16).
    // `None` means we fall back to per-pixel sqrt with row-hoisted constants.
    let shepards = ShepardsLut::try_new(width, height, gainmap.width, gainmap.height);

    // Create output image
    let mut output = match output_format {
        HdrOutputFormat::LinearFloat => new_pixel_buffer(
            width,
            height,
            PixelFormat::RgbaF32,
            sdr_primaries,
            TransferFunction::Linear,
        )?,
        #[cfg(feature = "f16")]
        HdrOutputFormat::LinearF16 => new_pixel_buffer(
            width,
            height,
            PixelFormat::RgbaF16,
            sdr_primaries,
            TransferFunction::Linear,
        )?,
        HdrOutputFormat::Srgb8 => new_pixel_buffer(
            width,
            height,
            PixelFormat::Rgba8,
            sdr_primaries,
            TransferFunction::Srgb,
        )?,
    };

    // Row-reusable scratch buffers
    let row_pixels = width as usize;
    let mut sdr_row = vec![[0.0f32; 3]; row_pixels];
    let mut gains_row = vec![[0.0f32; 3]; row_pixels];
    let mut hdr_row = vec![[0.0f32; 3]; row_pixels];

    // Pre-broadcast metadata offsets into `[f32; 3]` arrays once per image.
    let base_offset = [
        metadata.channels[0].base_offset as f32,
        metadata.channels[1].base_offset as f32,
        metadata.channels[2].base_offset as f32,
    ];
    let alternate_offset = [
        metadata.channels[0].alternate_offset as f32,
        metadata.channels[1].alternate_offset as f32,
        metadata.channels[2].alternate_offset as f32,
    ];

    let out_stride = output.stride();
    let out_format = output.descriptor().pixel_format();
    let mut out_slice = output.as_slice_mut();
    let out_data = out_slice.as_strided_bytes_mut();

    // Process each row, checking for cancellation periodically
    for y in 0..height {
        // Check for cancellation once per row (not per pixel for performance)
        stop.check().map_err(|r| at!(Error::Stopped(r)))?;

        read_sdr_row_linear(&sdr, y, &mut sdr_row);
        sample_gainmap_row_lut(
            gainmap,
            &lut,
            shepards.as_ref(),
            y,
            width,
            height,
            &mut gains_row,
        );
        super::apply_simd::apply_gain_row_presampled(
            &sdr_row,
            &gains_row,
            base_offset,
            alternate_offset,
            &mut hdr_row,
        );
        write_hdr_row(out_data, out_stride, out_format, y, &hdr_row, output_format);
    }

    drop(out_slice);

    // Tag the reconstruction's absolute-luminance anchor. Gain-map HDR is
    // reconstructed relative to SDR diffuse white = BT.2408 (203 cd/m²), so a
    // relative-linear `1.0` maps to 203 nits. Carrying it on the buffer's
    // `ColorContext` lets a downstream PQ/HLG encode read the real anchor
    // instead of assuming the default — every codec that reconstructs through
    // `apply_gainmap` (heic/zenjpeg/zenavif/zenjxl) inherits this. Linear
    // outputs only: the `Srgb8` path is gamma-encoded SDR, not relative-linear.
    let is_linear = match output_format {
        HdrOutputFormat::LinearFloat => true,
        #[cfg(feature = "f16")]
        HdrOutputFormat::LinearF16 => true,
        HdrOutputFormat::Srgb8 => false,
    };
    let output = if is_linear {
        output.with_color_context(alloc::sync::Arc::new(
            zenpixels::ColorContext::default().with_diffuse_white(zenpixels::DiffuseWhite::BT2408),
        ))
    } else {
        output
    };
    Ok(output)
}

/// Read a row of the SDR image into linear f32 RGB.
///
/// `out.len()` must equal `sdr.width() as usize`. Supports the pixel formats
/// that the per-pixel `get_sdr_linear` supports — other formats yield
/// `[0, 0, 0]` per-pixel as a fallback.
fn read_sdr_row_linear(sdr: &PixelSlice<'_>, y: u32, out: &mut [[f32; 3]]) {
    debug_assert_eq!(out.len(), sdr.width() as usize);
    for (x, pixel) in out.iter_mut().enumerate() {
        *pixel = get_sdr_linear(sdr, x as u32, y);
    }
}

/// Sample a full row of gains from the gain map (Shepard's IDW, LUT-accelerated).
///
/// `out.len()` must equal `img_width as usize`. For single-channel gain maps,
/// the same gain is broadcast to R/G/B.
///
/// `shepards` MUST be `Some(_)` when `img_width % gainmap.width == 0` and
/// `img_height % gainmap.height == 0`; the caller builds it once via
/// [`ShepardsLut::try_new`] and passes it to every row. When `None`, the
/// per-pixel sqrt fallback runs (still computes weights once per pixel and
/// shares them across channels).
pub(crate) fn sample_gainmap_row_lut(
    gainmap: &GainMap,
    lut: &GainMapLut,
    shepards: Option<&ShepardsLut>,
    y: u32,
    img_width: u32,
    img_height: u32,
    out: &mut [[f32; 3]],
) {
    debug_assert_eq!(out.len(), img_width as usize);
    match shepards {
        Some(shep) => sample_row_lut_int(gainmap, lut, shep, y, out),
        None => sample_row_lut_float(gainmap, lut, y, img_width, img_height, out),
    }
}

/// Write a row of HDR pixels to the output image in the requested format.
///
/// `out_data` must be the strided-byte buffer of the output image; `out_stride`
/// is its row stride; `out_format` is the format (Rgba8 or RgbaF32).
fn write_hdr_row(
    out_data: &mut [u8],
    out_stride: usize,
    out_format: PixelFormat,
    y: u32,
    hdr_row: &[[f32; 3]],
    format: HdrOutputFormat,
) {
    let row_start = (y as usize) * out_stride;
    match format {
        HdrOutputFormat::LinearFloat => {
            debug_assert_eq!(out_format, PixelFormat::RgbaF32);
            for (x, &hdr) in hdr_row.iter().enumerate() {
                let idx = row_start + x * 16;
                out_data[idx..idx + 4].copy_from_slice(&hdr[0].to_le_bytes());
                out_data[idx + 4..idx + 8].copy_from_slice(&hdr[1].to_le_bytes());
                out_data[idx + 8..idx + 12].copy_from_slice(&hdr[2].to_le_bytes());
                out_data[idx + 12..idx + 16].copy_from_slice(&1.0f32.to_le_bytes());
            }
        }
        #[cfg(feature = "f16")]
        HdrOutputFormat::LinearF16 => {
            debug_assert_eq!(out_format, PixelFormat::RgbaF16);
            // 8 bytes/pixel: 4 channels × f16 (2 bytes). Alpha is 1.0 (constant).
            const F16_ONE: u16 = 0x3C00; // half::f16::ONE.to_bits()
            for (x, &hdr) in hdr_row.iter().enumerate() {
                let idx = row_start + x * 8;
                let r = half::f16::from_f32(hdr[0]).to_bits().to_le_bytes();
                let g = half::f16::from_f32(hdr[1]).to_bits().to_le_bytes();
                let b = half::f16::from_f32(hdr[2]).to_bits().to_le_bytes();
                let a = F16_ONE.to_le_bytes();
                out_data[idx..idx + 2].copy_from_slice(&r);
                out_data[idx + 2..idx + 4].copy_from_slice(&g);
                out_data[idx + 4..idx + 6].copy_from_slice(&b);
                out_data[idx + 6..idx + 8].copy_from_slice(&a);
            }
        }
        HdrOutputFormat::Srgb8 => {
            debug_assert_eq!(out_format, PixelFormat::Rgba8);
            for (x, &hdr) in hdr_row.iter().enumerate() {
                let r = srgb_oetf(hdr[0].clamp(0.0, 1.0));
                let g = srgb_oetf(hdr[1].clamp(0.0, 1.0));
                let b = srgb_oetf(hdr[2].clamp(0.0, 1.0));
                let idx = row_start + x * 4;
                out_data[idx] = (r * 255.0).round() as u8;
                out_data[idx + 1] = (g * 255.0).round() as u8;
                out_data[idx + 2] = (b * 255.0).round() as u8;
                out_data[idx + 3] = 255;
            }
        }
    }
}

/// Calculate the weight factor for gain map application.
///
/// Headroom values are in log2 domain. `display_boost` is linear.
///
/// Mirrors `avifGetGainMapWeight` in libavif and the equivalent in
/// libultrahdr. The output is `clamp((log2(display_boost) - base) /
/// (alt - base), 0, 1)` where `base` and `alt` are the metadata's
/// HDR-headroom log2 values.
pub fn calculate_weight(display_boost: f32, metadata: &GainMapMetadata) -> f32 {
    let log_display = display_boost.max(1.0).log2() as f64;
    let log_min = metadata.base_hdr_headroom.max(0.0);
    let log_max = metadata.alternate_hdr_headroom.max(0.0);

    if log_max <= log_min {
        return 1.0;
    }

    ((log_display - log_min) / (log_max - log_min)).clamp(0.0, 1.0) as f32
}

/// Get linear RGB from SDR image.
fn get_sdr_linear(sdr: &PixelSlice<'_>, x: u32, y: u32) -> [f32; 3] {
    let format = sdr.descriptor().pixel_format();
    let stride = sdr.stride();
    let data = sdr.as_strided_bytes();
    match format {
        PixelFormat::Rgba8 | PixelFormat::Rgb8 => {
            let bpp = if format == PixelFormat::Rgba8 { 4 } else { 3 };
            let idx = (y as usize) * stride + (x as usize) * bpp;
            let r = data[idx] as f32 / 255.0;
            let g = data[idx + 1] as f32 / 255.0;
            let b = data[idx + 2] as f32 / 255.0;
            [srgb_eotf(r), srgb_eotf(g), srgb_eotf(b)]
        }
        PixelFormat::RgbaF32 => {
            let idx = (y as usize) * stride + (x as usize) * 16;
            let r = f32::from_le_bytes([data[idx], data[idx + 1], data[idx + 2], data[idx + 3]]);
            let g =
                f32::from_le_bytes([data[idx + 4], data[idx + 5], data[idx + 6], data[idx + 7]]);
            let b =
                f32::from_le_bytes([data[idx + 8], data[idx + 9], data[idx + 10], data[idx + 11]]);
            [r, g, b]
        }
        #[cfg(feature = "f16")]
        PixelFormat::RgbaF16 | PixelFormat::RgbF16 => {
            let bpp = if format == PixelFormat::RgbaF16 { 8 } else { 6 };
            let idx = (y as usize) * stride + (x as usize) * bpp;
            let r = half::f16::from_le_bytes([data[idx], data[idx + 1]]).to_f32();
            let g = half::f16::from_le_bytes([data[idx + 2], data[idx + 3]]).to_f32();
            let b = half::f16::from_le_bytes([data[idx + 4], data[idx + 5]]).to_f32();
            [r, g, b]
        }
        PixelFormat::Gray8 => {
            let idx = (y as usize) * stride + (x as usize);
            let v = data[idx] as f32 / 255.0;
            [v, v, v]
        }
        _ => [0.0, 0.0, 0.0],
    }
}

/// Precomputed Shepard's IDW weight tables for integer-scale gain map upsample.
///
/// Mirrors libultrahdr's `ShepardsIDW` struct (see `gainmapmath.h:228`,
/// `gainmapmath.cpp:49`). Per-pixel sample-time work drops from "4 sqrt
/// plus 4 div" to "4 mul plus 3 add" by precomputing weights for every
/// distinct sub-pixel position in the unit cell. Holds four tables
/// (interior, no-right edge, no-bottom edge, corner) to handle gain-map
/// boundary clamping without per-pixel branches on bounds.
///
/// Only valid when image dimensions are an integer multiple of gain-map
/// dimensions. Storage is `4 * scale_x * scale_y * 4` floats (≤ 16 KB
/// for any sane scale).
#[derive(Debug)]
pub struct ShepardsLut {
    scale_x: u32,
    scale_y: u32,
    /// Indexed `[oy * scale_x + ox] * 4 + corner` where corner is
    /// 0=TL, 1=BL, 2=TR, 3=BR. Same memory layout in all four tables.
    full: Box<[f32]>,
    no_right: Box<[f32]>,
    no_bottom: Box<[f32]>,
    corner: Box<[f32]>,
}

impl ShepardsLut {
    /// Build weight tables for an arbitrary integer scale (`scale_x`, `scale_y`).
    /// Most callers should use [`Self::try_new`] which infers the scale from
    /// image and gain-map dimensions.
    ///
    /// Returns `None` when `scale_x * scale_y` exceeds
    /// [`crate::limits::MAX_SHEPARDS_TABLE_ENTRIES`] (the LUT would not fit
    /// in a sane allocation) or when `checked_mul` overflows. Callers that
    /// hit this case must fall back to the per-pixel float path.
    ///
    /// Also returns `None` when either scale is zero.
    pub fn new(scale_x: u32, scale_y: u32) -> Option<Self> {
        if scale_x == 0 || scale_y == 0 {
            return None;
        }
        let entries = scale_x.checked_mul(scale_y)?;
        if entries > crate::limits::MAX_SHEPARDS_TABLE_ENTRIES {
            return None;
        }
        // `entries * 4` cannot overflow u32 once `entries <= 4096`.
        let n = entries.checked_mul(4)? as usize;
        let mut full = vec![0.0f32; n].into_boxed_slice();
        let mut no_right = vec![0.0f32; n].into_boxed_slice();
        let mut no_bottom = vec![0.0f32; n].into_boxed_slice();
        let mut corner = vec![0.0f32; n].into_boxed_slice();
        fill_shepards(&mut full, scale_x, scale_y, 1, 1);
        fill_shepards(&mut no_right, scale_x, scale_y, 0, 1);
        fill_shepards(&mut no_bottom, scale_x, scale_y, 1, 0);
        fill_shepards(&mut corner, scale_x, scale_y, 0, 0);
        Some(Self {
            scale_x,
            scale_y,
            full,
            no_right,
            no_bottom,
            corner,
        })
    }

    /// Build a LUT iff image dims are an exact integer multiple of gain-map
    /// dims AND the resulting scale fits in
    /// [`crate::limits::MAX_SHEPARDS_TABLE_ENTRIES`]. `None` means the caller
    /// must take the per-pixel sqrt fallback.
    pub fn try_new(img_width: u32, img_height: u32, gm_width: u32, gm_height: u32) -> Option<Self> {
        if gm_width == 0 || gm_height == 0 || img_width == 0 || img_height == 0 {
            return None;
        }
        if !img_width.is_multiple_of(gm_width) || !img_height.is_multiple_of(gm_height) {
            return None;
        }
        let sx = img_width / gm_width;
        let sy = img_height / gm_height;
        Self::new(sx, sy)
    }

    #[inline(always)]
    fn pick(&self, no_right: bool, no_bottom: bool) -> &[f32] {
        match (no_right, no_bottom) {
            (false, false) => &self.full,
            (true, false) => &self.no_right,
            (false, true) => &self.no_bottom,
            (true, true) => &self.corner,
        }
    }
}

fn fill_shepards(weights: &mut [f32], sx: u32, sy: u32, inc_r: u32, inc_b: u32) {
    let sx_f = sx as f32;
    let sy_f = sy as f32;
    for y in 0..sy {
        for x in 0..sx {
            let pos_x = x as f32 / sx_f;
            let pos_y = y as f32 / sy_f;
            let next_x = inc_r as f32;
            let next_y = inc_b as f32;
            let idx = ((y * sx + x) * 4) as usize;
            let d_tl = (pos_x * pos_x + pos_y * pos_y).sqrt();
            if d_tl == 0.0 {
                weights[idx] = 1.0;
                weights[idx + 1] = 0.0;
                weights[idx + 2] = 0.0;
                weights[idx + 3] = 0.0;
                continue;
            }
            let dy_b = pos_y - next_y;
            let dx_r = pos_x - next_x;
            let d_bl = (pos_x * pos_x + dy_b * dy_b).sqrt();
            let d_tr = (dx_r * dx_r + pos_y * pos_y).sqrt();
            let d_br = (dx_r * dx_r + dy_b * dy_b).sqrt();
            let w_tl = 1.0 / d_tl;
            let w_bl = 1.0 / d_bl;
            let w_tr = 1.0 / d_tr;
            let w_br = 1.0 / d_br;
            let inv_total = 1.0 / (w_tl + w_bl + w_tr + w_br);
            weights[idx] = w_tl * inv_total;
            weights[idx + 1] = w_bl * inv_total;
            weights[idx + 2] = w_tr * inv_total;
            weights[idx + 3] = w_br * inv_total;
        }
    }
}

/// Shepard's IDW on 4 corners with weights computed in-place.
///
/// Used as the per-pixel fallback when the image-to-gain-map ratio is
/// non-integer (so the precomputed LUT can't be used). Weights are
/// computed once and reused across channels by callers.
#[inline(always)]
fn shepards_weights(fx: f32, fy: f32) -> [f32; 4] {
    let dx_r = 1.0 - fx;
    let dy_b = 1.0 - fy;
    let d_tl = (fx * fx + fy * fy).sqrt();
    if d_tl == 0.0 {
        return [1.0, 0.0, 0.0, 0.0];
    }
    let d_bl = (fx * fx + dy_b * dy_b).sqrt();
    if d_bl == 0.0 {
        return [0.0, 1.0, 0.0, 0.0];
    }
    let d_tr = (dx_r * dx_r + fy * fy).sqrt();
    if d_tr == 0.0 {
        return [0.0, 0.0, 1.0, 0.0];
    }
    let d_br = (dx_r * dx_r + dy_b * dy_b).sqrt();
    if d_br == 0.0 {
        return [0.0, 0.0, 0.0, 1.0];
    }
    let w_tl = 1.0 / d_tl;
    let w_bl = 1.0 / d_bl;
    let w_tr = 1.0 / d_tr;
    let w_br = 1.0 / d_br;
    let inv_total = 1.0 / (w_tl + w_bl + w_tr + w_br);
    [
        w_tl * inv_total,
        w_bl * inv_total,
        w_tr * inv_total,
        w_br * inv_total,
    ]
}

#[inline(always)]
fn dot4(c: [f32; 4], w: [f32; 4]) -> f32 {
    c[0] * w[0] + c[1] * w[1] + c[2] * w[2] + c[3] * w[3]
}

/// Fast integer-scale row sampler. Picks weights from `shepards`
/// (no per-pixel sqrt/div) and shares them across channels.
fn sample_row_lut_int(
    gainmap: &GainMap,
    lut: &GainMapLut,
    shepards: &ShepardsLut,
    y: u32,
    out: &mut [[f32; 3]],
) {
    let sx = shepards.scale_x;
    let sy = shepards.scale_y;
    let gw = gainmap.width;
    let gh = gainmap.height;
    debug_assert!(gw > 0 && gh > 0);

    // Row-constant pieces: y0/y1 = enclosing gainmap rows; oy = sub-pixel
    // offset; no_bottom = row sits at the gainmap's bottom edge so y1 was
    // clamped back to y0.
    let y0 = (y / sy).min(gh - 1);
    let y1 = (y0 + 1).min(gh - 1);
    let oy = y % sy;
    let no_bottom = y0 == y1;

    let row0_off = (y0 * gw) as usize;
    let row1_off = (y1 * gw) as usize;

    if gainmap.channels == 1 {
        for (x_out, gain) in out.iter_mut().enumerate() {
            let x = x_out as u32;
            let x0 = (x / sx).min(gw - 1);
            let x1 = (x0 + 1).min(gw - 1);
            let ox = x % sx;
            let no_right = x0 == x1;

            let table = shepards.pick(no_right, no_bottom);
            let base = ((oy * sx + ox) * 4) as usize;
            let w = [
                table[base],
                table[base + 1],
                table[base + 2],
                table[base + 3],
            ];

            let g_tl = lut.lookup(gainmap.data[row0_off + x0 as usize], 0);
            let g_bl = lut.lookup(gainmap.data[row1_off + x0 as usize], 0);
            let g_tr = lut.lookup(gainmap.data[row0_off + x1 as usize], 0);
            let g_br = lut.lookup(gainmap.data[row1_off + x1 as usize], 0);
            let g = dot4([g_tl, g_bl, g_tr, g_br], w);
            *gain = [g, g, g];
        }
    } else {
        for (x_out, gain) in out.iter_mut().enumerate() {
            let x = x_out as u32;
            let x0 = (x / sx).min(gw - 1);
            let x1 = (x0 + 1).min(gw - 1);
            let ox = x % sx;
            let no_right = x0 == x1;

            let table = shepards.pick(no_right, no_bottom);
            let base = ((oy * sx + ox) * 4) as usize;
            let w = [
                table[base],
                table[base + 1],
                table[base + 2],
                table[base + 3],
            ];

            let tl = (row0_off + x0 as usize) * 3;
            let bl = (row1_off + x0 as usize) * 3;
            let tr = (row0_off + x1 as usize) * 3;
            let br = (row1_off + x1 as usize) * 3;
            for (c, dst) in gain.iter_mut().enumerate() {
                let corners = [
                    lut.lookup(gainmap.data[tl + c], c),
                    lut.lookup(gainmap.data[bl + c], c),
                    lut.lookup(gainmap.data[tr + c], c),
                    lut.lookup(gainmap.data[br + c], c),
                ];
                *dst = dot4(corners, w);
            }
        }
    }
}

/// Non-integer scale fallback. Computes weights once per pixel (4 sqrt
/// plus 1 reciprocal-divide) and shares them across channels. Hoists
/// `gm_y`/`y0`/`y1`/`fy` to row constants.
fn sample_row_lut_float(
    gainmap: &GainMap,
    lut: &GainMapLut,
    y: u32,
    img_width: u32,
    img_height: u32,
    out: &mut [[f32; 3]],
) {
    let gw = gainmap.width;
    let gh = gainmap.height;
    debug_assert!(gw > 0 && gh > 0);
    debug_assert!(img_width > 0 && img_height > 0);

    let inv_iw = 1.0 / img_width as f32;
    let inv_ih = 1.0 / img_height as f32;
    let gw_f = gw as f32;
    let gh_f = gh as f32;

    let gm_y = (y as f32 * inv_ih) * gh_f;
    let gm_y_floor = gm_y.floor();
    let y0 = (gm_y_floor as u32).min(gh - 1);
    let y1 = (y0 + 1).min(gh - 1);
    let fy = gm_y - gm_y_floor;
    let row0_off = (y0 * gw) as usize;
    let row1_off = (y1 * gw) as usize;

    if gainmap.channels == 1 {
        for (x_out, gain) in out.iter_mut().enumerate() {
            let gm_x = (x_out as f32 * inv_iw) * gw_f;
            let gm_x_floor = gm_x.floor();
            let x0 = (gm_x_floor as u32).min(gw - 1);
            let x1 = (x0 + 1).min(gw - 1);
            let fx = gm_x - gm_x_floor;
            let w = shepards_weights(fx, fy);

            let g_tl = lut.lookup(gainmap.data[row0_off + x0 as usize], 0);
            let g_bl = lut.lookup(gainmap.data[row1_off + x0 as usize], 0);
            let g_tr = lut.lookup(gainmap.data[row0_off + x1 as usize], 0);
            let g_br = lut.lookup(gainmap.data[row1_off + x1 as usize], 0);
            let g = dot4([g_tl, g_bl, g_tr, g_br], w);
            *gain = [g, g, g];
        }
    } else {
        for (x_out, gain) in out.iter_mut().enumerate() {
            let gm_x = (x_out as f32 * inv_iw) * gw_f;
            let gm_x_floor = gm_x.floor();
            let x0 = (gm_x_floor as u32).min(gw - 1);
            let x1 = (x0 + 1).min(gw - 1);
            let fx = gm_x - gm_x_floor;
            let w = shepards_weights(fx, fy);

            let tl = (row0_off + x0 as usize) * 3;
            let bl = (row1_off + x0 as usize) * 3;
            let tr = (row0_off + x1 as usize) * 3;
            let br = (row1_off + x1 as usize) * 3;
            for (c, dst) in gain.iter_mut().enumerate() {
                let corners = [
                    lut.lookup(gainmap.data[tl + c], c),
                    lut.lookup(gainmap.data[bl + c], c),
                    lut.lookup(gainmap.data[tr + c], c),
                    lut.lookup(gainmap.data[br + c], c),
                ];
                *dst = dot4(corners, w);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Single-pixel convenience wrapper over `apply_gain_row_presampled` for tests.
    fn apply_gain_one(metadata: &GainMapMetadata, sdr: [f32; 3], gain: [f32; 3]) -> [f32; 3] {
        let base = [
            metadata.channels[0].base_offset as f32,
            metadata.channels[1].base_offset as f32,
            metadata.channels[2].base_offset as f32,
        ];
        let alt = [
            metadata.channels[0].alternate_offset as f32,
            metadata.channels[1].alternate_offset as f32,
            metadata.channels[2].alternate_offset as f32,
        ];
        let sdr_row = [sdr];
        let gains_row = [gain];
        let mut out_row = [[0.0f32; 3]];
        super::super::apply_simd::apply_gain_row_presampled(
            &sdr_row,
            &gains_row,
            base,
            alt,
            &mut out_row,
        );
        out_row[0]
    }
    use crate::types::ColorPrimaries;

    #[test]
    fn test_calculate_weight() {
        let mut metadata = GainMapMetadata::default();
        metadata.base_hdr_headroom = 0.0;
        metadata.alternate_hdr_headroom = 2.0;

        // No boost
        let w = calculate_weight(1.0, &metadata);
        assert!((w - 0.0).abs() < 0.01);

        // Full boost
        let w = calculate_weight(4.0, &metadata);
        assert!((w - 1.0).abs() < 0.01);

        // Half boost (log scale)
        let w = calculate_weight(2.0, &metadata);
        assert!(w > 0.4 && w < 0.6);
    }

    #[test]
    fn test_gain_map_lut() {
        let mut metadata = GainMapMetadata::default();
        for ch in &mut metadata.channels {
            ch.max = 2.0;
        }

        let lut = GainMapLut::new(&metadata, 1.0);

        // Min gain (byte 0 = normalized 0.0)
        let gain = lut.lookup(0, 0);
        assert!((gain - 1.0).abs() < 0.01, "min gain: {}", gain);

        // Max gain (byte 255 = normalized 1.0)
        let gain = lut.lookup(255, 0);
        assert!((gain - 4.0).abs() < 0.1, "max gain: {}", gain);

        // Mid gain should be between min and max
        let gain = lut.lookup(128, 0);
        assert!(gain > 1.5 && gain < 2.5, "mid gain: {}", gain);
    }

    #[test]
    fn test_apply_gainmap_basic() {
        // Create SDR image
        let mut sdr = crate::types::new_pixel_buffer(
            4,
            4,
            PixelFormat::Rgba8,
            ColorPrimaries::Bt709,
            TransferFunction::Srgb,
        )
        .unwrap();
        {
            let mut slice = sdr.as_slice_mut();
            let bytes = slice.as_strided_bytes_mut();
            for i in 0..bytes.len() / 4 {
                bytes[i * 4] = 128;
                bytes[i * 4 + 1] = 128;
                bytes[i * 4 + 2] = 128;
                bytes[i * 4 + 3] = 255;
            }
        }

        // Create gain map (all same boost)
        let mut gainmap = GainMap::new(2, 2).unwrap();
        for v in &mut gainmap.data {
            *v = 200; // High gain
        }

        let metadata = crate::types::metadata_from_arrays(
            [0.0; 3],
            [2.0; 3],
            [1.0; 3],
            [0.015625; 3],
            [0.015625; 3],
            0.0,
            2.0,
            true,
            false,
        );

        let result = apply_gainmap(
            &sdr,
            &gainmap,
            &metadata,
            4.0,
            HdrOutputFormat::Srgb8,
            enough::Unstoppable,
        )
        .unwrap();

        assert_eq!(result.width(), 4);
        assert_eq!(result.height(), 4);
        assert_eq!(result.descriptor().pixel_format(), PixelFormat::Rgba8);
    }

    /// Linear (HDR) reconstruction carries the SDR diffuse-white anchor
    /// (BT.2408 = 203 cd/m²) on its `ColorContext` so a downstream PQ/HLG
    /// encode reads the real anchor; the Srgb8 (SDR) path does not.
    #[test]
    fn apply_gainmap_tags_diffuse_white_on_linear_output() {
        let sdr = crate::types::new_pixel_buffer(
            2,
            2,
            PixelFormat::Rgba8,
            ColorPrimaries::Bt709,
            TransferFunction::Srgb,
        )
        .unwrap();
        let gainmap = GainMap::new(1, 1).unwrap();
        let metadata = crate::types::metadata_from_arrays(
            [0.0; 3],
            [2.0; 3],
            [1.0; 3],
            [0.015625; 3],
            [0.015625; 3],
            0.0,
            2.0,
            true,
            false,
        );

        let linear = apply_gainmap(
            &sdr,
            &gainmap,
            &metadata,
            4.0,
            HdrOutputFormat::LinearFloat,
            enough::Unstoppable,
        )
        .unwrap();
        assert_eq!(
            linear.color_context().and_then(|c| c.diffuse_white),
            Some(zenpixels::DiffuseWhite::BT2408),
            "linear reconstruction must carry the 203-nit anchor"
        );

        let srgb = apply_gainmap(
            &sdr,
            &gainmap,
            &metadata,
            4.0,
            HdrOutputFormat::Srgb8,
            enough::Unstoppable,
        )
        .unwrap();
        assert_eq!(
            srgb.color_context().and_then(|c| c.diffuse_white),
            None,
            "the gamma-encoded SDR path is not relative-linear; no anchor"
        );
    }

    // ========================================================================
    // Gain application reference values (C++ libultrahdr parity)
    //
    // Tests the LUT-based gain application against known-correct values.
    // The LUT maps byte values to linear gain multipliers:
    //   normalized = byte / 255.0
    //   linear = normalized^(1/gamma)  [undo gamma]
    //   log_gain = ln(min_boost) + linear * (ln(max_boost) - ln(min_boost))
    //   gain = exp(log_gain * weight)
    //
    // Then HDR = (sdr + offset_sdr) * gain - offset_hdr
    // ========================================================================

    /// Test gain application at 5 weight levels for white pixel.
    ///
    /// White (sdr=1.0) at gain map value 255 (max boost),
    /// with weight from 0.0 to 1.0 in steps of 0.25.
    #[test]
    fn test_gain_application_weight_levels() {
        let metadata = crate::types::metadata_from_arrays(
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

        let sdr_val = 1.0_f32; // White pixel (linear)
        let offset = 1.0_f32 / 64.0;
        let log_min = 1.0_f32.ln(); // 0.0
        let log_max = 4.0_f32.ln(); // ~1.386

        // At byte=255 (normalized=1.0, gamma=1.0 → linear=1.0):
        //   log_gain = 0.0 + 1.0 * (ln(4) - ln(1)) = ln(4) ≈ 1.386
        //   gain = exp(log_gain * weight)
        //   hdr = (sdr + offset) * gain - offset

        let weights: [(f32, &str); 5] = [
            (0.0, "SDR (no boost)"),
            (0.25, "25% boost"),
            (0.5, "50% boost"),
            (0.75, "75% boost"),
            (1.0, "full boost"),
        ];

        for &(weight, desc) in &weights {
            let lut = GainMapLut::new(&metadata, weight);
            let gain = lut.lookup(255, 0);

            let log_gain = log_min + 1.0 * (log_max - log_min);
            let expected_gain = (log_gain * weight).exp();
            let expected_hdr = (sdr_val + offset) * expected_gain - offset;

            // Verify LUT gain matches formula
            assert!(
                (gain - expected_gain).abs() < 0.01,
                "{}: LUT gain={}, expected={}",
                desc,
                gain,
                expected_gain
            );

            // Verify HDR output
            let hdr = apply_gain_one(&metadata, [sdr_val; 3], [gain; 3]);
            assert!(
                (hdr[0] - expected_hdr).abs() < 0.02,
                "{}: hdr={}, expected={}",
                desc,
                hdr[0],
                expected_hdr
            );
        }
    }

    /// Test gain application for black pixel (sdr=0.0).
    ///
    /// Black pixels should remain close to black regardless of gain,
    /// because the offset dominates: hdr = (0 + 1/64) * gain - 1/64
    #[test]
    fn test_gain_application_black_pixel() {
        let metadata = crate::types::metadata_from_arrays(
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

        let offset = 1.0_f32 / 64.0;

        // At full weight with max gain byte
        let lut = GainMapLut::new(&metadata, 1.0);
        let gain = lut.lookup(255, 0);

        // hdr = (0 + 1/64) * 4.0 - 1/64 = 4/64 - 1/64 = 3/64 ≈ 0.047
        let expected_hdr = offset * gain - offset;
        let hdr = apply_gain_one(&metadata, [0.0; 3], [gain; 3]);

        assert!(
            (hdr[0] - expected_hdr).abs() < 0.01,
            "Black pixel HDR: {} vs expected {}",
            hdr[0],
            expected_hdr
        );

        // Black with zero gain (byte=0) should stay near zero
        let gain_min = lut.lookup(0, 0);
        let hdr_min = apply_gain_one(&metadata, [0.0; 3], [gain_min; 3]);
        // gain_min = exp(0 * 1.0) = 1.0 for weight=1.0 and min_boost=1.0
        // hdr = (0 + 1/64) * 1.0 - 1/64 = 0
        assert!(
            hdr_min[0].abs() < 0.01,
            "Black at min gain should be ~0, got {}",
            hdr_min[0]
        );
    }

    /// Verify gain LUT covers the full [min_boost, max_boost] range.
    #[test]
    fn test_gain_lut_range_coverage() {
        let metadata = crate::types::metadata_from_arrays(
            [-1.0; 3],
            [3.0; 3],
            [1.0; 3],
            [1.0 / 64.0; 3],
            [1.0 / 64.0; 3],
            0.0,
            3.0,
            true,
            false,
        );

        let lut = GainMapLut::new(&metadata, 1.0);

        // Byte 0 → min gain = exp(ln(0.5)) = 0.5
        let gain_0 = lut.lookup(0, 0);
        assert!(
            (gain_0 - 0.5).abs() < 0.01,
            "Byte 0 should give min gain 0.5, got {}",
            gain_0
        );

        // Byte 255 → max gain = exp(ln(8)) = 8.0
        let gain_255 = lut.lookup(255, 0);
        assert!(
            (gain_255 - 8.0).abs() < 0.1,
            "Byte 255 should give max gain 8.0, got {}",
            gain_255
        );

        // Monotonically increasing
        for i in 1..=255u8 {
            let prev = lut.lookup(i - 1, 0);
            let curr = lut.lookup(i, 0);
            assert!(
                curr >= prev,
                "LUT not monotonic at byte {}: {} < {}",
                i,
                curr,
                prev
            );
        }
    }

    /// Helper: create a 4x4 SDR image (Rgba8, Srgb, BT.709) filled with a uniform color.
    fn make_sdr_4x4(r: u8, g: u8, b: u8) -> PixelBuffer {
        let mut data = vec![0u8; 4 * 4 * 4];
        for i in 0..16 {
            data[i * 4] = r;
            data[i * 4 + 1] = g;
            data[i * 4 + 2] = b;
            data[i * 4 + 3] = 255;
        }
        crate::types::pixel_buffer_from_vec(
            data,
            4,
            4,
            PixelFormat::Rgba8,
            ColorPrimaries::Bt709,
            TransferFunction::Srgb,
        )
        .unwrap()
    }

    /// Helper: create a 2x2 single-channel gain map filled with a uniform value.
    fn make_gainmap_2x2(value: u8) -> GainMap {
        let mut gm = GainMap::new(2, 2).unwrap();
        for v in &mut gm.data {
            *v = value;
        }
        gm
    }

    /// Helper: create standard test metadata.
    fn test_metadata() -> GainMapMetadata {
        // log2(1.0)=0.0, log2(4.0)=2.0
        crate::types::metadata_from_arrays(
            [0.0; 3],
            [2.0; 3],
            [1.0; 3],
            [1.0 / 64.0; 3],
            [1.0 / 64.0; 3],
            0.0,
            2.0,
            true,
            false,
        )
    }

    #[test]
    fn test_apply_gainmap_linear_float_format() {
        let sdr = make_sdr_4x4(128, 128, 128);
        let gainmap = make_gainmap_2x2(128);
        let metadata = test_metadata();

        let result = apply_gainmap(
            &sdr,
            &gainmap,
            &metadata,
            4.0,
            HdrOutputFormat::LinearFloat,
            enough::Unstoppable,
        )
        .unwrap();

        assert_eq!(result.descriptor().pixel_format(), PixelFormat::RgbaF32);
        assert_eq!(result.width(), 4);
        assert_eq!(result.height(), 4);
        // RgbaF32: 16 bytes per pixel (4 f32 channels)
        assert_eq!(result.as_slice().as_strided_bytes().len(), 4 * 4 * 16);
    }

    #[cfg(feature = "f16")]
    #[test]
    fn test_apply_gainmap_linear_f16_format() {
        let sdr = make_sdr_4x4(128, 128, 128);
        let gainmap = make_gainmap_2x2(128);
        let metadata = test_metadata();

        let f32_out = apply_gainmap(
            &sdr,
            &gainmap,
            &metadata,
            4.0,
            HdrOutputFormat::LinearFloat,
            enough::Unstoppable,
        )
        .unwrap();
        let f16_out = apply_gainmap(
            &sdr,
            &gainmap,
            &metadata,
            4.0,
            HdrOutputFormat::LinearF16,
            enough::Unstoppable,
        )
        .unwrap();

        assert_eq!(f16_out.descriptor().pixel_format(), PixelFormat::RgbaF16);
        assert_eq!(f16_out.width(), 4);
        assert_eq!(f16_out.height(), 4);
        // RgbaF16: 8 bytes per pixel (4 f16 channels).
        assert_eq!(f16_out.as_slice().as_strided_bytes().len(), 4 * 4 * 8);

        // f32 vs f16 must agree within f16 rounding (~1e-3 for values near 1).
        let f32_bytes = f32_out.as_slice();
        let f32_data = f32_bytes.as_strided_bytes();
        let f16_bytes = f16_out.as_slice();
        let f16_data = f16_bytes.as_strided_bytes();
        for px in 0..16 {
            let f32_idx = px * 16;
            let f16_idx = px * 8;
            for ch in 0..3 {
                let want = f32::from_le_bytes(
                    f32_data[f32_idx + ch * 4..f32_idx + ch * 4 + 4]
                        .try_into()
                        .unwrap(),
                );
                let got = half::f16::from_le_bytes(
                    f16_data[f16_idx + ch * 2..f16_idx + ch * 2 + 2]
                        .try_into()
                        .unwrap(),
                )
                .to_f32();
                let err = (want - got).abs();
                // f16 has ~3-4 sig figs near 1.0; allow generous tolerance for
                // values up to ~50 (full HDR boost range).
                let tol = (want.abs() * 5e-4).max(5e-4);
                assert!(
                    err < tol,
                    "px {px} ch {ch}: f32={want} f16={got} err={err} tol={tol}",
                );
            }
        }
    }

    #[test]
    fn test_apply_gainmap_srgb8_format() {
        let sdr = make_sdr_4x4(128, 128, 128);
        let gainmap = make_gainmap_2x2(128);
        let metadata = test_metadata();

        let result = apply_gainmap(
            &sdr,
            &gainmap,
            &metadata,
            4.0,
            HdrOutputFormat::Srgb8,
            enough::Unstoppable,
        )
        .unwrap();

        assert_eq!(result.descriptor().pixel_format(), PixelFormat::Rgba8);
        assert_eq!(result.width(), 4);
        assert_eq!(result.height(), 4);
    }

    #[test]
    fn test_apply_gainmap_boost_1() {
        // display_boost=1.0 → weight=0.0 → gain=1.0 everywhere → output ≈ SDR
        let sdr = make_sdr_4x4(128, 128, 128);
        let gainmap = make_gainmap_2x2(200); // High gain value, but weight=0 should negate it
        let metadata = test_metadata();

        let result = apply_gainmap(
            &sdr,
            &gainmap,
            &metadata,
            1.0,
            HdrOutputFormat::Srgb8,
            enough::Unstoppable,
        )
        .unwrap();

        // With boost=1.0, weight=0.0, gain=exp(0)=1.0 for all LUT entries.
        // HDR = (sdr_linear + offset) * 1.0 - offset = sdr_linear
        // So output should be very close to the input SDR values.
        let result_bytes = result.as_slice().as_strided_bytes();
        for i in 0..16 {
            let r = result_bytes[i * 4];
            let g = result_bytes[i * 4 + 1];
            let b = result_bytes[i * 4 + 2];
            assert!(
                (r as i16 - 128).unsigned_abs() <= 2,
                "boost=1 R should be ~128, got {}",
                r
            );
            assert!(
                (g as i16 - 128).unsigned_abs() <= 2,
                "boost=1 G should be ~128, got {}",
                g
            );
            assert!(
                (b as i16 - 128).unsigned_abs() <= 2,
                "boost=1 B should be ~128, got {}",
                b
            );
        }
    }

    #[test]
    fn test_apply_gainmap_boost_max() {
        // display_boost = hdr_capacity_max → weight=1.0 → full HDR enhancement
        let sdr = make_sdr_4x4(128, 128, 128);
        let gainmap = make_gainmap_2x2(255); // Max gain
        let metadata = test_metadata();

        let result_max = apply_gainmap(
            &sdr,
            &gainmap,
            &metadata,
            2.0f32.powf(metadata.alternate_hdr_headroom as f32), // linear display boost
            HdrOutputFormat::LinearFloat,
            enough::Unstoppable,
        )
        .unwrap();

        // Also compute with boost=1.0 for comparison
        let result_sdr = apply_gainmap(
            &sdr,
            &gainmap,
            &metadata,
            1.0,
            HdrOutputFormat::LinearFloat,
            enough::Unstoppable,
        )
        .unwrap();

        // Read first pixel from each
        let max_bytes = result_max.as_slice().as_strided_bytes();
        let sdr_bytes = result_sdr.as_slice().as_strided_bytes();
        let hdr_r = f32::from_le_bytes([max_bytes[0], max_bytes[1], max_bytes[2], max_bytes[3]]);
        let sdr_r = f32::from_le_bytes([sdr_bytes[0], sdr_bytes[1], sdr_bytes[2], sdr_bytes[3]]);

        // Full boost should produce significantly brighter output than no boost
        assert!(
            hdr_r > sdr_r * 1.5,
            "max boost ({}) should be much brighter than sdr ({})",
            hdr_r,
            sdr_r
        );
    }

    #[test]
    fn test_gain_map_lut_monotonic() {
        let metadata = test_metadata();
        let lut = GainMapLut::new(&metadata, 1.0);

        // LUT values should be monotonically non-decreasing from byte 0 to 255
        for channel in 0..3 {
            for i in 1..=255u8 {
                let prev = lut.lookup(i - 1, channel);
                let curr = lut.lookup(i, channel);
                assert!(
                    curr >= prev,
                    "LUT not monotonic at byte {} channel {}: {} < {}",
                    i,
                    channel,
                    curr,
                    prev
                );
            }
        }
    }

    #[test]
    fn test_gain_map_lut_endpoints() {
        let metadata = test_metadata();
        let lut = GainMapLut::new(&metadata, 1.0);

        // At weight=1.0:
        // Byte 0 → normalized=0.0 → gain = 2^gain_map_min = 2^0 = 1.0
        let gain_0 = lut.lookup(0, 0);
        let expected_min = 2.0f32.powf(metadata.channels[0].min as f32);
        assert!(
            (gain_0 - expected_min).abs() < 0.01,
            "byte 0 should give 2^gain_map_min={}, got {}",
            expected_min,
            gain_0
        );

        // Byte 255 → normalized=1.0 → gain = 2^gain_map_max = 2^2 = 4.0
        let gain_255 = lut.lookup(255, 0);
        let expected_max = 2.0f32.powf(metadata.channels[0].max as f32);
        assert!(
            (gain_255 - expected_max).abs() < 0.1,
            "byte 255 should give 2^gain_map_max={}, got {}",
            expected_max,
            gain_255
        );
    }

    #[test]
    fn test_apply_gainmap_multichannel() {
        let sdr = make_sdr_4x4(128, 128, 128);

        // Create a 2x2 multichannel (3-channel) gain map
        let mut gainmap = GainMap::new_multichannel(2, 2).unwrap();
        assert_eq!(gainmap.channels, 3);
        // Fill with different values per channel
        for i in 0..(2 * 2) {
            gainmap.data[i * 3] = 200; // R channel - high gain
            gainmap.data[i * 3 + 1] = 128; // G channel - mid gain
            gainmap.data[i * 3 + 2] = 50; // B channel - low gain
        }

        let metadata = test_metadata();

        let result = apply_gainmap(
            &sdr,
            &gainmap,
            &metadata,
            4.0,
            HdrOutputFormat::LinearFloat,
            enough::Unstoppable,
        )
        .unwrap();

        assert_eq!(result.width(), 4);
        assert_eq!(result.height(), 4);
        assert_eq!(result.descriptor().pixel_format(), PixelFormat::RgbaF32);
        assert_eq!(result.as_slice().as_strided_bytes().len(), 4 * 4 * 16);
    }

    #[test]
    fn test_apply_gainmap_invalid_boost() {
        // display_boost=0.5 (< 1.0) is clamped to 1.0 internally, not an error.
        // Verify it behaves exactly like boost=1.0.
        let sdr = make_sdr_4x4(128, 128, 128);
        let gainmap = make_gainmap_2x2(200);
        let metadata = test_metadata();

        let result_low = apply_gainmap(
            &sdr,
            &gainmap,
            &metadata,
            0.5,
            HdrOutputFormat::Srgb8,
            enough::Unstoppable,
        )
        .unwrap();

        let result_one = apply_gainmap(
            &sdr,
            &gainmap,
            &metadata,
            1.0,
            HdrOutputFormat::Srgb8,
            enough::Unstoppable,
        )
        .unwrap();

        // Both should produce identical output since 0.5 is clamped to 1.0
        assert_eq!(
            result_low.as_slice().as_strided_bytes(),
            result_one.as_slice().as_strided_bytes()
        );
    }

    #[test]
    fn test_apply_gainmap_cancellation() {
        /// A Stop implementation that cancels immediately
        struct ImmediateCancel;

        impl enough::Stop for ImmediateCancel {
            fn check(&self) -> std::result::Result<(), enough::StopReason> {
                Err(enough::StopReason::Cancelled)
            }
        }

        // Create minimal images
        let sdr = crate::types::new_pixel_buffer(
            4,
            4,
            PixelFormat::Rgba8,
            ColorPrimaries::Bt709,
            TransferFunction::Srgb,
        )
        .unwrap();
        let gainmap = GainMap::new(2, 2).unwrap();
        let metadata = GainMapMetadata::default();

        // Should return Stopped error due to cancellation
        let err = apply_gainmap(
            &sdr,
            &gainmap,
            &metadata,
            4.0,
            HdrOutputFormat::Srgb8,
            ImmediateCancel,
        )
        .unwrap_err();

        assert!(matches!(
            err.error(),
            crate::Error::Stopped(enough::StopReason::Cancelled)
        ));
    }

    #[test]
    fn shepards_lut_try_new_rejects_non_integer_ratio() {
        // 5x5 image with 2x2 gainmap → no exact integer scale.
        assert!(ShepardsLut::try_new(5, 5, 2, 2).is_none());
        // 0-dim gainmap is rejected.
        assert!(ShepardsLut::try_new(8, 8, 0, 2).is_none());
        // Exact 4x scale on both axes.
        assert!(ShepardsLut::try_new(8, 8, 2, 2).is_some());
        // Asymmetric integer scales are still valid.
        assert!(ShepardsLut::try_new(8, 12, 2, 3).is_some());
    }

    #[test]
    fn shepards_lut_weights_at_sample_center_collapse_to_nearest() {
        // Sub-pixel offset (0, 0) sits exactly on the top-left sample.
        // C++ libultrahdr short-circuits to weights = [1, 0, 0, 0]; we mirror
        // that so the LUT and per-pixel paths agree at sample centers.
        let lut = ShepardsLut::new(4, 4).expect("4x4 fits in MAX_SHEPARDS_TABLE_ENTRIES");
        let table = lut.pick(false, false);
        assert_eq!(table[0], 1.0);
        assert_eq!(table[1], 0.0);
        assert_eq!(table[2], 0.0);
        assert_eq!(table[3], 0.0);
    }

    #[test]
    fn shepards_weights_normalize_to_one() {
        // For any non-degenerate (fx, fy) the four weights must sum to 1.0
        // (within f32 rounding). This is what makes the result independent
        // of the underlying gain values.
        for &fx in &[0.1f32, 0.25, 0.5, 0.75, 0.9] {
            for &fy in &[0.1f32, 0.25, 0.5, 0.75, 0.9] {
                let w = shepards_weights(fx, fy);
                let total: f32 = w.iter().sum();
                assert!(
                    (total - 1.0).abs() < 1e-5,
                    "weights at ({fx}, {fy}) sum to {total}",
                );
            }
        }
    }

    #[test]
    fn shepards_int_lut_matches_float_path_at_sample_centers() {
        // Walk a 4x4 image over a 2x2 gainmap (scale=2). At every output
        // pixel that lands on a gainmap sample (offsets 0 in both axes —
        // here that's every other pixel), both paths must agree.
        let mut gainmap = GainMap::new(2, 2).unwrap();
        gainmap.data = vec![10, 200, 50, 150];
        let metadata = GainMapMetadata::default();
        let lut = GainMapLut::new(&metadata, 1.0);
        let shepards = ShepardsLut::try_new(4, 4, 2, 2).unwrap();

        let mut row_int = vec![[0.0f32; 3]; 4];
        let mut row_float = vec![[0.0f32; 3]; 4];

        for y in [0u32, 2u32] {
            sample_row_lut_int(&gainmap, &lut, &shepards, y, &mut row_int);
            sample_row_lut_float(&gainmap, &lut, y, 4, 4, &mut row_float);
            for x in [0usize, 2usize] {
                // Sample-center pixels: weights collapse to nearest, both
                // paths must produce identical output (no f32 drift).
                assert_eq!(
                    row_int[x], row_float[x],
                    "mismatch at ({x}, {y}): int={:?} float={:?}",
                    row_int[x], row_float[x]
                );
            }
        }
    }

    #[test]
    fn shepards_int_lut_matches_float_path_within_rounding() {
        // Same setup, full-row comparison. Off-center pixels use precomputed
        // vs per-pixel weights; the operations differ in associativity so
        // bit-equality is not guaranteed, but values must agree to 1e-6.
        let mut gainmap = GainMap::new(2, 2).unwrap();
        gainmap.data = vec![10, 200, 50, 150];
        let metadata = GainMapMetadata::default();
        let lut = GainMapLut::new(&metadata, 1.0);
        let shepards = ShepardsLut::try_new(8, 8, 2, 2).unwrap();

        for y in 0..8 {
            let mut row_int = vec![[0.0f32; 3]; 8];
            let mut row_float = vec![[0.0f32; 3]; 8];
            sample_row_lut_int(&gainmap, &lut, &shepards, y, &mut row_int);
            sample_row_lut_float(&gainmap, &lut, y, 8, 8, &mut row_float);
            for x in 0..8 {
                for c in 0..3 {
                    let diff = (row_int[x][c] - row_float[x][c]).abs();
                    assert!(
                        diff < 1e-6,
                        "({x}, {y})[{c}]: int={} float={} diff={}",
                        row_int[x][c],
                        row_float[x][c],
                        diff,
                    );
                }
            }
        }
    }

    // Regression: large `scale_x * scale_y` used to either wrap a u32
    // multiplication or allocate ~16 GB (security audit 2026-05-06,
    // finding C3). `try_new` must early-return `None` so callers fall
    // through to the per-pixel float path.
    #[test]
    fn shepards_lut_try_new_caps_pathological_scale() {
        // 65535x65535 image, 1x1 gainmap → scale 65535 each axis.
        // 65535 * 65535 * 4 wraps u32; the cap rejects it before that.
        assert!(ShepardsLut::try_new(65535, 65535, 1, 1).is_none());

        // 32768x32768 image, 2x2 gainmap → scale 16384 each axis.
        // 16384 * 16384 = 268M, far above MAX_SHEPARDS_TABLE_ENTRIES.
        assert!(ShepardsLut::try_new(32768, 32768, 2, 2).is_none());

        // 64x64 image, 1x1 gainmap → scale 64 each axis.
        // 64 * 64 = 4096 — exactly at the cap, must succeed.
        assert!(ShepardsLut::try_new(64, 64, 1, 1).is_some());

        // 65x65 image, 1x1 gainmap → scale 65 each axis.
        // 65 * 65 = 4225, just over the cap. Must reject.
        assert!(ShepardsLut::try_new(65, 65, 1, 1).is_none());
    }

    #[test]
    fn shepards_lut_try_new_rejects_zero_dims() {
        assert!(ShepardsLut::try_new(0, 64, 8, 8).is_none());
        assert!(ShepardsLut::try_new(64, 0, 8, 8).is_none());
        assert!(ShepardsLut::try_new(64, 64, 0, 8).is_none());
        assert!(ShepardsLut::try_new(64, 64, 8, 0).is_none());
    }

    #[test]
    fn shepards_lut_new_rejects_zero_scale() {
        assert!(ShepardsLut::new(0, 4).is_none());
        assert!(ShepardsLut::new(4, 0).is_none());
    }

    // Regression: a forged GainMap with `width = 0` used to underflow
    // `gh - 1` and panic in samplers (security audit 2026-05-06, finding
    // C2). `apply_gainmap_slice` must reject it before any arithmetic.
    #[test]
    fn apply_gainmap_slice_rejects_zero_dim_gainmap() {
        use crate::types::{ColorPrimaries, PixelBuffer, TransferFunction};
        use enough::Unstoppable;

        let desc = zenpixels::PixelDescriptor::from_pixel_format(PixelFormat::Rgba8)
            .with_primaries(ColorPrimaries::Bt709)
            .with_transfer(TransferFunction::Srgb);
        let sdr = PixelBuffer::try_new(8, 8, desc).unwrap();
        let metadata = GainMapMetadata::default();
        let gm = GainMap {
            width: 0,
            height: 8,
            channels: 1,
            data: alloc::vec![0u8; 0],
        };
        let res = apply_gainmap(
            &sdr,
            &gm,
            &metadata,
            1.0,
            HdrOutputFormat::LinearFloat,
            Unstoppable,
        );
        assert!(res.is_err());
    }

    // Regression: large finite metadata used to silently produce `+inf`
    // gain values in the LUT (security audit 2026-05-06, finding C1).
    #[test]
    fn gainmap_lut_clamps_large_metadata_to_finite() {
        let mut metadata = GainMapMetadata::default();
        // Bypass external validation — simulate a caller that did not
        // run `validate_gainmap_magnitude`. The LUT must still produce
        // finite values via `clamp_gain_finite`.
        for ch in &mut metadata.channels {
            ch.min = 1.0e30;
            ch.max = 1.0e30;
        }
        let lut = GainMapLut::new(&metadata, 1.0);
        for i in 0..256 {
            for c in 0..3 {
                let g = lut.lookup(i as u8, c);
                assert!(g.is_finite(), "lut[{c}][{i}] = {g} should be finite");
            }
        }
    }
}
