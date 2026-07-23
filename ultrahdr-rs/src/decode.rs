//! Ultra HDR decoder.

use ultrahdr_core::gainmap::apply::{HdrOutputFormat, apply_gainmap};
use ultrahdr_core::{
    ColorPrimaries, Error, GainMap, GainMapMetadata, PixelBuffer, PixelFormat, Result, Stop,
    TransferFunction, Unstoppable, limits, pixel_buffer_from_vec, validate_ultrahdr_dimensions,
};
use whereat::at;
use zenjpeg::container::marker::find_jpeg_boundaries;
use zenjpeg::container::xmp::parse_xmp;

use crate::container::{self, AppSegment};

/// Caller-supplied resource limits for [`Decoder`] decode paths.
///
/// Bounds what the bundled zenjpeg codec and the HDR reconstruction are
/// allowed to allocate when decoding untrusted input. Attach limits with
/// [`Decoder::new_with_limits`]; the plain [`Decoder::new`] keeps the
/// historical uncapped behavior.
///
/// Two independent caps:
/// - **Pixel cap** ([`with_max_pixels`](Self::with_max_pixels)): maximum
///   `width * height` for any decoded image — the base JPEG, the gain-map
///   JPEG, and the reconstructed HDR output. Enforced against the JPEG
///   header dimensions *before* pixel allocation, and always clamped to the
///   crate-wide hard caps ([`limits::MAX_TOTAL_PIXELS`],
///   [`limits::MAX_IMAGE_DIMENSION`]) — caller limits can tighten the caps,
///   never loosen them. Defaults to [`limits::MAX_TOTAL_PIXELS`] (500 MP).
/// - **Memory cap** ([`with_max_memory`](Self::with_max_memory)): maximum
///   bytes for any single decode output allocation, and forwarded to the
///   JPEG codec's internal memory limit. Unset by default.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ResourceLimits {
    max_pixels: u64,
    max_memory_bytes: Option<u64>,
}

impl Default for ResourceLimits {
    /// Pixel cap at [`limits::MAX_TOTAL_PIXELS`] (500 MP), no memory cap.
    fn default() -> Self {
        Self {
            max_pixels: limits::MAX_TOTAL_PIXELS,
            max_memory_bytes: None,
        }
    }
}

impl ResourceLimits {
    /// Create limits with the default caps (same as [`Default`]).
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the maximum decoded pixel count (`width * height`).
    ///
    /// Values above [`limits::MAX_TOTAL_PIXELS`] are clamped down to it at
    /// enforcement time — the crate-wide hard cap always applies.
    #[must_use]
    pub fn with_max_pixels(mut self, max_pixels: u64) -> Self {
        self.max_pixels = max_pixels;
        self
    }

    /// Set the maximum bytes for any single decode output allocation.
    #[must_use]
    pub fn with_max_memory(mut self, max_bytes: u64) -> Self {
        self.max_memory_bytes = Some(max_bytes);
        self
    }

    /// The configured pixel cap (before hard-cap clamping).
    pub fn max_pixels(&self) -> u64 {
        self.max_pixels
    }

    /// The configured memory cap in bytes, if any.
    pub fn max_memory(&self) -> Option<u64> {
        self.max_memory_bytes
    }

    /// The pixel cap that is actually enforced: the caller's value clamped
    /// to the crate-wide hard cap.
    fn effective_max_pixels(&self) -> u64 {
        self.max_pixels.min(limits::MAX_TOTAL_PIXELS)
    }

    /// Validate decoded dimensions and the projected output allocation.
    ///
    /// Runs [`validate_ultrahdr_dimensions`] (non-zero, per-dimension cap,
    /// 500 MP hard cap) and then the caller's tighter pixel/memory caps.
    fn check_output(&self, width: u32, height: u32, bytes_per_pixel: u64) -> Result<()> {
        validate_ultrahdr_dimensions(width, height)?;
        let px = u64::from(width) * u64::from(height);
        if px > self.effective_max_pixels() {
            return Err(at!(Error::LimitExceeded(format!(
                "decoded pixel count {} ({}x{}) exceeds configured limit {}",
                px,
                width,
                height,
                self.effective_max_pixels()
            ))));
        }
        if let Some(max_mem) = self.max_memory_bytes {
            let bytes = px.saturating_mul(bytes_per_pixel);
            if bytes > max_mem {
                return Err(at!(Error::LimitExceeded(format!(
                    "decoded output of {} bytes ({}x{}x{}) exceeds configured memory limit {}",
                    bytes, width, height, bytes_per_pixel, max_mem
                ))));
            }
        }
        Ok(())
    }
}

/// Ultra HDR decoder.
///
/// Decodes Ultra HDR JPEGs, extracting the SDR base image, gain map,
/// and metadata. Can reconstruct HDR content at various display
/// brightness levels.
///
/// The decoder borrows the input data to avoid an unconditional copy.
///
/// # Resource limits
///
/// [`Decoder::new`] decodes with no resource caps — appropriate for trusted
/// input. For untrusted input (servers, user uploads), construct with
/// [`Decoder::new_with_limits`] so JPEG header dimensions are validated
/// against a pixel/memory budget *before* any pixel allocation happens:
///
/// ```no_run
/// use ultrahdr_rs::{Decoder, ResourceLimits};
///
/// # fn main() -> ultrahdr_rs::Result<()> {
/// let data = std::fs::read("untrusted.jpg").expect("read");
/// // 100 MP pixel cap + 1 GiB output-allocation cap
/// let limits = ResourceLimits::new()
///     .with_max_pixels(100_000_000)
///     .with_max_memory(1 << 30);
/// let decoder = Decoder::new_with_limits(&data, limits)?;
/// let sdr = decoder.decode_sdr()?; // over-budget input => clean Err
/// # Ok(()) }
/// ```
pub struct Decoder<'a> {
    data: &'a [u8],
    metadata: Option<GainMapMetadata>,
    primary_jpeg: Option<(usize, usize)>,
    gainmap_jpeg: Option<(usize, usize)>,
    is_ultrahdr: bool,
    limits: Option<ResourceLimits>,
}

impl<'a> Decoder<'a> {
    /// Create a new decoder from JPEG data.
    ///
    /// The decoder borrows the data — no copy is made.
    ///
    /// No resource limits are applied on this path; for untrusted input use
    /// [`Decoder::new_with_limits`].
    pub fn new(data: &'a [u8]) -> Result<Self> {
        Self::build(data, None)
    }

    /// Create a new decoder with caller-supplied [`ResourceLimits`].
    ///
    /// All decode paths ([`decode_sdr`](Self::decode_sdr),
    /// [`decode_gainmap`](Self::decode_gainmap),
    /// [`decode_hdr`](Self::decode_hdr) and variants) validate JPEG header
    /// dimensions against the limits before allocating pixel buffers, and
    /// the bundled JPEG codec enforces the same caps internally. Over-budget
    /// input yields [`Error::LimitExceeded`] — never an unbounded
    /// allocation.
    pub fn new_with_limits(data: &'a [u8], limits: ResourceLimits) -> Result<Self> {
        Self::build(data, Some(limits))
    }

    fn build(data: &'a [u8], limits: Option<ResourceLimits>) -> Result<Self> {
        let mut decoder = Self {
            data,
            metadata: None,
            primary_jpeg: None,
            gainmap_jpeg: None,
            is_ultrahdr: false,
            limits,
        };

        decoder.parse()?;
        Ok(decoder)
    }

    /// The resource limits this decoder was constructed with, if any.
    pub fn resource_limits(&self) -> Option<&ResourceLimits> {
        self.limits.as_ref()
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
        self.decode_sdr_with_stop(Unstoppable)
    }

    /// [`decode_sdr`](Self::decode_sdr) with cooperative cancellation.
    ///
    /// The `stop` token is checked before and during the JPEG decode.
    /// Cancellation surfaces as [`Error::Stopped`].
    pub fn decode_sdr_with_stop(&self, stop: impl Stop) -> Result<PixelBuffer> {
        let primary_data = self
            .primary_jpeg()
            .ok_or_else(|| at!(Error::DecodeError("No primary image found".into())))?;
        decode_jpeg_to_rgb(primary_data, self.limits.as_ref(), stop)
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
        self.decode_gainmap_with_stop(Unstoppable)
    }

    /// [`decode_gainmap`](Self::decode_gainmap) with cooperative
    /// cancellation.
    ///
    /// The `stop` token is checked before and during the gain-map JPEG
    /// decode. Cancellation surfaces as [`Error::Stopped`].
    pub fn decode_gainmap_with_stop(&self, stop: impl Stop) -> Result<GainMap> {
        let gainmap_data = self
            .gainmap_jpeg()
            .ok_or_else(|| at!(Error::DecodeError("No gain map found".into())))?;

        let single_channel = self.metadata.as_ref().map(|m| m.is_single_channel());

        if single_channel != Some(false) {
            // Single-channel per metadata (or unknown): the historical Gray
            // decode is the exact luma plane — keep it as the fast path.
            // Some color encodings can't produce Gray output ("unsupported
            // color conversion", #27); those fall through to the RGB path.
            // (A cancelled Gray decode also falls through, but the RGB
            // path's own stop check re-raises `Stopped` immediately.)
            if let Ok((width, height, data)) =
                decode_jpeg_to_grayscale_bytes(gainmap_data, self.limits.as_ref(), &stop)
            {
                return Ok(GainMap {
                    width,
                    height,
                    channels: 1,
                    data,
                });
            }
        }

        let (width, height, rgb) =
            decode_jpeg_to_rgb_bytes(gainmap_data, self.limits.as_ref(), &stop)?;
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
        self.decode_hdr_with_format_and_stop(
            display_boost,
            HdrOutputFormat::LinearFloat,
            Unstoppable,
        )
    }

    /// [`decode_hdr`](Self::decode_hdr) with cooperative cancellation.
    ///
    /// The `stop` token is checked throughout the base-JPEG decode, the
    /// gain-map decode, and the gain-map application. Cancellation surfaces
    /// as [`Error::Stopped`].
    pub fn decode_hdr_with_stop(&self, display_boost: f32, stop: impl Stop) -> Result<PixelBuffer> {
        self.decode_hdr_with_format_and_stop(display_boost, HdrOutputFormat::LinearFloat, stop)
    }

    /// Decode to HDR with a specific output format.
    pub fn decode_hdr_with_format(
        &self,
        display_boost: f32,
        format: HdrOutputFormat,
    ) -> Result<PixelBuffer> {
        self.decode_hdr_with_format_and_stop(display_boost, format, Unstoppable)
    }

    /// [`decode_hdr_with_format`](Self::decode_hdr_with_format) with
    /// cooperative cancellation.
    pub fn decode_hdr_with_format_and_stop(
        &self,
        display_boost: f32,
        format: HdrOutputFormat,
        stop: impl Stop,
    ) -> Result<PixelBuffer> {
        if !self.is_ultrahdr {
            return Err(at!(Error::DecodeError("Not an Ultra HDR image".into())));
        }

        if !display_boost.is_finite() || display_boost < 1.0 {
            return Err(at!(Error::DecodeError(format!(
                "display_boost must be >= 1.0, got {}",
                display_boost
            ))));
        }

        let metadata = self
            .metadata
            .as_ref()
            .ok_or_else(|| at!(Error::DecodeError("No gain map metadata".into())))?;

        let sdr = self.decode_sdr_with_stop(&stop)?;
        let gainmap = self.decode_gainmap_with_stop(&stop)?;

        // The HDR output buffer is up to 16 bytes/pixel (RgbaF32) — 4x the
        // SDR decode. Check it against the limits before apply_gainmap
        // allocates it.
        if let Some(lim) = &self.limits {
            lim.check_output(
                sdr.width(),
                sdr.height(),
                hdr_output_bytes_per_pixel(format),
            )?;
        }

        apply_gainmap(&sdr, &gainmap, metadata, display_boost, format, stop)
    }

    /// Decode the UltraHDR JPEG and apply the **audited-default HDR→SDR
    /// tone map** in one call, returning an SDR `Rgba8` `PixelBuffer` ready
    /// for sRGB display.
    ///
    /// Pipeline:
    /// 1. Decode the SDR base + gain map and reconstruct linear-light HDR
    ///    (`apply_gainmap` → `HdrOutputFormat::LinearFloat`,
    ///    `1.0` = BT.2408 SDR white = 203 nits).
    /// 2. Auto-measure the source peak via
    ///    [`CllMeasure::measure_max`](ultrahdr_core::CllMeasure::measure_max)
    ///    (MaxRgb reduction, BT.2408 anchor) — the audited-winner peak
    ///    measurement that won 3 of 6 ranking criteria in the 2026-06-22
    ///    shootout.
    /// 3. Apply the [`Bt2446A`](ultrahdr_core::Bt2446A) tone curve in
    ///    linear-light BT.2020 — the audited-winner curve that won mean
    ///    ΔE2000 by 2-5× over every channel-independent curve tested.
    /// 4. Encode through the sRGB OETF and write 8-bit RGBA.
    ///
    /// `target_primaries` is currently informational — the SDR output is
    /// always tagged with [`ColorPrimaries::Bt709`] (the BT.2446 §4 output
    /// gamut and the de-facto SDR display gamut). Accepting it here so the
    /// parameter shape stays stable when a BT.2020-SDR or DCI-P3 output
    /// surface lands in 0.6.0 or later.
    ///
    /// Skips the public HDR-roundtrip API entirely if all the caller needs
    /// is SDR display.
    ///
    /// Gated behind the `tonemap-bt2446a` Cargo feature (forwards to
    /// `ultrahdr-core/tonemap-bt2446a`).
    #[cfg(feature = "tonemap-bt2446a")]
    pub fn decode_full_sdr(&self, target_primaries: ColorPrimaries) -> Result<PixelBuffer> {
        use ultrahdr_core::color::transfer::srgb_oetf;
        use ultrahdr_core::{
            Bt2446A, CllMeasure, ContentLightLevel, DiffuseWhite, LightLevelMethod,
            new_pixel_buffer,
        };

        let _ = target_primaries; // informational for now; see docs

        if !self.is_ultrahdr {
            return Err(at!(Error::DecodeError("Not an Ultra HDR image".into())));
        }

        // Step 1: reconstruct linear-light HDR. Use the metadata's full
        // `alternate_hdr_headroom` boost so the buffer carries the full
        // dynamic range we then tone-map down. `1.0` in the resulting buffer
        // = BT.2408 SDR white = 203 nits per the apply_gainmap contract.
        let metadata = self
            .metadata
            .as_ref()
            .ok_or_else(|| at!(Error::DecodeError("No gain map metadata".into())))?;
        let display_boost = 2.0f32
            .powf((metadata.alternate_hdr_headroom as f32).max(0.0))
            .max(1.0);
        let hdr = self.decode_hdr_with_format(display_boost, HdrOutputFormat::LinearFloat)?;

        // Step 2: measure peak via audited-winner `measure_max` (MaxRgb /
        // BT.2408). The reconstructed buffer is RgbaF32 linear, so this hits
        // the `CllMeasure::measure_max` happy path directly.
        let cll: ContentLightLevel = ContentLightLevel::measure_max(
            hdr.as_slice(),
            DiffuseWhite::BT2408,
            LightLevelMethod::MaxRgb,
        )
        .ok_or_else(|| {
            at!(Error::DecodeError(
                "measure_max returned None on RgbaF32 HDR buffer".into(),
            ))
        })?;
        // `measure_max` writes the MaxCLL field directly in cd/m². Clamp to
        // SDR peak so peaks below 100 nits don't break the curve.
        let hdr_peak_nits = (cll.max_content_light_level as f32).max(100.0);

        // Step 3: apply Bt2446A. The curve expects normalized
        // `1.0 = hdr_peak_nits` in, normalized `1.0 = sdr_peak_nits` out.
        // Our buffer is normalized `1.0 = 203 nits`, so scale by
        // `203 / hdr_peak_nits` before the curve and scale by
        // `100 / 203` * (target SDR peak 100) afterward — combined: the
        // curve's output is `1.0 = 100 nits`, which is `100 / 255 ≈ 0.39`
        // in the sRGB-display convention. We renormalize so 100 nits → 1.0
        // (SDR-display 100% white) before sRGB encoding.
        let sdr_peak_nits: f32 = 100.0;
        let curve = Bt2446A::new(hdr_peak_nits, sdr_peak_nits);
        let to_curve = DiffuseWhite::BT2408.nits() / hdr_peak_nits;
        let width = hdr.width();
        let height = hdr.height();

        // Step 4: re-encode through sRGB OETF and write 8-bit RGBA. Tag the
        // output as Bt709 + Srgb (the BT.2446 §4 output gamut + standard
        // SDR display transfer).
        let mut out = new_pixel_buffer(
            width,
            height,
            PixelFormat::Rgba8,
            ColorPrimaries::Bt709,
            TransferFunction::Srgb,
        )?;
        let hdr_slice = hdr.as_slice();
        let hdr_stride = hdr_slice.stride();
        let hdr_bytes = hdr_slice.as_strided_bytes();
        let out_stride = out.stride();
        let mut out_view = out.as_slice_mut();
        let out_bytes = out_view.as_strided_bytes_mut();

        for y in 0..height as usize {
            let in_row = y * hdr_stride;
            let out_row = y * out_stride;
            for x in 0..width as usize {
                let idx = in_row + x * 16;
                let r = f32::from_le_bytes([
                    hdr_bytes[idx],
                    hdr_bytes[idx + 1],
                    hdr_bytes[idx + 2],
                    hdr_bytes[idx + 3],
                ]);
                let g = f32::from_le_bytes([
                    hdr_bytes[idx + 4],
                    hdr_bytes[idx + 5],
                    hdr_bytes[idx + 6],
                    hdr_bytes[idx + 7],
                ]);
                let b = f32::from_le_bytes([
                    hdr_bytes[idx + 8],
                    hdr_bytes[idx + 9],
                    hdr_bytes[idx + 10],
                    hdr_bytes[idx + 11],
                ]);
                let mapped = curve.map_rgb([r * to_curve, g * to_curve, b * to_curve]);
                let r_srgb = srgb_oetf(mapped[0].clamp(0.0, 1.0));
                let g_srgb = srgb_oetf(mapped[1].clamp(0.0, 1.0));
                let b_srgb = srgb_oetf(mapped[2].clamp(0.0, 1.0));
                let o = out_row + x * 4;
                out_bytes[o] = (r_srgb * 255.0).round() as u8;
                out_bytes[o + 1] = (g_srgb * 255.0).round() as u8;
                out_bytes[o + 2] = (b_srgb * 255.0).round() as u8;
                out_bytes[o + 3] = 255;
            }
        }
        drop(out_view);
        Ok(out)
    }

    /// Parse the Ultra HDR structure.
    ///
    /// Uses `container::scan_segments` for efficient marker-to-marker scanning
    /// instead of byte-by-byte search.
    fn parse(&mut self) -> Result<()> {
        // Check for valid JPEG
        if self.data.len() < 4 || self.data[0] != 0xFF || self.data[1] != 0xD8 {
            return Err(at!(Error::DecodeError("Not a valid JPEG".into())));
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
                    return Err(at!(Error::DecodeError(
                        "MPF entry offset+size overflows".into()
                    )));
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

/// Bytes per pixel for an [`HdrOutputFormat`] output buffer.
fn hdr_output_bytes_per_pixel(format: HdrOutputFormat) -> u64 {
    match format {
        HdrOutputFormat::LinearFloat => 16,
        #[cfg(feature = "f16")]
        HdrOutputFormat::LinearF16 => 8,
        HdrOutputFormat::Srgb8 => 4,
        // `HdrOutputFormat` is non_exhaustive: assume the widest layout so a
        // future variant can never under-count against a memory cap.
        _ => 16,
    }
}

/// Map a zenjpeg decode error to a typed ultrahdr error.
///
/// Limit rejections and cancellations keep their types
/// ([`Error::LimitExceeded`] / [`Error::Stopped`] /
/// [`Error::AllocationFailed`]) instead of collapsing into a
/// [`Error::DecodeError`] string.
///
/// Returns the `At<Error>` wrapper directly: converting the built `At` back
/// into a bare `Error` would route through core's
/// `From<zenpixels::At<E>> for Error` blanket impl, which stringifies the
/// variant into `InvalidPixelData` and loses the type.
fn map_jpeg_decode_error(e: zenjpeg::decoder::Error) -> whereat::At<Error> {
    use zenjpeg::decoder::ErrorKind;
    match e.kind() {
        ErrorKind::Cancelled(reason) => at!(Error::Stopped(*reason)),
        ErrorKind::ImageTooLarge { pixels, limit } => at!(Error::LimitExceeded(format!(
            "JPEG header declares {} pixels, over the configured limit {}",
            pixels, limit
        ))),
        ErrorKind::AllocationFailed { bytes, .. } => at!(Error::AllocationFailed(*bytes)),
        _ => at!(Error::DecodeError(format!("JPEG decode failed: {}", e))),
    }
}

/// Validate the JPEG header dimensions against the limits *before* the
/// decode allocates anything.
///
/// A header-only probe ([`zenjpeg::decoder::Decoder::read_info`]) reads the
/// SOF dimensions at parse cost. Probe failures are ignored — the real
/// decode immediately after will report the parse error properly — but if
/// the dimensions are readable, an over-budget image is rejected here with
/// a typed [`Error::LimitExceeded`], before any pixel allocation.
fn precheck_jpeg_header(
    jpeg_data: &[u8],
    limits: &ResourceLimits,
    output_bytes_per_pixel: u64,
) -> Result<()> {
    if let Ok(info) = zenjpeg::decoder::Decoder::new().read_info(jpeg_data) {
        limits.check_output(
            info.dimensions.width,
            info.dimensions.height,
            output_bytes_per_pixel,
        )?;
    }
    Ok(())
}

/// Construct a zenjpeg decoder with the caller's resource limits applied.
///
/// zenjpeg enforces `max_pixels` against the JPEG header (SOF) dimensions
/// *before* allocating pixel planes, so an over-budget bomb is rejected at
/// header-parse cost.
fn jpeg_decoder_with_limits(
    format: zenjpeg::decoder::PixelFormat,
    limits: Option<&ResourceLimits>,
) -> zenjpeg::decoder::Decoder {
    let mut dec = zenjpeg::decoder::Decoder::new().output_format(format);
    if let Some(lim) = limits {
        dec = dec.max_pixels(lim.effective_max_pixels());
        if let Some(max_mem) = lim.max_memory() {
            dec = dec.max_memory(max_mem);
        }
    }
    dec
}

/// `width * height * bytes_per_pixel` in overflow-checked arithmetic.
///
/// The historical code multiplied in fixed-width integers, which can wrap
/// for attacker-controlled dimensions (and at a much smaller threshold on
/// 32-bit targets). A wrap here would produce an undersized capacity hint —
/// harmless for `Vec` growth but a symptom of unchecked size math — so the
/// product is computed in u64 and range-checked into usize.
fn checked_output_len(width: u32, height: u32, bytes_per_pixel: u64) -> Result<usize> {
    u64::from(width)
        .checked_mul(u64::from(height))
        .and_then(|px| px.checked_mul(bytes_per_pixel))
        .and_then(|bytes| usize::try_from(bytes).ok())
        .ok_or_else(|| {
            at!(Error::LimitExceeded(format!(
                "output size {}x{}x{} overflows addressable memory",
                width, height, bytes_per_pixel
            )))
        })
}

/// Allocate a `Vec<u8>` of exactly `len` capacity, failing with a clean
/// [`Error::AllocationFailed`] instead of aborting the process on OOM.
fn try_vec_with_capacity(len: usize) -> Result<Vec<u8>> {
    let mut v = Vec::new();
    v.try_reserve_exact(len)
        .map_err(|_| at!(Error::AllocationFailed(len)))?;
    Ok(v)
}

/// Decode JPEG to RGB.
fn decode_jpeg_to_rgb(
    jpeg_data: &[u8],
    limits: Option<&ResourceLimits>,
    stop: impl Stop,
) -> Result<PixelBuffer> {
    use zenjpeg::decoder::PixelFormat as JpegPixelFormat;
    stop.check().map_err(|r| at!(Error::Stopped(r)))?;
    if let Some(lim) = limits {
        precheck_jpeg_header(jpeg_data, lim, 4)?;
    }
    let decoded = jpeg_decoder_with_limits(JpegPixelFormat::Rgb, limits)
        .decode(jpeg_data, stop)
        .map_err(map_jpeg_decode_error)?;

    let width = decoded.width();
    let height = decoded.height();
    if let Some(lim) = limits {
        lim.check_output(width, height, 4)?;
    }
    let pixels = decoded
        .pixels_u8()
        .ok_or_else(|| at!(Error::DecodeError("No pixel data in decoded JPEG".into())))?;
    let bpp = decoded.bytes_per_pixel();

    // Convert to RGBA if needed
    let data = if bpp == 3 {
        // RGB -> RGBA
        let mut rgba = try_vec_with_capacity(checked_output_len(width, height, 4)?)?;
        for chunk in pixels.chunks_exact(3) {
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
        let mut rgba = try_vec_with_capacity(checked_output_len(width, height, 4)?)?;
        for &g in pixels {
            rgba.push(g);
            rgba.push(g);
            rgba.push(g);
            rgba.push(255);
        }
        rgba
    } else {
        return Err(at!(Error::DecodeError(format!(
            "Unsupported bytes per pixel: {}",
            bpp
        ))));
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
fn decode_jpeg_to_grayscale_bytes(
    jpeg_data: &[u8],
    limits: Option<&ResourceLimits>,
    stop: impl Stop,
) -> Result<(u32, u32, Vec<u8>)> {
    use zenjpeg::decoder::PixelFormat as JpegPixelFormat;
    stop.check().map_err(|r| at!(Error::Stopped(r)))?;
    if let Some(lim) = limits {
        precheck_jpeg_header(jpeg_data, lim, 1)?;
    }
    let decoded = jpeg_decoder_with_limits(JpegPixelFormat::Gray, limits)
        .decode(jpeg_data, stop)
        .map_err(map_jpeg_decode_error)?;

    let width = decoded.width();
    let height = decoded.height();
    if let Some(lim) = limits {
        lim.check_output(width, height, 1)?;
    }
    let pixels = decoded
        .pixels_u8()
        .ok_or_else(|| at!(Error::DecodeError("No pixel data in decoded JPEG".into())))?;
    match decoded.bytes_per_pixel() {
        1 => Ok((width, height, pixels.to_vec())),
        bpp => Err(at!(Error::DecodeError(format!(
            "Unsupported bytes per pixel for grayscale gain-map decode: {bpp}"
        )))),
    }
}

/// Decode a gain-map JPEG to tight interleaved RGB8 bytes.
///
/// Used for per-channel (multi-channel) maps and as the fallback when
/// Gray output is unavailable: RGB output is universally supported
/// (grayscale codestreams expand to identical channels), and per-channel
/// maps must not be flattened to luma (#27).
fn decode_jpeg_to_rgb_bytes(
    jpeg_data: &[u8],
    limits: Option<&ResourceLimits>,
    stop: impl Stop,
) -> Result<(u32, u32, Vec<u8>)> {
    use zenjpeg::decoder::PixelFormat as JpegPixelFormat;
    stop.check().map_err(|r| at!(Error::Stopped(r)))?;
    if let Some(lim) = limits {
        precheck_jpeg_header(jpeg_data, lim, 3)?;
    }
    let decoded = jpeg_decoder_with_limits(JpegPixelFormat::Rgb, limits)
        .decode(jpeg_data, stop)
        .map_err(map_jpeg_decode_error)?;

    let width = decoded.width();
    let height = decoded.height();
    if let Some(lim) = limits {
        lim.check_output(width, height, 3)?;
    }
    let pixels = decoded
        .pixels_u8()
        .ok_or_else(|| at!(Error::DecodeError("No pixel data in decoded JPEG".into())))?;
    match decoded.bytes_per_pixel() {
        3 => Ok((width, height, pixels.to_vec())),
        bpp => Err(at!(Error::DecodeError(format!(
            "Unsupported bytes per pixel for RGB gain-map decode: {bpp}"
        )))),
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
    fn test_checked_output_len_normal() {
        assert_eq!(checked_output_len(64, 64, 4).unwrap(), 64 * 64 * 4);
        assert_eq!(checked_output_len(1, 1, 16).unwrap(), 16);
    }

    #[test]
    fn test_checked_output_len_overflow_is_err_not_panic() {
        // u32::MAX * u32::MAX * 4 overflows u64::try_from(usize) on every
        // target — must be a clean Err, never a wrap or panic.
        let r = checked_output_len(u32::MAX, u32::MAX, 4);
        assert!(matches!(r.unwrap_err().error(), Error::LimitExceeded(_)));
    }

    #[test]
    fn test_resource_limits_defaults() {
        let lim = ResourceLimits::default();
        assert_eq!(lim.max_pixels(), ultrahdr_core::limits::MAX_TOTAL_PIXELS);
        assert_eq!(lim.max_memory(), None);
        // In-budget dims pass
        assert!(lim.check_output(1920, 1080, 4).is_ok());
        // Over the 500 MP hard cap fails even at default
        assert!(matches!(
            lim.check_output(30000, 30000, 4).unwrap_err().error(),
            Error::LimitExceeded(_)
        ));
    }

    #[test]
    fn test_resource_limits_pixel_cap() {
        let lim = ResourceLimits::new().with_max_pixels(16);
        assert!(lim.check_output(4, 4, 4).is_ok());
        assert!(matches!(
            lim.check_output(5, 4, 4).unwrap_err().error(),
            Error::LimitExceeded(_)
        ));
    }

    #[test]
    fn test_resource_limits_cannot_loosen_hard_cap() {
        // A caller cap above 500 MP is clamped down to the crate hard cap.
        let lim = ResourceLimits::new().with_max_pixels(u64::MAX);
        assert!(matches!(
            lim.check_output(30000, 30000, 4).unwrap_err().error(),
            Error::LimitExceeded(_)
        ));
    }

    #[test]
    fn test_resource_limits_memory_cap() {
        // 64x64x4 = 16384 bytes
        let lim = ResourceLimits::new().with_max_memory(16384);
        assert!(lim.check_output(64, 64, 4).is_ok());
        assert!(matches!(
            lim.check_output(64, 65, 4).unwrap_err().error(),
            Error::LimitExceeded(_)
        ));
    }

    #[test]
    fn test_decoder_new_with_limits_plain_jpeg() {
        let data = vec![0xFF, 0xD8, 0xFF, 0xD9];
        let decoder = Decoder::new_with_limits(&data, ResourceLimits::default()).unwrap();
        assert!(decoder.resource_limits().is_some());
        assert!(Decoder::new(&data).unwrap().resource_limits().is_none());
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
