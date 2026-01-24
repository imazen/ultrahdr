//! Streaming APIs for low-memory gain map processing.
//!
//! This module provides four streaming processors, two for decode (SDR+gainmap → HDR)
//! and two for encode (HDR+SDR → gainmap):
//!
//! | Type | Direction | Memory Model | Use When |
//! |------|-----------|--------------|----------|
//! | [`RowDecoder`] | Decode | Full gainmap in memory | Gainmap fits in RAM |
//! | [`StreamDecoder`] | Decode | Ring buffer (16 rows) | Parallel JPEG decode |
//! | [`RowEncoder`] | Encode | Synchronized batches | Same-rate HDR/SDR |
//! | [`StreamEncoder`] | Encode | Independent buffers | Parallel JPEG decode |
//!
//! # Memory Comparison (4K image, 3840x2160)
//!
//! | API | Peak Memory |
//! |-----|-------------|
//! | Full decode | ~166 MB |
//! | Streaming decode (16 rows) | ~2 MB |
//! | Full encode | ~170 MB |
//! | Streaming encode (16 rows) | ~4 MB |
//!
//! # Choosing a Decoder
//!
//! - **[`RowDecoder`]**: Load gainmap fully, then stream SDR rows. Best when gainmap
//!   is small (e.g., 1/4 resolution) and can remain in memory.
//! - **[`StreamDecoder`]**: Stream both SDR and gamut rows in parallel. Best for
//!   true parallel decode of MPF primary/secondary images with minimal memory.
//!
//! # Choosing an Encoder
//!
//! - **[`RowEncoder`]**: Feed synchronized HDR+SDR batches. Best when both inputs
//!   come from the same decode loop at the same rate.
//! - **[`StreamEncoder`]**: Feed HDR and SDR rows independently at different rates.
//!   Best for parallel decode of separate HDR/SDR sources.
//!
//! # Example: Streaming HDR Reconstruction
//!
//! ```ignore
//! use ultrahdr_core::gainmap::streaming::{RowDecoder, DecodeInput};
//!
//! let mut decoder = RowDecoder::new(
//!     gainmap, metadata, width, height, 4.0, HdrOutputFormat::LinearFloat, gamut
//! )?;
//!
//! // Process in 16-row batches (typical JPEG MCU height)
//! for batch_start in (0..height).step_by(16) {
//!     let batch_height = 16.min(height - batch_start);
//!     let sdr_batch = decode_next_rows(batch_height);
//!     let hdr_batch = decoder.process_rows(&sdr_batch, batch_height)?;
//!     write_hdr_rows(batch_start, &hdr_batch);
//! }
//! ```

use alloc::format;
use alloc::vec;
use alloc::vec::Vec;

use crate::color::gamut::rgb_to_luminance;
use crate::color::transfer::{pq_oetf, srgb_eotf, srgb_oetf};
use crate::types::{
    ColorGamut, ColorTransfer, Error, GainMap, GainMapMetadata, PixelFormat, Result,
};

use super::apply::HdrOutputFormat;
use super::compute::GainMapConfig;

// ============================================================================
// Row Decoder (full gainmap in memory)
// ============================================================================

/// Row-based HDR decoder that holds the full gainmap in memory.
///
/// Accepts SDR image rows in configurable batches and outputs reconstructed HDR rows.
/// The gainmap must be loaded entirely before processing begins—typically acceptable
/// since gainmaps are often 1/4 resolution or smaller.
///
/// # Memory Usage
///
/// - Gainmap: `gm_width × gm_height × channels` bytes (e.g., 960×540×1 = ~500KB for 4K)
/// - Per batch: `width × batch_rows × output_bpp` bytes
///
/// For streaming gainmap input, use [`StreamDecoder`] instead.
#[derive(Debug)]
pub struct RowDecoder {
    /// Full gain map (typically small, e.g., 1/4 resolution)
    gainmap: GainMap,
    /// Gain map metadata
    metadata: GainMapMetadata,
    /// Output image width
    width: u32,
    /// Output image height
    height: u32,
    /// Weight factor for display boost
    weight: f32,
    /// Output format
    output_format: HdrOutputFormat,
    /// Bytes per output pixel
    output_bpp: usize,
    /// Current row being processed
    current_row: u32,
    /// Source gamut
    gamut: ColorGamut,
    /// Input format configuration
    input_config: DecodeInput,
}

/// Input configuration for streaming decoders ([`RowDecoder`] and [`StreamDecoder`]).
///
/// Describes the pixel format, stride, and processing mode for SDR input data.
#[derive(Debug, Clone)]
pub struct DecodeInput {
    /// Pixel format of SDR input data.
    pub format: PixelFormat,
    /// Bytes per row. Set to 0 for auto-calculation from width.
    pub stride: u32,
    /// If true, expect Y plane only (single channel) for luminance-based processing.
    pub y_only: bool,
}

impl Default for DecodeInput {
    fn default() -> Self {
        Self {
            format: PixelFormat::Rgba8,
            stride: 0,
            y_only: false,
        }
    }
}

impl DecodeInput {
    /// Create config for RGBA8 input with packed stride.
    pub fn rgba8(width: u32) -> Self {
        Self {
            format: PixelFormat::Rgba8,
            stride: width * 4,
            y_only: false,
        }
    }

    /// Create config for RGB8 input with packed stride.
    pub fn rgb8(width: u32) -> Self {
        Self {
            format: PixelFormat::Rgb8,
            stride: width * 3,
            y_only: false,
        }
    }

    /// Create config for Y-only luminance input (8-bit grayscale).
    pub fn y_only(width: u32) -> Self {
        Self {
            format: PixelFormat::Gray8,
            stride: width,
            y_only: true,
        }
    }

    /// Builder: set custom stride (for row padding from decoders).
    pub fn with_stride(mut self, stride: u32) -> Self {
        self.stride = stride;
        self
    }
}

impl RowDecoder {
    /// Create a new streaming HDR reconstructor.
    ///
    /// # Arguments
    /// * `gainmap` - The gain map (loaded entirely, typically small)
    /// * `metadata` - Gain map metadata
    /// * `width` - Output image width
    /// * `height` - Output image height
    /// * `display_boost` - HDR boost factor (e.g., 4.0)
    /// * `output_format` - Desired output format
    /// * `gamut` - Color gamut
    pub fn new(
        gainmap: GainMap,
        metadata: GainMapMetadata,
        width: u32,
        height: u32,
        display_boost: f32,
        output_format: HdrOutputFormat,
        gamut: ColorGamut,
    ) -> Result<Self> {
        Self::with_input_config(
            gainmap,
            metadata,
            width,
            height,
            display_boost,
            output_format,
            gamut,
            DecodeInput::rgba8(width),
        )
    }

    /// Create with custom input configuration.
    #[allow(clippy::too_many_arguments)]
    pub fn with_input_config(
        gainmap: GainMap,
        metadata: GainMapMetadata,
        width: u32,
        height: u32,
        display_boost: f32,
        output_format: HdrOutputFormat,
        gamut: ColorGamut,
        input_config: DecodeInput,
    ) -> Result<Self> {
        let weight = calculate_weight(display_boost, &metadata);

        let output_bpp = match output_format {
            HdrOutputFormat::LinearFloat => 16,
            HdrOutputFormat::Pq1010102 => 4,
            HdrOutputFormat::Srgb8 => 4,
        };

        Ok(Self {
            gainmap,
            metadata,
            width,
            height,
            weight,
            output_format,
            output_bpp,
            current_row: 0,
            gamut,
            input_config,
        })
    }

    /// Get the source color gamut.
    pub fn gamut(&self) -> ColorGamut {
        self.gamut
    }

    /// Process multiple SDR rows and return corresponding HDR rows.
    ///
    /// # Arguments
    /// * `sdr_data` - SDR pixel data for `num_rows` rows
    /// * `num_rows` - Number of rows in the batch
    ///
    /// # Returns
    /// HDR pixel data for all rows in the batch.
    pub fn process_rows(&mut self, sdr_data: &[u8], num_rows: u32) -> Result<Vec<u8>> {
        let remaining = self.height - self.current_row;
        let actual_rows = num_rows.min(remaining);

        if actual_rows == 0 {
            return Err(Error::InvalidPixelData("all rows already processed".into()));
        }

        let input_stride = if self.input_config.stride > 0 {
            self.input_config.stride as usize
        } else {
            self.width as usize * self.input_config.format.bytes_per_pixel().unwrap_or(4)
        };

        let expected_len = input_stride * actual_rows as usize;
        if sdr_data.len() < expected_len {
            return Err(Error::InvalidPixelData(format!(
                "input data too short: {} < {}",
                sdr_data.len(),
                expected_len
            )));
        }

        let output_stride = self.width as usize * self.output_bpp;
        let mut output = vec![0u8; output_stride * actual_rows as usize];

        for row_offset in 0..actual_rows {
            let y = self.current_row + row_offset;
            let input_row = &sdr_data[row_offset as usize * input_stride..];
            let output_row = &mut output
                [row_offset as usize * output_stride..(row_offset as usize + 1) * output_stride];

            self.process_single_row(y, input_row, output_row);
        }

        self.current_row += actual_rows;
        Ok(output)
    }

    /// Process a single row (convenience wrapper).
    pub fn process_row(&mut self, sdr_row: &[u8]) -> Result<Vec<u8>> {
        self.process_rows(sdr_row, 1)
    }

    /// Process single row internally.
    fn process_single_row(&self, y: u32, input_row: &[u8], output_row: &mut [u8]) {
        for x in 0..self.width {
            // Get SDR pixel
            let sdr_linear = if self.input_config.y_only {
                // Y-only mode: treat as grayscale
                let y_val = input_row[x as usize] as f32 / 255.0;
                let linear = srgb_eotf(y_val);
                [linear, linear, linear]
            } else {
                get_pixel_linear(input_row, x, &self.input_config)
            };

            // Sample gain map
            let gain = self.sample_gainmap(x, y);

            // Apply gain
            let hdr_linear = apply_gain(sdr_linear, gain, &self.metadata);

            // Write to output
            self.write_pixel(output_row, x, hdr_linear);
        }
    }

    /// Check if all rows have been processed.
    pub fn is_complete(&self) -> bool {
        self.current_row >= self.height
    }

    /// Get current row index.
    pub fn current_row(&self) -> u32 {
        self.current_row
    }

    /// Get total rows.
    pub fn total_rows(&self) -> u32 {
        self.height
    }

    /// Rows remaining to process.
    pub fn rows_remaining(&self) -> u32 {
        self.height - self.current_row
    }

    /// Reset for reprocessing.
    pub fn reset(&mut self) {
        self.current_row = 0;
    }

    /// Sample gainmap with bilinear interpolation.
    fn sample_gainmap(&self, x: u32, y: u32) -> [f32; 3] {
        let gm_x = (x as f32 / self.width as f32) * self.gainmap.width as f32;
        let gm_y = (y as f32 / self.height as f32) * self.gainmap.height as f32;

        let x0 = (gm_x.floor() as u32).min(self.gainmap.width - 1);
        let y0 = (gm_y.floor() as u32).min(self.gainmap.height - 1);
        let x1 = (x0 + 1).min(self.gainmap.width - 1);
        let y1 = (y0 + 1).min(self.gainmap.height - 1);

        let fx = gm_x - gm_x.floor();
        let fy = gm_y - gm_y.floor();

        if self.gainmap.channels == 1 {
            let v00 = self.gainmap.data[(y0 * self.gainmap.width + x0) as usize] as f32 / 255.0;
            let v10 = self.gainmap.data[(y0 * self.gainmap.width + x1) as usize] as f32 / 255.0;
            let v01 = self.gainmap.data[(y1 * self.gainmap.width + x0) as usize] as f32 / 255.0;
            let v11 = self.gainmap.data[(y1 * self.gainmap.width + x1) as usize] as f32 / 255.0;

            let v = bilinear(v00, v10, v01, v11, fx, fy);
            let gain = decode_gain(v, &self.metadata, 0, self.weight);
            [gain, gain, gain]
        } else {
            let mut gains = [0.0f32; 3];
            #[allow(clippy::needless_range_loop)]
            for c in 0..3 {
                let v00 = self.gainmap.data[(y0 * self.gainmap.width + x0) as usize * 3 + c] as f32
                    / 255.0;
                let v10 = self.gainmap.data[(y0 * self.gainmap.width + x1) as usize * 3 + c] as f32
                    / 255.0;
                let v01 = self.gainmap.data[(y1 * self.gainmap.width + x0) as usize * 3 + c] as f32
                    / 255.0;
                let v11 = self.gainmap.data[(y1 * self.gainmap.width + x1) as usize * 3 + c] as f32
                    / 255.0;

                let v = bilinear(v00, v10, v01, v11, fx, fy);
                gains[c] = decode_gain(v, &self.metadata, c, self.weight);
            }
            gains
        }
    }

    /// Write HDR pixel to output buffer.
    fn write_pixel(&self, output: &mut [u8], x: u32, hdr: [f32; 3]) {
        match self.output_format {
            HdrOutputFormat::LinearFloat => {
                let idx = x as usize * 16;
                output[idx..idx + 4].copy_from_slice(&hdr[0].to_le_bytes());
                output[idx + 4..idx + 8].copy_from_slice(&hdr[1].to_le_bytes());
                output[idx + 8..idx + 12].copy_from_slice(&hdr[2].to_le_bytes());
                output[idx + 12..idx + 16].copy_from_slice(&1.0f32.to_le_bytes());
            }
            HdrOutputFormat::Pq1010102 => {
                let scale = 203.0 / 10000.0;
                let r_pq = pq_oetf(hdr[0].max(0.0) * scale);
                let g_pq = pq_oetf(hdr[1].max(0.0) * scale);
                let b_pq = pq_oetf(hdr[2].max(0.0) * scale);

                let r = (r_pq * 1023.0).round().clamp(0.0, 1023.0) as u32;
                let g = (g_pq * 1023.0).round().clamp(0.0, 1023.0) as u32;
                let b = (b_pq * 1023.0).round().clamp(0.0, 1023.0) as u32;
                let a = 3u32;

                let packed = r | (g << 10) | (b << 20) | (a << 30);
                let idx = x as usize * 4;
                output[idx..idx + 4].copy_from_slice(&packed.to_le_bytes());
            }
            HdrOutputFormat::Srgb8 => {
                let r = srgb_oetf(hdr[0].clamp(0.0, 1.0));
                let g = srgb_oetf(hdr[1].clamp(0.0, 1.0));
                let b = srgb_oetf(hdr[2].clamp(0.0, 1.0));

                let idx = x as usize * 4;
                output[idx] = (r * 255.0).round() as u8;
                output[idx + 1] = (g * 255.0).round() as u8;
                output[idx + 2] = (b * 255.0).round() as u8;
                output[idx + 3] = 255;
            }
        }
    }
}

// ============================================================================
// Stream Decoder (streamed gainmap, minimal memory)
// ============================================================================

/// Streaming HDR decoder that accepts both SDR and gainmap rows independently.
///
/// Unlike [`RowDecoder`] which loads the full gainmap upfront, this variant
/// streams both the SDR image and the gainmap simultaneously. This enables
/// true parallel decode of MPF primary (SDR) and secondary (gainmap) JPEG
/// images with minimal memory.
///
/// # Memory Model
///
/// - Gainmap ring buffer: 16 rows × `gm_width × channels` bytes
/// - At 4× scale factor (4K → 960px gainmap): ~15-46KB depending on channels
///
/// # Usage Pattern
///
/// ```ignore
/// let mut decoder = StreamDecoder::new(
///     metadata,
///     3840, 2160,  // SDR dimensions
///     960, 540, 1, // Gainmap dimensions + channels
///     4.0,         // display boost
///     HdrOutputFormat::LinearFloat,
///     ColorGamut::Bt709,
/// )?;
///
/// // Decode both JPEGs in parallel, feeding rows as they arrive
/// loop {
///     // Feed gainmap rows (can be ahead of SDR)
///     if let Some(gm_row) = gainmap_decoder.next_row() {
///         decoder.push_gainmap_row(&gm_row)?;
///     }
///
///     // Process SDR rows when gainmap data is ready
///     if let Some(sdr_rows) = sdr_decoder.next_rows(16) {
///         if decoder.can_process(16) {
///             let hdr = decoder.process_sdr_rows(&sdr_rows, 16)?;
///             output.write(&hdr);
///         }
///     }
/// }
/// ```
#[derive(Debug)]
pub struct StreamDecoder {
    /// Gain map metadata
    metadata: GainMapMetadata,
    /// SDR image width
    sdr_width: u32,
    /// SDR image height
    sdr_height: u32,
    /// Gainmap width
    gm_width: u32,
    /// Gainmap height
    gm_height: u32,
    /// Gainmap channels (1 or 3)
    gm_channels: u8,
    /// Weight factor for display boost
    weight: f32,
    /// Output format
    output_format: HdrOutputFormat,
    /// Bytes per output pixel
    output_bpp: usize,
    /// Current SDR row being processed
    current_sdr_row: u32,
    /// Current gainmap row received
    current_gm_row: u32,
    /// Source gamut
    gamut: ColorGamut,
    /// Input format configuration
    input_config: DecodeInput,
    /// Ring buffer of gainmap rows (up to 16)
    gm_buffer: GainMapRingBuffer,
}

/// Ring buffer for gainmap rows during streaming decode.
#[derive(Debug)]
struct GainMapRingBuffer {
    /// Row data storage
    rows: Vec<Vec<u8>>,
    /// First row index in buffer
    first_row: u32,
    /// Number of valid rows in buffer
    count: u32,
    /// Bytes per gainmap row
    row_bytes: usize,
    /// Buffer capacity (max rows)
    capacity: u32,
}

impl GainMapRingBuffer {
    fn new(gm_width: u32, gm_channels: u8, capacity: u32) -> Self {
        let row_bytes = gm_width as usize * gm_channels as usize;
        Self {
            rows: vec![vec![0u8; row_bytes]; capacity as usize],
            first_row: 0,
            count: 0,
            row_bytes,
            capacity,
        }
    }

    fn push(&mut self, row_index: u32, data: &[u8]) {
        let slot = (row_index % self.capacity) as usize;
        let copy_len = data.len().min(self.row_bytes);
        self.rows[slot][..copy_len].copy_from_slice(&data[..copy_len]);

        if self.count == 0 {
            self.first_row = row_index;
        }
        // Update range
        let new_last = row_index + 1;
        let new_first = new_last.saturating_sub(self.capacity);
        if new_first > self.first_row {
            self.first_row = new_first;
        }
        self.count = (new_last - self.first_row).min(self.capacity);
    }

    fn contains(&self, row: u32) -> bool {
        if self.count == 0 {
            return false;
        }
        row >= self.first_row && row < self.first_row + self.count
    }

    fn get(&self, row: u32) -> Option<&[u8]> {
        if !self.contains(row) {
            return None;
        }
        let slot = (row % self.capacity) as usize;
        Some(&self.rows[slot])
    }
}

impl StreamDecoder {
    /// Create a new streaming decoder.
    ///
    /// # Arguments
    /// * `metadata` - Gain map metadata
    /// * `sdr_width`, `sdr_height` - SDR image dimensions
    /// * `gm_width`, `gm_height` - Gainmap dimensions
    /// * `gm_channels` - Gainmap channels (1 for luminance, 3 for RGB)
    /// * `display_boost` - HDR boost factor (e.g., 4.0)
    /// * `output_format` - Desired output format
    /// * `gamut` - Color gamut
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        metadata: GainMapMetadata,
        sdr_width: u32,
        sdr_height: u32,
        gm_width: u32,
        gm_height: u32,
        gm_channels: u8,
        display_boost: f32,
        output_format: HdrOutputFormat,
        gamut: ColorGamut,
    ) -> Result<Self> {
        Self::with_input_config(
            metadata,
            sdr_width,
            sdr_height,
            gm_width,
            gm_height,
            gm_channels,
            display_boost,
            output_format,
            gamut,
            DecodeInput::rgba8(sdr_width),
        )
    }

    /// Create with custom input configuration.
    #[allow(clippy::too_many_arguments)]
    pub fn with_input_config(
        metadata: GainMapMetadata,
        sdr_width: u32,
        sdr_height: u32,
        gm_width: u32,
        gm_height: u32,
        gm_channels: u8,
        display_boost: f32,
        output_format: HdrOutputFormat,
        gamut: ColorGamut,
        input_config: DecodeInput,
    ) -> Result<Self> {
        let weight = calculate_weight(display_boost, &metadata);

        let output_bpp = match output_format {
            HdrOutputFormat::LinearFloat => 16,
            HdrOutputFormat::Pq1010102 => 4,
            HdrOutputFormat::Srgb8 => 4,
        };

        // Buffer 16 gainmap rows
        let gm_buffer = GainMapRingBuffer::new(gm_width, gm_channels, 16);

        Ok(Self {
            metadata,
            sdr_width,
            sdr_height,
            gm_width,
            gm_height,
            gm_channels,
            weight,
            output_format,
            output_bpp,
            current_sdr_row: 0,
            current_gm_row: 0,
            gamut,
            input_config,
            gm_buffer,
        })
    }

    /// Get the source color gamut.
    pub fn gamut(&self) -> ColorGamut {
        self.gamut
    }

    /// Push a gainmap row into the buffer.
    ///
    /// Rows should be pushed in order. The buffer holds up to 16 rows.
    pub fn push_gainmap_row(&mut self, data: &[u8]) -> Result<()> {
        if self.current_gm_row >= self.gm_height {
            return Err(Error::InvalidPixelData(
                "all gainmap rows already received".into(),
            ));
        }

        self.gm_buffer.push(self.current_gm_row, data);
        self.current_gm_row += 1;
        Ok(())
    }

    /// Push multiple gainmap rows at once.
    pub fn push_gainmap_rows(&mut self, data: &[u8], num_rows: u32) -> Result<()> {
        let row_bytes = self.gm_width as usize * self.gm_channels as usize;
        for i in 0..num_rows {
            let start = i as usize * row_bytes;
            let end = start + row_bytes;
            if end > data.len() {
                return Err(Error::InvalidPixelData("gainmap data too short".into()));
            }
            self.push_gainmap_row(&data[start..end])?;
        }
        Ok(())
    }

    /// Check if we have enough gainmap data to process the given number of SDR rows.
    pub fn can_process(&self, num_sdr_rows: u32) -> bool {
        if self.current_sdr_row >= self.sdr_height {
            return false;
        }

        let last_sdr_row = (self.current_sdr_row + num_sdr_rows - 1).min(self.sdr_height - 1);

        // Calculate which gainmap rows we need for bilinear interpolation
        let gm_y_last = (last_sdr_row as f32 / self.sdr_height as f32) * self.gm_height as f32;
        let gm_y1_needed = (gm_y_last.ceil() as u32).min(self.gm_height - 1);

        // We need up to gm_y1_needed in the buffer
        self.current_gm_row > gm_y1_needed || self.gm_buffer.contains(gm_y1_needed)
    }

    /// Get the next gainmap row index needed for processing.
    ///
    /// Returns `None` if all needed rows are already buffered.
    pub fn next_gainmap_row_needed(&self) -> Option<u32> {
        if self.current_gm_row >= self.gm_height {
            return None;
        }

        // What's the last SDR row we might want to process?
        let look_ahead_sdr = (self.current_sdr_row + 64).min(self.sdr_height); // ~4 batches ahead
        let gm_y_needed = ((look_ahead_sdr as f32 / self.sdr_height as f32) * self.gm_height as f32)
            .ceil() as u32;
        let gm_y_needed = gm_y_needed.min(self.gm_height);

        if self.current_gm_row < gm_y_needed {
            Some(self.current_gm_row)
        } else {
            None
        }
    }

    /// Process SDR rows and return HDR output.
    ///
    /// Call `can_process()` first to ensure enough gainmap data is buffered.
    pub fn process_sdr_rows(&mut self, sdr_data: &[u8], num_rows: u32) -> Result<Vec<u8>> {
        let remaining = self.sdr_height - self.current_sdr_row;
        let actual_rows = num_rows.min(remaining);

        if actual_rows == 0 {
            return Err(Error::InvalidPixelData(
                "all SDR rows already processed".into(),
            ));
        }

        // Verify we have gainmap data
        if !self.can_process(actual_rows) {
            return Err(Error::InvalidPixelData(
                "insufficient gainmap data buffered - call push_gainmap_row first".into(),
            ));
        }

        let input_stride = if self.input_config.stride > 0 {
            self.input_config.stride as usize
        } else {
            self.sdr_width as usize * self.input_config.format.bytes_per_pixel().unwrap_or(4)
        };

        let expected_len = input_stride * actual_rows as usize;
        if sdr_data.len() < expected_len {
            return Err(Error::InvalidPixelData(format!(
                "SDR data too short: {} < {}",
                sdr_data.len(),
                expected_len
            )));
        }

        let output_stride = self.sdr_width as usize * self.output_bpp;
        let mut output = vec![0u8; output_stride * actual_rows as usize];

        for row_offset in 0..actual_rows {
            let y = self.current_sdr_row + row_offset;
            let input_row = &sdr_data[row_offset as usize * input_stride..];
            let output_row = &mut output
                [row_offset as usize * output_stride..(row_offset as usize + 1) * output_stride];

            self.process_single_row(y, input_row, output_row);
        }

        self.current_sdr_row += actual_rows;
        Ok(output)
    }

    /// Process a single SDR row.
    fn process_single_row(&self, y: u32, input_row: &[u8], output_row: &mut [u8]) {
        for x in 0..self.sdr_width {
            let sdr_linear = self.get_sdr_linear(input_row, x);
            let gain = self.sample_gainmap(x, y);
            let hdr = apply_gain_dual(sdr_linear, gain, &self.metadata);
            self.write_pixel(output_row, x, hdr);
        }
    }

    /// Get linear RGB from SDR input.
    fn get_sdr_linear(&self, row: &[u8], x: u32) -> [f32; 3] {
        if self.input_config.y_only {
            let y_val = row.get(x as usize).copied().unwrap_or(128);
            let linear = srgb_eotf(y_val as f32 / 255.0);
            [linear, linear, linear]
        } else {
            match self.input_config.format {
                PixelFormat::Rgba8 => {
                    let idx = x as usize * 4;
                    let r = row.get(idx).copied().unwrap_or(128) as f32 / 255.0;
                    let g = row.get(idx + 1).copied().unwrap_or(128) as f32 / 255.0;
                    let b = row.get(idx + 2).copied().unwrap_or(128) as f32 / 255.0;
                    [srgb_eotf(r), srgb_eotf(g), srgb_eotf(b)]
                }
                PixelFormat::Rgb8 => {
                    let idx = x as usize * 3;
                    let r = row.get(idx).copied().unwrap_or(128) as f32 / 255.0;
                    let g = row.get(idx + 1).copied().unwrap_or(128) as f32 / 255.0;
                    let b = row.get(idx + 2).copied().unwrap_or(128) as f32 / 255.0;
                    [srgb_eotf(r), srgb_eotf(g), srgb_eotf(b)]
                }
                _ => [0.18, 0.18, 0.18],
            }
        }
    }

    /// Sample gainmap with bilinear interpolation from ring buffer.
    fn sample_gainmap(&self, x: u32, y: u32) -> [f32; 3] {
        let gm_x = (x as f32 / self.sdr_width as f32) * self.gm_width as f32;
        let gm_y = (y as f32 / self.sdr_height as f32) * self.gm_height as f32;

        let x0 = (gm_x.floor() as u32).min(self.gm_width - 1);
        let y0 = (gm_y.floor() as u32).min(self.gm_height - 1);
        let x1 = (x0 + 1).min(self.gm_width - 1);
        let y1 = (y0 + 1).min(self.gm_height - 1);

        let fx = gm_x - gm_x.floor();
        let fy = gm_y - gm_y.floor();

        // Get rows from ring buffer (fall back to 128 if not available)
        let row0 = self.gm_buffer.get(y0);
        let row1 = self.gm_buffer.get(y1);

        if self.gm_channels == 1 {
            let v00 = Self::sample_row_gray(row0, x0);
            let v10 = Self::sample_row_gray(row0, x1);
            let v01 = Self::sample_row_gray(row1, x0);
            let v11 = Self::sample_row_gray(row1, x1);

            let v = bilinear(v00, v10, v01, v11, fx, fy);
            let gain = decode_gain(v, &self.metadata, 0, self.weight);
            [gain, gain, gain]
        } else {
            let mut gains = [0.0f32; 3];
            #[allow(clippy::needless_range_loop)]
            for c in 0..3 {
                let v00 = Self::sample_row_rgb(row0, x0, c);
                let v10 = Self::sample_row_rgb(row0, x1, c);
                let v01 = Self::sample_row_rgb(row1, x0, c);
                let v11 = Self::sample_row_rgb(row1, x1, c);

                let v = bilinear(v00, v10, v01, v11, fx, fy);
                gains[c] = decode_gain(v, &self.metadata, c, self.weight);
            }
            gains
        }
    }

    #[inline]
    fn sample_row_gray(row: Option<&[u8]>, x: u32) -> f32 {
        row.and_then(|r| r.get(x as usize).copied()).unwrap_or(128) as f32 / 255.0
    }

    #[inline]
    fn sample_row_rgb(row: Option<&[u8]>, x: u32, c: usize) -> f32 {
        row.and_then(|r| r.get(x as usize * 3 + c).copied())
            .unwrap_or(128) as f32
            / 255.0
    }

    /// Write HDR pixel to output buffer.
    fn write_pixel(&self, output: &mut [u8], x: u32, hdr: [f32; 3]) {
        match self.output_format {
            HdrOutputFormat::LinearFloat => {
                let idx = x as usize * 16;
                output[idx..idx + 4].copy_from_slice(&hdr[0].to_le_bytes());
                output[idx + 4..idx + 8].copy_from_slice(&hdr[1].to_le_bytes());
                output[idx + 8..idx + 12].copy_from_slice(&hdr[2].to_le_bytes());
                output[idx + 12..idx + 16].copy_from_slice(&1.0f32.to_le_bytes());
            }
            HdrOutputFormat::Pq1010102 => {
                let scale = 203.0 / 10000.0;
                let r_pq = pq_oetf(hdr[0].max(0.0) * scale);
                let g_pq = pq_oetf(hdr[1].max(0.0) * scale);
                let b_pq = pq_oetf(hdr[2].max(0.0) * scale);

                let r = (r_pq * 1023.0).round().clamp(0.0, 1023.0) as u32;
                let g = (g_pq * 1023.0).round().clamp(0.0, 1023.0) as u32;
                let b = (b_pq * 1023.0).round().clamp(0.0, 1023.0) as u32;
                let a = 3u32;

                let packed = r | (g << 10) | (b << 20) | (a << 30);
                let idx = x as usize * 4;
                output[idx..idx + 4].copy_from_slice(&packed.to_le_bytes());
            }
            HdrOutputFormat::Srgb8 => {
                let r = srgb_oetf(hdr[0].clamp(0.0, 1.0));
                let g = srgb_oetf(hdr[1].clamp(0.0, 1.0));
                let b = srgb_oetf(hdr[2].clamp(0.0, 1.0));

                let idx = x as usize * 4;
                output[idx] = (r * 255.0).round() as u8;
                output[idx + 1] = (g * 255.0).round() as u8;
                output[idx + 2] = (b * 255.0).round() as u8;
                output[idx + 3] = 255;
            }
        }
    }

    /// Get remaining SDR rows to process.
    pub fn sdr_rows_remaining(&self) -> u32 {
        self.sdr_height - self.current_sdr_row
    }

    /// Get remaining gainmap rows to receive.
    pub fn gainmap_rows_remaining(&self) -> u32 {
        self.gm_height - self.current_gm_row
    }

    /// Check if reconstruction is complete.
    pub fn is_complete(&self) -> bool {
        self.current_sdr_row >= self.sdr_height
    }
}

/// Apply gain to SDR pixel (duplicated to avoid borrowing issues).
#[inline]
fn apply_gain_dual(sdr_linear: [f32; 3], gain: [f32; 3], metadata: &GainMapMetadata) -> [f32; 3] {
    [
        (sdr_linear[0] + metadata.offset_sdr[0]) * gain[0] - metadata.offset_hdr[0],
        (sdr_linear[1] + metadata.offset_sdr[1]) * gain[1] - metadata.offset_hdr[1],
        (sdr_linear[2] + metadata.offset_sdr[2]) * gain[2] - metadata.offset_hdr[2],
    ]
}

// ============================================================================
// Row Encoder (synchronized HDR+SDR batches)
// ============================================================================

/// Row-based gainmap encoder for synchronized HDR+SDR input.
///
/// Accepts HDR and SDR rows in synchronized batches (same rows at the same time),
/// buffering only what's needed for block sampling. Best when both inputs come
/// from the same decode loop.
///
/// For independent HDR/SDR streams at different rates, use [`StreamEncoder`].
#[derive(Debug)]
pub struct RowEncoder {
    /// Configuration
    config: GainMapConfig,
    /// Image width
    width: u32,
    /// Image height
    height: u32,
    /// Gain map width
    gm_width: u32,
    /// Gain map height
    gm_height: u32,
    /// Scale factor
    scale: u32,
    /// Current input row
    current_input_row: u32,
    /// Current gainmap output row
    current_gm_row: u32,
    /// HDR row buffer
    hdr_buffer: RowBuffer,
    /// SDR row buffer
    sdr_buffer: RowBuffer,
    /// Accumulated min/max boost
    actual_min_boost: f32,
    actual_max_boost: f32,
    /// Output gainmap rows
    gainmap_rows: Vec<Vec<u8>>,
    /// Gamuts for luminance calculation
    hdr_gamut: ColorGamut,
    sdr_gamut: ColorGamut,
}

/// Input configuration for streaming encoders ([`RowEncoder`] and [`StreamEncoder`]).
///
/// Describes pixel formats, strides, transfer functions, and gamuts for both
/// HDR and SDR input streams.
#[derive(Debug, Clone)]
pub struct EncodeInput {
    /// HDR pixel format.
    pub hdr_format: PixelFormat,
    /// HDR row stride in bytes. Set to 0 for auto-calculation.
    pub hdr_stride: u32,
    /// HDR transfer function (typically Linear).
    pub hdr_transfer: ColorTransfer,
    /// HDR color gamut.
    pub hdr_gamut: ColorGamut,
    /// SDR pixel format.
    pub sdr_format: PixelFormat,
    /// SDR row stride in bytes. Set to 0 for auto-calculation.
    pub sdr_stride: u32,
    /// SDR color gamut.
    pub sdr_gamut: ColorGamut,
    /// If true, use Y-only luminance mode for faster single-channel processing.
    pub y_only: bool,
}

impl Default for EncodeInput {
    fn default() -> Self {
        Self {
            hdr_format: PixelFormat::Rgba16F,
            hdr_stride: 0,
            hdr_transfer: ColorTransfer::Linear,
            hdr_gamut: ColorGamut::Bt709,
            sdr_format: PixelFormat::Rgba8,
            sdr_stride: 0,
            sdr_gamut: ColorGamut::Bt709,
            y_only: false,
        }
    }
}

impl EncodeInput {
    /// Create config for RGBA16F HDR + RGBA8 SDR (common case).
    pub fn hdr16f_sdr8(width: u32) -> Self {
        Self {
            hdr_format: PixelFormat::Rgba16F,
            hdr_stride: width * 8,
            hdr_transfer: ColorTransfer::Linear,
            hdr_gamut: ColorGamut::Bt709,
            sdr_format: PixelFormat::Rgba8,
            sdr_stride: width * 4,
            sdr_gamut: ColorGamut::Bt709,
            y_only: false,
        }
    }

    /// Create config for Y-only luminance mode (faster single-channel processing).
    pub fn y_only(width: u32) -> Self {
        Self {
            hdr_format: PixelFormat::Gray8,
            hdr_stride: width,
            hdr_transfer: ColorTransfer::Linear,
            hdr_gamut: ColorGamut::Bt709,
            sdr_format: PixelFormat::Gray8,
            sdr_stride: width,
            sdr_gamut: ColorGamut::Bt709,
            y_only: true,
        }
    }
}

/// Buffer for storing a sliding window of rows.
#[derive(Debug)]
struct RowBuffer {
    rows: Vec<Vec<u8>>,
    first_row: u32,
    count: u32,
    format: PixelFormat,
    transfer: ColorTransfer,
    stride: usize,
}

impl RowBuffer {
    fn new(
        capacity: usize,
        width: u32,
        format: PixelFormat,
        transfer: ColorTransfer,
        stride: u32,
    ) -> Self {
        let row_stride = if stride > 0 {
            stride as usize
        } else {
            width as usize * format.bytes_per_pixel().unwrap_or(4)
        };
        Self {
            rows: vec![vec![0u8; row_stride]; capacity],
            first_row: 0,
            count: 0,
            format,
            transfer,
            stride: row_stride,
        }
    }

    fn push_row(&mut self, row: u32, data: &[u8]) {
        let capacity = self.rows.len() as u32;
        let idx = (row % capacity) as usize;
        let copy_len = data.len().min(self.rows[idx].len());
        self.rows[idx][..copy_len].copy_from_slice(&data[..copy_len]);

        if self.count == 0 {
            self.first_row = row;
        }
        self.count = self.count.saturating_add(1).min(capacity);
    }

    fn push_rows(&mut self, start_row: u32, data: &[u8], num_rows: u32) {
        for i in 0..num_rows {
            let row_data = &data[i as usize * self.stride..];
            self.push_row(start_row + i, row_data);
        }
    }

    fn get_pixel(&self, row: u32, x: u32, y_only: bool) -> Option<[f32; 3]> {
        let capacity = self.rows.len() as u32;

        if row < self.first_row || row >= self.first_row + self.count {
            return None;
        }

        let idx = (row % capacity) as usize;
        let row_data = &self.rows[idx];

        Some(get_linear_rgb_from_row(
            row_data,
            x,
            self.format,
            self.transfer,
            y_only,
        ))
    }
}

impl RowEncoder {
    /// Create a new row-based encoder.
    pub fn new(
        width: u32,
        height: u32,
        config: GainMapConfig,
        input_config: EncodeInput,
    ) -> Result<Self> {
        let scale = config.scale_factor.max(1) as u32;
        let gm_width = width.div_ceil(scale);
        let gm_height = height.div_ceil(scale);

        // Buffer size: scale rows + margin for batch processing
        let buffer_size = (scale as usize + 16).max(32);

        Ok(Self {
            config,
            width,
            height,
            gm_width,
            gm_height,
            scale,
            current_input_row: 0,
            current_gm_row: 0,
            hdr_buffer: RowBuffer::new(
                buffer_size,
                width,
                input_config.hdr_format,
                input_config.hdr_transfer,
                input_config.hdr_stride,
            ),
            sdr_buffer: RowBuffer::new(
                buffer_size,
                width,
                input_config.sdr_format,
                ColorTransfer::Srgb,
                input_config.sdr_stride,
            ),
            actual_min_boost: f32::MAX,
            actual_max_boost: f32::MIN,
            gainmap_rows: Vec::new(),
            hdr_gamut: input_config.hdr_gamut,
            sdr_gamut: input_config.sdr_gamut,
        })
    }

    /// Process multiple rows at once.
    ///
    /// # Returns
    /// Vector of completed gainmap rows (may be empty if waiting for more data).
    pub fn process_rows(
        &mut self,
        hdr_data: &[u8],
        sdr_data: &[u8],
        num_rows: u32,
    ) -> Result<Vec<Vec<u8>>> {
        let remaining = self.height - self.current_input_row;
        let actual_rows = num_rows.min(remaining);

        if actual_rows == 0 {
            return Ok(Vec::new());
        }

        // Buffer the rows
        self.hdr_buffer
            .push_rows(self.current_input_row, hdr_data, actual_rows);
        self.sdr_buffer
            .push_rows(self.current_input_row, sdr_data, actual_rows);

        self.current_input_row += actual_rows;

        // Compute any gainmap rows we now have data for
        let mut output_rows = Vec::new();

        while self.current_gm_row < self.gm_height {
            let target_y = self.current_gm_row * self.scale + self.scale / 2;
            let target_y = target_y.min(self.height - 1);

            if self.current_input_row > target_y {
                let gm_row = self.compute_gainmap_row()?;
                self.gainmap_rows.push(gm_row.clone());
                output_rows.push(gm_row);
                self.current_gm_row += 1;
            } else {
                break;
            }
        }

        Ok(output_rows)
    }

    /// Process a single pair of rows.
    pub fn process_row(&mut self, hdr_row: &[u8], sdr_row: &[u8]) -> Result<Option<Vec<u8>>> {
        let rows = self.process_rows(hdr_row, sdr_row, 1)?;
        Ok(rows.into_iter().next())
    }

    /// Finish and return the complete gainmap with metadata.
    pub fn finish(mut self) -> Result<(GainMap, GainMapMetadata)> {
        // Compute remaining rows
        while self.current_gm_row < self.gm_height {
            let gm_row = self.compute_gainmap_row()?;
            self.gainmap_rows.push(gm_row);
            self.current_gm_row += 1;
        }

        // Assemble gainmap
        let channels = if self.config.multi_channel { 3 } else { 1 };
        let mut gainmap = if self.config.multi_channel {
            GainMap::new_multichannel(self.gm_width, self.gm_height)?
        } else {
            GainMap::new(self.gm_width, self.gm_height)?
        };

        for (gy, row) in self.gainmap_rows.iter().enumerate() {
            let start = gy * self.gm_width as usize * channels;
            let end = start + row.len();
            gainmap.data[start..end].copy_from_slice(row);
        }

        let actual_min = self.actual_min_boost.max(self.config.min_content_boost);
        let actual_max = self.actual_max_boost.min(self.config.max_content_boost);

        let metadata = GainMapMetadata {
            max_content_boost: [actual_max; 3],
            min_content_boost: [actual_min; 3],
            gamma: [self.config.gamma; 3],
            offset_sdr: [self.config.offset_sdr; 3],
            offset_hdr: [self.config.offset_hdr; 3],
            hdr_capacity_min: self.config.hdr_capacity_min,
            hdr_capacity_max: self.config.hdr_capacity_max.max(actual_max),
            use_base_color_space: true,
        };

        Ok((gainmap, metadata))
    }

    /// Get progress info.
    pub fn progress(&self) -> (u32, u32) {
        (self.current_input_row, self.height)
    }

    /// Compute a single gainmap row.
    fn compute_gainmap_row(&mut self) -> Result<Vec<u8>> {
        let gy = self.current_gm_row;
        let channels = if self.config.multi_channel { 3 } else { 1 };
        let mut row = vec![0u8; self.gm_width as usize * channels];

        let log_min = self.config.min_content_boost.ln();
        let log_max = self.config.max_content_boost.ln();
        let log_range = log_max - log_min;

        let y_only = !self.config.multi_channel;

        for gx in 0..self.gm_width {
            let x = (gx * self.scale + self.scale / 2).min(self.width - 1);
            let y = (gy * self.scale + self.scale / 2).min(self.height - 1);

            let hdr_rgb = self
                .hdr_buffer
                .get_pixel(y, x, y_only)
                .unwrap_or([0.5, 0.5, 0.5]);
            let sdr_rgb = self
                .sdr_buffer
                .get_pixel(y, x, y_only)
                .unwrap_or([0.5, 0.5, 0.5]);

            if self.config.multi_channel {
                for c in 0..3 {
                    let gain = (hdr_rgb[c] + self.config.offset_hdr)
                        / (sdr_rgb[c] + self.config.offset_sdr).max(0.001);

                    self.actual_min_boost = self.actual_min_boost.min(gain);
                    self.actual_max_boost = self.actual_max_boost.max(gain);

                    let encoded = encode_gain(gain, log_min, log_range, &self.config);
                    row[gx as usize * 3 + c] = encoded;
                }
            } else {
                let hdr_lum = rgb_to_luminance(hdr_rgb, self.hdr_gamut);
                let sdr_lum = rgb_to_luminance(sdr_rgb, self.sdr_gamut);

                let gain = (hdr_lum + self.config.offset_hdr) / (sdr_lum + self.config.offset_sdr);

                self.actual_min_boost = self.actual_min_boost.min(gain);
                self.actual_max_boost = self.actual_max_boost.max(gain);

                let encoded = encode_gain(gain, log_min, log_range, &self.config);
                row[gx as usize] = encoded;
            }
        }

        Ok(row)
    }
}

// ============================================================================
// Helper Functions
// ============================================================================

fn calculate_weight(display_boost: f32, metadata: &GainMapMetadata) -> f32 {
    let log_display = display_boost.max(1.0).ln();
    let log_min = metadata.hdr_capacity_min.max(1.0).ln();
    let log_max = metadata.hdr_capacity_max.max(1.0).ln();

    if log_max <= log_min {
        return 1.0;
    }

    ((log_display - log_min) / (log_max - log_min)).clamp(0.0, 1.0)
}

#[inline]
fn bilinear(v00: f32, v10: f32, v01: f32, v11: f32, fx: f32, fy: f32) -> f32 {
    let top = v00 * (1.0 - fx) + v10 * fx;
    let bottom = v01 * (1.0 - fx) + v11 * fx;
    top * (1.0 - fy) + bottom * fy
}

fn decode_gain(normalized: f32, metadata: &GainMapMetadata, channel: usize, weight: f32) -> f32 {
    let gamma = metadata.gamma[channel];
    let linear = if gamma != 1.0 && gamma > 0.0 {
        normalized.powf(1.0 / gamma)
    } else {
        normalized
    };

    let log_min = metadata.min_content_boost[channel].ln();
    let log_max = metadata.max_content_boost[channel].ln();
    let log_gain = log_min + linear * (log_max - log_min);

    (log_gain * weight).exp()
}

fn apply_gain(sdr_linear: [f32; 3], gain: [f32; 3], metadata: &GainMapMetadata) -> [f32; 3] {
    [
        (sdr_linear[0] + metadata.offset_sdr[0]) * gain[0] - metadata.offset_hdr[0],
        (sdr_linear[1] + metadata.offset_sdr[1]) * gain[1] - metadata.offset_hdr[1],
        (sdr_linear[2] + metadata.offset_sdr[2]) * gain[2] - metadata.offset_hdr[2],
    ]
}

fn encode_gain(gain: f32, log_min: f32, log_range: f32, config: &GainMapConfig) -> u8 {
    let gain_clamped = gain.clamp(config.min_content_boost, config.max_content_boost);
    let log_gain = gain_clamped.ln();

    let normalized = if log_range > 0.0 {
        (log_gain - log_min) / log_range
    } else {
        0.5
    };

    let gamma_corrected = normalized.powf(config.gamma);
    (gamma_corrected * 255.0).round().clamp(0.0, 255.0) as u8
}

fn get_pixel_linear(row: &[u8], x: u32, config: &DecodeInput) -> [f32; 3] {
    get_linear_rgb_from_row(row, x, config.format, ColorTransfer::Srgb, config.y_only)
}

fn get_linear_rgb_from_row(
    row: &[u8],
    x: u32,
    format: PixelFormat,
    transfer: ColorTransfer,
    y_only: bool,
) -> [f32; 3] {
    if y_only || format == PixelFormat::Gray8 {
        let idx = x as usize;
        if idx < row.len() {
            let v = row[idx] as f32 / 255.0;
            let linear = srgb_eotf(v);
            return [linear, linear, linear];
        }
        return [0.5, 0.5, 0.5];
    }

    match format {
        PixelFormat::Rgba8 | PixelFormat::Rgb8 => {
            let bpp = if format == PixelFormat::Rgba8 { 4 } else { 3 };
            let idx = x as usize * bpp;
            if idx + 2 < row.len() {
                let r = row[idx] as f32 / 255.0;
                let g = row[idx + 1] as f32 / 255.0;
                let b = row[idx + 2] as f32 / 255.0;
                match transfer {
                    ColorTransfer::Srgb => [srgb_eotf(r), srgb_eotf(g), srgb_eotf(b)],
                    ColorTransfer::Linear => [r, g, b],
                    _ => [srgb_eotf(r), srgb_eotf(g), srgb_eotf(b)],
                }
            } else {
                [0.5, 0.5, 0.5]
            }
        }
        PixelFormat::Rgba16F => {
            let idx = x as usize * 8;
            if idx + 5 < row.len() {
                let r = half::f16::from_le_bytes([row[idx], row[idx + 1]]).to_f32();
                let g = half::f16::from_le_bytes([row[idx + 2], row[idx + 3]]).to_f32();
                let b = half::f16::from_le_bytes([row[idx + 4], row[idx + 5]]).to_f32();
                [r, g, b]
            } else {
                [0.5, 0.5, 0.5]
            }
        }
        PixelFormat::Rgba32F => {
            let idx = x as usize * 16;
            if idx + 11 < row.len() {
                let r = f32::from_le_bytes([row[idx], row[idx + 1], row[idx + 2], row[idx + 3]]);
                let g =
                    f32::from_le_bytes([row[idx + 4], row[idx + 5], row[idx + 6], row[idx + 7]]);
                let b =
                    f32::from_le_bytes([row[idx + 8], row[idx + 9], row[idx + 10], row[idx + 11]]);
                [r, g, b]
            } else {
                [0.5, 0.5, 0.5]
            }
        }
        _ => [0.5, 0.5, 0.5],
    }
}

// ============================================================================
// Stream Encoder (independent HDR/SDR streams)
// ============================================================================

/// Streaming gainmap encoder that accepts HDR and SDR rows independently.
///
/// Unlike [`RowEncoder`] which requires synchronized HDR+SDR batches,
/// this variant allows feeding HDR and SDR rows from separate decode streams
/// at different rates. Outputs gainmap rows as soon as sufficient data is buffered.
///
/// # Memory Model
///
/// - HDR ring buffer: 16–32 rows × `width × hdr_bpp` bytes
/// - SDR ring buffer: 16–32 rows × `width × sdr_bpp` bytes
/// - At 4K with RGBA8: ~240KB HDR + ~240KB SDR = ~480KB total
///
/// # Usage Pattern
///
/// ```ignore
/// let mut encoder = StreamEncoder::new(
///     3840, 2160,  // Image dimensions
///     config,      // GainMapConfig
///     input_config,
/// )?;
///
/// // Decode HDR and SDR JPEGs in parallel
/// loop {
///     // Feed rows as they arrive from decoders
///     if let Some(hdr_rows) = hdr_decoder.next_rows() {
///         encoder.push_hdr_rows(&hdr_rows, 16)?;
///     }
///     if let Some(sdr_rows) = sdr_decoder.next_rows() {
///         encoder.push_sdr_rows(&sdr_rows, 16)?;
///     }
///
///     // Collect gainmap rows as they become available
///     while let Some(gm_row) = encoder.take_gainmap_row() {
///         gainmap_output.push(gm_row);
///     }
/// }
///
/// let (gainmap, metadata) = encoder.finish()?;
/// ```
#[derive(Debug)]
pub struct StreamEncoder {
    /// Configuration
    config: GainMapConfig,
    /// Image width
    width: u32,
    /// Image height
    height: u32,
    /// Gain map width
    gm_width: u32,
    /// Gain map height
    gm_height: u32,
    /// Scale factor
    scale: u32,
    /// HDR row buffer
    hdr_rows: InputRingBuffer,
    /// SDR row buffer
    sdr_rows: InputRingBuffer,
    /// Next HDR row to receive
    next_hdr_row: u32,
    /// Next SDR row to receive
    next_sdr_row: u32,
    /// Next gainmap row to output
    next_gm_row: u32,
    /// Accumulated min/max boost for metadata
    actual_min_boost: f32,
    actual_max_boost: f32,
    /// Pending gainmap rows ready for output
    pending_gm_rows: Vec<Vec<u8>>,
    /// Gamuts for luminance calculation
    hdr_gamut: ColorGamut,
    sdr_gamut: ColorGamut,
    /// Input configurations
    hdr_format: PixelFormat,
    hdr_transfer: ColorTransfer,
    hdr_stride: usize,
    sdr_format: PixelFormat,
    sdr_stride: usize,
    y_only: bool,
}

/// Ring buffer for input rows.
#[derive(Debug)]
struct InputRingBuffer {
    rows: Vec<Vec<u8>>,
    first_row: u32,
    count: u32,
    row_bytes: usize,
    capacity: u32,
}

impl InputRingBuffer {
    fn new(row_bytes: usize, capacity: u32) -> Self {
        Self {
            rows: vec![vec![0u8; row_bytes]; capacity as usize],
            first_row: 0,
            count: 0,
            row_bytes,
            capacity,
        }
    }

    fn push(&mut self, row_index: u32, data: &[u8]) {
        let slot = (row_index % self.capacity) as usize;
        let copy_len = data.len().min(self.row_bytes);
        self.rows[slot][..copy_len].copy_from_slice(&data[..copy_len]);

        if self.count == 0 {
            self.first_row = row_index;
        }
        let new_last = row_index + 1;
        let new_first = new_last.saturating_sub(self.capacity);
        if new_first > self.first_row {
            self.first_row = new_first;
        }
        self.count = (new_last - self.first_row).min(self.capacity);
    }

    fn get(&self, row: u32) -> Option<&[u8]> {
        if self.count == 0 || row < self.first_row || row >= self.first_row + self.count {
            return None;
        }
        let slot = (row % self.capacity) as usize;
        Some(&self.rows[slot])
    }

    fn has_row(&self, row: u32) -> bool {
        self.count > 0 && row >= self.first_row && row < self.first_row + self.count
    }
}

impl StreamEncoder {
    /// Create a new streaming encoder.
    pub fn new(
        width: u32,
        height: u32,
        config: GainMapConfig,
        input_config: EncodeInput,
    ) -> Result<Self> {
        let scale = config.scale_factor.max(1) as u32;
        let gm_width = width.div_ceil(scale);
        let gm_height = height.div_ceil(scale);

        let hdr_stride = if input_config.hdr_stride > 0 {
            input_config.hdr_stride as usize
        } else {
            width as usize * input_config.hdr_format.bytes_per_pixel().unwrap_or(4)
        };

        let sdr_stride = if input_config.sdr_stride > 0 {
            input_config.sdr_stride as usize
        } else {
            width as usize * input_config.sdr_format.bytes_per_pixel().unwrap_or(4)
        };

        // Buffer enough rows for block sampling: scale + 1 for center sampling
        let buffer_capacity = (scale + 16).min(32);

        Ok(Self {
            config,
            width,
            height,
            gm_width,
            gm_height,
            scale,
            hdr_rows: InputRingBuffer::new(hdr_stride, buffer_capacity),
            sdr_rows: InputRingBuffer::new(sdr_stride, buffer_capacity),
            next_hdr_row: 0,
            next_sdr_row: 0,
            next_gm_row: 0,
            actual_min_boost: f32::MAX,
            actual_max_boost: f32::MIN,
            pending_gm_rows: Vec::new(),
            hdr_gamut: input_config.hdr_gamut,
            sdr_gamut: input_config.sdr_gamut,
            hdr_format: input_config.hdr_format,
            hdr_transfer: input_config.hdr_transfer,
            hdr_stride,
            sdr_format: input_config.sdr_format,
            sdr_stride,
            y_only: input_config.y_only,
        })
    }

    /// Push HDR rows into the buffer.
    pub fn push_hdr_rows(&mut self, data: &[u8], num_rows: u32) -> Result<()> {
        let remaining = self.height - self.next_hdr_row;
        let actual = num_rows.min(remaining);

        for i in 0..actual {
            let start = i as usize * self.hdr_stride;
            let end = start + self.hdr_stride;
            if end > data.len() {
                return Err(Error::InvalidPixelData("HDR data too short".into()));
            }
            self.hdr_rows.push(self.next_hdr_row + i, &data[start..end]);
        }
        self.next_hdr_row += actual;

        // Try to produce gainmap rows
        self.try_produce_gainmap_rows();
        Ok(())
    }

    /// Push a single HDR row.
    pub fn push_hdr_row(&mut self, data: &[u8]) -> Result<()> {
        if self.next_hdr_row >= self.height {
            return Err(Error::InvalidPixelData(
                "all HDR rows already received".into(),
            ));
        }
        self.hdr_rows.push(self.next_hdr_row, data);
        self.next_hdr_row += 1;
        self.try_produce_gainmap_rows();
        Ok(())
    }

    /// Push SDR rows into the buffer.
    pub fn push_sdr_rows(&mut self, data: &[u8], num_rows: u32) -> Result<()> {
        let remaining = self.height - self.next_sdr_row;
        let actual = num_rows.min(remaining);

        for i in 0..actual {
            let start = i as usize * self.sdr_stride;
            let end = start + self.sdr_stride;
            if end > data.len() {
                return Err(Error::InvalidPixelData("SDR data too short".into()));
            }
            self.sdr_rows.push(self.next_sdr_row + i, &data[start..end]);
        }
        self.next_sdr_row += actual;

        self.try_produce_gainmap_rows();
        Ok(())
    }

    /// Push a single SDR row.
    pub fn push_sdr_row(&mut self, data: &[u8]) -> Result<()> {
        if self.next_sdr_row >= self.height {
            return Err(Error::InvalidPixelData(
                "all SDR rows already received".into(),
            ));
        }
        self.sdr_rows.push(self.next_sdr_row, data);
        self.next_sdr_row += 1;
        self.try_produce_gainmap_rows();
        Ok(())
    }

    /// Take a completed gainmap row if available.
    pub fn take_gainmap_row(&mut self) -> Option<Vec<u8>> {
        if self.pending_gm_rows.is_empty() {
            None
        } else {
            Some(self.pending_gm_rows.remove(0))
        }
    }

    /// Check how many gainmap rows are ready.
    pub fn pending_gainmap_rows(&self) -> usize {
        self.pending_gm_rows.len()
    }

    /// Try to produce gainmap rows from buffered data.
    fn try_produce_gainmap_rows(&mut self) {
        while self.next_gm_row < self.gm_height {
            // Calculate which input rows we need for this gainmap row
            let center_y = self.next_gm_row * self.scale + self.scale / 2;
            let center_y = center_y.min(self.height - 1);

            // Check if we have the needed row
            if !self.hdr_rows.has_row(center_y) || !self.sdr_rows.has_row(center_y) {
                break;
            }

            // Compute this gainmap row
            let gm_row = self.compute_gainmap_row(self.next_gm_row);
            self.pending_gm_rows.push(gm_row);
            self.next_gm_row += 1;
        }
    }

    /// Compute a single gainmap row.
    fn compute_gainmap_row(&mut self, gm_y: u32) -> Vec<u8> {
        let channels = if self.config.multi_channel { 3 } else { 1 };
        let mut row = vec![0u8; self.gm_width as usize * channels];

        let center_y = (gm_y * self.scale + self.scale / 2).min(self.height - 1);
        let hdr_row_data = self.hdr_rows.get(center_y);
        let sdr_row_data = self.sdr_rows.get(center_y);

        let log_min = self.config.min_content_boost.ln();
        let log_max = self.config.max_content_boost.ln();
        let log_range = log_max - log_min;

        for gx in 0..self.gm_width {
            let center_x = (gx * self.scale + self.scale / 2).min(self.width - 1);

            let hdr_rgb = self.get_hdr_pixel(hdr_row_data, center_x);
            let sdr_rgb = self.get_sdr_pixel(sdr_row_data, center_x);

            if self.config.multi_channel {
                #[allow(clippy::needless_range_loop)]
                for c in 0..3 {
                    let hdr_c = hdr_rgb[c] + self.config.offset_hdr;
                    let sdr_c = sdr_rgb[c] + self.config.offset_sdr;
                    let gain = hdr_c / sdr_c.max(1e-6);

                    self.actual_min_boost = self.actual_min_boost.min(gain);
                    self.actual_max_boost = self.actual_max_boost.max(gain);

                    row[gx as usize * 3 + c] = encode_gain(gain, log_min, log_range, &self.config);
                }
            } else {
                let hdr_lum = rgb_to_luminance(hdr_rgb, self.hdr_gamut);
                let sdr_lum = rgb_to_luminance(sdr_rgb, self.sdr_gamut);
                let gain = (hdr_lum + self.config.offset_hdr) / (sdr_lum + self.config.offset_sdr);

                self.actual_min_boost = self.actual_min_boost.min(gain);
                self.actual_max_boost = self.actual_max_boost.max(gain);

                row[gx as usize] = encode_gain(gain, log_min, log_range, &self.config);
            }
        }

        row
    }

    /// Get HDR pixel in linear RGB.
    fn get_hdr_pixel(&self, row: Option<&[u8]>, x: u32) -> [f32; 3] {
        let row = match row {
            Some(r) => r,
            None => return [0.5, 0.5, 0.5],
        };

        if self.y_only || self.hdr_format == PixelFormat::Gray8 {
            let idx = x as usize;
            let v = row.get(idx).copied().unwrap_or(128) as f32 / 255.0;
            let linear = if self.hdr_transfer == ColorTransfer::Linear {
                v
            } else {
                srgb_eotf(v)
            };
            return [linear, linear, linear];
        }

        get_linear_rgb_from_row(row, x, self.hdr_format, self.hdr_transfer, false)
    }

    /// Get SDR pixel in linear RGB.
    fn get_sdr_pixel(&self, row: Option<&[u8]>, x: u32) -> [f32; 3] {
        let row = match row {
            Some(r) => r,
            None => return [0.5, 0.5, 0.5],
        };

        if self.y_only || self.sdr_format == PixelFormat::Gray8 {
            let idx = x as usize;
            let v = row.get(idx).copied().unwrap_or(128) as f32 / 255.0;
            return [srgb_eotf(v), srgb_eotf(v), srgb_eotf(v)];
        }

        get_linear_rgb_from_row(row, x, self.sdr_format, ColorTransfer::Srgb, false)
    }

    /// Finish encoding and return the gainmap and metadata.
    ///
    /// Consumes remaining pending rows.
    pub fn finish(mut self) -> Result<(GainMap, GainMapMetadata)> {
        // Collect any remaining pending rows
        let mut all_rows = Vec::new();
        all_rows.append(&mut self.pending_gm_rows);

        // Verify we got all rows
        if all_rows.len() != self.gm_height as usize {
            return Err(Error::InvalidPixelData(format!(
                "incomplete gainmap: {} of {} rows",
                all_rows.len(),
                self.gm_height
            )));
        }

        let channels = if self.config.multi_channel { 3u8 } else { 1u8 };
        let row_bytes = self.gm_width as usize * channels as usize;
        let mut data = Vec::with_capacity(row_bytes * self.gm_height as usize);

        for row in all_rows {
            data.extend_from_slice(&row);
        }

        let gainmap = GainMap {
            width: self.gm_width,
            height: self.gm_height,
            channels,
            data,
        };

        let actual_max = if self.actual_max_boost > f32::MIN {
            self.actual_max_boost
        } else {
            self.config.max_content_boost
        };
        let actual_min = if self.actual_min_boost < f32::MAX {
            self.actual_min_boost
        } else {
            self.config.min_content_boost
        };

        let metadata = GainMapMetadata {
            max_content_boost: [actual_max; 3],
            min_content_boost: [actual_min; 3],
            gamma: [self.config.gamma; 3],
            offset_sdr: [self.config.offset_sdr; 3],
            offset_hdr: [self.config.offset_hdr; 3],
            hdr_capacity_min: 1.0,
            hdr_capacity_max: actual_max,
            use_base_color_space: true,
        };

        Ok((gainmap, metadata))
    }

    /// Check if all input rows have been received.
    pub fn inputs_complete(&self) -> bool {
        self.next_hdr_row >= self.height && self.next_sdr_row >= self.height
    }

    /// Check if encoding is complete (all gainmap rows produced).
    pub fn is_complete(&self) -> bool {
        self.next_gm_row >= self.gm_height && self.pending_gm_rows.is_empty()
    }

    /// Get remaining HDR rows to receive.
    pub fn hdr_rows_remaining(&self) -> u32 {
        self.height - self.next_hdr_row
    }

    /// Get remaining SDR rows to receive.
    pub fn sdr_rows_remaining(&self) -> u32 {
        self.height - self.next_sdr_row
    }

    /// Get remaining gainmap rows to produce.
    pub fn gainmap_rows_remaining(&self) -> u32 {
        self.gm_height - self.next_gm_row
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_row_decoder_multi_row() {
        let mut gainmap = GainMap::new(2, 2).unwrap();
        for v in &mut gainmap.data {
            *v = 128;
        }

        let metadata = GainMapMetadata {
            min_content_boost: [1.0; 3],
            max_content_boost: [4.0; 3],
            gamma: [1.0; 3],
            offset_sdr: [0.015625; 3],
            offset_hdr: [0.015625; 3],
            hdr_capacity_min: 1.0,
            hdr_capacity_max: 4.0,
            use_base_color_space: true,
        };

        let mut decoder = RowDecoder::new(
            gainmap,
            metadata,
            16,
            16,
            4.0,
            HdrOutputFormat::Srgb8,
            ColorGamut::Bt709,
        )
        .unwrap();

        // Process in batches of 4 rows
        let sdr_batch = vec![128u8; 16 * 4 * 4]; // 4 rows, 16 pixels, RGBA
        for _ in 0..4 {
            let hdr_batch = decoder.process_rows(&sdr_batch, 4).unwrap();
            assert_eq!(hdr_batch.len(), 16 * 4 * 4);
        }

        assert!(decoder.is_complete());
    }

    #[test]
    fn test_row_encoder_multi_row() {
        let config = GainMapConfig {
            scale_factor: 2,
            ..Default::default()
        };

        let input_config = EncodeInput {
            hdr_format: PixelFormat::Rgba8,
            hdr_stride: 16 * 4,
            hdr_transfer: ColorTransfer::Srgb,
            hdr_gamut: ColorGamut::Bt709,
            sdr_format: PixelFormat::Rgba8,
            sdr_stride: 16 * 4,
            sdr_gamut: ColorGamut::Bt709,
            y_only: false,
        };

        let mut encoder = RowEncoder::new(16, 16, config, input_config).unwrap();

        let hdr_batch = vec![180u8; 16 * 4 * 4]; // 4 rows
        let sdr_batch = vec![128u8; 16 * 4 * 4];

        // Process in batches of 4
        for _ in 0..4 {
            let _gm_rows = encoder.process_rows(&hdr_batch, &sdr_batch, 4).unwrap();
        }

        let (gainmap, metadata) = encoder.finish().unwrap();
        assert_eq!(gainmap.width, 8);
        assert_eq!(gainmap.height, 8);
        assert!(metadata.max_content_boost[0] >= 1.0);
    }

    #[test]
    fn test_y_only_mode() {
        let config = GainMapConfig {
            scale_factor: 4,
            multi_channel: false,
            ..Default::default()
        };

        let input_config = EncodeInput::y_only(8);

        let mut encoder = RowEncoder::new(8, 8, config, input_config).unwrap();

        // Y-only data (1 byte per pixel)
        let hdr_row = vec![200u8; 8];
        let sdr_row = vec![128u8; 8];

        for _ in 0..8 {
            let _ = encoder.process_row(&hdr_row, &sdr_row).unwrap();
        }

        let (gainmap, _) = encoder.finish().unwrap();
        assert_eq!(gainmap.width, 2);
        assert_eq!(gainmap.height, 2);
    }
}
