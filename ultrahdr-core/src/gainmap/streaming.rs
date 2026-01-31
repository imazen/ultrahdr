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
//! # Linear f32 Input/Output
//!
//! All streaming APIs work with **linear f32 RGB** data. The caller is responsible
//! for converting encoded formats (sRGB, PQ, HLG) to/from linear using an external
//! CMS such as moxcms.
//!
//! ```text
//! Decode: SDR (linear f32) + GainMap → HDR (linear f32)
//! Encode: HDR (linear f32) + SDR (linear f32) → GainMap (u8)
//! ```
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
//! # Example: Streaming HDR Reconstruction
//!
//! ```ignore
//! use ultrahdr_core::gainmap::streaming::RowDecoder;
//!
//! // Caller converts sRGB JPEG output to linear f32 using moxcms
//! let sdr_linear: Vec<f32> = moxcms::srgb_to_linear(&sdr_srgb);
//!
//! let mut decoder = RowDecoder::new(
//!     gainmap, metadata, width, height, 4.0, gamut
//! )?;
//!
//! // Process in 16-row batches
//! for batch_start in (0..height).step_by(16) {
//!     let batch_height = 16.min(height - batch_start);
//!     let sdr_batch = &sdr_linear[batch_start as usize * width as usize * 3..];
//!     let hdr_batch = decoder.process_rows(sdr_batch, batch_height)?;
//!     // hdr_batch is linear f32 RGBA - convert to PQ/sRGB as needed
//! }
//! ```

use alloc::format;
use alloc::vec;
use alloc::vec::Vec;

use crate::color::gamut::rgb_to_luminance;
use crate::types::{ColorGamut, Error, GainMap, GainMapMetadata, Result};

use super::compute::GainMapConfig;

// ============================================================================
// Row Decoder (full gainmap in memory, linear f32 I/O)
// ============================================================================

/// Row-based HDR decoder that holds the full gainmap in memory.
///
/// Accepts SDR image rows as **linear f32 RGB** and outputs reconstructed HDR rows
/// as **linear f32 RGBA**. The caller is responsible for converting encoded formats
/// (sRGB, PQ, etc.) to linear before calling this decoder.
///
/// # Input Format
///
/// Input must be linear f32 RGB with values in `[0, 1]` for SDR content.
/// Layout: `[R, G, B, R, G, B, ...]` - 3 floats per pixel.
///
/// # Output Format
///
/// Output is linear f32 RGBA where `1.0 = SDR white (203 nits)`.
/// Values above 1.0 represent HDR highlights. Layout: `[R, G, B, A, R, G, B, A, ...]`.
///
/// # Memory Usage
///
/// - Gainmap: `gm_width × gm_height × channels` bytes
/// - Per batch: `width × batch_rows × 16` bytes (RGBA f32)
#[derive(Debug)]
pub struct RowDecoder {
    gainmap: GainMap,
    metadata: GainMapMetadata,
    width: u32,
    height: u32,
    weight: f32,
    current_row: u32,
    gamut: ColorGamut,
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
    /// * `gamut` - Color gamut of the SDR input
    pub fn new(
        gainmap: GainMap,
        metadata: GainMapMetadata,
        width: u32,
        height: u32,
        display_boost: f32,
        gamut: ColorGamut,
    ) -> Result<Self> {
        let weight = calculate_weight(display_boost, &metadata);

        Ok(Self {
            gainmap,
            metadata,
            width,
            height,
            weight,
            current_row: 0,
            gamut,
        })
    }

    /// Get the source color gamut.
    pub fn gamut(&self) -> ColorGamut {
        self.gamut
    }

    /// Process multiple SDR rows (linear f32 RGB) and return HDR rows (linear f32 RGBA).
    ///
    /// # Arguments
    /// * `sdr_linear` - Linear f32 RGB data, 3 floats per pixel, `width * num_rows` pixels
    /// * `num_rows` - Number of rows in the batch
    ///
    /// # Returns
    /// Linear f32 RGBA data, 4 floats per pixel.
    pub fn process_rows(&mut self, sdr_linear: &[f32], num_rows: u32) -> Result<Vec<f32>> {
        let remaining = self.height - self.current_row;
        let actual_rows = num_rows.min(remaining);

        if actual_rows == 0 {
            return Err(Error::InvalidPixelData("all rows already processed".into()));
        }

        let input_stride = self.width as usize * 3; // RGB
        let expected_len = input_stride * actual_rows as usize;
        if sdr_linear.len() < expected_len {
            return Err(Error::InvalidPixelData(format!(
                "input data too short: {} < {} floats",
                sdr_linear.len(),
                expected_len
            )));
        }

        let output_stride = self.width as usize * 4; // RGBA
        let mut output = vec![0.0f32; output_stride * actual_rows as usize];

        for row_offset in 0..actual_rows {
            let y = self.current_row + row_offset;
            let input_start = row_offset as usize * input_stride;
            let output_start = row_offset as usize * output_stride;

            for x in 0..self.width {
                let in_idx = input_start + x as usize * 3;
                let out_idx = output_start + x as usize * 4;

                let sdr = [
                    sdr_linear[in_idx],
                    sdr_linear[in_idx + 1],
                    sdr_linear[in_idx + 2],
                ];
                let gain = self.sample_gainmap(x, y);
                let hdr = apply_gain(sdr, gain, &self.metadata);

                output[out_idx] = hdr[0];
                output[out_idx + 1] = hdr[1];
                output[out_idx + 2] = hdr[2];
                output[out_idx + 3] = 1.0;
            }
        }

        self.current_row += actual_rows;
        Ok(output)
    }

    /// Process a single row (convenience wrapper).
    pub fn process_row(&mut self, sdr_linear: &[f32]) -> Result<Vec<f32>> {
        self.process_rows(sdr_linear, 1)
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
}

// ============================================================================
// Stream Decoder (streamed gainmap, minimal memory, linear f32 I/O)
// ============================================================================

/// Streaming HDR decoder that accepts both SDR and gainmap rows independently.
///
/// Unlike [`RowDecoder`] which loads the full gainmap upfront, this variant
/// streams both the SDR image and the gainmap simultaneously.
///
/// # Input/Output Format
///
/// - SDR input: Linear f32 RGB, 3 floats per pixel
/// - HDR output: Linear f32 RGBA, 4 floats per pixel
///
/// # Memory Model
///
/// - Gainmap ring buffer: 16 rows × `gm_width × channels` bytes
#[derive(Debug)]
pub struct StreamDecoder {
    metadata: GainMapMetadata,
    sdr_width: u32,
    sdr_height: u32,
    gm_width: u32,
    gm_height: u32,
    gm_channels: u8,
    weight: f32,
    current_sdr_row: u32,
    current_gm_row: u32,
    gamut: ColorGamut,
    gm_buffer: GainMapRingBuffer,
}

/// Ring buffer for gainmap rows during streaming decode.
#[derive(Debug)]
struct GainMapRingBuffer {
    rows: Vec<Vec<u8>>,
    first_row: u32,
    count: u32,
    row_bytes: usize,
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
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        metadata: GainMapMetadata,
        sdr_width: u32,
        sdr_height: u32,
        gm_width: u32,
        gm_height: u32,
        gm_channels: u8,
        display_boost: f32,
        gamut: ColorGamut,
    ) -> Result<Self> {
        let weight = calculate_weight(display_boost, &metadata);
        let gm_buffer = GainMapRingBuffer::new(gm_width, gm_channels, 16);

        Ok(Self {
            metadata,
            sdr_width,
            sdr_height,
            gm_width,
            gm_height,
            gm_channels,
            weight,
            current_sdr_row: 0,
            current_gm_row: 0,
            gamut,
            gm_buffer,
        })
    }

    /// Get the source color gamut.
    pub fn gamut(&self) -> ColorGamut {
        self.gamut
    }

    /// Push a gainmap row into the buffer.
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
        let gm_y_last = (last_sdr_row as f32 / self.sdr_height as f32) * self.gm_height as f32;
        let gm_y1_needed = (gm_y_last.ceil() as u32).min(self.gm_height - 1);

        self.current_gm_row > gm_y1_needed || self.gm_buffer.contains(gm_y1_needed)
    }

    /// Process SDR rows (linear f32 RGB) and return HDR output (linear f32 RGBA).
    pub fn process_sdr_rows(&mut self, sdr_linear: &[f32], num_rows: u32) -> Result<Vec<f32>> {
        let remaining = self.sdr_height - self.current_sdr_row;
        let actual_rows = num_rows.min(remaining);

        if actual_rows == 0 {
            return Err(Error::InvalidPixelData(
                "all SDR rows already processed".into(),
            ));
        }

        if !self.can_process(actual_rows) {
            return Err(Error::InvalidPixelData(
                "insufficient gainmap data buffered".into(),
            ));
        }

        let input_stride = self.sdr_width as usize * 3;
        let expected_len = input_stride * actual_rows as usize;
        if sdr_linear.len() < expected_len {
            return Err(Error::InvalidPixelData(format!(
                "SDR data too short: {} < {} floats",
                sdr_linear.len(),
                expected_len
            )));
        }

        let output_stride = self.sdr_width as usize * 4;
        let mut output = vec![0.0f32; output_stride * actual_rows as usize];

        for row_offset in 0..actual_rows {
            let y = self.current_sdr_row + row_offset;
            let input_start = row_offset as usize * input_stride;
            let output_start = row_offset as usize * output_stride;

            for x in 0..self.sdr_width {
                let in_idx = input_start + x as usize * 3;
                let out_idx = output_start + x as usize * 4;

                let sdr = [
                    sdr_linear[in_idx],
                    sdr_linear[in_idx + 1],
                    sdr_linear[in_idx + 2],
                ];
                let gain = self.sample_gainmap(x, y);
                let hdr = apply_gain(sdr, gain, &self.metadata);

                output[out_idx] = hdr[0];
                output[out_idx + 1] = hdr[1];
                output[out_idx + 2] = hdr[2];
                output[out_idx + 3] = 1.0;
            }
        }

        self.current_sdr_row += actual_rows;
        Ok(output)
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

// ============================================================================
// Row Encoder (synchronized HDR+SDR batches, linear f32 input)
// ============================================================================

/// Row-based gainmap encoder for synchronized HDR+SDR input.
///
/// Accepts HDR and SDR rows as **linear f32 RGB** and outputs 8-bit gain map values.
///
/// # Input Format
///
/// - HDR: Linear f32 RGB, 3 floats per pixel, values typically `[0, ~10]`
/// - SDR: Linear f32 RGB, 3 floats per pixel, values in `[0, 1]`
#[derive(Debug)]
pub struct RowEncoder {
    config: GainMapConfig,
    width: u32,
    height: u32,
    gm_width: u32,
    gm_height: u32,
    scale: u32,
    current_input_row: u32,
    current_gm_row: u32,
    hdr_buffer: LinearRowBuffer,
    sdr_buffer: LinearRowBuffer,
    actual_min_boost: f32,
    actual_max_boost: f32,
    gainmap_rows: Vec<Vec<u8>>,
    hdr_gamut: ColorGamut,
    sdr_gamut: ColorGamut,
}

/// Buffer for storing a sliding window of linear f32 rows.
#[derive(Debug)]
struct LinearRowBuffer {
    rows: Vec<Vec<f32>>,
    first_row: u32,
    count: u32,
    width: u32,
}

impl LinearRowBuffer {
    fn new(capacity: usize, width: u32) -> Self {
        let row_len = width as usize * 3; // RGB
        Self {
            rows: vec![vec![0.0f32; row_len]; capacity],
            first_row: 0,
            count: 0,
            width,
        }
    }

    fn push_row(&mut self, row: u32, data: &[f32]) {
        let capacity = self.rows.len() as u32;
        let idx = (row % capacity) as usize;
        let copy_len = data.len().min(self.rows[idx].len());
        self.rows[idx][..copy_len].copy_from_slice(&data[..copy_len]);

        if self.count == 0 {
            self.first_row = row;
        }
        self.count = self.count.saturating_add(1).min(capacity);
    }

    fn push_rows(&mut self, start_row: u32, data: &[f32], num_rows: u32) {
        let stride = self.width as usize * 3;
        for i in 0..num_rows {
            let row_data = &data[i as usize * stride..];
            self.push_row(start_row + i, row_data);
        }
    }

    fn get_pixel(&self, row: u32, x: u32) -> Option<[f32; 3]> {
        let capacity = self.rows.len() as u32;

        if row < self.first_row || row >= self.first_row + self.count {
            return None;
        }

        let idx = (row % capacity) as usize;
        let row_data = &self.rows[idx];
        let px_idx = x as usize * 3;

        if px_idx + 2 < row_data.len() {
            Some([row_data[px_idx], row_data[px_idx + 1], row_data[px_idx + 2]])
        } else {
            None
        }
    }
}

impl RowEncoder {
    /// Create a new row-based encoder.
    ///
    /// # Arguments
    /// * `width`, `height` - Image dimensions
    /// * `config` - Gain map configuration
    /// * `hdr_gamut` - HDR input color gamut
    /// * `sdr_gamut` - SDR input color gamut
    pub fn new(
        width: u32,
        height: u32,
        config: GainMapConfig,
        hdr_gamut: ColorGamut,
        sdr_gamut: ColorGamut,
    ) -> Result<Self> {
        let scale = config.scale_factor.max(1) as u32;
        let gm_width = width.div_ceil(scale);
        let gm_height = height.div_ceil(scale);

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
            hdr_buffer: LinearRowBuffer::new(buffer_size, width),
            sdr_buffer: LinearRowBuffer::new(buffer_size, width),
            actual_min_boost: f32::MAX,
            actual_max_boost: f32::MIN,
            gainmap_rows: Vec::new(),
            hdr_gamut,
            sdr_gamut,
        })
    }

    /// Process multiple rows at once.
    ///
    /// # Arguments
    /// * `hdr_linear` - HDR linear f32 RGB data, 3 floats per pixel
    /// * `sdr_linear` - SDR linear f32 RGB data, 3 floats per pixel
    /// * `num_rows` - Number of rows
    ///
    /// # Returns
    /// Vector of completed gainmap rows (may be empty if waiting for more data).
    pub fn process_rows(
        &mut self,
        hdr_linear: &[f32],
        sdr_linear: &[f32],
        num_rows: u32,
    ) -> Result<Vec<Vec<u8>>> {
        let remaining = self.height - self.current_input_row;
        let actual_rows = num_rows.min(remaining);

        if actual_rows == 0 {
            return Ok(Vec::new());
        }

        self.hdr_buffer
            .push_rows(self.current_input_row, hdr_linear, actual_rows);
        self.sdr_buffer
            .push_rows(self.current_input_row, sdr_linear, actual_rows);

        self.current_input_row += actual_rows;

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
    pub fn process_row(
        &mut self,
        hdr_linear: &[f32],
        sdr_linear: &[f32],
    ) -> Result<Option<Vec<u8>>> {
        let rows = self.process_rows(hdr_linear, sdr_linear, 1)?;
        Ok(rows.into_iter().next())
    }

    /// Finish and return the complete gainmap with metadata.
    pub fn finish(mut self) -> Result<(GainMap, GainMapMetadata)> {
        while self.current_gm_row < self.gm_height {
            let gm_row = self.compute_gainmap_row()?;
            self.gainmap_rows.push(gm_row);
            self.current_gm_row += 1;
        }

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

        for gx in 0..self.gm_width {
            let x = (gx * self.scale + self.scale / 2).min(self.width - 1);
            let y = (gy * self.scale + self.scale / 2).min(self.height - 1);

            let hdr_rgb = self.hdr_buffer.get_pixel(y, x).unwrap_or([0.5, 0.5, 0.5]);
            let sdr_rgb = self.sdr_buffer.get_pixel(y, x).unwrap_or([0.5, 0.5, 0.5]);

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
// Stream Encoder (independent HDR/SDR streams, linear f32 input)
// ============================================================================

/// Streaming gainmap encoder that accepts HDR and SDR rows independently.
///
/// Unlike [`RowEncoder`] which requires synchronized batches, this variant
/// allows feeding HDR and SDR rows from separate sources at different rates.
///
/// # Input Format
///
/// - HDR: Linear f32 RGB, 3 floats per pixel
/// - SDR: Linear f32 RGB, 3 floats per pixel
#[derive(Debug)]
pub struct StreamEncoder {
    config: GainMapConfig,
    width: u32,
    height: u32,
    gm_width: u32,
    gm_height: u32,
    scale: u32,
    hdr_rows: LinearInputRingBuffer,
    sdr_rows: LinearInputRingBuffer,
    next_hdr_row: u32,
    next_sdr_row: u32,
    next_gm_row: u32,
    actual_min_boost: f32,
    actual_max_boost: f32,
    pending_gm_rows: Vec<Vec<u8>>,
    hdr_gamut: ColorGamut,
    sdr_gamut: ColorGamut,
}

/// Ring buffer for linear f32 input rows.
#[derive(Debug)]
struct LinearInputRingBuffer {
    rows: Vec<Vec<f32>>,
    first_row: u32,
    count: u32,
    row_floats: usize,
    capacity: u32,
}

impl LinearInputRingBuffer {
    fn new(row_floats: usize, capacity: u32) -> Self {
        Self {
            rows: vec![vec![0.0f32; row_floats]; capacity as usize],
            first_row: 0,
            count: 0,
            row_floats,
            capacity,
        }
    }

    fn push(&mut self, row_index: u32, data: &[f32]) {
        let slot = (row_index % self.capacity) as usize;
        let copy_len = data.len().min(self.row_floats);
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

    fn get(&self, row: u32) -> Option<&[f32]> {
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
        hdr_gamut: ColorGamut,
        sdr_gamut: ColorGamut,
    ) -> Result<Self> {
        let scale = config.scale_factor.max(1) as u32;
        let gm_width = width.div_ceil(scale);
        let gm_height = height.div_ceil(scale);

        let row_floats = width as usize * 3; // RGB
        let buffer_capacity = (scale + 16).min(32);

        Ok(Self {
            config,
            width,
            height,
            gm_width,
            gm_height,
            scale,
            hdr_rows: LinearInputRingBuffer::new(row_floats, buffer_capacity),
            sdr_rows: LinearInputRingBuffer::new(row_floats, buffer_capacity),
            next_hdr_row: 0,
            next_sdr_row: 0,
            next_gm_row: 0,
            actual_min_boost: f32::MAX,
            actual_max_boost: f32::MIN,
            pending_gm_rows: Vec::new(),
            hdr_gamut,
            sdr_gamut,
        })
    }

    /// Push HDR rows (linear f32 RGB) into the buffer.
    pub fn push_hdr_rows(&mut self, data: &[f32], num_rows: u32) -> Result<()> {
        let remaining = self.height - self.next_hdr_row;
        let actual = num_rows.min(remaining);
        let stride = self.width as usize * 3;

        for i in 0..actual {
            let start = i as usize * stride;
            let end = start + stride;
            if end > data.len() {
                return Err(Error::InvalidPixelData("HDR data too short".into()));
            }
            self.hdr_rows.push(self.next_hdr_row + i, &data[start..end]);
        }
        self.next_hdr_row += actual;

        self.try_produce_gainmap_rows();
        Ok(())
    }

    /// Push a single HDR row.
    pub fn push_hdr_row(&mut self, data: &[f32]) -> Result<()> {
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

    /// Push SDR rows (linear f32 RGB) into the buffer.
    pub fn push_sdr_rows(&mut self, data: &[f32], num_rows: u32) -> Result<()> {
        let remaining = self.height - self.next_sdr_row;
        let actual = num_rows.min(remaining);
        let stride = self.width as usize * 3;

        for i in 0..actual {
            let start = i as usize * stride;
            let end = start + stride;
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
    pub fn push_sdr_row(&mut self, data: &[f32]) -> Result<()> {
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
            let center_y = self.next_gm_row * self.scale + self.scale / 2;
            let center_y = center_y.min(self.height - 1);

            if !self.hdr_rows.has_row(center_y) || !self.sdr_rows.has_row(center_y) {
                break;
            }

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

            let hdr_rgb = self.get_pixel(hdr_row_data, center_x);
            let sdr_rgb = self.get_pixel(sdr_row_data, center_x);

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

    /// Get pixel from row data.
    fn get_pixel(&self, row: Option<&[f32]>, x: u32) -> [f32; 3] {
        match row {
            Some(r) => {
                let idx = x as usize * 3;
                if idx + 2 < r.len() {
                    [r[idx], r[idx + 1], r[idx + 2]]
                } else {
                    [0.5, 0.5, 0.5]
                }
            }
            None => [0.5, 0.5, 0.5],
        }
    }

    /// Finish encoding and return the gainmap and metadata.
    pub fn finish(mut self) -> Result<(GainMap, GainMapMetadata)> {
        let mut all_rows = Vec::new();
        all_rows.append(&mut self.pending_gm_rows);

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

    /// Check if encoding is complete.
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_row_decoder_linear_f32() {
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

        let mut decoder = RowDecoder::new(gainmap, metadata, 4, 4, 4.0, ColorGamut::Bt709).unwrap();

        // Linear f32 RGB input (mid-gray = 0.18)
        let sdr_linear = vec![0.18f32; 4 * 3]; // One row, 4 pixels, RGB
        let hdr = decoder.process_row(&sdr_linear).unwrap();

        // Output is RGBA f32
        assert_eq!(hdr.len(), 4 * 4); // 4 pixels × 4 floats
        assert!(hdr[0] > 0.0); // Non-zero output
    }

    #[test]
    fn test_row_encoder_linear_f32() {
        let config = GainMapConfig {
            scale_factor: 2,
            ..Default::default()
        };

        let mut encoder =
            RowEncoder::new(4, 4, config, ColorGamut::Bt709, ColorGamut::Bt709).unwrap();

        // HDR brighter than SDR (both linear f32)
        let hdr_linear = vec![0.5f32; 4 * 3]; // Brighter
        let sdr_linear = vec![0.18f32; 4 * 3]; // Mid-gray

        for _ in 0..4 {
            let _ = encoder.process_row(&hdr_linear, &sdr_linear).unwrap();
        }

        let (gainmap, metadata) = encoder.finish().unwrap();
        assert_eq!(gainmap.width, 2);
        assert_eq!(gainmap.height, 2);
        assert!(metadata.max_content_boost[0] >= 1.0);
    }

    #[test]
    fn test_stream_encoder_linear_f32() {
        let config = GainMapConfig {
            scale_factor: 2,
            ..Default::default()
        };

        let mut encoder =
            StreamEncoder::new(4, 4, config, ColorGamut::Bt709, ColorGamut::Bt709).unwrap();

        let hdr_linear = vec![0.5f32; 4 * 3];
        let sdr_linear = vec![0.18f32; 4 * 3];

        for _ in 0..4 {
            encoder.push_hdr_row(&hdr_linear).unwrap();
            encoder.push_sdr_row(&sdr_linear).unwrap();
        }

        // DON'T drain pending rows before finish() - let finish() collect them
        let (gainmap, _) = encoder.finish().unwrap();
        assert_eq!(gainmap.width, 2);
        assert_eq!(gainmap.height, 2);
    }
}
