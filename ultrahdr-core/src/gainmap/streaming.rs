//! Streaming APIs for low-memory gain map processing.
//!
//! These APIs process images in row batches (typically 8-16 rows for JPEG MCU alignment),
//! drastically reducing memory usage compared to full-image APIs.
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
//! # Features
//!
//! - **Flexible batch sizes**: Process 1-N rows at a time (16 is typical for JPEG MCUs)
//! - **Stride support**: Handle row padding from various decoders
//! - **YCbCr passthrough**: Work directly on Y channel for luminance-based gainmaps
//! - **Zero-copy**: Borrow input data rather than copying when possible
//!
//! # Example: Streaming HDR Reconstruction (16 rows at a time)
//!
//! ```ignore
//! use ultrahdr_core::gainmap::streaming::StreamingHdrReconstructor;
//!
//! let mut reconstructor = StreamingHdrReconstructor::new(
//!     gainmap, metadata, width, height, 4.0, HdrOutputFormat::LinearFloat
//! )?;
//!
//! // Process in 16-row batches (typical JPEG MCU height)
//! for batch_start in (0..height).step_by(16) {
//!     let batch_height = 16.min(height - batch_start);
//!     let sdr_batch = decode_next_rows(batch_height);  // Your decoder
//!     let hdr_batch = reconstructor.process_rows(&sdr_batch, batch_height)?;
//!     write_hdr_rows(batch_start, &hdr_batch);
//! }
//! ```

use crate::color::gamut::rgb_to_luminance;
use crate::color::transfer::{pq_oetf, srgb_eotf, srgb_oetf};
use crate::types::{
    ColorGamut, ColorTransfer, Error, GainMap, GainMapMetadata, PixelFormat, Result,
};

use super::apply::HdrOutputFormat;
use super::compute::GainMapConfig;

// ============================================================================
// Streaming HDR Reconstruction (Decode)
// ============================================================================

/// Streaming HDR reconstructor for batch row processing.
///
/// Accepts SDR rows in configurable batches and outputs HDR rows.
/// Memory usage scales with batch size, not image size.
#[derive(Debug)]
pub struct StreamingHdrReconstructor {
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
    input_config: InputConfig,
}

/// Configuration for input data format.
#[derive(Debug, Clone)]
pub struct InputConfig {
    /// Pixel format of input data
    pub format: PixelFormat,
    /// Bytes per row (0 = auto-calculate from width)
    pub stride: u32,
    /// YCbCr mode: if true, expect Y plane only for luminance-based processing
    pub ycbcr_y_only: bool,
}

impl Default for InputConfig {
    fn default() -> Self {
        Self {
            format: PixelFormat::Rgba8,
            stride: 0,
            ycbcr_y_only: false,
        }
    }
}

impl InputConfig {
    /// Create config for RGBA8 input.
    pub fn rgba8(width: u32) -> Self {
        Self {
            format: PixelFormat::Rgba8,
            stride: width * 4,
            ycbcr_y_only: false,
        }
    }

    /// Create config for RGB8 input.
    pub fn rgb8(width: u32) -> Self {
        Self {
            format: PixelFormat::Rgb8,
            stride: width * 3,
            ycbcr_y_only: false,
        }
    }

    /// Create config for Y-only luminance input (8-bit).
    pub fn y_only(width: u32) -> Self {
        Self {
            format: PixelFormat::Gray8,
            stride: width,
            ycbcr_y_only: true,
        }
    }

    /// Create config with custom stride.
    pub fn with_stride(mut self, stride: u32) -> Self {
        self.stride = stride;
        self
    }
}

impl StreamingHdrReconstructor {
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
            InputConfig::rgba8(width),
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
        input_config: InputConfig,
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
            let sdr_linear = if self.input_config.ycbcr_y_only {
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
// Streaming Gain Map Computation (Encode)
// ============================================================================

/// Streaming gain map computer for batch row processing.
///
/// Accepts HDR and SDR rows in configurable batches, buffering only
/// what's needed for block sampling.
#[derive(Debug)]
pub struct StreamingGainMapComputer {
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

/// Input configuration for streaming encoder.
#[derive(Debug, Clone)]
pub struct EncoderInputConfig {
    /// HDR pixel format
    pub hdr_format: PixelFormat,
    /// HDR row stride (0 = auto)
    pub hdr_stride: u32,
    /// HDR transfer function
    pub hdr_transfer: ColorTransfer,
    /// HDR gamut
    pub hdr_gamut: ColorGamut,
    /// SDR pixel format
    pub sdr_format: PixelFormat,
    /// SDR row stride (0 = auto)
    pub sdr_stride: u32,
    /// SDR gamut
    pub sdr_gamut: ColorGamut,
    /// Y-only mode for luminance processing
    pub y_only: bool,
}

impl Default for EncoderInputConfig {
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

impl EncoderInputConfig {
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

    /// Create config for Y-only luminance mode (faster, for luminance gainmaps).
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

impl StreamingGainMapComputer {
    /// Create a new streaming gain map computer.
    pub fn new(
        width: u32,
        height: u32,
        config: GainMapConfig,
        input_config: EncoderInputConfig,
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

fn get_pixel_linear(row: &[u8], x: u32, config: &InputConfig) -> [f32; 3] {
    get_linear_rgb_from_row(
        row,
        x,
        config.format,
        ColorTransfer::Srgb,
        config.ycbcr_y_only,
    )
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_streaming_reconstructor_multi_row() {
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

        let mut reconstructor = StreamingHdrReconstructor::new(
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
            let hdr_batch = reconstructor.process_rows(&sdr_batch, 4).unwrap();
            assert_eq!(hdr_batch.len(), 16 * 4 * 4);
        }

        assert!(reconstructor.is_complete());
    }

    #[test]
    fn test_streaming_computer_multi_row() {
        let config = GainMapConfig {
            scale_factor: 2,
            ..Default::default()
        };

        let input_config = EncoderInputConfig {
            hdr_format: PixelFormat::Rgba8,
            hdr_stride: 16 * 4,
            hdr_transfer: ColorTransfer::Srgb,
            hdr_gamut: ColorGamut::Bt709,
            sdr_format: PixelFormat::Rgba8,
            sdr_stride: 16 * 4,
            sdr_gamut: ColorGamut::Bt709,
            y_only: false,
        };

        let mut computer = StreamingGainMapComputer::new(16, 16, config, input_config).unwrap();

        let hdr_batch = vec![180u8; 16 * 4 * 4]; // 4 rows
        let sdr_batch = vec![128u8; 16 * 4 * 4];

        // Process in batches of 4
        for _ in 0..4 {
            let _gm_rows = computer.process_rows(&hdr_batch, &sdr_batch, 4).unwrap();
        }

        let (gainmap, metadata) = computer.finish().unwrap();
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

        let input_config = EncoderInputConfig::y_only(8);

        let mut computer = StreamingGainMapComputer::new(8, 8, config, input_config).unwrap();

        // Y-only data (1 byte per pixel)
        let hdr_row = vec![200u8; 8];
        let sdr_row = vec![128u8; 8];

        for _ in 0..8 {
            let _ = computer.process_row(&hdr_row, &sdr_row).unwrap();
        }

        let (gainmap, _) = computer.finish().unwrap();
        assert_eq!(gainmap.width, 2);
        assert_eq!(gainmap.height, 2);
    }
}
