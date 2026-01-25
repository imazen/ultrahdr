//! Streaming tonemapper with local adaptation.
//!
//! This module provides a high-quality tonemapper that works in a single streaming
//! pass with local adaptation, using a lookahead buffer for context.
//!
//! # Design
//!
//! - **Single pass**: Rows flow in, tonemapped rows flow out (with delay)
//! - **Local adaptation**: Grid-based local key/white tracking
//! - **Top-down learning**: Statistics improve as more rows arrive
//! - **AgX-style color**: Smooth highlight desaturation toward white
//!
//! # Memory Usage
//!
//! For 4K (3840×2160) with 1/8 grid and 64-row lookahead:
//! - Grid: 480×270 × 16 bytes = ~2 MB
//! - Row buffer: 64 × 3840 × 16 bytes = ~4 MB
//! - Total: ~6 MB (vs 130 MB for full buffer)

use alloc::vec;
use alloc::vec::Vec;

use crate::color::transfer::srgb_oetf;
use crate::types::Result;

// ============================================================================
// Configuration
// ============================================================================

/// Configuration for the streaming tonemapper.
#[derive(Debug, Clone)]
pub struct StreamingTonemapConfig {
    /// Grid cell size in pixels (default: 8 = 1/8 resolution grid).
    pub cell_size: u32,
    /// Number of lookahead rows to buffer (default: 64).
    pub lookahead_rows: u32,
    /// Target mid-gray (key) value in linear (default: 0.18).
    pub target_key: f32,
    /// Contrast boost factor (default: 1.1, subtle boost).
    pub contrast: f32,
    /// Saturation preservation factor (default: 0.95, slight reduction in highlights).
    pub saturation: f32,
    /// Shadow lift amount (default: 0.02).
    pub shadow_lift: f32,
    /// Highlight desaturation threshold (fraction of white point, default: 0.5).
    pub desat_threshold: f32,
    /// Number of channels in input data (3 for RGB, 4 for RGBA). Default: 4.
    pub channels: u8,
}

impl Default for StreamingTonemapConfig {
    fn default() -> Self {
        Self {
            cell_size: 8,
            lookahead_rows: 64,
            target_key: 0.18,
            contrast: 1.1,
            saturation: 0.95,
            shadow_lift: 0.02,
            desat_threshold: 0.5,
            channels: 4,
        }
    }
}

impl StreamingTonemapConfig {
    /// Configure for RGB input (3 channels, no alpha).
    pub fn rgb() -> Self {
        Self {
            channels: 3,
            ..Default::default()
        }
    }

    /// Configure for RGBA input (4 channels with alpha).
    pub fn rgba() -> Self {
        Self::default()
    }
}

// ============================================================================
// Local Adaptation Grid
// ============================================================================

/// Statistics accumulated for a grid cell.
#[derive(Debug, Clone, Copy)]
struct CellStats {
    /// Sum of log-luminance (for geometric mean).
    sum_log_lum: f64,
    /// Count of valid samples.
    count: u32,
    /// Running max luminance.
    max_lum: f32,
    /// Running min luminance (above black).
    min_lum: f32,
    /// Sum of luminance for arithmetic mean.
    sum_lum: f64,
}

impl Default for CellStats {
    fn default() -> Self {
        Self {
            sum_log_lum: 0.0,
            count: 0,
            max_lum: 0.0,
            min_lum: f32::MAX,
            sum_lum: 0.0,
        }
    }
}

impl CellStats {
    /// Add a sample.
    #[inline]
    fn add(&mut self, lum: f32) {
        if lum > 1e-6 {
            self.sum_log_lum += (lum as f64).ln();
            self.count += 1;
            self.max_lum = self.max_lum.max(lum);
            self.min_lum = self.min_lum.min(lum);
            self.sum_lum += lum as f64;
        }
    }

    /// Compute local parameters.
    fn compute_params(self, global_key: f32, global_white: f32) -> LocalParams {
        if self.count == 0 {
            return LocalParams {
                key: global_key,
                white: global_white,
                black: 0.0,
            };
        }

        let key = (self.sum_log_lum / self.count as f64).exp() as f32;
        // Use 95th percentile estimate: between arithmetic mean and max
        let mean = (self.sum_lum / self.count as f64) as f32;
        let white = mean + (self.max_lum - mean) * 0.8; // Approximate 95th percentile

        LocalParams {
            key: key.max(0.001),
            white: white.max(key * 2.0),
            black: if self.min_lum < f32::MAX {
                self.min_lum
            } else {
                0.0
            },
        }
    }
}

/// Local adaptation parameters for a region.
#[derive(Debug, Clone, Copy)]
pub struct LocalParams {
    /// Local key (geometric mean luminance).
    pub key: f32,
    /// Local white point (approximate max).
    pub white: f32,
    /// Local black point (approximate min).
    pub black: f32,
}

impl Default for LocalParams {
    fn default() -> Self {
        Self {
            key: 0.18,
            white: 4.0,
            black: 0.0,
        }
    }
}

/// Grid of local adaptation parameters.
struct AdaptationGrid {
    /// Grid dimensions.
    width: u32,
    height: u32,
    /// Cell size in pixels.
    cell_size: u32,
    /// Per-cell statistics (accumulated during analysis).
    stats: Vec<CellStats>,
    /// Computed parameters (finalized after blur).
    params: Vec<LocalParams>,
    /// Global statistics (for fallback and blending).
    global_stats: CellStats,
    /// How many rows have been fully processed.
    rows_processed: u32,
}

impl AdaptationGrid {
    fn new(image_width: u32, image_height: u32, cell_size: u32) -> Self {
        let width = image_width.div_ceil(cell_size);
        let height = image_height.div_ceil(cell_size);
        let num_cells = (width * height) as usize;

        Self {
            width,
            height,
            cell_size,
            stats: vec![CellStats::default(); num_cells],
            params: vec![LocalParams::default(); num_cells],
            global_stats: CellStats::default(),
            rows_processed: 0,
        }
    }

    /// Add samples from a row of HDR data.
    fn add_row(&mut self, row_data: &[f32], y: u32, image_width: u32, channels: usize) {
        let cell_y = y / self.cell_size;
        if cell_y >= self.height {
            return;
        }

        for (x, pixel) in row_data.chunks(channels).enumerate().take(image_width as usize) {
            let lum = luminance_bt709(pixel[0], pixel[1], pixel[2]);

            // Update global stats
            self.global_stats.add(lum);

            // Update cell stats
            let cell_x = (x as u32) / self.cell_size;
            if cell_x < self.width {
                let cell_idx = (cell_y * self.width + cell_x) as usize;
                self.stats[cell_idx].add(lum);
            }
        }
    }

    /// Mark that a row of cells is complete and can be finalized.
    fn finalize_row(&mut self, cell_y: u32) {
        if cell_y >= self.height {
            return;
        }

        let global = self.global_stats.compute_params(0.18, 4.0);

        for cell_x in 0..self.width {
            let idx = (cell_y * self.width + cell_x) as usize;
            self.params[idx] = self.stats[idx].compute_params(global.key, global.white);
        }

        self.rows_processed = self.rows_processed.max(cell_y + 1);
    }

    /// Sample local parameters at image coordinates with bilinear interpolation.
    fn sample(&self, x: f32, y: f32) -> LocalParams {
        // Convert to grid coordinates
        let gx = x / self.cell_size as f32;
        let gy = y / self.cell_size as f32;

        let x0 = (gx.floor() as u32).min(self.width.saturating_sub(1));
        let y0 = (gy.floor() as u32).min(self.height.saturating_sub(1));
        let x1 = (x0 + 1).min(self.width - 1);
        let y1 = (y0 + 1).min(self.height - 1);

        // Clamp y1 to processed rows
        let y1 = y1.min(self.rows_processed.saturating_sub(1));
        let y0 = y0.min(y1);

        let fx = gx - gx.floor();
        let fy = gy - gy.floor();

        // Sample four corners
        let p00 = &self.params[(y0 * self.width + x0) as usize];
        let p10 = &self.params[(y0 * self.width + x1) as usize];
        let p01 = &self.params[(y1 * self.width + x0) as usize];
        let p11 = &self.params[(y1 * self.width + x1) as usize];

        // Bilinear interpolate each field
        LocalParams {
            key: bilinear(p00.key, p10.key, p01.key, p11.key, fx, fy),
            white: bilinear(p00.white, p10.white, p01.white, p11.white, fx, fy),
            black: bilinear(p00.black, p10.black, p01.black, p11.black, fx, fy),
        }
    }

    /// Get global parameters (fallback).
    fn global_params(&self) -> LocalParams {
        self.global_stats.compute_params(0.18, 4.0)
    }

    /// Blur the grid for smoother transitions.
    fn blur_params(&mut self, radius: u32) {
        if radius == 0 || self.width < 3 || self.height < 3 {
            return;
        }

        let mut blurred = vec![LocalParams::default(); self.params.len()];

        for y in 0..self.height {
            for x in 0..self.width {
                let mut sum_key = 0.0f32;
                let mut sum_white = 0.0f32;
                let mut sum_black = 0.0f32;
                let mut count = 0.0f32;

                let y_start = y.saturating_sub(radius);
                let y_end = (y + radius + 1).min(self.height);
                let x_start = x.saturating_sub(radius);
                let x_end = (x + radius + 1).min(self.width);

                for ny in y_start..y_end {
                    for nx in x_start..x_end {
                        let idx = (ny * self.width + nx) as usize;
                        let p = &self.params[idx];
                        sum_key += p.key;
                        sum_white += p.white;
                        sum_black += p.black;
                        count += 1.0;
                    }
                }

                let idx = (y * self.width + x) as usize;
                blurred[idx] = LocalParams {
                    key: sum_key / count,
                    white: sum_white / count,
                    black: sum_black / count,
                };
            }
        }

        self.params = blurred;
    }
}

// ============================================================================
// Streaming Tonemapper
// ============================================================================

/// Streaming tonemapper with local adaptation.
///
/// Processes HDR rows in a single pass with a lookahead buffer,
/// producing high-quality SDR output with local contrast preservation.
pub struct StreamingTonemapper {
    config: StreamingTonemapConfig,
    /// Image dimensions.
    width: u32,
    height: u32,
    /// Local adaptation grid.
    grid: AdaptationGrid,
    /// Lookahead row buffer (ring buffer of HDR rows).
    row_buffer: Vec<Vec<f32>>,
    /// Index of first row in buffer.
    buffer_start_row: u32,
    /// Number of rows currently in buffer.
    buffer_count: u32,
    /// Next row to output.
    next_output_row: u32,
    /// Whether we've finished receiving input.
    input_complete: bool,
}

/// Output from processing a row.
pub struct TonemapOutput {
    /// The tonemapped SDR row (linear RGB, ready for sRGB OETF).
    pub sdr_linear: Vec<f32>,
    /// The row index this corresponds to.
    pub row_index: u32,
}

impl StreamingTonemapper {
    /// Create a new streaming tonemapper.
    pub fn new(width: u32, height: u32, config: StreamingTonemapConfig) -> Result<Self> {
        let grid = AdaptationGrid::new(width, height, config.cell_size);
        let buffer_size = config.lookahead_rows as usize;

        Ok(Self {
            config,
            width,
            height,
            grid,
            row_buffer: vec![Vec::new(); buffer_size],
            buffer_start_row: 0,
            buffer_count: 0,
            next_output_row: 0,
            input_complete: false,
        })
    }

    /// Push HDR rows from a slice with stride.
    ///
    /// # Arguments
    /// - `data`: Slice containing row data (linear HDR, f32)
    /// - `stride`: Number of f32 elements between row starts (>= width * channels)
    /// - `num_rows`: Number of rows to process from this slice
    ///
    /// Stride allows for padding between rows or processing a sub-region of a larger buffer.
    /// For tightly-packed data, use `stride = width * channels`.
    ///
    /// Returns tonemapped rows that are ready (may be empty if still buffering).
    pub fn push_rows(
        &mut self,
        data: &[f32],
        stride: usize,
        num_rows: usize,
    ) -> Result<Vec<TonemapOutput>> {
        let channels = self.config.channels as usize;
        let row_width = self.width as usize * channels;

        for row_idx in 0..num_rows {
            let input_row = self.buffer_start_row + self.buffer_count;
            if input_row >= self.height {
                break;
            }

            let start = row_idx * stride;
            if start + row_width > data.len() {
                break;
            }
            let row_data = &data[start..start + row_width];

            // Add to grid statistics
            self.grid.add_row(row_data, input_row, self.width, channels);

            // Store in ring buffer (copy into pre-allocated slot)
            let buffer_idx = (input_row % self.config.lookahead_rows) as usize;
            let buffer_slot = &mut self.row_buffer[buffer_idx];
            buffer_slot.clear();
            buffer_slot.extend_from_slice(row_data);
            self.buffer_count += 1;

            // Finalize grid cells for completed rows
            let completed_cell_y = input_row / self.config.cell_size;
            if input_row % self.config.cell_size == self.config.cell_size - 1 {
                self.grid.finalize_row(completed_cell_y);
            }
        }

        self.try_output_rows()
    }

    /// Push a single HDR row.
    ///
    /// Convenience method for `push_rows(row, row.len(), 1)`.
    #[inline]
    pub fn push_row(&mut self, row: &[f32]) -> Result<Vec<TonemapOutput>> {
        self.push_rows(row, row.len(), 1)
    }

    /// Signal that all input has been provided. Flush remaining rows.
    pub fn finish(&mut self) -> Result<Vec<TonemapOutput>> {
        self.input_complete = true;

        // Finalize all remaining grid cells
        let last_cell_y = (self.height - 1) / self.config.cell_size;
        for y in 0..=last_cell_y {
            self.grid.finalize_row(y);
        }

        // Blur the grid for smoother output
        self.grid.blur_params(1);

        // Output all remaining rows
        self.try_output_rows()
    }

    /// Try to output any rows that have enough context.
    fn try_output_rows(&mut self) -> Result<Vec<TonemapOutput>> {
        let mut outputs = Vec::new();

        // Determine how far ahead we can see
        let last_input_row = self.buffer_start_row + self.buffer_count;

        // We need lookahead_rows/2 rows ahead for good context,
        // unless input is complete
        let required_ahead = if self.input_complete {
            0
        } else {
            self.config.lookahead_rows / 2
        };

        while self.next_output_row < self.height {
            // Check if we have enough lookahead
            let rows_ahead = last_input_row.saturating_sub(self.next_output_row);
            if rows_ahead < required_ahead && !self.input_complete {
                break;
            }

            // Check if row is still in buffer
            if self.next_output_row < self.buffer_start_row {
                // Row has scrolled out - shouldn't happen with proper sizing
                self.next_output_row += 1;
                continue;
            }

            let buffer_idx = (self.next_output_row % self.config.lookahead_rows) as usize;
            let hdr_row = &self.row_buffer[buffer_idx];

            if hdr_row.is_empty() {
                break;
            }

            // Tonemap this row
            let sdr_row = self.tonemap_row(hdr_row, self.next_output_row);

            outputs.push(TonemapOutput {
                sdr_linear: sdr_row,
                row_index: self.next_output_row,
            });

            self.next_output_row += 1;

            // Update buffer start (allow rows to be overwritten)
            if self.next_output_row > self.config.lookahead_rows {
                self.buffer_start_row = self.next_output_row - self.config.lookahead_rows;
            }
        }

        Ok(outputs)
    }

    /// Tonemap a single row.
    fn tonemap_row(&self, hdr_row: &[f32], y: u32) -> Vec<f32> {
        let channels = self.config.channels as usize;
        let mut sdr_row = vec![0.0f32; self.width as usize * channels];
        let global = self.grid.global_params();

        for (x, (hdr_pixel, sdr_pixel)) in hdr_row
            .chunks(channels)
            .zip(sdr_row.chunks_mut(channels))
            .enumerate()
            .take(self.width as usize)
        {
            // Get local adaptation parameters
            let local = self.grid.sample(x as f32, y as f32);

            // Blend local with global for stability (especially at edges)
            let blend = 0.7; // 70% local, 30% global
            let params = LocalParams {
                key: local.key * blend + global.key * (1.0 - blend),
                white: local.white * blend + global.white * (1.0 - blend),
                black: local.black * blend + global.black * (1.0 - blend),
            };

            // Apply tonemapping
            let rgb = self.tonemap_pixel(
                [hdr_pixel[0], hdr_pixel[1], hdr_pixel[2]],
                &params,
            );

            sdr_pixel[0] = rgb[0];
            sdr_pixel[1] = rgb[1];
            sdr_pixel[2] = rgb[2];

            // Preserve alpha if present
            if channels >= 4 {
                sdr_pixel[3] = hdr_pixel.get(3).copied().unwrap_or(1.0);
            }
        }

        sdr_row
    }

    /// Tonemap a single pixel with local adaptation.
    fn tonemap_pixel(&self, rgb: [f32; 3], local: &LocalParams) -> [f32; 3] {
        let lum = luminance_bt709(rgb[0], rgb[1], rgb[2]);

        if lum <= 0.0 {
            return [self.config.shadow_lift; 3];
        }

        // 1. Local adaptation: scale to target key
        let key_scale = self.config.target_key / local.key.max(0.001);
        let adapted = lum * key_scale;

        // 2. Apply sigmoid tonemap curve
        let white_adapted = local.white * key_scale;
        let mapped = self.sigmoid_tonemap(adapted, white_adapted);

        // 3. Calculate luminance ratio
        let ratio = mapped / adapted;

        // 4. Apply to RGB with slight contrast boost
        let mut out = [
            (rgb[0] * key_scale * ratio).max(0.0),
            (rgb[1] * key_scale * ratio).max(0.0),
            (rgb[2] * key_scale * ratio).max(0.0),
        ];

        // 5. Highlight desaturation (AgX-style)
        let desat_start = white_adapted * self.config.desat_threshold;
        if adapted > desat_start {
            let t = ((adapted - desat_start) / (white_adapted - desat_start)).clamp(0.0, 1.0);
            let desat_factor = t * t * (1.0 - self.config.saturation);
            let out_lum = luminance_bt709(out[0], out[1], out[2]);
            out[0] = out[0] * (1.0 - desat_factor) + out_lum * desat_factor;
            out[1] = out[1] * (1.0 - desat_factor) + out_lum * desat_factor;
            out[2] = out[2] * (1.0 - desat_factor) + out_lum * desat_factor;
        }

        // 6. Shadow lift
        out[0] = out[0] + self.config.shadow_lift * (1.0 - out[0]);
        out[1] = out[1] + self.config.shadow_lift * (1.0 - out[1]);
        out[2] = out[2] + self.config.shadow_lift * (1.0 - out[2]);

        // 7. Clamp to valid range
        [
            out[0].clamp(0.0, 1.0),
            out[1].clamp(0.0, 1.0),
            out[2].clamp(0.0, 1.0),
        ]
    }

    /// Apply sigmoid tonemap curve.
    fn sigmoid_tonemap(&self, x: f32, white: f32) -> f32 {
        // Extended Reinhard with configurable contrast
        let x_scaled = x * self.config.contrast;
        let w2 = white * white;

        // Attempt at a punchier curve than basic Reinhard
        let knee = 0.5;
        if x_scaled < knee {
            // Linear region with slight boost
            x_scaled * 1.05
        } else {
            // Soft rolloff
            let base = knee * 1.05;
            let over = x_scaled - knee;
            let compressed = over * (1.0 + over / w2) / (1.0 + over);
            (base + compressed * (1.0 - knee)).min(1.0)
        }
    }

    /// Convert the tonemapped linear output to sRGB bytes (RGBA).
    pub fn linear_to_srgb8_rgba(linear: &[f32]) -> Vec<u8> {
        linear
            .chunks(4)
            .flat_map(|p| {
                [
                    (srgb_oetf(p[0]) * 255.0).round().clamp(0.0, 255.0) as u8,
                    (srgb_oetf(p[1]) * 255.0).round().clamp(0.0, 255.0) as u8,
                    (srgb_oetf(p[2]) * 255.0).round().clamp(0.0, 255.0) as u8,
                    (p.get(3).unwrap_or(&1.0) * 255.0).round().clamp(0.0, 255.0) as u8,
                ]
            })
            .collect()
    }

    /// Convert the tonemapped linear output to sRGB bytes (RGB, no alpha).
    pub fn linear_to_srgb8_rgb(linear: &[f32]) -> Vec<u8> {
        linear
            .chunks(3)
            .flat_map(|p| {
                [
                    (srgb_oetf(p[0]) * 255.0).round().clamp(0.0, 255.0) as u8,
                    (srgb_oetf(p[1]) * 255.0).round().clamp(0.0, 255.0) as u8,
                    (srgb_oetf(p[2]) * 255.0).round().clamp(0.0, 255.0) as u8,
                ]
            })
            .collect()
    }

    /// Convert linear output to sRGB bytes, auto-detecting channel count.
    ///
    /// Uses the configured channel count from this tonemapper instance.
    pub fn linear_to_srgb8(&self, linear: &[f32]) -> Vec<u8> {
        if self.config.channels >= 4 {
            Self::linear_to_srgb8_rgba(linear)
        } else {
            Self::linear_to_srgb8_rgb(linear)
        }
    }

    /// Get progress info: (rows_output, total_rows).
    pub fn progress(&self) -> (u32, u32) {
        (self.next_output_row, self.height)
    }

    /// Get the configured number of channels.
    pub fn channels(&self) -> u8 {
        self.config.channels
    }
}

// ============================================================================
// Helper Functions
// ============================================================================

#[inline]
fn luminance_bt709(r: f32, g: f32, b: f32) -> f32 {
    0.2126 * r + 0.7152 * g + 0.0722 * b
}

#[inline]
fn bilinear(v00: f32, v10: f32, v01: f32, v11: f32, fx: f32, fy: f32) -> f32 {
    let top = v00 * (1.0 - fx) + v10 * fx;
    let bottom = v01 * (1.0 - fx) + v11 * fx;
    top * (1.0 - fy) + bottom * fy
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_streaming_tonemapper_basic() {
        let width = 64u32;
        let height = 64u32;
        let config = StreamingTonemapConfig {
            lookahead_rows: 16,
            cell_size: 8,
            ..Default::default()
        };

        let mut tonemapper = StreamingTonemapper::new(width, height, config).unwrap();

        // Generate test HDR rows
        let mut output_rows = Vec::new();

        for _y in 0..height {
            let mut row = vec![0.0f32; width as usize * 4];
            for x in 0..width {
                let t = x as f32 / width as f32;
                let lum = t * 4.0; // 0 to 4 (HDR range)
                let idx = x as usize * 4;
                row[idx] = lum;
                row[idx + 1] = lum;
                row[idx + 2] = lum;
                row[idx + 3] = 1.0;
            }

            let outputs = tonemapper.push_row(&row).unwrap();
            output_rows.extend(outputs);
        }

        // Finish
        let final_outputs = tonemapper.finish().unwrap();
        output_rows.extend(final_outputs);

        // Should have all rows
        assert_eq!(output_rows.len(), height as usize);

        // Check output is valid
        for output in &output_rows {
            assert_eq!(output.sdr_linear.len(), width as usize * 4);

            // All values should be in [0, 1]
            for &val in &output.sdr_linear {
                assert!(val >= 0.0 && val <= 1.0, "Value out of range: {}", val);
            }
        }
    }

    #[test]
    fn test_local_adaptation() {
        let width = 128u32;
        let height = 128u32;
        let config = StreamingTonemapConfig::default();

        let mut tonemapper = StreamingTonemapper::new(width, height, config).unwrap();
        let mut all_outputs = Vec::new();

        // Create image with bright left half, dark right half
        for _y in 0..height {
            let mut row = vec![0.0f32; width as usize * 4];
            for x in 0..width {
                let lum = if x < width / 2 { 4.0 } else { 0.2 };
                let idx = x as usize * 4;
                row[idx] = lum;
                row[idx + 1] = lum;
                row[idx + 2] = lum;
                row[idx + 3] = 1.0;
            }
            all_outputs.extend(tonemapper.push_row(&row).unwrap());
        }

        all_outputs.extend(tonemapper.finish().unwrap());

        // Sort by row index
        all_outputs.sort_by_key(|o| o.row_index);

        // Should have all rows
        assert_eq!(all_outputs.len(), height as usize);

        // With local adaptation, both regions should produce valid SDR output
        // Check middle row
        let mid_row = &all_outputs[height as usize / 2];

        // Sample from bright region (input: 4.0 linear HDR)
        let bright_val = mid_row.sdr_linear[width as usize / 4 * 4];
        // Sample from dark region (input: 0.2 linear)
        let dark_val = mid_row.sdr_linear[width as usize * 3 / 4 * 4];

        // Both values should be in valid SDR range
        assert!(bright_val >= 0.0 && bright_val <= 1.0, "Bright val out of range: {}", bright_val);
        assert!(dark_val >= 0.0 && dark_val <= 1.0, "Dark val out of range: {}", dark_val);

        // Bright should map to a reasonable SDR value (tonemapped down from 4.0)
        assert!(bright_val > 0.1, "Bright region should not be too dark: {}", bright_val);

        // Dark region should be visible (lifted from potential black crush)
        assert!(dark_val > 0.02, "Dark region should be lifted: {}", dark_val);

        // Bright should be brighter than dark (basic sanity check)
        assert!(dark_val < bright_val, "Bright ({}) should be brighter than dark ({})", bright_val, dark_val);
    }

    #[test]
    fn test_highlight_desaturation() {
        let config = StreamingTonemapConfig::default();
        let grid = AdaptationGrid::new(64, 64, 8);

        let tonemapper = StreamingTonemapper {
            config,
            width: 64,
            height: 64,
            grid,
            row_buffer: vec![],
            buffer_start_row: 0,
            buffer_count: 0,
            next_output_row: 0,
            input_complete: false,
        };

        let local = LocalParams {
            key: 0.18,
            white: 4.0,
            black: 0.0,
        };

        // Test saturated red in highlights
        let bright_red = [4.0f32, 0.5, 0.5];
        let result = tonemapper.tonemap_pixel(bright_red, &local);

        // Should be desaturated toward white
        let r_g_diff = result[0] - result[1];
        let r_b_diff = result[0] - result[2];

        // Original input had r-g = 3.5, r-b = 3.5
        // After tonemapping and desaturation, differences should be reduced
        // (but not eliminated - we're just testing that desaturation is working)
        assert!(r_g_diff < 0.8, "Red-green diff should be reduced: {}", r_g_diff);
        assert!(r_b_diff < 0.8, "Red-blue diff should be reduced: {}", r_b_diff);

        // Also verify all channels are in valid range
        assert!(result[0] >= 0.0 && result[0] <= 1.0, "R out of range: {}", result[0]);
        assert!(result[1] >= 0.0 && result[1] <= 1.0, "G out of range: {}", result[1]);
        assert!(result[2] >= 0.0 && result[2] <= 1.0, "B out of range: {}", result[2]);
    }

    #[test]
    fn test_batch_push_rows() {
        let width = 64u32;
        let height = 64u32;
        let channels = 4usize;
        let config = StreamingTonemapConfig {
            lookahead_rows: 16,
            cell_size: 8,
            ..Default::default()
        };

        let mut tonemapper = StreamingTonemapper::new(width, height, config).unwrap();

        // Create batch of rows (simulate tile decoder giving us 8 rows at once)
        let batch_size = 8usize;
        let stride = width as usize * channels;
        let mut all_outputs = Vec::new();

        for batch_start in (0..height as usize).step_by(batch_size) {
            let rows_in_batch = batch_size.min(height as usize - batch_start);
            let mut batch = vec![0.0f32; rows_in_batch * stride];

            for row_in_batch in 0..rows_in_batch {
                let y = batch_start + row_in_batch;
                for x in 0..width as usize {
                    let t = (x as f32 + y as f32) / (width + height) as f32;
                    let lum = t * 4.0;
                    let idx = row_in_batch * stride + x * channels;
                    batch[idx] = lum;
                    batch[idx + 1] = lum * 0.8;
                    batch[idx + 2] = lum * 0.6;
                    batch[idx + 3] = 1.0;
                }
            }

            let outputs = tonemapper.push_rows(&batch, stride, rows_in_batch).unwrap();
            all_outputs.extend(outputs);
        }

        all_outputs.extend(tonemapper.finish().unwrap());
        all_outputs.sort_by_key(|o| o.row_index);

        assert_eq!(all_outputs.len(), height as usize);
    }

    #[test]
    fn test_rgb_3channel_mode() {
        let width = 32u32;
        let height = 32u32;
        let config = StreamingTonemapConfig::rgb();

        assert_eq!(config.channels, 3);

        let mut tonemapper = StreamingTonemapper::new(width, height, config).unwrap();
        let mut all_outputs = Vec::new();

        for _y in 0..height {
            let mut row = vec![0.0f32; width as usize * 3]; // RGB, no alpha
            for x in 0..width {
                let t = x as f32 / width as f32;
                let lum = t * 2.0;
                let idx = x as usize * 3;
                row[idx] = lum;
                row[idx + 1] = lum;
                row[idx + 2] = lum;
            }
            all_outputs.extend(tonemapper.push_row(&row).unwrap());
        }

        all_outputs.extend(tonemapper.finish().unwrap());

        // Should have all rows
        assert_eq!(all_outputs.len(), height as usize);

        // Each output row should be 3 channels
        for output in &all_outputs {
            assert_eq!(output.sdr_linear.len(), width as usize * 3);
        }

        // Convert to sRGB8
        let row0_srgb = tonemapper.linear_to_srgb8(&all_outputs[0].sdr_linear);
        assert_eq!(row0_srgb.len(), width as usize * 3);
    }

    #[test]
    fn test_contiguous_buffer() {
        let width = 64u32;
        let height = 64u32;
        let channels = 4usize;
        let stride = width as usize * channels;
        let config = StreamingTonemapConfig::default();

        let mut tonemapper = StreamingTonemapper::new(width, height, config).unwrap();

        // Create contiguous buffer (like from a decoded image)
        let mut buffer = vec![0.0f32; height as usize * stride];
        for y in 0..height as usize {
            for x in 0..width as usize {
                let idx = y * stride + x * channels;
                let lum = (x as f32 / width as f32) * 3.0;
                buffer[idx] = lum;
                buffer[idx + 1] = lum;
                buffer[idx + 2] = lum;
                buffer[idx + 3] = 1.0;
            }
        }

        // Push all rows at once using stride-based API
        let mut all_outputs = tonemapper.push_rows(&buffer, stride, height as usize).unwrap();
        all_outputs.extend(tonemapper.finish().unwrap());
        all_outputs.sort_by_key(|o| o.row_index);

        assert_eq!(all_outputs.len(), height as usize);
    }

    #[test]
    fn test_stride_with_padding() {
        let width = 60u32; // Not aligned
        let height = 32u32;
        let channels = 4usize;
        let stride = 64 * channels; // Padded to 64 pixels per row
        let config = StreamingTonemapConfig::default();

        let mut tonemapper = StreamingTonemapper::new(width, height, config).unwrap();

        // Create buffer with padding (simulates aligned memory)
        let mut buffer = vec![0.0f32; height as usize * stride];
        for y in 0..height as usize {
            for x in 0..width as usize {
                let idx = y * stride + x * channels;
                let lum = (x as f32 / width as f32) * 2.0;
                buffer[idx] = lum;
                buffer[idx + 1] = lum;
                buffer[idx + 2] = lum;
                buffer[idx + 3] = 1.0;
            }
            // Padding pixels (64-60=4 pixels) are left as zeros
        }

        let mut all_outputs = tonemapper.push_rows(&buffer, stride, height as usize).unwrap();
        all_outputs.extend(tonemapper.finish().unwrap());

        assert_eq!(all_outputs.len(), height as usize);

        // Output should be width * channels, not stride
        for output in &all_outputs {
            assert_eq!(output.sdr_linear.len(), width as usize * channels);
        }
    }
}
