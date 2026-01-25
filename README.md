# ultrahdr

[![CI](https://github.com/imazen/ultrahdr/actions/workflows/ci.yml/badge.svg)](https://github.com/imazen/ultrahdr/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/imazen/ultrahdr/graph/badge.svg)](https://codecov.io/gh/imazen/ultrahdr)
[![crates.io](https://img.shields.io/crates/v/ultrahdr-rs.svg)](https://crates.io/crates/ultrahdr-rs)
[![docs.rs](https://docs.rs/ultrahdr-rs/badge.svg)](https://docs.rs/ultrahdr-rs)
[![MSRV](https://img.shields.io/badge/MSRV-1.92-blue)](https://blog.rust-lang.org/2025/07/03/Rust-1.92.0.html)
[![License](https://img.shields.io/crates/l/ultrahdr.svg)](LICENSE)

Pure Rust implementation of [Ultra HDR](https://developer.android.com/media/platform/hdr-image-format) (gain map HDR) encoding and decoding.

Ultra HDR is a backward-compatible HDR image format that embeds a gain map in a standard JPEG, allowing HDR-capable displays to reconstruct the full HDR image while remaining viewable as SDR on legacy displays.

## Crates

| Crate | Description |
|-------|-------------|
| [`ultrahdr-rs`](ultrahdr/) | Full encoder/decoder with zenjpeg JPEG codec |
| [`ultrahdr-core`](ultrahdr-core/) | Pure math and metadata - no codec dependency, WASM-compatible |

## Features

- **Encode**: Create Ultra HDR JPEGs from HDR images (with optional SDR input)
- **Decode**: Extract and apply gain maps to reconstruct HDR content
- **Tone mapping**: Automatic SDR generation from HDR-only input
- **Adaptive tonemapping**: Learn tone curves from existing HDR/SDR pairs
- **Metadata**: Full XMP (hdrgm namespace) and ISO 21496-1 support
- **Pure Rust**: No C dependencies, uses [zenjpeg](https://github.com/imazen/zenjpeg) for JPEG
- **WASM**: `ultrahdr-core` compiles to WebAssembly

## Usage

### Encoding

```rust
use ultrahdr_rs::{Encoder, RawImage, PixelFormat, ColorGamut, ColorTransfer};

// Create HDR image (linear float RGB, BT.2020 gamut)
let hdr_image = RawImage {
    width: 1920,
    height: 1080,
    format: PixelFormat::Rgba32F,
    gamut: ColorGamut::Bt2100,
    transfer: ColorTransfer::Linear,
    data: hdr_pixels,
    stride: 1920 * 16,
};

// Encode to Ultra HDR JPEG (SDR is auto-generated via tone mapping)
let ultrahdr_jpeg = Encoder::new()
    .set_hdr_image(hdr_image)
    .set_quality(90, 85)  // base quality, gainmap quality
    .set_gainmap_scale(4) // 1/4 resolution gain map
    .set_target_display_peak(1000.0) // nits
    .encode()?;

std::fs::write("output.jpg", &ultrahdr_jpeg)?;
```

### Decoding

```rust
use ultrahdr_rs::{Decoder, HdrOutputFormat};

let data = std::fs::read("ultrahdr.jpg")?;
let decoder = Decoder::new(&data)?;

if decoder.is_ultrahdr() {
    // Get HDR output (4x display boost)
    let hdr = decoder.decode_hdr(4.0, HdrOutputFormat::LinearFloat)?;

    // Or just get SDR
    let sdr = decoder.decode_sdr()?;

    // Inspect metadata
    let metadata = decoder.metadata();
    println!("HDR capacity: {:.1}x", metadata.hdr_capacity_max);
}
```

### Adaptive Tonemapping (Preserve Artistic Intent)

When editing HDR content, use `AdaptiveTonemapper` to learn the original tone curve and reproduce it:

```rust
use ultrahdr_core::color::{AdaptiveTonemapper, FitConfig};

// Learn tone curve from original HDR/SDR pair
let tonemapper = AdaptiveTonemapper::fit(&original_hdr, &original_sdr)?;

// Apply to edited HDR - preserves the original artistic intent
let new_sdr = tonemapper.apply(&edited_hdr)?;
```

## Supported Formats

### Input (HDR)
- `Rgba32F` - Linear float RGBA
- `Rgba16F` - Half-float RGBA
- `P010` - 10-bit YUV (BT.2020)

### Input (SDR)
- `Rgba8` - 8-bit sRGB RGBA
- `Rgb8` - 8-bit sRGB RGB

### Output (HDR)
- `LinearFloat` - Linear RGB float
- `Pq1010102` - PQ-encoded 10-bit packed
- `Srgb8` - Clipped to SDR range

## Metadata Formats

Both XMP and ISO 21496-1 metadata are supported for maximum compatibility:

- **XMP**: Adobe hdrgm namespace, embedded in APP1 marker
- **ISO 21496-1**: Binary format with fractions, typically in APP2

## Transfer Functions

- sRGB (IEC 61966-2-1)
- PQ/ST.2084 (HDR10)
- HLG (ITU-R BT.2100)

## Color Gamuts

- BT.709 (sRGB)
- Display P3
- BT.2100/BT.2020

## Pipeline Architecture

Understanding the correct sequencing is critical for both quality and memory efficiency.

### Streaming Encode Pipeline (Recommended)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         STREAMING ENCODE PIPELINE                           │
│                        (4 MB peak vs 165 MB batch)                          │
└─────────────────────────────────────────────────────────────────────────────┘

  HDR Source                                              Output Files
  (AVIF/JXL/                                              ┌──────────────┐
   EXR/etc)                                               │ Ultra HDR    │
      │                                                   │ JPEG         │
      ▼                                                   │ ┌──────────┐ │
┌───────────┐     ┌─────────────────────────────────┐     │ │ SDR JPEG │ │
│ Streaming │     │      COLOR MANAGEMENT           │     │ │ (primary)│ │
│ Decoder   │────▶│  ┌─────────────────────────┐    │     │ ├──────────┤ │
│ (rows)    │     │  │ 1. Input Transform      │    │     │ │ Gain Map │ │
└───────────┘     │  │    PQ/HLG → Linear      │    │     │ │ (APP15)  │ │
                  │  │    BT.2100 → Working    │    │     │ ├──────────┤ │
   16 rows        │  │    (use moxcms)         │    │     │ │ XMP      │ │
   at a time      │  └───────────┬─────────────┘    │     │ │ Metadata │ │
                  │              │                  │     │ └──────────┘ │
                  │              ▼                  │     └──────────────┘
                  │  ┌─────────────────────────┐    │
                  │  │ 2. Linear Working Space │    │
                  │  │    (HDR, scene-referred)│    │
                  │  └───────────┬─────────────┘    │
                  │              │                  │
                  │      ┌───────┴───────┐         │
                  │      │               │         │
                  │      ▼               ▼         │
                  │  ┌───────┐    ┌────────────┐   │
                  │  │ Keep  │    │ 3. Tonemap │   │
                  │  │ HDR   │    │ HDR → SDR  │   │
                  │  │ Linear│    │ (filmic/   │   │
                  │  └───┬───┘    │  adaptive) │   │
                  │      │        └─────┬──────┘   │
                  │      │              │          │
                  │      │              ▼          │
                  │      │    ┌─────────────────┐  │
                  │      │    │ 4. Output OETF  │  │
                  │      │    │    Linear→sRGB  │  │
                  │      │    │    (use moxcms) │  │
                  │      │    └────────┬────────┘  │
                  └──────│─────────────│───────────┘
                         │             │
                         ▼             ▼
               ┌─────────────────────────────────┐
               │        GAIN MAP ENCODER         │
               │  (RowEncoder / StreamEncoder)   │
               │                                 │
               │  Computes: gain = HDR/SDR       │
               │  Per-block, streaming output    │
               └────────────────┬────────────────┘
                                │
                    ┌───────────┴───────────┐
                    │                       │
                    ▼                       ▼
          ┌─────────────────┐    ┌─────────────────┐
          │  SDR JPEG       │    │  Gain Map JPEG  │
          │  Encoder        │    │  Encoder        │
          │  (streaming)    │    │  (streaming)    │
          │                 │    │                 │
          │  push_row()     │    │  push_row()     │
          └────────┬────────┘    └────────┬────────┘
                   │                      │
                   └──────────┬───────────┘
                              │
                              ▼
                   ┌─────────────────────┐
                   │   MPF Container     │
                   │   Assembly          │
                   │   + XMP Metadata    │
                   └─────────────────────┘
```

### Color Management: Where moxcms Fits

```
┌────────────────────────────────────────────────────────────────────────┐
│                    COLOR MANAGEMENT STAGES                              │
│                                                                         │
│  ┌─────────────┐      ┌─────────────┐      ┌─────────────┐            │
│  │   INPUT     │      │   WORKING   │      │   OUTPUT    │            │
│  │   SPACE     │ ───▶ │   SPACE     │ ───▶ │   SPACE     │            │
│  └─────────────┘      └─────────────┘      └─────────────┘            │
│                                                                         │
│  Examples:            Always:               Examples:                   │
│  • PQ BT.2100        • Linear              • sRGB (SDR output)        │
│  • HLG BT.2100       • Scene-referred      • Display P3               │
│  • Linear BT.2020    • Wide gamut          • PQ (HDR output)          │
│                        (BT.2020 or         • Linear (gain map)        │
│                         AP0/ACES)                                      │
│                                                                         │
│  ┌──────────────────────────────────────────────────────────────────┐  │
│  │                         moxcms handles:                          │  │
│  │  • EOTF/OETF (PQ, HLG, sRGB transfer functions)                 │  │
│  │  • Chromatic adaptation (D65 ↔ D50)                              │  │
│  │  • Gamut mapping (BT.2100 → sRGB with perceptual intent)        │  │
│  │  • ICC profile generation and parsing                            │  │
│  └──────────────────────────────────────────────────────────────────┘  │
│                                                                         │
│  ⚠️  CRITICAL: Tonemapping happens in LINEAR WORKING SPACE            │
│      Never tonemap PQ-encoded or sRGB-encoded values!                  │
│                                                                         │
└────────────────────────────────────────────────────────────────────────┘
```

### Streaming Decode Pipeline

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        STREAMING DECODE PIPELINE                        │
│                         (2 MB peak vs 166 MB)                           │
└─────────────────────────────────────────────────────────────────────────┘

  Ultra HDR JPEG
        │
        ▼
┌───────────────────┐
│ Parse MPF Header  │──────────────────────────────────────┐
│ Extract offsets   │                                      │
└────────┬──────────┘                                      │
         │                                                 │
    ┌────┴────┐                                           │
    │         │                                           │
    ▼         ▼                                           ▼
┌────────┐  ┌────────────┐                        ┌─────────────┐
│ SDR    │  │ Gain Map   │                        │ XMP/ISO     │
│ JPEG   │  │ JPEG       │                        │ Metadata    │
│ Decode │  │ Decode     │                        │ Parse       │
│(stream)│  │ (full or   │                        └──────┬──────┘
└───┬────┘  │  stream)   │                               │
    │       └─────┬──────┘                               │
    │             │                                      │
    │    ┌────────┴─────────────────────────────────────┐│
    │    │         GainMapMetadata                      ││
    │    │  • min/max_content_boost                     ││
    │    │  • gamma, offsets                            ││
    │    │  • hdr_capacity_min/max                      ││
    │    └────────┬─────────────────────────────────────┘│
    │             │                                      │
    ▼             ▼                                      │
┌─────────────────────────────────────┐                  │
│        HDR RECONSTRUCTION           │◀─────────────────┘
│        (RowDecoder/StreamDecoder)   │
│                                     │
│  For each pixel:                    │
│  1. Decode gain from gain map       │
│  2. Apply: HDR = (SDR + offset_sdr) │
│            × gain^weight            │
│            - offset_hdr             │
│  3. Bilinear upsample gain map      │
└──────────────────┬──────────────────┘
                   │
                   ▼
           ┌─────────────────┐
           │ OUTPUT TRANSFORM│
           │ (if needed)     │
           │ Linear → PQ/HLG │
           └────────┬────────┘
                    │
                    ▼
              HDR Output
```

### Memory Comparison

```
┌────────────────────────────────────────────────────────────────────────┐
│                     MEMORY USAGE: 4K (3840×2160)                       │
├────────────────────────────────────────────────────────────────────────┤
│                                                                        │
│  BATCH ENCODE (full images in memory)                                  │
│  ═══════════════════════════════════                                   │
│                                                                        │
│  Stage              Memory                                             │
│  ─────              ──────                                             │
│  Decode HDR         132 MB  ████████████████████████████████████████  │
│  + Resize buffer    +33 MB  ██████████                                │
│  + SDR copy         +33 MB  ██████████                                │
│  + Gain map          +1 MB  ▌                                         │
│  ─────────────────────────                                             │
│  PEAK:              165 MB                                             │
│                                                                        │
│  STREAMING ENCODE (row buffers)                                        │
│  ══════════════════════════════                                        │
│                                                                        │
│  Component          Memory                                             │
│  ─────────          ──────                                             │
│  Decoder buffer      1.0 MB  ███                                       │
│  Resize buffer       0.5 MB  ██                                        │
│  Tonemap (in-place)  0.0 MB                                            │
│  RowEncoder buffers  1.0 MB  ███                                       │
│  JPEG encoders       1.5 MB  █████                                     │
│  ─────────────────────────                                             │
│  PEAK:               4.0 MB  (40× reduction!)                          │
│                                                                        │
└────────────────────────────────────────────────────────────────────────┘
```

### Common Mistakes to Avoid

```
┌────────────────────────────────────────────────────────────────────────┐
│                          ❌ WRONG                                      │
├────────────────────────────────────────────────────────────────────────┤
│                                                                        │
│  1. Tonemapping PQ-encoded values                                      │
│     ✗ let sdr = tonemap(pq_pixel);  // PQ is perceptual, not linear!  │
│     ✓ let linear = pq_eotf(pq_pixel);                                 │
│       let sdr = tonemap(linear);                                       │
│                                                                        │
│  2. Computing gain map from sRGB (not linear)                          │
│     ✗ gain = srgb_hdr / srgb_sdr;  // Wrong! sRGB is nonlinear        │
│     ✓ gain = linear_hdr / linear_sdr;                                 │
│                                                                        │
│  3. Loading full image when streaming works                            │
│     ✗ let full_image = decoder.decode_all()?;  // 132 MB              │
│     ✓ for row in decoder.rows() { ... }        // 1 MB                │
│                                                                        │
│  4. Applying sRGB OETF before gain map computation                     │
│     ✗ let sdr = srgb_oetf(linear_sdr);                                │
│       compute_gainmap(hdr_linear, sdr);  // Mixing linear and sRGB!   │
│     ✓ compute_gainmap(hdr_linear, sdr_linear);                        │
│       let sdr_output = srgb_oetf(sdr_linear);                         │
│                                                                        │
│  5. Ignoring color gamut conversion                                    │
│     ✗ SDR in BT.2020 gamut (out-of-range values)                      │
│     ✓ Convert BT.2020 → sRGB with gamut mapping before SDR output     │
│                                                                        │
└────────────────────────────────────────────────────────────────────────┘
```

### Correct Pipeline Order

```
┌────────────────────────────────────────────────────────────────────────┐
│                          ✓ CORRECT ORDER                               │
├────────────────────────────────────────────────────────────────────────┤
│                                                                        │
│  ENCODE:                                                               │
│  ═══════                                                               │
│  1. Decode HDR source (get encoded pixels)                             │
│  2. Apply EOTF (PQ/HLG → Linear)           ← moxcms                   │
│  3. Convert gamut to working space         ← moxcms                   │
│  4. Tonemap (linear HDR → linear SDR)      ← ultrahdr-core            │
│  5. Compute gain map (both in linear!)     ← ultrahdr-core            │
│  6. Convert SDR gamut to output space      ← moxcms                   │
│  7. Apply OETF (Linear → sRGB)             ← moxcms                   │
│  8. Encode SDR JPEG                        ← zenjpeg                  │
│  9. Encode gain map JPEG                   ← zenjpeg                  │
│  10. Assemble MPF container + XMP          ← ultrahdr-core            │
│                                                                        │
│  DECODE:                                                               │
│  ═══════                                                               │
│  1. Parse MPF, extract SDR + gain map JPEGs                            │
│  2. Parse XMP/ISO metadata                  ← ultrahdr-core           │
│  3. Decode SDR JPEG                         ← zenjpeg                 │
│  4. Decode gain map JPEG                    ← zenjpeg                 │
│  5. Apply EOTF to SDR (sRGB → Linear)      ← moxcms                   │
│  6. Apply gain map (in linear space!)       ← ultrahdr-core           │
│  7. Convert gamut if needed                 ← moxcms                   │
│  8. Apply OETF for output (Linear → PQ)    ← moxcms                   │
│                                                                        │
└────────────────────────────────────────────────────────────────────────┘
```

## Streaming APIs (Low Memory)

For memory-constrained environments, `ultrahdr-core` provides streaming APIs that process images row-by-row:

```rust
use ultrahdr_core::gainmap::streaming::{RowDecoder, RowEncoder, DecodeInput, EncodeInput};
```

| Type | Direction | Memory | Use Case |
|------|-----------|--------|----------|
| `RowDecoder` | SDR+gainmap→HDR | Full gainmap in RAM | Gainmap fits in memory |
| `StreamDecoder` | SDR+gainmap→HDR | 16-row ring buffer | Parallel JPEG decode |
| `RowEncoder` | HDR+SDR→gainmap | Synchronized batches | Same-rate inputs |
| `StreamEncoder` | HDR+SDR→gainmap | Independent buffers | Parallel decode sources |

### Streaming Decode Example

```rust
use ultrahdr_core::gainmap::streaming::{RowDecoder, DecodeInput};
use ultrahdr_core::{HdrOutputFormat, ColorGamut};

// Load gainmap fully, then stream SDR rows
let mut decoder = RowDecoder::new(
    gainmap, metadata, width, height, 4.0, HdrOutputFormat::LinearFloat, ColorGamut::Bt709
)?;

// Process in 16-row batches (JPEG MCU alignment)
for batch_start in (0..height).step_by(16) {
    let batch_height = 16.min(height - batch_start);
    let sdr_batch = jpeg_decoder.next_rows(batch_height);
    let hdr_batch = decoder.process_rows(&sdr_batch, batch_height)?;
    write_output(&hdr_batch);
}
```

### Memory Savings (4K image)

| API | Peak Memory |
|-----|-------------|
| Full decode | ~166 MB |
| Streaming (16 rows) | ~2 MB |

## Streaming Tonemapper

`StreamingTonemapper` provides high-quality HDR→SDR tonemapping in a single streaming pass with local adaptation.

### Semantics

```
┌────────────────────────────────────────────────────────────────────────┐
│                    STREAMING TONEMAPPER FLOW                           │
│                                                                         │
│   Input                    Internal                      Output         │
│   ──────                   ────────                      ──────         │
│                                                                         │
│   Row 0  ─────┐                                                        │
│   Row 1  ─────┤       ┌─────────────────────┐                          │
│   Row 2  ─────┼──────▶│   Lookahead Buffer  │                          │
│    ...   ─────┤       │   (ring buffer)     │                          │
│   Row N  ─────┘       │   Default: 64 rows  │                          │
│                       └──────────┬──────────┘                          │
│                                  │                                     │
│                                  ▼                                     │
│                       ┌─────────────────────┐                          │
│                       │  Local Adaptation   │                          │
│                       │  Grid (1/8 res)     │                          │
│                       │  • Per-cell stats   │                          │
│                       │  • Key (geo mean)   │                          │
│                       │  • White point      │                          │
│                       └──────────┬──────────┘                          │
│                                  │                                     │
│   ⚠️ OUTPUT LAG                  │                                     │
│   ═════════════                  ▼                                     │
│                       ┌─────────────────────┐       Row 0 ────▶       │
│   After pushing       │    Tonemap with     │       Row 1 ────▶       │
│   row 32, you get     │  local adaptation   │       Row 2 ────▶       │
│   row 0 out           │  • AgX highlights   │        ...              │
│                       │  • Shadow lift      │                          │
│   Lag = lookahead/2   └─────────────────────┘                          │
│       = 32 rows                                                        │
│                                                                         │
└────────────────────────────────────────────────────────────────────────┘
```

**Key points:**
- **Output lag**: Rows come out `lookahead_rows / 2` behind input (default: 32 rows)
- **Row order preserved**: Output row indices match input, just delayed
- **Call `finish()`**: Required to flush remaining rows after all input is pushed
- **Memory**: ~6 MB for 4K (grid + row buffer)

### API

```rust
use ultrahdr_core::color::{StreamingTonemapper, StreamingTonemapConfig};

// Configure (defaults shown)
let config = StreamingTonemapConfig {
    channels: 4,          // 3 for RGB, 4 for RGBA
    lookahead_rows: 64,   // Buffer size (affects quality & lag)
    cell_size: 8,         // Local adaptation grid: image_size / cell_size
    target_key: 0.18,     // Target mid-gray
    contrast: 1.1,        // Subtle contrast boost
    saturation: 0.95,     // Slight highlight desaturation
    shadow_lift: 0.02,    // Lift shadows slightly
    desat_threshold: 0.5, // Start desaturating at 50% of white
};

let mut tm = StreamingTonemapper::new(width, height, config)?;

// Push rows: (data, stride, num_rows)
// stride = elements between row starts (>= width * channels)
let outputs = tm.push_rows(&hdr_buffer, stride, num_rows)?;

// Process outputs as they become ready
for out in outputs {
    // out.row_index: which row this is (may not be sequential!)
    // out.sdr_linear: linear f32 SDR data, ready for OETF
    let srgb = tm.linear_to_srgb8(&out.sdr_linear);
    jpeg_encoder.push_row(&srgb)?;
}

// Flush remaining rows (REQUIRED!)
for out in tm.finish()? {
    let srgb = tm.linear_to_srgb8(&out.sdr_linear);
    jpeg_encoder.push_row(&srgb)?;
}
```

### Output Ordering

Because of the lookahead buffer, outputs may not arrive in order during streaming.
The `row_index` field tells you which row each output corresponds to.

```rust
// If you need sequential output (e.g., for JPEG encoder), buffer and sort:
let mut pending: BTreeMap<u32, Vec<f32>> = BTreeMap::new();
let mut next_to_emit = 0u32;

for out in tm.push_rows(&data, stride, rows)? {
    pending.insert(out.row_index, out.sdr_linear);

    // Emit any consecutive rows starting from next_to_emit
    while let Some(row) = pending.remove(&next_to_emit) {
        jpeg_encoder.push_row(&tm.linear_to_srgb8(&row))?;
        next_to_emit += 1;
    }
}
```

### Memory Usage

| Image Size | Lookahead | Grid | Buffers | Total |
|------------|-----------|------|---------|-------|
| 1920×1080  | 64 rows   | 0.5 MB | 2 MB  | ~2.5 MB |
| 3840×2160  | 64 rows   | 2 MB   | 4 MB  | ~6 MB   |
| 7680×4320  | 64 rows   | 8 MB   | 8 MB  | ~16 MB  |

Compare to full-frame tonemapping: 132 MB for 4K (entire image in RAM).

## Using ultrahdr-core with zenjpeg Directly

For more control, use `ultrahdr-core` (math + metadata only) with `zenjpeg` for JPEG operations:

### Encoding UltraHDR

```rust
use ultrahdr_core::{
    gainmap::compute::{compute_gainmap, GainMapConfig},
    metadata::xmp::generate_xmp,
    RawImage, PixelFormat, ColorGamut, ColorTransfer, Unstoppable,
};
use zenjpeg::encoder::{EncoderConfig, PixelLayout, ChromaSubsampling, Unstoppable as ZenjpegStop};

// 1. Compute gain map from HDR + SDR
let config = GainMapConfig::default();
let (gainmap, metadata) = compute_gainmap(&hdr_image, &sdr_image, &config, Unstoppable)?;

// 2. Encode gain map to JPEG
let gainmap_jpeg = {
    let cfg = EncoderConfig::grayscale(75.0);
    let mut enc = cfg.encode_from_bytes(gainmap.width, gainmap.height, PixelLayout::Gray8Srgb)?;
    enc.push_packed(&gainmap.data, ZenjpegStop)?;
    enc.finish()?
};

// 3. Generate XMP metadata
let xmp = generate_xmp(&metadata, gainmap_jpeg.len());

// 4. Encode UltraHDR with embedded gain map
let ultrahdr = {
    let cfg = EncoderConfig::ycbcr(90.0, ChromaSubsampling::Quarter)
        .xmp(xmp.as_bytes().to_vec())
        .add_gainmap(gainmap_jpeg);
    let mut enc = cfg.encode_from_bytes(width, height, PixelLayout::Rgb8Srgb)?;
    enc.push_packed(&sdr_rgb, ZenjpegStop)?;
    enc.finish()?
};
```

### Decoding UltraHDR

```rust
use ultrahdr_core::{
    gainmap::apply::{apply_gainmap, HdrOutputFormat},
    metadata::xmp::parse_xmp,
    GainMap, RawImage, Unstoppable,
};
use zenjpeg::decoder::{Decoder, PreserveConfig};

// 1. Decode with metadata preservation
let decoded = Decoder::new()
    .preserve(PreserveConfig::default())
    .decode(&ultrahdr_jpeg)?;

let extras = decoded.extras().expect("extras");

// 2. Parse XMP metadata
let xmp_str = extras.xmp().expect("XMP");
let (metadata, _) = parse_xmp(xmp_str)?;

// 3. Decode gain map JPEG
let gainmap_jpeg = extras.gainmap().expect("gainmap");
let gainmap_decoded = Decoder::new().decode(gainmap_jpeg)?;

// 4. Build RawImage and GainMap structs
let sdr = RawImage::from_data(
    decoded.width, decoded.height,
    PixelFormat::Rgba8, ColorGamut::Bt709, ColorTransfer::Srgb,
    rgba_pixels,
)?;
let gainmap = GainMap {
    width: gainmap_decoded.width,
    height: gainmap_decoded.height,
    channels: 1,
    data: gainmap_decoded.data,
};

// 5. Apply gain map to reconstruct HDR
let hdr = apply_gainmap(&sdr, &gainmap, &metadata, 4.0, HdrOutputFormat::LinearFloat, Unstoppable)?;
```

### Lossless Round-Trip (Edit SDR, Preserve Gain Map)

```rust
// Decode
let decoded = Decoder::new().preserve(PreserveConfig::default()).decode(&ultrahdr)?;
let extras = decoded.extras().unwrap();

// Edit SDR pixels...
let edited_sdr: Vec<u8> = /* your edits */;

// Re-encode preserving XMP + gainmap
let encoder_segments = extras.to_encoder_segments();
let cfg = EncoderConfig::ycbcr(90.0, ChromaSubsampling::Quarter)
    .with_segments(encoder_segments);  // Preserves XMP + gainmap
let mut enc = cfg.encode_from_bytes(width, height, PixelLayout::Rgb8Srgb)?;
enc.push_packed(&edited_sdr, ZenjpegStop)?;
let re_encoded = enc.finish()?;
```

## Cooperative Cancellation

Long-running operations accept an `impl Stop` parameter from the [`enough`](https://crates.io/crates/enough) crate for cooperative cancellation:

```rust
use ultrahdr_core::{Unstoppable, Stop};
use enough::AtomicStop;

// Simple usage - no cancellation
let (gainmap, metadata) = compute_gainmap(&hdr, &sdr, &config, Unstoppable)?;

// With cancellation support
let stop = AtomicStop::new();
let stop_clone = stop.clone();
std::thread::spawn(move || {
    std::thread::sleep(Duration::from_secs(5));
    stop_clone.stop();
});
let result = compute_gainmap(&hdr, &sdr, &config, &stop);
```

## License

Apache-2.0

## AI-Generated Code Notice

This library was developed with assistance from Claude (Anthropic). The implementation has been tested against reference Ultra HDR images and passes comprehensive unit tests. Not all code has been manually reviewed - please review critical paths before production use.
