# ultrahdr

[![CI](https://github.com/imazen/ultrahdr/actions/workflows/ci.yml/badge.svg)](https://github.com/imazen/ultrahdr/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/imazen/ultrahdr/graph/badge.svg)](https://codecov.io/gh/imazen/ultrahdr)
[![crates.io](https://img.shields.io/crates/v/ultrahdr.svg)](https://crates.io/crates/ultrahdr)
[![docs.rs](https://docs.rs/ultrahdr/badge.svg)](https://docs.rs/ultrahdr)
[![MSRV](https://img.shields.io/badge/MSRV-1.92-blue)](https://blog.rust-lang.org/2025/07/03/Rust-1.92.0.html)
[![License](https://img.shields.io/crates/l/ultrahdr.svg)](LICENSE)

Pure Rust implementation of [Ultra HDR](https://developer.android.com/media/platform/hdr-image-format) (gain map HDR) encoding and decoding.

Ultra HDR is a backward-compatible HDR image format that embeds a gain map in a standard JPEG, allowing HDR-capable displays to reconstruct the full HDR image while remaining viewable as SDR on legacy displays.

## Crates

| Crate | Description |
|-------|-------------|
| [`ultrahdr`](ultrahdr/) | Full encoder/decoder with jpegli-rs JPEG codec |
| [`ultrahdr-core`](ultrahdr-core/) | Pure math and metadata - no codec dependency, WASM-compatible |

## Features

- **Encode**: Create Ultra HDR JPEGs from HDR images (with optional SDR input)
- **Decode**: Extract and apply gain maps to reconstruct HDR content
- **Tone mapping**: Automatic SDR generation from HDR-only input
- **Adaptive tonemapping**: Learn tone curves from existing HDR/SDR pairs
- **Metadata**: Full XMP (hdrgm namespace) and ISO 21496-1 support
- **Pure Rust**: No C dependencies, uses [jpegli-rs](https://github.com/imazen/jpegli-rs) for JPEG
- **WASM**: `ultrahdr-core` compiles to WebAssembly

## Usage

### Encoding

```rust
use ultrahdr::{Encoder, RawImage, PixelFormat, ColorGamut, ColorTransfer};

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
use ultrahdr::{Decoder, HdrOutputFormat};

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

## Using ultrahdr-core with jpegli-rs Directly

For more control, use `ultrahdr-core` (math + metadata only) with `jpegli-rs` for JPEG operations:

### Encoding UltraHDR

```rust
use ultrahdr_core::{
    gainmap::compute::{compute_gainmap, GainMapConfig},
    metadata::xmp::generate_xmp,
    RawImage, PixelFormat, ColorGamut, ColorTransfer, Unstoppable,
};
use jpegli::encoder::{EncoderConfig, PixelLayout, ChromaSubsampling, Unstoppable as JpegliStop};

// 1. Compute gain map from HDR + SDR
let config = GainMapConfig::default();
let (gainmap, metadata) = compute_gainmap(&hdr_image, &sdr_image, &config, Unstoppable)?;

// 2. Encode gain map to JPEG
let gainmap_jpeg = {
    let cfg = EncoderConfig::grayscale(75.0);
    let mut enc = cfg.encode_from_bytes(gainmap.width, gainmap.height, PixelLayout::Gray8Srgb)?;
    enc.push_packed(&gainmap.data, JpegliStop)?;
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
    enc.push_packed(&sdr_rgb, JpegliStop)?;
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
use jpegli::decoder::{Decoder, PreserveConfig};

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
enc.push_packed(&edited_sdr, JpegliStop)?;
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
