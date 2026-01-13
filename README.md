# ultrahdr

Pure Rust implementation of [Ultra HDR](https://developer.android.com/media/platform/hdr-image-format) (gain map HDR) encoding and decoding.

Ultra HDR is a backward-compatible HDR image format that embeds a gain map in a standard JPEG, allowing HDR-capable displays to reconstruct the full HDR image while remaining viewable as SDR on legacy displays.

## Features

- **Encode**: Create Ultra HDR JPEGs from HDR images (with optional SDR input)
- **Decode**: Extract and apply gain maps to reconstruct HDR content
- **Tone mapping**: Automatic SDR generation from HDR-only input
- **Metadata**: Full XMP (hdrgm namespace) and ISO 21496-1 support
- **Pure Rust**: No C dependencies, uses [jpegli-rs](https://github.com/imazen/jpegli-rs) for JPEG

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

## License

MIT OR Apache-2.0

## AI-Generated Code Notice

This library was developed with assistance from Claude (Anthropic). The implementation has been tested against reference Ultra HDR images and passes comprehensive unit tests. Not all code has been manually reviewed - please review critical paths before production use.
