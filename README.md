# ultrahdr [![CI](https://img.shields.io/github/actions/workflow/status/imazen/ultrahdr/ci.yml?style=flat-square)](https://github.com/imazen/ultrahdr/actions/workflows/ci.yml) [![crates.io](https://img.shields.io/crates/v/ultrahdr-rs?style=flat-square)](https://crates.io/crates/ultrahdr-rs) [![docs.rs](https://img.shields.io/docsrs/ultrahdr-rs?style=flat-square)](https://docs.rs/ultrahdr-rs) [![codecov](https://img.shields.io/codecov/c/github/imazen/ultrahdr?style=flat-square)](https://codecov.io/gh/imazen/ultrahdr) [![MSRV](https://img.shields.io/badge/MSRV-1.92-blue?style=flat-square)](https://doc.rust-lang.org/cargo/reference/manifest.html#the-rust-version-field) [![license](https://img.shields.io/crates/l/ultrahdr-rs?style=flat-square)](https://github.com/imazen/ultrahdr#license)

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

## Comparison with C++ libultrahdr

| Feature | ultrahdr-rs | C++ libultrahdr |
|---------|:-----------:|:---------------:|
| **Encoding** | | |
| HDR + SDR → Ultra HDR JPEG | Yes | Yes |
| HDR-only (auto-tonemap SDR) | Yes | Yes |
| Adaptive tonemapping (learn curves) | Yes | No |
| Streaming encode (low memory) | Yes | No |
| Multi-channel gain map | Yes | Yes |
| **Decoding** | | |
| Ultra HDR → HDR reconstruct | Yes | Yes |
| Display boost parameter | Yes | Yes |
| Gain map extraction (raw JPEG) | Yes | Yes |
| **Metadata** | | |
| XMP (hdrgm namespace) | Yes | Yes |
| ISO 21496-1 binary | Yes | Yes |
| MPF (Multi-Picture Format) | Yes | Yes |
| **Pixel Formats** | | |
| RGBA 8-bit (SDR) | Yes | Yes |
| RGBA 32F / 16F (HDR) | Yes | Yes |
| P010 (10-bit YUV) | Yes | Yes |
| RGBA 1010102 (PQ/HLG) | Yes | Yes |
| **Transfer Functions** | | |
| sRGB | Yes | Yes |
| PQ (ST.2084) | Yes | Yes |
| HLG (BT.2100) | Yes | Yes |
| **Color Gamuts** | | |
| BT.709 / sRGB | Yes | Yes |
| Display P3 | Yes | Yes |
| BT.2100 / BT.2020 | Yes | Yes |
| **Platform** | | |
| Pure Rust (no C deps) | Yes | No (C++) |
| WASM support | Yes (`ultrahdr-core`) | No |
| `no_std` support | Yes (`ultrahdr-core`) | No |
| JPEG codec bundled | Optional (zenjpeg) | Yes (built-in) |
| **Not Yet Implemented** | | |
| JPEG-R (ISO 21496-1 container) | No | Yes |
| Editing API (in-place metadata update) | No | Yes |
| GPU acceleration | No | Yes (OpenGL) |

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
use ultrahdr_rs::Decoder;

let data = std::fs::read("ultrahdr.jpg")?;
let decoder = Decoder::new(&data)?;

if decoder.is_ultrahdr() {
    // Get HDR output (4x display boost)
    let hdr = decoder.decode_hdr(4.0)?;

    // Or just get SDR
    let sdr = decoder.decode_sdr()?;

    // Inspect metadata
    let metadata = decoder.metadata();
    println!("HDR capacity: {:.1}x", metadata.unwrap().hdr_capacity_max);
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

Gain map metadata tells the decoder how to reconstruct HDR from the SDR base image: per-channel min/max boost, gamma curve, offsets, and HDR headroom. This metadata can be serialized two different ways, and Ultra HDR JPEGs often carry both.

### XMP vs ISO 21496-1

| | XMP (`hdrgm:` namespace) | ISO 21496-1 binary |
|---|---|---|
| **Location** | APP1 marker in gain map JPEG | APP2 marker in gain map JPEG |
| **Encoding** | XML text with float strings | Big-endian rational fractions |
| **Precision** | Limited by decimal string formatting | Exact rational representation |
| **Support** | Universal (all Ultra HDR readers) | Android 15+, iOS 18+, Chromium |
| **Priority** | Fallback when ISO is absent | Authoritative when present |

Use `GainMapEncodingFormat::Both` (the default) to emit both. Cost is ~60 extra bytes for the ISO binary block.

```rust
use ultrahdr_core::GainMapEncodingFormat;

// Default: maximum compatibility
let format = GainMapEncodingFormat::Both;

// XMP-only: legacy compatibility, no ISO binary
let format = GainMapEncodingFormat::Xmp;

// ISO-only: newer readers, smaller output
let format = GainMapEncodingFormat::Iso21496;
```

### ISO 21496-1 Wire Format

The same metadata payload (flags byte + rational fraction pairs) appears in three container formats. The only difference is whether the payload has a leading `version(u8)` byte:

| Container | Header | Size | Used by |
|-----------|--------|------|---------|
| JPEG APP2 | `min_ver(u16) + writer_ver(u16) + flags(u8)` | 5 bytes | libultrahdr, this crate |
| JXL `jhgm` box | Same as JPEG (no version byte) | 5 bytes | libjxl |
| AVIF `tmap` item | `version(u8)` + JPEG header | 6 bytes | libavif |

JPEG doesn't need the version byte because the APP2 URN namespace (`urn:iso:std:iso:ts:21496:-1`) already identifies the format. AVIF wraps the payload in an ISOBMFF box that needs its own versioning.

Select the wire format variant with `Iso21496Format`:

```rust
use ultrahdr_core::Iso21496Format;

// JPEG APP2 and JXL jhgm (no version byte prefix)
let bytes = serialize_iso21496(&metadata, Iso21496Format::JpegApp2);

// AVIF tmap (with version byte prefix)
let bytes = serialize_iso21496(&metadata, Iso21496Format::AvifTmap);
```

### Fraction Encoding

Each metadata value (min/max boost, gamma, offsets, headroom) is stored as a rational fraction: `i32` or `u32` numerator over `u32` denominator, big-endian. The serializer uses a continued fraction algorithm (ported from libultrahdr) that finds the simplest exact representation:

| Value | Wire fraction | Bytes |
|-------|--------------|-------|
| `0.0` | `0/1` | `00000000 00000001` |
| `4.0` | `4/1` | `00000004 00000001` |
| `1/64` (0.015625) | `1/64` | `00000001 00000040` |
| `5.622376...` | `5895489/1048576` | `0059F541 00100000` |

This matters for browser interop. Chromium prefers ISO over XMP when both are present, and non-canonical fraction encodings (like a fixed 1,000,000 denominator) cause HDR rendering failures. See [jcayzac/ultrajpeg#6](https://github.com/jcayzac/ultrajpeg/issues/6) for the full story.

### JPEG Structure

A canonical Ultra HDR JPEG has this structure:

```
Primary JPEG (SDR base image):
  SOI
  APP1  XMP: Container:Directory pointing to gain map by byte length
  APP2  ISO 21496-1: version-only block [00 00 00 00]
  APP2  ICC profile
  APP2  MPF directory (offsets to secondary image)
  [entropy-coded image data]
  EOI

Gain Map JPEG (secondary image, appended after primary EOI):
  SOI
  APP1  XMP: hdrgm:* namespace with metadata values        <- if Xmp or Both
  APP2  ISO 21496-1: full metadata payload (61+ bytes)     <- if Iso21496 or Both
  APP2  ICC profile (gain map color space)
  [entropy-coded gain map data]
  EOI
```

The primary JPEG's 4-byte version-only ISO APP2 (`min_version=0, writer_version=0`) signals ISO 21496-1 awareness without carrying gain map parameters. The actual metadata lives in the secondary JPEG. Legacy readers ignore both APP2 blocks and see a normal JPEG.

### High-Level API

For JPEG encoding, `create_jpeg_iso_markers` produces both APP2 markers in one call:

```rust
use ultrahdr_core::metadata::iso21496::create_jpeg_iso_markers;

let markers = create_jpeg_iso_markers(&metadata);
// markers.primary  -> version-only APP2 for the primary JPEG
// markers.gain_map -> full metadata APP2 for the gain map JPEG
```

This is what `encode_ultrahdr` uses internally. You only need the low-level API if you're assembling JPEG segments yourself.

### Cross-Format Comparison

| Format | Metadata location | Serializer | Notes |
|--------|-------------------|------------|-------|
| **JPEG** (Ultra HDR) | APP2 in gain map JPEG + XMP | `ultrahdr-core` | Both XMP and ISO supported |
| **AVIF** (`tmap`) | ISOBMFF `tmap` derived image item | `zencodec` | ISO 21496-1 only (with version byte) |
| **JXL** (`jhgm`) | `jhgm` codestream box | `zencodec` | ISO 21496-1 only (JPEG wire variant) |
| **HEIC** (Apple) | Auxiliary image + XMP | `heic` crate | Apple-proprietary, not ISO 21496-1 |

AVIF and JXL always use ISO 21496-1 binary; there's no XMP option. HEIC uses Apple's own gain map mechanism (`urn:com:apple:photo:2020:aux:hdrgainmap`) with a simpler headroom-only metadata model.

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
use ultrahdr_core::gainmap::streaming::{RowDecoder, RowEncoder};
```

| Type | Direction | Memory | Use Case |
|------|-----------|--------|----------|
| `RowDecoder` | SDR+gainmap→HDR | Full gainmap in RAM | Gainmap fits in memory |
| `StreamDecoder` | SDR+gainmap→HDR | 16-row ring buffer | Parallel JPEG decode |
| `RowEncoder` | HDR+SDR→gainmap | Synchronized batches | Same-rate inputs |
| `StreamEncoder` | HDR+SDR→gainmap | Independent buffers | Parallel decode sources |

### Streaming Decode Example

```rust
use ultrahdr_core::gainmap::streaming::RowDecoder;
use ultrahdr_core::ColorGamut;

// Load gainmap fully, then stream SDR rows (linear f32)
let mut decoder = RowDecoder::new(
    gainmap, metadata, width, height, 4.0, ColorGamut::Bt709
)?;

// Process in 16-row batches (JPEG MCU alignment)
// Note: SDR input must be linear f32 RGB (3 floats per pixel)
for batch_start in (0..height).step_by(16) {
    let batch_height = 16.min(height - batch_start);
    let sdr_batch = jpeg_decoder.next_rows(batch_height); // linear f32
    let hdr_batch = decoder.process_sdr_rows(&sdr_batch, batch_height)?;
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
        .add_gainmap(gainmap_jpeg);
    let mut enc = cfg.request()
        .xmp(xmp.as_bytes())
        .encode_from_bytes(width, height, PixelLayout::Rgb8Srgb)?;
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
    .decode(&ultrahdr_jpeg, Unstoppable)?;

let extras = decoded.extras().expect("extras");

// 2. Parse XMP metadata
let xmp_str = extras.xmp().expect("XMP");
let (metadata, _) = parse_xmp(xmp_str)?;

// 3. Decode gain map JPEG
let gainmap_jpeg = extras.gainmap().expect("gainmap");
let gainmap_decoded = Decoder::new().decode(gainmap_jpeg, Unstoppable)?;

// 4. Build RawImage and GainMap structs
let sdr = RawImage::from_data(
    decoded.width(), decoded.height(),
    PixelFormat::Rgba8, ColorGamut::Bt709, ColorTransfer::Srgb,
    rgba_pixels,
)?;
let gainmap = GainMap {
    width: gainmap_decoded.width(),
    height: gainmap_decoded.height(),
    channels: 1,
    data: gainmap_decoded.pixels_u8().unwrap().to_vec(),
};

// 5. Apply gain map to reconstruct HDR
let hdr = apply_gainmap(&sdr, &gainmap, &metadata, 4.0, HdrOutputFormat::LinearFloat, Unstoppable)?;
```

### Lossless Round-Trip (Edit SDR, Preserve Gain Map)

```rust
// Decode
let decoded = Decoder::new().preserve(PreserveConfig::default()).decode(&ultrahdr, Unstoppable)?;
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

## Known Differences from libultrahdr

This implementation aims for compatibility with [Google's libultrahdr](https://github.com/google/libultrahdr) reference implementation, but has the following known differences:

### XMP Metadata Validation (ultrahdr-core)

| Behavior | libultrahdr | This Implementation |
|----------|-------------|---------------------|
| `BaseRenditionIsHDR="True"` | **Rejected** with error | ⚠️ Accepted (should reject) |
| Required fields (Version, GainMapMax, HDRCapacityMax) | **All required** | ⚠️ Only checks if Version OR GainMapMax present |
| Unparseable field values | **Error** | ⚠️ Silently uses defaults |

### JPEG Boundary Detection

| Behavior | libultrahdr | This Implementation |
|----------|-------------|---------------------|
| Primary method | JpegScanner (SOI/EOI markers) | MPF directory parsing |
| Fallback | N/A | SOI/EOI marker scanning |
| Marker-aware scanning | Yes (skips marker payloads) | ⚠️ Simple scan in ultrahdr-core, robust scan in zenjpeg |
| >2 images warning | Yes | No |

### Practical Impact

- Files with `BaseRenditionIsHDR="True"` (rare) may decode incorrectly
- Files with missing required XMP fields may use incorrect default values
- Detection should work for all standard Ultra HDR files

### Tracking

These differences are tracked for future fixes. Contributions welcome.

## Image tech I maintain

| | |
|:--|:--|
| State of the art codecs* | [zenjpeg] · [zenpng] · [zenwebp] · [zengif] · [zenavif] ([rav1d-safe] · [zenrav1e] · [zenavif-parse] · [zenavif-serialize]) · [zenjxl] ([jxl-encoder] · [zenjxl-decoder]) · [zentiff] · [zenbitmaps] · [heic] · [zenraw] · [zenpdf] · **ultrahdr** · [mozjpeg-rs] · [webpx] |
| Compression | [zenflate] · [zenzop] |
| Processing | [zenresize] · [zenfilters] · [zenquant] · [zenblend] |
| Metrics | [zensim] · [fast-ssim2] · [butteraugli] · [resamplescope-rs] · [codec-eval] · [codec-corpus] |
| Pixel types & color | [zenpixels] · [zenpixels-convert] · [linear-srgb] · [garb] |
| Pipeline | [zenpipe] · [zencodec] · [zencodecs] · [zenlayout] · [zennode] |
| ImageResizer | [ImageResizer] (C#) — 24M+ NuGet downloads across all packages |
| [Imageflow][] | Image optimization engine (Rust) — [.NET][imageflow-dotnet] · [node][imageflow-node] · [go][imageflow-go] — 9M+ NuGet downloads across all packages |
| [Imageflow Server][] | [The fast, safe image server](https://www.imazen.io/) (Rust+C#) — 552K+ NuGet downloads, deployed by Fortune 500s and major brands |

<sub>* as of 2026</sub>

### General Rust awesomeness

[archmage] · [magetypes] · [enough] · [whereat] · [zenbench] · [cargo-copter]

[And other projects](https://www.imazen.io/open-source) · [GitHub @imazen](https://github.com/imazen) · [GitHub @lilith](https://github.com/lilith) · [lib.rs/~lilith](https://lib.rs/~lilith) · [NuGet](https://www.nuget.org/profiles/imazen) (over 30 million downloads / 87 packages)

## License

Apache-2.0

## AI-Generated Code Notice

This library was developed with assistance from Claude (Anthropic). The implementation has been tested against reference Ultra HDR images and passes comprehensive unit tests. Not all code has been manually reviewed - please review critical paths before production use.

[zenjpeg]: https://github.com/imazen/zenjpeg
[zenpng]: https://github.com/imazen/zenpng
[zenwebp]: https://github.com/imazen/zenwebp
[zengif]: https://github.com/imazen/zengif
[zenavif]: https://github.com/imazen/zenavif
[zenjxl]: https://github.com/imazen/zenjxl
[zentiff]: https://github.com/imazen/zentiff
[zenbitmaps]: https://github.com/imazen/zenbitmaps
[heic]: https://github.com/imazen/heic-decoder-rs
[zenraw]: https://github.com/imazen/zenraw
[zenpdf]: https://github.com/imazen/zenpdf
[jxl-encoder]: https://github.com/imazen/jxl-encoder
[zenjxl-decoder]: https://github.com/imazen/zenjxl-decoder
[rav1d-safe]: https://github.com/imazen/rav1d-safe
[zenrav1e]: https://github.com/imazen/zenrav1e
[mozjpeg-rs]: https://github.com/imazen/mozjpeg-rs
[zenavif-parse]: https://github.com/imazen/zenavif-parse
[zenavif-serialize]: https://github.com/imazen/zenavif-serialize
[webpx]: https://github.com/imazen/webpx
[zenflate]: https://github.com/imazen/zenflate
[zenzop]: https://github.com/imazen/zenzop
[zenresize]: https://github.com/imazen/zenresize
[zenfilters]: https://github.com/imazen/zenfilters
[zenquant]: https://github.com/imazen/zenquant
[zenblend]: https://github.com/imazen/zenblend
[zensim]: https://github.com/imazen/zensim
[fast-ssim2]: https://github.com/imazen/fast-ssim2
[butteraugli]: https://github.com/imazen/butteraugli
[zenpixels]: https://github.com/imazen/zenpixels
[zenpixels-convert]: https://github.com/imazen/zenpixels
[linear-srgb]: https://github.com/imazen/linear-srgb
[garb]: https://github.com/imazen/garb
[zenpipe]: https://github.com/imazen/zenpipe
[zencodec]: https://github.com/imazen/zencodec
[zencodecs]: https://github.com/imazen/zencodecs
[zenlayout]: https://github.com/imazen/zenlayout
[zennode]: https://github.com/imazen/zennode
[Imageflow]: https://github.com/imazen/imageflow
[Imageflow Server]: https://github.com/imazen/imageflow-server
[imageflow-dotnet]: https://github.com/imazen/imageflow-dotnet
[imageflow-node]: https://github.com/imazen/imageflow-node
[imageflow-go]: https://github.com/imazen/imageflow-go
[ImageResizer]: https://github.com/imazen/resizer
[archmage]: https://github.com/imazen/archmage
[magetypes]: https://github.com/imazen/archmage
[enough]: https://github.com/imazen/enough
[whereat]: https://github.com/lilith/whereat
[zenbench]: https://github.com/imazen/zenbench
[cargo-copter]: https://github.com/imazen/cargo-copter
[resamplescope-rs]: https://github.com/imazen/resamplescope-rs
[codec-eval]: https://github.com/imazen/codec-eval
[codec-corpus]: https://github.com/imazen/codec-corpus
