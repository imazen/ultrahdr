# ultrahdr [![CI](https://img.shields.io/github/actions/workflow/status/imazen/ultrahdr/ci.yml?style=flat-square&label=CI)](https://github.com/imazen/ultrahdr/actions/workflows/ci.yml) [![crates.io](https://img.shields.io/crates/v/ultrahdr-core?style=flat-square)](https://crates.io/crates/ultrahdr-core) [![lib.rs](https://img.shields.io/crates/v/ultrahdr-core?style=flat-square&label=lib.rs&color=blue)](https://lib.rs/crates/ultrahdr-core) [![docs.rs](https://img.shields.io/docsrs/ultrahdr-core?style=flat-square)](https://docs.rs/ultrahdr-core) [![codecov](https://img.shields.io/codecov/c/github/imazen/ultrahdr?style=flat-square)](https://codecov.io/gh/imazen/ultrahdr) [![MSRV](https://img.shields.io/badge/MSRV-1.92-blue?style=flat-square)](https://doc.rust-lang.org/cargo/reference/manifest.html#the-rust-version-field) [![license](https://img.shields.io/crates/l/ultrahdr-core?style=flat-square)](https://github.com/imazen/ultrahdr#license)

Pure-Rust encoder and decoder for [Ultra HDR](https://developer.android.com/media/platform/hdr-image-format) gain map JPEGs. An Ultra HDR file is a normal SDR JPEG with a second JPEG (the gain map) and a small block of metadata stapled onto it; HDR-capable readers reconstruct an HDR image, everything else sees the SDR base. This workspace ships the gain map math (`ultrahdr-core`) and a JPEG-bundled encoder/decoder (`ultrahdr-rs`) built on [zenjpeg](https://github.com/imazen/zenjpeg).

> **Pre-1.0, still being shaped.** Expect renames and re-exports between point releases — for example, `ultrahdr_core::metadata` is being retired in favor of `zenjpeg::container` and `zencodec::gainmap` ([#8](https://github.com/imazen/ultrahdr/issues/8)), and the streaming types are `#[doc(hidden)]` pending removal. Pin patch versions and read `CHANGELOG.md` before upgrading. Semver 0.x rules apply: breaking changes ride a minor bump.

## Acknowledgments

Built on the foundation of Google's [Ultra HDR Image Format specification](https://developer.android.com/media/platform/hdr-image-format) and the [libultrahdr](https://github.com/google/libultrahdr) reference implementation (BSD-3-Clause). The gain map math, ISO 21496-1 metadata layout, and most of the algorithm choices follow libultrahdr directly. This Rust port would not exist without that work.

The wire format is also defined by ISO/IEC 21496-1 (Gain map metadata for image conversion), which formalizes the on-disk gain map interchange.

## Crates

| Crate | Description |
|-------|-------------|
| [`ultrahdr-core`](ultrahdr-core/) | Core gain map math and metadata for Ultra HDR — no codec dependencies (`no_std + alloc`, WASM-compatible) |
| [`ultrahdr-rs`](ultrahdr-rs/) | Pure Rust Ultra HDR (JPEG with gain map) encoder/decoder, with [zenjpeg](https://github.com/imazen/zenjpeg) bundled |

`ultrahdr-core` carries the per-pixel kernels, the `GainMapMetadata` types, and the validators. `ultrahdr-rs` adds JPEG container assembly, MPF, XMP, ISO 21496-1 APP2 wiring, and the `Encoder` / `Decoder` types that hand you HDR or SDR pixels back. Pull `ultrahdr-core` directly if you already have your own JPEG codec, are decoding gain maps stored in AVIF/JXL/HEIF containers, or want to run on WASM.

## Getting started

Add `ultrahdr-rs` if you want a JPEG codec bundled, or `ultrahdr-core` if you bring your own:

```toml
[dependencies]
ultrahdr-rs = "0.3"        # full encoder + decoder, zenjpeg included
# or
ultrahdr-core = "0.5"      # math + metadata only, BYO codec
```

### Decode an Ultra HDR JPEG

```rust
use ultrahdr_rs::{Decoder, HdrOutputFormat};

fn decode(bytes: &[u8]) -> ultrahdr_rs::Result<()> {
    let decoder = Decoder::new(bytes)?;
    if !decoder.is_ultrahdr() {
        return Ok(()); // plain SDR JPEG
    }

    // 4× display boost = a typical "HDR display" target. 1.0 = SDR.
    let hdr = decoder.decode_hdr(4.0)?;        // PixelBuffer, RgbaF32, linear
    let sdr = decoder.decode_sdr()?;           // PixelBuffer, Rgba8, sRGB

    // Or hand off f16 directly to a compositor / GPU texture upload:
    let _hdr_f16 = decoder.decode_hdr_with_format(4.0, HdrOutputFormat::LinearF16)?;

    let _meta = decoder.metadata().cloned();   // GainMapMetadata, log2 domain
    let _ = (hdr, sdr);
    Ok(())
}
```

### Encode HDR + SDR into an Ultra HDR JPEG

```rust
use ultrahdr_rs::{
    ColorPrimaries, Encoder, PixelFormat, TransferFunction, new_pixel_buffer,
};

fn encode_hdr_plus_sdr() -> ultrahdr_rs::Result<Vec<u8>> {
    // HDR linear-light, BT.2020 primaries.
    let hdr = new_pixel_buffer(
        1920, 1080,
        PixelFormat::RgbaF32,
        ColorPrimaries::Bt2020,
        TransferFunction::Linear,
    )?;

    // SDR sRGB 8-bit, BT.709.
    let sdr = new_pixel_buffer(
        1920, 1080,
        PixelFormat::Rgba8,
        ColorPrimaries::Bt709,
        TransferFunction::Srgb,
    )?;

    let mut enc = Encoder::new();
    enc.set_hdr_image(hdr)
        .set_sdr_image(sdr)
        .set_quality(90, 85)            // base, gainmap
        .set_gainmap_scale(4)            // gain map at 1/4 resolution
        .set_target_display_peak(1000.0); // nits
    enc.encode()
}
```

### Encode HDR-only (auto-generate the SDR base)

If you don't supply an SDR image, the encoder tone-maps HDR → SDR using the curves in `ultrahdr_core::color::tonemap` (filmic by default). The same `Encoder` API otherwise:

```rust
use ultrahdr_rs::{
    ColorPrimaries, Encoder, PixelFormat, TransferFunction, new_pixel_buffer,
};

fn encode_hdr_only() -> ultrahdr_rs::Result<Vec<u8>> {
    let hdr = new_pixel_buffer(
        1920, 1080,
        PixelFormat::RgbaF32,
        ColorPrimaries::Bt2020,
        TransferFunction::Pq,         // PQ-encoded HDR, EOTF'd before tone mapping
    )?;
    let mut enc = Encoder::new();
    enc.set_hdr_image(hdr).set_quality(90, 85);
    enc.encode()
}
```

For finer control over the SDR generation, build the SDR yourself with `ultrahdr_core::color::tonemap` (BT.2408 / BT.2446 A/B/C / AgX / Reinhard family / Hable / ACES AP1 / darktable filmic spline) or `zentone::LumaGainMapSplitter`, then pass both to the encoder.

### Apply a gain map directly (math only)

If you've already parsed an Ultra HDR JPEG (or are pulling a gain map out of an AVIF / JXL container), `ultrahdr_core::gainmap::apply_gainmap` does the per-pixel reconstruction. No JPEG codec involved.

```rust
use ultrahdr_core::{
    ColorPrimaries, GainMap, GainMapMetadata, HdrOutputFormat, PixelFormat,
    TransferFunction, Unstoppable, new_pixel_buffer,
    gainmap::apply::apply_gainmap,
};

fn reconstruct(
    sdr_bytes: Vec<u8>,
    sdr_w: u32,
    sdr_h: u32,
    gainmap: GainMap,
    metadata: GainMapMetadata,
) -> ultrahdr_core::Result<()> {
    let sdr = ultrahdr_core::pixel_buffer_from_vec(
        sdr_bytes, sdr_w, sdr_h,
        PixelFormat::Rgba8,
        ColorPrimaries::Bt709,
        TransferFunction::Srgb,
    )?;
    let _ = new_pixel_buffer; // silence unused-import lint in this snippet

    let _hdr = apply_gainmap(
        &sdr,
        &gainmap,
        &metadata,
        4.0,                              // display boost
        HdrOutputFormat::LinearFloat,     // RgbaF32, linear
        Unstoppable,
    )?;
    Ok(())
}
```

## What's supported

Pixel inputs (gain map math kernels):

- **HDR**: `RgbaF32`, `RgbF32`, `RgbaF16`, `RgbF16` — linear, sRGB, PQ, or HLG transfer (the descriptor's `TransferFunction` is honored; non-linear inputs are EOTF-decoded first).
- **SDR**: `Rgba8`, `Rgb8`, `RgbaF32`, `RgbaF16`, `RgbF16`, `Gray8`.

Output formats from `apply_gainmap` (`HdrOutputFormat`):

- `LinearFloat` — linear f32 RGBA, 16 bytes/pixel. 1.0 = SDR white.
- `LinearF16` — linear f16 RGBA, 8 bytes/pixel. Mirrors libultrahdr's `UHDR_IMG_FMT_64bppRGBAHalfFloat`.
- `Srgb8` — sRGB 8-bit RGBA, clipped to SDR range.

10:10:10:2 packed PQ/HLG (`UHDR_IMG_FMT_32bppRGBA1010102` paired with `UHDR_CT_PQ` / `UHDR_CT_HLG`) and YCbCr P010 input are tracked in [#10](https://github.com/imazen/ultrahdr/issues/10).

Color spaces (`ColorPrimaries`):

- BT.709 / sRGB
- Display P3
- BT.2020 / BT.2100

Transfer functions (`TransferFunction`):

- sRGB (IEC 61966-2-1)
- PQ / ST.2084 (HDR10)
- HLG (BT.2100)
- Linear

Tone mapping curves (in `ultrahdr_core::color::tonemap`, gated by the `tonemap` feature):

- BT.2408 (PQ-domain, YRGB or MaxRGB)
- BT.2446 Methods A, B, C
- AgX (with `AgxLook` presets)
- Reinhard simple, Reinhard extended, Reinhard-Jodie, "tuned" Reinhard
- Hable filmic
- ACES AP1
- Filmic Narkowicz
- Darktable / Blender-style filmic spline (`CompiledFilmicSpline`)
- BT.2390 (with extended min-luminance form)
- An adaptive tone mapper (`AdaptiveTonemapper`) that fits an existing HDR/SDR pair and replays the curve

The streaming tonemapper and the row-streaming gain map encode/decode glue (`RowEncoder`, `RowDecoder`, `StreamEncoder`, `StreamDecoder`, `StreamingTonemapper`) are still present but `#[doc(hidden)]` and slated for removal in a future release. Drive `zentone::experimental::StreamingTonemapper` directly if you need streaming tone mapping.

Container support (`ultrahdr-rs`):

- Read and write XMP (Adobe `hdrgm:` namespace, GContainer directory)
- Read and write ISO 21496-1 APP2 binary (the version-only primary marker plus the full secondary payload)
- MPF (Multi-Picture Format) directory parsing and emission
- ICC profile injection for the primary JPEG's color space

## Comparison with libultrahdr

This is a partial port aimed at the encode/decode and gain-map-application paths a web/server use case actually exercises. Scope differences:

| | `ultrahdr` (this) | `libultrahdr` |
|---|---|---|
| HDR + SDR → Ultra HDR | yes | yes |
| HDR-only (auto SDR) | yes | yes |
| Multi-channel gain map | yes | yes |
| Ultra HDR → HDR reconstruct | yes | yes |
| Display boost parameter | yes | yes |
| Adaptive tone mapping (fit a curve from existing HDR/SDR) | yes | no |
| 10:10:10:2 / P010 pixel formats | tracked in [#10](https://github.com/imazen/ultrahdr/issues/10) | yes |
| In-place metadata edit API | no | yes |
| GPU acceleration | no | yes (OpenGL) |
| Pure Rust, no C deps | yes | C++ |
| WASM build target | yes (`ultrahdr-core`) | no |
| `no_std + alloc` build | yes (`ultrahdr-core`) | no |

Bit-exact `applyGain` and `applyGainCore` parity against libultrahdr and libavif goldens is enforced in `ultrahdr-core/tests/reference_parity.rs` (5 reference parity tests, see CHANGELOG entry for 0.5.0). One documented divergence: libultrahdr's `computeGain` clamps near-black pixels with hardcoded constants; this crate uses configurable `min_boost` / `max_boost` from `GainMapConfig` instead.

## Features

`ultrahdr-core`:

- `std` (default) — enables `std`-dependent transitive features in `enough`, `linear-srgb`, `archmage`, `magetypes`, `zentone`.
- `tonemap` (default) — gates the `zentone` re-exports at the crate root and the `color::tonemap` module. Decoder-only consumers can build with `--no-default-features --features std` to drop the transitive `zentone` dep.
- `simd` — enables explicit SIMD via `archmage` / `magetypes` (NEON, SSE4, AVX2, AVX-512, WASM SIMD128).
- `resize` — high-quality gain map downsampling via `zenresize`.

`ultrahdr-rs`:

- `simd` — forwards to `ultrahdr-core/simd`.
- `ffi-tests` — pulls in a libultrahdr Rust binding for FFI parity tests (CI only).
- `__pixel-parity` — runs pixel-parity tests against Google's `ultrahdr_app` subprocess (CI only, requires the binary on `PATH`).
- `zencodec` — opt-in `zencodec` trait integration for the unified codec dispatch in `zencodecs`.

## Compatibility

- MSRV 1.92 (workspace `rust-version`).
- Edition 2024.
- `#![forbid(unsafe_code)]` in both crates.
- CI tests: Linux x86_64, Linux aarch64, Windows x86_64, Windows aarch64 (`windows-11-arm`), macOS Intel, macOS aarch64, plus `i686-unknown-linux-gnu` (32-bit), WASM, and a Gain Map Interop workflow that runs pixel-parity tests against Google's `ultrahdr_app`.

## Links

- API docs: [`ultrahdr-core`](https://docs.rs/ultrahdr-core) · [`ultrahdr-rs`](https://docs.rs/ultrahdr-rs)
- [`CHANGELOG.md`](CHANGELOG.md)
- [Ultra HDR Image Format spec (Android)](https://developer.android.com/media/platform/hdr-image-format)
- [ISO/IEC 21496-1](https://www.iso.org/standard/86775.html) (Gain map metadata for image conversion)
- [`libultrahdr`](https://github.com/google/libultrahdr) (reference implementation)

## License

Apache-2.0. See [`LICENSE`](LICENSE).

## AI-Generated Code Notice

Parts of this library were developed with assistance from Claude (Anthropic). The implementation has been tested against reference Ultra HDR images, libultrahdr / libavif goldens, and Google's `ultrahdr_app` for pixel parity. Not all code has been manually reviewed — please review critical paths before production use.
