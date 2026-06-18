# ABLATION-ultrahdr-core.md — conservative public-API ablation report

**Date:** 2026-06-11
**Snapshot commit:** 0908181c (main@origin before ablation change)
**Snapshot file:** `docs/public-api/ultrahdr-core.txt` (811 items default, 813 all-features)
**Grep template (run from `/home/lilith/work`, exclude target/.jj/ultrahdr, zen-arm-src):**
```
grep -r --include="*.rs" "<symbol>" /home/lilith/work/ \
  --exclude-dir="target" --exclude-dir=".jj" \
  --exclude-dir="zen-arm-src" --exclude-dir="ultrahdr"
```

---

## Summary

| Snapshot items | Flagged A | Flagged B | % flagged |
|----------------|-----------|-----------|-----------|
| 811 | 0 | 2 groups | ~0.5% |

**Conservative stance:** 809 of 811 items are KEEP. The two B-flagged groups are SIMD row-level primitives and a bplist parser that are internal implementation details with zero external consumers.

---

## Known consumers (evidence gathered this scan)

| Consumer | Items used |
|----------|-----------|
| `hdr-corpus-convert/src/main.rs` | `color::pq_oetf`, `gainmap::apply::apply_gainmap`, `ColorPrimaries`, `GainMap`, `HdrOutputFormat`, `PixelBuffer`, `PixelFormat`, `TransferFunction`, `Unstoppable`, `from_apple_headroom`, `parse_exif_for_apple_hdr`, `pixel_buffer_from_vec` |
| `zenmetrics/zenhdr-corpus/src/main.rs` | `gainmap::apply::apply_gainmap`, `ColorPrimaries`, `GainMap`, `HdrOutputFormat`, `Iso21496Format`, `PixelBuffer`, `PixelFormat`, `TransferFunction`, `parse_iso21496_fmt`, `pixel_buffer_from_vec` |
| `zenmetrics/zenmetrics-cli/src/hdr.rs` | `gainmap::apply::apply_gainmap`, `ColorPrimaries`, `GainMap`, `HdrOutputFormat`, `Iso21496Format`, `PixelFormat`, `TransferFunction`, `parse_iso21496_fmt`, `pixel_buffer_from_vec`, `Unstoppable`, `PixelBuffer` |
| `jxl-encoder/src/hdr/from_sdr.rs` | `GainMapParams` (= `GainMapMetadata`), `Iso21496Format`, `gainmap::GainMapConfig`, `ColorPrimaries`, `TransferFunction`, `GainMapEncodingFormat`, `descriptor_for`, `pixel_buffer_from_vec`, `serialize_iso21496_fmt` |
| `zenjpeg/zenjpeg/tests/` | `pixel_buffer_from_vec`, `gainmap::HdrOutputFormat`, `ColorPrimaries`, `PixelBuffer` |
| Today's Apple MakerNote work (fresh) | `metadata::apple::AppleHdrInfo`, `parse_apple_makernote`, `parse_exif_for_apple_hdr`, `from_apple_headroom`, `metadata::apple::tags::*` |

---

## Flagged items

### B — `pub(crate)` candidates (zero external consumers, internal SIMD plumbing)

**Group 1: `ultrahdr_core::gainmap::apply_simd` module — 3 functions**

```
pub fn apply_gain_row_presampled(&[[f32; 3]], &[[f32; 3]], [f32; 3], [f32; 3], &mut [[f32; 3]])
pub fn apply_gain_row_scalar(&[[f32; 3]], &[u8], &[f32; 256], &mut [[f32; 3]])
pub fn apply_gain_row_simd(&[[f32; 3]], &[u8], &[f32; 256], &mut [[f32; 3]])
```

These are the row-level SIMD dispatch functions that `apply_gainmap` / `apply_gainmap_slice` call internally. They take SIMD-register-width arrays, raw LUT pointers (`&[f32; 256]`), and pre-sampled gain rows — callers are expected to use the high-level `apply_gainmap` / `apply_gainmap_slice` APIs instead.

Consumer grep: zero hits outside the ultrahdr workspace (grep run 2026-06-11). Also re-exported as `ultrahdr_core::gainmap::apply_gain_row_*` at the module level — the re-exports are equally zero-consumer.

The module is also re-exported at `ultrahdr_core::gainmap::apply_gain_row_presampled`, `apply_gain_row_scalar`, `apply_gain_row_simd` — those three flat re-exports are included in this same B proposal.

**B proposal:** Make `pub mod apply_simd` into `pub(crate)`. Eliminates 3 submodule items + 3 flat re-exports at the gainmap level (6 total). The high-level `apply_gainmap` / `apply_gainmap_slice` / `GainMapLut` / `ShepardsLut` APIs remain fully public and are what external consumers need.

**Conservative note:** `apply_gain_row_simd` could theoretically serve a custom hot-loop consumer that wants to drive the SIMD kernel directly (e.g., zenjpeg's streaming gain-map path). However: (a) the live zenjpeg tests use only `apply_gainmap` at the pixel-buffer level; (b) the arm-src stale snapshot also uses `apply_gainmap`, not the row-level primitives. No current consumer warrants keeping these public. If a future consumer needs per-row dispatch it should be a new public API designed for that purpose.

---

**Group 2: `ultrahdr_core::metadata::bplist` module — 10 items**

```
pub enum PlistValue  (Array / Bool / Data / Date / Dict / Integer / Real / String / Uid)
pub fn parse_bplist(&[u8]) -> Option<PlistValue>
```

This is the Apple binary plist parser used internally by `parse_apple_makernote` to decode the `MakerNote` blob from iPhone HEIC EXIF. It is an implementation detail of the Apple HDR metadata path — the public entry points are `parse_apple_makernote(&[u8])` and `parse_exif_for_apple_hdr(&[u8])`, which return `Option<AppleHdrInfo>`. No caller outside ultrahdr-core needs to inspect `PlistValue` directly.

Consumer grep: zero hits outside the ultrahdr workspace (grep run 2026-06-11). The `parse_bplist` function is called only in `ultrahdr-core/src/metadata/apple.rs` where it is used to extract three specific IFD tag values; the `PlistValue` tree is never returned across an API boundary.

**B proposal:** Make `pub mod bplist` into `pub(crate)`. Eliminates 10 items. `parse_apple_makernote` / `parse_exif_for_apple_hdr` / `from_apple_headroom` / `AppleHdrInfo` / `metadata::apple::tags::*` all remain public — they are the intended published surface of the Apple HDR metadata work (freshly shipped 2026-06-09, KEEP unconditionally).

---

## Items reviewed and explicitly kept

**Core types and re-exports (bulk: ~760 items):** `GainMap`, `GainMapMetadata`, `GainMapConfig`, `GainMapEncodingFormat`, `HdrOutputFormat`, `Fraction`, `Error`, `Result`, `Iso21496Format`, type aliases (`ColorPrimaries`, `TransferFunction`, `PixelBuffer`, `PixelFormat`, `PixelSlice`, `PixelSliceMut`, `Stop`, `StopReason`, `Unstoppable`) — all confirmed consumed by multiple live callers. KEEP.

**`gainmap::apply` module** (`apply_gainmap`, `apply_gainmap_slice`, `calculate_weight`, `GainMapLut`, `ShepardsLut`, `HdrOutputFormat`): `apply_gainmap` confirmed consumed by hdr-corpus-convert, zenmetrics, zenjpeg. `GainMapLut` and `ShepardsLut` are LUT helpers for the same path. KEEP.

**`gainmap::compute` module** (`compute_gainmap`, `compute_gainmap_slice`, `compute_gain_row`, `GainMapConfig`): `compute_gainmap` confirmed consumed by jxl-encoder (hdr-gainmap feature). `compute_gain_row` is consumed by the stale arm-src zenjpeg encode path (likely still needed for the live zenjpeg gainmap encoder once it uses ultrahdr-core's encoder). KEEP — confirmed hit.

**`color` module** (all): `pq_oetf`, `pq_eotf`, `hlg_*`, `srgb_*`, `tonemap_*`, `convert_gamut`, `luma_coefficients`, `rgb_to_luminance`, `AdaptiveTonemapper`, `FitConfig`, `FitStats`, `GainMapInverter`, `LuminanceCurve`, `PerChannelLut`, `ProfileToneCurve`, `ToneMapConfig`, luma constants, `BT2100_LUMA`/`BT709_LUMA`/`P3_LUMA`, `scale_gainmap`, `crop_gainmap`, `soft_clip_gamut`, `tonemap_image_to_srgb8`: `pq_oetf` confirmed consumed by hdr-corpus-convert; `apply_gainmap` uses color math internally; `AdaptiveTonemapper` and `GainMapInverter` are deliberate API surface for tone-mapping tooling. `tonemap_image_to_srgb8` is a convenience for the live hdr-corpus-convert. KEEP.

**`limits` module** (all 10 constants): Caller-accessible bounds for validation. `MAX_XMP_LENGTH` referenced by zenjpeg container/xmp.rs comment. KEEP.

**`luminance` module** (`HLG_WHITE_NITS`, `PQ_PEAK_NITS`, `PQ_WHITE_NITS`, `SDR_WHITE_NITS`): hdr-corpus-convert defines its own `SDR_WHITE_NITS` matching this value; the constants serve callers who want reference nits. KEEP.

**`metadata::apple` module** (`AppleHdrInfo`, `parse_apple_makernote`, `parse_exif_for_apple_hdr`, `from_apple_headroom`, `tags::HDR_GAIN`, `tags::HDR_HEADROOM`, `tags::HDR_IMAGE_TYPE`): FRESH work shipped 2026-06-09. `parse_exif_for_apple_hdr` and `from_apple_headroom` confirmed consumed by hdr-corpus-convert. KEEP unconditionally.

**Root-level free functions:**
- `clone_pixel_buffer`: Zero consumer grep hits in live workspace; however it is a convenience companion to `new_pixel_buffer`/`pixel_buffer_from_vec` that is expected to be used as the API grows. **Conservative: KEEP** — the missing hit reflects thin coverage, not deliberate non-use. This is a simple wrapper around `PixelBuffer::clone`; it is not internal plumbing.
- `descriptor_for`: Confirmed consumed by jxl-encoder/src/hdr/from_sdr.rs (doc comment references it; code calls `ultrahdr_core::descriptor_for`). KEEP.
- `new_pixel_buffer`: Same API family as `pixel_buffer_from_vec`; `pixel_buffer_from_vec` confirmed consumed. KEEP.
- `pixel_buffer_from_vec`: Confirmed consumed by hdr-corpus-convert and zenmetrics. KEEP.
- `require_supported_format`: Zero external hits. **Conservative: KEEP** — validates `PixelFormat` against the codec's supported set; likely used in encoder tooling as APIs grow. Not internal plumbing.
- `validate_gainmap_magnitude`, `validate_gainmap_metadata`, `validate_ultrahdr_dimensions`, `validate_ultrahdr_image`, `validate_ultrahdr_slice`: Zero external hits. These are validation guard functions for the ultrahdr pipeline inputs. **Conservative: KEEP** — they are the clear public contract for callers assembling ultrahdr encode pipelines who want to validate inputs before encoding. Keeping them consistent with `GainMap::validate()` being public.

**`GainMap` struct** (with all pub fields `channels`, `data`, `height`, `width`): Returned by `Decoder::decode_gainmap()` and passed through `Encoder::set_existing_gainmap()`. Fields are legitimately readable by callers (e.g. zenmetrics). KEEP.

---

## Queued breaking changes (for next minor bump)

```
### QUEUED BREAKING CHANGES
- `ultrahdr_core::gainmap::apply_simd` module: make `pub(crate)` — 6 items (3 in submodule + 3 flat gainmap re-exports: `apply_gain_row_presampled`, `apply_gain_row_scalar`, `apply_gain_row_simd`)
- `ultrahdr_core::metadata::bplist` module: make `pub(crate)` — 10 items (`PlistValue` enum + `parse_bplist`)
```
