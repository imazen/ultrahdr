# ABLATION-ultrahdr-rs.md — conservative public-API ablation report

**Date:** 2026-06-11
**Snapshot commit:** 0908181c (main@origin before ablation change)
**Snapshot file:** `docs/public-api/ultrahdr-rs.txt` (213 items default, 299 all-features)
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
| 213 (default) | 0 | 1 group | ~4% |
| +86 (all-features diff) | 0 | 0 | — |

**Conservative stance:** The `ultrahdr-rs` crate is the high-level consumer-facing codec crate. Its `Decoder`, `Encoder`, `encode_ultrahdr*` functions, and the zencodec adapter (`codec::*`) are all deliberate API. The one flagged group is the JPEG internals module — container-parse helpers that duplicate capabilities already in zenjpeg and were likely meant as a short-term bridge.

---

## Known consumers (evidence gathered this scan)

| Consumer | Items used |
|----------|-----------|
| `hdr-corpus-convert/src/main.rs` | `ultrahdr_rs::Decoder::new` |
| `zenmetrics/zenhdr-corpus/src/main.rs` | `ultrahdr_rs::Decoder` |
| `zenmetrics/zenmetrics-cli/src/hdr.rs` | `ultrahdr_rs::Decoder::new` |
| `zenjpeg/tests/bundled/ultrahdr_roundtrip.rs` | `ultrahdr_rs::Decoder::new`, `ultrahdr_rs::Decoder::*` methods |
| `zenmetrics/zenhdr-corpus/examples/make_distorted.rs` | `ultrahdr_rs::{Decoder, Encoder}` |
| zencodecs (via `codec::UltraHdrDecoderConfig`) | (zencodecs depends on ultrahdr-rs for its codec registry) |

---

## Flagged items

### B — `pub(crate)` candidate: `ultrahdr_rs::jpeg` module and `ultrahdr_rs::container` module

The `jpeg` module exports:
- `Marker` enum (JPEG SOI/EOI/SOF0/App0-2/DQT/DHT/SOS/EOI/DRI/COM — 12 variants)
- `JpegSegment` struct (pub fields `marker: u8`, `data: Vec<u8>`, `offset: usize`)
- `parse_jpeg_segments`, `reconstruct_jpeg`, `insert_segment_after_soi` free functions
- `icc` submodule: `ICC_IDENTIFIER`, `create_icc_markers`, `extract_icc_profile`, `get_icc_profile_for_gamut`
- Both items appear duplicated at module level (`ultrahdr_rs::jpeg::*`) and in the submodule (`ultrahdr_rs::jpeg::markers::*` and `ultrahdr_rs::jpeg::icc::*`)

The `container` module exports:
- `AppSegment` struct (pub fields `marker_num: u8`, `data: Vec<u8>`, `offset: usize`) + `is_exif/is_icc/is_jfif/is_mpf/is_xmp` classifiers
- `parse_mpf`, `primary_bounds`, `scan_segments` free functions
- Re-exports: `use MpImageType`, `use MpfEntry` (from zenjpeg)

Consumer grep evidence:
- `JpegSegment`, `parse_jpeg_segments`, `reconstruct_jpeg`, `insert_segment_after_soi`: **zero hits** outside ultrahdr workspace (grep run 2026-06-11). The zenjpeg arm-src hit (`JBRD / reconstruct_jpeg`) refers to `djxl --reconstruct_jpeg` CLI flag in a test, not this struct.
- `AppSegment`, `scan_segments`, `primary_bounds`: **zero hits** outside ultrahdr workspace. The parse_mpf hits in arm-src are for zenjpeg's own `parse_mpf`, not `ultrahdr_rs::container::parse_mpf`.
- `ICC_IDENTIFIER`, `create_icc_markers`, `get_icc_profile_for_gamut`, `extract_icc_profile`: The arm-src zenjpeg hits use `zenjpeg::color::icc::extract_icc_profile` (a zenjpeg function), not `ultrahdr_rs::jpeg::icc::extract_icc_profile`. Zero hits for the `ultrahdr_rs` version.
- `Marker` (the JPEG marker enum): zero external hits.

**Conservative assessment:** These modules look like implementation helpers that were made pub during initial development of the UltraHDR container parser, before zenjpeg's own container types existed. `AppSegment` is structurally the same as zenjpeg's `AppSegment`; `JpegSegment` is similarly parallel. No live external consumer uses any of these. The CLAUDE.md TODO section describes a planned generalization of the MPF/GContainer container in ultrahdr-core — that work should produce proper public container types, not these low-level JPEG scanner helpers.

**B proposal:** Make `pub mod jpeg` and `pub mod container` into `pub(crate)` within ultrahdr-rs. This removes ~40 items from the default surface. The `Decoder`, `Encoder`, `encode_ultrahdr*`, and all codec adapter items remain public.

**Conservative note:** The `jpeg::icc` functions (`get_icc_profile_for_gamut` in particular) could be useful to callers building custom ICC-tagged JPEG output. However, zenjpeg already provides equivalent ICC functionality and no current consumer uses the `ultrahdr_rs` versions. If a future consumer needs ICC profile helpers from ultrahdr-rs specifically, this is a deliberate additive re-publication — not a reason to keep currently-unused functions public now.

---

## Items reviewed and explicitly kept

**`Decoder<'a>`** (12 items): `new`, `is_ultrahdr`, `metadata`, `gainmap_jpeg`, `primary_jpeg`, `icc_profile`, `dimensions`, `decode_sdr`, `decode_gainmap`, `decode_hdr`, `decode_hdr_with_format`. Confirmed consumed by hdr-corpus-convert, zenmetrics, zenjpeg tests. KEEP.

**`Encoder`** (14 items): `new`, `default`, all `set_*` builder methods, `encode`, `encode_from_jpegs`, `clear_existing_gainmap`, `has_existing_gainmap`. Confirmed consumed by zenmetrics (make_distorted.rs). KEEP.

**`encode_ultrahdr`** and **`encode_ultrahdr_with_format`**: Entry-point encoding API. KEEP.

**Root-level re-exports** (the `pub use` block: `ColorPrimaries`, `Error`, `Fraction`, `GainMap`, `GainMapConfig`, `GainMapEncodingFormat`, `GainMapMetadata`, `HdrOutputFormat`, `Iso21496Format`, `PixelBuffer`, `PixelFormat`, `PixelSlice`, `PixelSliceMut`, `Result`, `Stop`, `StopReason`, `TransferFunction`, `Unstoppable`, `clone_pixel_buffer`, `color`, `descriptor_for`, `gainmap`, `limits`, `luminance`, `new_pixel_buffer`, `pixel_buffer_from_vec`): These re-export the same types as ultrahdr-core, providing a single import point for callers who depend on ultrahdr-rs rather than ultrahdr-core directly. Consistent with how the crate is designed as a convenience wrapper. KEEP.

**`codec::UltraHdrDecoderConfig`** (all-features, 4 items): Zencodec adapter entry point. KEEP.

**`codec::UltraHdrDecodeJob`** (all-features, 10 items): The per-decode-request object for the zencodec adapter. KEEP.

**`codec::UltraHdrDecoder<'a>`** (all-features, 1 item): The `Decode` impl. KEEP.

**`codec::ZenDecodeError`** (all-features, 4 variants + impls): Error type for the zencodec adapter. KEEP.

**`codec::UltraHdrExtras`** (all-features, 2 pub fields: `gainmap_jpeg`, `metadata`): Extension payload for zencodec decode output. KEEP.

---

## Queued breaking changes (for next minor bump)

```
### QUEUED BREAKING CHANGES
- `ultrahdr_rs::jpeg` module: make `pub(crate)` — ~25 items (`Marker`, `JpegSegment`, `parse_jpeg_segments`, `reconstruct_jpeg`, `insert_segment_after_soi`, `ICC_IDENTIFIER`, `create_icc_markers`, `extract_icc_profile`, `get_icc_profile_for_gamut`, plus their submodule duplicates in `jpeg::markers` and `jpeg::icc`)
- `ultrahdr_rs::container` module: make `pub(crate)` — ~15 items (`AppSegment` + methods, `parse_mpf`, `primary_bounds`, `scan_segments`, `MpImageType` re-export, `MpfEntry` re-export)
```
