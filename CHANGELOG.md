# Changelog

All notable changes to this repository are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/); crates follow [Semantic
Versioning](https://semver.org/).

The workspace ships two publishable crates — `ultrahdr-core` and `ultrahdr-rs` —
with independent version numbers, plus shared workspace tooling. Each has its
own section below.

## ultrahdr-core

### [Unreleased]

#### Added
- **Content-fit gain-map grid** (`compute_gainmap_content_fit` + `CONTENT_FIT_MIN_SPAN_STOPS`; zensim campaign appendix AA "measure, don't configure"): the quantization grid is SELECTED from the MEASURED content gain range (one extra subsampled gain scan, bit-identical math shared with the ordinary encode) with the config `min_boost..=max_boost` as the outer policy bound. Spends the 8-bit code space on gains that exist — a ~2-stop content on the 10,000-nit default grid gets ~2.8× finer quantization — and is interop-safe by construction: the declared metadata is exactly the narrowed grid the bytes were quantized on (#33 invariant untouched). Uniform content is guarded to a 1/16-stop minimum span. `compute_gainmap`/`compute_gainmap_slice` behavior is byte-identical to before (they share a new `compute_gainmap_slice_observed` core). Tests: measured-not-configured declaration, precision-beats-config, uniform-content non-degeneracy, span-constant pin.

#### Fixed
- **Gain-map metadata now declares the range the bytes were actually quantized on (#33).** The encode kernel has always normalized gain-map bytes over the CONFIG boost grid (`GainMapConfig::min_boost ..= max_boost`), but `compute_gainmap`, `RowEncoder::finish`, and `StreamEncoder::finish` stored the content's observed (actual) gain range in the per-channel metadata `min`/`max` — the fields every conformant reader dequantizes on. Whenever the content range was narrower than the config range (almost always with `ultrahdr-rs`'s 10,000-nit default `target_display_peak`), every reader — including this workspace's own decoder — reconstructed under-boosted: a 2000-nit ramp decoded at ~732 nits. All three sites now declare the config grid via a shared helper (`metadata_for_config_grid`); the observed content max only widens `alternate_hdr_headroom`. Gain-map **bytes are unchanged** — only the declared mapping moved, so the single-pass row/streaming contracts and the `compute_gain_row` batch-parity guarantee are intact. Round-trip gates: 2000-nit ramp now decodes at 1976.3 nits (HDR-only and SDR+HDR paths), a 2000-nit specular highlight in a structured scene at 1858.9 nits.
- `StreamEncoder::finish` (deprecated, `#[doc(hidden)]`) previously hardcoded `base_hdr_headroom = 0.0` and derived `alternate_hdr_headroom` from the content max alone, ignoring the configured headrooms; it now shares the same config-driven metadata derivation as the other encode paths. (#33)

#### Interop note — files written before this fix
Every Ultra HDR file produced by the affected paths since the initial implementation (all ultrahdr-core releases 0.1.0–0.6.0, all ultrahdr-rs releases through 0.4.1, 2026-01-13 → 2026-08-06) whose content gain range was narrower than the configured range carries **mis-declared metadata and decodes under-boosted in every spec-compliant reader**. The written file does not record the grid the bytes were quantized on, so readers cannot repair it; **re-encoding from the original HDR source is the only remedy.** Files where the content range reached the configured range (e.g. `max_boost` set to the true content peak) are unaffected.

### [0.6.0] - 2026-07-24

First publish since 0.5.0. Drafted incrementally since 2026-06-23 (Cargo.toml
was pre-bumped to 0.6.0 then, `cargo publish` never ran); everything below
through this section ships together now, including the f16-gating and
`wide`-removal work that was still sitting under `[Unreleased]`.

#### QUEUED BREAKING CHANGES
<!-- Breaking changes queued for the next major (or minor for 0.x) release.
     Batch them here instead of shipping piecemeal. -->
- Move `gainmap::apply_simd` module to `pub(crate)` (6 items: 3 in submodule + 3 flat `gainmap::apply_gain_row_*` re-exports). Zero external consumers per the 2026-06-11 ablation (`docs/public-api/ABLATION-ultrahdr-core.md`); high-level `apply_gainmap` / `apply_gainmap_slice` cover every public use.
- Move `metadata::bplist` module to `pub(crate)` (10 items: `PlistValue` enum + `parse_bplist`). Apple plist parsing is an implementation detail of `metadata::apple::parse_apple_makernote`; zero external consumers per the same ablation.
- f16 pixel support is now gated behind the new default-off `f16` feature. In a default build, `HdrOutputFormat::LinearF16` is no longer compiled (it leaves the default public-API surface) and `RgbaF16`/`RgbF16` input is rejected with `Error::UnsupportedFormat` instead of decoded. Enable `features = ["f16"]` to restore the previous behavior. Queued because removing the `LinearF16` variant from the default surface is a semver break.

#### Added
- `f16` Cargo feature (default-off) — gates f16 (IEEE 754 half-precision) pixel I/O (`RgbaF16`/`RgbF16` input) and the `HdrOutputFormat::LinearF16` decode output via the `half` crate. `half` is now an **optional** dependency with `default-features = false`, keeping the `--no-default-features --features f16` build no_std-clean (`half`'s `std` folds into the crate's `std` feature). Without the feature, f16 input formats are rejected with a loud `Error::UnsupportedFormat` rather than silently decoded to black.

#### Removed
- Dropped the unused `wide` dependency. Explicit SIMD is provided by `magetypes` (`gainmap::apply_simd`, behind the `simd` feature); `wide` was declared in the manifest but never referenced in source.

#### Changed
- Bumped `zenpixels-convert` to `>=0.2.16, <0.3` (was pinned to an unpublished git rev) now that 0.2.16 is on crates.io.

### [0.6.0-draft] - 2026-06-23 (folded into 0.6.0 above)

#### Breaking changes
- `Result<T>` now carries a source location: `Result<T, whereat::At<Error>>` (was `Result<T, Error>`), for server-side error stack traces. Match the inner error with `e.error()` (borrow) or `e.decompose().0` (owned); read the capture site with `e.location()`. The bare `Error` type is unchanged, so `#[from] ultrahdr_core::Error` keeps working. `ultrahdr-rs` is instrumented in lockstep. (commit 60b642f)

#### Added
- `color::audited` — production-recommended HDR→SDR primitives behind the new `tonemap-bt2446a` Cargo feature (default-off). Re-exports `Bt2446A` (ITU-R BT.2446 Method A tone curve), `CllMeasure` + `LightLevelMethod` (`measure_max` peak measurement), `ContentLightLevel`, and `DiffuseWhite` from `zenpixels-convert::hdr`. Empirical basis: the 2026-06-22 audited HDR→SDR shootout (76 imazen-26 samples × 20 curves × 4 peak methods) — `Bt2446A` wins mean ΔE2000 by 2-5× over every channel-independent curve tested; `measure_max` wins 3 of 6 ranking criteria including `pct_above_de5` by 11 % over the closest alternative. See `zen/zentone/benchmarks/shootout_2026-06-22_findings_v2.md`. Default-off because `zenpixels-convert` brings `archmage` / `magetypes` / `garb` / `libm` beyond the crate's minimal-deps mandate. (commit 602d2152)
- Crate root re-exports for `Bt2446A`, `CllMeasure`, `LightLevelMethod`, `ContentLightLevel`, `DiffuseWhite` under the same feature gate, so consumers can `use ultrahdr_core::Bt2446A;` directly. (commit 602d2152)
- `metadata::apple` — Apple iOS MakerNote HDR headroom parser. Extracts `0x21 HDRHeadroom`, `0x30 HDRGain`, `0x0a HDRImageType` (per exiftool `Apple.pm`) from EXIF TIFF bytes, computes HDR headroom via the Apple stops formula, and maps to `GainMapMetadata` (`from_apple_headroom`). Validated against 49 real iPhone 8/13/16/17 HEIC captures (parsed values match exiftool, tol 1e-3). `no_std` + `alloc`, zero new deps. Public API: `parse_exif_for_apple_hdr`, `parse_apple_makernote`, `from_apple_headroom`, `AppleHdrInfo`. (commit 4ab18d5)
- `metadata::bplist` — minimal `bplist00` (Apple binary property list) reader for bplist-encoded MakerNote values (`RunTime`, AE state, …). Depth-bounded against cyclic refs. Public API: `parse_bplist`, `PlistValue`. (commit 4ab18d5)
- `full_reconstruction_boost(&GainMapMetadata) -> f32` — canonical f32 boost route for the `GainMapRender::ReconstructHdr { target_headroom: None }` semantics. Adapters must use this (not `2f32.powf(stops as f32)`) so reconstructions of the same parameters are bit-identical across codecs (heic#20). (commit 3ac20f9)

#### Fixed
- **bplist parser: bound attacker-controlled allocations.** The Apple binary plist reader (`metadata/bplist.rs`) called `Vec::with_capacity(count)` with an untrusted `count` for arrays/sets/dicts (no preceding length bound) and could overflow `count * 2` / `count * ref_size`, defeating the slice bound. Reservations are now capped by the input length and the multiplies are `checked_*`; the element loops already fail fast on the first out-of-range reference. No behavior change on well-formed input. (commit 2eb5329)
- Bound gain-map metadata magnitudes via `validate_gainmap_magnitude` and `validate_gainmap_metadata`; the gain-map LUT and decode pipeline now reject non-spec-compliant metadata whose `f64` values would cast to `f32` `±inf` / `NaN` and silently poison the HDR output buffer (#21). (commit cd8b785)
- Tag reconstruction output with the BT.2408 diffuse-white anchor so downstream tone-mapping math (e.g. `Bt2446A` + `measure_max` in `decode_full_sdr`) reads the canonical 203 cd/m² white point instead of guessing. (commit a5461cc)
- Fixed two broken intra-doc-links (`[color::audited]` in the crate root and `[0,1]` in `color::tonemap::decode_gain_value` — markdown intent, not a doc reference). (this prep)

#### Changed
- Exclude `tests/` and `benches/` from published package to slim the tarball; local `cargo test`/`cargo bench` are unaffected (target declarations kept intact).

### [0.5.0] - 2026-04-26

#### Breaking changes
- Removed `RawImage`, `RawImageRef`, `RawImageRefMut`. All kernels now take
  `zenpixels::PixelBuffer` (owning) and `zenpixels::PixelSlice` /
  `PixelSliceMut` (borrowed) directly. Replace:
  - `RawImage::new(w, h, fmt)` → `ultrahdr_core::new_pixel_buffer(w, h, fmt, primaries, transfer)`
  - `RawImage::from_data(w, h, fmt, primaries, transfer, data)` →
    `ultrahdr_core::pixel_buffer_from_vec(data, w, h, fmt, primaries, transfer)`
    (note: `data` is now the first argument)
  - Field access `img.data` / `img.stride` / `img.width` / `img.height` /
    `img.format` / `img.gamut` / `img.transfer` →
    `img.as_slice().as_strided_bytes()` / `img.stride()` / `img.width()` /
    `img.height()` / `img.descriptor().pixel_format()` /
    `img.descriptor().primaries` / `img.descriptor().transfer()`.
  - `img.clone()` on a `PixelBuffer` → `ultrahdr_core::clone_pixel_buffer(&img)`
    (zenpixels intentionally doesn't derive `Clone` on `PixelBuffer` to
    discourage silent large-pixel copies).
  - `RawImageRef<'_>` / `RawImageRefMut<'_>` → `PixelSlice<'_>` /
    `PixelSliceMut<'_>`.
- **Deletions queued** (marked `#[doc(hidden)]` in the current release,
  dropped in 0.5.0). None of these are reached from ultrahdr-rs's
  documented decode/encode paths. If you use any of these, switch now.
  - `color::transfer::{SrgbEotfLut, PqEotfLut, HlgOetfInvLut}` — dead LUT
    types. Callers needing per-byte linearization should call
    `linear_srgb::tf::*` directly (no setup cost).
  - `color::transfer::{apply_oetf, apply_eotf}` — generic dispatchers
    with silent sRGB fallback on unknown transfers (footgun).
  - `color::transfer::{pq_to_nits, nits_to_pq}` — trivial `* 10000` /
    `/ 10000`; inline at call sites.
  - `color::transfer::hlg_oetf` (forward direction). The only useful
    direction is `hlg_oetf_inv` (decode); use `linear_srgb::tf::linear_to_hlg`
    directly if you need the forward path.
  - `color::streaming_tonemap::*` — pass-through re-export of
    `zentone::experimental::{StreamingTonemapConfig, StreamingTonemapper}`.
    Import directly from `zentone` instead.
  - `color::tonemap` zentone re-exports: `Bt2408Tonemapper`, `EetfSpace`,
    `CompiledFilmicSpline`, `FilmicSplineConfig`, `AgxLook`, `ToneMap`,
    `ToneMapCurve`, and the named curve functions (`aces_ap1`,
    `agx_tonemap`, `bt2390_tonemap`, `bt2390_tonemap_ext`,
    `filmic_narkowicz`, `hable_filmic`, `reinhard_extended`,
    `reinhard_jodie`, `reinhard_simple`). Pure pass-throughs; import
    from `zentone` / `zentone::curves` directly.
  - `gainmap::streaming::*` — `RowEncoder`, `StreamEncoder`,
    `RowDecoder`, `StreamDecoder` (+ their configs and stats types).
    The per-row kernels these wrap (`sample_gainmap_row_lut`,
    `apply_gain_row_presampled`) stay; the state-machine/ring-buffer
    glue is Ultra-HDR-specific and doesn't generalize (no imageflow or
    zenjpeg consumer for it).

#### Changed
- Re-introduced the `tonemap` feature flag (default-on) gating the
  zentone re-exports at the crate root (`LumaToneMap`,
  `LumaGainMapSplitter`, etc.). Decoder-only consumers can build with
  `--no-default-features --features std` to drop the transitive zentone
  dependency. Replaces the prior `zentone` feature that was removed in
  the splitter consolidation; this version gates only the re-exports
  (no in-core splitter to fall back on when off).

#### Added
- `HdrOutputFormat::LinearF16` — linear f16 RGBA HDR output.
  Mirrors libultrahdr's `UHDR_IMG_FMT_64bppRGBAHalfFloat`. 8 bytes/pixel
  vs `LinearFloat`'s 16. Use for direct compositor / GPU-texture handoff.
- `RgbaF16` and `RgbF16` accepted as encode HDR input (`compute_gainmap`)
  and as decode SDR input (`apply_gainmap`). PQ / HLG / sRGB transfers
  on float inputs are now properly EOTF-decoded — previously RgbaF32
  silently assumed Linear regardless of `descriptor().transfer()`.
- Reference parity tests against libultrahdr and libavif goldens
  (`tests/reference_parity.rs`). 5 tests: bit-exact agreement on
  `applyGain` (105 rows from libultrahdr), `applyGainCore` (35 rows
  from libavif), `avifGetGainMapWeight` (6 rows), a 35-point cross-check
  proving libultrahdr/libavif/ours all agree on shared inputs, and a
  documented-divergence test for libultrahdr's `computeGain` near-black
  clamp (we don't replicate it; ours uses configurable `min_boost`/
  `max_boost` from `GainMapConfig`).
- `gainmap::apply::calculate_weight` is now `pub` (was `pub(crate)`).
  Mirrors `avifGetGainMapWeight`; useful for callers that want to
  precompute the apply weight without going through `apply_gainmap`.
- Shepard's Inverse Distance Weighting upsample for gain map apply
  with integer-scale precomputed weight LUT, plus shared weights across
  channels and row-hoisted constants in the float fallback. Bit-exact
  parity with libultrahdr's CPU `sampleMap` (see f425292).
- Re-export `PixelBuffer`, `PixelSlice`, `PixelSliceMut` at the crate root.
- `new_pixel_buffer(w, h, fmt, primaries, transfer)` — allocate a
  zero-filled `PixelBuffer` with ultrahdr-core's stricter
  dimension/format caps applied.
- `pixel_buffer_from_vec(data, w, h, fmt, primaries, transfer)` — wrap
  an existing `Vec<u8>` as a `PixelBuffer` with the same validators.
- `clone_pixel_buffer(&buf)` — deep-copy a `PixelBuffer` for callers that
  need an owned duplicate.
- `validate_ultrahdr_dimensions`, `validate_ultrahdr_image`,
  `validate_ultrahdr_slice`, `require_supported_format`,
  `descriptor_for` — the validator/descriptor helpers used internally
  at public entry points.
- New `apply_gainmap_slice(sdr: PixelSlice, ...)` — the borrowed
  counterpart to `apply_gainmap(&PixelBuffer, ...)`.
- New `compute_gainmap_slice(hdr: PixelSlice, sdr: PixelSlice, ...)` —
  the borrowed counterpart to `compute_gainmap(&PixelBuffer, ...)`.

#### Removed
- `RawImage`, `RawImageRef`, `RawImageRefMut` — eliminated in favor of
  zenpixels types. (~1,100 LOC removed from `types.rs`.)
- `gainmap::splitter::*` (`LumaGainMapSplitter`, `SplitConfig`,
  `SplitStats`, `LumaToneMap`, `LumaFn`, `HableFilmic`) — moved to
  `zentone`; ultrahdr-core re-exports them at the crate root for
  back-compat. Use `zentone::LumaGainMapSplitter` etc. directly.
- `gainmap::compute::compute_gainmap_tonemap` — niche HDR-only compute
  with explicit tone-curve injection. Callers should drive
  `zentone::LumaGainMapSplitter::new(curve, config).split_row(hdr,
  sdr_out, gain_out, channels, &mut stats)` per row, then call
  `compute_gainmap(hdr, sdr, …)` (or pack the gain map themselves) on
  the resulting SDR.

#### Changed
- The `zentone` dependency is gated behind the new `tonemap` feature
  (default-on). The luma gain map splitter API + curve catalog comes
  from zentone; ultrahdr-core re-exports `LumaGainMapSplitter`,
  `SplitConfig`, `SplitStats`, `LumaToneMap`, `LumaFn`, `HableFilmic`,
  `Bt2408Yrgb`, and `ExtendedReinhardLuma` at the crate root when the
  feature is on. Decoder-only consumers can build with
  `--no-default-features --features std` to drop the transitive
  zentone dep. (Replaces the old `zentone` Cargo feature; same shape,
  renamed and rewired.)
- `color::gamut` owns `apply_matrix` / `apply_matrix_row` /
  `soft_clip_gamut` directly rather than re-exporting from zentone.

#### Fixed
- `gainmap::compute` now imports `alloc::vec` so the
  `--no-default-features` (no_std + alloc) build of `cargo build -p
  ultrahdr-core` compiles. Was a regression from earlier refactor work.
- `gainmap::streaming::RowEncoder` no longer keeps an internal
  `Vec<Vec<u8>>` of per-row gainmap bytes alongside the rows it returns
  for streaming output. Replaced with a single preallocated
  `gainmap_data: Vec<u8>` written in place by `compute_gainmap_row`;
  saves one full per-row `Vec` allocation per gainmap row (~50% fewer
  allocs in the streaming encode path; ~3 MB churn per 4K multi-channel
  encode).

### [0.4.1] - 2026-04-10

#### Added
- ISO 21496-1 `common_denominator` parsing and `Iso21496Format` conversion helpers (2474e8e)
- `backward_direction` field on `GainMapMetadata`, parsed from ISO 21496-1 flags (dab5e27)
- Comprehensive ISO 21496-1 metadata format documentation (80f7970)
- `create_jpeg_iso_markers` high-level API for emitting both APP2 markers in one call (5f36b0d)
- SIMD tier consistency tests for gain map application (fc4f30e)
- aarch64 NEON SIMD implementation for gain map application (e85f7ce)
- Overflow guard on gain map math (269061c)

#### Changed
- Replaced platform-specific SIMD with portable magetypes generics (264ffff)
- Bumped `archmage` / `magetypes` 0.9.4 → 0.9.16 and `linear-srgb` 0.6.5 → 0.6.7 (e9c2931)
- Gated `proptest` behind `cfg(not(wasm32))` so wasm tests build cleanly (ba9310f)

#### Fixed
- Emit version-only ISO APP2 in the primary JPEG (4a14af1)
- Strip version byte from ISO 21496-1 JPEG APP2 and default wire format to `Both` (626063c)
- Use continued-fraction algorithm for ISO 21496-1 fraction encoding to match Chromium expectations (23f81c9)
- Resolved broken doc links in `ultrahdr-core` (22d851a)
- Collapsed nested `if let` in decoder XMP parsing to satisfy new clippy lints (e31032b)
- Prevent slice panic in MPF parsing when segment length < 2 (14b9423)
- Prevent panics from out-of-bounds JPEG segment offsets (aee9692)

### [0.3.0] - 2026-03-29

#### Changed
- 8 breaking API changes flagged by `cargo semver-checks`; see commit for details (8c07820)
- Switched gain map metadata types to the canonical zencodec versions, deleting ~1700 lines of duplication (0524d44)
- Introduced borrow-only `RawImageRef` / `RawImageRefMut` and made public enums `#[non_exhaustive]` (7acea4d)

### [0.2.0] - 2026-02-21

#### Changed
- Adapted to the new zenjpeg API, refreshed transitive deps (45e4e40)
- MSRV bumped to 1.92 (b22fdde)

### [0.1.1] - 2026-01-23

#### Added
- `set_existing_gainmap_jpeg()` for raw JPEG passthrough on the gain map side (eca5894)
- Dual streaming APIs for parallel decode/encode (13a609a)
- Low-memory streaming APIs for gain map processing (c80dd91)
- `AdaptiveTonemapper` for learned HDR-to-SDR curves (5e3eab8)
- `enough::Stop` cooperative cancellation hooks (e0d20b7)
- `GainMapLut` giving a 32-37% speedup on `apply_gainmap` (cf70399)
- `no_std` support for `ultrahdr-core` (db8594a)

### [0.1.0] - 2026-01-13

#### Added
- Initial pure-Rust Ultra HDR implementation (46ad2be).
- Split into `ultrahdr-core` (math + metadata, no codec dep) and `ultrahdr` crates (9f76f93).

## ultrahdr-rs

### [Unreleased]

#### Changed
- **`Encoder` selects its gain-map grid from MEASURED content by default** (appendix AA): `compute_new_gainmap` routes through `compute_gainmap_content_fit`, so `target_display_peak / 203` becomes the grid's UPPER BOUND and the actual quantization grid is the content's observed gain range within it (finer code-space use; declared metadata still exactly the quantization grid per #33). Opt out with the new `set_content_fit_grid(false)` to reproduce the configured-grid bytes (byte-stable corpora / external grid contracts). Gate: `encoder_default_grid_is_measured_not_configured` (measured top ≪ configured top, both grids reconstruct the 2000-nit ramp).

#### Fixed
- HDR-only (`set_hdr_image` + `encode()`) and SDR+HDR (`set_sdr_image` + `set_hdr_image`) encodes now produce files that reconstruct at the source luminance in conformant readers, via the ultrahdr-core gain-map metadata fix (#33 — see the ultrahdr-core Unreleased entry, including the interop note on re-encoding files written by earlier versions). With the 10,000-nit default `target_display_peak`, a 2000-nit ramp previously decoded at ~732 nits; it now decodes at 1976.3 nits. New round-trip gates: `tests/hdr_range_roundtrip.rs` (ramp + structured scene, both paths, peak + mid-tone + shadow assertions).

### [0.4.1] - 2026-07-24

Publishes as 0.4.1, not 0.4.0: the original 0.4.0 was published and yanked
same-day back on 2026-04-10 (semver-checks showed no real break, corrected
to 0.3.5 instead — see that entry below), and crates.io permanently burns
yanked version numbers. This release bundles that content plus everything
below through the `[0.4.0] - 2026-06-23` section (also never published) —
all of it ships together now.

#### QUEUED BREAKING CHANGES
<!-- Breaking changes queued for the next major (or minor for 0.x) release. -->
- Remove the `ffi-tests` Cargo feature stub (soft-removed in 0.4.0; was the gate for the now-deleted `libultrahdr_rs` C++ parity tests).
- f16 decode output requires the new default-off `f16` feature (forwards to `ultrahdr-core/f16`). In a default build, `decode_hdr_with_format(_, HdrOutputFormat::LinearF16)` and `RgbaF16`/`RgbF16` input are unavailable (the variant leaves the default `HdrOutputFormat` surface; f16 input is rejected with `Error::UnsupportedFormat`). Enable `features = ["f16"]` to restore.

#### Added
- `ResourceLimits` + `Decoder::new_with_limits` (#28): caller pixel/memory caps for untrusted-input decoding. All decode paths (base JPEG, gain-map JPEG, HDR output) validate JPEG header dimensions via `validate_ultrahdr_dimensions` + the caller caps *before* pixel allocation (header probe + zenjpeg `max_pixels`/`max_memory` + post-decode re-check); over-budget input returns a typed `Error::LimitExceeded`. `Decoder::new` behavior is unchanged (5d1122c).
- Cooperative cancellation (#28): `Decoder::decode_sdr_with_stop` / `decode_gainmap_with_stop` / `decode_hdr_with_stop` / `decode_hdr_with_format_and_stop` and `Encoder::encode_with_stop`; a cancelled `Stop` token surfaces as a typed `Error::Stopped` from every path (306aa5d).
- `f16` Cargo feature (default-off) — forwards to `ultrahdr-core/f16`, enabling `RgbaF16`/`RgbF16` input and the `HdrOutputFormat::LinearF16` decode output.

#### Fixed
- Decode output size arithmetic is now overflow-checked (u64 multiply + `usize::try_from`) and the RGBA expansion allocates via `try_reserve_exact` — a clean `Error::AllocationFailed` instead of an abort on OOM; `chunks` → `chunks_exact` removes trailing-partial-chunk panics (#28, 5d1122c).
- zenjpeg limit / cancellation / allocation errors keep their types (`LimitExceeded` / `Stopped` / `AllocationFailed`) instead of collapsing into `DecodeError`/`JpegEncode` strings (5d1122c, 306aa5d).
- Index-guard hardening in `jpeg::icc::extract_icc_profile` and `jpeg::markers::parse_jpeg_segments`: `.get()`-guarded reads so truncated segments or lying length fields can never index past the input (#28, 16818da).

#### Removed
- Dropped the unused `wide` dependency (it was declared but never referenced in source).

#### Changed
- `half` moved from a runtime dependency to a test-only dev-dependency (it was only used by the `__pixel-parity` test suite's f16 comparison against `ultrahdr_app -O 4`). Consumers no longer build `half` unless they opt into the `f16` feature, which pulls it transitively through `ultrahdr-core`.
- Bumped `zenpixels-convert` to `>=0.2.16, <0.3` and dropped the workspace's git-rev patch for it and `zenpixels` now that both publish at 0.2.16 on crates.io (see the Workspace section below).

### [0.4.0] - 2026-06-23 (drafted; never published, see 0.4.1 above)

#### Breaking changes
- `Result<T>` re-export now carries the `whereat::At<Error>` location annotation that `ultrahdr-core` adopted in lockstep. Match `e.error()` (borrow) or `e.into_inner()` (owned) on the returned error; bare `#[from] ultrahdr_rs::Error` still works for downstream error enums (the inner `Error` type is unchanged). (commit 60b642f)
- Dropped the `libultrahdr_rs` FFI binding dep (the Google libultrahdr Rust binding at `imazen/libultrahdr-rs`) and the `tests/parity_libultrahdr.rs` C++ parity harness. The pure-Rust impl stands alone — parity is validated through the corpus-based parity tests + the CI Gain Map Interop workflow (which runs Google's `ultrahdr_app` subprocess against our output). The `ffi-tests` Cargo feature is now an empty stub (soft-removal) so `cargo build --features ffi-tests` is a no-op for one release; the stub will be deleted in the next breaking release. (commit bc0e12f4 + this prep)

#### Added
- `Decoder::decode_full_sdr(target_primaries)` — one-call HDR→SDR decode for display paths. Reconstructs linear-light HDR via `apply_gainmap` at the metadata's full `alternate_hdr_headroom`, auto-measures the source peak via the audited-winner `CllMeasure::measure_max` (MaxRgb, BT.2408), applies the audited-winner `Bt2446A` tone curve, and writes 8-bit sRGB RGBA. Skips the public HDR-roundtrip API for callers who only need SDR. Empirical basis: the 2026-06-22 audited shootout (`zen/zentone/benchmarks/shootout_2026-06-22_findings_v2.md`) — `Bt2446A` wins mean ΔE2000 by 2-5× over every channel-independent curve; `measure_max` wins 3 of 6 ranking criteria including `pct_above_de5` by 11 %. Gated behind the new `tonemap-bt2446a` Cargo feature (forwards to `ultrahdr-core/tonemap-bt2446a`; default-off, pulls archmage / magetypes / garb / libm via `zenpixels-convert`). (commit 602d2152)
- `codec::ZenDecodeError` implements `From<whereat::At<ultrahdr_core::Error>>`, bridging the `whereat`-annotated `Result<T>` that core entry points return onto the zencodec trait's bare-error contract. Lets `Decode::probe` / `decode` use `?` directly without manually unwrapping the location annotation. (this prep)
- `UltraHdrDecoderConfig::probe` now attaches gain-map presence + metadata to the returned `ZenImageInfo` (`GainMapPresence::Available(GainMapInfo {…})` when Ultra HDR is detected, `Absent` for plain JPEGs). Callers can drive routing decisions before decode without a full decode pass. (this prep)

#### Fixed
- `decode.rs` RGB/grayscale→RGBA `Vec::with_capacity` computed `width * height * 4` in `u32` (wraps for large images / sooner on 32-bit); now computed in `usize`.
- Crate-level doc example used `use ultrahdr::…`; the crate is `ultrahdr_rs`.
- **`decode_gainmap` decodes RGB (multi-channel) gain maps** (#27), with the channel count driven by the ISO 21496-1 **metadata** (`is_single_channel`), not pixel inspection: single-channel maps keep the historical Gray decode (the exact luma plane — immune to the ±1 chroma noise a YCbCr-coded map picks up, which pixel-scanning would promote to spurious per-channel gain), falling back to RGB+BT.709-luma collapse when Gray output is unavailable; per-channel maps decode as 3-channel interleaved RGB; the full achromatic scan decides only when no metadata exists. Previously grayscale output was requested unconditionally — failing outright for some color encodings ("unsupported color conversion"; e.g. the libavif seine sample, whose hdrgm metadata carries distinct per-channel triples) and silently luma-averaging any RGB map that did decode. Per-channel maps are mainstream (Adobe exports, iOS 18). Regressions in `tests/rgb_gainmap.rs` cover the 3-channel and single-channel-metadata paths via fully synthetic round-trips.
- **`Decoder::new` no longer aborts when the MPF index fails to parse** (#26): MPF is one of several gain-map discovery routes, so a malformed or unsupported index (e.g. zenjpeg#148 — valid big-endian `MM` MPF read as "zero images") now degrades to the JPEG-boundary fallback instead of erroring out of detection that the XMP scan already established. Files like the committed 7.6 KB MPF-first fixture previously lost their HDR rendition silently in every consumer; regression-pinned in `tests/decoder_robustness.rs::mpf_first_sample_detected_as_ultrahdr` (asserts detection, exact hdrgm values, and gain-map decode).
- README's HTTP-error matching example was matching against bare `Error` directly; updated to call `e.error()` to reach the inner enum through the `whereat::At` wrapper.

#### Changed
- Exclude `tests/` from published package.
- `Encoder::set_hdr_image` / `set_sdr_image` now take `PixelBuffer` (from zenpixels) instead of the former `RawImage`. `Decoder::decode_sdr` / `decode_hdr` / `decode_hdr_with_format` return `PixelBuffer`. See the ultrahdr-core section for the call-site migration table. ultrahdr-rs re-exports `PixelBuffer`, `PixelSlice`, `PixelSliceMut`, `new_pixel_buffer`, `pixel_buffer_from_vec`, `clone_pixel_buffer`, `descriptor_for` at the crate root.

### [0.3.5] - 2026-04-10

#### Changed
- Yanked 0.4.0 — `cargo semver-checks` confirmed no breaking changes vs 0.3.4, so this is a patch release instead (b448736)
- Pinned `moxcms` to a concrete version for crates.io publish (d9f1305)

### [0.3.0] - 2026-03-29

#### Changed
- Added explicit `zenjpeg 0.7` version constraint (44081c0)
- Updated to `zencodec` 0.1.8 API (b2841c5)
- Required `zenpixels` 0.2.1 for gamut matrix, serde, ICC profiles, and bug fixes (126ba04)

### [0.2.0] - 2026-02-21

#### Changed
- Adapted to new zenjpeg API and refreshed deps (45e4e40)
- Updated README code examples for the 0.2.0 API (537e73b)

### [0.1.1] - 2026-01-23

#### Added
- Streaming encode/decode pipelines and tonemapper exposed through the top-level crate (1be054d, 8887946, 149aff2)
- `set_existing_gainmap()` for gain map reuse (4b25c6d)

#### Changed
- Renamed the `ultrahdr` crate to `ultrahdr-rs` to free the `ultrahdr` name on crates.io (c84d85d)

### [0.1.0] - 2026-01-13

#### Added
- Initial pure-Rust Ultra HDR encoder/decoder built on zenjpeg (46ad2be)
- WASM build and test infrastructure (a127e64, d6f9679)
- FFI parity test suite against libultrahdr (19b37e2)

## Workspace

### [Unreleased]

#### QUEUED BREAKING CHANGES
<!-- None queued. -->

#### Changed
- Overhauled the repo README to the zen-family conventions (inline shields.io badge row, `## Quick start`, absolute links, crosslink footer) and added a generated crates.io variant `README.crates.md` (CI-badge-only, regenerated from `README.md`); both `ultrahdr-core` and `ultrahdr-rs` now point `readme` at it so crates.io renders the trimmed version.

#### Fixed
- **Fuzz CI: the `tonemap` target builds again (#32).** `fuzz/Cargo.toml` carried a direct path dependency on the sibling `zentone` 0.2.0 plus a `[patch.crates-io]` for it, while `ultrahdr-core` requires registry `zentone ^0.1.0` — the patch could never apply, so two distinct `zentone` crates were linked and `zentone::ToneMap` did not provide `map_rgb` for `ultrahdr_core::color::tonemap::Bt2408Tonemapper` (`E0599`). The target now imports `ToneMap` through ultrahdr-core's re-export (`ultrahdr_core::color::tonemap::ToneMap`), and the fuzz workspace no longer declares or patches `zentone` at all. The other cause listed in #32 (unpublished `zenpixels-convert ^0.2.15`) was already resolved by the 0.2.16 publish on 2026-07-24; all 7 fuzz targets build locally with `cargo +nightly fuzz build`.

#### Removed
- Removed the unused `wide` SIMD crate from `[workspace.dependencies]` and both member manifests (`magetypes` is the SIMD path). `half` becomes opt-in via the per-crate `f16` feature — see the `ultrahdr-core` / `ultrahdr-rs` sections above.

### 2026-06 — Publish prep

- Versioned public-API surface snapshots: `docs/public-api/<crate>.txt` for `ultrahdr-core` and `ultrahdr-rs`, regenerated by `ultrahdr-core/tests/public_api_doc.rs` on every `cargo test` (`ZEN_API_DOC=check` verified by the new ci.yml `api-doc` job, `=off` elsewhere; `just api-doc` / `api-doc-check` recipes; `just fmt` regenerates).
- Credits / Acknowledgments section in README for Google's Ultra HDR spec and the libultrahdr reference implementation.
- `.gitignore` entry for the `.workongoing` agent coordination marker.
- Cleaned up `unused manifest key` warnings by removing redundant `path =` / `version =` overrides on workspace dependencies in `ultrahdr-core` and `ultrahdr-rs`.
- Removed the stale `.github/workflows/ffi-tests.yml` workflow — its `cargo test --features ffi-tests` step targeted a feature that no longer enables anything (the soft-removed `ffi-tests` stub) and the platform-matrix `base-tests` job duplicated `ci.yml`. The CI Gain Map Interop workflow remains the live integration test against Google's `ultrahdr_app`.

### 2026-04 — Tooling

- Nightly fuzz workflow: 60 s on push, 5 min nightly (aa66e35)
- Added 5 cargo-fuzz targets plus a custom JPEG dictionary (2f77e48)
- Committed fuzz infrastructure — seeds, dictionary, lockfile (9cd277b)
- Gitignored tooling noise (`.superwork/`, `.zenbench/`, etc.) and excluded it from published packages (b93ba96)

### 2026-03 — CI & Deps

- Tested the `zencodec` feature in CI alongside XMP / ISO 21496 parity tests (b71611a)
- Fixed `wasm-bench` build flags for edition 2024 and WASI cap-lints (fe21c23, 7bb494b, 642db55, f07c1b8, cd094fb, d24e3b3)
- Replaced local path dependencies with crates.io versions across the workspace (31e0bb1)

### 2026-01 — Initial Infrastructure

- Added cross-platform SIMD benchmarking via archmage / magetypes (c58f543)
- Added criterion benchmarks for core gain map operations (f1d2f39)
- Added WASI benchmark testing with wasmtime (1253735)
- Added WASM SIMD128 build support and `justfile` targets (413bede)
