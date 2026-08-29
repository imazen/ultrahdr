# ultrahdr - Project Notes

## Feedback Log

- 2026-01-31: User requested porting C++ libultrahdr test logic. Implemented: ISO 21496-1 multi-channel roundtrip tests, transfer function reference values, gain map math reference tests, metadata validation tests, decoder parameter validation, README comparison table.
- 2026-03-05: Code review + 6 recommendations implemented (branch: refactor/review-fixes). Test coverage expanded 207 → 318 tests across all modules.
- 2026-07-23: Issue #28 (untrusted-input hardening) fixed, additive-only: `ResourceLimits` + `Decoder::new_with_limits` (pixel cap default 500 MP clamped to core hard caps, optional memory cap; header-probe precheck + zenjpeg max_pixels/max_memory + post-decode re-check on base/gain-map/HDR paths); cancellation via `decode_sdr_with_stop`/`decode_gainmap_with_stop`/`decode_hdr_with_stop`/`decode_hdr_with_format_and_stop`/`Encoder::encode_with_stop`; overflow-checked `try_reserve_exact` allocations; typed error mapping (zenjpeg ImageTooLarge→LimitExceeded, Cancelled→Stopped — mappers return `At<Error>` because converting a built `At` to bare `Error` routes through core's `From<zenpixels::At<E>>` and stringifies into `InvalidPixelData`); index-guard hardening in jpeg/icc.rs + jpeg/markers.rs + encode.rs chunks_exact. Commits 5d1122cf, 306aa5dd, 16818da2, 9a86754c.
- 2026-06-27: User requested "use magetypes instead of the wide crate, and make f16/half an opt-in compile feature." Dropped the unused `wide` dep from all manifests (the SIMD path was already on `magetypes`); made `half` an optional dep behind a new default-off `f16` feature gating `RgbaF16`/`RgbF16` I/O + `HdrOutputFormat::LinearF16`. Without `f16`, f16 input is rejected with `UnsupportedFormat` (no silent decode-to-black). Commits 708d68a (refactor) + 6e8bf60 (doc-link fix for the gated variant under `cargo doc -D warnings`).
- 2026-08-06: Issue #33 fixed (a09478f0): gain-map bytes have always been quantized on the CONFIG boost grid, but batch/RowEncoder/StreamEncoder metadata declared the content ACTUAL range — every conformant reader under-reconstructed (2000-nit ramp → ~732 nits; now 1976.3). Fix = declare the config grid via shared `metadata_for_config_grid` (libultrahdr convention; bytes unchanged; quantize-on-actual rejected — impossible for the streaming/row contracts and would break the `compute_gain_row`==batch byte-parity that zenjpeg's fused path relies on). Round-trip gates in `ultrahdr-rs/tests/hdr_range_roundtrip.rs` + metadata==grid structural tests ×3 paths; CHANGELOG carries the interop/defect window (all releases ≤0.6.0/≤0.4.1 wrote mis-declared files; re-encode is the only remedy). Same-class bug in zenjpeg's `build_gainmap_metadata` → zenjpeg#193; zenmetrics workaround retirement → zenmetrics#40. Follow-up c0d15f44: main CI had been RED on all arm64 runners since bb27ac06 (2026-07-28) — `apply_gain_inner` went uncalled on aarch64 (dead_code under `-D warnings`); now cfg'd off aarch64. CI runner-set gap noted: no `i686-unknown-linux-gnu` (cross) job.

- 2026-08-26: `target_quality.rs` added to ultrahdr-rs (per-codec copy of zenjpeg's loop, injected scorer, 9/9 tests; commit b8c58d63). Same session: repaired local git-object corruption — 25 corrupt loose objects (all in pushed history, so plain `git fetch` would NOT resend them; `git fetch --refetch` is the fix), quarantined to ~/tmp/ultrahdr-rescue with the damaged .jj state; jj re-initialized colocated on the refetched store. If old op-log history is ever needed it lives in the rescue dir, not in .jj.

## Known Bugs

- **Fuzz CI (#32) — FIXED 2026-08-27.** Was red since 602d215 for two sibling-version-skew reasons in the standalone `fuzz/` workspace (which does NOT inherit the workspace-root `[patch.crates-io]`): (1) unpublished `zenpixels-convert ^0.2.15` — resolved by the 0.2.16 publish on 2026-07-24; (2) `fuzz_targets/tonemap.rs` imported `zentone::ToneMap` from a direct path dep on the sibling zentone 0.2.0 while `ultrahdr-core` requires registry `zentone ^0.1.0` — a `[patch]` can't satisfy `^0.1.0` with 0.2.0, so cargo silently ignored it, two `zentone` crates were linked, and `map_rgb` didn't apply (E0599). Fix: import `ToneMap` via `ultrahdr_core::color::tonemap::ToneMap` and drop fuzz's direct zentone dep + dead patch. **Lesson: a `[patch.crates-io]` whose path crate's version is outside the dependent's semver requirement is silently unused** — the workspace-root `zentone = { path = "../zentone" }` patch is in the same state (root `Cargo.lock` resolves registry 0.1.0 for ultrahdr-core and records path 0.2.0 under `[[patch.unused]]`); reconciling that is the still-open zentone 0.1↔0.2 publish question, separate from the fuzz build. The main `CI` workflow (all platforms, WASM, docs, MSRV, clippy, coverage) is green.

- **Fuzz CI — the SAME trap recurred 2026-08-29, different dependency; FIXED (`cbc14ecb`, Fuzz run 33238835462 green).** Red from `33230647323` onward, every target dying with `failed to select a version for the requirement zenanalyze = "^0.2.0"`. Cause: zenjpeg's 2026-08-28 migration made it take `zenanalyze` from the registry and supply the unpublished 0.2.x line through **zenjpeg's own** `[patch.crates-io]` — and a dependency's patch table is ignored once that dependency is consumed as a path dep; only the root workspace's applies. The root manifest was fixed in `127b058e`, but `fuzz/` is a standalone `[workspace]` and inherits nothing, so it needed the entry repeated (the lesson already recorded above, hit again from a new direction). Fixing that exposed a third layer: `fuzz/Cargo.lock` still pinned `archmage`/`magetypes` at 0.9.26 while current zenanalyze requires `^0.9.27` (bumped to 0.9.28; `ultrahdr-core`'s `^0.9.16` is still satisfied), and the lock also moved zenanalyze off a stale rev-pin (`13d40c3b`) onto the patch. **Both patch entries are removable once `zenanalyze 0.2.0` publishes — keep the root and `fuzz/` copies in step until then.** Detecting this class cheaply: `cargo metadata --format-version 1 --locked` per workspace is non-mutating and distinguishes a genuine `failed to select a version` from a merely stale lock; run it in `fuzz/` too, not just the root.

## Untracked Files

- `ultrahdr/examples/test_ultrahdr_parse.rs` — ad-hoc MPF debugging script, hardcodes absolute path to zenjpeg fixtures. Not suitable as a proper example without cleanup.

## TODO: Generalize Container for Depth Maps & Multi-Item JPEG

ultrahdr-core owns the container primitives (MPF, GContainer XMP, Extended XMP) that are
shared across Ultra HDR gain maps, depth maps, and other multi-image JPEG use cases. The
current code is hardcoded for exactly 2 items (Primary + GainMap). It needs to become an
N-item container so that zenjpeg (which already has depth extraction in `decode/depth.rs`
and `MpfImageType::Disparity` in `encode/extras.rs`) can consume generalized types instead
of reimplementing container parsing.

### Background: How Depth Maps Are Stored in JPEG

Three formats exist in the wild, all using the same container primitives ultrahdr-core
already parses:

- **Apple MPF (iPhone portrait)**: Secondary JPEG after primary EOI, referenced by APP2
  MPF directory. Depth is a disparity map (normalized, grayscale JPEG). Same MPF mechanism
  as Ultra HDR gain maps — only `MpImageType` differs.

- **Android GDepth XMP (pre-2019)**: Base64-encoded grayscale JPEG/PNG in `GDepth:Data`
  XMP attribute, namespace `http://ns.google.com/photos/1.0/depthmap/`. Large maps spill
  into Extended XMP (same chunking ultrahdr-core already handles). Includes `GDepth:Near`,
  `GDepth:Far`, `GDepth:Format` (RangeLinear/RangeInverse), optional confidence map.

- **Android Dynamic Depth (Android Q+)**: Uses GContainer — the same `Container:Directory`
  / `Item:Semantic` / `Item:Mime` / `Item:Length` XMP structure as Ultra HDR. Depth and
  confidence maps appended as raw JPEG/PNG trailers after EOI (not base64). Just adds
  `Item:Semantic="DepthMap"` and `"ConfidenceMap"` alongside `"GainMap"`.

### Phase 1: Generalize GContainer XMP (ultrahdr-core)

`xmp.rs` currently hardcodes a 2-item directory template. Generalize to N items.

- [ ] Add `ItemSemantic` enum: `Primary`, `GainMap`, `DepthMap`, `ConfidenceMap` (extensible
  with `Other(String)` for forward compat)
- [ ] Add `ContainerItem` struct: `{ semantic: ItemSemantic, mime: String, length: Option<usize> }`
- [ ] Refactor `generate_xmp()` to accept `&[ContainerItem]` instead of a bare
  `gainmap_length: usize`. Keep a convenience wrapper for the Ultra HDR 2-item case so
  existing callers don't break.
- [ ] Refactor `parse_xmp()` to return `Vec<ContainerItem>` alongside `GainMapMetadata`.
  Currently it only extracts `hdrgm:*` attributes and `Item:Length` for the gain map —
  it should parse ALL items in the `Container:Directory` `rdf:Seq`.

### Phase 2: Generalize MPF (ultrahdr-core)

`mpf.rs` returns bare `Vec<(usize, usize)>` byte ranges with no semantic info.

- [ ] Add `MpfEntry` struct: `{ image_type: MpImageType, offset: usize, size: usize }`
  where `MpImageType` covers BaselinePrimary, DependentChild, LargeThumbnail, Disparity,
  MultiAngle, etc. (zenjpeg's `MpfImageType` enum already has these — reconcile or share)
- [ ] Refactor `parse_mpf()` to return `Vec<MpfEntry>` instead of `Vec<(usize, usize)>`
- [ ] Refactor `create_mpf_header()` to accept `&[MpfEntry]` (or at minimum a
  `&[(MpImageType, usize)]` of type+length pairs) instead of hardcoded primary+gainmap

### Phase 3: GDepth XMP Namespace (ultrahdr-core, optional)

Parsing GDepth attributes is currently in zenjpeg's `decode/depth.rs`. Decide whether to
keep it there (JPEG-specific) or move the XMP attribute extraction here (reusable).

- [ ] Evaluate: does any non-JPEG format embed GDepth XMP? If not, leave in zenjpeg.
- [ ] If moving: add `GDepthMetadata` struct and `parse_gdepth_xmp()` to a new
  `metadata/gdepth.rs` module. Keep it behind a feature flag if it adds weight.

### Non-Goals

- **Depth rendering** (blur, 3D, disparity→meters): out of scope, belongs in consumer code
- **HEIF/AVIF depth**: different container (ISOBMFF auxiliary images), handled by
  heic-decoder-rs and zenavif-parse respectively
- **Depth map codec**: the depth image is just a grayscale JPEG or PNG — zenjpeg/zenpng
  decode it, ultrahdr-core doesn't touch pixel data

### Cross-Crate Coordination

| Crate | Role | Existing Code |
|-------|------|---------------|
| **ultrahdr-core** | Container primitives (MPF, GContainer XMP) | `metadata/mpf.rs`, `metadata/xmp.rs` |
| **zenjpeg** | JPEG-specific depth extraction + encode | `decode/depth.rs`, `encode/extras.rs` |
| **zencodecs** | Format-agnostic `DecodedDepthMap` | `depthmap.rs` |
| **heic-decoder-rs** | HEIF auxiliary depth (`auxid:2`) | `auxiliary.rs` |

After phases 1-2, zenjpeg's depth code should consume `ContainerItem` / `MpfEntry` from
ultrahdr-core instead of doing its own parallel container parsing.

## Remaining Test Coverage Gaps

### Medium Priority Decoder Tests (not yet covered)
- [ ] JPEG with multiple XMP segments (which one wins?)
- [ ] JPEG with corrupted segment markers (invalid marker bytes)
