# ultrahdr - Project Notes

## Feedback Log

- 2026-01-31: User requested porting C++ libultrahdr test logic. Implemented: ISO 21496-1 multi-channel roundtrip tests, transfer function reference values, gain map math reference tests, metadata validation tests, decoder parameter validation, README comparison table.
- 2026-03-05: Code review + 6 recommendations implemented (branch: refactor/review-fixes). Test coverage expanded 207 → 318 tests across all modules.
- 2026-06-27: User requested "use magetypes instead of the wide crate, and make f16/half an opt-in compile feature." Dropped the unused `wide` dep from all manifests (the SIMD path was already on `magetypes`); made `half` an optional dep behind a new default-off `f16` feature gating `RgbaF16`/`RgbF16` I/O + `HdrOutputFormat::LinearF16`. Without `f16`, f16 input is rejected with `UnsupportedFormat` (no silent decode-to-black). Commits 708d68a (refactor) + 6e8bf60 (doc-link fix for the gated variant under `cargo doc -D warnings`).

## Known Bugs

- **Fuzz CI workflow is pre-existing-RED (since commit 602d215), independent of crate code.** Two sibling-version-skew causes, both in the standalone `fuzz/` workspace (which does NOT inherit the workspace-root `[patch.crates-io]`): (1) a sibling requires the unpublished `zenpixels-convert = "0.2.15"` — `fuzz/Cargo.toml` lacks the git-rev patch the root has, so resolution fails; (2) `fuzz_targets/tonemap.rs` calls `Bt2408Tonemapper::map_rgb` via `zentone::ToneMap`, but `ultrahdr-core` pins `zentone = "0.1.0"` (registry) while `fuzz`'s direct `zentone` path dep is the sibling `0.2.0` — two different `zentone` crates, so the trait method doesn't apply ("no method `map_rgb`"). Real fixes need publishing decisions (publish `zenpixels-convert 0.2.15`; reconcile `zentone` 0.1↔0.2), not a code change. The main `CI` workflow (all platforms, WASM, docs, MSRV, clippy, coverage) is green.

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
