# Shared metadata integration

`ultrahdr-core` keeps its published Apple MakerNote and binary-plist types while
privately delegating parsing to zenmeta. `ultrahdr-rs` reuses the metadata-policy
assembly from [PR 35](https://github.com/imazen/ultrahdr/pull/35), revision
`1dee8471bb32d3189add54e63d8f45bfeebfb440`.

The additive `encode_ultrahdr_with_metadata_policy` function filters both JPEG
components through zenjpeg, checks display dependencies through zenmeta, and
regenerates gain-map XMP, ISO metadata and MPF. Its signature contains only existing
ultrahdr and zencodec types. It refuses unsupported gamut/direction contracts and
policies retaining source XMP. ICC is retained as a rendering dependency, not
certified free of private profile tags.

The coordinated dependencies are pinned to zenjpeg `db07f503` and zencodec/zenmeta
`c1c3292f` until release. No local path override is needed. No crate was published.

## Validation

On 2026-10-12, native and i686 production-encoder tests passed. The policy test
retains encoded scan bytes and verifies bit-exact reconstructed HDR against the
same assembler without source metadata. ExifTool independently resolves the
regenerated MPF directory to exactly the retained secondary JPEG and reads the
primary dimensions. Strict Clippy, doctests and public API snapshots passed.
Reproduce with `just metadata-policy-integration metadata-policy-targets`.

## Published API audit

The registry baseline for ultrahdr-rs is **0.3.5**, retrieved with `cargo read
ultrahdr-rs --path-only`; requesting 0.4.1 returns no matching version. The local
0.4.1 version and dated changelog section do not establish a published release.

`just metadata-policy-api` forces a patch comparison with 0.3.5, including the
zencodec feature: 218 checks passed, five failed, and 31 were skipped. Three
failures describe existing breaking removals in `2b7df54`: `container::MpfImageType`,
`MpfDirectory`, `MpfEntry`, `generate_mpf`, `assemble`, `extract_secondary_images`,
and `parse_mpf_segment`. Two failures report added `must_use` annotations on
`primary_bounds`, `scan_segments` and the five `AppSegment::is_*` predicates.
These precede the metadata integration.

Manual review also retains the previously queued `whereat::At<Error>` return
migration (`60b642f`) and feature-gated f16 output in the release audit; a semver
tool alone does not fully capture dependency-owned aliases. Snapshot refreshes
include already-existing target-quality and content-fit APIs from main. The only
new public function in this policy integration is the assembly function above.
