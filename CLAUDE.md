# ultrahdr - Project Notes

## Feedback Log

- 2026-01-31: User requested porting C++ libultrahdr test logic. Implemented: ISO 21496-1 multi-channel roundtrip tests, transfer function reference values, gain map math reference tests, metadata validation tests, decoder parameter validation, README comparison table.

## TODO: Regression Tests for Non-UltraHDR / Malformed Input

The decoder gracefully handles non-UltraHDR files in code (returns `Ok()` with `is_ultrahdr() == false`, fallback chain: XMP → MPF → JPEG boundary scan), but almost none of this behavior is pinned by tests. Currently only two cases are tested: garbage bytes and a minimal `FF D8 FF D9` JPEG.

### High Priority

- [ ] JPEG with XMP but no `hdrgm` namespace (e.g. camera JPEG with EXIF/XMP) — should not set `is_ultrahdr`
- [ ] JPEG with MPF segment but only 1 image (stereo camera, not UltraHDR) — should degrade gracefully
- [ ] JPEG with corrupted/truncated MPF TIFF structure — should return meaningful error or degrade
- [ ] JPEG with malformed XMP XML in hdrgm namespace — should return `XmpParse` error
- [ ] JPEG with XMP claiming gain map exists but no MPF or secondary image — should handle inconsistency
- [ ] Truncated file (cuts off mid-segment) — should not panic
- [ ] Confirm gain map access returns `None` on plain JPEG (not just `is_ultrahdr() == false`)

### Medium Priority

- [ ] Dimension mismatch between primary and gain map images
- [ ] Missing required metadata fields in XMP/ISO 21496-1 (defaults vs errors)
- [ ] Metadata with NaN/infinity/out-of-range values in hdrgm attributes
- [ ] JPEG with corrupted segment markers (invalid marker bytes)
- [ ] JPEG with multiple XMP segments (which one wins?)

## Untracked Files

- `ultrahdr/examples/test_ultrahdr_parse.rs` — ad-hoc MPF debugging script, hardcodes absolute path to zenjpeg fixtures. Not suitable as a proper example without cleanup.
