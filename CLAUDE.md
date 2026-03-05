# ultrahdr - Project Notes

## Feedback Log

- 2026-01-31: User requested porting C++ libultrahdr test logic. Implemented: ISO 21496-1 multi-channel roundtrip tests, transfer function reference values, gain map math reference tests, metadata validation tests, decoder parameter validation, README comparison table.
- 2026-03-05: Code review + 6 recommendations implemented (branch: refactor/review-fixes). Test coverage expanded 207 → 318 tests across all modules.

## Untracked Files

- `ultrahdr/examples/test_ultrahdr_parse.rs` — ad-hoc MPF debugging script, hardcodes absolute path to zenjpeg fixtures. Not suitable as a proper example without cleanup.

## Remaining Test Coverage Gaps

### Medium Priority Decoder Tests (not yet covered)
- [ ] JPEG with multiple XMP segments (which one wins?)
- [ ] JPEG with corrupted segment markers (invalid marker bytes)
