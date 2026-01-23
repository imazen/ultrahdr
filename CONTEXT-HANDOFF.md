# Context Handoff: Streaming API Refactoring

## Current Task
Refactor the streaming API in `ultrahdr-core/src/gainmap/streaming.rs` for optimal naming, visibility, and documentation.

## What Was Done
1. Added `StreamingHdrReconstructor` - decode with full gainmap owned
2. Added `DualStreamingReconstructor` - decode with streamed gainmap (16-row ring buffer)
3. Added `StreamingGainMapComputer` - encode with synchronized HDR+SDR batches
4. Added `DualStreamingEncoder` - encode with independent HDR/SDR row streams

All committed:
- `c80dd91` feat(core): add streaming APIs for low-memory gain map processing
- `13a609a` feat(core): add dual-streaming APIs for parallel decode/encode

## What Needs To Be Done
User requested: "take a look at names, modules, interfaces, pub vs pub crate, and refactor interface optimally and with docs"

### Identified Issues

**1. Naming inconsistency:**
- `StreamingHdrReconstructor` vs `DualStreamingReconstructor` (both decode)
- `StreamingGainMapComputer` vs `DualStreamingEncoder` (Computer vs Encoder)
- `InputConfig` too generic (decode-only)
- `EncoderInputConfig` (encode-only)

**2. Proposed Renames:**
```
Decode (SDR + gainmap → HDR):
  StreamingHdrReconstructor → RowDecoder (takes full gainmap)
  DualStreamingReconstructor → StreamDecoder (streams both)

Encode (HDR + SDR → gainmap):
  StreamingGainMapComputer → RowEncoder (synchronized batches)
  DualStreamingEncoder → StreamEncoder (async inputs)

Config:
  InputConfig → DecodeInput
  EncoderInputConfig → EncodeInput
```

**3. Visibility fixes needed:**
- `GainMapRingBuffer` - should be private (line 469)
- `InputRingBuffer` - should be private (line 1480)
- `RowBuffer` - should be private (line 993)
- Helper functions already private ✓

**4. API simplification:**
- Too many constructor arguments (9+ params)
- Consider builder pattern: `RowDecoder::builder().width(w).height(h)...build()`
- Or config structs that group related params

**5. Module organization options:**
- Option A: Keep flat `streaming.rs` with renamed types
- Option B: Split into `streaming/decode.rs` and `streaming/encode.rs`

### Current Public API Surface
```rust
// streaming.rs exports:
pub struct StreamingHdrReconstructor     // line 58
pub struct InputConfig                   // line 83
pub struct DualStreamingReconstructor    // line 436
pub struct StreamingGainMapComputer      // line 894
pub struct EncoderInputConfig            // line 927
pub struct DualStreamingEncoder          // line 1435
```

### File Locations
- Main file: `/home/lilith/work/ultrahdr/ultrahdr-core/src/gainmap/streaming.rs` (1945 lines)
- Module: `/home/lilith/work/ultrahdr/ultrahdr-core/src/gainmap/mod.rs`
- Lib: `/home/lilith/work/ultrahdr/ultrahdr-core/src/lib.rs`

### Tests
All 48 tests passing. Tests are at the bottom of streaming.rs:
- `test_streaming_reconstructor_multi_row`
- `test_streaming_computer_multi_row`
- `test_y_only_mode`

## Git Status
```
On branch main
Your branch is ahead of 'origin/main' by 27 commits.
Working tree clean.
```

## Commands
```bash
cargo test --package ultrahdr-core
cargo clippy --package ultrahdr-core --all-targets -- -D warnings
```

## Notes
- MSRV is 1.75
- `div_ceil` is available (stabilized 1.73)
- No external dependents to worry about
