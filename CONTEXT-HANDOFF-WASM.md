# Context Handoff: Fix ultrahdr Crate for WASM

**Date:** 2026-01-22
**Priority:** High - Blocks HDR editing in zenimage-web

## Problem

The ultrahdr crate causes "unreachable" WASM traps when calling `decode_hdr()` or `decode_hdr_with_format()`. This blocks the HDR editing workflow in zenimage-web.

### Error
```
RuntimeError: unreachable
```

This typically means:
1. A panic occurred in Rust code
2. An unimplemented WASM intrinsic was called
3. Memory access violation

### Where It Fails

In zenimage-web's `src/decode.rs`:
```rust
let hdr_image = decoder
    .decode_hdr_with_format(4.0, HdrOutputFormat::LinearFloat)
    .map_err(|e| format!("UltraHDR decode error: {e}"))?;
```

The error happens inside the ultrahdr crate before the Result is returned.

## Investigation Needed

### 1. Check jpegli-rs Dependency
ultrahdr uses `jpegli-rs` for JPEG decode:
```toml
# /home/lilith/work/ultrahdr/Cargo.toml
jpegli-rs = { path = "../jpegli-rs/jpegli-rs", features = ["decoder"] }
```

We already fixed one WASM issue in jpegli-rs (`profile.rs` using `std::time::Instant`), but there may be more.

**Action:** Grep jpegli-rs for other std-only features:
```bash
grep -r "std::" /home/lilith/work/jpegli-rs/jpegli-rs/src/ | grep -v "//\|test"
```

### 2. Check moxcms Dependency
ultrahdr uses moxcms 0.6 for color management:
```toml
moxcms = "0.6"
```

Current version is 0.8. The older version may have WASM issues.

**Action:** Update to latest moxcms and test.

### 3. Trace the Decode Path

The decode path is:
```
Decoder::decode_hdr_with_format()
  → apply_gainmap() in gainmap/apply.rs
    → get_sdr_linear() - calls jpegli decode
    → sample_gainmap() - pure math
    → write_output() - format conversion
```

**Action:** Add `web_sys::console::log_1()` calls at each step to find where it fails.

### 4. Check for Allocations

Large allocations can fail in WASM. The HDR output is `Rgba32F` which is 16 bytes per pixel.

For a 4000x3000 image: 4000 × 3000 × 16 = 192MB

**Action:** Test with a small image first (100x100).

## Proposed Fix Strategy

### Phase 1: Add WASM Test Target
```bash
# In ultrahdr directory
cargo test --target wasm32-unknown-unknown
```

Add to CI to catch WASM issues early.

### Phase 2: Create Minimal WASM Test
```rust
#[cfg(target_arch = "wasm32")]
#[wasm_bindgen_test]
fn test_decode_small_ultrahdr() {
    let test_data = include_bytes!("../test_data/small_ultrahdr.jpg");
    let decoder = Decoder::new(test_data).expect("create decoder");
    assert!(decoder.is_ultrahdr());

    // This is what fails
    let hdr = decoder.decode_hdr(4.0).expect("decode HDR");
    assert!(hdr.width > 0);
}
```

### Phase 3: Isolate the Failure
Add logging to narrow down:
```rust
// In decode.rs decode_hdr_with_format()
#[cfg(target_arch = "wasm32")]
web_sys::console::log_1(&"decode_hdr: starting".into());

let sdr = self.decode_sdr()?;  // Does this work?

#[cfg(target_arch = "wasm32")]
web_sys::console::log_1(&format!("decode_hdr: sdr decoded {}x{}", sdr.width, sdr.height).into());

let gainmap = self.decode_gainmap()?;  // Does this work?

#[cfg(target_arch = "wasm32")]
web_sys::console::log_1(&"decode_hdr: gainmap decoded".into());

// etc.
```

### Phase 4: Fix Dependencies
If jpegli-rs is the issue:
- Check for `std::time`, `std::fs`, `std::thread` usage
- Ensure `default-features = false` if needed
- Consider using zune-jpeg as fallback for WASM

If moxcms is the issue:
- Update to 0.8
- Check their WASM support

## Files to Examine

### ultrahdr crate
- `/home/lilith/work/ultrahdr/src/decode.rs` - Main decode logic
- `/home/lilith/work/ultrahdr/src/gainmap/apply.rs` - Gain map application
- `/home/lilith/work/ultrahdr/Cargo.toml` - Dependencies

### jpegli-rs crate
- `/home/lilith/work/jpegli-rs/jpegli-rs/src/decode/mod.rs` - Decoder entry
- `/home/lilith/work/jpegli-rs/jpegli-rs/src/profile.rs` - Already fixed Instant issue

### zenimage-web (consumer)
- `/home/lilith/work/zenimage/zenimage-web/src/decode.rs` - Where HDR decode is called
- Contains disabled `decode_ultrahdr()` function ready to re-enable

## Test Data Needed

Create a small UltraHDR test image:
```bash
# Use Android Camera app or libultrahdr to create
# Or extract from Google's test corpus
```

Place in `/home/lilith/work/ultrahdr/test_data/small_ultrahdr.jpg`

## Success Criteria

1. `cargo test --target wasm32-unknown-unknown` passes
2. `wasm-pack test --headless --chrome` passes
3. zenimage-web can load and export UltraHDR without errors
4. Round-trip test: load → edit → export → load again → verify

## Commands to Get Started

```bash
cd /home/lilith/work/ultrahdr

# Check current WASM compatibility
cargo build --target wasm32-unknown-unknown 2>&1 | head -50

# Add wasm-bindgen-test
cargo add wasm-bindgen-test --dev

# Run WASM tests (after adding test)
wasm-pack test --headless --chrome
```
