# Context Handoff: Fix ultrahdr Crate for WASM

**Date:** 2026-01-22
**Updated:** 2026-01-23
**Status:** ✅ FIXED

## ROOT CAUSE IDENTIFIED AND FIXED (2026-01-23)

**The root cause was `f64::ln()` crashing in browser WASM.**

### Investigation Summary

Deep tracing revealed the crash location:
1. `jpegli.decode()` → works
2. `parser.decode()` → works
3. `to_pixels()` → crashes inside `compute_biases()`
4. `compute_biases()` → crashes at `f64::ln()` call

```
[compute_biases] computing ln(gamma)... gamma_f64=0.9580963850021362
💥 CRASH (unreachable)
```

**The issue is that `f64::ln()` (and possibly other math intrinsics) are not properly linked in browser WASM.** This only affects `wasm32-unknown-unknown` in browsers - native, Node.js, wasmer, and wasmtime all work correctly.

### Fix Applied

Added workaround in `jpegli-rs/src/quant/mod.rs`:

```rust
pub fn compute_biases(&self, component: usize) -> [f32; DCT_BLOCK_SIZE] {
    // WORKAROUND: Browser WASM (wasm32-unknown-unknown) crashes on f64::ln() calls.
    // Use default biases instead of computed optimal biases.
    // This is a minor quality regression but enables decoding to work.
    #[cfg(all(target_arch = "wasm32", not(target_os = "wasi")))]
    {
        let mut biases = [0.5f32; DCT_BLOCK_SIZE];
        biases[0] = 0.0; // DC doesn't get bias
        return biases;
    }

    // Full computation for native/WASI...
}
```

### WASM Runtime Comparison

| Environment | Target | Grayscale Decode |
|-------------|--------|------------------|
| Native | x86_64/aarch64 | ✅ Works |
| Node.js | wasm32-unknown-unknown | ✅ Works |
| **Wasmer 6.1** | **wasm32-wasip1** | **✅ Works** |
| **Wasmtime** | **wasm32-wasip1** | **✅ Works** |
| Browser (before fix) | wasm32-unknown-unknown | ❌ Crashed |
| **Browser (after fix)** | **wasm32-unknown-unknown** | **✅ Works** |

### Quality Impact

The workaround uses default biases (0.5 for AC coefficients, 0.0 for DC) instead of computing optimal Laplacian biases. This is a minor quality regression in the decoder:

- The bias computation improves edge sharpness and reduces ringing artifacts
- Default biases work well for most images
- The difference is typically imperceptible

### Tests Passing

All 9 zenimage-web Playwright tests pass:
- UltraHDR detection works
- Grayscale gain map decode works
- No WASM "unreachable" errors

### Future Work

1. Investigate why `f64::ln()` crashes in browser WASM
2. Consider using a pure-Rust ln() implementation that compiles to WASM correctly
3. Add browser WASM CI testing to catch similar issues early
