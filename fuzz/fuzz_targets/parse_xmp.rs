#![no_main]

use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    // XMP metadata parsing — exercises XML string parsing for gain map metadata,
    // container directory items, and hdrgm namespace extraction.
    // High exec/s since it operates on strings, not binary data.
    if let Ok(xmp_str) = std::str::from_utf8(data) {
        let _ = ultrahdr_core::metadata::xmp::parse_xmp(xmp_str);
        let _ = ultrahdr_core::metadata::xmp::parse_xmp_full(xmp_str);
    }
});
