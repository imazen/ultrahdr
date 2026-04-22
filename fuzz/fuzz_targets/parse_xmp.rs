#![no_main]

use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    // XMP metadata parsing — exercises XML string parsing for the `hdrgm:`
    // namespace + generic GContainer directory items. Both entry points
    // live in zenjpeg::container::xmp now; ultrahdr-core's duplicate was
    // deleted in issue #8.
    if let Ok(xmp_str) = std::str::from_utf8(data) {
        let _ = zenjpeg::container::xmp::parse_xmp(xmp_str);
        let _ = zenjpeg::container::xmp::parse_xmp_full(xmp_str);
    }
});
