#![no_main]

use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    // Multi-Picture Format parsing — exercises TIFF IFD parsing, endianness
    // detection, tag extraction, and MP entry decode.
    let _ = ultrahdr_core::metadata::mpf::parse_mpf(data);
    let _ = ultrahdr_core::metadata::mpf::parse_mpf_entries(data);
});
