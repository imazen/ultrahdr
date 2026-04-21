#![no_main]

use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    // Multi-Picture Format parsing — exercises TIFF IFD parsing, endianness
    // detection, tag extraction, and MP entry decode.
    let _ = zenjpeg::container::mpf::parse_mpf(data);
    // Feed the same bytes as if the APP2 segment already had the MPF
    // identifier stripped, exercising the lower-level TIFF-directory parser.
    let _ = zenjpeg::container::mpf::parse_mpf_segment(data, 0);
});
