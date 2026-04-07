#![no_main]

use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    // JPEG segment parsing — exercises marker scanning, segment length parsing,
    // and APP segment extraction. Covers the container-level JPEG structure
    // without any pixel decoding.
    let _ = ultrahdr_rs::jpeg::parse_jpeg_segments(data);
    let _ = ultrahdr_rs::container::scan_segments(data);
});
