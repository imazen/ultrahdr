#![no_main]

use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    // Full Ultra HDR container decode — exercises JPEG SOI/EOI scanning,
    // APP segment extraction, XMP parsing, MPF directory parsing,
    // secondary image extraction, and gain map metadata discovery.
    // Does not decode JPEG pixels (that requires a JPEG codec).
    if let Ok(decoder) = ultrahdr_rs::Decoder::new(data) {
        let _ = decoder.is_ultrahdr();
        let _ = decoder.metadata();
        let _ = decoder.primary_jpeg();
        let _ = decoder.gainmap_jpeg();
        let _ = decoder.icc_profile();
    }
});
