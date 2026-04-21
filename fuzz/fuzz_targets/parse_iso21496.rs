#![no_main]

use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    // ISO 21496-1 binary gain map metadata parsing — exercises the AVIF tmap
    // (version-byte prefixed), JXL `jhgm` (bare payload), and full JPEG APP2
    // body-with-URN wire format variants. Fast binary parser, very high exec/s.
    let _ = ultrahdr_core::metadata::iso_jpeg::parse_iso21496(
        data,
        ultrahdr_core::Iso21496Format::AvifTmap,
    );
    let _ = ultrahdr_core::metadata::iso_jpeg::parse_iso21496(
        data,
        ultrahdr_core::Iso21496Format::JxlJhgm,
    );
    let _ = ultrahdr_core::metadata::iso_jpeg::parse_iso21496(
        data,
        ultrahdr_core::Iso21496Format::JpegApp2BodyWithUrn,
    );
});
