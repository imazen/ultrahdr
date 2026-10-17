//! Incremental (sans-I/O) decode through `dyn_incremental_decoder`: the
//! testkit's incremental contract on committed Ultra HDR JPEGs.
#![cfg(feature = "zencodec")]

use ultrahdr_rs::codec::UltraHdrDecoderConfig;
use zencodec::decode::DecoderConfig as _;

#[test]
fn incremental_contract_holds() {
    assert!(UltraHdrDecoderConfig::capabilities().incremental());
    for (name, bytes) in [
        (
            "test_ultrahdr.jpg",
            include_bytes!("../../test_ultrahdr.jpg").as_slice(),
        ),
        (
            "mpf_first_attribute_xmp.jpg",
            include_bytes!("images/mpf_first_attribute_xmp.jpg").as_slice(),
        ),
    ] {
        zencodec_testkit::check_incremental_capability_honesty(UltraHdrDecoderConfig, bytes)
            .unwrap_or_else(|e| panic!("{name}: {e:?}"));
    }
}
