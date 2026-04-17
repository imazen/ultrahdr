//! Pixel-and-byte-level parity against Google's `ultrahdr_app` binary.
//!
//! # How this file is gated
//!
//! This suite is compiled **only** when the `__pixel-parity` feature is set.
//! Per CLAUDE.md: "the skip decision must be made by the *caller*, not
//! buried inside the test body" — the caller (CI's Gain Map Interop
//! workflow, or a developer's `just` target) enables the feature when the
//! `ultrahdr_app` binary is available. Under that feature the tests MUST
//! complete; a missing binary is a hard failure, not a silent pass.
//!
//! To run locally:
//! ```sh
//! cargo test -p ultrahdr-rs --features __pixel-parity \
//!     --test libultrahdr_pixel_parity
//! ```
//!
//! Set `ULTRAHDR_APP=/path/to/ultrahdr_app` to override PATH discovery.
//!
//! # What this covers (from the correctness-gap audit)
//!
//! | Item | Coverage |
//! |------|----------|
//! | #13 MPF byte offsets       | `mpf_offsets_match_libultrahdr` |
//! | #14 Probe field-by-field   | `probe_metadata_matches_libultrahdr` |
//! | #15 SDR bit parity         | `sdr_decode_matches_libultrahdr` |
//! | #16 HDR similarity         | `hdr_decode_matches_libultrahdr` |
//!
//! # What a "match" means per test
//!
//! - **Probe fields** — we use our Rust decoder's `metadata()` and compare
//!   field-by-field to the values `ultrahdr_app -P` prints. Numeric fields
//!   use a small tolerance (1e-3 linear) to absorb libultrahdr's
//!   continued-fraction rounding on both sides.
//! - **SDR pixel bytes** — libultrahdr writes the decoded primary to a
//!   raw `.rgb` file (`-o 3 -O 3 -z out.rgb`). We decode via `decode_sdr()`
//!   and compare byte-for-byte. Both sides use libjpeg-turbo-derived IDCT,
//!   so we assert max delta ≤ 3 and MAE < 0.2 (per ISO/IEC 10918-2
//!   conformance ±1 per channel).
//! - **HDR pixel similarity** — libultrahdr outputs linear f32 RGBA via
//!   `-o 0 -O 4`. We output the same via `decode_hdr_with_format(..,
//!   LinearFloat)`. Mean absolute error per channel is asserted < 0.05.

#![cfg(all(not(target_arch = "wasm32"), feature = "__pixel-parity"))]

use codec_corpus::Corpus;
use std::path::PathBuf;
use std::process::Command;
use ultrahdr_rs::Decoder;

// ---------------------------------------------------------------------------
// Binary discovery (hard-fails when missing — the feature gate is the switch)
// ---------------------------------------------------------------------------

fn ultrahdr_app() -> PathBuf {
    if let Ok(path) = std::env::var("ULTRAHDR_APP") {
        let p = PathBuf::from(&path);
        assert!(
            p.is_file(),
            "ULTRAHDR_APP={path} does not point at an existing file"
        );
        return p;
    }
    if let Ok(p) = which::which("ultrahdr_app") {
        return p;
    }
    for guess in [
        "/usr/local/bin/ultrahdr_app",
        "/usr/bin/ultrahdr_app",
        "/opt/homebrew/bin/ultrahdr_app",
    ] {
        let p = PathBuf::from(guess);
        if p.is_file() {
            return p;
        }
    }
    panic!(
        "ultrahdr_app not found. The `__pixel-parity` feature is enabled — \
         either put `ultrahdr_app` on PATH, set ULTRAHDR_APP, or disable the \
         feature. CI's Gain Map Interop workflow installs it automatically."
    )
}

fn corpus() -> Corpus {
    Corpus::new().expect(
        "codec-corpus initialisation failed — pixel-parity tests require \
         network access on first run to fetch the imazen/codec-corpus ZIP. \
         Subsequent runs use the local cache.",
    )
}

fn first_pixel_sample(corpus: &Corpus) -> PathBuf {
    let dir = corpus
        .get("ultrahdr-conformance/valid/jpeg/pixel-ultrahdr")
        .expect("ultrahdr-conformance/valid/jpeg/pixel-ultrahdr must exist in codec-corpus");
    std::fs::read_dir(&dir)
        .expect("read pixel-ultrahdr dir")
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .find(|p| {
            p.extension()
                .and_then(|s| s.to_str())
                .is_some_and(|s| s.eq_ignore_ascii_case("jpg"))
        })
        .expect("at least one .jpg in pixel-ultrahdr")
}

// ---------------------------------------------------------------------------
// #14 — Probe metadata, field-by-field
// ---------------------------------------------------------------------------

#[test]
fn probe_metadata_matches_libultrahdr() {
    let bin = ultrahdr_app();
    let c = corpus();
    let fixture = first_pixel_sample(&c);

    let out = Command::new(&bin)
        .args(["-m", "1", "-P", "-j"])
        .arg(&fixture)
        .output()
        .expect("invoke ultrahdr_app");
    assert!(
        out.status.success(),
        "ultrahdr_app probe failed: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    let probe = String::from_utf8_lossy(&out.stdout);

    fn field(probe: &str, key: &str) -> Option<f64> {
        for line in probe.lines() {
            let line = line.trim_start();
            if let Some(rest) = line.strip_prefix(&format!("--{key}")) {
                return rest.trim().parse::<f64>().ok();
            }
        }
        None
    }

    let lib_max = field(&probe, "maxContentBoost").expect("libultrahdr printed maxContentBoost");
    let lib_min = field(&probe, "minContentBoost").expect("libultrahdr printed minContentBoost");
    let lib_gamma = field(&probe, "gamma").expect("libultrahdr printed gamma");
    let lib_sdr_offset = field(&probe, "offsetSdr").expect("libultrahdr printed offsetSdr");
    let lib_hdr_offset = field(&probe, "offsetHdr").expect("libultrahdr printed offsetHdr");
    let lib_cap_max = field(&probe, "hdrCapacityMax").expect("libultrahdr printed hdrCapacityMax");
    let lib_cap_min = field(&probe, "hdrCapacityMin").expect("libultrahdr printed hdrCapacityMin");

    let data = std::fs::read(&fixture).expect("read fixture");
    let dec = Decoder::new(&data).expect("our Decoder::new");
    let meta = dec.metadata().expect("our metadata()");

    // libultrahdr's probe prints linear-domain boost/capacity. Our metadata
    // is log2; convert with 2^x.
    let our_max = 2f64.powf(meta.channels[0].max);
    let our_min = 2f64.powf(meta.channels[0].min);
    let our_gamma = meta.channels[0].gamma;
    let our_sdr_offset = meta.channels[0].base_offset;
    let our_hdr_offset = meta.channels[0].alternate_offset;
    let our_cap_max = 2f64.powf(meta.alternate_hdr_headroom);
    let our_cap_min = 2f64.powf(meta.base_hdr_headroom);

    fn close(tag: &str, ours: f64, theirs: f64, tol: f64) {
        let diff = (ours - theirs).abs();
        assert!(
            diff <= tol,
            "{tag}: ours={ours:.6}, libultrahdr={theirs:.6}, diff={diff:.6} > tol={tol}"
        );
    }

    close("maxContentBoost", our_max, lib_max, 1e-3);
    close("minContentBoost", our_min, lib_min, 1e-3);
    close("gamma", our_gamma, lib_gamma, 1e-4);
    close("offsetSdr", our_sdr_offset, lib_sdr_offset, 1e-4);
    close("offsetHdr", our_hdr_offset, lib_hdr_offset, 1e-4);
    close("hdrCapacityMax", our_cap_max, lib_cap_max, 1e-3);
    close("hdrCapacityMin", our_cap_min, lib_cap_min, 1e-3);

    eprintln!(
        "probe_metadata_matches_libultrahdr: {} — all 7 fields within tolerance",
        fixture.file_name().and_then(|s| s.to_str()).unwrap_or("<?>")
    );
}

// ---------------------------------------------------------------------------
// #13 — MPF byte offsets in our-encoded output
//
// We re-encode a known SDR+HDR pair through our encoder, then probe with
// ultrahdr_app. If MPF is wrong ultrahdr_app will fail to find the
// secondary or report wrong dimensions. Semantic check: libultrahdr agrees
// with our MPF.
// ---------------------------------------------------------------------------

#[test]
fn mpf_offsets_match_libultrahdr() {
    let bin = ultrahdr_app();
    let c = corpus();
    let fixture = first_pixel_sample(&c);

    let data = std::fs::read(&fixture).expect("read");
    let dec = Decoder::new(&data).expect("parse");
    let hdr = dec.decode_hdr(4.0).expect("decode HDR");
    let sdr = dec.decode_sdr().expect("decode SDR");

    let our_jpeg = ultrahdr_rs::Encoder::new()
        .set_hdr_image(hdr)
        .set_sdr_image(sdr)
        .set_quality(90, 85)
        .encode()
        .expect("re-encode");

    let tmp = tempfile::tempdir().expect("tempdir");
    let reenc = tmp.path().join("reenc.jpg");
    std::fs::write(&reenc, &our_jpeg).expect("write reenc");

    let out = Command::new(&bin)
        .args(["-m", "1", "-P", "-j"])
        .arg(&reenc)
        .output()
        .expect("invoke ultrahdr_app on our re-encoded output");
    let probe = String::from_utf8_lossy(&out.stdout);
    assert!(
        out.status.success()
            && probe.contains("Ultra HDR Image: Yes")
            && probe.contains("maxContentBoost"),
        "libultrahdr rejected our re-encoded UltraHDR (MPF likely wrong):\n\
         stdout: {}\nstderr: {}",
        probe,
        String::from_utf8_lossy(&out.stderr)
    );
    eprintln!("mpf_offsets_match_libultrahdr: libultrahdr probed our re-encode cleanly");
}

// ---------------------------------------------------------------------------
// #15 — SDR pixel parity against `ultrahdr_app -m 1 -o 3 -O 3`
// ---------------------------------------------------------------------------

#[test]
fn sdr_decode_matches_libultrahdr() {
    let bin = ultrahdr_app();
    let c = corpus();
    let fixture = first_pixel_sample(&c);

    let tmp = tempfile::tempdir().expect("tempdir");
    let rgb_path = tmp.path().join("out.rgb");

    // -o 3 = sRGB transfer, -O 3 = rgba8888 color format.
    let status = Command::new(&bin)
        .args(["-m", "1", "-j"])
        .arg(&fixture)
        .args(["-o", "3", "-O", "3", "-z"])
        .arg(&rgb_path)
        .status()
        .expect("invoke ultrahdr_app sdr decode");
    assert!(status.success(), "ultrahdr_app sdr decode failed");

    let lib_raw = std::fs::read(&rgb_path).expect("read libultrahdr output");

    let data = std::fs::read(&fixture).expect("read");
    let dec = Decoder::new(&data).expect("parse");
    let our_sdr = dec.decode_sdr().expect("our decode_sdr");
    let expected_bytes = (our_sdr.width as usize) * (our_sdr.height as usize) * 4;
    assert_eq!(
        lib_raw.len(),
        expected_bytes,
        "libultrahdr raw size mismatch: {} bytes vs expected {}",
        lib_raw.len(),
        expected_bytes
    );

    let mut max_delta = 0i32;
    let mut sum_abs_delta: u64 = 0;
    let mut count: u64 = 0;
    let row_bytes = our_sdr.width as usize * 4;
    for y in 0..our_sdr.height as usize {
        let our_row_start = y * our_sdr.stride as usize;
        let lib_row_start = y * row_bytes;
        for x in 0..row_bytes {
            let a = our_sdr.data[our_row_start + x] as i32;
            let b = lib_raw[lib_row_start + x] as i32;
            let d = (a - b).abs();
            if d > max_delta {
                max_delta = d;
            }
            sum_abs_delta += d as u64;
            count += 1;
        }
    }
    let mae = sum_abs_delta as f64 / count as f64;
    eprintln!(
        "sdr_decode_matches_libultrahdr: MAE={mae:.3}, max_delta={max_delta} over {count} bytes"
    );
    assert!(
        max_delta <= 3,
        "max per-byte delta {max_delta} is too high; zenjpeg and libultrahdr should agree within \
         ≤ 3 per ISO/IEC 10918-2 IDCT tolerance"
    );
    assert!(mae < 0.2, "MAE {mae} is too high for the same JPEG input");
}

// ---------------------------------------------------------------------------
// #16 — HDR pixel similarity
//
// -o 0 -O 4 → linear f32 RGBA. 16 bytes per pixel. Same layout as our
// RawImage::Rgba32F, so we can interpret both as `&[f32]` and compare.
// ---------------------------------------------------------------------------

#[test]
fn hdr_decode_matches_libultrahdr() {
    let bin = ultrahdr_app();
    let c = corpus();
    let fixture = first_pixel_sample(&c);

    let tmp = tempfile::tempdir().expect("tempdir");
    let hdr_path = tmp.path().join("out.hdr");

    let status = Command::new(&bin)
        .args(["-m", "1", "-j"])
        .arg(&fixture)
        .args(["-o", "0", "-O", "4", "-z"])
        .arg(&hdr_path)
        .status()
        .expect("invoke ultrahdr_app hdr decode");
    assert!(status.success(), "ultrahdr_app hdr decode failed");

    let lib_raw = std::fs::read(&hdr_path).expect("read libultrahdr hdr output");
    let lib_floats: &[f32] = bytemuck::cast_slice(&lib_raw);

    let data = std::fs::read(&fixture).expect("read");
    let dec = Decoder::new(&data).expect("parse");
    let our_hdr = dec
        .decode_hdr_with_format(
            1.0, // libultrahdr default applies the stored headroom; 1.0 = no boost from us
            ultrahdr_rs::HdrOutputFormat::LinearFloat,
        )
        .expect("our decode_hdr");

    assert_eq!(
        lib_raw.len() as u32,
        our_hdr.width * our_hdr.height * 16,
        "HDR byte size mismatch"
    );

    let our_floats: &[f32] = bytemuck::cast_slice(&our_hdr.data);

    let mut sum_abs: f64 = 0.0;
    let mut count: u64 = 0;
    let mut max_abs: f32 = 0.0;
    for (i, (&a, &b)) in our_floats.iter().zip(lib_floats.iter()).enumerate() {
        if i % 4 == 3 {
            continue; // alpha
        }
        if !(a.is_finite() && b.is_finite()) {
            continue;
        }
        let d = (a - b).abs();
        if d > max_abs {
            max_abs = d;
        }
        sum_abs += d as f64;
        count += 1;
    }
    let mae = sum_abs / count as f64;
    eprintln!("hdr_decode_matches_libultrahdr: MAE={mae:.6}, max_abs={max_abs:.6}");

    assert!(
        mae < 0.05,
        "HDR MAE {mae} too high — pixel reconstruction diverges from libultrahdr"
    );
}
