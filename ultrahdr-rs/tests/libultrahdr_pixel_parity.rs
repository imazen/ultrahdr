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
//! # Thresholds
//!
//! Pixel thresholds are grounded in zenjpeg's own chroma-upsample regression
//! gate (`zenjpeg/tests/bundled/chroma_upsample_regression.rs`): post-fix
//! `boundary_max <= 6` at MCU boundaries (IDCT rounding only, ISO/IEC
//! 10918-2 allows ±1 per channel but both sides use libjpeg-turbo-family
//! IDCT, so realistic maxes are 2-5). We assert `max_delta <= 8` to keep
//! one byte of headroom over that gate.
//!
//! The HDR format is `UHDR_IMG_FMT_64bppRGBAHalfFloat` — 8 bytes/pixel
//! (f16 RGBA), not f32. We decode f32, convert to f16 via `half::f16`,
//! then compare.

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

/// Pick a deterministic fixture from the pixel-ultrahdr corpus.
///
/// CI runs used to see different fixtures between runs because `read_dir`
/// is filesystem-order dependent. We pin to `_05.jpg` here for single-fixture
/// probe/MPF/SDR tests — it's a stable, conformant sample.
///
/// The HDR parity test (`hdr_decode_matches_libultrahdr`) sweeps all three
/// corpus fixtures rather than relying on this pinned pick.
fn first_pixel_sample(corpus: &Corpus) -> PathBuf {
    let dir = corpus
        .get("ultrahdr-conformance/valid/jpeg/pixel-ultrahdr")
        .expect("ultrahdr-conformance/valid/jpeg/pixel-ultrahdr must exist in codec-corpus");
    const PINNED: &str = "Ultra_HDR_Samples_Originals_05.jpg";
    let pinned = dir.join(PINNED);
    if pinned.is_file() {
        return pinned;
    }
    // Fallback: sorted first jpg (so at least it's deterministic).
    let mut jpgs: Vec<PathBuf> = std::fs::read_dir(&dir)
        .expect("read pixel-ultrahdr dir")
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| {
            p.extension()
                .and_then(|s| s.to_str())
                .is_some_and(|s| s.eq_ignore_ascii_case("jpg"))
        })
        .collect();
    jpgs.sort();
    jpgs.into_iter()
        .next()
        .expect("at least one .jpg in pixel-ultrahdr")
}

/// Every `.jpg` in `ultrahdr-conformance/valid/jpeg/pixel-ultrahdr`, sorted.
///
/// Used by [`hdr_decode_matches_libultrahdr`] — every fixture must come under
/// the MAE threshold against libultrahdr, not just one representative sample.
fn all_pixel_samples(corpus: &Corpus) -> Vec<PathBuf> {
    let dir = corpus
        .get("ultrahdr-conformance/valid/jpeg/pixel-ultrahdr")
        .expect("ultrahdr-conformance/valid/jpeg/pixel-ultrahdr must exist in codec-corpus");
    let mut jpgs: Vec<PathBuf> = std::fs::read_dir(&dir)
        .expect("read pixel-ultrahdr dir")
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| {
            p.extension()
                .and_then(|s| s.to_str())
                .is_some_and(|s| s.eq_ignore_ascii_case("jpg"))
        })
        .collect();
    jpgs.sort();
    assert!(
        !jpgs.is_empty(),
        "no .jpg fixtures in pixel-ultrahdr corpus"
    );
    jpgs
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
        fixture
            .file_name()
            .and_then(|s| s.to_str())
            .unwrap_or("<?>")
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
//
// Threshold is grounded in zenjpeg/tests/bundled/chroma_upsample_regression.rs,
// which asserts boundary_max <= 6 at MCU boundaries after the h2v2-fancy
// fix (3ba1f1ab, in zenjpeg >= 0.8.0). One byte of headroom → 8.
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
    let expected_bytes = (our_sdr.width() as usize) * (our_sdr.height() as usize) * 4;
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
    let mut histogram = [0u64; 256];
    // Per-row max delta buckets in groups of 16 rows to find MCU-row
    // patterns without 10K-row spam.
    let w = our_sdr.width() as usize;
    let h = our_sdr.height() as usize;
    let row_bytes = w * 4;
    let row_bucket = |y: usize| y / 16;
    let n_buckets = (h + 15) / 16;
    let mut row_max = vec![0i32; n_buckets];
    // Per-column bucket max (columns in groups of 16).
    let col_bucket = |x: usize| (x / 4) / 16;
    let n_col_buckets = (w + 15) / 16;
    let mut col_max = vec![0i32; n_col_buckets];
    // Sample high-delta hotspots.
    let mut hot: Vec<(usize, usize, u8, i32)> = Vec::new(); // (y, x, channel, delta)
    for y in 0..h {
        let our_row_start = y * our_sdr.stride();
        let lib_row_start = y * row_bytes;
        for x in 0..row_bytes {
            let a = our_sdr.as_slice().as_strided_bytes()[our_row_start + x] as i32;
            let b = lib_raw[lib_row_start + x] as i32;
            let d = (a - b).abs();
            if d > max_delta {
                max_delta = d;
            }
            sum_abs_delta += d as u64;
            count += 1;
            histogram[d as usize] += 1;
            let rb = row_bucket(y);
            if d > row_max[rb] {
                row_max[rb] = d;
            }
            let cb = col_bucket(x);
            if d > col_max[cb] {
                col_max[cb] = d;
            }
            if d >= 30 && hot.len() < 20 {
                let px = x / 4;
                let ch = (x % 4) as u8;
                hot.push((y, px, ch, d));
            }
        }
    }
    let mae = sum_abs_delta as f64 / count as f64;
    eprintln!(
        "sdr_decode_matches_libultrahdr: MAE={mae:.6}, max_delta={max_delta} over {count} bytes"
    );
    eprintln!("  size: {w}x{h}  stride={}", our_sdr.stride());
    // Delta histogram (top bins only).
    let mut cum: u64 = 0;
    eprintln!("  delta histogram (cumulative %):");
    for (d, c) in histogram.iter().enumerate() {
        if *c == 0 {
            continue;
        }
        cum += *c;
        if d <= 10 || d % 5 == 0 || *c > count / 1000 {
            let pct = (cum as f64 / count as f64) * 100.0;
            eprintln!("    delta={d:>3}: count={c:>10} cum={pct:7.4}%");
        }
    }
    // Row bucket max — look for MCU-row bottom boundaries (y % 16 == 15)
    // or MCU-row top (y % 16 == 0).
    eprintln!("  per-16-row max delta (first 8 + last 4):");
    for rb in 0..n_buckets.min(8) {
        eprintln!(
            "    rows {:>5}..{:>5}: max={}",
            rb * 16,
            rb * 16 + 15,
            row_max[rb]
        );
    }
    for rb in n_buckets.saturating_sub(4)..n_buckets {
        eprintln!(
            "    rows {:>5}..{:>5}: max={}",
            rb * 16,
            rb * 16 + 15,
            row_max[rb]
        );
    }
    // Column bucket max — should be roughly flat unless there's a horizontal edge issue.
    let col_max_overall = *col_max.iter().max().unwrap_or(&0);
    let col_min_of_max = *col_max.iter().min().unwrap_or(&0);
    eprintln!(
        "  col buckets: {} total, max-across={col_max_overall}, min-of-maxes={col_min_of_max}",
        n_col_buckets
    );
    // High-delta hotspots.
    eprintln!("  sample pixels with delta >= 30:");
    for (y, px, ch, d) in hot.iter().take(10) {
        let name = match ch {
            0 => "R",
            1 => "G",
            2 => "B",
            _ => "A",
        };
        let y_mod16 = y % 16;
        eprintln!(
            "    ({:>5},{:>5}) ch={name} delta={d}  y%16={y_mod16}",
            y, px
        );
    }

    assert!(
        max_delta <= 8,
        "max per-byte delta {max_delta} exceeds zenjpeg's own chroma-upsample \
         regression gate (boundary_max <= 6 + 1 byte headroom). Diagnostics \
         above show where the deltas live."
    );
    assert!(mae < 0.1, "MAE {mae} is too high for the same JPEG input");
}

// ---------------------------------------------------------------------------
// #16 — HDR pixel similarity
//
// libultrahdr `-O 4` is UHDR_IMG_FMT_64bppRGBAHalfFloat (8 bytes/pixel,
// f16 RGBA). We decode to f32 RGBA via LinearFloat, convert to f16, and
// compare.
// ---------------------------------------------------------------------------

#[test]
fn hdr_decode_matches_libultrahdr() {
    let bin = ultrahdr_app();
    let c = corpus();
    let fixtures = all_pixel_samples(&c);

    // CRITICAL: libultrahdr's `ultrahdr_app -o 0 -O 4` (no explicit
    // `--max_display_boost` flag) defaults `max_display_boost` to `FLT_MAX`
    // inside `uhdr_dec_set_out_max_display_boost`. The decoder then clamps
    // that to `gainmap_metadata->hdr_capacity_max`, so the effective
    // `gainmap_weight` becomes `1.0` (full boost) — NOT zero. We mirror
    // that by passing `2^alternate_hdr_headroom` as our `display_boost`,
    // which our `calculate_weight` maps to weight=1.0 for this fixture.
    //
    // Passing `display_boost = 1.0` (the previous behaviour) was the root
    // cause of the apparent "HDR divergence": every pixel with a non-zero
    // gain-map byte produced SDR output instead of full HDR, yielding
    // MAEs of 0.2–0.8 vs libultrahdr's full-boost output.
    let mut failures: Vec<String> = Vec::new();
    for fixture in &fixtures {
        let name = fixture
            .file_name()
            .and_then(|s| s.to_str())
            .unwrap_or("<?>")
            .to_string();

        let tmp = tempfile::tempdir().expect("tempdir");
        let hdr_path = tmp.path().join("out.hdr");
        let status = Command::new(&bin)
            .args(["-m", "1", "-j"])
            .arg(fixture)
            .args(["-o", "0", "-O", "4", "-z"])
            .arg(&hdr_path)
            .status()
            .expect("invoke ultrahdr_app hdr decode");
        assert!(
            status.success(),
            "ultrahdr_app hdr decode failed for {name}"
        );

        let lib_raw = std::fs::read(&hdr_path).expect("read libultrahdr hdr output");
        // 8 bytes/pixel (f16 RGBA). f16 lacks AnyBitPattern, so go via u16 bits.
        let lib_halfs: Vec<half::f16> = bytemuck::cast_slice::<u8, u16>(&lib_raw)
            .iter()
            .map(|&bits| half::f16::from_bits(bits))
            .collect();

        let data = std::fs::read(fixture).expect("read");
        let dec = Decoder::new(&data).expect("parse");
        let meta = dec.metadata().expect("metadata present");
        let boost = 2f32.powf(meta.alternate_hdr_headroom as f32);
        let our_hdr = dec
            .decode_hdr_with_format(boost, ultrahdr_rs::HdrOutputFormat::LinearFloat)
            .expect("our decode_hdr");

        assert_eq!(
            lib_raw.len() as u32,
            our_hdr.width() * our_hdr.height() * 8,
            "{name}: HDR byte size mismatch (expected 8 bytes/pixel for f16 RGBA)"
        );

        let our_floats: &[f32] = bytemuck::cast_slice(&our_hdr.as_slice().as_strided_bytes());
        assert_eq!(
            our_floats.len(),
            lib_halfs.len(),
            "{name}: pixel count mismatch"
        );

        let mut sum_abs: f64 = 0.0;
        let mut count: u64 = 0;
        let mut max_abs: f32 = 0.0;
        for (i, (&ours_f32, &theirs_f16)) in our_floats.iter().zip(lib_halfs.iter()).enumerate() {
            if i % 4 == 3 {
                continue; // alpha
            }
            let theirs = theirs_f16.to_f32();
            if !(ours_f32.is_finite() && theirs.is_finite()) {
                continue;
            }
            let d = (ours_f32 - theirs).abs();
            if d > max_abs {
                max_abs = d;
            }
            sum_abs += d as f64;
            count += 1;
        }
        let mae = sum_abs / count as f64;
        eprintln!(
            "hdr_decode_matches_libultrahdr: {name} boost={boost:.4} MAE={mae:.6} max_abs={max_abs:.4}"
        );
        // Linear HDR values span ~[0, 16] for 4× boost content; 0.1 MAE is
        // ~0.6% of dynamic range. f16 quantizes to ~10-bit mantissa so we
        // can't go tighter than a few ULPs at large magnitudes.
        if mae >= 0.1 {
            failures.push(format!(
                "{name}: MAE={mae:.6} >= 0.1 (max_abs={max_abs:.4})"
            ));
        }
    }

    assert!(
        failures.is_empty(),
        "HDR parity failed for {} fixture(s):\n  {}",
        failures.len(),
        failures.join("\n  ")
    );
}
