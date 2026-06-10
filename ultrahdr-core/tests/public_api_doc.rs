//! Regenerates the committed public-API surface snapshots
//! (`docs/public-api/<crate>.txt`, one per published crate) on every
//! `cargo test` run, so API changes always show up as a git diff next to the
//! code change that caused them, and the surface size stays one glance away.
//!
//! Modes (`ZEN_API_DOC` env var — set per-job in CI workflows, see ci.yml):
//! - unset / `regen` → regenerate the files in place (local default; commit the diff)
//! - `check`         → regenerate to memory, FAIL if a committed file is stale
//! - `off`           → skipped (CI matrix jobs without nightly rustdoc / the tool)
//!
//! Two sections per crate: default features, and "all features except `_*`" —
//! every feature from the crate's manifest except underscore-prefixed ones,
//! which are internal/research gates and not public surface. The list is
//! computed from `cargo metadata`, so new features appear automatically.
//!
//! Adaptations for this workspace:
//! - The test lives in ultrahdr-core (not ultrahdr-rs) because ci.yml's
//!   standard jobs strip ultrahdr-rs from the workspace (sibling path-deps);
//!   the dedicated api-doc CI job clones the siblings and runs the check.
//! - `ffi-tests` is excluded from ultrahdr-rs's all-features section: it only
//!   gates dev/test deps (the cmake-built libultrahdr C++ parity harness, a
//!   git dep) and contributes zero public surface.
//!
//! Requires `cargo-public-api` (0.52+) and a nightly toolchain for rustdoc
//! JSON: `cargo install cargo-public-api --locked && rustup toolchain install nightly`

use std::path::PathBuf;
use std::process::Command;

/// Published crates in this workspace (wasm-bench is publish = false).
const CRATES: &[&str] = &["ultrahdr-core", "ultrahdr-rs"];

/// Features excluded from the all-features section beyond the automatic
/// underscore-prefix rule. `ffi-tests` only toggles dev/test-side deps
/// (libultrahdr C++ via cmake) — no public-API impact, heavy build cost.
fn excluded_features(package: &str) -> &'static [&'static str] {
    match package {
        "ultrahdr-rs" => &["ffi-tests"],
        _ => &[],
    }
}

fn run(args: &[&str]) -> Vec<u8> {
    let out = Command::new("cargo")
        .current_dir(env!("CARGO_MANIFEST_DIR"))
        .args(args)
        .output()
        .unwrap_or_else(|e| {
            panic!(
                "failed to run cargo {} ({e}); for public-api: install with \
                 `cargo install cargo-public-api --locked` and ensure a nightly \
                 toolchain exists (`rustup toolchain install nightly`), or set \
                 ZEN_API_DOC=off to skip this test",
                args[0]
            )
        });
    assert!(
        out.status.success(),
        "cargo {} failed (set ZEN_API_DOC=off to skip):\n{}",
        args.join(" "),
        String::from_utf8_lossy(&out.stderr)
    );
    out.stdout
}

fn surface(package: &str, feature_args: &[&str]) -> Vec<String> {
    let mut args = vec!["public-api", "-p", package, "--simplified"];
    args.extend_from_slice(feature_args);
    String::from_utf8(run(&args))
        .expect("cargo public-api emitted non-UTF8")
        .lines()
        .map(str::to_owned)
        .filter(|l| !l.is_empty())
        .collect()
}

/// All manifest features of `package` except `default`, underscore-prefixed
/// internal gates, and the per-crate exclusions, sorted for determinism.
fn public_features(package: &str) -> Vec<String> {
    let meta: serde_json::Value =
        serde_json::from_slice(&run(&["metadata", "--no-deps", "--format-version", "1"]))
            .expect("cargo metadata JSON");
    let pkg = meta["packages"]
        .as_array()
        .expect("packages array")
        .iter()
        .find(|p| p["name"] == package)
        .unwrap_or_else(|| panic!("{package} not in workspace metadata"));
    let excluded = excluded_features(package);
    let mut feats: Vec<String> = pkg["features"]
        .as_object()
        .expect("features map")
        .keys()
        .filter(|k| *k != "default" && !k.starts_with('_') && !excluded.contains(&k.as_str()))
        .cloned()
        .collect();
    feats.sort();
    feats
}

#[test]
fn public_api_surface_docs_are_current() {
    match std::env::var("ZEN_API_DOC").as_deref() {
        Ok("off") => {
            eprintln!("ZEN_API_DOC=off — public-API snapshot regen skipped by caller");
            return;
        }
        Ok("check") | Ok("regen") | Err(_) => {}
        Ok(other) => panic!("unknown ZEN_API_DOC value {other:?} (off|check|regen)"),
    }
    let check = std::env::var("ZEN_API_DOC").as_deref() == Ok("check");

    for package in CRATES {
        let features = public_features(package);
        let feature_csv = features.join(",");
        let sections: [(&str, Vec<&str>); 2] = [
            ("default features", vec![]),
            (
                "all features except _*",
                if features.is_empty() {
                    vec![]
                } else {
                    vec!["--features", &feature_csv]
                },
            ),
        ];

        let mut doc = String::new();
        doc.push_str(&format!(
            "# {package} public API surface\n\
             # Generated by ultrahdr-core/tests/public_api_doc.rs via `cargo public-api --simplified`\n\
             # (regenerated on every `cargo test`; ZEN_API_DOC=check verifies, =off skips).\n\
             # Underscore-prefixed features are internal and excluded from the\n\
             # all-features section (so is ultrahdr-rs's test-only `ffi-tests`).\n\
             # DO NOT EDIT BY HAND — commit regenerated changes together with the code.\n"
        ));
        for (label, extra) in &sections {
            let items = surface(package, extra);
            doc.push_str(&format!("\n## {label} ({} items)\n\n", items.len()));
            for line in &items {
                doc.push_str(line);
                doc.push('\n');
            }
            eprintln!("{package} [{label}]: {} public items", items.len());
        }

        let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../docs/public-api")
            .join(format!("{package}.txt"));
        let existing = std::fs::read_to_string(&path).ok();

        if check {
            assert_eq!(
                existing.as_deref(),
                Some(doc.as_str()),
                "committed public-API snapshot for {package} is stale: run \
                 `cargo test` locally and commit the regenerated {}",
                path.display()
            );
        } else if existing.as_deref() != Some(doc.as_str()) {
            std::fs::create_dir_all(path.parent().unwrap()).unwrap();
            std::fs::write(&path, &doc).unwrap();
            eprintln!(
                "regenerated {} — review and commit the diff",
                path.display()
            );
        }
    }
}
