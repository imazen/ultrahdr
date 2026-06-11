//! Public-API surface snapshots for the PARENT workspace (docs/public-api/).
//! Shared implementation + format docs: the `zenutils-apidoc` crate.
#[test]
fn public_api_surface_docs_are_current() {
    zenutils_apidoc::ApiDoc::new()
        .workspace_dir("..")
        // `ffi-tests` only toggles dev/test-side deps (libultrahdr C++ via
        // cmake, a git dep) — no public-API impact, heavy build cost. Its
        // (empty) surface is documented in ultrahdr-rs.internal.txt instead
        // of the supported files.
        .exclude_features("ultrahdr-rs", ["ffi-tests"])
        .run();
}
