# Reviewed SIMD and BT.2446 cohort — 2026-11-06

This candidate keeps the public API, kernels, numeric assertions and existing SIMD floors. The SIMD ceiling explicitly admits 0.9.31-beta.2. It is temporary, excludes stable 0.9.31, and must be replaced with a stable-compatible range after publication and unchanged numeric/full-feature qualification.

The BT.2446 dependency is different: `bt2446a` first exists in converter commit `e70065f77aef0fbe9a7bc3349eb9f2cb5de6af58`, whose manifest says 0.3.1. Older same-version contents at fa836abf do not define the feature. Published converter 0.2.16 also lacks it. Therefore a version label cannot prove availability. The separate feature-forwarding commit pins reviewed main-cohort contents at `fde59b96632557592cffa9b7301acd8db96ed078` (descendant of e70065f), retains the >=0.3.1 floor and requests `zenpixels-convert/bt2446a` explicitly. Remove that git pin only after a published package contains both the feature and exported type. No public signature or arithmetic changes are made.

## Qualification

Outside ambient workspace overrides, an export retaining only the core workspace member used zencodec 1fb997bb88e7405e6a315bf1f0b22f71e99be272, pixels/converter fde59b96632557592cffa9b7301acd8db96ed078, garb 605276d717ea0b31a1cd2a36ae05301383c7b8b5 and linear-srgb 52896f4b24664315586bd60e5f048253b1f69052. PR199 was not merged or substituted. This is core qualification, not the ultrahdr-rs workspace or release gate.

Separate exact-source dependency candidates, with only SIMD manifest requirements edited:

- zenresize: `4ad64b3c0a29469e68ddfdda3798de20c60c3421` on `review/sansio-simd-beta2`.
- zenblend: `c61d051bb0a1fd0afb0a73d15ed6de8dcbf1eeed` on `review/sansio-simd-beta2`.
- zentone: `a8e3394cc0a2c8444f0cd74e8ab392964ab87f62` on `review/sansio-simd-beta2`.

Commands (executed from the isolated export):

```sh
cargo test -p ultrahdr-core --all-features --lib --tests
cargo clippy -p ultrahdr-core --all-features --lib --tests -- -D warnings
cargo clippy -p ultrahdr-core --all-features --all-targets -- -D warnings
```

The all-feature library/integration tests pass: 180 unit, five reference-parity and three SIMD-consistency controls, zero ignored or filtered. Their existing expected values/tolerances remain unchanged. Strict library/test Clippy passes. The first broader all-target Clippy failed at the simd_xplat benchmark's deprecated `mul_add` call (line 283). A separate reviewed benchmark-only migration now names `mul_add_relaxed`. Exact beta.2 f32x8 implementation lines342–344 show that the old method delegates directly to it, so this preserves the selected cohort's operation and rounding. A dev-only beta.2 requirement supplies that method without raising the runtime dependency floors. No numeric assertion or warning suppression changes. Strict all-target Clippy and all three all-feature doctests now pass. At this core-only stage the complete workspace/platform gates remained open; the full follow-through below supersedes that stage.

The pre-forwarding full-feature test failed because Bt2446A was configured out; the same command succeeds after the separate feature commit. Default+simd,f16,resize numeric controls also passed before forwarding. All logs are retained by the coordinating integration lane; these results establish neither performance nor a process-memory bound.

Historical full JPEG compilation then failed three PNG private-helper calls. Its owner supplied runtime6b740345; current JPEG full runtime/lint/API and independent color-model oracle gates pass on the reviewed cohort. PNG source remains separately owned.

## Lock and contents

The isolated core lock has one identity each for core, pixels, converter, enough 0.4.5 and whereat 0.1.5; archmage, macros and magetypes all select beta.2. No tracked workspace lock was changed: its old multi-member graph is not represented as a passing baseline. The exact qualification lock SHA-256 is `06b6f7b5643a05f69afe273b0aa7f6fb016c99701c4b773f9a5d1af6fe4cbf6b`.

The initial source workspace used older codec/pixel patches, which were not qualified by the isolated core export. The full follow-through below supplies explicit root pins. Do not land or publish this candidate independently.

PR199 compatibility: in a separate export/target with only core88fc5871, pixelsb6b12bbe and garb58ce9cad substituted together, the same 180+5+3 all-feature core tests pass unchanged. This does not merge PR199 or change the main-cohort pin in this candidate.

## Full workspace follow-through

The separate full export retains all three workspace members and pins JPEG
162a1f353864713f60f272b8fff5638d78219437, main core1fb997bb, pixels/converter
fde59b96, garb605276d7, linear2969ee84 and the already-reviewed resize/blend/
tone manifest cohorts. Main and PR199 remain distinct; the PR199 core-only
qualification above is not a full PR199 wrapper result. The exact new root
lock is `sansio_full_dependencies.lock`; identities and the seeded standalone
fuzz delta are in `sansio_full_graph_2026-11-06.json`. There was no tracked root
workspace lock baseline. Fuzz preserves unrelated versions: twelve removed/
added identities, with only the three SIMD versions changing to beta.2 and
reviewed source identities replacing the remaining edges. The earlier broad
fuzz update was replaced with a reseed and narrow SIMD unlock.

The full std workspace all-feature library/integration tests, all-feature
doctests and strict all-target Clippy pass (ultra-full-qualification-2.log rc0,
71s, peakRSS0.80GiB). All four existing independent pixel controls run: SDR,
HDR, MPF offsets and probe metadata. Their thresholds, assertions and corpus
fixtures are unchanged; existing ignored tests/doctests remain ignored.
Reference libultrahdr source5f117294cb701170b9198ed8eae16270aa84896c was
independently built with CMake and system JPEG (rc0,11s, peakRSS0.22GiB).
Five strict lint failures in pixel-parity diagnostics were repaired with
div_ceil, iterator enumeration and removal of a needless borrow; no sample
math, codec path or assertion changed.

Default workspace tests/doctests and minimal-std core tests also pass. The
standalone fuzz binary check passes with the seeded lock. Full std tests,
pixel controls and doctests pass again after the feature-ownership change
below; the all-feature core WASM build also passes (the final combined log
then fails only its stale API snapshot). These are qualification resources,
not codec performance or memory bounds.

### Current blocker: genuine no_std math

The genuine x86_64-unknown-none check first fails on implicit thiserror/std.
The manifest now disables that default edge and forwards it only from core/std,
preserving ordinary std behavior. That exposes43 existing unsupported float
methods in color/transfer.rs and gainmap/{apply,compute,streaming}.rs: log/exp,
powf, sqrt and floor/round. Both genuine freestanding and WASM with no default
features fail. The final independent command records WASM_NO_DEFAULT_RC=101
and FUZZ_RC=0. A host-target no-default test is not freestanding evidence.
No portable math backend, numeric tolerance or expectation was substituted.
Portable math requires separate numeric qualification and review. The failing
portability recipe stays enabled, and README/crate docs now state this limit.
Keep the draft; platform/release gates also remain open.

### API inventory

The standalone API runner uses published0.1.1 and pinned nightly2026-07-13,
matching JPEG's featureless-workspace tooling repair. The first strict check
finds stale snapshots; regeneration changes only ultrahdr-core's supported
snapshot to include the already-existing tonemap_image_to_srgb8_with_stop,
verified in parent30249993 at color/tonemap.rs1108. No new function or public
signature was introduced by these root/feature repairs. Feature and internal
snapshots are unchanged. Regeneration passes (rc0,12s); the strict final check
passes (ultra-api-check-final.log rc0,4s, peakRSS0.35GiB), using the same
pinned toolchain and every workspace API package.
