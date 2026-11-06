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

The all-feature library/integration tests pass: 180 unit, five reference-parity and three SIMD-consistency controls, zero ignored or filtered. Their existing expected values/tolerances remain unchanged. Strict library/test Clippy passes. The first broader all-target Clippy failed at the simd_xplat benchmark's deprecated `mul_add` call (line 283). A separate reviewed benchmark-only migration now names `mul_add_relaxed`. Exact beta.2 f32x8 implementation lines342–344 show that the old method delegates directly to it, so this preserves the selected cohort's operation and rounding. A dev-only beta.2 requirement supplies that method without raising the runtime dependency floors. No numeric assertion or warning suppression changes. Strict all-target Clippy and all three all-feature doctests now pass. The complete workspace/platform gates remain open, so retain the draft.

The pre-forwarding full-feature test failed because Bt2446A was configured out; the same command succeeds after the separate feature commit. Default+simd,f16,resize numeric controls also passed before forwarding. All logs are retained by the coordinating integration lane; these results establish neither performance nor a process-memory bound.

Full JPEG graph Clippy and native PNG controls get past resolution, then fail three PNG private-helper calls; PNG source migration is owned separately. Resolution is not qualification.

## Lock and contents

The isolated core lock has one identity each for core, pixels, converter, enough 0.4.5 and whereat 0.1.5; archmage, macros and magetypes all select beta.2. No tracked workspace lock was changed: its old multi-member graph is not represented as a passing baseline. The exact qualification lock SHA-256 is `06b6f7b5643a05f69afe273b0aa7f6fb016c99701c4b773f9a5d1af6fe4cbf6b`.

The reviewed source still requires coordinated root graph pins. The source workspace's older codec/pixel patches are not qualified by this isolated export. Do not land or publish this candidate independently.

PR199 compatibility: in a separate export/target with only core88fc5871, pixelsb6b12bbe and garb58ce9cad substituted together, the same 180+5+3 all-feature core tests pass unchanged. This does not merge PR199 or change the main-cohort pin in this candidate.
