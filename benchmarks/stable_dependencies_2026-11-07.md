# Stable dependency cohort, 2026-11-07

This separate candidate preserves all implementation/test files from full HDR
review `6df45a2eb58536b0f4f1c5d6276c7d551c6abe09`. It retains the distinct
JPEG `162a1f35` and linear-srgb `2969ee84` graph. The JPEG review's HDR302
graph is separate and is not substituted here. No main or prior review ref
was overwritten.

The committed full-root seed lock selects registry archmage, archmage-macros
and magetypes 0.9.31 and moxcms 0.9.2. Garb `8f9aa10d`, zenblend `d477b29c`,
zentone `2598054c` and zenresize `5c5fe0d5` are stable-cap descendants with
unchanged kernels. Core `1fb997bb` and pixel `fde59b96` families remain fixed.

`just cohort-full-tests` passes formatting, full workspace all-feature/all-target
strict Clippy, all 435 library/integration tests (one existing ignore), and five
doctests (one existing ignore). The existing libultrahdr numeric/metadata/pixel
oracle controls run with the pinned external oracle enabled, including all
four pixel-parity controls. No pixel threshold, fixture, golden or ignore changes.
Wrapper: 77 seconds, peak RSS 0.78 GiB, minimum available 15946 MiB,
peak load 1.82. These are validation resources, not codec throughput or
per-image memory.

## Genuine portability blocker retained

`just stable-freestanding-check` fails on both x86_64-unknown-none and
wasm32-unknown-unknown with default features disabled. Each target reports
43 float-method/type errors, matching the existing blocker. Both commands
run and the recipe returns nonzero; supported std results are not relabeled
as no_std. Failure wrapper: seven seconds, peak RSS 0.32 GiB.

Platform CI, API snapshots and standalone fuzz locks were not requalified
by this candidate. The original source, numeric expectations and all earlier
failing receipts are retained. No merge or publication is requested.
