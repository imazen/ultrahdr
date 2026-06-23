//! Color space handling: transfer functions, gamut matrices, HDR→SDR tone mapping.
//!
//! YCbCr conversions live in `zenyuv` / `yuv` / `zenjpeg::color`; RGBA1010102
//! pack/unpack lives in `garb`; f16 pixel storage is handled by `zenpixels`.
//! This module owns only what's codec-agnostic and Ultra-HDR-specific.

pub mod gamut;
pub mod transfer;

/// The full `color::tonemap` surface (AdaptiveTonemapper, ProfileToneCurve,
/// tonemap_image_to_srgb8, and the zentone curve re-exports). Callers
/// computing the SDR base for a gain map can also drive the splitter
/// directly via [`zentone::LumaGainMapSplitter`] (re-exported at the
/// crate root) with [`zentone::HableFilmic`] or any other
/// [`zentone::LumaToneMap`] implementation.
///
/// Gated behind the `tonemap` feature (default-on); the entire module is
/// zentone-backed so decoder-only consumers (`--no-default-features
/// --features std`) opt out of it along with the transitive zentone dep.
#[cfg(feature = "tonemap")]
pub mod tonemap;

/// Production-recommended HDR→SDR primitives (audited 2026-06-22 shootout).
///
/// Lives in its own module so consumers can opt into [`Bt2446A`] and
/// [`measure_max`](CllMeasure::measure_max) by enabling
/// `features = ["tonemap-bt2446a"]` without also pulling in the broader
/// zentone-backed `tonemap` module (which is what the default tone curve
/// grab-bag uses). Both flags are independent.
///
/// `Bt2446A` is the **production-recommended** curve per the 2026-06-22
/// audited shootout (76 imazen-26 samples × 20 curves × 4 peak methods),
/// winning mean ΔE2000 by 2-5× over every channel-independent curve tested.
/// `measure_max` is the production-recommended peak measurement, winning 3
/// of 6 ranking criteria including the user-visible `pct_above_de5` by 11 %.
///
/// See `zen/zentone/benchmarks/shootout_2026-06-22_findings_v2.md`.
#[cfg(feature = "tonemap-bt2446a")]
pub mod audited {
    /// **Production-recommended HDR → SDR tone curve** — re-exported from
    /// [`zenpixels_convert::hdr::Bt2446A`].
    ///
    /// ITU-R BT.2446 Method A. The 2026-06-22 audited shootout (76 imazen-26
    /// samples × 20 curves × 4 peak methods) crowned this curve the winner by
    /// 2-5× mean ΔE2000 over every channel-independent curve tested — the only
    /// tone-map with a peer-reviewed psychophysical study showing imperceptible
    /// degradation after a full HDR → SDR → HDR round-trip on graded content.
    ///
    /// Pick this over the zentone Reinhard / Filmic re-exports in
    /// [`super::tonemap`] for any new HDR→SDR code. See
    /// `zen/zentone/benchmarks/shootout_2026-06-22_findings_v2.md`.
    pub use zenpixels_convert::hdr::Bt2446A;

    /// **Production-recommended HDR peak measurement** — re-exported from
    /// [`zenpixels_convert::hdr::CllMeasure`].
    ///
    /// Extension trait on `zenpixels::hdr::ContentLightLevel`. Carries
    /// `measure_max` (the spec-conformant CTA-861.3 MaxCLL + MaxFALL scan)
    /// which won 3 of 6 ranking criteria in the 2026-06-22 audited shootout
    /// — including the user-visible `pct_above_de5` by 11 % over the closest
    /// alternative.
    pub use zenpixels_convert::hdr::{CllMeasure, LightLevelMethod};

    /// Diffuse-white anchor — sample `1.0` maps to
    /// [`DiffuseWhite::BT2408`] (203 cd/m²) by convention. Required argument
    /// for [`CllMeasure::measure_max`].
    pub use zenpixels::hdr::{ContentLightLevel, DiffuseWhite};
}

pub use gamut::*;
pub use transfer::*;

#[cfg(feature = "tonemap")]
pub use tonemap::*;

#[cfg(feature = "tonemap-bt2446a")]
pub use audited::*;

/// **Deprecated** — slated for removal in 0.5.0. Pass-through re-export
/// of `zentone::experimental::{StreamingTonemapConfig, StreamingTonemapper}`;
/// no ultrahdr-core-specific logic here. Import directly from `zentone`.
#[cfg(feature = "tonemap")]
#[doc(hidden)]
pub mod streaming_tonemap {
    pub use zentone::experimental::{StreamingTonemapConfig, StreamingTonemapper};
}
#[cfg(feature = "tonemap")]
#[doc(hidden)]
pub use streaming_tonemap::*;
