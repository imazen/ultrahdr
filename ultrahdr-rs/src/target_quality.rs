//! ultrahdr-owned perceptual-target encode loop.
//!
//! Per the per-codec-loop-ownership directive (2026-08-25), each codec owns
//! its own target/secant loop. This module is Ultra HDR's — a per-codec copy
//! of zenjpeg's `target_quality` (same math, by directive; zenavif's is the
//! family exemplar).
//!
//! [`search_target`] is the pure, dependency-free bracketed secant/bisection
//! core: given a monotone score-vs-quality relationship and a trial closure
//! that encodes at a quality and returns the achieved score, it converges on
//! the *lowest* quality whose achieved score reaches the target band (the
//! smallest file that hits the target, by monotonicity). The metric is
//! INJECTED — ultrahdr does not (and must not) depend on `zensim`; the caller
//! supplies the scorer, typically: build an [`crate::encode::Encoder`] with
//! `set_hdr_image(..)` + `set_quality(q, gainmap_q)`, `encode()`, decode the
//! Ultra HDR back to HDR, and score against the source with an HDR metric.
//!
//! The searched knob is the BASE JPEG quality; `gainmap_quality` stays
//! caller-fixed inside the trial closure (searching both is a later, separate
//! design — the base quality dominates the RD point).
//!
//! [`anchor_guess`] is the zenjpeg-derived content-blind seed (target 0 →
//! q40, 100 → q98). It is UNCALIBRATED for HDR-metric targets — the bracketed
//! search corrects it within a couple of iterations; a gainmap-specific
//! anchor (or a Zq seed head) is registered follow-up work, not claimed here.

// (std crate — Vec via prelude)

/// Search configuration for [`search_target`] / [`encode_with_target`].
#[derive(Debug, Clone)]
pub struct TargetOptions {
    /// Lowest quality the search may try (inclusive), 0–100.
    pub min_quality: f32,
    /// Highest quality the search may try (inclusive), 0–100.
    pub max_quality: f32,
    /// Convergence half-width: a trial within `target ± tolerance` stops the loop.
    pub tolerance: f64,
    /// Hard cap on encode→decode→score iterations.
    pub max_encodes: u8,
    /// Optional starting quality (e.g. a zenpredict Zq seed). `None` uses
    /// [`anchor_guess`].
    pub q_start: Option<f32>,
}

impl Default for TargetOptions {
    fn default() -> Self {
        Self {
            min_quality: 1.0,
            max_quality: 100.0,
            tolerance: 0.5,
            max_encodes: 8,
            q_start: None,
        }
    }
}

/// Outcome of [`search_target`]: which quality won, its achieved score, and
/// whether the target band was actually reached.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TargetSearchResult {
    /// The selected quality (lowest quality reaching the target band, or the
    /// closest-scoring quality if the target was unreachable).
    pub quality: f32,
    /// The achieved score at [`Self::quality`].
    pub score: f64,
    /// `true` iff some trial reached `>= target - tolerance`.
    pub converged: bool,
    /// Number of trial encodes performed.
    pub encodes: u8,
}

/// Fixed anchor initial guess for a `[0,100]`-scaled perceptual target, used
/// when no `q_start` seed is supplied. Deliberately conservative (mid-high),
/// since most web-quality targets land there; the bracketed search corrects it
/// within a couple of iterations regardless.
#[must_use]
pub fn anchor_guess(target: f64) -> f32 {
    // Linear-ish map: target 0 → q40, target 100 → q98, clamped.
    ((0.40 + target / 100.0 * 0.58) * 100.0).clamp(1.0, 100.0) as f32
}

/// Pure bracketed secant/bisection search over a monotone score-vs-quality
/// curve. `trial(q)` performs one encode-at-`q` and returns the achieved score
/// (higher score = higher fidelity). Returns the lowest quality whose score
/// reaches `target - tolerance` (the smallest file at the target, by
/// monotonicity), or the closest-scoring quality if the target is unreachable in
/// `[min_quality, max_quality]`.
///
/// The metric is fully injected via `trial`; this function has no codec or
/// metric dependency and is unit-tested with synthetic monotone curves.
///
/// # Errors
/// Propagates the first error returned by `trial`.
pub fn search_target<E, Err>(
    target: f64,
    options: &TargetOptions,
    mut trial: E,
) -> Result<TargetSearchResult, Err>
where
    E: FnMut(f32) -> Result<f64, Err>,
{
    let tol = options.tolerance.max(0.0);
    let min_q = options.min_quality.clamp(0.0, 100.0);
    let max_q = options.max_quality.clamp(0.0, 100.0);
    let (min_q, max_q) = if min_q <= max_q {
        (min_q, max_q)
    } else {
        (max_q, min_q)
    };
    let max_encodes = options.max_encodes.max(1);

    // Lowest quality known to reach the band (smallest file, by monotonicity),
    // and the overall closest-scoring iterate as a fallback.
    let mut best_reaching: Option<(f32, f64)> = None;
    let mut best_any: Option<(f32, f64)> = None;

    // Bracket: lo = highest q known BELOW target, hi = lowest q known AT/ABOVE.
    let mut lo: Option<(f32, f64)> = None;
    let mut hi: Option<(f32, f64)> = None;

    let mut q = options
        .q_start
        .filter(|v| v.is_finite())
        .unwrap_or_else(|| anchor_guess(target))
        .clamp(min_q, max_q);
    let mut encodes = 0u8;

    while encodes < max_encodes {
        let s = trial(q)?;
        encodes += 1;

        if best_any.is_none_or(|(_, bs)| (s - target).abs() < (bs - target).abs()) {
            best_any = Some((q, s));
        }
        if s >= target - tol && best_reaching.is_none_or(|(bq, _)| q < bq) {
            best_reaching = Some((q, s));
        }

        if (s - target).abs() <= tol {
            break;
        }
        if s < target {
            lo = Some((q, s));
        } else {
            hi = Some((q, s));
        }

        let next = match (lo, hi) {
            (Some((lq, ls)), Some((hq, hs))) => {
                let span = hq - lq;
                if span <= 1.0 {
                    break; // adjacent integer qualities — quantization floor
                }
                let sec = if (hs - ls).abs() > 1e-9 {
                    lq + ((target - ls) / (hs - ls)) as f32 * span
                } else {
                    lq + span / 2.0
                };
                // Clamp away from endpoints so the bracket provably shrinks.
                sec.clamp(lq + span * 0.1, hq - span * 0.1)
            }
            (Some((lq, ls)), None) => {
                // Under target, no upper bracket yet: extrapolate up.
                let step = (((target - ls) as f32) * 1.2).max(4.0);
                let n = (lq + step).min(max_q);
                if n <= lq + 0.5 {
                    break; // pinned at max_quality and still short
                }
                n
            }
            (None, Some((hq, hs))) => {
                // At/above target, no lower bracket yet: extrapolate down.
                let step = (((hs - target) as f32) * 1.2).max(4.0);
                let n = (hq - step).max(min_q);
                if n >= hq - 0.5 {
                    break; // pinned at min_quality and still over
                }
                n
            }
            (None, None) => unreachable!("one of lo/hi is set after a trial"),
        };

        // Round to an integer quality (JPEG quality is integer-valued) and stop
        // if the search would repeat the same quality.
        let next = next.round().clamp(min_q, max_q);
        if (next - q).abs() < 0.5 {
            break;
        }
        q = next;
    }

    let (quality, score, converged) = match (best_reaching, best_any) {
        (Some((q, s)), _) => (q, s, true),
        (None, Some((q, s))) => (q, s, false),
        // max_encodes >= 1 guarantees at least one trial populated best_any.
        (None, None) => unreachable!("at least one trial always runs"),
    };
    Ok(TargetSearchResult {
        quality,
        score,
        converged,
        encodes,
    })
}

/// Result of [`encode_with_target`]: the winning encode plus the search outcome.
#[derive(Debug, Clone)]
pub struct TargetedEncode {
    /// The encoded bytes at the selected quality.
    pub data: Vec<u8>,
    /// The search outcome (quality, achieved score, converged, encodes).
    pub search: TargetSearchResult,
}

/// Run the target-quality loop and return the actual winning encode.
///
/// `trial(quality)` performs one encode-at-`quality` and returns
/// `(encoded_bytes, achieved_score)` — the caller wires zenjpeg's real
/// `JpegEncoderConfig` encode + `JpegDecoderConfig` decode + an injected metric
/// there (caller-supplied). This function owns the bracketed
/// secant search ([`search_target`]), the selection policy (lowest quality
/// reaching the target band = smallest file, by monotonicity), and returns that
/// iterate's bytes without re-encoding.
///
/// The metric+codec are injected via `trial`, so zenjpeg depends on neither
/// `zensim` nor a specific decoder here — keeping this the reusable, always-
/// available core of zenjpeg's owned loop.
///
/// # Errors
/// Propagates the first error returned by `trial`.
pub fn encode_with_target<E, Err>(
    target: f64,
    options: &TargetOptions,
    mut trial: E,
) -> Result<TargetedEncode, Err>
where
    E: FnMut(f32) -> Result<(Vec<u8>, f64), Err>,
{
    // Keep each trial's bytes so the winning quality is returned without a
    // redundant final encode.
    let mut cache: Vec<(u8, Vec<u8>)> = Vec::new();
    let search = search_target(target, options, |q| {
        let (bytes, score) = trial(q)?;
        cache.push((q.round().clamp(0.0, 100.0) as u8, bytes));
        Ok(score)
    })?;
    let qi = search.quality.round().clamp(0.0, 100.0) as u8;
    let data = cache
        .into_iter()
        .rev()
        .find(|(cq, _)| *cq == qi)
        .map(|(_, b)| b)
        .unwrap_or_default();
    Ok(TargetedEncode { data, search })
}

#[cfg(test)]
mod tests {
    use super::*;

    // A synthetic monotone score-vs-quality curve: score(q) = q (identity), so
    // the target IS the quality that hits it.
    #[test]
    fn identity_curve_converges_to_target() {
        let opts = TargetOptions {
            tolerance: 0.5,
            max_encodes: 12,
            ..Default::default()
        };
        let r = search_target(72.0, &opts, |q| Ok::<_, ()>(q as f64)).unwrap();
        assert!(r.converged);
        assert!((r.score - 72.0).abs() <= 0.5, "score {} off band", r.score);
        assert!((r.quality - 72.0).abs() <= 1.0, "quality {}", r.quality);
    }

    // Selection + walk-down: a SATURATED curve (many high qualities all score
    // the same, over target beyond tolerance) must not stop at the high seed —
    // it brackets DOWN and returns a low quality near the saturation knee (the
    // lowest reaching quality = smallest file). Small tolerance so the seed does
    // NOT immediately satisfy.
    #[test]
    fn selects_lowest_reaching_quality() {
        let opts = TargetOptions {
            tolerance: 1.0,
            max_encodes: 24,
            q_start: Some(95.0), // start high, well above the band, force walk-down
            ..Default::default()
        };
        // score(q) = 92 for q>=40 (saturated), else q*2.3. Target 85, band [84,∞):
        // reached for q >= ~36.5. Seed q95 scores 92 (over by 7 > tol) → must walk
        // down past the saturation plateau to the knee, not stop at 95.
        let curve = |q: f32| if q >= 40.0 { 92.0 } else { q as f64 * 2.3 };
        let r = search_target(85.0, &opts, |q| Ok::<_, ()>(curve(q))).unwrap();
        assert!(r.converged, "should reach the band");
        assert!(r.score >= 84.0, "score {} not in band", r.score);
        assert!(
            r.quality < 50.0,
            "did not walk down to the low reaching region from q95 seed: {}",
            r.quality
        );
    }

    // Unreachable target: even at max_quality the score falls short → converged
    // is false and the best (highest) score is reported.
    #[test]
    fn unreachable_target_reports_best_not_converged() {
        let opts = TargetOptions {
            tolerance: 0.5,
            max_encodes: 12,
            ..Default::default()
        };
        // score(q) = q*0.5: max at q100 is 50, target 80 is unreachable.
        let r = search_target(80.0, &opts, |q| Ok::<_, ()>(q as f64 * 0.5)).unwrap();
        assert!(!r.converged);
        assert!(
            (r.quality - 100.0).abs() <= 1.0,
            "should pin at max_quality: {}",
            r.quality
        );
        assert!(r.score <= 50.5 && r.score >= 49.0, "best score {}", r.score);
    }

    // Trivially-reachable target (score always over): converge at the LOWEST
    // quality via downward extrapolation.
    #[test]
    fn always_over_walks_to_min_quality() {
        let opts = TargetOptions {
            min_quality: 5.0,
            tolerance: 0.5,
            max_encodes: 20,
            ..Default::default()
        };
        // score(q) = 95 constant: every q reaches target 60 → lowest = min_quality.
        let r = search_target(60.0, &opts, |_q| Ok::<_, ()>(95.0)).unwrap();
        assert!(r.converged);
        assert!(
            (r.quality - 5.0).abs() <= 1.0,
            "should reach min_quality: {}",
            r.quality
        );
    }

    // The search respects max_encodes (never exceeds the budget).
    #[test]
    fn honors_encode_budget() {
        let opts = TargetOptions {
            tolerance: 0.0, // impossible exact match forces max iterations
            max_encodes: 5,
            ..Default::default()
        };
        let mut calls = 0u32;
        let r = search_target(33.3, &opts, |q| {
            calls += 1;
            Ok::<_, ()>(q as f64)
        })
        .unwrap();
        assert!(calls <= 5, "exceeded encode budget: {calls}");
        assert!(r.encodes <= 5);
    }

    // Error propagation: a failing trial surfaces the error.
    #[test]
    fn propagates_trial_error() {
        let opts = TargetOptions::default();
        let r = search_target(50.0, &opts, |_q| Err::<f64, _>("boom"));
        assert_eq!(r, Err("boom"));
    }

    #[test]
    fn anchor_guess_is_monotone_and_clamped() {
        assert!(anchor_guess(0.0) >= 1.0);
        assert!(anchor_guess(100.0) <= 100.0);
        assert!(anchor_guess(90.0) > anchor_guess(10.0));
    }

    // encode_with_target returns the WINNING iterate's bytes (not a re-encode):
    // each quality's "bytes" are a marker (q as a single byte); the returned
    // data must be the marker for the selected quality.
    #[test]
    fn encode_with_target_returns_winning_bytes() {
        let opts = TargetOptions {
            tolerance: 0.5,
            max_encodes: 12,
            ..Default::default()
        };
        // score(q) = q (identity); target 70 → selected quality ~70.
        let out = encode_with_target(70.0, &opts, |q| {
            let qi = q.round().clamp(0.0, 100.0) as u8;
            Ok::<_, ()>((vec![qi], q as f64))
        })
        .unwrap();
        assert!(out.search.converged);
        assert_eq!(out.data.len(), 1);
        assert_eq!(
            out.data[0],
            out.search.quality.round() as u8,
            "returned bytes are not the selected quality's encode"
        );
        assert!((out.search.quality - 70.0).abs() <= 1.0);
    }

    // encode_with_target propagates a trial error.
    #[test]
    fn encode_with_target_propagates_error() {
        let opts = TargetOptions::default();
        let r = encode_with_target(50.0, &opts, |_q| Err::<(Vec<u8>, f64), _>("enc-fail"));
        assert!(matches!(r, Err("enc-fail")));
    }
}
