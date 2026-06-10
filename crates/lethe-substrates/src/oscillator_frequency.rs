//! Oscillator intrinsic-frequency adaptive variant.
//!
//! T21 hypothesis: the live DOF on the Kuramoto-style oscillator is the
//! **intrinsic frequency band**, not the phase-memory retention band. The
//! T08 `α=0.99` result (`-8.6` to `+0.17`) suggests `λᵢ` is destructive
//! across the swept `α` grid and that frequency adaptation is the natural
//! knob to search.
//!
//! This variant is intentionally a separate file from `oscillator.rs` so
//! the original Kuramoto adapter remains the control. The two share
//! Kuramoto-style stepping but differ in which DOF is plastic.
//! `OscillatorSubstrate` keeps `intrinsic_omega` static; this variant
//! keeps a per-cell plastic `intrinsic_omega: Vec<f64>` driven by
//! `eta_omega`, with `lambda` held as a static scalar.
//!
//! Primary Goldilocks knob: `eta_omega`.
//! Natural DOF: [`DofKind::Frequency`].

use lethe_core::{DofKind, NaturalDof, State, Substrate, SubstrateParams};
use rand_chacha::ChaCha8Rng;
use rand_core::RngCore;

const PI: f64 = std::f64::consts::PI;
const TWO_PI: f64 = std::f64::consts::PI * 2.0;

pub const OSCILLATOR_FREQUENCY_SEED_BASE: u64 = 92_500;

/// T23 Prong A — extended `eta_omega` Goldilocks sweep range.
///
/// The T21 sweep tested `{0.001, 0.005, 0.01, 0.05}` and produced a
/// monotonically destructive live-Δ curve. T23 hypothesis: Kuramoto
/// intrinsic-frequency drift is a slow process in physical
/// oscillators; the T21 sweep may simply have been over-aggressive.
/// The T23 range extends the low end with four micro-rates while
/// retaining all four T21 rates so the regression baseline is
/// preserved.
///
/// Sweep order: monotonically increasing. Used by `lethe-cli pivot`'s
/// `oscillator-frequency` row (see `pivot.rs`).
pub const ETA_OMEGA_SWEEP: &[f64] = &[0.00001, 0.00003, 0.0001, 0.0003, 0.001, 0.003, 0.01, 0.05];

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct OscillatorFrequencyConfig {
    pub size: usize,
    pub base_coupling: f64,
    pub base_omega: f64,
    pub omega_spread: f64,
    pub noise_scale: f64,
    pub phase_memory_lambda: f64,
    pub dt: f64,
    pub eta_omega: f64,
    pub omega_min: f64,
    pub omega_max: f64,
    pub seed: u64,
}

impl Default for OscillatorFrequencyConfig {
    fn default() -> Self {
        Self {
            size: 16,
            base_coupling: 2.2,
            base_omega: 1.0,
            omega_spread: 0.25,
            noise_scale: 0.04,
            phase_memory_lambda: 0.93,
            dt: 0.05,
            eta_omega: 0.01,
            omega_min: 0.5,
            omega_max: 2.0,
            seed: OSCILLATOR_FREQUENCY_SEED_BASE,
        }
    }
}

impl OscillatorFrequencyConfig {
    #[must_use]
    pub const fn cell_count(self) -> usize {
        self.size * self.size
    }
}

#[derive(Debug, Clone)]
pub struct OscillatorFrequencySubstrate {
    config: OscillatorFrequencyConfig,
    phases: Vec<f64>,
    previous_phases: Vec<f64>,
    memory_state: Vec<f64>,
    intrinsic_omega: Vec<f64>,
    phase_scratch: Vec<f64>,
    activity_scratch: Vec<f64>,
    state: State,
}

impl OscillatorFrequencySubstrate {
    #[must_use]
    pub fn new(config: OscillatorFrequencyConfig) -> Self {
        let mut seed_stream = config.seed;
        let cell_count = config.cell_count();

        let mut phases = Vec::with_capacity(cell_count);
        let mut intrinsic_omega = Vec::with_capacity(cell_count);
        for _ in 0..cell_count {
            let phase = splitmix_signed_unit(&mut seed_stream) * PI;
            phases.push(phase);
            let spread = config.omega_spread * splitmix_signed_unit(&mut seed_stream);
            intrinsic_omega
                .push((config.base_omega + spread).clamp(config.omega_min, config.omega_max));
        }

        let memory_state: Vec<f64> = phases.iter().map(|phase| phase.sin()).collect();
        let state = State::new(memory_state.clone(), intrinsic_omega.clone());

        Self {
            config,
            previous_phases: phases.clone(),
            phases,
            memory_state,
            intrinsic_omega,
            phase_scratch: vec![0.0; cell_count],
            activity_scratch: vec![0.0; cell_count],
            state,
        }
    }

    const fn neighbor_indices(&self, idx: usize) -> [usize; 4] {
        let size = self.config.size;
        let row = idx / size;
        let col = idx % size;
        let up = (row + size - 1) % size;
        let down = (row + 1) % size;
        let left = (col + size - 1) % size;
        let right = (col + 1) % size;
        [
            up * size + col,
            down * size + col,
            row * size + left,
            row * size + right,
        ]
    }

    fn coupling_term(&self, idx: usize) -> f64 {
        let theta_i = self.phases.get(idx).copied().unwrap_or(0.0);
        let neighbors = self.neighbor_indices(idx);
        let mut sum = 0.0;
        for neighbor in neighbors {
            let theta_j = self.phases.get(neighbor).copied().unwrap_or(theta_i);
            sum += (theta_j - theta_i).sin();
        }
        self.config.base_coupling * (sum / 4.0)
    }

    fn update_intrinsic_omega(&mut self) {
        let eta = self.config.eta_omega;
        let min = self.config.omega_min;
        let max = self.config.omega_max;
        let n = self.config.cell_count();
        for idx in 0..n {
            let phase = self.phases.get(idx).copied().unwrap_or(0.0);
            let neighbors = self.neighbor_indices(idx);
            let mut neighbor_phase_sum = 0.0;
            for neighbor in neighbors {
                neighbor_phase_sum += self.phases.get(neighbor).copied().unwrap_or(phase);
            }
            let neighbor_phase_mean = neighbor_phase_sum / 4.0;
            // Standard Kuramoto adaptation: shift `omega` to align the
            // cell's phase with the local neighborhood mean. `sin()` keeps
            // the per-step update bounded by `eta_omega` and sign-correct.
            let update = eta * (neighbor_phase_mean - phase).sin();
            let current = self.intrinsic_omega.get(idx).copied().unwrap_or(0.0);
            let updated = (current + update).clamp(min, max);
            if let Some(slot) = self.intrinsic_omega.get_mut(idx) {
                *slot = updated;
            }
        }
    }

    fn refresh_state(&mut self) {
        self.state.activities.clone_from(&self.activity_scratch);
        self.state.lambda_i.clone_from(&self.intrinsic_omega);
    }
}

impl Substrate for OscillatorFrequencySubstrate {
    fn step(&mut self, rng: &mut ChaCha8Rng) -> &State {
        self.previous_phases.clone_from(&self.phases);

        for idx in 0..self.config.cell_count() {
            let phase = self.phases.get(idx).copied().unwrap_or(0.0);
            let omega = self.intrinsic_omega.get(idx).copied().unwrap_or(0.0);
            let coupling = self.coupling_term(idx);
            let noise = self.config.noise_scale * standard_normal(rng);
            let next_phase = wrap_phase((omega + coupling + noise).mul_add(self.config.dt, phase));
            if let Some(slot) = self.phase_scratch.get_mut(idx) {
                *slot = next_phase;
            }
        }

        for idx in 0..self.config.cell_count() {
            let next_phase = self.phase_scratch.get(idx).copied().unwrap_or(0.0);
            if let Some(slot) = self.phases.get_mut(idx) {
                *slot = next_phase;
            }

            let lambda = self.config.phase_memory_lambda;
            let memory = self.memory_state.get(idx).copied().unwrap_or(0.0);
            let updated_memory = lambda.mul_add(memory, (1.0 - lambda) * next_phase.sin());
            if let Some(slot) = self.memory_state.get_mut(idx) {
                *slot = updated_memory;
            }
            if let Some(slot) = self.activity_scratch.get_mut(idx) {
                *slot = updated_memory;
            }
        }

        self.update_intrinsic_omega();
        self.refresh_state();
        &self.state
    }

    fn params(&self) -> SubstrateParams {
        SubstrateParams {
            cell_count: self.config.cell_count(),
        }
    }

    fn reset(&mut self, seed: u64) {
        self.config.seed = seed;
        let reset = Self::new(self.config);
        self.phases = reset.phases;
        self.previous_phases = reset.previous_phases;
        self.memory_state = reset.memory_state;
        self.intrinsic_omega = reset.intrinsic_omega;
        self.phase_scratch = reset.phase_scratch;
        self.activity_scratch = reset.activity_scratch;
        self.state = reset.state;
    }
}

impl NaturalDof for OscillatorFrequencySubstrate {
    fn natural_dof(&self) -> DofKind {
        DofKind::Frequency
    }
}

#[inline]
fn wrap_phase(value: f64) -> f64 {
    (value + PI).rem_euclid(TWO_PI) - PI
}

const fn splitmix64(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

fn splitmix_signed_unit(state: &mut u64) -> f64 {
    let raw = splitmix64(state);
    let top = u32::try_from(raw >> 32).unwrap_or(u32::MAX);
    let unit = f64::from(top) / f64::from(u32::MAX);
    unit.mul_add(2.0, -1.0)
}

fn standard_normal(rng: &mut ChaCha8Rng) -> f64 {
    let u1 = (f64::from(rng.next_u32()) + 1.0) / (f64::from(u32::MAX) + 2.0);
    let u2 = (f64::from(rng.next_u32()) + 1.0) / (f64::from(u32::MAX) + 2.0);
    let radius = (-2.0 * u1.ln()).sqrt();
    let angle = TWO_PI * u2;
    radius * angle.cos()
}

#[cfg(test)]
mod tests {
    use super::{
        ETA_OMEGA_SWEEP, OSCILLATOR_FREQUENCY_SEED_BASE, OscillatorFrequencyConfig,
        OscillatorFrequencySubstrate,
    };
    use lethe_core::{DofKind, NaturalDof, Substrate};
    use rand_chacha::ChaCha8Rng;
    use rand_core::SeedableRng;

    #[test]
    fn oscillator_frequency_natural_dof_is_frequency() {
        let config = OscillatorFrequencyConfig::default();
        let substrate = OscillatorFrequencySubstrate::new(config);
        assert_eq!(substrate.natural_dof(), DofKind::Frequency);
    }

    #[test]
    fn oscillator_frequency_step_is_deterministic_under_fixed_seed() {
        let config = OscillatorFrequencyConfig::default();
        let mut sub_a = OscillatorFrequencySubstrate::new(config);
        let mut sub_b = OscillatorFrequencySubstrate::new(config);
        let mut rng_a = ChaCha8Rng::seed_from_u64(OSCILLATOR_FREQUENCY_SEED_BASE);
        let mut rng_b = ChaCha8Rng::seed_from_u64(OSCILLATOR_FREQUENCY_SEED_BASE);
        for _ in 0..100 {
            sub_a.step(&mut rng_a);
            sub_b.step(&mut rng_b);
        }
        assert_eq!(sub_a.state, sub_b.state);
    }

    #[test]
    fn oscillator_frequency_config_exposes_eta_omega_as_primary_knob() {
        // T21 hypothesis: `eta_omega` is the primary Goldilocks knob.
        // Pin the field's existence and that the default is positive finite.
        let config = OscillatorFrequencyConfig::default();
        assert!(config.eta_omega.is_finite());
        assert!(config.eta_omega > 0.0);
    }

    #[test]
    fn oscillator_frequency_seed_base_is_distinct_from_oscillator_seed_base() {
        // Seed base must not collide with the original oscillator adapter's
        // seed base, otherwise cross-experiment re-seeding would not be safe.
        assert_ne!(
            OSCILLATOR_FREQUENCY_SEED_BASE,
            super::super::oscillator::OSCILLATOR_SEED_BASE
        );
    }

    // ============== T23 Prong A — Extended sweep range ==============
    //
    // A1 (always-on): the sweep-set constant carries the T23-extended
    // range with micro-rates AND retains the four T21 rates so the
    // regression baseline is preserved.
    //
    // A2/A3 (ignored): the T21 replay and non-monotonic / plateau
    // checks. Both depend on external evidence files
    // (`results/t21_pivot_local/t21_pivot_evidence.csv` for the replay
    // and `results/t23_oscillator/t23_pivot_evidence.csv` for the
    // non-monotonic check) and are invoked manually after the
    // evidence is on disk:
    //
    //   cargo test -p lethe-substrates oscillator_frequency -- --ignored
    //
    // See `tasks/T23-oscillator-dof-rescope.md` for the spec and exit
    // criteria.

    #[test]
    fn a1_eta_omega_sweep_includes_micro_rates_and_t21_rates() {
        // The T23 sweep must include the four T21 rates (so the
        // regression baseline is preserved) AND four new micro-rates
        // (so the T21 "monotonically destructive" pattern is
        // falsifiable at a finer resolution).
        let sweep = ETA_OMEGA_SWEEP;
        for t21_rate in [0.001_f64, 0.003_f64, 0.01_f64, 0.05_f64] {
            assert!(
                sweep.contains(&t21_rate),
                "T23 sweep {sweep:?} is missing T21 rate {t21_rate}",
            );
        }
        for micro_rate in [0.00001_f64, 0.00003_f64, 0.0001_f64, 0.0003_f64] {
            assert!(
                sweep.contains(&micro_rate),
                "T23 sweep {sweep:?} is missing micro-rate {micro_rate}",
            );
        }
        // Sweep must be monotonically increasing (so the pivot's
        // formatted CSV rows are in ascending order).
        for window in sweep.windows(2) {
            let [left, right] = window else { continue };
            assert!(
                left < right,
                "T23 sweep is not monotonically increasing at {left} -> {right}",
            );
        }
    }

    /// Inline re-implementation of the pivot's score function: the
    /// sum of `ais_binning + te + tc`. Duplicated here to keep the
    /// substrate crate free of CLI dependencies.
    fn metric_score(metrics: &lethe_core::ObserverMetrics) -> f64 {
        metrics.ais_binning + metrics.te + metrics.tc
    }

    fn collect_metrics<S: Substrate>(
        substrate: &mut S,
        seed: u64,
        burn_in: usize,
        samples: usize,
    ) -> lethe_core::ObserverMetrics {
        let observer = lethe_core::Observer::new(lethe_core::ObserverConfig::default());
        let mut trace = lethe_core::StateTrace::new();
        let mut rng = ChaCha8Rng::seed_from_u64(seed);
        for tick in 0..(burn_in + samples) {
            let state = substrate.step(&mut rng).clone();
            if tick >= burn_in {
                trace.push(tick - burn_in, state);
            }
        }
        observer.observe(&trace)
    }

    /// Re-run the T21 oscillator-frequency row at a single `eta_omega`
    /// value, returning the live Δ relative to the fixed baseline
    /// (the same `OscillatorSubstrate` defaults the T21 row used).
    fn oscillator_frequency_live_delta(eta_omega: f64, burn_in: usize, samples: usize) -> f64 {
        use super::super::oscillator::{
            OSCILLATOR_SEED_BASE, OscillatorConfig, OscillatorPlasticity, OscillatorSubstrate,
        };

        let fixed_config = OscillatorConfig {
            size: 8,
            coupling: 2.4,
            base_frequency: 1.0,
            frequency_spread: 0.35,
            noise_scale: 0.05,
            phase_memory_lambda: 0.93,
            dt: 0.05,
            plasticity: OscillatorPlasticity::Fixed,
            seed: OSCILLATOR_SEED_BASE,
        };
        let fixed_score = metric_score(&collect_metrics(
            &mut OscillatorSubstrate::new(fixed_config),
            OSCILLATOR_SEED_BASE + 100,
            burn_in,
            samples,
        ));

        let live_config = OscillatorFrequencyConfig {
            size: 8,
            base_coupling: 2.4,
            base_omega: 1.0,
            omega_spread: 0.35,
            noise_scale: 0.05,
            phase_memory_lambda: 0.93,
            dt: 0.05,
            eta_omega,
            omega_min: 0.5,
            omega_max: 2.0,
            seed: OSCILLATOR_FREQUENCY_SEED_BASE,
        };
        let live_score = metric_score(&collect_metrics(
            &mut OscillatorFrequencySubstrate::new(live_config),
            OSCILLATOR_FREQUENCY_SEED_BASE + 200,
            burn_in,
            samples,
        ));

        live_score - fixed_score
    }

    /// Load the T21 oscillator row's `live_natural_delta` from the
    /// evidence CSV at `path`, filtered by `target_knob` (the formatted
    /// knob value, e.g. `"0.050000"`). Returns `None` on any I/O or
    /// parse failure; the caller decides whether to fail loud.
    fn load_t21_oscillator_live_delta(path: &str, target_knob: &str) -> Option<f64> {
        use std::fs;
        let body = fs::read_to_string(path).ok()?;
        // CSV columns: substrate,dof_kind,goldilocks_knob,knob_value,
        // fixed_score,dead_score,live_natural_score,dead_delta,
        // live_natural_delta. `knob_value` is formatted `{:.6}` so
        // 0.05 -> "0.050000". The local T21 evidence uses the
        // pre-fix truncated knob name (`eta_om`) — match on
        // `oscillator,` prefix to be agnostic to the knob-name column.
        body.lines()
            .skip(1) // header
            .find(|line| {
                line.starts_with("oscillator,") && line.contains(&format!(",{target_knob},"))
            })
            .and_then(|line| line.split(',').nth(8))
            .and_then(|value| value.parse::<f64>().ok())
    }

    /// **A2 — Replays the T21 oscillator row from the manifest.**
    ///
    /// Reads the T21 oscillator evidence row at `eta_omega=0.05` from
    /// `results/t21_pivot_local/t21_pivot_evidence.csv` (overridable
    /// via `LETHE_T21_EVIDENCE_CSV`), then re-runs the
    /// `OscillatorFrequencySubstrate` at the same `eta_omega` and
    /// asserts the new live Δ matches the T21 baseline within 0.01.
    /// The T21 row is the regression baseline for the extended
    /// sweep; we do not break it.
    #[ignore = "requires T21 evidence CSV; run with --ignored after the local T21 evidence is in place"]
    #[test]
    fn a2_replays_t21_oscillator_row_with_extended_sweep() {
        use std::env;

        // Resolve default path via CARGO_MANIFEST_DIR so the test
        // works regardless of which directory cargo runs from
        // (cargo test sets CWD = the package dir, not the workspace
        // root).
        let path = env::var("LETHE_T21_EVIDENCE_CSV").unwrap_or_else(|_| {
            let manifest_dir = env!("CARGO_MANIFEST_DIR");
            format!("{manifest_dir}/../../results/t21_pivot_local/t21_pivot_evidence.csv")
        });
        let target_knob = "0.050000";
        let Some(t21_live_delta) = load_t21_oscillator_live_delta(&path, target_knob) else {
            eprintln!(
                "T21 oscillator row at knob={target_knob} not found in {path}; set LETHE_T21_EVIDENCE_CSV if the file is at a non-default location",
            );
            std::process::exit(1);
        };

        // Replay uses the full evidence-run sample count (160 samples,
        // 40 burn-in) so the in-source result is representative of
        // what the pivot CLI produces.
        let new_live_delta = oscillator_frequency_live_delta(0.05, 40, 160);
        let drift = (new_live_delta - t21_live_delta).abs();
        assert!(
            drift < 0.01,
            "T23 Prong A regression: new live Δ {new_live_delta:.4} drifted from T21 baseline {t21_live_delta:.4} by {drift:.4} (threshold 0.01)",
        );
    }

    /// **A3 — Non-monotonic / plateau check on the T23 sweep.**
    ///
    /// Reads the T23 oscillator evidence CSV (overridable via
    /// `LETHE_T23_EVIDENCE_CSV`) and asserts the live-Δ curve across
    /// the new 8-point grid is *not* monotonically destructive. The
    /// T21 monotonically-negative pattern is the falsification; the
    /// T23 hypothesis is that the curve has a sign change (some live
    /// Δ ≥ 0) or a plateau (`max − min < 0.5`). The pivot evidence is
    /// the source of truth — the in-source test only checks the
    /// structural property.
    #[ignore = "requires T23 evidence CSV; run with --ignored after the local T23 evidence is in place"]
    #[test]
    fn a3_t23_sweep_is_not_monotonically_destructive() {
        use std::env;

        // Resolve default path via CARGO_MANIFEST_DIR (cargo test
        // sets CWD = the package dir, not the workspace root).
        let path = env::var("LETHE_T23_EVIDENCE_CSV").unwrap_or_else(|_| {
            let manifest_dir = env!("CARGO_MANIFEST_DIR");
            format!("{manifest_dir}/../../results/t23_oscillator/t23_pivot_evidence.csv")
        });

        let Ok(body) = std::fs::read_to_string(&path) else {
            eprintln!(
                "T23 evidence CSV not found at {path}. Set LETHE_T23_EVIDENCE_CSV if the file is at a non-default location.",
            );
            std::process::exit(1);
        };

        // Read the oscillator-frequency rows (NOT the new
        // oscillator-coupling-hebbian rows) from the T23 evidence
        // CSV. Columns: substrate,dof_kind,goldilocks_knob,
        // knob_value,fixed_score,dead_score,live_natural_score,
        // dead_delta,live_natural_delta. A row is part of the
        // oscillator-frequency grid iff substrate == "oscillator" and
        // dof_kind == "frequency".
        let mut live_deltas: Vec<f64> = Vec::new();
        for line in body.lines().skip(1) {
            let cols: Vec<&str> = line.split(',').collect();
            let Some(substrate) = cols.first() else {
                continue;
            };
            let Some(dof_kind) = cols.get(1) else {
                continue;
            };
            if *substrate != "oscillator" || *dof_kind != "frequency" {
                continue;
            }
            let Some(live_s) = cols.get(8) else { continue };
            if let Ok(live_delta) = live_s.parse::<f64>() {
                live_deltas.push(live_delta);
            }
        }

        assert!(
            !live_deltas.is_empty(),
            "T23 oscillator-frequency rows not found in {path}",
        );

        let max = live_deltas
            .iter()
            .copied()
            .fold(f64::NEG_INFINITY, f64::max);
        let min = live_deltas.iter().copied().fold(f64::INFINITY, f64::min);
        let plateau = max - min < 0.5;
        let sign_change = live_deltas.iter().any(|delta| *delta >= 0.0);

        assert!(
            plateau || sign_change,
            "T23 sweep is monotonically destructive (live Δ range [{min:.4}, {max:.4}], {} samples) — T21 pattern is not broken; Falsification: Kuramoto frequency band is not live at any tested rate",
            live_deltas.len(),
        );
    }
}
