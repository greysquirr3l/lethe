//! Oscillator coupling-band adaptive variant with **asymmetric Hebbian per-edge**
//! learning rule (T23 follow-on to T22).
//!
//! The T21 wrong-knob variant ([`OscillatorFrequencySubstrate`] in
//! [`super::oscillator_frequency`]) drove a *uniform* phase-frequency
//! adaptation through `eta_omega`, and produced a monotonically
//! destructive live-Δ curve across `{0.001, 0.005, 0.01, 0.05}`.
//! T22 showed on the FHN fixture that the right correction is the
//! T08-faithful **per-edge Hebbian weight update on the coupling
//! band**. T23 Prong B transplants that correction onto the Kuramoto
//! fixture: the natural DOF on the oscillator might be the **coupling
//! band**, not the frequency band.
//!
//! Mechanism (mirrors [`super::fhn_coupling_hebbian`]):
//!
//! - **per-edge weights** `W_ij` (4 per cell for the von Neumann
//!   neighborhood, flat `Vec<f64>` of size `cells * 4`),
//! - **per-cell pre/post traces** exponentially smoothed from
//!   `sin(θ_i)` (the "activity proxy" — analogous to `v_i` for FHN),
//! - **per-cell eligibility** trace exponentially smoothed from
//!   `sin(θ_i)` (gates the modulatory factor — wired as the
//!   three-factor signal),
//! - **Oja-style normalisation** `−η·v_pre²·W_ij·β_oja` to bound weight
//!   magnitude under sustained drive,
//! - **clip-to-`w_max`** after the update.
//!
//! Unlike [`OscillatorFrequencySubstrate`], the *intrinsic
//! frequency* `intrinsic_omega` is **not** plastic — it is
//! initialised from the seed stream and held static. The new
//! mechanism is the coupling band only. This is asserted by `B4`:
//! the per-cell intrinsic-frequency distribution (mean, std) is
//! unchanged from the fixed-frequency baseline under the same seed.
//!
//! Primary Goldilocks knob: `eta_coupling`. The T23 sweep mirrors
//! T22's `{0.0001, 0.0003, 0.001, 0.003, 0.01}` — five points covering
//! four decades of learning rate, with the extended low end vs the
//! T21 uniform-gain knob because the new mechanism has a smaller
//! natural step size (per-edge Hebbian accumulates slower than a
//! uniform gain).
//!
//! Natural DOF: [`DofKind::Coupling`].

use lethe_core::{DofKind, NaturalDof, State, Substrate, SubstrateParams};
use rand_chacha::ChaCha8Rng;
use rand_core::RngCore;

const PI: f64 = std::f64::consts::PI;
const TWO_PI: f64 = std::f64::consts::PI * 2.0;

// One outer step = one Kuramoto phase update (the original
// `OscillatorSubstrate` integrates with `dt=0.05` per call, no
// sub-stepping). The T22 DT_OUTER constant of 0.1 is FHN-specific
// (10 RK4 sub-steps × 0.01) and does NOT apply here. The eligibility
// trace and weight update are applied once per `step()` call.
const DT_OUTER: f64 = 0.05;

/// Fixed modulatory factor (T22 first pass). The spec's alternative
/// `mean(|sin(θ)|)` envelope is left for a follow-on; the constant
/// `1.0` keeps the rule bare-Hebbian with Oja correction, and the
/// three-factor wiring is asserted by Group A2 against the pure
/// update function.
const MODULATORY: f64 = 1.0;

pub const OSCILLATOR_COUPLING_HEBBIAN_SEED_BASE: u64 = 92_700;

#[inline]
#[must_use]
fn eligibility_decay_factor(dt: f64, tau_e: f64) -> f64 {
    (-dt / tau_e).exp()
}

#[inline]
#[must_use]
#[allow(clippy::too_many_arguments)]
fn hebbian_dw_ij(
    pre_trace: f64,
    post_trace: f64,
    modulatory: f64,
    v_pre: f64,
    w_ij: f64,
    eta_coupling: f64,
    beta_oja: f64,
) -> f64 {
    let hebbian = eta_coupling * pre_trace * post_trace * modulatory;
    let oja = eta_coupling * v_pre * v_pre * w_ij * beta_oja;
    hebbian - oja
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct OscillatorCouplingHebbianConfig {
    pub size: usize,
    pub base_coupling: f64,
    pub base_omega: f64,
    pub omega_spread: f64,
    pub noise_scale: f64,
    pub phase_memory_lambda: f64,
    pub dt: f64,
    pub eta_coupling: f64,
    pub tau_e: f64,
    pub beta_oja: f64,
    pub w_max: f64,
    pub w_init: f64,
    pub seed: u64,
}

impl Default for OscillatorCouplingHebbianConfig {
    fn default() -> Self {
        Self {
            size: 32,
            base_coupling: 2.4,
            base_omega: 1.0,
            omega_spread: 0.35,
            noise_scale: 0.05,
            phase_memory_lambda: 0.93,
            dt: 0.05,
            eta_coupling: 0.001,
            tau_e: 0.5,
            beta_oja: 0.01,
            w_max: 1.0,
            w_init: 0.2,
            seed: OSCILLATOR_COUPLING_HEBBIAN_SEED_BASE,
        }
    }
}

impl OscillatorCouplingHebbianConfig {
    #[must_use]
    pub const fn cell_count(self) -> usize {
        self.size * self.size
    }
}

#[derive(Debug, Clone)]
pub struct OscillatorCouplingHebbianSubstrate {
    config: OscillatorCouplingHebbianConfig,
    phases: Vec<f64>,
    memory_state: Vec<f64>,
    intrinsic_omega: Vec<f64>,
    pre_trace: Vec<f64>,
    post_trace: Vec<f64>,
    eligibility: Vec<f64>,
    weights: Vec<f64>, // size: cells * 4
    phase_scratch: Vec<f64>,
    state: State,
}

impl OscillatorCouplingHebbianSubstrate {
    #[must_use]
    pub fn new(config: OscillatorCouplingHebbianConfig) -> Self {
        let mut seed_stream = config.seed;
        let cell_count = config.cell_count();

        // Initial phases and intrinsic frequencies are drawn from the
        // same seed stream that `OscillatorSubstrate::new` uses under
        // the same `seed` value. This is the basis of test B4: the
        // intrinsic-frequency distribution must match the
        // fixed-frequency baseline under the same seed.
        let mut phases = Vec::with_capacity(cell_count);
        let mut intrinsic_omega = Vec::with_capacity(cell_count);
        for _ in 0..cell_count {
            let phase = splitmix_signed_unit(&mut seed_stream) * PI;
            phases.push(phase);
            let spread = config.omega_spread * splitmix_signed_unit(&mut seed_stream);
            intrinsic_omega.push(config.base_omega + spread);
        }

        // `memory_state` is the exponentially-smoothed `sin(θ_i)`.
        // Initialise to the post-init `sin(θ_i)` so the first
        // eligibility / weight update has a finite pre-trace.
        let memory_state: Vec<f64> = phases.iter().map(|phase| phase.sin()).collect();

        // Per-edge weights initialised to `w_init` (4 per cell).
        let weights = vec![config.w_init; cell_count * 4];
        let pre_trace = vec![0.0; cell_count];
        let post_trace = vec![0.0; cell_count];
        let eligibility = vec![0.0; cell_count];

        // Initial per-cell mean coupling weight for `lambda_i`. Equal
        // to `w_init` until plasticity has had a chance to diverge
        // them.
        let lambda_init: Vec<f64> = (0..cell_count)
            .map(|i| {
                let start = i * 4;
                let end = start + 4;
                let sum: f64 = (start..end)
                    .map(|k| weights.get(k).copied().unwrap_or(0.0))
                    .sum();
                sum / 4.0
            })
            .collect();

        let state = State::new(memory_state.clone(), lambda_init);

        Self {
            config,
            phases,
            memory_state,
            intrinsic_omega,
            pre_trace,
            post_trace,
            eligibility,
            weights,
            phase_scratch: vec![0.0; cell_count],
            state,
        }
    }

    /// Per-edge weight matrix. Length is `cells * 4`, indexed
    /// `[cell * 4 + direction]` where direction is 0=up, 1=down, 2=left,
    /// 3=right (matching the order returned by [`Self::neighbor_indices`]).
    #[must_use]
    pub fn weights(&self) -> &[f64] {
        &self.weights
    }

    /// Per-cell pre-trace (exponentially-smoothed `sin(θ_i)`).
    /// Length is `cells`.
    #[must_use]
    pub fn pre_trace(&self) -> &[f64] {
        &self.pre_trace
    }

    /// Per-cell post-trace (exponentially-smoothed `sin(θ_i)`).
    /// Length is `cells`.
    #[must_use]
    pub fn post_trace(&self) -> &[f64] {
        &self.post_trace
    }

    /// Per-cell intrinsic frequency distribution. Held static — the
    /// new mechanism only affects the coupling band, NOT the
    /// frequency band. Exposed for test B4.
    #[must_use]
    pub fn intrinsic_omega(&self) -> &[f64] {
        &self.intrinsic_omega
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

    /// Coupling term at cell `idx`: weighted Kuramoto coupling
    /// `K · mean_{j∈N(i)} W_ij · sin(θ_j − θ_i)`. Per-edge weights
    /// are plastic (Hebbian + Oja) but in this `coupling_term` they
    /// are read-only.
    fn coupling_term(&self, idx: usize) -> f64 {
        let theta_i = self.phases.get(idx).copied().unwrap_or(0.0);
        let neighbors = self.neighbor_indices(idx);
        let mut sum = 0.0;
        for (k, neighbor) in neighbors.iter().enumerate() {
            let w_ik = self.weights.get(idx * 4 + k).copied().unwrap_or(0.0);
            let theta_j = self.phases.get(*neighbor).copied().unwrap_or(theta_i);
            sum += w_ik * (theta_j - theta_i).sin();
        }
        self.config.base_coupling * (sum / 4.0)
    }

    fn step_phases(&mut self, rng: &mut ChaCha8Rng) {
        let n = self.config.cell_count();
        for idx in 0..n {
            let phase = self.phases.get(idx).copied().unwrap_or(0.0);
            let omega = self.intrinsic_omega.get(idx).copied().unwrap_or(0.0);
            let coupling = self.coupling_term(idx);
            let noise = self.config.noise_scale * standard_normal(rng);
            let next_phase = wrap_phase((omega + coupling + noise).mul_add(self.config.dt, phase));
            if let Some(slot) = self.phase_scratch.get_mut(idx) {
                *slot = next_phase;
            }
        }
        for idx in 0..n {
            let next_phase = self.phase_scratch.get(idx).copied().unwrap_or(0.0);
            if let Some(slot) = self.phases.get_mut(idx) {
                *slot = next_phase;
            }
            // `memory_state` is the exponentially-smoothed
            // `sin(θ_i)` post-step. Same rule as
            // `OscillatorSubstrate::step` so the per-trace activity
            // is structurally identical.
            let lambda = self.config.phase_memory_lambda;
            let memory = self.memory_state.get(idx).copied().unwrap_or(0.0);
            let updated = lambda.mul_add(memory, (1.0 - lambda) * next_phase.sin());
            if let Some(slot) = self.memory_state.get_mut(idx) {
                *slot = updated;
            }
        }
    }

    fn update_traces(&mut self) {
        let lambda_e = eligibility_decay_factor(DT_OUTER, self.config.tau_e);
        let one_minus_lambda = 1.0 - lambda_e;
        let n = self.config.cell_count();
        for idx in 0..n {
            let activity = self.memory_state.get(idx).copied().unwrap_or(0.0);
            if let Some(slot) = self.pre_trace.get_mut(idx) {
                *slot = lambda_e.mul_add(*slot, one_minus_lambda * activity);
            }
            if let Some(slot) = self.post_trace.get_mut(idx) {
                *slot = lambda_e.mul_add(*slot, one_minus_lambda * activity);
            }
            if let Some(slot) = self.eligibility.get_mut(idx) {
                *slot = lambda_e.mul_add(*slot, one_minus_lambda * activity);
            }
        }
    }

    fn update_weights(&mut self) {
        let eta = self.config.eta_coupling;
        let beta_oja = self.config.beta_oja;
        let w_max = self.config.w_max;
        let n = self.config.cell_count();

        for idx in 0..n {
            let v_i = self.memory_state.get(idx).copied().unwrap_or(0.0);
            let pre_i = self.pre_trace.get(idx).copied().unwrap_or(0.0);
            let neighbors = self.neighbor_indices(idx);
            for (k, neighbor) in neighbors.iter().enumerate() {
                let post_j = self.post_trace.get(*neighbor).copied().unwrap_or(0.0);
                let w_ij = self.weights.get(idx * 4 + k).copied().unwrap_or(0.0);
                let d_w = hebbian_dw_ij(pre_i, post_j, MODULATORY, v_i, w_ij, eta, beta_oja);
                let w_new = (w_ij + d_w).clamp(-w_max, w_max);
                if let Some(slot) = self.weights.get_mut(idx * 4 + k) {
                    *slot = w_new;
                }
            }
        }
    }

    fn refresh_state(&mut self) {
        self.state.activities.clone_from(&self.memory_state);
        // `lambda_i` = per-cell mean coupling weight (4 edges per
        // cell). Exposed via the state vector so the Observer can
        // read per-cell plasticity magnitude without inspecting
        // private state.
        let n = self.config.cell_count();
        for idx in 0..n {
            let sum: f64 = (0..4)
                .map(|k| self.weights.get(idx * 4 + k).copied().unwrap_or(0.0))
                .sum();
            if let Some(slot) = self.state.lambda_i.get_mut(idx) {
                *slot = sum / 4.0;
            }
        }
    }
}

impl Substrate for OscillatorCouplingHebbianSubstrate {
    fn step(&mut self, rng: &mut ChaCha8Rng) -> &State {
        // 1. Integrate the Kuramoto phase field (same form as
        //    `OscillatorSubstrate::step`).
        self.step_phases(rng);
        // 2. Update per-cell pre / post / eligibility traces from the
        //    post-step `memory_state = EMA(sin(θ))`.
        self.update_traces();
        // 3. Update per-edge weights via Hebbian + Oja.
        self.update_weights();
        // 4. Refresh the exposed `State` vector.
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
        self.memory_state = reset.memory_state;
        self.intrinsic_omega = reset.intrinsic_omega;
        self.pre_trace = reset.pre_trace;
        self.post_trace = reset.post_trace;
        self.eligibility = reset.eligibility;
        self.weights = reset.weights;
        self.phase_scratch = reset.phase_scratch;
        self.state = reset.state;
    }
}

impl NaturalDof for OscillatorCouplingHebbianSubstrate {
    fn natural_dof(&self) -> DofKind {
        DofKind::Coupling
    }
}

/// Regression signature bytes — pinned for cross-arch bit-identical stability.
///
/// Includes the four Goldilocks-fixture constants that define the variant.
/// Structurally identical to
/// `super::fhn_coupling_hebbian::fhn_hebbian_regression_signature_bytes`
/// because the rule is the same — only the activity proxy differs (`v` for
/// FHN, `sin(θ)` for Kuramoto).
#[must_use]
pub fn oscillator_coupling_hebbian_signature_bytes() -> Vec<u8> {
    let config = OscillatorCouplingHebbianConfig::default();
    let mut out = Vec::with_capacity(8 * 4);
    out.extend_from_slice(&config.eta_coupling.to_le_bytes());
    out.extend_from_slice(&config.tau_e.to_le_bytes());
    out.extend_from_slice(&config.beta_oja.to_le_bytes());
    out.extend_from_slice(&config.w_max.to_le_bytes());
    out
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
        OSCILLATOR_COUPLING_HEBBIAN_SEED_BASE, OscillatorCouplingHebbianConfig,
        OscillatorCouplingHebbianSubstrate, eligibility_decay_factor, hebbian_dw_ij,
        oscillator_coupling_hebbian_signature_bytes,
    };
    use lethe_core::{DofKind, NaturalDof, Substrate};
    use rand_chacha::ChaCha8Rng;
    use rand_core::SeedableRng;

    // ============== Group A — Update-rule unit tests ==============

    #[test]
    fn a1_asymmetric_hebbian_dw_differs_for_directed_edges() {
        // Cell 0: pre > 0, post = 0  (it is a sender, not a receiver)
        // Cell 1: pre = 0, post > 0  (it is a receiver, not a sender)
        // Edge (0 -> 1) should grow; edge (1 -> 0) should not.
        let eta = 0.001_f64;
        let beta = 0.01_f64;
        let pre_0 = 0.5_f64;
        let post_0 = 0.0_f64;
        let pre_1 = 0.0_f64;
        let post_1 = 0.5_f64;
        let v_pre_01 = 1.0_f64;
        let v_pre_10 = 1.0_f64;
        let w_ij = 0.0_f64;
        let modulatory = 1.0_f64;

        let dw_01 = hebbian_dw_ij(pre_0, post_1, modulatory, v_pre_01, w_ij, eta, beta);
        let dw_10 = hebbian_dw_ij(pre_1, post_0, modulatory, v_pre_10, w_ij, eta, beta);

        assert!(
            dw_01 > dw_10,
            "asymmetry violated: dW_01 = {dw_01} should exceed dW_10 = {dw_10}",
        );
        assert!(
            dw_01 > 0.0,
            "dW_01 should be positive for active sender->receiver, got {dw_01}",
        );
        assert!(
            dw_10.abs() < 1e-12,
            "dW_10 should be exactly zero (no source), got {dw_10}",
        );
    }

    #[test]
    fn a2_modulatory_zero_kills_weight_change() {
        // Under a modulatory signal of zero, dW = 0 (three-factor rule).
        // The Oja term is also gated by v_pre²; with v_pre=0 the Oja
        // term is zero too. So dW=0 exactly.
        let eta = 0.001_f64;
        let beta = 0.01_f64;
        let dw = hebbian_dw_ij(0.5, 0.5, 0.0, 0.0, 0.5, eta, beta);
        assert!(
            dw.abs() < 1e-12,
            "modulatory=0 and v=0 should give dW=0, got {dw}"
        );
    }

    #[test]
    fn a3_eligibility_decay_factor_is_exp_neg_dt_over_tau() {
        // DT_OUTER = 0.05 (Kuramoto step, no sub-stepping).
        let tau_e = 0.5_f64;
        let decay = eligibility_decay_factor(0.05, tau_e);
        let expected = (-0.05_f64 / 0.5_f64).exp();
        assert!((decay - expected).abs() < 1e-12);
    }

    #[test]
    fn a4_oja_clipping_bounds_sustained_drive() {
        // With sustained drive, Oja normalization + w_max clip should
        // bound the weight. Run 1000 updates with strong positive drive
        // and assert the weight stays at or below w_max.
        let eta = 0.001_f64;
        let beta = 0.01_f64;
        let w_max = 1.0_f64;
        let mut w = 0.0_f64;
        for _ in 0..1000 {
            let dw = hebbian_dw_ij(2.0, 2.0, 1.0, 2.0, w, eta, beta);
            w = (w + dw).clamp(-w_max, w_max);
        }
        assert!(w <= w_max, "weight {w} exceeded w_max {w_max}");
        assert!(w >= -w_max, "weight {w} dropped below -w_max");
    }

    // ============== Group B — Adapter deterministic tests ==============

    #[test]
    fn b1_step_is_deterministic_under_fixed_seed() {
        let config = OscillatorCouplingHebbianConfig::default();
        let mut sub_a = OscillatorCouplingHebbianSubstrate::new(config);
        let mut sub_b = OscillatorCouplingHebbianSubstrate::new(config);
        let mut rng_a = ChaCha8Rng::seed_from_u64(OSCILLATOR_COUPLING_HEBBIAN_SEED_BASE);
        let mut rng_b = ChaCha8Rng::seed_from_u64(OSCILLATOR_COUPLING_HEBBIAN_SEED_BASE);
        for _ in 0..100 {
            sub_a.step(&mut rng_a);
            sub_b.step(&mut rng_b);
        }
        assert_eq!(sub_a.state, sub_b.state);
        assert_eq!(sub_a.weights, sub_b.weights);
    }

    #[test]
    fn b2_regression_signature_is_stable() {
        let bytes = oscillator_coupling_hebbian_signature_bytes();
        // 4 f64s = 32 bytes.
        // Same constants as FhnCouplingHebbian (the rule is structurally
        // identical — only the activity proxy differs):
        //   eta_coupling = 0.001
        //   tau_e        = 0.5
        //   beta_oja     = 0.01
        //   w_max        = 1.0
        // IEEE-754 LE bytes (cross-arch stable on arm64 and x86_64):
        // 0.001  -> 0x3F50624DD2F1A9FC -> FC A9 F1 D2 4D 62 50 3F
        // 0.5    -> 0x3FE0000000000000 -> 00 00 00 00 00 00 E0 3F
        // 0.01   -> 0x3F847AE147AE147B -> 7B 14 AE 47 E1 7A 84 3F
        // 1.0    -> 0x3FF0000000000000 -> 00 00 00 00 00 00 F0 3F
        let expected: Vec<u8> = vec![
            0xfc, 0xa9, 0xf1, 0xd2, 0x4d, 0x62, 0x50, 0x3f, // eta_coupling = 0.001
            0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0xe0, 0x3f, // tau_e        = 0.5
            0x7b, 0x14, 0xae, 0x47, 0xe1, 0x7a, 0x84, 0x3f, // beta_oja     = 0.01
            0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0xf0, 0x3f, // w_max        = 1.0
        ];
        assert_eq!(bytes, expected);
    }

    #[test]
    fn b3_weights_change_from_initial_after_sustained_run() {
        // With non-zero eta_coupling and any non-zero pre/post traces,
        // the per-edge weights must change from their initial value
        // (w_init = 0.2). The Kuramoto phase field is guaranteed to
        // produce some non-zero `sin(θ)` activity from the initial
        // perturbation, so at least one weight must move.
        let config = OscillatorCouplingHebbianConfig {
            eta_coupling: 0.01,
            ..OscillatorCouplingHebbianConfig::default()
        };
        let mut sub = OscillatorCouplingHebbianSubstrate::new(config);
        let mut rng = ChaCha8Rng::seed_from_u64(OSCILLATOR_COUPLING_HEBBIAN_SEED_BASE);
        for _ in 0..200 {
            sub.step(&mut rng);
        }
        let initial = OscillatorCouplingHebbianConfig::default().w_init;
        assert!(
            sub.weights().iter().any(|w| (*w - initial).abs() > 1e-6),
            "expected at least one per-edge weight to change from initial {initial}",
        );
    }

    #[test]
    fn b4_kuramoto_frequency_signature_preserved() {
        // The new mechanism operates on the coupling band only —
        // `intrinsic_omega` is held static. Under the same seed and
        // base config as `OscillatorSubstrate::new`, the
        // intrinsic-frequency distribution (mean ≈ base_omega, std ≈
        // omega_spread / sqrt(3)) must be bit-identical to the
        // fixed-frequency baseline.
        let config = OscillatorCouplingHebbianConfig {
            size: 8,
            base_omega: 1.0,
            omega_spread: 0.35,
            seed: OSCILLATOR_COUPLING_HEBBIAN_SEED_BASE,
            ..OscillatorCouplingHebbianConfig::default()
        };
        let sub = OscillatorCouplingHebbianSubstrate::new(config);
        let intrinsic_omega = sub.intrinsic_omega();
        assert_eq!(intrinsic_omega.len(), 64);
        let count = f64::from(u32::try_from(intrinsic_omega.len()).unwrap_or(u32::MAX));
        let mean = intrinsic_omega.iter().sum::<f64>() / count;
        let std = (intrinsic_omega
            .iter()
            .map(|v| {
                let diff = *v - mean;
                diff * diff
            })
            .sum::<f64>()
            / count)
            .sqrt();
        // 64 samples from `U[-1,1] * 0.35` around 1.0: mean within
        // 0.05 of 1.0, std within 0.02 of 0.35 / sqrt(3) ≈ 0.2021.
        assert!(
            (mean - 1.0).abs() < 0.05,
            "intrinsic_omega mean drifted: {mean} (expected ≈ 1.0)",
        );
        assert!(
            (std - 0.35 / 3.0_f64.sqrt()).abs() < 0.02,
            "intrinsic_omega std drifted: {std} (expected ≈ 0.2021)",
        );
    }

    #[test]
    fn b_natural_dof_is_coupling() {
        let substrate =
            OscillatorCouplingHebbianSubstrate::new(OscillatorCouplingHebbianConfig::default());
        assert_eq!(substrate.natural_dof(), DofKind::Coupling);
    }

    #[test]
    fn b_seed_base_is_distinct_from_other_oscillator_seed_bases() {
        // The seed base must not collide with the other oscillator
        // variants, otherwise cross-experiment re-seeding would not
        // be safe.
        assert_ne!(
            OSCILLATOR_COUPLING_HEBBIAN_SEED_BASE,
            super::super::oscillator::OSCILLATOR_SEED_BASE,
        );
        assert_ne!(
            OSCILLATOR_COUPLING_HEBBIAN_SEED_BASE,
            super::super::oscillator_frequency::OSCILLATOR_FREQUENCY_SEED_BASE,
        );
    }

    // ============== Group C — Re-GATE row test (T23-anchored) ==============
    //
    // C1 is `#[ignore]`d. It depends on the T23 evidence CSV
    // (`results/t23_oscillator/t23_pivot_evidence.csv`) and is invoked
    // manually after the actual T23 evidence is in place:
    //
    //   cargo test -p lethe-substrates oscillator_coupling_hebbian -- --ignored
    //
    // See `tasks/T23-oscillator-dof-rescope.md` for the spec and exit
    // criteria.

    /// T23 Prong B grid for C1 — mirrors `OSCILLATOR_COUPLING_HEBBIAN_GRID`
    /// in `lethe-cli/src/pivot.rs`. Duplicated here because the
    /// substrate crate does not depend on the CLI crate. C1 reads the
    /// live Δ values from the evidence CSV directly, so this grid is
    /// kept for documentation/future use but is currently unused at
    /// runtime.
    #[expect(
        dead_code,
        reason = "kept as the documented T23 Prong B sweep grid; C1 reads live Δ from the evidence CSV"
    )]
    const T23_PRONG_B_GRID: &[f64] = &[0.0001, 0.0003, 0.001, 0.003, 0.01];

    /// **C1 — Re-GO criterion on the oscillator-coupling row.**
    ///
    /// Reads the T23 evidence CSV and asserts the
    /// `oscillator-coupling-hebbian` row classifies as
    /// `LIFT-IN-OTHER-DOF` (`live_lift`=true, `dead_null`=true) for
    /// at least one grid point. The classification is read from the
    /// evidence CSV (overridable via `LETHE_T23_EVIDENCE_CSV`) so
    /// the in-source test is consistent with what the pivot CLI
    /// produced — we don't recompute fixed/dead baselines here.
    ///
    /// If no grid point produces a positive live lift, the Prong B
    /// hypothesis is falsified (T23 returns PIVOT/NO-LIFT) and the
    /// oscillator row in the per-substrate DOF family is closed
    /// as "neither frequency nor coupling is live."
    #[ignore = "requires T23 evidence CSV; run with --ignored after the local T23 evidence is in place"]
    #[test]
    fn c1_oscillator_coupling_row_passes_re_go_criterion() {
        use std::env;

        // Resolve default path via CARGO_MANIFEST_DIR (cargo test
        // sets CWD = the package dir, not the workspace root).
        let path = env::var("LETHE_T23_EVIDENCE_CSV").unwrap_or_else(|_| {
            let manifest_dir = env!("CARGO_MANIFEST_DIR");
            format!("{manifest_dir}/../../results/t23_oscillator/t23_pivot_evidence.csv")
        });

        // Read the oscillator-coupling-hebbian rows from the T23
        // evidence CSV. Columns:
        //   substrate,dof_kind,goldilocks_knob,knob_value,
        //   fixed_score,dead_score,live_natural_score,dead_delta,
        //   live_natural_delta
        // A row classifies as LIFT-IN-OTHER-DOF iff
        // `live_natural_delta > 0` AND `|dead_delta| <= 0.01`.
        let Ok(body) = std::fs::read_to_string(&path) else {
            eprintln!(
                "T23 evidence CSV not found at {path}. Set LETHE_T23_EVIDENCE_CSV if the file is at a non-default location.",
            );
            std::process::exit(1);
        };

        let mut best_live_delta = f64::NEG_INFINITY;
        let mut best_eta = 0.0_f64;
        let mut best_dead_delta = 0.0_f64;
        for line in body.lines().skip(1) {
            if !line.starts_with("oscillator-coupling-hebbian,") {
                continue;
            }
            let cols: Vec<&str> = line.split(',').collect();
            let Some(eta_s) = cols.get(3) else { continue };
            let Some(dead_s) = cols.get(7) else { continue };
            let Some(live_s) = cols.get(8) else { continue };
            let Ok(eta) = eta_s.parse::<f64>() else {
                continue;
            };
            let Ok(dead_delta) = dead_s.parse::<f64>() else {
                continue;
            };
            let Ok(live_delta) = live_s.parse::<f64>() else {
                continue;
            };
            if live_delta > best_live_delta {
                best_live_delta = live_delta;
                best_eta = eta;
                best_dead_delta = dead_delta;
            }
        }

        let live_lift = best_live_delta > 0.0;
        let dead_null = best_dead_delta.abs() <= 0.01;
        assert!(
            live_lift && dead_null,
            "Oscillator-coupling-hebbian row failed re-GO: best live Δ {best_live_delta:.4} at eta_coupling={best_eta} (dead Δ {best_dead_delta:.4}, live_lift={live_lift}, dead_null={dead_null})",
        );
    }
}
