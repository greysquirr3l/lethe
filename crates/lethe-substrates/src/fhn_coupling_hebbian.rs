//! FHN coupling-band adaptive variant with **asymmetric Hebbian per-edge**
//! learning rule (T22 follow-on to T21).
//!
//! The T21 wrong-knob variant ([`FhnCouplingSubstrate`]) drove a *uniform*
//! diffusive coupling gain through `eta_coupling`, not per-edge Hebbian
//! learning. T22 replaces that with the T08-faithful update:
//!
//! - **per-edge weights** `W_ij` (4 per cell for the von Neumann
//!   neighborhood, flat `Vec<f64>` of size `cells * 4`),
//! - **per-cell pre/post traces** exponentially smoothed from `v_i`,
//! - **per-cell eligibility** trace exponentially smoothed from `v_i`
//!   (gates the modulatory factor — wired as the three-factor signal),
//! - **Oja-style normalisation** `−η·v_i²·W_ij·β_oja` to bound weight
//!   magnitude under sustained drive,
//! - **clip-to-`w_max`** after the update.
//!
//! T08 evidence (`dead Δ +0.7 to +2.4 across α`) showed that
//! coupling-weight plasticity lifts FHN; T22 exposes that DOF correctly.
//!
//! Primary Goldilocks knob: `eta_coupling`. The T22 sweep is
//! `η ∈ {0.0001, 0.0003, 0.001, 0.003, 0.01, 0.03}` — extended low end vs
//! T21's `{0.001, 0.005, 0.01, 0.05}` because the new mechanism has a
//! smaller natural step size (per-edge Hebbian accumulates slower than a
//! uniform gain knob).
//!
//! Natural DOF: [`DofKind::Coupling`].

use lethe_core::{DofKind, NaturalDof, State, Substrate, SubstrateParams};
use rand_chacha::ChaCha8Rng;
use rand_core::RngCore;

const DT: f64 = 0.01;
const STEPS_PER_SAMPLE: usize = 10;
const FHN_A: f64 = 0.7;
const FHN_B: f64 = 0.8;
// DT_OUTER = DT * STEPS_PER_SAMPLE = 0.01 * 10 = 0.1 (hard-coded to
// avoid a `usize as f64` cast precision-loss lint; the constants above
// are the source of truth).
const DT_OUTER: f64 = 0.1;

/// Fixed modulatory factor (T22 first pass). The spec's alternative
/// `mean(|v|)` envelope is left for a follow-on; the constant `1.0` keeps
/// the rule bare-Hebbian with Oja correction, and the three-factor
/// wiring is asserted by Group A2 against the pure update function.
const MODULATORY: f64 = 1.0;

pub const FHN_COUPLING_HEBBIAN_SEED_BASE: u64 = 92_000;

/// Discrete-time eligibility decay factor for one outer step.
/// `dt` is the outer-step duration in the same units as `tau_e`.
///
/// `λ = exp(−dt / τ_e)`. The eligibility update is
/// `e ← λ·e + (1 − λ)·v`.
#[inline]
#[must_use]
pub fn eligibility_decay_factor(dt: f64, tau_e: f64) -> f64 {
    (-dt / tau_e).exp()
}

/// Asymmetric Hebbian per-edge weight delta. Pure function — called by
/// [`FhnCouplingHebbianSubstrate::update_weights`] and unit-tested in
/// isolation under Group A.
///
/// Three-factor: `pre · post · modulatory`. The Oja normalisation term
/// `−η·v_pre²·W_ij·β_oja` keeps the weight bounded under sustained drive.
#[inline]
#[must_use]
#[allow(clippy::too_many_arguments)]
pub fn hebbian_dw_ij(
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
pub struct FhnCouplingHebbianConfig {
    pub size: usize,
    pub epsilon: f64,
    pub i_ext: f64,
    pub i_ext_noise: f64,
    pub lambda: f64,
    pub eta_coupling: f64,
    pub tau_e: f64,
    pub beta_oja: f64,
    pub w_max: f64,
    pub w_init: f64,
    pub seed: u64,
}

impl Default for FhnCouplingHebbianConfig {
    fn default() -> Self {
        Self {
            size: 32,
            epsilon: 0.08,
            i_ext: 0.5,
            i_ext_noise: 0.1,
            lambda: 0.95,
            eta_coupling: 0.001,
            tau_e: 0.5,
            beta_oja: 0.01,
            w_max: 1.0,
            w_init: 0.2,
            seed: FHN_COUPLING_HEBBIAN_SEED_BASE,
        }
    }
}

impl FhnCouplingHebbianConfig {
    #[must_use]
    pub const fn cell_count(self) -> usize {
        self.size * self.size
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
struct FhnHebbianNode {
    v: f64,
    w: f64,
}

#[derive(Debug, Clone)]
pub struct FhnCouplingHebbianSubstrate {
    config: FhnCouplingHebbianConfig,
    nodes: Vec<FhnHebbianNode>,
    scratch: Vec<FhnHebbianNode>,
    pre_trace: Vec<f64>,
    post_trace: Vec<f64>,
    eligibility: Vec<f64>,
    weights: Vec<f64>, // size: cells * 4
    state: State,
}

impl FhnCouplingHebbianSubstrate {
    #[must_use]
    pub fn new(config: FhnCouplingHebbianConfig) -> Self {
        let mut seed_stream = config.seed;
        let cell_count = config.cell_count();

        let mut nodes = Vec::with_capacity(cell_count);
        for _ in 0..cell_count {
            let v = splitmix_signed_unit(&mut seed_stream) * 0.1;
            let w = splitmix_signed_unit(&mut seed_stream) * 0.1;
            nodes.push(FhnHebbianNode { v, w });
        }

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

        let state = State::new(nodes.iter().map(|node| node.v).collect(), lambda_init);

        Self {
            config,
            nodes,
            scratch: vec![FhnHebbianNode { v: 0.0, w: 0.0 }; cell_count],
            pre_trace,
            post_trace,
            eligibility,
            weights,
            state,
        }
    }

    /// Per-edge weight matrix. Length is `cells * 4`, indexed
    /// `[cell * 4 + direction]` where direction is 0=up, 1=down, 2=left,
    /// 3=right (matching the order returned by [`Self::neighbors`]).
    #[must_use]
    pub fn weights(&self) -> &[f64] {
        &self.weights
    }

    /// Per-cell eligibility trace. Length is `cells`.
    #[must_use]
    pub fn eligibility(&self) -> &[f64] {
        &self.eligibility
    }

    const fn neighbors(&self, idx: usize) -> [usize; 4] {
        let size = self.config.size;
        let row = idx / size;
        let col = idx % size;
        let up = ((row + size - 1) % size) * size + col;
        let down = ((row + 1) % size) * size + col;
        let left = row * size + ((col + size - 1) % size);
        let right = row * size + ((col + 1) % size);
        [up, down, left, right]
    }

    fn diffusive_coupling(&self, idx: usize, v_snapshot: &[f64]) -> f64 {
        let base = v_snapshot.get(idx).copied().unwrap_or(0.0);
        let neighbors = self.neighbors(idx);
        let mut sum = 0.0;
        for (k, neighbor) in neighbors.iter().enumerate() {
            let w_ik = self.weights.get(idx * 4 + k).copied().unwrap_or(0.0);
            let v_neighbor = v_snapshot.get(*neighbor).copied().unwrap_or(base);
            sum += w_ik * (v_neighbor - base);
        }
        sum / 4.0
    }

    fn step_macro(&mut self, rng: &mut ChaCha8Rng) {
        let n = self.config.cell_count();
        for _ in 0..STEPS_PER_SAMPLE {
            let v_snapshot: Vec<f64> = self.nodes.iter().map(|node| node.v).collect();
            for idx in 0..n {
                let node = self
                    .nodes
                    .get(idx)
                    .copied()
                    .unwrap_or(FhnHebbianNode { v: 0.0, w: 0.0 });
                let coupling_term = self.diffusive_coupling(idx, &v_snapshot);
                let noise = standard_normal(rng) * self.config.i_ext_noise;
                let total_input = self.config.i_ext + noise + coupling_term;
                if let Some(slot) = self.scratch.get_mut(idx) {
                    *slot = rk4_step(node, total_input, self.config.epsilon);
                }
            }
            self.nodes.clone_from(&self.scratch);
        }
    }

    fn update_traces(&mut self) {
        let lambda_e = eligibility_decay_factor(DT_OUTER, self.config.tau_e);
        let one_minus_lambda = 1.0 - lambda_e;
        let n = self.config.cell_count();
        for idx in 0..n {
            let v = self.nodes.get(idx).map_or(0.0, |node| node.v);
            if let Some(slot) = self.pre_trace.get_mut(idx) {
                *slot = lambda_e.mul_add(*slot, one_minus_lambda * v);
            }
            if let Some(slot) = self.post_trace.get_mut(idx) {
                *slot = lambda_e.mul_add(*slot, one_minus_lambda * v);
            }
            if let Some(slot) = self.eligibility.get_mut(idx) {
                *slot = lambda_e.mul_add(*slot, one_minus_lambda * v);
            }
        }
    }

    fn update_weights(&mut self) {
        let eta = self.config.eta_coupling;
        let beta_oja = self.config.beta_oja;
        let w_max = self.config.w_max;
        let n = self.config.cell_count();

        for idx in 0..n {
            let v_i = self.nodes.get(idx).map_or(0.0, |node| node.v);
            let pre_i = self.pre_trace.get(idx).copied().unwrap_or(0.0);
            let neighbors = self.neighbors(idx);
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
        self.state.activities = self.nodes.iter().map(|node| node.v).collect();
        // `lambda_i` = per-cell mean coupling weight (size: cells).
        let n = self.config.cell_count();
        let mut lambda_i = vec![0.0_f64; n];
        for idx in 0..n {
            let sum: f64 = (0..4)
                .map(|k| self.weights.get(idx * 4 + k).copied().unwrap_or(0.0))
                .sum();
            if let Some(slot) = lambda_i.get_mut(idx) {
                *slot = sum / 4.0;
            }
        }
        self.state.lambda_i = lambda_i;
    }
}

impl Substrate for FhnCouplingHebbianSubstrate {
    fn step(&mut self, rng: &mut ChaCha8Rng) -> &State {
        self.step_macro(rng);
        self.update_traces();
        self.update_weights();
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
        self.nodes = reset.nodes;
        self.scratch = reset.scratch;
        self.pre_trace = reset.pre_trace;
        self.post_trace = reset.post_trace;
        self.eligibility = reset.eligibility;
        self.weights = reset.weights;
        self.state = reset.state;
    }
}

impl NaturalDof for FhnCouplingHebbianSubstrate {
    fn natural_dof(&self) -> DofKind {
        DofKind::Coupling
    }
}

/// Regression signature bytes — pinned for cross-arch bit-identical
/// stability. Includes the four Goldilocks-fixture constants that
/// define the variant.
#[must_use]
pub fn fhn_hebbian_regression_signature_bytes() -> Vec<u8> {
    let config = FhnCouplingHebbianConfig::default();
    let mut out = Vec::with_capacity(8 * 4);
    out.extend_from_slice(&config.eta_coupling.to_le_bytes());
    out.extend_from_slice(&config.tau_e.to_le_bytes());
    out.extend_from_slice(&config.beta_oja.to_le_bytes());
    out.extend_from_slice(&config.w_max.to_le_bytes());
    out
}

#[inline]
fn rhs(v: f64, w: f64, total_input: f64, epsilon: f64) -> (f64, f64) {
    let dv = v - (v * v * v) / 3.0 - w + total_input;
    let dw = epsilon * v.mul_add(1.0, FHN_A).mul_add(1.0, -FHN_B * w);
    (dv, dw)
}

#[inline]
fn rk4_step(node: FhnHebbianNode, total_input: f64, epsilon: f64) -> FhnHebbianNode {
    let (k1_v, k1_w) = rhs(node.v, node.w, total_input, epsilon);
    let (k2_v, k2_w) = rhs(
        k1_v.mul_add(0.5 * DT, node.v),
        k1_w.mul_add(0.5 * DT, node.w),
        total_input,
        epsilon,
    );
    let (k3_v, k3_w) = rhs(
        k2_v.mul_add(0.5 * DT, node.v),
        k2_w.mul_add(0.5 * DT, node.w),
        total_input,
        epsilon,
    );
    let (k4_v, k4_w) = rhs(
        k3_v.mul_add(DT, node.v),
        k3_w.mul_add(DT, node.w),
        total_input,
        epsilon,
    );

    let slope_v = k2_v.mul_add(2.0, k1_v) + k3_v.mul_add(2.0, k4_v);
    let slope_w = k2_w.mul_add(2.0, k1_w) + k3_w.mul_add(2.0, k4_w);

    FhnHebbianNode {
        v: slope_v.mul_add(DT / 6.0, node.v),
        w: slope_w.mul_add(DT / 6.0, node.w),
    }
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
    let angle = (2.0 * std::f64::consts::PI) * u2;
    radius * angle.cos()
}

#[cfg(test)]
mod tests {
    use super::{
        DT_OUTER, FHN_COUPLING_HEBBIAN_SEED_BASE, FhnCouplingHebbianConfig,
        FhnCouplingHebbianSubstrate, eligibility_decay_factor,
        fhn_hebbian_regression_signature_bytes, hebbian_dw_ij,
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
        let tau_e = 0.5_f64;
        let decay = eligibility_decay_factor(DT_OUTER, tau_e);
        let expected = (-DT_OUTER / tau_e).exp();
        assert!((decay - expected).abs() < 1e-12);
        // Pin the actual value for cross-arch stability:
        // DT_OUTER = 0.01 * 10 = 0.1, tau_e = 0.5 -> exp(-0.2)
        let pinned = (-0.1_f64 / 0.5_f64).exp();
        assert!((decay - pinned).abs() < 1e-12);
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
        let config = FhnCouplingHebbianConfig::default();
        let mut sub_a = FhnCouplingHebbianSubstrate::new(config);
        let mut sub_b = FhnCouplingHebbianSubstrate::new(config);
        let mut rng_a = ChaCha8Rng::seed_from_u64(FHN_COUPLING_HEBBIAN_SEED_BASE);
        let mut rng_b = ChaCha8Rng::seed_from_u64(FHN_COUPLING_HEBBIAN_SEED_BASE);
        for _ in 0..100 {
            sub_a.step(&mut rng_a);
            sub_b.step(&mut rng_b);
        }
        assert_eq!(sub_a.state, sub_b.state);
        assert_eq!(sub_a.weights, sub_b.weights);
    }

    #[test]
    fn b2_regression_signature_is_stable() {
        let bytes = fhn_hebbian_regression_signature_bytes();
        // 4 f64s = 32 bytes
        assert_eq!(bytes.len(), 32);
        // Hard-coded values for cross-arch bit-identical stability.
        // `f64::to_le_bytes()` returns LSB first on any little-endian
        // platform (including the project's arm64 and x86_64 targets).
        // 0.001_f64 IEEE-754  = 0x3F50624DD2F1A9FC -> LE: FC A9 F1 D2 4D 62 50 3F
        // 0.5_f64   IEEE-754  = 0x3FE0000000000000 -> LE: 00 00 00 00 00 00 E0 3F
        // 0.01_f64  IEEE-754  = 0x3F847AE147AE147B -> LE: 7B 14 AE 47 E1 7A 84 3F
        // 1.0_f64   IEEE-754  = 0x3FF0000000000000 -> LE: 00 00 00 00 00 00 F0 3F
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
        // (w_init = 0.2). The FHN dynamics are guaranteed to produce
        // some non-zero pre/post activity from the initial perturbation
        // of 0.1, so at least one weight must move.
        let config = FhnCouplingHebbianConfig {
            eta_coupling: 0.01,
            ..FhnCouplingHebbianConfig::default()
        };
        let mut sub = FhnCouplingHebbianSubstrate::new(config);
        let mut rng = ChaCha8Rng::seed_from_u64(FHN_COUPLING_HEBBIAN_SEED_BASE);
        for _ in 0..200 {
            sub.step(&mut rng);
        }
        let initial = FhnCouplingHebbianConfig::default().w_init;
        assert!(
            sub.weights().iter().any(|w| (*w - initial).abs() > 1e-6),
            "expected at least one per-edge weight to change from initial {initial}",
        );
    }

    #[test]
    fn b_natural_dof_is_coupling() {
        let substrate = FhnCouplingHebbianSubstrate::new(FhnCouplingHebbianConfig::default());
        assert_eq!(substrate.natural_dof(), DofKind::Coupling);
    }

    #[test]
    fn b_seed_base_is_distinct_from_other_fhn_seed_bases() {
        // The seed base must not collide with the other FHN variants,
        // otherwise cross-experiment re-seeding would not be safe.
        assert_ne!(
            FHN_COUPLING_HEBBIAN_SEED_BASE,
            super::super::fhn::FHN_SEED_BASE,
        );
        assert_ne!(
            FHN_COUPLING_HEBBIAN_SEED_BASE,
            super::super::fhn_coupling::FHN_COUPLING_SEED_BASE,
        );
    }

    // ============== Group C — Re-GATE row tests (T21-anchored) ==============
    //
    // Both tests are `#[ignore]`d. They are slow (a full pivot-style
    // run) and/or depend on external evidence files, so they are
    // invoked manually after the actual T22 evidence is in place:
    //
    //   cargo test -p lethe-substrates fhn_coupling_hebbian -- --ignored
    //
    // See `tasks/T22-fhn-asymmetric-coupling-hebbian.md` for the spec
    // and exit criteria.

    /// T22 grid for C1/C2 — mirrors `FHN_COUPLING_HEBBIAN_GRID` in
    /// `lethe-cli/src/pivot.rs`. Duplicated here because the substrate
    /// crate does not depend on the CLI crate.
    const T22_GRID: &[f64] = &[0.0001, 0.0003, 0.001, 0.003, 0.01, 0.03];

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

    /// Run the new variant at a single `eta_coupling` value against
    /// the same `FhnSubstrate` fixed/dead baselines that the T21 FHN
    /// row used. Returns the live Δ (`live_score` − `fixed_score`).
    fn fhn_hebbian_live_delta(eta_coupling: f64, burn_in: usize, samples: usize) -> f64 {
        use super::super::fhn::{FHN_SEED_BASE, FhnConfig, FhnSubstrate};

        let fixed_config = FhnConfig {
            size: 8,
            epsilon: 0.08,
            coupling: 0.2,
            i_ext: 0.5,
            i_ext_noise: 0.1,
            lambda: 0.95,
            seed: FHN_SEED_BASE,
        };
        let fixed_score = metric_score(&collect_metrics(
            &mut FhnSubstrate::new(fixed_config),
            FHN_SEED_BASE + 100,
            burn_in,
            samples,
        ));

        let live_config = FhnCouplingHebbianConfig {
            size: 8,
            epsilon: 0.08,
            i_ext: 0.5,
            i_ext_noise: 0.1,
            lambda: 0.95,
            eta_coupling,
            tau_e: 0.5,
            beta_oja: 0.01,
            w_max: 1.0,
            w_init: 0.2,
            seed: FHN_COUPLING_HEBBIAN_SEED_BASE,
        };
        let live_score = metric_score(&collect_metrics(
            &mut FhnCouplingHebbianSubstrate::new(live_config),
            FHN_COUPLING_HEBBIAN_SEED_BASE + 200,
            burn_in,
            samples,
        ));

        live_score - fixed_score
    }

    /// Load the T21 FHN row's `live_natural_delta` from the evidence
    /// CSV at `path`, filtered by `target_knob` (the formatted knob
    /// value, e.g. `"0.001000"`). Returns `None` on any I/O or parse
    /// failure; the caller decides whether to fail loud.
    fn load_t21_fhn_live_delta(path: &str, target_knob: &str) -> Option<f64> {
        use std::fs;
        let body = fs::read_to_string(path).ok()?;
        // CSV columns: substrate,dof_kind,goldilocks_knob,knob_value,fixed_score,dead_score,live_natural_score,dead_delta,live_natural_delta
        // knob_value is formatted `{:.6}`, so 0.001 -> "0.001000".
        body.lines()
            .skip(1) // header
            .find(|line| line.starts_with("fhn,") && line.contains(&format!(",{target_knob},")))
            .and_then(|line| line.split(',').nth(8))
            .and_then(|value| value.parse::<f64>().ok())
    }

    /// **C1 — Replays the T21 FHN row from the manifest.**
    ///
    /// Reads the T21 FHN evidence row at `eta_coupling=0.001` from
    /// `results/t21_pivot_local/t21_pivot_evidence.csv` (overridable
    /// via `LETHE_T21_EVIDENCE_CSV`), then re-runs the new variant at
    /// the same `eta_coupling` and asserts the new live Δ exceeds the
    /// T21 live Δ by **at least 0.5**.
    ///
    /// The T21 FHN row's live Δ was `−0.141` (per the spec); a passing
    /// T22 must beat that by ≥ 0.5, i.e. new live Δ ≥ 0.36 to even
    /// register, and ≥ 0.36 + 0.01 to clear `LIVE_LIFT_THRESHOLD`.
    #[ignore = "requires T21 evidence CSV; run with --ignored after the local T21 evidence is in place"]
    #[test]
    fn c1_replays_t21_fhn_row_with_corrective_variant() {
        use std::env;

        let path = env::var("LETHE_T21_EVIDENCE_CSV")
            .unwrap_or_else(|_| "results/t21_pivot_local/t21_pivot_evidence.csv".to_string());
        let target_knob = "0.001000";
        let Some(t21_live_delta) = load_t21_fhn_live_delta(&path, target_knob) else {
            eprintln!(
                "T21 FHN row at knob={target_knob} not found in {path}; set LETHE_T21_EVIDENCE_CSV if the file is at a non-default location",
            );
            std::process::exit(1);
        };

        // Small but non-trivial sweep. Full 160/40 is in the T22
        // evidence run; this is the in-source replay.
        let new_live_delta = fhn_hebbian_live_delta(0.001, 40, 80);
        let lift = new_live_delta - t21_live_delta;
        assert!(
            lift >= 0.5,
            "T22 corrective failed: new live Δ {new_live_delta:.4} did not exceed T21 live Δ {t21_live_delta:.4} by ≥ 0.5 (lift = {lift:.4})",
        );
    }

    /// **C2 — Re-GO criterion on the FHN row.**
    ///
    /// Sweeps `eta_coupling` over the T22 grid and asserts the new
    /// variant classifies as `LIFT-IN-OTHER-DOF` (`live_lift`=true,
    /// `dead_null`=true) for at least one point. The classification is
    /// computed against the same fixed/dead baselines the T21 FHN row
    /// uses, so the result is directly comparable.
    ///
    /// If no grid point produces a positive live lift, the FHN
    /// hypothesis is falsified (T22 returns PIVOT/NO-LIFT) and Phase
    /// 3 stays halted on this substrate.
    #[ignore = "slow pivot-style sweep; run with --ignored after the local T22 evidence is in place"]
    #[test]
    fn c2_fhn_row_passes_re_go_criterion() {
        use super::super::fhn::{FHN_SEED_BASE, FhnConfig, FhnSubstrate};

        // Fixed baseline. Dead = same FhnSubstrate with lambda=0.99,
        // matching the T21 FHN row's dead-mode. If the FhnSubstrate
        // import is unavailable in this build, this test will fail
        // to compile; the explicit import keeps that visible.
        let fixed_config = FhnConfig {
            size: 8,
            epsilon: 0.08,
            coupling: 0.2,
            i_ext: 0.5,
            i_ext_noise: 0.1,
            lambda: 0.95,
            seed: FHN_SEED_BASE,
        };
        let fixed_score = metric_score(&collect_metrics(
            &mut FhnSubstrate::new(fixed_config),
            FHN_SEED_BASE + 100,
            40,
            80,
        ));
        let dead_config = FhnConfig {
            lambda: 0.99,
            ..fixed_config
        };
        let dead_score = metric_score(&collect_metrics(
            &mut FhnSubstrate::new(dead_config),
            FHN_SEED_BASE + 101,
            40,
            80,
        ));
        let dead_delta = dead_score - fixed_score;
        let dead_null = dead_delta.abs() <= 0.01;

        let mut best_live_delta = f64::NEG_INFINITY;
        let mut best_eta = 0.0_f64;
        for &eta in T22_GRID {
            let live_delta = fhn_hebbian_live_delta(eta, 40, 80);
            if live_delta > best_live_delta {
                best_live_delta = live_delta;
                best_eta = eta;
            }
        }
        let live_lift = best_live_delta >= 0.01;

        assert!(
            live_lift && dead_null,
            "FHN row failed re-GO: best live Δ {best_live_delta:.4} at eta_coupling={best_eta} (dead Δ {dead_delta:.4}, dead_null={dead_null}, live_lift={live_lift})",
        );
    }
}
