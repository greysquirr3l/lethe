//! Conductance retention-adaptive variant.
//!
//! T21 hypothesis: the live DOF on the conductance substrate is the
//! **retention band**, not the coupling band. The T08 `λᵢ` failure (`Δ -58`
//! to `-77` at every `α`) is consistent with the T08 default `eta_lambda=0.03`
//! being over-aggressive for memristive timescales — the DOF itself is
//! correct, the Goldilocks knob just needs to be parameterised at a
//! timescale that lets `lambda_i` find the sliver.
//!
//! This variant is intentionally a separate file from `conductance.rs` so
//! the original conductance adapter remains the control. The two share the
//! conductance physics (weighted-neighbor drive, hyperbolic-tanh coupling,
//! per-cell retention mixing) but differ in their parameterisation of the
//! retention DOF. The `ConductanceSubstrate` default `eta_lambda=0.03` is
//! the T08 fixture value; this variant ships with a smaller default
//! `eta_lambda=0.005` to demonstrate the Goldilocks window on the
//! retention band.
//!
//! Primary Goldilocks knob: `eta_lambda`.
//! Natural DOF: [`DofKind::Retention`].

use lethe_core::{DofKind, NaturalDof, State, Substrate, SubstrateParams};
use rand_chacha::ChaCha8Rng;
use rand_core::RngCore;

pub const CONDUCTANCE_RETENTION_SEED_BASE: u64 = 93_500;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ConductanceRetentionConfig {
    pub size: usize,
    pub coupling_gain: f64,
    pub base_retention: f64,
    pub noise_scale: f64,
    pub activity_clip: f64,
    pub eta_lambda: f64,
    pub lambda_min: f64,
    pub lambda_max: f64,
    pub seed: u64,
}

impl Default for ConductanceRetentionConfig {
    fn default() -> Self {
        Self {
            size: 16,
            coupling_gain: 1.2,
            base_retention: 0.92,
            noise_scale: 0.06,
            activity_clip: 5.0,
            eta_lambda: 0.005,
            lambda_min: 0.5,
            lambda_max: 0.99,
            seed: CONDUCTANCE_RETENTION_SEED_BASE,
        }
    }
}

impl ConductanceRetentionConfig {
    #[must_use]
    pub const fn cell_count(self) -> usize {
        self.size * self.size
    }
}

#[derive(Debug, Clone)]
pub struct ConductanceRetentionSubstrate {
    config: ConductanceRetentionConfig,
    previous_activities: Vec<f64>,
    lambda_i: Vec<f64>,
    scratch: Vec<f64>,
    state: State,
}

impl ConductanceRetentionSubstrate {
    #[must_use]
    pub fn new(config: ConductanceRetentionConfig) -> Self {
        let mut stream = config.seed;
        let cell_count = config.cell_count();

        let mut activities = Vec::with_capacity(cell_count);
        for _ in 0..cell_count {
            activities.push(splitmix_signed_unit(&mut stream) * 0.1);
        }

        let lambda_i = vec![config.base_retention; cell_count];
        let state = State::new(activities.clone(), lambda_i.clone());
        Self {
            config,
            previous_activities: activities,
            lambda_i,
            scratch: vec![0.0; cell_count],
            state,
        }
    }

    const fn neighbors(&self, idx: usize) -> [usize; 4] {
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

    fn neighbor_drive(&self, idx: usize) -> f64 {
        let neighbors = self.neighbors(idx);
        let mut sum = 0.0;
        for neighbor in neighbors {
            sum += self.state.activities.get(neighbor).copied().unwrap_or(0.0);
        }
        self.config.coupling_gain * (sum / 4.0)
    }

    fn update_adaptive_retention(&mut self) {
        let eta = self.config.eta_lambda;
        let min = self.config.lambda_min;
        let max = self.config.lambda_max;
        for idx in 0..self.config.cell_count() {
            let current = self.state.activities.get(idx).copied().unwrap_or(0.0);
            let previous = self.previous_activities.get(idx).copied().unwrap_or(0.0);
            let current_lambda = self
                .lambda_i
                .get(idx)
                .copied()
                .unwrap_or(self.config.base_retention);
            // Persistence signal: tanh of the current×previous product,
            // mapped to [0, 1]. Tanh saturates so the signal is bounded
            // even under spiking transients.
            let persistence = ((current * previous).tanh() + 1.0) * 0.5;
            let updated = eta
                .mul_add(persistence - current_lambda, current_lambda)
                .clamp(min, max);
            if let Some(slot) = self.lambda_i.get_mut(idx) {
                *slot = updated;
            }
        }
    }

    fn refresh_state(&mut self) {
        self.state.activities.clone_from(&self.scratch);
        self.state.lambda_i.clone_from(&self.lambda_i);
    }
}

impl Substrate for ConductanceRetentionSubstrate {
    fn step(&mut self, rng: &mut ChaCha8Rng) -> &State {
        self.previous_activities.clone_from(&self.state.activities);

        for idx in 0..self.config.cell_count() {
            let activity = self.state.activities.get(idx).copied().unwrap_or(0.0);
            let drive = self.neighbor_drive(idx).tanh();
            let noise = self.config.noise_scale * standard_normal(rng);
            let lambda = self
                .lambda_i
                .get(idx)
                .copied()
                .unwrap_or(self.config.base_retention);

            let mixed = lambda.mul_add(activity, (1.0 - lambda) * (drive + noise));
            let clipped = mixed.clamp(-self.config.activity_clip, self.config.activity_clip);
            if let Some(slot) = self.scratch.get_mut(idx) {
                *slot = clipped;
            }
        }

        self.update_adaptive_retention();
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
        self.previous_activities = reset.previous_activities;
        self.lambda_i = reset.lambda_i;
        self.scratch = reset.scratch;
        self.state = reset.state;
    }
}

impl NaturalDof for ConductanceRetentionSubstrate {
    fn natural_dof(&self) -> DofKind {
        DofKind::Retention
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
        CONDUCTANCE_RETENTION_SEED_BASE, ConductanceRetentionConfig, ConductanceRetentionSubstrate,
    };
    use lethe_core::{DofKind, NaturalDof, Substrate};
    use rand_chacha::ChaCha8Rng;
    use rand_core::SeedableRng;

    #[test]
    fn conductance_retention_natural_dof_is_retention() {
        let config = ConductanceRetentionConfig::default();
        let substrate = ConductanceRetentionSubstrate::new(config);
        assert_eq!(substrate.natural_dof(), DofKind::Retention);
    }

    #[test]
    fn conductance_retention_step_is_deterministic_under_fixed_seed() {
        let config = ConductanceRetentionConfig::default();
        let mut sub_a = ConductanceRetentionSubstrate::new(config);
        let mut sub_b = ConductanceRetentionSubstrate::new(config);
        let mut rng_a = ChaCha8Rng::seed_from_u64(CONDUCTANCE_RETENTION_SEED_BASE);
        let mut rng_b = ChaCha8Rng::seed_from_u64(CONDUCTANCE_RETENTION_SEED_BASE);
        for _ in 0..100 {
            sub_a.step(&mut rng_a);
            sub_b.step(&mut rng_b);
        }
        assert_eq!(sub_a.state, sub_b.state);
    }

    #[test]
    fn conductance_retention_config_exposes_eta_lambda_as_primary_knob() {
        // T21 hypothesis: `eta_lambda` is the primary Goldilocks knob.
        // Pin the field's existence and that the default is positive finite.
        let config = ConductanceRetentionConfig::default();
        assert!(config.eta_lambda.is_finite());
        assert!(config.eta_lambda > 0.0);
    }

    #[test]
    fn conductance_retention_default_eta_lambda_is_in_goldilocks_window() {
        // T08 fixture `eta_lambda=0.03` was over-aggressive (Δ -58 to -77).
        // T21 hypothesis: a smaller default sits inside the Goldilocks window.
        // Pin that the default is strictly smaller than the T08 fixture.
        let config = ConductanceRetentionConfig::default();
        assert!(config.eta_lambda < 0.03);
    }

    #[test]
    fn conductance_retention_seed_base_is_distinct_from_conductance_seed_base() {
        // Seed base must not collide with the original conductance adapter's
        // seed base, otherwise cross-experiment re-seeding would not be safe.
        assert_ne!(
            CONDUCTANCE_RETENTION_SEED_BASE,
            super::super::conductance::CONDUCTANCE_SEED_BASE
        );
    }
}
