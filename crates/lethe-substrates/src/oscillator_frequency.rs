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
        OSCILLATOR_FREQUENCY_SEED_BASE, OscillatorFrequencyConfig, OscillatorFrequencySubstrate,
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
}
