use lethe_core::{DofKind, NaturalDof, State, Substrate, SubstrateParams};
use rand_chacha::ChaCha8Rng;
use rand_core::RngCore;

const PI: f64 = std::f64::consts::PI;
const TWO_PI: f64 = std::f64::consts::PI * 2.0;

pub const OSCILLATOR_SEED_BASE: u64 = 92_000;

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum OscillatorPlasticity {
    Fixed,
    Hebbian {
        eta: f64,
        lambda_w: f64,
        w_max: f64,
    },
    AdaptiveLambda {
        eta_lambda: f64,
        lambda_min: f64,
        lambda_max: f64,
    },
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct OscillatorConfig {
    pub size: usize,
    pub coupling: f64,
    pub base_frequency: f64,
    pub frequency_spread: f64,
    pub noise_scale: f64,
    pub phase_memory_lambda: f64,
    pub dt: f64,
    pub plasticity: OscillatorPlasticity,
    pub seed: u64,
}

impl Default for OscillatorConfig {
    fn default() -> Self {
        Self {
            size: 16,
            coupling: 2.2,
            base_frequency: 1.0,
            frequency_spread: 0.25,
            noise_scale: 0.04,
            phase_memory_lambda: 0.93,
            dt: 0.05,
            plasticity: OscillatorPlasticity::Fixed,
            seed: OSCILLATOR_SEED_BASE,
        }
    }
}

impl OscillatorConfig {
    #[must_use]
    pub const fn cell_count(self) -> usize {
        self.size * self.size
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct OscillatorRegression {
    pub ais_threshold: f64,
    pub te_minimum: f64,
    pub tc_cutoff: f64,
}

pub const OSCILLATOR_REGRESSION: OscillatorRegression = OscillatorRegression {
    ais_threshold: 0.01,
    te_minimum: 0.001,
    tc_cutoff: 0.01,
};

#[must_use]
pub fn oscillator_regression_signature_bytes() -> Vec<u8> {
    let mut out = Vec::with_capacity(8 * 4);
    out.extend_from_slice(&OSCILLATOR_REGRESSION.ais_threshold.to_le_bytes());
    out.extend_from_slice(&OSCILLATOR_REGRESSION.te_minimum.to_le_bytes());
    out.extend_from_slice(&OSCILLATOR_REGRESSION.tc_cutoff.to_le_bytes());
    out.extend_from_slice(&OSCILLATOR_SEED_BASE.to_le_bytes());
    out
}

#[derive(Debug, Clone)]
pub struct OscillatorSubstrate {
    config: OscillatorConfig,
    phases: Vec<f64>,
    previous_phases: Vec<f64>,
    memory_state: Vec<f64>,
    lambda_i: Vec<f64>,
    intrinsic_omega: Vec<f64>,
    weights: Vec<f64>,
    phase_scratch: Vec<f64>,
    activity_scratch: Vec<f64>,
    state: State,
}

impl OscillatorSubstrate {
    #[must_use]
    pub fn new(config: OscillatorConfig) -> Self {
        let mut seed_stream = config.seed;
        let cell_count = config.cell_count();

        let mut phases = Vec::with_capacity(cell_count);
        let mut intrinsic_omega = Vec::with_capacity(cell_count);

        for _ in 0..cell_count {
            let phase = splitmix_signed_unit(&mut seed_stream) * PI;
            phases.push(phase);
            let spread = config.frequency_spread * splitmix_signed_unit(&mut seed_stream);
            intrinsic_omega.push(config.base_frequency + spread);
        }

        let memory_state: Vec<f64> = phases.iter().map(|phase| phase.sin()).collect();
        let lambda_i = vec![config.phase_memory_lambda; cell_count];
        let state = State::new(memory_state.clone(), lambda_i.clone());

        Self {
            config,
            previous_phases: phases.clone(),
            phases,
            memory_state,
            lambda_i,
            intrinsic_omega,
            weights: vec![1.0; cell_count * 4],
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
        for (edge, neighbor) in neighbors.iter().enumerate() {
            let theta_j = self.phases.get(*neighbor).copied().unwrap_or(theta_i);
            let base = (theta_j - theta_i).sin();
            let contribution = match self.config.plasticity {
                OscillatorPlasticity::Hebbian { .. } => {
                    let weight = self.weights.get(idx * 4 + edge).copied().unwrap_or(1.0);
                    weight * base
                }
                _ => base,
            };
            sum += contribution;
        }
        self.config.coupling * (sum / 4.0)
    }

    fn update_hebbian_weights(&mut self) {
        let cell_count = self.config.cell_count();
        let OscillatorPlasticity::Hebbian {
            eta,
            lambda_w,
            w_max,
        } = self.config.plasticity
        else {
            return;
        };

        for idx in 0..cell_count {
            let theta_i = self.phases.get(idx).copied().unwrap_or(0.0);
            let neighbors = self.neighbor_indices(idx);
            for (edge, neighbor) in neighbors.iter().enumerate() {
                let theta_j = self.phases.get(*neighbor).copied().unwrap_or(theta_i);
                let lock = (theta_j - theta_i).cos();
                let weight_idx = idx * 4 + edge;
                let weight = self.weights.get(weight_idx).copied().unwrap_or(1.0);
                let updated = eta
                    .mul_add((-lambda_w).mul_add(weight, lock), weight)
                    .clamp(-w_max, w_max);
                if let Some(slot) = self.weights.get_mut(weight_idx) {
                    *slot = updated;
                }
            }
        }
    }

    fn update_adaptive_lambda(&mut self) {
        let OscillatorPlasticity::AdaptiveLambda {
            eta_lambda,
            lambda_min,
            lambda_max,
        } = self.config.plasticity
        else {
            return;
        };

        for idx in 0..self.config.cell_count() {
            let now = self.phases.get(idx).copied().unwrap_or(0.0);
            let previous = self.previous_phases.get(idx).copied().unwrap_or(0.0);
            let current_lambda = self
                .lambda_i
                .get(idx)
                .copied()
                .unwrap_or(self.config.phase_memory_lambda);
            let phase_persistence = ((now - previous).cos() + 1.0) * 0.5;
            let updated = eta_lambda
                .mul_add(phase_persistence - current_lambda, current_lambda)
                .clamp(lambda_min, lambda_max);
            if let Some(slot) = self.lambda_i.get_mut(idx) {
                *slot = updated;
            }
        }
    }

    fn refresh_state(&mut self) {
        self.state.activities.clone_from(&self.activity_scratch);
        self.state.lambda_i.clone_from(&self.lambda_i);
    }

    #[must_use]
    pub fn lambda_std(&self) -> f64 {
        if self.lambda_i.is_empty() {
            return 0.0;
        }

        let count = usize_to_f64(self.lambda_i.len());
        let mean = self.lambda_i.iter().sum::<f64>() / count;
        let variance = self
            .lambda_i
            .iter()
            .map(|value| {
                let diff = *value - mean;
                diff * diff
            })
            .sum::<f64>()
            / count;

        variance.sqrt()
    }
}

impl Substrate for OscillatorSubstrate {
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

            let lambda = match self.config.plasticity {
                OscillatorPlasticity::AdaptiveLambda { .. } => self
                    .lambda_i
                    .get(idx)
                    .copied()
                    .unwrap_or(self.config.phase_memory_lambda),
                _ => self.config.phase_memory_lambda,
            };

            let memory = self.memory_state.get(idx).copied().unwrap_or(0.0);
            let updated_memory = lambda.mul_add(memory, (1.0 - lambda) * next_phase.sin());
            if let Some(slot) = self.memory_state.get_mut(idx) {
                *slot = updated_memory;
            }
            if let Some(slot) = self.activity_scratch.get_mut(idx) {
                *slot = updated_memory;
            }
        }

        self.update_hebbian_weights();
        self.update_adaptive_lambda();
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
        self.lambda_i = reset.lambda_i;
        self.intrinsic_omega = reset.intrinsic_omega;
        self.weights = reset.weights;
        self.phase_scratch = reset.phase_scratch;
        self.activity_scratch = reset.activity_scratch;
        self.state = reset.state;
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

fn usize_to_f64(value: usize) -> f64 {
    let narrowed = u32::try_from(value).unwrap_or(u32::MAX);
    f64::from(narrowed)
}

impl NaturalDof for OscillatorSubstrate {
    fn natural_dof(&self) -> DofKind {
        DofKind::Frequency
    }
}

#[cfg(test)]
mod tests {
    use super::{
        OSCILLATOR_REGRESSION, OSCILLATOR_SEED_BASE, OscillatorConfig, OscillatorPlasticity,
        OscillatorSubstrate, oscillator_regression_signature_bytes,
    };
    use lethe_core::{Observer, ObserverConfig, StateTrace, Substrate};
    use rand_chacha::ChaCha8Rng;
    use rand_core::SeedableRng;

    fn base_config(plasticity: OscillatorPlasticity) -> OscillatorConfig {
        OscillatorConfig {
            size: 8,
            coupling: 2.4,
            base_frequency: 1.0,
            frequency_spread: 0.35,
            noise_scale: 0.05,
            phase_memory_lambda: 0.93,
            dt: 0.05,
            plasticity,
            seed: OSCILLATOR_SEED_BASE,
        }
    }

    #[test]
    fn oscillator_substrate_is_deterministic_under_seeded_rng() {
        let config = base_config(OscillatorPlasticity::Fixed);
        let mut left = OscillatorSubstrate::new(config);
        let mut right = OscillatorSubstrate::new(config);
        let mut rng_left = ChaCha8Rng::seed_from_u64(OSCILLATOR_SEED_BASE + 1);
        let mut rng_right = ChaCha8Rng::seed_from_u64(OSCILLATOR_SEED_BASE + 1);

        for _ in 0..32 {
            let left_state = left.step(&mut rng_left).clone();
            let right_state = right.step(&mut rng_right).clone();
            assert_eq!(left_state, right_state);
        }
    }

    #[test]
    fn adaptive_lambda_creates_nonzero_lambda_dispersion() {
        let config = base_config(OscillatorPlasticity::AdaptiveLambda {
            eta_lambda: 0.04,
            lambda_min: 0.5,
            lambda_max: 0.99,
        });
        let mut substrate = OscillatorSubstrate::new(config);
        let mut rng = ChaCha8Rng::seed_from_u64(OSCILLATOR_SEED_BASE + 2);

        let initial = substrate.lambda_std();
        for _ in 0..300 {
            let _ = substrate.step(&mut rng);
        }
        let final_value = substrate.lambda_std();

        assert!(final_value > initial);
    }

    #[test]
    fn observer_metrics_are_nonzero_for_active_oscillator_region() {
        let config = base_config(OscillatorPlasticity::Fixed);
        let mut substrate = OscillatorSubstrate::new(config);
        let mut rng = ChaCha8Rng::seed_from_u64(OSCILLATOR_SEED_BASE + 3);

        let mut trace = StateTrace::new();
        for tick in 0..240 {
            let state = substrate.step(&mut rng).clone();
            if tick >= 40 {
                trace.push(tick - 40, state);
            }
        }

        let observer = Observer::new(ObserverConfig {
            history_depth: 4,
            bin_count: 8,
            ksg_k: 3,
        });
        let metrics = observer.observe(&trace);

        assert!(metrics.ais_binning >= OSCILLATOR_REGRESSION.ais_threshold);
        assert!(metrics.te >= OSCILLATOR_REGRESSION.te_minimum);
        assert!(metrics.tc >= OSCILLATOR_REGRESSION.tc_cutoff);
    }

    #[test]
    fn oscillator_regression_signature_is_stable() {
        let bytes = oscillator_regression_signature_bytes();
        let expected: Vec<u8> = vec![
            123, 20, 174, 71, 225, 122, 132, 63, 252, 169, 241, 210, 77, 98, 80, 63, 123, 20, 174,
            71, 225, 122, 132, 63, 96, 103, 1, 0, 0, 0, 0, 0,
        ];
        assert_eq!(bytes, expected);
    }

    #[test]
    fn oscillator_natural_dof_is_frequency() {
        use lethe_core::DofKind;
        use lethe_core::NaturalDof;

        let substrate = OscillatorSubstrate::new(OscillatorConfig::default());

        assert_eq!(substrate.natural_dof(), DofKind::Frequency);
    }
}
