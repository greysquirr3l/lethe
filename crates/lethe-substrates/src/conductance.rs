use lethe_core::{State, Substrate, SubstrateParams};
use rand_chacha::ChaCha8Rng;
use rand_core::RngCore;

const TWO_PI: f64 = std::f64::consts::PI * 2.0;

pub const CONDUCTANCE_SEED_BASE: u64 = 93_000;

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ConductancePlasticity {
    Fixed,
    HebbianConductance {
        eta: f64,
        leak: f64,
        min_weight: f64,
        max_weight: f64,
    },
    AdaptiveRetention {
        eta_lambda: f64,
        lambda_min: f64,
        lambda_max: f64,
    },
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ConductanceConfig {
    pub size: usize,
    pub coupling_gain: f64,
    pub base_retention: f64,
    pub noise_scale: f64,
    pub activity_clip: f64,
    pub plasticity: ConductancePlasticity,
    pub seed: u64,
}

impl Default for ConductanceConfig {
    fn default() -> Self {
        Self {
            size: 16,
            coupling_gain: 1.2,
            base_retention: 0.92,
            noise_scale: 0.06,
            activity_clip: 5.0,
            plasticity: ConductancePlasticity::Fixed,
            seed: CONDUCTANCE_SEED_BASE,
        }
    }
}

impl ConductanceConfig {
    #[must_use]
    pub const fn cell_count(self) -> usize {
        self.size * self.size
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ConductanceRegression {
    pub ais_threshold: f64,
    pub te_minimum: f64,
    pub tc_cutoff: f64,
}

pub const CONDUCTANCE_REGRESSION: ConductanceRegression = ConductanceRegression {
    ais_threshold: 0.01,
    te_minimum: 0.001,
    tc_cutoff: 0.01,
};

#[must_use]
pub fn conductance_regression_signature_bytes() -> Vec<u8> {
    let mut out = Vec::with_capacity(8 * 4);
    out.extend_from_slice(&CONDUCTANCE_REGRESSION.ais_threshold.to_le_bytes());
    out.extend_from_slice(&CONDUCTANCE_REGRESSION.te_minimum.to_le_bytes());
    out.extend_from_slice(&CONDUCTANCE_REGRESSION.tc_cutoff.to_le_bytes());
    out.extend_from_slice(&CONDUCTANCE_SEED_BASE.to_le_bytes());
    out
}

#[derive(Debug, Clone)]
pub struct ConductanceSubstrate {
    config: ConductanceConfig,
    previous_activities: Vec<f64>,
    lambda_i: Vec<f64>,
    edge_weights: Vec<f64>,
    scratch: Vec<f64>,
    state: State,
}

impl ConductanceSubstrate {
    #[must_use]
    pub fn new(config: ConductanceConfig) -> Self {
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
            edge_weights: vec![1.0; cell_count * 4],
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

    fn weighted_neighbor_drive(&self, idx: usize) -> f64 {
        let neighbors = self.neighbors(idx);
        let mut sum = 0.0;
        for (edge, neighbor) in neighbors.iter().enumerate() {
            let activity_neighbor = self.state.activities.get(*neighbor).copied().unwrap_or(0.0);
            let weighted = match self.config.plasticity {
                ConductancePlasticity::HebbianConductance { .. } => {
                    let w = self
                        .edge_weights
                        .get(idx * 4 + edge)
                        .copied()
                        .unwrap_or(1.0);
                    w * activity_neighbor
                }
                _ => activity_neighbor,
            };
            sum += weighted;
        }
        self.config.coupling_gain * (sum / 4.0)
    }

    fn update_hebbian_conductance(&mut self) {
        let ConductancePlasticity::HebbianConductance {
            eta,
            leak,
            min_weight,
            max_weight,
        } = self.config.plasticity
        else {
            return;
        };

        for idx in 0..self.config.cell_count() {
            let activity_i = self.state.activities.get(idx).copied().unwrap_or(0.0);
            let neighbors = self.neighbors(idx);
            for (edge, neighbor) in neighbors.iter().enumerate() {
                let activity_j = self.state.activities.get(*neighbor).copied().unwrap_or(0.0);
                let weight_idx = idx * 4 + edge;
                let weight = self.edge_weights.get(weight_idx).copied().unwrap_or(1.0);
                let updated = eta
                    .mul_add((-leak).mul_add(weight, activity_i * activity_j), weight)
                    .clamp(min_weight, max_weight);
                if let Some(slot) = self.edge_weights.get_mut(weight_idx) {
                    *slot = updated;
                }
            }
        }
    }

    fn update_adaptive_retention(&mut self) {
        let ConductancePlasticity::AdaptiveRetention {
            eta_lambda,
            lambda_min,
            lambda_max,
        } = self.config.plasticity
        else {
            return;
        };

        for idx in 0..self.config.cell_count() {
            let current = self.state.activities.get(idx).copied().unwrap_or(0.0);
            let previous = self.previous_activities.get(idx).copied().unwrap_or(0.0);
            let current_lambda = self
                .lambda_i
                .get(idx)
                .copied()
                .unwrap_or(self.config.base_retention);
            let persistence = ((current * previous).tanh() + 1.0) * 0.5;
            let updated = eta_lambda
                .mul_add(persistence - current_lambda, current_lambda)
                .clamp(lambda_min, lambda_max);
            if let Some(slot) = self.lambda_i.get_mut(idx) {
                *slot = updated;
            }
        }
    }

    fn refresh_state(&mut self) {
        self.state.activities.clone_from(&self.scratch);
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

impl Substrate for ConductanceSubstrate {
    fn step(&mut self, rng: &mut ChaCha8Rng) -> &State {
        self.previous_activities.clone_from(&self.state.activities);

        for idx in 0..self.config.cell_count() {
            let activity = self.state.activities.get(idx).copied().unwrap_or(0.0);
            let drive = self.weighted_neighbor_drive(idx).tanh();
            let noise = self.config.noise_scale * standard_normal(rng);
            let lambda = match self.config.plasticity {
                ConductancePlasticity::AdaptiveRetention { .. } => self
                    .lambda_i
                    .get(idx)
                    .copied()
                    .unwrap_or(self.config.base_retention),
                _ => self.config.base_retention,
            };

            let mixed = lambda.mul_add(activity, (1.0 - lambda) * (drive + noise));
            let clipped = mixed.clamp(-self.config.activity_clip, self.config.activity_clip);
            if let Some(slot) = self.scratch.get_mut(idx) {
                *slot = clipped;
            }
        }

        self.update_hebbian_conductance();
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
        self.edge_weights = reset.edge_weights;
        self.scratch = reset.scratch;
        self.state = reset.state;
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
    let angle = TWO_PI * u2;
    radius * angle.cos()
}

fn usize_to_f64(value: usize) -> f64 {
    let narrowed = u32::try_from(value).unwrap_or(u32::MAX);
    f64::from(narrowed)
}

#[cfg(test)]
mod tests {
    use super::{
        CONDUCTANCE_REGRESSION, CONDUCTANCE_SEED_BASE, ConductanceConfig, ConductancePlasticity,
        ConductanceSubstrate, conductance_regression_signature_bytes,
    };
    use lethe_core::{Observer, ObserverConfig, StateTrace, Substrate};
    use rand_chacha::ChaCha8Rng;
    use rand_core::SeedableRng;

    fn base_config(plasticity: ConductancePlasticity) -> ConductanceConfig {
        ConductanceConfig {
            size: 8,
            coupling_gain: 1.35,
            base_retention: 0.92,
            noise_scale: 0.05,
            activity_clip: 4.0,
            plasticity,
            seed: CONDUCTANCE_SEED_BASE,
        }
    }

    #[test]
    fn conductance_substrate_is_deterministic_under_seeded_rng() {
        let config = base_config(ConductancePlasticity::Fixed);
        let mut left = ConductanceSubstrate::new(config);
        let mut right = ConductanceSubstrate::new(config);
        let mut rng_left = ChaCha8Rng::seed_from_u64(CONDUCTANCE_SEED_BASE + 1);
        let mut rng_right = ChaCha8Rng::seed_from_u64(CONDUCTANCE_SEED_BASE + 1);

        for _ in 0..32 {
            let left_state = left.step(&mut rng_left).clone();
            let right_state = right.step(&mut rng_right).clone();
            assert_eq!(left_state, right_state);
        }
    }

    #[test]
    fn adaptive_retention_produces_lambda_dispersion() {
        let config = base_config(ConductancePlasticity::AdaptiveRetention {
            eta_lambda: 0.03,
            lambda_min: 0.5,
            lambda_max: 0.99,
        });
        let mut substrate = ConductanceSubstrate::new(config);
        let mut rng = ChaCha8Rng::seed_from_u64(CONDUCTANCE_SEED_BASE + 2);

        let start = substrate.lambda_std();
        for _ in 0..300 {
            let _ = substrate.step(&mut rng);
        }
        let end = substrate.lambda_std();

        assert!(end > start);
    }

    #[test]
    fn observer_metrics_are_nonzero_for_active_conductance_region() {
        let config = base_config(ConductancePlasticity::HebbianConductance {
            eta: 0.02,
            leak: 0.01,
            min_weight: 0.4,
            max_weight: 1.6,
        });
        let mut substrate = ConductanceSubstrate::new(config);
        let mut rng = ChaCha8Rng::seed_from_u64(CONDUCTANCE_SEED_BASE + 3);

        let mut trace = StateTrace::new();
        for tick in 0..260 {
            let state = substrate.step(&mut rng).clone();
            if tick >= 60 {
                trace.push(tick - 60, state);
            }
        }

        let observer = Observer::new(ObserverConfig {
            history_depth: 4,
            bin_count: 8,
            ksg_k: 3,
        });
        let metrics = observer.observe(&trace);

        assert!(metrics.ais_binning >= CONDUCTANCE_REGRESSION.ais_threshold);
        assert!(metrics.te >= CONDUCTANCE_REGRESSION.te_minimum);
        assert!(metrics.tc >= CONDUCTANCE_REGRESSION.tc_cutoff);
    }

    #[test]
    fn conductance_regression_signature_is_stable() {
        let bytes = conductance_regression_signature_bytes();
        let expected: Vec<u8> = vec![
            123, 20, 174, 71, 225, 122, 132, 63, 252, 169, 241, 210, 77, 98, 80, 63, 123, 20, 174,
            71, 225, 122, 132, 63, 72, 107, 1, 0, 0, 0, 0, 0,
        ];
        assert_eq!(bytes, expected);
    }
}
