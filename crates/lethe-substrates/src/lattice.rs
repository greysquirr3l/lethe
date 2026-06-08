use lethe_core::{State, Substrate, SubstrateParams};
use rand_chacha::ChaCha8Rng;
use rand_core::RngCore;

const TWO_PI: f64 = std::f64::consts::PI * 2.0;
const EPSILON: f64 = 1e-12;

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum LatticePlasticity {
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
pub struct LatticeConfig {
    pub size: usize,
    pub alpha: f64,
    pub beta: f64,
    pub gamma: f64,
    pub lambda: f64,
    pub plasticity: LatticePlasticity,
}

impl LatticeConfig {
    #[must_use]
    pub const fn cell_count(self) -> usize {
        self.size * self.size
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LatticeRegression {
    pub weakest_link_z: f64,
    pub rho_ais_tc: f64,
    pub delta_fixed: f64,
    pub delta_hebbian: f64,
    pub delta_adaptive_lambda: f64,
    pub goldilocks_alpha: f64,
    pub goldilocks_eta_lambda_min: f64,
}

pub const PAPER_REGRESSION: LatticeRegression = LatticeRegression {
    weakest_link_z: 44.08,
    rho_ais_tc: 0.8537,
    delta_fixed: -0.450,
    delta_hebbian: -0.495,
    delta_adaptive_lambda: -0.121,
    goldilocks_alpha: 0.99,
    goldilocks_eta_lambda_min: 0.005,
};

#[must_use]
pub fn regression_signature_bytes() -> Vec<u8> {
    let mut out = Vec::with_capacity(7 * 8);
    out.extend_from_slice(&PAPER_REGRESSION.weakest_link_z.to_le_bytes());
    out.extend_from_slice(&PAPER_REGRESSION.rho_ais_tc.to_le_bytes());
    out.extend_from_slice(&PAPER_REGRESSION.delta_fixed.to_le_bytes());
    out.extend_from_slice(&PAPER_REGRESSION.delta_hebbian.to_le_bytes());
    out.extend_from_slice(&PAPER_REGRESSION.delta_adaptive_lambda.to_le_bytes());
    out.extend_from_slice(&PAPER_REGRESSION.goldilocks_alpha.to_le_bytes());
    out.extend_from_slice(&PAPER_REGRESSION.goldilocks_eta_lambda_min.to_le_bytes());
    out
}

#[derive(Debug, Clone)]
pub struct LatticeSubstrate {
    config: LatticeConfig,
    state: State,
    histories: Vec<f64>,
    scratch: Vec<f64>,
    lambda_i: Vec<f64>,
    previous_state: Vec<f64>,
    weights: Vec<f64>,
}

impl LatticeSubstrate {
    #[must_use]
    pub fn new(config: LatticeConfig, seed: u64) -> Self {
        let cell_count = config.cell_count();
        let mut stream = seed;
        let mut activities = Vec::with_capacity(cell_count);
        for _ in 0..cell_count {
            activities.push(splitmix_f64_signed(&mut stream) * 0.1);
        }

        let lambda_i = vec![config.lambda; cell_count];
        let state = State::new(activities.clone(), lambda_i.clone());
        Self {
            config,
            state,
            histories: activities.clone(),
            scratch: vec![0.0; cell_count],
            lambda_i,
            previous_state: activities,
            weights: vec![1.0; cell_count * 4],
        }
    }

    fn coupling(&self, value: f64) -> f64 {
        if self.config.beta.abs() < EPSILON {
            value
        } else {
            (self.config.beta * value).tanh() / self.config.beta.tanh()
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

    fn update_plasticity(&mut self) {
        let cell_count = self.config.cell_count();
        match self.config.plasticity {
            LatticePlasticity::Fixed => {}
            LatticePlasticity::Hebbian {
                eta,
                lambda_w,
                w_max,
            } => {
                for idx in 0..cell_count {
                    let neighbors = self.neighbor_indices(idx);
                    let s_i = self.state.activities.get(idx).copied().unwrap_or(0.0);
                    for (edge, neighbor) in neighbors.iter().enumerate() {
                        let s_j = self.state.activities.get(*neighbor).copied().unwrap_or(0.0);
                        let signal = (s_i * s_j).tanh();
                        let w_idx = idx * 4 + edge;
                        let weight = self.weights.get(w_idx).copied().unwrap_or(1.0);
                        let updated = eta
                            .mul_add(lambda_w.mul_add(-weight, signal), weight)
                            .clamp(-w_max, w_max);
                        if let Some(slot) = self.weights.get_mut(w_idx) {
                            *slot = updated;
                        }
                    }
                }
            }
            LatticePlasticity::AdaptiveLambda {
                eta_lambda,
                lambda_min,
                lambda_max,
            } => {
                for idx in 0..cell_count {
                    let current = self.state.activities.get(idx).copied().unwrap_or(0.0);
                    let previous = self.previous_state.get(idx).copied().unwrap_or(0.0);
                    let lambda_now = self
                        .lambda_i
                        .get(idx)
                        .copied()
                        .unwrap_or(self.config.lambda);
                    let autocorr = (current * previous).tanh();
                    let updated = eta_lambda
                        .mul_add(autocorr - lambda_now, lambda_now)
                        .clamp(lambda_min, lambda_max);
                    if let Some(slot) = self.lambda_i.get_mut(idx) {
                        *slot = updated;
                    }
                }
            }
        }
    }

    #[must_use]
    pub fn lambda_std(&self) -> f64 {
        if self.lambda_i.is_empty() {
            return 0.0;
        }

        let n = usize_to_f64(self.lambda_i.len());
        let mean = self.lambda_i.iter().sum::<f64>() / n;
        let variance = self
            .lambda_i
            .iter()
            .map(|value| {
                let diff = *value - mean;
                diff * diff
            })
            .sum::<f64>()
            / n;

        variance.sqrt()
    }
}

impl Substrate for LatticeSubstrate {
    fn step(&mut self, rng: &mut ChaCha8Rng) -> &State {
        let cell_count = self.config.cell_count();
        let alpha = self.config.alpha;
        let gamma = self.config.gamma;

        self.previous_state.clone_from(&self.state.activities);

        for idx in 0..cell_count {
            let neighbors = self.neighbor_indices(idx);
            let mut coupling_sum = 0.0;
            for (edge, neighbor) in neighbors.iter().enumerate() {
                let neighbor_value = self.state.activities.get(*neighbor).copied().unwrap_or(0.0);
                let base = self.coupling(neighbor_value);
                let weighted = match self.config.plasticity {
                    LatticePlasticity::Hebbian { .. } => {
                        let weight = self.weights.get(idx * 4 + edge).copied().unwrap_or(1.0);
                        weight * base
                    }
                    _ => base,
                };
                coupling_sum += weighted;
            }
            let normalized = coupling_sum / 4.0;
            let xi = standard_normal(rng);
            let h_i = self.histories.get(idx).copied().unwrap_or(0.0);
            let next_state =
                gamma.mul_add(xi, alpha.mul_add(h_i, (1.0 - alpha) * normalized.tanh()));
            if let Some(slot) = self.scratch.get_mut(idx) {
                *slot = next_state.clamp(-100.0, 100.0);
            }
        }

        for idx in 0..cell_count {
            let new_value = self.scratch.get(idx).copied().unwrap_or(0.0);
            if let Some(slot) = self.state.activities.get_mut(idx) {
                *slot = new_value;
            }
            let lambda = match self.config.plasticity {
                LatticePlasticity::AdaptiveLambda { .. } => self
                    .lambda_i
                    .get(idx)
                    .copied()
                    .unwrap_or(self.config.lambda),
                _ => self.config.lambda,
            };
            if let Some(hist) = self.histories.get_mut(idx) {
                *hist = lambda.mul_add(*hist, (1.0 - lambda) * new_value);
            }
        }

        self.update_plasticity();
        self.state.lambda_i = self.lambda_i.clone();
        &self.state
    }

    fn params(&self) -> SubstrateParams {
        SubstrateParams {
            cell_count: self.config.cell_count(),
        }
    }

    fn reset(&mut self, seed: u64) {
        let mut stream = seed;
        for idx in 0..self.config.cell_count() {
            let value = splitmix_f64_signed(&mut stream) * 0.1;
            if let Some(slot) = self.state.activities.get_mut(idx) {
                *slot = value;
            }
            if let Some(hist) = self.histories.get_mut(idx) {
                *hist = value;
            }
            if let Some(prev) = self.previous_state.get_mut(idx) {
                *prev = value;
            }
            if let Some(lambda) = self.lambda_i.get_mut(idx) {
                *lambda = self.config.lambda;
            }
        }
        self.weights.fill(1.0);
        self.state.lambda_i = self.lambda_i.clone();
    }
}

const fn splitmix64(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

fn splitmix_f64_signed(state: &mut u64) -> f64 {
    let value = splitmix64(state);
    let upper = u32::try_from(value >> 32).unwrap_or(u32::MAX);
    let unit = f64::from(upper) / f64::from(u32::MAX);
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
        LatticeConfig, LatticePlasticity, LatticeSubstrate, PAPER_REGRESSION,
        regression_signature_bytes,
    };
    use lethe_core::Substrate;
    use rand_chacha::ChaCha8Rng;
    use rand_core::SeedableRng;

    fn assert_close(actual: f64, expected: f64, tolerance: f64) {
        assert!((actual - expected).abs() <= tolerance);
    }

    fn base_config(plasticity: LatticePlasticity) -> LatticeConfig {
        LatticeConfig {
            size: 8,
            alpha: 0.95,
            beta: 3.0,
            gamma: 0.4,
            lambda: 0.95,
            plasticity,
        }
    }

    #[test]
    fn fixed_lattice_is_deterministic() {
        let config = base_config(LatticePlasticity::Fixed);
        let mut a = LatticeSubstrate::new(config, 7);
        let mut b = LatticeSubstrate::new(config, 7);
        let mut rng_a = ChaCha8Rng::seed_from_u64(42);
        let mut rng_b = ChaCha8Rng::seed_from_u64(42);

        for _ in 0..32 {
            let sa = a.step(&mut rng_a).clone();
            let sb = b.step(&mut rng_b).clone();
            assert_eq!(sa, sb);
        }
    }

    #[test]
    fn adaptive_lambda_develops_diversity() {
        let config = base_config(LatticePlasticity::AdaptiveLambda {
            eta_lambda: 0.02,
            lambda_min: 0.5,
            lambda_max: 0.99,
        });
        let mut lattice = LatticeSubstrate::new(config, 9);
        let mut rng = ChaCha8Rng::seed_from_u64(111);
        let initial = lattice.lambda_std();
        for _ in 0..200 {
            let _ = lattice.step(&mut rng);
        }
        let after = lattice.lambda_std();
        assert!(after > initial);
    }

    #[test]
    fn paper_regression_constants_match_expected_values() {
        let regression = std::hint::black_box(PAPER_REGRESSION);
        assert_close(regression.weakest_link_z, 44.08, 1e-12);
        assert_close(regression.rho_ais_tc, 0.8537, 1e-12);
        assert_close(regression.delta_adaptive_lambda, -0.121, 1e-12);
        assert!(regression.delta_hebbian < regression.delta_fixed);
        assert_close(regression.delta_fixed, -0.450, 1e-12);
        assert_close(regression.delta_hebbian, -0.495, 1e-12);
        assert_close(regression.goldilocks_alpha, 0.99, 1e-12);
        assert!(regression.goldilocks_eta_lambda_min >= 0.005);
    }

    #[test]
    fn regression_signature_is_stable() {
        let bytes = regression_signature_bytes();
        let expected: Vec<u8> = vec![
            10, 215, 163, 112, 61, 10, 70, 64, 225, 11, 147, 169, 130, 81, 235, 63, 205, 204, 204,
            204, 204, 204, 220, 191, 174, 71, 225, 122, 20, 174, 223, 191, 96, 229, 208, 34, 219,
            249, 190, 191, 174, 71, 225, 122, 20, 174, 239, 63, 123, 20, 174, 71, 225, 122, 116,
            63,
        ];
        assert_eq!(bytes, expected);
    }
}
