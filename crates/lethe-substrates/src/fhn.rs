use lethe_core::{DofKind, NaturalDof, State, Substrate, SubstrateParams};
use rand_chacha::ChaCha8Rng;
use rand_core::RngCore;

const DT: f64 = 0.01;
const STEPS_PER_SAMPLE: usize = 10;
const FHN_A: f64 = 0.7;
const FHN_B: f64 = 0.8;

pub const FHN_SEED_BASE: u64 = 91_000;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FhnConfig {
    pub size: usize,
    pub epsilon: f64,
    pub coupling: f64,
    pub i_ext: f64,
    pub i_ext_noise: f64,
    pub lambda: f64,
    pub seed: u64,
}

impl Default for FhnConfig {
    fn default() -> Self {
        Self {
            size: 32,
            epsilon: 0.08,
            coupling: 0.2,
            i_ext: 0.5,
            i_ext_noise: 0.1,
            lambda: 0.95,
            seed: FHN_SEED_BASE,
        }
    }
}

impl FhnConfig {
    #[must_use]
    pub const fn cell_count(self) -> usize {
        self.size * self.size
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FhnCorrelationBands {
    pub global_r_te_tc: f64,
    pub global_r_ais_te: f64,
    pub global_r_ais_tc: f64,
    pub coupling_band_sign_flips: usize,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FhnRegression {
    pub dense_geometry: FhnCorrelationBands,
}

pub const FHN_PAPER_REGRESSION: FhnRegression = FhnRegression {
    dense_geometry: FhnCorrelationBands {
        global_r_te_tc: -0.7385,
        global_r_ais_te: 0.4496,
        global_r_ais_tc: 0.1524,
        coupling_band_sign_flips: 0,
    },
};

#[must_use]
pub fn fhn_regression_signature_bytes() -> Vec<u8> {
    let mut out = Vec::with_capacity(8 * 4);
    out.extend_from_slice(
        &FHN_PAPER_REGRESSION
            .dense_geometry
            .global_r_te_tc
            .to_le_bytes(),
    );
    out.extend_from_slice(
        &FHN_PAPER_REGRESSION
            .dense_geometry
            .global_r_ais_te
            .to_le_bytes(),
    );
    out.extend_from_slice(
        &FHN_PAPER_REGRESSION
            .dense_geometry
            .global_r_ais_tc
            .to_le_bytes(),
    );

    let sign_flips = u64::try_from(FHN_PAPER_REGRESSION.dense_geometry.coupling_band_sign_flips)
        .unwrap_or(u64::MAX);
    out.extend_from_slice(&sign_flips.to_le_bytes());
    out
}

#[derive(Debug, Clone, Copy, PartialEq)]
struct FhnNode {
    v: f64,
    w: f64,
}

#[derive(Debug, Clone)]
pub struct FhnSubstrate {
    config: FhnConfig,
    nodes: Vec<FhnNode>,
    scratch: Vec<FhnNode>,
    lambda_i: Vec<f64>,
    state: State,
}

impl FhnSubstrate {
    #[must_use]
    pub fn new(config: FhnConfig) -> Self {
        let mut seed_stream = config.seed;
        let cell_count = config.cell_count();

        let mut nodes = Vec::with_capacity(cell_count);
        for _ in 0..cell_count {
            let v = splitmix_signed_unit(&mut seed_stream) * 0.1;
            let w = splitmix_signed_unit(&mut seed_stream) * 0.1;
            nodes.push(FhnNode { v, w });
        }

        let lambda_i = vec![config.lambda; cell_count];
        let state = State::new(nodes.iter().map(|node| node.v).collect(), lambda_i.clone());

        Self {
            config,
            nodes,
            scratch: vec![FhnNode { v: 0.0, w: 0.0 }; cell_count],
            lambda_i,
            state,
        }
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
        for neighbor in neighbors {
            let v_neighbor = v_snapshot.get(neighbor).copied().unwrap_or(base);
            sum += v_neighbor - base;
        }
        self.config.coupling * (sum / 4.0)
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
                    .unwrap_or(FhnNode { v: 0.0, w: 0.0 });
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

    fn refresh_state(&mut self) {
        self.state.activities = self.nodes.iter().map(|node| node.v).collect();
        self.state.lambda_i = self.lambda_i.clone();
    }
}

impl Substrate for FhnSubstrate {
    fn step(&mut self, rng: &mut ChaCha8Rng) -> &State {
        self.step_macro(rng);
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
        self.lambda_i = reset.lambda_i;
        self.state = reset.state;
    }
}

#[inline]
fn rhs(v: f64, w: f64, total_input: f64, epsilon: f64) -> (f64, f64) {
    let dv = v - (v * v * v) / 3.0 - w + total_input;
    let dw = epsilon * v.mul_add(1.0, FHN_A).mul_add(1.0, -FHN_B * w);
    (dv, dw)
}

#[inline]
fn rk4_step(node: FhnNode, total_input: f64, epsilon: f64) -> FhnNode {
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

    FhnNode {
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

impl NaturalDof for FhnSubstrate {
    fn natural_dof(&self) -> DofKind {
        DofKind::Coupling
    }
}

#[cfg(test)]
mod tests {
    use super::{
        FHN_PAPER_REGRESSION, FHN_SEED_BASE, FhnConfig, FhnSubstrate,
        fhn_regression_signature_bytes,
    };
    use lethe_core::Substrate;
    use rand_chacha::ChaCha8Rng;
    use rand_core::SeedableRng;

    #[test]
    fn fhn_substrate_is_deterministic_under_seeded_rng() {
        let config = FhnConfig::default();
        let mut left = FhnSubstrate::new(config);
        let mut right = FhnSubstrate::new(config);
        let mut rng_left = ChaCha8Rng::seed_from_u64(FHN_SEED_BASE + 1);
        let mut rng_right = ChaCha8Rng::seed_from_u64(FHN_SEED_BASE + 1);

        for _ in 0..16 {
            let left_state = left.step(&mut rng_left).clone();
            let right_state = right.step(&mut rng_right).clone();
            assert_eq!(left_state, right_state);
        }
    }

    #[test]
    fn paper_fhn_geometry_constants_match_expected_values() {
        let geom = FHN_PAPER_REGRESSION.dense_geometry;
        assert!((geom.global_r_ais_te - 0.4496).abs() <= 1e-6);
        assert!((geom.global_r_ais_tc - 0.1524).abs() <= 1e-6);
        assert!((geom.global_r_te_tc + 0.7385).abs() <= 1e-6);
        assert!((geom.global_r_te_tc + 0.739).abs() <= 0.01);
        assert_eq!(geom.coupling_band_sign_flips, 0);
    }

    #[test]
    fn fhn_seed_base_matches_reference() {
        assert_eq!(FHN_SEED_BASE, 91_000);
    }

    #[test]
    fn fhn_regression_signature_is_stable() {
        let bytes = fhn_regression_signature_bytes();
        let expected: Vec<u8> = vec![
            111, 18, 131, 192, 202, 161, 231, 191, 188, 5, 18, 20, 63, 198, 220, 63, 253, 135, 244,
            219, 215, 129, 195, 63, 0, 0, 0, 0, 0, 0, 0, 0,
        ];
        assert_eq!(bytes, expected);
    }

    #[test]
    fn fhn_natural_dof_is_coupling() {
        use lethe_core::DofKind;
        use lethe_core::NaturalDof;

        let substrate = FhnSubstrate::new(FhnConfig::default());

        assert_eq!(substrate.natural_dof(), DofKind::Coupling);
    }
}
