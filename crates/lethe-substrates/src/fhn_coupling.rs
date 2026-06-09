//! FHN coupling-band adaptive variant.
//!
//! T21 hypothesis: the live DOF on FHN is the **coupling band**, not the
//! retention band. The T08 dead-term lift (`+0.7` to `+2.4` across `α`)
//! suggests coupling-weight Hebbian learning is a real live DOF on FHN, not
//! just a foil.
//!
//! This variant is intentionally a separate file from `fhn.rs` so the
//! original FHN adapter remains the control. The two share the FHN physics
//! (RK4 stepping, excitable-media dynamics) but differ in which DOF is
//! plastic. `FhnSubstrate` keeps `coupling` as a static scalar on the
//! config; this variant replaces that scalar with a per-cell plastic
//! `coupling_band: Vec<f64>` driven by `eta_coupling`, leaving `lambda`
//! held as a static scalar.
//!
//! Primary Goldilocks knob: `eta_coupling`.
//! Natural DOF: [`DofKind::Coupling`].

use lethe_core::{DofKind, NaturalDof, State, Substrate, SubstrateParams};
use rand_chacha::ChaCha8Rng;
use rand_core::RngCore;

const DT: f64 = 0.01;
const STEPS_PER_SAMPLE: usize = 10;
const FHN_A: f64 = 0.7;
const FHN_B: f64 = 0.8;

pub const FHN_COUPLING_SEED_BASE: u64 = 91_500;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FhnCouplingConfig {
    pub size: usize,
    pub epsilon: f64,
    pub i_ext: f64,
    pub i_ext_noise: f64,
    pub lambda: f64,
    pub eta_coupling: f64,
    pub coupling_leak: f64,
    pub coupling_min: f64,
    pub coupling_max: f64,
    pub seed: u64,
}

impl Default for FhnCouplingConfig {
    fn default() -> Self {
        Self {
            size: 32,
            epsilon: 0.08,
            i_ext: 0.5,
            i_ext_noise: 0.1,
            lambda: 0.95,
            eta_coupling: 0.01,
            coupling_leak: 0.01,
            coupling_min: 0.0,
            coupling_max: 1.0,
            seed: FHN_COUPLING_SEED_BASE,
        }
    }
}

impl FhnCouplingConfig {
    #[must_use]
    pub const fn cell_count(self) -> usize {
        self.size * self.size
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
struct FhnCouplingNode {
    v: f64,
    w: f64,
}

#[derive(Debug, Clone)]
pub struct FhnCouplingSubstrate {
    config: FhnCouplingConfig,
    nodes: Vec<FhnCouplingNode>,
    scratch: Vec<FhnCouplingNode>,
    coupling_band: Vec<f64>,
    state: State,
}

impl FhnCouplingSubstrate {
    #[must_use]
    pub fn new(config: FhnCouplingConfig) -> Self {
        let mut seed_stream = config.seed;
        let cell_count = config.cell_count();

        let mut nodes = Vec::with_capacity(cell_count);
        for _ in 0..cell_count {
            let v = splitmix_signed_unit(&mut seed_stream) * 0.1;
            let w = splitmix_signed_unit(&mut seed_stream) * 0.1;
            nodes.push(FhnCouplingNode { v, w });
        }

        let coupling_band = vec![config.coupling_min; cell_count];
        let state = State::new(
            nodes.iter().map(|node| node.v).collect(),
            coupling_band.clone(),
        );

        Self {
            config,
            nodes,
            scratch: vec![FhnCouplingNode { v: 0.0, w: 0.0 }; cell_count],
            coupling_band,
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
        let weight = self.coupling_band.get(idx).copied().unwrap_or(0.0);
        weight * (sum / 4.0)
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
                    .unwrap_or(FhnCouplingNode { v: 0.0, w: 0.0 });
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

    fn update_coupling_band(&mut self) {
        let eta = self.config.eta_coupling;
        let leak = self.config.coupling_leak;
        let min = self.config.coupling_min;
        let max = self.config.coupling_max;
        let n = self.config.cell_count();
        for idx in 0..n {
            let activity = self.nodes.get(idx).map(|node| node.v).unwrap_or(0.0);
            let neighbors = self.neighbors(idx);
            let mut neighbor_mean = 0.0;
            for neighbor in neighbors {
                neighbor_mean += self
                    .nodes
                    .get(neighbor)
                    .map(|node| node.v)
                    .unwrap_or(activity);
            }
            neighbor_mean /= 4.0;
            // Hebbian: strengthen the local coupling weight when the
            // cell's activity covaries positively with its neighbor
            // mean; anti-Hebbian leak pulls the band back toward zero.
            let hebbian = (activity - neighbor_mean) * activity;
            let current = self.coupling_band.get(idx).copied().unwrap_or(0.0);
            let updated = eta.mul_add(hebbian, (1.0 - leak) * current).clamp(min, max);
            if let Some(slot) = self.coupling_band.get_mut(idx) {
                *slot = updated;
            }
        }
    }

    fn refresh_state(&mut self) {
        self.state.activities = self.nodes.iter().map(|node| node.v).collect();
        self.state.lambda_i = self.coupling_band.clone();
    }
}

impl Substrate for FhnCouplingSubstrate {
    fn step(&mut self, rng: &mut ChaCha8Rng) -> &State {
        self.step_macro(rng);
        self.update_coupling_band();
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
        self.coupling_band = reset.coupling_band;
        self.state = reset.state;
    }
}

impl NaturalDof for FhnCouplingSubstrate {
    fn natural_dof(&self) -> DofKind {
        DofKind::Coupling
    }
}

#[inline]
fn rhs(v: f64, w: f64, total_input: f64, epsilon: f64) -> (f64, f64) {
    let dv = v - (v * v * v) / 3.0 - w + total_input;
    let dw = epsilon * v.mul_add(1.0, FHN_A).mul_add(1.0, -FHN_B * w);
    (dv, dw)
}

#[inline]
fn rk4_step(node: FhnCouplingNode, total_input: f64, epsilon: f64) -> FhnCouplingNode {
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

    FhnCouplingNode {
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
    use super::{FHN_COUPLING_SEED_BASE, FhnCouplingConfig, FhnCouplingSubstrate};
    use lethe_core::{DofKind, NaturalDof, Substrate};
    use rand_chacha::ChaCha8Rng;
    use rand_core::SeedableRng;

    #[test]
    fn fhn_coupling_natural_dof_is_coupling() {
        let config = FhnCouplingConfig::default();
        let substrate = FhnCouplingSubstrate::new(config);
        assert_eq!(substrate.natural_dof(), DofKind::Coupling);
    }

    #[test]
    fn fhn_coupling_step_is_deterministic_under_fixed_seed() {
        let config = FhnCouplingConfig::default();
        let mut sub_a = FhnCouplingSubstrate::new(config);
        let mut sub_b = FhnCouplingSubstrate::new(config);
        let mut rng_a = ChaCha8Rng::seed_from_u64(FHN_COUPLING_SEED_BASE);
        let mut rng_b = ChaCha8Rng::seed_from_u64(FHN_COUPLING_SEED_BASE);
        for _ in 0..100 {
            sub_a.step(&mut rng_a);
            sub_b.step(&mut rng_b);
        }
        assert_eq!(sub_a.state, sub_b.state);
    }

    #[test]
    fn fhn_coupling_config_exposes_eta_coupling_as_primary_knob() {
        // T21 hypothesis: `eta_coupling` is the primary Goldilocks knob.
        // Pin the field's existence and that the default is positive finite.
        let config = FhnCouplingConfig::default();
        assert!(config.eta_coupling.is_finite());
        assert!(config.eta_coupling > 0.0);
    }

    #[test]
    fn fhn_coupling_seed_base_is_distinct_from_fhn_seed_base() {
        // Seed base must not collide with the original FHN adapter's seed
        // base, otherwise cross-experiment re-seeding would not be safe.
        assert_ne!(FHN_COUPLING_SEED_BASE, super::super::fhn::FHN_SEED_BASE);
    }
}
