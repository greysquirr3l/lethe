#![forbid(unsafe_code)]

pub mod domain;
pub mod ports;

pub use domain::{
    Observer, ObserverConfig, ObserverMetrics, PlasticDOF, State, StateTrace, TraceFrame,
};
pub use ports::{Substrate, SubstrateParams};

#[cfg(test)]
mod tests {
    use rand_chacha::ChaCha8Rng;
    use rand_chacha::rand_core::{RngCore, SeedableRng};

    use crate::domain::{PlasticDOF, State};
    use crate::ports::{Substrate, SubstrateParams};

    struct NullSubstrate {
        params: SubstrateParams,
        state: State,
        reset_seed: u32,
    }

    impl NullSubstrate {
        fn new(cell_count: usize) -> Self {
            let params = SubstrateParams { cell_count };
            let state = State::new(vec![0.0; cell_count], vec![0.0; cell_count]);
            Self {
                params,
                state,
                reset_seed: 0,
            }
        }
    }

    impl Substrate for NullSubstrate {
        fn step(&mut self, rng: &mut ChaCha8Rng) -> &State {
            let cell_count = self.params.cell_count;
            let mut activities = Vec::with_capacity(cell_count);
            for _ in 0..cell_count {
                activities.push(f64::from(rng.next_u32()));
            }

            let reset_value = f64::from(self.reset_seed);
            self.state = State::new(activities, vec![reset_value; cell_count]);
            &self.state
        }

        fn params(&self) -> SubstrateParams {
            self.params
        }

        fn reset(&mut self, seed: u64) {
            let seed32 = u32::try_from(seed).unwrap_or(u32::MAX);
            self.reset_seed = seed32;
            let reset_value = f64::from(seed32);
            let cell_count = self.params.cell_count;
            self.state = State::new(vec![reset_value; cell_count], vec![reset_value; cell_count]);
        }
    }

    #[test]
    fn null_substrate_steps_deterministically_from_fixed_seed() {
        let mut substrate_a = NullSubstrate::new(4);
        let mut substrate_b = NullSubstrate::new(4);

        substrate_a.reset(99);
        substrate_b.reset(99);

        let mut rng_a = ChaCha8Rng::seed_from_u64(424_242);
        let mut rng_b = ChaCha8Rng::seed_from_u64(424_242);

        let first = substrate_a.step(&mut rng_a).clone();
        let second = substrate_b.step(&mut rng_b).clone();
        assert_eq!(first, second);
    }

    #[test]
    fn plastic_dof_distinguishes_dead_and_live_terms() {
        let dead_term = PlasticDOF::CouplingWeight;
        let live_term = PlasticDOF::MemoryDepth;

        assert!(dead_term.is_dead_term());
        assert!(!dead_term.is_live_term());
        assert!(live_term.is_live_term());
        assert!(!live_term.is_dead_term());
    }
}
