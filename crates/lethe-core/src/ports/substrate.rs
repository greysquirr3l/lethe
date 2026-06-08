use rand_chacha::ChaCha8Rng;

use crate::domain::State;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SubstrateParams {
    pub cell_count: usize,
}

pub trait Substrate {
    fn step(&mut self, rng: &mut ChaCha8Rng) -> &State;
    fn params(&self) -> SubstrateParams;
    fn reset(&mut self, seed: u64);
}
