use rand_chacha::ChaCha8Rng;
use rand_core::{RngCore, SeedableRng};

#[must_use]
pub fn seeded_micro_experiment(seed: u64, steps: usize) -> Vec<u8> {
    let mut seed_bytes = [0_u8; 32];
    seed_bytes[..8].copy_from_slice(&seed.to_le_bytes());
    let mut rng = ChaCha8Rng::from_seed(seed_bytes);

    let mut output = vec![0_u8; steps];
    rng.fill_bytes(&mut output);
    output
}

#[cfg(test)]
mod tests {
    use crate::seeded_micro_experiment;

    #[test]
    fn same_seed_produces_same_bytes() {
        let first = seeded_micro_experiment(424_242, 1024);
        let second = seeded_micro_experiment(424_242, 1024);
        assert_eq!(first, second);
    }

    #[test]
    fn different_seeds_produce_different_bytes() {
        let first = seeded_micro_experiment(1, 1024);
        let second = seeded_micro_experiment(2, 1024);
        assert_ne!(first, second);
    }

    #[test]
    fn output_has_fixed_width() {
        let output = seeded_micro_experiment(7, 1024);
        assert_eq!(output.len(), 1024);
    }
}
