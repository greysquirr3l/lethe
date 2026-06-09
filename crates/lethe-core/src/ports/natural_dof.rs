//! Extension port trait: query a substrate's natural live-DOF.
//!
//! See `tasks/T21-pivot-per-substrate-live-dof.md`. The `NaturalDof` trait
//! is per-substrate-class: a property of the substrate's physics, not of
//! the experiment-time plasticity knob.
//!
//! Implementors return the `DofKind` that, when perturbed, produces the
//! largest response in this substrate's correlation geometry. The T08
//! result tells us that for the lattice that DOF is `Retention`; T21 is
//! the gate that determines the same for FHN, oscillator, and conductance.

use crate::DofKind;

pub trait NaturalDof {
    fn natural_dof(&self) -> DofKind;
}

#[cfg(test)]
mod tests {
    use crate::DofKind;
    use crate::NaturalDof;

    struct Probe(DofKind);

    impl NaturalDof for Probe {
        fn natural_dof(&self) -> DofKind {
            self.0
        }
    }

    #[test]
    fn natural_dof_trait_returns_implementor_value() {
        let probe = Probe(DofKind::Coupling);
        assert_eq!(probe.natural_dof(), DofKind::Coupling);
    }
}
