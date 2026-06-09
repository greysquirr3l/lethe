//! Per-substrate live degree-of-freedom taxonomy.
//!
//! See `tasks/T21-pivot-per-substrate-live-dof.md`. The T08 generalisation
//! gate established that per-cell adaptive memory depth `λᵢ` is the
//! *lattice* natural DOF, not the universal one. This taxonomy names each
//! substrate's natural live-DOF so that the gate can sweep the right knob
//! per substrate, and so that the per-substrate re-GO criterion can be
//! expressed in substrate-aware terms rather than a single-axis `PlasticDOF`.
//!
//! Distinct from [`crate::domain::plastic_dof::PlasticDOF`], which is the
//! single-axis taxonomy swept in T08 (`CouplingWeight` vs `MemoryDepth`).
//! `DofKind` is a property of the substrate *class*; `PlasticDOF` is a
//! property of one experiment's plasticity mode.

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DofKind {
    /// Per-cell retention band (λ / λᵢ). Lattice and conductance.
    Retention,
    /// Coupling band (`η_coupling` / diffusive coupling gain). FHN.
    Coupling,
    /// Intrinsic frequency (`η_ω`). Oscillator.
    Frequency,
}

#[cfg(test)]
mod tests {
    use crate::DofKind;

    #[test]
    fn dof_kind_variants_distinguish_substrate_classes() {
        // Pin the T21 per-substrate natural DOF set.
        // Lattice/conductance: Retention (different rates)
        // FHN: Coupling
        // Oscillator: Frequency
        // Distinctness prevents silent variant collapse in a future refactor.
        assert_ne!(DofKind::Retention, DofKind::Coupling);
        assert_ne!(DofKind::Retention, DofKind::Frequency);
        assert_ne!(DofKind::Coupling, DofKind::Frequency);
    }
}
