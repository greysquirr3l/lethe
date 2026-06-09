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

impl DofKind {
    /// Stable lowercase identifier used in evidence and decision artefacts.
    /// Pin this in tests so a rename cannot silently change downstream
    /// decision files.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Retention => "retention",
            Self::Coupling => "coupling",
            Self::Frequency => "frequency",
        }
    }
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

    #[test]
    fn dof_kind_as_str_is_stable() {
        // Pin the lowercase identifiers used in evidence and decision files.
        // A rename here would silently invalidate downstream artefacts.
        assert_eq!(DofKind::Retention.as_str(), "retention");
        assert_eq!(DofKind::Coupling.as_str(), "coupling");
        assert_eq!(DofKind::Frequency.as_str(), "frequency");
    }
}
