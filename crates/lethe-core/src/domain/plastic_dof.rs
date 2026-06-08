#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PlasticDOF {
    CouplingWeight,
    MemoryDepth,
}

impl PlasticDOF {
    #[must_use]
    pub const fn is_dead_term(self) -> bool {
        matches!(self, Self::CouplingWeight)
    }

    #[must_use]
    pub const fn is_live_term(self) -> bool {
        matches!(self, Self::MemoryDepth)
    }
}
