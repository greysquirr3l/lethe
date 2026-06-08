#![forbid(unsafe_code)]

mod conductance;
mod fhn;
mod lattice;
mod oscillator;

pub use conductance::{
    CONDUCTANCE_REGRESSION, CONDUCTANCE_SEED_BASE, ConductanceConfig, ConductancePlasticity,
    ConductanceSubstrate, conductance_regression_signature_bytes,
};
pub use fhn::{
    FHN_PAPER_REGRESSION, FHN_SEED_BASE, FhnConfig, FhnCorrelationBands, FhnRegression,
    FhnSubstrate, fhn_regression_signature_bytes,
};
pub use lattice::{
    LatticeConfig, LatticePlasticity, LatticeRegression, LatticeSubstrate, PAPER_REGRESSION,
    regression_signature_bytes,
};
pub use oscillator::{
    OSCILLATOR_REGRESSION, OSCILLATOR_SEED_BASE, OscillatorConfig, OscillatorPlasticity,
    OscillatorSubstrate, oscillator_regression_signature_bytes,
};
