#![forbid(unsafe_code)]

mod conductance;
mod conductance_retention;
mod fhn;
mod fhn_coupling;
mod fhn_coupling_hebbian;
mod lattice;
mod oscillator;
mod oscillator_frequency;

pub use conductance::{
    CONDUCTANCE_REGRESSION, CONDUCTANCE_SEED_BASE, ConductanceConfig, ConductancePlasticity,
    ConductanceSubstrate, conductance_regression_signature_bytes,
};
pub use conductance_retention::{
    CONDUCTANCE_RETENTION_SEED_BASE, ConductanceRetentionConfig, ConductanceRetentionSubstrate,
};
pub use fhn::{
    FHN_PAPER_REGRESSION, FHN_SEED_BASE, FhnConfig, FhnCorrelationBands, FhnRegression,
    FhnSubstrate, fhn_regression_signature_bytes,
};
pub use fhn_coupling::{FHN_COUPLING_SEED_BASE, FhnCouplingConfig, FhnCouplingSubstrate};
pub use fhn_coupling_hebbian::{
    FHN_COUPLING_HEBBIAN_SEED_BASE, FhnCouplingHebbianConfig, FhnCouplingHebbianSubstrate,
    fhn_hebbian_regression_signature_bytes,
};
pub use lattice::{
    LatticeConfig, LatticePlasticity, LatticeRegression, LatticeSubstrate, PAPER_REGRESSION,
    regression_signature_bytes,
};
pub use oscillator::{
    OSCILLATOR_REGRESSION, OSCILLATOR_SEED_BASE, OscillatorConfig, OscillatorPlasticity,
    OscillatorSubstrate, oscillator_regression_signature_bytes,
};
pub use oscillator_frequency::{
    OSCILLATOR_FREQUENCY_SEED_BASE, OscillatorFrequencyConfig, OscillatorFrequencySubstrate,
};
