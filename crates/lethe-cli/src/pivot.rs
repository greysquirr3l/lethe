//! T21 per-substrate DOF-characterisation pivot harness.
//!
//! See `tasks/T21-pivot-per-substrate-live-dof.md`. The T08 generalisation
//! gate returned PIVOT: a single-knob `λᵢ` lift on lattice is real, but
//! the live DOF is substrate-dependent. This pivot re-runs the gate with
//! per-substrate Goldilocks knobs — the substrate's own *natural* live DOF
//! — and applies the relaxed re-GO criterion from the T21 spec:
//!
//! > **GO**: ≥2 non-lattice substrates in their natural DOF (λᵢ or
//! > otherwise) each satisfy the lift + dead-null criterion.
//!
//! The verdict is 3-class: GO | PIVOT | NO-GO. NO-GO is reserved for
//! substrate-independent failure (the lattice control itself breaks the
//! lift-or-null condition), in which case the underlying growth claim is
//! in question and the project should pause for human review.

#![allow(
    clippy::module_name_repetitions,
    reason = "pivot types are domain-specific"
)]

use std::ffi::OsString;
use std::fmt::Write as _;
use std::fs;
use std::path::PathBuf;

use anyhow::{Context, Result, anyhow, bail};
use lethe_core::{DofKind, Observer, ObserverConfig, ObserverMetrics, StateTrace, Substrate};
use lethe_substrates::{
    CONDUCTANCE_RETENTION_SEED_BASE, CONDUCTANCE_SEED_BASE, ConductanceConfig,
    ConductancePlasticity, ConductanceRetentionConfig, ConductanceRetentionSubstrate,
    ConductanceSubstrate, FHN_COUPLING_HEBBIAN_SEED_BASE, FHN_COUPLING_SEED_BASE, FHN_SEED_BASE,
    FhnConfig, FhnCouplingConfig, FhnCouplingHebbianConfig, FhnCouplingHebbianSubstrate,
    FhnCouplingSubstrate, FhnSubstrate, LatticeConfig, LatticePlasticity, LatticeSubstrate,
    OSCILLATOR_FREQUENCY_SEED_BASE, OSCILLATOR_SEED_BASE, OscillatorConfig,
    OscillatorFrequencyConfig, OscillatorFrequencySubstrate, OscillatorPlasticity,
    OscillatorSubstrate,
};
use rand_chacha::ChaCha8Rng;
use rand_core::SeedableRng;

const DEFAULT_PIVOT_SAMPLES: usize = 160;
const DEFAULT_PIVOT_BURN_IN: usize = 40;
const LIVE_LIFT_THRESHOLD: f64 = 0.01;
const DEAD_NULL_THRESHOLD: f64 = 0.01;

// Per-substrate Goldilocks knob grids. The natural-DOF knob is swept while
// the substrate's *other* (T08-was-wrong) DOF is held at the substrate's
// default. The T08 fixture for conductance (`eta_lambda=0.03`) is retained
// as the rightmost grid point so the T21 evidence explicitly brackets the
// T08 failure mode.
const FHN_COUPLING_GRID: &[f64] = &[0.001, 0.005, 0.01, 0.05];
// T22 follow-on: extended low end vs T21 because the per-edge Hebbian
// rule accumulates slower than a uniform gain knob. The rightmost
// `0.03` point is below T21's `0.05` because the new mechanism has a
// smaller natural step size.
const FHN_COUPLING_HEBBIAN_GRID: &[f64] = &[0.0001, 0.0003, 0.001, 0.003, 0.01, 0.03];
const OSCILLATOR_OMEGA_GRID: &[f64] = &[0.001, 0.005, 0.01, 0.05];
const CONDUCTANCE_LAMBDA_GRID: &[f64] = &[0.001, 0.003, 0.005, 0.01, 0.03];
const LATTICE_ALPHA_GRID: &[f64] = &[0.85, 0.90, 0.95, 0.99];

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Verdict {
    Go,
    Pivot,
    NoGo,
}

impl Verdict {
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Go => "GO",
            Self::Pivot => "PIVOT",
            Self::NoGo => "NO-GO",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SubstrateKind {
    Lattice,
    Fhn,
    /// T22 follow-on to T21: FHN re-tested with a T08-faithful
    /// asymmetric Hebbian per-edge coupling rule. Sits *alongside*
    /// `Fhn` (the T21 wrong-knob baseline) rather than replacing it;
    /// see `tasks/T22-fhn-asymmetric-coupling-hebbian.md`.
    FhnHebbian,
    Oscillator,
    Conductance,
}

impl SubstrateKind {
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Lattice => "lattice",
            Self::Fhn => "fhn",
            Self::FhnHebbian => "fhn-hebbian",
            Self::Oscillator => "oscillator",
            Self::Conductance => "conductance",
        }
    }

    #[must_use]
    pub const fn is_non_lattice(self) -> bool {
        !matches!(self, Self::Lattice)
    }

    #[must_use]
    pub const fn dof_kind(self) -> DofKind {
        match self {
            Self::Fhn | Self::FhnHebbian => DofKind::Coupling,
            Self::Oscillator => DofKind::Frequency,
            Self::Lattice | Self::Conductance => DofKind::Retention,
        }
    }

    #[must_use]
    pub const fn goldilocks_knob(self) -> &'static str {
        match self {
            Self::Lattice => "alpha",
            Self::Fhn | Self::FhnHebbian => "eta_coupling",
            Self::Oscillator => "eta_omega",
            Self::Conductance => "eta_lambda",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Classification {
    LiftWithRightDof,
    LiftInOtherDof,
    NoLift,
}

impl Classification {
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::LiftWithRightDof => "LIFT-WITH-RIGHT-DOF",
            Self::LiftInOtherDof => "LIFT-IN-OTHER-DOF",
            Self::NoLift => "NO-LIFT",
        }
    }

    /// Derive the per-substrate classification from the (`live_lift`, `dead_null`)
    /// tuple. See `tasks/T21-pivot-per-substrate-live-dof.md` for the
    /// 3-class taxonomy.
    #[must_use]
    pub const fn from_flags(live_lift: bool, dead_null: bool) -> Self {
        if live_lift && dead_null {
            Self::LiftWithRightDof
        } else if live_lift {
            Self::LiftInOtherDof
        } else {
            Self::NoLift
        }
    }
}

#[derive(Debug, Clone)]
pub struct PivotArgs {
    pub output_dir: PathBuf,
    pub samples: usize,
    pub burn_in: usize,
}

#[derive(Debug, Clone)]
pub struct EvidenceRow {
    pub substrate: SubstrateKind,
    pub knob: f64,
    pub fixed_score: f64,
    pub dead_score: f64,
    pub live_natural_score: f64,
    pub dead_delta: f64,
    pub live_natural_delta: f64,
}

#[derive(Debug, Clone)]
pub struct SubstrateSummary {
    pub substrate: SubstrateKind,
    pub knob: f64,
    pub fixed_score: f64,
    pub dead_score: f64,
    pub live_natural_score: f64,
    pub dead_delta: f64,
    pub live_natural_delta: f64,
    pub dead_null: bool,
    pub live_lift: bool,
    pub classification: Classification,
}

#[derive(Debug, Clone)]
pub struct PivotOutcome {
    pub summaries: Vec<SubstrateSummary>,
    pub rows: Vec<EvidenceRow>,
    pub verdict: Verdict,
    pub non_lattice_pass_count: usize,
}

pub fn parse_pivot_args(args: &[OsString]) -> Result<PivotArgs> {
    let mut output_dir = PathBuf::from("results/t21_pivot");
    let mut samples = DEFAULT_PIVOT_SAMPLES;
    let mut burn_in = DEFAULT_PIVOT_BURN_IN;

    let mut i = 0_usize;
    while i < args.len() {
        let key = args
            .get(i)
            .and_then(|value| value.to_str())
            .context("argument keys must be valid UTF-8")?;
        let value = args
            .get(i + 1)
            .context("missing value for argument")?
            .to_str()
            .context("argument values must be valid UTF-8")?;

        match key {
            "--output-dir" => output_dir = PathBuf::from(value),
            "--samples" => {
                samples = value
                    .parse::<usize>()
                    .context("--samples must be a non-negative integer")?;
            }
            "--burn-in" => {
                burn_in = value
                    .parse::<usize>()
                    .context("--burn-in must be a non-negative integer")?;
            }
            other => bail!("unsupported argument: {other}"),
        }
        i += 2;
    }

    if samples == 0 {
        bail!("--samples must be greater than zero");
    }

    Ok(PivotArgs {
        output_dir,
        samples,
        burn_in,
    })
}

fn metric_score(metrics: &ObserverMetrics) -> f64 {
    metrics.ais_binning + metrics.te + metrics.tc
}

fn collect_metrics<S: Substrate>(
    substrate: &mut S,
    seed: u64,
    burn_in: usize,
    samples: usize,
) -> ObserverMetrics {
    let observer = Observer::new(ObserverConfig::default());
    let mut trace = StateTrace::new();
    let mut rng = ChaCha8Rng::seed_from_u64(seed);

    for tick in 0..(burn_in + samples) {
        let state = substrate.step(&mut rng).clone();
        if tick >= burn_in {
            trace.push(tick - burn_in, state);
        }
    }

    observer.observe(&trace)
}

// ---------------------------------------------------------------------
// Lattice control — replicates the T08 `evaluate_lattice` per `alpha`,
// kept as a methodology check. The lattice substrate's natural DOF is
// Retention (controlled by `alpha`); the T08 lift + null signature is
// the gold standard. If this breaks, the methodology itself is broken
// and the verdict is NO-GO.
// ---------------------------------------------------------------------

fn evaluate_lattice_at(alpha: f64, burn_in: usize, samples: usize) -> EvidenceRow {
    let base = LatticeConfig {
        size: 8,
        alpha,
        beta: 3.0,
        gamma: 0.4,
        lambda: 0.95,
        plasticity: LatticePlasticity::Fixed,
    };
    let fixed_score = metric_score(&collect_metrics(
        &mut LatticeSubstrate::new(base, FHN_SEED_BASE),
        FHN_SEED_BASE + 1,
        burn_in,
        samples,
    ));

    let dead_config = LatticeConfig {
        plasticity: LatticePlasticity::Hebbian {
            eta: 0.01,
            lambda_w: 0.02,
            w_max: 2.0,
        },
        ..base
    };
    let dead_score = metric_score(&collect_metrics(
        &mut LatticeSubstrate::new(dead_config, FHN_SEED_BASE + 2),
        FHN_SEED_BASE + 3,
        burn_in,
        samples,
    ));

    let live_config = LatticeConfig {
        plasticity: LatticePlasticity::AdaptiveLambda {
            eta_lambda: 0.02,
            lambda_min: 0.5,
            lambda_max: alpha,
        },
        ..base
    };
    let live_natural_score = metric_score(&collect_metrics(
        &mut LatticeSubstrate::new(live_config, FHN_SEED_BASE + 4),
        FHN_SEED_BASE + 5,
        burn_in,
        samples,
    ));

    EvidenceRow {
        substrate: SubstrateKind::Lattice,
        knob: alpha,
        fixed_score,
        dead_score,
        live_natural_score,
        dead_delta: dead_score - fixed_score,
        live_natural_delta: live_natural_score - fixed_score,
    }
}

fn evaluate_lattice(burn_in: usize, samples: usize) -> Vec<EvidenceRow> {
    LATTICE_ALPHA_GRID
        .iter()
        .map(|alpha| evaluate_lattice_at(*alpha, burn_in, samples))
        .collect()
}

// ---------------------------------------------------------------------
// FHN coupling-band live-DOF evaluation. T21 hypothesis: the natural DOF
// on FHN is the coupling band, not λᵢ. The "dead" mode is the T08 "live"
// mode re-purposed — a static `lambda: 0.99` perturbation of the parent
// FhnSubstrate. FHN's T08 evidence (`dead_delta: +2.45`, `live_delta:
// +1.91` at `alpha=0.99`) tells us to expect the dead mode to fail the
// null criterion; FhnCoupling's lift-or-failure is what the pivot tests.
// ---------------------------------------------------------------------

fn evaluate_fhn_coupling_at(eta_coupling: f64, burn_in: usize, samples: usize) -> EvidenceRow {
    // Fixed: FhnSubstrate, no plasticity perturbation beyond defaults.
    let fixed_config = FhnConfig {
        size: 8,
        epsilon: 0.08,
        coupling: 0.2,
        i_ext: 0.5,
        i_ext_noise: 0.1,
        lambda: 0.95,
        seed: FHN_SEED_BASE,
    };
    let fixed_score = metric_score(&collect_metrics(
        &mut FhnSubstrate::new(fixed_config),
        FHN_SEED_BASE + 100,
        burn_in,
        samples,
    ));

    // Dead: FhnSubstrate with the *other* DOF (lambda) statically
    // perturbed. Re-purposed from the T08 "live" mode. T08 evidence at
    // alpha=0.99: live_delta=+1.91 → this dead mode fails the null test
    // for FHN. The pivot records that explicitly.
    let dead_config = FhnConfig {
        lambda: 0.99,
        ..fixed_config
    };
    let dead_score = metric_score(&collect_metrics(
        &mut FhnSubstrate::new(dead_config),
        FHN_SEED_BASE + 101,
        burn_in,
        samples,
    ));

    // Live-natural: FhnCouplingSubstrate sweeping the natural-DOF knob.
    let live_config = FhnCouplingConfig {
        size: 8,
        epsilon: 0.08,
        i_ext: 0.5,
        i_ext_noise: 0.1,
        lambda: 0.95,
        eta_coupling,
        coupling_leak: 0.01,
        coupling_min: 0.0,
        coupling_max: 1.0,
        seed: FHN_COUPLING_SEED_BASE,
    };
    let live_natural_score = metric_score(&collect_metrics(
        &mut FhnCouplingSubstrate::new(live_config),
        FHN_COUPLING_SEED_BASE + 200,
        burn_in,
        samples,
    ));

    EvidenceRow {
        substrate: SubstrateKind::Fhn,
        knob: eta_coupling,
        fixed_score,
        dead_score,
        live_natural_score,
        dead_delta: dead_score - fixed_score,
        live_natural_delta: live_natural_score - fixed_score,
    }
}

fn evaluate_fhn_coupling(burn_in: usize, samples: usize) -> Vec<EvidenceRow> {
    FHN_COUPLING_GRID
        .iter()
        .map(|eta| evaluate_fhn_coupling_at(*eta, burn_in, samples))
        .collect()
}

// ---------------------------------------------------------------------
// T22 follow-on: FHN re-tested with the T08-faithful asymmetric Hebbian
// per-edge coupling rule. The fixed/dead baselines are *identical* to
// the T21 FHN row (`FhnSubstrate` with default and `lambda=0.99`),
// which makes the live Δ directly comparable to the T21 baseline and
// is what the spec's C1 "T21 replay" assertion depends on. See
// `tasks/T22-fhn-asymmetric-coupling-hebbian.md`.
// ---------------------------------------------------------------------

fn evaluate_fhn_coupling_hebbian_at(
    eta_coupling: f64,
    burn_in: usize,
    samples: usize,
) -> EvidenceRow {
    // Fixed: FhnSubstrate, no plasticity perturbation beyond defaults.
    let fixed_config = FhnConfig {
        size: 8,
        epsilon: 0.08,
        coupling: 0.2,
        i_ext: 0.5,
        i_ext_noise: 0.1,
        lambda: 0.95,
        seed: FHN_SEED_BASE,
    };
    let fixed_score = metric_score(&collect_metrics(
        &mut FhnSubstrate::new(fixed_config),
        FHN_SEED_BASE + 100,
        burn_in,
        samples,
    ));

    // Dead: FhnSubstrate with the *other* DOF (lambda) statically
    // perturbed, identical to the T21 FHN row.
    let dead_config = FhnConfig {
        lambda: 0.99,
        ..fixed_config
    };
    let dead_score = metric_score(&collect_metrics(
        &mut FhnSubstrate::new(dead_config),
        FHN_SEED_BASE + 101,
        burn_in,
        samples,
    ));

    // Live-natural: FhnCouplingHebbianSubstrate sweeping the natural-DOF
    // knob. The `+200` offset on the RNG seed is the project convention
    // for the live-DOF mode (matches the FHN row and the other
    // substrates in this file).
    let live_config = FhnCouplingHebbianConfig {
        size: 8,
        epsilon: 0.08,
        i_ext: 0.5,
        i_ext_noise: 0.1,
        lambda: 0.95,
        eta_coupling,
        tau_e: 0.5,
        beta_oja: 0.01,
        w_max: 1.0,
        w_init: 0.2,
        seed: FHN_COUPLING_HEBBIAN_SEED_BASE,
    };
    let live_natural_score = metric_score(&collect_metrics(
        &mut FhnCouplingHebbianSubstrate::new(live_config),
        FHN_COUPLING_HEBBIAN_SEED_BASE + 200,
        burn_in,
        samples,
    ));

    EvidenceRow {
        substrate: SubstrateKind::FhnHebbian,
        knob: eta_coupling,
        fixed_score,
        dead_score,
        live_natural_score,
        dead_delta: dead_score - fixed_score,
        live_natural_delta: live_natural_score - fixed_score,
    }
}

fn evaluate_fhn_coupling_hebbian(burn_in: usize, samples: usize) -> Vec<EvidenceRow> {
    FHN_COUPLING_HEBBIAN_GRID
        .iter()
        .map(|eta| evaluate_fhn_coupling_hebbian_at(*eta, burn_in, samples))
        .collect()
}

// ---------------------------------------------------------------------
// Oscillator intrinsic-frequency live-DOF evaluation. T21 hypothesis:
// the natural DOF on the Kuramoto-style oscillator is the per-cell
// intrinsic omega, not the phase-memory retention band. The "dead" mode
// is OscillatorPlasticity::Hebbian (coupling-weight Hebbian, the T08
// dead mode). T08 oscillator `dead_delta` was negative across the
// alpha grid → expected to be null.
// ---------------------------------------------------------------------

fn evaluate_oscillator_frequency_at(eta_omega: f64, burn_in: usize, samples: usize) -> EvidenceRow {
    let base = OscillatorConfig {
        size: 8,
        coupling: 2.4,
        base_frequency: 1.0,
        frequency_spread: 0.35,
        noise_scale: 0.05,
        phase_memory_lambda: 0.93,
        dt: 0.05,
        plasticity: OscillatorPlasticity::Fixed,
        seed: OSCILLATOR_SEED_BASE,
    };
    let fixed_score = metric_score(&collect_metrics(
        &mut OscillatorSubstrate::new(base),
        OSCILLATOR_SEED_BASE + 100,
        burn_in,
        samples,
    ));

    let dead_config = OscillatorConfig {
        plasticity: OscillatorPlasticity::Hebbian {
            eta: 0.02,
            lambda_w: 0.01,
            w_max: 2.0,
        },
        ..base
    };
    let dead_score = metric_score(&collect_metrics(
        &mut OscillatorSubstrate::new(dead_config),
        OSCILLATOR_SEED_BASE + 101,
        burn_in,
        samples,
    ));

    let live_config = OscillatorFrequencyConfig {
        size: 8,
        base_coupling: 2.4,
        base_omega: 1.0,
        omega_spread: 0.35,
        noise_scale: 0.05,
        phase_memory_lambda: 0.93,
        dt: 0.05,
        eta_omega,
        omega_min: 0.5,
        omega_max: 2.0,
        seed: OSCILLATOR_FREQUENCY_SEED_BASE,
    };
    let live_natural_score = metric_score(&collect_metrics(
        &mut OscillatorFrequencySubstrate::new(live_config),
        OSCILLATOR_FREQUENCY_SEED_BASE + 200,
        burn_in,
        samples,
    ));

    EvidenceRow {
        substrate: SubstrateKind::Oscillator,
        knob: eta_omega,
        fixed_score,
        dead_score,
        live_natural_score,
        dead_delta: dead_score - fixed_score,
        live_natural_delta: live_natural_score - fixed_score,
    }
}

fn evaluate_oscillator_frequency(burn_in: usize, samples: usize) -> Vec<EvidenceRow> {
    OSCILLATOR_OMEGA_GRID
        .iter()
        .map(|eta| evaluate_oscillator_frequency_at(*eta, burn_in, samples))
        .collect()
}

// ---------------------------------------------------------------------
// Conductance retention-band live-DOF evaluation. T21 hypothesis: the
// natural DOF is the retention band (correct DOF, wrong parameterisation
// in T08). The "dead" mode is ConductancePlasticity::HebbianConductance
// (coupling-weight Hebbian, the T08 dead mode). The T08 fixture
// `eta_lambda=0.03` is the rightmost grid point so the pivot explicitly
// brackets the T08 failure mode.
// ---------------------------------------------------------------------

fn evaluate_conductance_retention_at(
    eta_lambda: f64,
    burn_in: usize,
    samples: usize,
) -> EvidenceRow {
    let base = ConductanceConfig {
        size: 8,
        coupling_gain: 1.35,
        base_retention: 0.95,
        noise_scale: 0.05,
        activity_clip: 4.0,
        plasticity: ConductancePlasticity::Fixed,
        seed: CONDUCTANCE_SEED_BASE,
    };
    let fixed_score = metric_score(&collect_metrics(
        &mut ConductanceSubstrate::new(base),
        CONDUCTANCE_SEED_BASE + 100,
        burn_in,
        samples,
    ));

    let dead_config = ConductanceConfig {
        plasticity: ConductancePlasticity::HebbianConductance {
            eta: 0.02,
            leak: 0.01,
            min_weight: 0.4,
            max_weight: 1.6,
        },
        ..base
    };
    let dead_score = metric_score(&collect_metrics(
        &mut ConductanceSubstrate::new(dead_config),
        CONDUCTANCE_SEED_BASE + 101,
        burn_in,
        samples,
    ));

    let live_config = ConductanceRetentionConfig {
        size: 8,
        coupling_gain: 1.35,
        base_retention: 0.95,
        noise_scale: 0.05,
        activity_clip: 4.0,
        eta_lambda,
        lambda_min: 0.5,
        lambda_max: 0.99,
        seed: CONDUCTANCE_RETENTION_SEED_BASE,
    };
    let live_natural_score = metric_score(&collect_metrics(
        &mut ConductanceRetentionSubstrate::new(live_config),
        CONDUCTANCE_RETENTION_SEED_BASE + 200,
        burn_in,
        samples,
    ));

    EvidenceRow {
        substrate: SubstrateKind::Conductance,
        knob: eta_lambda,
        fixed_score,
        dead_score,
        live_natural_score,
        dead_delta: dead_score - fixed_score,
        live_natural_delta: live_natural_score - fixed_score,
    }
}

fn evaluate_conductance_retention(burn_in: usize, samples: usize) -> Vec<EvidenceRow> {
    CONDUCTANCE_LAMBDA_GRID
        .iter()
        .map(|eta| evaluate_conductance_retention_at(*eta, burn_in, samples))
        .collect()
}

fn pick_best(rows: &[EvidenceRow], substrate: SubstrateKind) -> Option<SubstrateSummary> {
    let best = rows
        .iter()
        .filter(|row| row.substrate == substrate)
        .max_by(|left, right| left.live_natural_delta.total_cmp(&right.live_natural_delta))?;

    let dead_null = best.dead_delta <= DEAD_NULL_THRESHOLD;
    let live_lift = best.live_natural_delta >= LIVE_LIFT_THRESHOLD;

    Some(SubstrateSummary {
        substrate,
        knob: best.knob,
        fixed_score: best.fixed_score,
        dead_score: best.dead_score,
        live_natural_score: best.live_natural_score,
        dead_delta: best.dead_delta,
        live_natural_delta: best.live_natural_delta,
        dead_null,
        live_lift,
        classification: Classification::from_flags(live_lift, dead_null),
    })
}

pub fn run_pivot(samples: usize, burn_in: usize) -> PivotOutcome {
    let mut rows = Vec::new();
    rows.extend(evaluate_lattice(burn_in, samples));
    rows.extend(evaluate_fhn_coupling(burn_in, samples));
    rows.extend(evaluate_fhn_coupling_hebbian(burn_in, samples));
    rows.extend(evaluate_oscillator_frequency(burn_in, samples));
    rows.extend(evaluate_conductance_retention(burn_in, samples));

    let kinds = [
        SubstrateKind::Lattice,
        SubstrateKind::Fhn,
        SubstrateKind::FhnHebbian,
        SubstrateKind::Oscillator,
        SubstrateKind::Conductance,
    ];
    let summaries: Vec<SubstrateSummary> = kinds
        .iter()
        .filter_map(|kind| pick_best(&rows, *kind))
        .collect();

    let lattice_pass = summaries
        .iter()
        .find(|summary| summary.substrate == SubstrateKind::Lattice)
        .is_some_and(|summary| summary.live_lift && summary.dead_null);

    let non_lattice_pass_count = summaries
        .iter()
        .filter(|summary| {
            summary.substrate.is_non_lattice()
                && summary.classification == Classification::LiftWithRightDof
        })
        .count();

    // NO-GO is reserved for substrate-independent methodology failure:
    // the lattice control itself fails the lift+null condition.
    let verdict = if !lattice_pass {
        Verdict::NoGo
    } else if non_lattice_pass_count >= 2 {
        Verdict::Go
    } else {
        Verdict::Pivot
    };

    PivotOutcome {
        summaries,
        rows,
        verdict,
        non_lattice_pass_count,
    }
}

fn write_evidence_csv(path: &PathBuf, rows: &[EvidenceRow]) -> Result<()> {
    let mut out = String::from(
        "substrate,dof_kind,goldilocks_knob,knob_value,fixed_score,dead_score,live_natural_score,dead_delta,live_natural_delta\n",
    );
    for row in rows {
        writeln!(
            &mut out,
            "{},{},{},{:.6},{:.8},{:.8},{:.8},{:.8},{:.8}",
            row.substrate.as_str(),
            row.substrate.dof_kind().as_str(),
            row.substrate.goldilocks_knob(),
            row.knob,
            row.fixed_score,
            row.dead_score,
            row.live_natural_score,
            row.dead_delta,
            row.live_natural_delta
        )
        .map_err(|_| anyhow!("failed to format evidence csv row"))?;
    }
    fs::write(path, out).with_context(|| format!("failed to write {}", path.display()))
}

fn write_summary_json(path: &PathBuf, outcome: &PivotOutcome, args: &PivotArgs) -> Result<()> {
    let mut summaries_json = String::new();
    for (index, summary) in outcome.summaries.iter().enumerate() {
        if index > 0 {
            summaries_json.push(',');
        }
        write!(
            &mut summaries_json,
            "{{\"substrate\":\"{}\",\"dof_kind\":\"{}\",\"goldilocks_knob\":\"{}\",\"knob\":{:.6},\"fixed_score\":{:.8},\"dead_score\":{:.8},\"live_natural_score\":{:.8},\"dead_delta\":{:.8},\"live_natural_delta\":{:.8},\"dead_null\":{},\"live_lift\":{},\"classification\":\"{}\"}}",
            summary.substrate.as_str(),
            summary.substrate.dof_kind().as_str(),
            summary.substrate.goldilocks_knob(),
            summary.knob,
            summary.fixed_score,
            summary.dead_score,
            summary.live_natural_score,
            summary.dead_delta,
            summary.live_natural_delta,
            summary.dead_null,
            summary.live_lift,
            summary.classification.as_str(),
        )
        .map_err(|_| anyhow!("failed to format pivot summary json"))?;
    }

    let payload = format!(
        "{{\"verdict\":\"{}\",\"arch\":\"{}\",\"samples\":{},\"burn_in\":{},\"live_lift_threshold\":{},\"dead_null_threshold\":{},\"non_lattice_pass_count\":{},\"summaries\":[{}]}}",
        outcome.verdict.as_str(),
        std::env::consts::ARCH,
        args.samples,
        args.burn_in,
        LIVE_LIFT_THRESHOLD,
        DEAD_NULL_THRESHOLD,
        outcome.non_lattice_pass_count,
        summaries_json
    );

    fs::write(path, payload).with_context(|| format!("failed to write {}", path.display()))
}

fn write_decision_markdown(path: &PathBuf, outcome: &PivotOutcome, args: &PivotArgs) -> Result<()> {
    let mut decision = String::new();
    decision.push_str("# T21 Pivot Decision\n\n");
    writeln!(&mut decision, "- Verdict: **{}**", outcome.verdict.as_str())
        .map_err(|_| anyhow!("failed to format decision markdown"))?;
    writeln!(
        &mut decision,
        "- Non-lattice substrates passing re-GO criterion: `{}`",
        outcome.non_lattice_pass_count
    )
    .map_err(|_| anyhow!("failed to format decision markdown"))?;
    writeln!(&mut decision, "- Host arch: `{}`", std::env::consts::ARCH)
        .map_err(|_| anyhow!("failed to format decision markdown"))?;
    writeln!(&mut decision, "- Samples: `{}`", args.samples)
        .map_err(|_| anyhow!("failed to format decision markdown"))?;
    writeln!(&mut decision, "- Burn-in: `{}`", args.burn_in)
        .map_err(|_| anyhow!("failed to format decision markdown"))?;
    writeln!(
        &mut decision,
        "- Live lift threshold: `{LIVE_LIFT_THRESHOLD:.4}`"
    )
    .map_err(|_| anyhow!("failed to format decision markdown"))?;
    writeln!(
        &mut decision,
        "- Dead-term null threshold: `{DEAD_NULL_THRESHOLD:.4}`\n"
    )
    .map_err(|_| anyhow!("failed to format decision markdown"))?;

    decision.push_str("## Per-substrate classification\n\n");
    decision.push_str(
        "| Substrate | DOF kind | Goldilocks knob | Knob value | Fixed | Dead | Live (natural) | Dead Δ | Live Δ | Dead null | Live lift | Classification |\n",
    );
    decision.push_str(
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | :---: | :---: | --- |\n",
    );
    for summary in &outcome.summaries {
        writeln!(
            &mut decision,
            "| {} | {} | {} | {:.4} | {:.6} | {:.6} | {:.6} | {:.6} | {:.6} | {} | {} | {} |\n",
            summary.substrate.as_str(),
            summary.substrate.dof_kind().as_str(),
            summary.substrate.goldilocks_knob(),
            summary.knob,
            summary.fixed_score,
            summary.dead_score,
            summary.live_natural_score,
            summary.dead_delta,
            summary.live_natural_delta,
            if summary.dead_null { "yes" } else { "no" },
            if summary.live_lift { "yes" } else { "no" },
            summary.classification.as_str(),
        )
        .map_err(|_| anyhow!("failed to format evidence row"))?;
    }

    decision.push_str("\n## Rationale\n\n");
    match outcome.verdict {
        Verdict::Go => {
            decision.push_str(
                "The per-substrate DOF characterisation found ≥2 non-lattice substrates in their **natural** live DOF (λᵢ or otherwise) satisfying the lift + dead-null criterion. Phase 3 dispatch unblocks; the machine model may use per-substrate live-DOF construction rather than a single-axis λᵢ.\n",
            );
        }
        Verdict::Pivot => {
            decision.push_str(
                "Fewer than 2 non-lattice substrates satisfy the re-GO criterion in their natural DOF. The single-axis λᵢ claim is **not** universal; per-substrate live-DOF characterisation is the prerequisite. Phase 3 must be re-scoped to a substrate-specific machine model family.\n",
            );
        }
        Verdict::NoGo => {
            decision.push_str(
                "The lattice control itself fails the lift + null condition. The methodology is broken (or the T08 result was substrate-specific in a way the pivot does not reproduce). The underlying growth claim is in question; the project should pause for human review before any Phase 3 work.\n",
            );
        }
    }

    fs::write(path, decision).with_context(|| format!("failed to write {}", path.display()))
}

pub fn run_pivot_command(args: &[OsString]) -> Result<()> {
    let params = parse_pivot_args(args)?;
    fs::create_dir_all(&params.output_dir).with_context(|| {
        format!(
            "failed to create output directory {}",
            params.output_dir.display()
        )
    })?;

    let outcome = run_pivot(params.samples, params.burn_in);

    let evidence_path = params.output_dir.join("t21_pivot_evidence.csv");
    let summary_path = params.output_dir.join("t21_pivot_summary.json");
    let decision_path = params.output_dir.join("decision.md");

    write_evidence_csv(&evidence_path, &outcome.rows)?;
    write_summary_json(&summary_path, &outcome, &params)?;
    write_decision_markdown(&decision_path, &outcome, &params)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{
        Classification, PivotArgs, SubstrateKind, Verdict, parse_pivot_args, run_pivot,
        write_evidence_csv,
    };
    use std::ffi::OsString;
    use std::path::PathBuf;

    #[test]
    fn parse_pivot_args_applies_defaults() {
        let parsed = parse_pivot_args(&[]);
        assert!(parsed.is_ok());
        let parsed = parsed.ok();
        assert_eq!(
            parsed
                .as_ref()
                .map(|params: &PivotArgs| params.output_dir.clone()),
            Some(PathBuf::from("results/t21_pivot"))
        );
        assert_eq!(parsed.as_ref().map(|params| params.samples), Some(160));
        assert_eq!(parsed.as_ref().map(|params| params.burn_in), Some(40));
    }

    #[test]
    fn parse_pivot_args_applies_overrides() {
        let args = vec![
            OsString::from("--output-dir"),
            OsString::from("tmp/pivot"),
            OsString::from("--samples"),
            OsString::from("80"),
            OsString::from("--burn-in"),
            OsString::from("20"),
        ];
        let parsed = parse_pivot_args(&args);
        assert!(parsed.is_ok());
        let parsed = parsed.ok();
        assert_eq!(
            parsed
                .as_ref()
                .map(|params: &PivotArgs| params.output_dir.clone()),
            Some(PathBuf::from("tmp/pivot"))
        );
        assert_eq!(parsed.as_ref().map(|params| params.samples), Some(80));
        assert_eq!(parsed.as_ref().map(|params| params.burn_in), Some(20));
    }

    #[test]
    fn parse_pivot_args_rejects_zero_samples() {
        let args = vec![OsString::from("--samples"), OsString::from("0")];
        let error = parse_pivot_args(&args);
        assert!(error.is_err());
    }

    #[test]
    fn parse_pivot_args_rejects_unknown_flag() {
        let args = vec![OsString::from("--nope"), OsString::from("1")];
        let error = parse_pivot_args(&args);
        assert!(error.is_err());
    }

    #[test]
    fn substrate_kind_natural_dof_matches_t21_taxonomy() {
        assert_eq!(SubstrateKind::Lattice.dof_kind(), super::DofKind::Retention);
        assert_eq!(SubstrateKind::Fhn.dof_kind(), super::DofKind::Coupling);
        // T22: FhnHebbian's natural DOF is Coupling (same DOF as Fhn,
        // different mechanism — asymmetric Hebbian per-edge, not a
        // uniform gain knob).
        assert_eq!(
            SubstrateKind::FhnHebbian.dof_kind(),
            super::DofKind::Coupling,
        );
        assert_eq!(
            SubstrateKind::Oscillator.dof_kind(),
            super::DofKind::Frequency
        );
        assert_eq!(
            SubstrateKind::Conductance.dof_kind(),
            super::DofKind::Retention
        );
    }

    #[test]
    fn classification_is_lift_with_right_dof_when_live_lifts_and_dead_null() {
        let class = Classification::from_flags(true, true);
        assert_eq!(class, Classification::LiftWithRightDof);
    }

    #[test]
    fn classification_is_lift_in_other_dof_when_live_lifts_but_dead_also_lifts() {
        let class = Classification::from_flags(true, false);
        assert_eq!(class, Classification::LiftInOtherDof);
    }

    #[test]
    fn classification_is_no_lift_when_live_does_not_lift() {
        let class_no_null = Classification::from_flags(false, true);
        let class_lifted_dead = Classification::from_flags(false, false);
        assert_eq!(class_no_null, Classification::NoLift);
        assert_eq!(class_lifted_dead, Classification::NoLift);
    }

    #[test]
    fn run_pivot_smoke_emits_one_summary_per_substrate() {
        // Cheap smoke: confirm the wiring runs end-to-end and produces
        // the expected 5-substrate summary (Lattice + Fhn +
        // FhnHebbian [T22] + Oscillator + Conductance). Per-substrate
        // lift/null assertions are not pinned here because the lattice
        // control's dead-delta sign depends on sample/burn-in ratio
        // (T08 evidence: dead_delta = -0.495 at 160/40). Real
        // coverage is the actual pivot run + T21/T22 task-exit
        // assertions.
        let outcome = run_pivot(8, 4);
        assert_eq!(outcome.summaries.len(), 5);
        for kind in [
            SubstrateKind::Lattice,
            SubstrateKind::Fhn,
            SubstrateKind::FhnHebbian,
            SubstrateKind::Oscillator,
            SubstrateKind::Conductance,
        ] {
            assert!(
                outcome
                    .summaries
                    .iter()
                    .any(|summary| summary.substrate == kind),
                "missing summary for {kind:?}",
            );
        }
    }

    #[test]
    fn verdict_strings_match_spec() {
        assert_eq!(Verdict::Go.as_str(), "GO");
        assert_eq!(Verdict::Pivot.as_str(), "PIVOT");
        assert_eq!(Verdict::NoGo.as_str(), "NO-GO");
    }

    #[test]
    fn evidence_csv_emits_full_knob_name() {
        // Regression: the CSV writer was passing `goldilocks_knob()`
        // through `{:.6}`, which silently truncates `&str` to its first
        // 6 chars (e.g. `eta_coupling` -> `eta_co`). Pin the full
        // identifier so a future format-string edit cannot silently
        // rename columns in the evidence file.
        let outcome = run_pivot(8, 4);
        let dir = std::env::temp_dir().join(format!("lethe-pivot-test-{}", std::process::id()));
        std::fs::create_dir_all(&dir).ok();
        let csv_path = dir.join("evidence.csv");
        write_evidence_csv(&csv_path, &outcome.rows).ok();
        let body = std::fs::read_to_string(&csv_path).ok();
        let body = body.unwrap_or_default();
        std::fs::remove_dir_all(&dir).ok();
        assert!(
            body.contains("eta_coupling"),
            "expected `eta_coupling` (full identifier) in evidence csv, got:\n{body}",
        );
        assert!(
            body.contains("eta_omega"),
            "expected `eta_omega` (full identifier) in evidence csv, got:\n{body}",
        );
        assert!(
            body.contains("eta_lambda"),
            "expected `eta_lambda` (full identifier) in evidence csv, got:\n{body}",
        );
    }
}
