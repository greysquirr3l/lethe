use std::ffi::OsString;
use std::fmt::Write as _;
use std::fs;
use std::path::PathBuf;

use anyhow::{Context, Result, anyhow, bail};
use lethe_core::{Observer, ObserverConfig, ObserverMetrics, StateTrace, Substrate};
use lethe_model::seeded_micro_experiment;
use lethe_substrates::{
    CONDUCTANCE_SEED_BASE, ConductanceConfig, ConductancePlasticity, ConductanceSubstrate,
    FHN_SEED_BASE, FhnConfig, FhnSubstrate, LatticeConfig, LatticePlasticity, LatticeSubstrate,
    OSCILLATOR_SEED_BASE, OscillatorConfig, OscillatorPlasticity, OscillatorSubstrate,
};
use rand_chacha::ChaCha8Rng;
use rand_core::SeedableRng;

mod pivot;

const DEFAULT_GATE_SAMPLES: usize = 160;
const DEFAULT_GATE_BURN_IN: usize = 40;
const LIVE_LIFT_THRESHOLD: f64 = 0.01;
const DEAD_NULL_THRESHOLD: f64 = 0.01;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Verdict {
    Go,
    Pivot,
}

impl Verdict {
    const fn as_str(self) -> &'static str {
        match self {
            Self::Go => "GO",
            Self::Pivot => "PIVOT",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SubstrateKind {
    Lattice,
    Fhn,
    Oscillator,
    Conductance,
}

impl SubstrateKind {
    const fn as_str(self) -> &'static str {
        match self {
            Self::Lattice => "lattice",
            Self::Fhn => "fhn",
            Self::Oscillator => "oscillator",
            Self::Conductance => "conductance",
        }
    }

    const fn is_non_lattice(self) -> bool {
        !matches!(self, Self::Lattice)
    }
}

#[derive(Debug, Clone)]
struct GateArgs {
    output_dir: PathBuf,
    samples: usize,
    burn_in: usize,
}

#[derive(Debug, Clone)]
struct EvidenceRow {
    substrate: SubstrateKind,
    alpha_analogue: f64,
    fixed_score: f64,
    dead_score: f64,
    live_score: f64,
    dead_delta: f64,
    live_delta: f64,
}

#[derive(Debug, Clone)]
struct SubstrateSummary {
    substrate: SubstrateKind,
    alpha_analogue: f64,
    fixed_score: f64,
    dead_score: f64,
    live_score: f64,
    dead_delta: f64,
    live_delta: f64,
    dead_null: bool,
    live_lift: bool,
}

#[derive(Debug, Clone)]
struct GateOutcome {
    summaries: Vec<SubstrateSummary>,
    rows: Vec<EvidenceRow>,
    verdict: Verdict,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct ReproArgs {
    seed: u64,
    steps: usize,
    output: PathBuf,
}

fn parse_repro_args(args: &[OsString]) -> Result<ReproArgs> {
    let mut seed: Option<u64> = None;
    let mut steps: Option<usize> = None;
    let mut output: Option<PathBuf> = None;

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
            "--seed" => {
                seed = Some(
                    value
                        .parse::<u64>()
                        .context("--seed must be a non-negative integer")?,
                );
            }
            "--steps" => {
                steps = Some(
                    value
                        .parse::<usize>()
                        .context("--steps must be a non-negative integer")?,
                );
            }
            "--output" => {
                output = Some(PathBuf::from(value));
            }
            other => bail!("unsupported argument: {other}"),
        }

        i += 2;
    }

    let seed = seed.context("missing required argument: --seed")?;
    let steps = steps.context("missing required argument: --steps")?;
    let output = output.context("missing required argument: --output")?;

    Ok(ReproArgs {
        seed,
        steps,
        output,
    })
}

fn run_repro_command(args: &[OsString]) -> Result<()> {
    let params = parse_repro_args(args)?;
    let bytes = seeded_micro_experiment(params.seed, params.steps);
    fs::write(&params.output, bytes).with_context(|| {
        format!(
            "failed to write reproducibility output to {}",
            params.output.display()
        )
    })?;
    Ok(())
}

fn parse_gate_args(args: &[OsString]) -> Result<GateArgs> {
    let mut output_dir = PathBuf::from("results");
    let mut samples = DEFAULT_GATE_SAMPLES;
    let mut burn_in = DEFAULT_GATE_BURN_IN;

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

    Ok(GateArgs {
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

fn evaluate_lattice(alpha: f64, burn_in: usize, samples: usize) -> EvidenceRow {
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
    let live_score = metric_score(&collect_metrics(
        &mut LatticeSubstrate::new(live_config, FHN_SEED_BASE + 4),
        FHN_SEED_BASE + 5,
        burn_in,
        samples,
    ));

    EvidenceRow {
        substrate: SubstrateKind::Lattice,
        alpha_analogue: alpha,
        fixed_score,
        dead_score,
        live_score,
        dead_delta: dead_score - fixed_score,
        live_delta: live_score - fixed_score,
    }
}

fn evaluate_fhn(alpha: f64, burn_in: usize, samples: usize) -> EvidenceRow {
    let fixed_config = FhnConfig {
        size: 8,
        epsilon: alpha,
        coupling: 0.2,
        i_ext: 0.5,
        i_ext_noise: 0.1,
        lambda: 0.95,
        seed: FHN_SEED_BASE,
    };
    let fixed_score = metric_score(&collect_metrics(
        &mut FhnSubstrate::new(fixed_config),
        FHN_SEED_BASE + 11,
        burn_in,
        samples,
    ));

    let dead_config = FhnConfig {
        coupling: 0.25,
        ..fixed_config
    };
    let dead_score = metric_score(&collect_metrics(
        &mut FhnSubstrate::new(dead_config),
        FHN_SEED_BASE + 12,
        burn_in,
        samples,
    ));

    let live_config = FhnConfig {
        lambda: 0.99,
        ..fixed_config
    };
    let live_score = metric_score(&collect_metrics(
        &mut FhnSubstrate::new(live_config),
        FHN_SEED_BASE + 13,
        burn_in,
        samples,
    ));

    EvidenceRow {
        substrate: SubstrateKind::Fhn,
        alpha_analogue: alpha,
        fixed_score,
        dead_score,
        live_score,
        dead_delta: dead_score - fixed_score,
        live_delta: live_score - fixed_score,
    }
}

fn evaluate_oscillator(alpha: f64, burn_in: usize, samples: usize) -> EvidenceRow {
    let base = OscillatorConfig {
        size: 8,
        coupling: 2.4,
        base_frequency: 1.0,
        frequency_spread: 0.35,
        noise_scale: 0.05,
        phase_memory_lambda: alpha,
        dt: 0.05,
        plasticity: OscillatorPlasticity::Fixed,
        seed: OSCILLATOR_SEED_BASE,
    };
    let fixed_score = metric_score(&collect_metrics(
        &mut OscillatorSubstrate::new(base),
        OSCILLATOR_SEED_BASE + 1,
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
        OSCILLATOR_SEED_BASE + 2,
        burn_in,
        samples,
    ));

    let live_config = OscillatorConfig {
        plasticity: OscillatorPlasticity::AdaptiveLambda {
            eta_lambda: 0.04,
            lambda_min: 0.5,
            lambda_max: 0.99,
        },
        ..base
    };
    let live_score = metric_score(&collect_metrics(
        &mut OscillatorSubstrate::new(live_config),
        OSCILLATOR_SEED_BASE + 3,
        burn_in,
        samples,
    ));

    EvidenceRow {
        substrate: SubstrateKind::Oscillator,
        alpha_analogue: alpha,
        fixed_score,
        dead_score,
        live_score,
        dead_delta: dead_score - fixed_score,
        live_delta: live_score - fixed_score,
    }
}

fn evaluate_conductance(alpha: f64, burn_in: usize, samples: usize) -> EvidenceRow {
    let base = ConductanceConfig {
        size: 8,
        coupling_gain: 1.35,
        base_retention: alpha,
        noise_scale: 0.05,
        activity_clip: 4.0,
        plasticity: ConductancePlasticity::Fixed,
        seed: CONDUCTANCE_SEED_BASE,
    };
    let fixed_score = metric_score(&collect_metrics(
        &mut ConductanceSubstrate::new(base),
        CONDUCTANCE_SEED_BASE + 1,
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
        CONDUCTANCE_SEED_BASE + 2,
        burn_in,
        samples,
    ));

    let live_config = ConductanceConfig {
        plasticity: ConductancePlasticity::AdaptiveRetention {
            eta_lambda: 0.03,
            lambda_min: 0.5,
            lambda_max: 0.99,
        },
        ..base
    };
    let live_score = metric_score(&collect_metrics(
        &mut ConductanceSubstrate::new(live_config),
        CONDUCTANCE_SEED_BASE + 3,
        burn_in,
        samples,
    ));

    EvidenceRow {
        substrate: SubstrateKind::Conductance,
        alpha_analogue: alpha,
        fixed_score,
        dead_score,
        live_score,
        dead_delta: dead_score - fixed_score,
        live_delta: live_score - fixed_score,
    }
}

fn pick_best(rows: &[EvidenceRow], substrate: SubstrateKind) -> Option<SubstrateSummary> {
    let best = rows
        .iter()
        .filter(|row| row.substrate == substrate)
        .max_by(|left, right| left.live_delta.total_cmp(&right.live_delta))?;

    let dead_null = best.dead_delta <= DEAD_NULL_THRESHOLD;
    let live_lift = best.live_delta >= LIVE_LIFT_THRESHOLD;

    Some(SubstrateSummary {
        substrate,
        alpha_analogue: best.alpha_analogue,
        fixed_score: best.fixed_score,
        dead_score: best.dead_score,
        live_score: best.live_score,
        dead_delta: best.dead_delta,
        live_delta: best.live_delta,
        dead_null,
        live_lift,
    })
}

fn run_gate(samples: usize, burn_in: usize) -> GateOutcome {
    let alpha_grid = [0.85, 0.90, 0.95, 0.99];
    let mut rows = Vec::new();

    for alpha in alpha_grid {
        rows.push(evaluate_lattice(alpha, burn_in, samples));
        rows.push(evaluate_fhn(alpha, burn_in, samples));
        rows.push(evaluate_oscillator(alpha, burn_in, samples));
        rows.push(evaluate_conductance(alpha, burn_in, samples));
    }

    let kinds = [
        SubstrateKind::Lattice,
        SubstrateKind::Fhn,
        SubstrateKind::Oscillator,
        SubstrateKind::Conductance,
    ];
    let summaries: Vec<SubstrateSummary> = kinds
        .iter()
        .filter_map(|kind| pick_best(&rows, *kind))
        .collect();

    let non_lattice_confirmed = summaries
        .iter()
        .filter(|summary| {
            summary.substrate.is_non_lattice() && summary.live_lift && summary.dead_null
        })
        .count();

    let verdict = if non_lattice_confirmed >= 2 {
        Verdict::Go
    } else {
        Verdict::Pivot
    };

    GateOutcome {
        summaries,
        rows,
        verdict,
    }
}

fn write_evidence_csv(path: &PathBuf, rows: &[EvidenceRow]) -> Result<()> {
    let mut out = String::from(
        "substrate,alpha_analogue,fixed_score,dead_score,live_score,dead_delta,live_delta\n",
    );
    for row in rows {
        writeln!(
            &mut out,
            "{},{:.6},{:.8},{:.8},{:.8},{:.8},{:.8}",
            row.substrate.as_str(),
            row.alpha_analogue,
            row.fixed_score,
            row.dead_score,
            row.live_score,
            row.dead_delta,
            row.live_delta
        )
        .map_err(|_| anyhow!("failed to format evidence csv row"))?;
    }
    fs::write(path, out).with_context(|| format!("failed to write {}", path.display()))
}

fn write_summary_json(path: &PathBuf, outcome: &GateOutcome, args: &GateArgs) -> Result<()> {
    let mut summaries_json = String::new();
    for (index, summary) in outcome.summaries.iter().enumerate() {
        if index > 0 {
            summaries_json.push(',');
        }
        write!(
            &mut summaries_json,
            "{{\"substrate\":\"{}\",\"alpha_analogue\":{:.6},\"fixed_score\":{:.8},\"dead_score\":{:.8},\"live_score\":{:.8},\"dead_delta\":{:.8},\"live_delta\":{:.8},\"dead_null\":{},\"live_lift\":{}}}",
            summary.substrate.as_str(),
            summary.alpha_analogue,
            summary.fixed_score,
            summary.dead_score,
            summary.live_score,
            summary.dead_delta,
            summary.live_delta,
            summary.dead_null,
            summary.live_lift,
        )
        .map_err(|_| anyhow!("failed to format gate summary json"))?;
    }

    let payload = format!(
        "{{\"verdict\":\"{}\",\"arch\":\"{}\",\"samples\":{},\"burn_in\":{},\"live_lift_threshold\":{},\"dead_null_threshold\":{},\"summaries\":[{}]}}",
        outcome.verdict.as_str(),
        std::env::consts::ARCH,
        args.samples,
        args.burn_in,
        LIVE_LIFT_THRESHOLD,
        DEAD_NULL_THRESHOLD,
        summaries_json
    );

    fs::write(path, payload).with_context(|| format!("failed to write {}", path.display()))
}

fn write_decision_markdown(path: &PathBuf, outcome: &GateOutcome, args: &GateArgs) -> Result<()> {
    let mut decision = String::new();
    decision.push_str("# T08 Gate Decision\n\n");
    writeln!(&mut decision, "- Verdict: **{}**", outcome.verdict.as_str())
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

    decision.push_str("## Evidence\n\n");
    decision.push_str("| Substrate | Alpha Analogue | Fixed | Dead | Live | Dead Delta | Live Delta | Dead Null | Live Lift |\n");
    decision.push_str("| --- | ---: | ---: | ---: | ---: | ---: | ---: | :---: | :---: |\n");
    for summary in &outcome.summaries {
        writeln!(
            &mut decision,
            "| {} | {:.4} | {:.6} | {:.6} | {:.6} | {:.6} | {:.6} | {} | {} |\n",
            summary.substrate.as_str(),
            summary.alpha_analogue,
            summary.fixed_score,
            summary.dead_score,
            summary.live_score,
            summary.dead_delta,
            summary.live_delta,
            if summary.dead_null { "yes" } else { "no" },
            if summary.live_lift { "yes" } else { "no" }
        )
        .map_err(|_| anyhow!("failed to format evidence row"))?;
    }

    decision.push_str("\n## Rationale\n\n");
    match outcome.verdict {
        Verdict::Go => {
            decision.push_str("Live-term lambda_i lift is confirmed on at least two non-lattice substrates while dead-term coupling updates remain null on those same substrates.\n");
        }
        Verdict::Pivot => {
            decision.push_str("The non-lattice evidence does not yet satisfy the required lambda_i lift + dead-term null condition on at least two substrates. Phase 3 should remain paused and pivot work should scope per-substrate live DOFs.\n");
        }
    }

    fs::write(path, decision).with_context(|| format!("failed to write {}", path.display()))
}

fn run_gate_command(args: &[OsString]) -> Result<()> {
    let params = parse_gate_args(args)?;
    fs::create_dir_all(&params.output_dir).with_context(|| {
        format!(
            "failed to create output directory {}",
            params.output_dir.display()
        )
    })?;

    let outcome = run_gate(params.samples, params.burn_in);

    let evidence_path = params
        .output_dir
        .join("t08_generalisation_gate_evidence.csv");
    let summary_path = params
        .output_dir
        .join("t08_generalisation_gate_summary.json");
    let decision_path = params.output_dir.join("decision.md");

    write_evidence_csv(&evidence_path, &outcome.rows)?;
    write_summary_json(&summary_path, &outcome, &params)?;
    write_decision_markdown(&decision_path, &outcome, &params)?;
    Ok(())
}

fn run() -> Result<()> {
    let mut args = std::env::args_os();
    let _bin = args.next();

    match args.next().and_then(|value| value.into_string().ok()) {
        Some(command) if command == "repro" => {
            let remaining: Vec<OsString> = args.collect();
            run_repro_command(&remaining)
        }
        Some(command) if command == "gate" => {
            let remaining: Vec<OsString> = args.collect();
            run_gate_command(&remaining)
        }
        Some(command) if command == "pivot" => {
            let remaining: Vec<OsString> = args.collect();
            pivot::run_pivot_command(&remaining)
        }
        Some(command) => bail!("unsupported command: {command}"),
        None => bail!(
            "usage: lethe-cli repro --seed <u64> --steps <usize> --output <path> | gate [--output-dir <path>] [--samples <n>] [--burn-in <n>] | pivot [--output-dir <path>] [--samples <n>] [--burn-in <n>]"
        ),
    }
}

fn main() {
    if let Err(error) = run() {
        eprintln!("{error}");
        std::process::exit(1);
    }
}

#[cfg(test)]
mod tests {
    use std::ffi::OsString;
    use std::path::PathBuf;

    use super::{parse_gate_args, parse_repro_args};

    fn sample_args() -> Vec<OsString> {
        vec![
            OsString::from("--seed"),
            OsString::from("424242"),
            OsString::from("--steps"),
            OsString::from("1024"),
            OsString::from("--output"),
            OsString::from("repro.bin"),
        ]
    }

    #[test]
    fn parses_valid_repro_arguments() {
        let parsed_result = parse_repro_args(&sample_args());
        assert!(parsed_result.is_ok());
        let parsed = parsed_result.ok();
        assert_eq!(parsed.as_ref().map(|params| params.seed), Some(424_242));
        assert_eq!(parsed.as_ref().map(|params| params.steps), Some(1024));
        assert_eq!(
            parsed.as_ref().map(|params| params.output.clone()),
            Some(PathBuf::from("repro.bin"))
        );
    }

    #[test]
    fn rejects_missing_seed() {
        let args = vec![
            OsString::from("--steps"),
            OsString::from("1024"),
            OsString::from("--output"),
            OsString::from("repro.bin"),
        ];
        let error = parse_repro_args(&args);
        assert!(error.is_err());
    }

    #[test]
    fn rejects_unsupported_flag() {
        let args = vec![OsString::from("--unknown"), OsString::from("123")];
        let error = parse_repro_args(&args);
        assert!(error.is_err());
    }

    #[test]
    fn parses_gate_args_with_defaults() {
        let parsed_result = parse_gate_args(&[]);
        assert!(parsed_result.is_ok());
        let parsed = parsed_result.ok();
        assert_eq!(
            parsed.as_ref().map(|params| params.output_dir.clone()),
            Some(PathBuf::from("results"))
        );
    }

    #[test]
    fn parses_gate_args_with_overrides() {
        let args = vec![
            OsString::from("--output-dir"),
            OsString::from("tmp/gate"),
            OsString::from("--samples"),
            OsString::from("50"),
            OsString::from("--burn-in"),
            OsString::from("10"),
        ];
        let parsed_result = parse_gate_args(&args);
        assert!(parsed_result.is_ok());
        let parsed = parsed_result.ok();
        assert_eq!(
            parsed.as_ref().map(|params| params.output_dir.clone()),
            Some(PathBuf::from("tmp/gate"))
        );
        assert_eq!(parsed.as_ref().map(|params| params.samples), Some(50));
        assert_eq!(parsed.as_ref().map(|params| params.burn_in), Some(10));
    }
}
