use std::collections::HashMap;

use super::StateTrace;

const EULER_GAMMA: f64 = 0.577_215_664_901_532_9;
const LOG2_E: f64 = std::f64::consts::LOG2_E;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ObserverConfig {
    pub history_depth: usize,
    pub bin_count: usize,
    pub ksg_k: usize,
}

impl Default for ObserverConfig {
    fn default() -> Self {
        Self {
            history_depth: 4,
            bin_count: 8,
            ksg_k: 3,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct ObserverMetrics {
    pub ais_binning: f64,
    pub ais_ksg: f64,
    pub ais_ksg_ratio: f64,
    pub te: f64,
    pub tc: f64,
    pub delta: f64,
    pub sigma_lambda: f64,
    pub attractor_dimension: f64,
}

#[derive(Debug, Default, Clone, Copy)]
pub struct Observer {
    config: ObserverConfig,
}

impl Observer {
    #[must_use]
    pub const fn new(config: ObserverConfig) -> Self {
        Self { config }
    }

    #[must_use]
    pub const fn config(&self) -> ObserverConfig {
        self.config
    }

    #[must_use]
    pub fn observe(&self, trace: &StateTrace) -> ObserverMetrics {
        let quantized = QuantizedTrace::from_trace(trace, self.config.bin_count);
        let ais_binning = self.ais_binning(&quantized);
        let ais_ksg = self.ais_ksg(trace);
        let ais_ksg_ratio = if ais_binning > 0.0 {
            ais_ksg / ais_binning
        } else {
            0.0
        };

        ObserverMetrics {
            ais_binning,
            ais_ksg,
            ais_ksg_ratio,
            te: Self::transfer_entropy(&quantized),
            tc: Self::total_correlation(&quantized),
            delta: self.delta(trace),
            sigma_lambda: sigma_lambda(trace),
            attractor_dimension: attractor_dimension(trace),
        }
    }

    fn ais_binning(&self, quantized: &QuantizedTrace) -> f64 {
        let mut cell_sum = 0.0;
        let mut counted_cells = 0_usize;

        for series in &quantized.per_cell {
            if series.len() <= self.config.history_depth {
                continue;
            }
            cell_sum += ais_for_series(series, self.config.history_depth, self.config.bin_count);
            counted_cells += 1;
        }

        if counted_cells == 0 {
            return 0.0;
        }

        cell_sum / usize_to_f64(counted_cells)
    }

    fn ais_ksg(&self, trace: &StateTrace) -> f64 {
        let history_depth = self.config.history_depth;
        let k = self.config.ksg_k;
        let frames = trace.frames();
        if frames.len() <= history_depth + k {
            return 0.0;
        }

        let first_cell_count = frames
            .first()
            .map_or(0, |frame| frame.state.activities.len());
        if first_cell_count == 0 {
            return 0.0;
        }

        let mut score_sum = 0.0;
        let mut scored_cells = 0_usize;

        for cell_index in 0..first_cell_count {
            let mut x_history: Vec<Vec<f64>> = Vec::new();
            let mut y_current: Vec<f64> = Vec::new();

            for t in history_depth..frames.len() {
                let current_opt = frames
                    .get(t)
                    .and_then(|frame| frame.state.activities.get(cell_index))
                    .copied();
                let Some(current) = current_opt else {
                    continue;
                };

                let mut history = Vec::with_capacity(history_depth);
                let mut valid = true;
                for lag in 1..=history_depth {
                    let value_opt = frames
                        .get(t - lag)
                        .and_then(|frame| frame.state.activities.get(cell_index))
                        .copied();
                    if let Some(value) = value_opt {
                        history.push(value);
                    } else {
                        valid = false;
                        break;
                    }
                }

                if valid {
                    x_history.push(history);
                    y_current.push(current);
                }
            }

            let estimate = ksg_mi(&x_history, &y_current, k);
            if estimate.is_finite() && estimate > 0.0 {
                let depth_f = usize_to_f64(history_depth);
                let dimensionality_penalty = 0.4_f64.mul_add(depth_f, 1.0);
                score_sum += estimate / dimensionality_penalty;
                scored_cells += 1;
            }
        }

        if scored_cells == 0 {
            0.0
        } else {
            score_sum / usize_to_f64(scored_cells)
        }
    }

    fn transfer_entropy(quantized: &QuantizedTrace) -> f64 {
        let cell_count = quantized.per_cell.len();
        if cell_count < 2 {
            return 0.0;
        }

        let mut pair_sum = 0.0;
        let mut pair_count = 0_usize;

        for target_idx in 0..cell_count {
            for source_idx in 0..cell_count {
                if source_idx == target_idx {
                    continue;
                }

                let source_opt = quantized.per_cell.get(source_idx);
                let target_opt = quantized.per_cell.get(target_idx);
                let (Some(source), Some(target)) = (source_opt, target_opt) else {
                    continue;
                };

                let sample_len = source.len().min(target.len());
                if sample_len < 2 {
                    continue;
                }

                let mut triple_counts: HashMap<(u8, u8, u8), usize> = HashMap::new();
                let mut source_target_prev_counts: HashMap<(u8, u8), usize> = HashMap::new();
                let mut target_pair_counts: HashMap<(u8, u8), usize> = HashMap::new();
                let mut target_prev_counts: HashMap<u8, usize> = HashMap::new();

                for t in 1..sample_len {
                    let a_opt = target.get(t).copied();
                    let b_opt = target.get(t - 1).copied();
                    let c_opt = source.get(t - 1).copied();
                    let (Some(a), Some(b), Some(c)) = (a_opt, b_opt, c_opt) else {
                        continue;
                    };

                    *triple_counts.entry((a, b, c)).or_insert(0) += 1;
                    *source_target_prev_counts.entry((b, c)).or_insert(0) += 1;
                    *target_pair_counts.entry((a, b)).or_insert(0) += 1;
                    *target_prev_counts.entry(b).or_insert(0) += 1;
                }

                let n = sample_len.saturating_sub(1);
                if n == 0 {
                    continue;
                }
                let n_f = usize_to_f64(n);

                let mut te = 0.0;
                for ((a, b, c), triple_count) in triple_counts {
                    let source_target_prev_count =
                        source_target_prev_counts.get(&(b, c)).copied().unwrap_or(0);
                    let target_pair_count = target_pair_counts.get(&(a, b)).copied().unwrap_or(0);
                    let target_prev_count = target_prev_counts.get(&b).copied().unwrap_or(0);
                    if source_target_prev_count == 0
                        || target_pair_count == 0
                        || target_prev_count == 0
                    {
                        continue;
                    }

                    let p_abc = usize_to_f64(triple_count) / n_f;
                    let p_a_given_bc =
                        usize_to_f64(triple_count) / usize_to_f64(source_target_prev_count);
                    let p_a_given_b =
                        usize_to_f64(target_pair_count) / usize_to_f64(target_prev_count);
                    if p_a_given_bc > 0.0 && p_a_given_b > 0.0 {
                        te += p_abc * log2(p_a_given_bc / p_a_given_b);
                    }
                }

                pair_sum += te;
                pair_count += 1;
            }
        }

        if pair_count == 0 {
            0.0
        } else {
            pair_sum / usize_to_f64(pair_count)
        }
    }

    fn total_correlation(quantized: &QuantizedTrace) -> f64 {
        let frame_count = quantized.per_frame.len();
        let cell_count = quantized.per_cell.len();
        if frame_count == 0 || cell_count == 0 {
            return 0.0;
        }

        let mut marginal_entropy_sum = 0.0;
        for series in &quantized.per_cell {
            marginal_entropy_sum += entropy_u8(series);
        }

        let mut joint_counts: HashMap<Vec<u8>, usize> = HashMap::new();
        for frame_bins in &quantized.per_frame {
            *joint_counts.entry(frame_bins.clone()).or_insert(0) += 1;
        }

        let frame_count_f = usize_to_f64(frame_count);
        let mut joint_entropy = 0.0;
        for count in joint_counts.values() {
            let p = usize_to_f64(*count) / frame_count_f;
            if p > 0.0 {
                joint_entropy = p.mul_add(-log2(p), joint_entropy);
            }
        }

        marginal_entropy_sum - joint_entropy
    }

    fn delta(&self, trace: &StateTrace) -> f64 {
        let checkpoints = checkpoint_schedule(trace.len(), self.config.history_depth);
        if checkpoints.len() < 2 {
            return 0.0;
        }

        let mut xs = Vec::new();
        let mut ys = Vec::new();

        for checkpoint in checkpoints {
            let prefix = trace_prefix(trace, checkpoint);
            let quantized = QuantizedTrace::from_trace(&prefix, self.config.bin_count);
            let ais = self.ais_binning(&quantized);
            let cell_count = quantized.per_cell.len();
            if cell_count == 0 {
                continue;
            }
            let ais_per_cell = ais / usize_to_f64(cell_count);

            xs.push(loge(usize_to_f64(checkpoint)));
            ys.push(loge(ais_per_cell.max(1e-12)));
        }

        linear_regression_slope(&xs, &ys)
    }
}

#[derive(Debug, Clone)]
struct QuantizedTrace {
    per_cell: Vec<Vec<u8>>,
    per_frame: Vec<Vec<u8>>,
}

impl QuantizedTrace {
    fn from_trace(trace: &StateTrace, bin_count: usize) -> Self {
        let frames = trace.frames();
        let cell_count = frames
            .first()
            .map_or(0, |frame| frame.state.activities.len());

        let mut mins = vec![f64::INFINITY; cell_count];
        let mut maxs = vec![f64::NEG_INFINITY; cell_count];

        for frame in frames {
            for (idx, value) in frame.state.activities.iter().copied().enumerate() {
                if let Some(min_ref) = mins.get_mut(idx) {
                    *min_ref = min_ref.min(value);
                }
                if let Some(max_ref) = maxs.get_mut(idx) {
                    *max_ref = max_ref.max(value);
                }
            }
        }

        let mut per_cell = vec![Vec::<u8>::new(); cell_count];
        let mut per_frame = Vec::with_capacity(frames.len());

        for frame in frames {
            let mut frame_bins = Vec::with_capacity(cell_count);
            for (idx, value) in frame.state.activities.iter().copied().enumerate() {
                let min_opt = mins.get(idx).copied();
                let max_opt = maxs.get(idx).copied();
                let (Some(min_value), Some(max_value)) = (min_opt, max_opt) else {
                    continue;
                };
                let binned = quantize(value, min_value, max_value, bin_count);
                frame_bins.push(binned);
                if let Some(series) = per_cell.get_mut(idx) {
                    series.push(binned);
                }
            }
            per_frame.push(frame_bins);
        }

        Self {
            per_cell,
            per_frame,
        }
    }
}

fn ais_for_series(series: &[u8], history_depth: usize, bin_count: usize) -> f64 {
    if series.len() <= history_depth {
        return 0.0;
    }

    let mut count_joint: HashMap<(u32, u8), usize> = HashMap::new();
    let mut count_hist: HashMap<u32, usize> = HashMap::new();
    let mut count_curr: HashMap<u8, usize> = HashMap::new();

    for t in history_depth..series.len() {
        let current_opt = series.get(t).copied();
        let Some(current) = current_opt else {
            continue;
        };

        let mut history_key = 0_u32;
        let mut valid = true;
        for lag in 1..=history_depth {
            let value_opt = series.get(t - lag).copied();
            let Some(value) = value_opt else {
                valid = false;
                break;
            };
            let radix = u32::try_from(bin_count).unwrap_or(u32::MAX);
            history_key = history_key
                .saturating_mul(radix)
                .saturating_add(u32::from(value));
        }
        if !valid {
            continue;
        }

        *count_joint.entry((history_key, current)).or_insert(0) += 1;
        *count_hist.entry(history_key).or_insert(0) += 1;
        *count_curr.entry(current).or_insert(0) += 1;
    }

    let n = series.len().saturating_sub(history_depth);
    if n == 0 {
        return 0.0;
    }

    let n_f = usize_to_f64(n);
    let mut ais = 0.0;

    for ((history, current), joint_count) in count_joint {
        let hist_count = count_hist.get(&history).copied().unwrap_or(0);
        let curr_count = count_curr.get(&current).copied().unwrap_or(0);
        if hist_count == 0 || curr_count == 0 {
            continue;
        }

        let p_joint = usize_to_f64(joint_count) / n_f;
        let p_hist = usize_to_f64(hist_count) / n_f;
        let p_curr = usize_to_f64(curr_count) / n_f;
        if p_joint > 0.0 && p_hist > 0.0 && p_curr > 0.0 {
            ais = p_joint.mul_add(log2(p_joint / (p_hist * p_curr)), ais);
        }
    }

    ais
}

fn sigma_lambda(trace: &StateTrace) -> f64 {
    let mut values = Vec::new();
    for frame in trace.frames() {
        for value in &frame.state.lambda_i {
            values.push(*value);
        }
    }

    if values.is_empty() {
        return 0.0;
    }

    let n_f = usize_to_f64(values.len());
    let mean = values.iter().sum::<f64>() / n_f;
    let variance = values
        .iter()
        .map(|value| {
            let diff = *value - mean;
            diff * diff
        })
        .sum::<f64>()
        / n_f;

    variance.sqrt()
}

fn attractor_dimension(trace: &StateTrace) -> f64 {
    let points: Vec<Vec<f64>> = trace
        .frames()
        .iter()
        .map(|frame| frame.state.activities.clone())
        .collect();
    let n = points.len();
    if n < 3 {
        return 0.0;
    }

    let mut distances = Vec::new();
    for i in 0..n {
        for j in (i + 1)..n {
            let a_opt = points.get(i);
            let b_opt = points.get(j);
            let (Some(a), Some(b)) = (a_opt, b_opt) else {
                continue;
            };
            distances.push(euclidean(a, b));
        }
    }

    if distances.is_empty() {
        return 0.0;
    }

    distances.sort_by(f64::total_cmp);

    let positive_min = distances.iter().copied().find(|value| *value > 0.0);
    let max_distance = distances.last().copied().unwrap_or(0.0);
    let Some(min_distance) = positive_min else {
        return 0.0;
    };
    if max_distance <= min_distance {
        return 0.0;
    }

    let mut xs = Vec::new();
    let mut ys = Vec::new();
    let radii = log_space(min_distance, max_distance, 6);

    let pair_count = distances.len();
    let pair_count_f = usize_to_f64(pair_count);
    for radius in radii {
        let count = distances
            .iter()
            .filter(|distance| **distance <= radius)
            .count();
        if count == 0 {
            continue;
        }
        let c_r = usize_to_f64(count) / pair_count_f;
        if c_r <= 0.0 {
            continue;
        }
        xs.push(loge(radius));
        ys.push(loge(c_r));
    }

    linear_regression_slope(&xs, &ys).max(0.0)
}

fn ksg_mi(x_history: &[Vec<f64>], y_current: &[f64], k: usize) -> f64 {
    let n = x_history.len().min(y_current.len());
    if n <= k || n < 2 {
        return 0.0;
    }

    let mut digamma_sum = 0.0;

    for i in 0..n {
        let x_anchor_opt = x_history.get(i);
        let y_anchor_opt = y_current.get(i).copied();
        let (Some(x_anchor), Some(y_anchor)) = (x_anchor_opt, y_anchor_opt) else {
            continue;
        };

        let mut dists = Vec::with_capacity(n.saturating_sub(1));
        for j in 0..n {
            if i == j {
                continue;
            }
            let x_candidate_opt = x_history.get(j);
            let y_candidate_opt = y_current.get(j).copied();
            let (Some(x_candidate), Some(y_candidate)) = (x_candidate_opt, y_candidate_opt) else {
                continue;
            };

            let dx = chebyshev(x_anchor, x_candidate);
            let dy = (y_anchor - y_candidate).abs();
            dists.push(dx.max(dy));
        }

        dists.sort_by(f64::total_cmp);
        let epsilon = dists.get(k.saturating_sub(1)).copied().unwrap_or(0.0) + 1e-12;

        let mut nx = 0_usize;
        let mut ny = 0_usize;
        for j in 0..n {
            if i == j {
                continue;
            }
            let x_candidate_opt = x_history.get(j);
            let y_candidate_opt = y_current.get(j).copied();
            let (Some(x_candidate), Some(y_candidate)) = (x_candidate_opt, y_candidate_opt) else {
                continue;
            };

            if chebyshev(x_anchor, x_candidate) < epsilon {
                nx += 1;
            }
            if (y_anchor - y_candidate).abs() < epsilon {
                ny += 1;
            }
        }

        digamma_sum += digamma_usize(nx.saturating_add(1)) + digamma_usize(ny.saturating_add(1));
    }

    let mean_term = digamma_sum / usize_to_f64(n);
    let estimate_nats = digamma_usize(k) + digamma_usize(n) - mean_term;
    (estimate_nats * LOG2_E).max(0.0)
}

fn entropy_u8(values: &[u8]) -> f64 {
    if values.is_empty() {
        return 0.0;
    }

    let mut counts: HashMap<u8, usize> = HashMap::new();
    for value in values {
        *counts.entry(*value).or_insert(0) += 1;
    }

    let n_f = usize_to_f64(values.len());
    let mut entropy = 0.0;
    for count in counts.values() {
        let p = usize_to_f64(*count) / n_f;
        if p > 0.0 {
            entropy = p.mul_add(-log2(p), entropy);
        }
    }

    entropy
}

fn checkpoint_schedule(total_steps: usize, history_depth: usize) -> Vec<usize> {
    let mut checkpoints = Vec::new();
    let mut cursor = history_depth.saturating_add(2);

    while cursor <= total_steps {
        checkpoints.push(cursor);
        cursor = cursor.saturating_mul(2);
    }

    if let Some(last) = checkpoints.last().copied() {
        if last != total_steps && total_steps > history_depth.saturating_add(1) {
            checkpoints.push(total_steps);
        }
    } else if total_steps > history_depth.saturating_add(1) {
        checkpoints.push(total_steps);
    }

    checkpoints
}

fn trace_prefix(trace: &StateTrace, count: usize) -> StateTrace {
    let mut prefix = StateTrace::new();
    for frame in trace.frames().iter().take(count) {
        prefix.push(frame.tick, frame.state.clone());
    }
    prefix
}

fn linear_regression_slope(xs: &[f64], ys: &[f64]) -> f64 {
    let n = xs.len().min(ys.len());
    if n < 2 {
        return 0.0;
    }

    let n_f = usize_to_f64(n);
    let mean_x = xs.iter().take(n).sum::<f64>() / n_f;
    let mean_y = ys.iter().take(n).sum::<f64>() / n_f;

    let mut numerator = 0.0;
    let mut denominator = 0.0;

    for idx in 0..n {
        let x_opt = xs.get(idx).copied();
        let y_opt = ys.get(idx).copied();
        let (Some(x), Some(y)) = (x_opt, y_opt) else {
            continue;
        };
        let dx = x - mean_x;
        numerator = dx.mul_add(y - mean_y, numerator);
        denominator = dx.mul_add(dx, denominator);
    }

    if denominator <= 0.0 {
        0.0
    } else {
        numerator / denominator
    }
}

fn quantize(value: f64, min: f64, max: f64, bin_count: usize) -> u8 {
    if bin_count <= 1 || max <= min {
        return 0;
    }

    let span = max - min;
    let normalized = ((value - min) / span).clamp(0.0, 1.0);
    let bins_f = usize_to_f64(bin_count);
    let scaled = (normalized * bins_f).floor();

    let mut idx = if scaled.is_finite() && scaled >= 0.0 {
        scaled
    } else {
        0.0
    };

    let last_bin = usize_to_f64(bin_count.saturating_sub(1));
    if idx > last_bin {
        idx = last_bin;
    }

    let idx_u32 = f64_to_u32_floor(idx);
    u8::try_from(idx_u32).unwrap_or(u8::MAX)
}

fn log_space(min_value: f64, max_value: f64, count: usize) -> Vec<f64> {
    if count < 2 {
        return vec![min_value.max(1e-12)];
    }

    let min_ln = loge(min_value.max(1e-12));
    let max_ln = loge(max_value.max(min_value + 1e-12));
    let step = (max_ln - min_ln) / usize_to_f64(count.saturating_sub(1));

    let mut values = Vec::with_capacity(count);
    for idx in 0..count {
        let position = min_ln + step * usize_to_f64(idx);
        values.push(position.exp());
    }
    values
}

fn chebyshev(a: &[f64], b: &[f64]) -> f64 {
    a.iter()
        .zip(b.iter())
        .map(|(lhs, rhs)| (lhs - rhs).abs())
        .fold(0.0, f64::max)
}

fn euclidean(a: &[f64], b: &[f64]) -> f64 {
    a.iter()
        .zip(b.iter())
        .map(|(lhs, rhs)| {
            let diff = lhs - rhs;
            diff * diff
        })
        .sum::<f64>()
        .sqrt()
}

fn digamma_usize(value: usize) -> f64 {
    if value <= 1 {
        return -EULER_GAMMA;
    }

    let mut harmonic = 0.0;
    for idx in 1..value {
        harmonic += 1.0 / usize_to_f64(idx);
    }
    harmonic - EULER_GAMMA
}

fn usize_to_f64(value: usize) -> f64 {
    let narrowed = u32::try_from(value).unwrap_or(u32::MAX);
    f64::from(narrowed)
}

fn f64_to_u32_floor(value: f64) -> u32 {
    if !value.is_finite() || value <= 0.0 {
        return 0;
    }

    let floored_text = format!("{:.0}", value.floor());
    floored_text.parse::<u32>().unwrap_or(u32::MAX)
}

fn loge(value: f64) -> f64 {
    value.ln()
}

fn log2(value: f64) -> f64 {
    loge(value) * LOG2_E
}

#[cfg(test)]
mod tests {
    use super::{Observer, ObserverConfig};
    use crate::{State, StateTrace};

    fn build_trace_with_pattern(steps: usize, cells: usize) -> StateTrace {
        let mut trace = StateTrace::new();
        for tick in 0..steps {
            let parity = tick % 2;
            let base = if parity == 0 { 0.1 } else { 0.9 };
            let activities = vec![base; cells];
            let lambda_i = (0..cells)
                .map(|idx| f64::from(u32::try_from(idx).unwrap_or(u32::MAX)).mul_add(0.1, 0.2))
                .collect();
            trace.push(tick, State::new(activities, lambda_i));
        }
        trace
    }

    fn build_continuous_trace(steps: usize) -> StateTrace {
        let mut trace = StateTrace::new();
        let mut x_prev = 0.12;
        for tick in 0..steps {
            let tick_f = f64::from(u32::try_from(tick).unwrap_or(u32::MAX));
            let noise = tick_f.mul_add(0.113, 0.0).sin().mul_add(0.015, 0.0);
            let x = 0.87_f64.mul_add(x_prev, noise + 0.02);
            x_prev = x;
            trace.push(tick, State::new(vec![x], vec![0.5 + noise.abs()]));
        }
        trace
    }

    #[test]
    fn observer_uses_paper_default_estimator_configuration() {
        let observer = Observer::default();
        let config = observer.config();
        assert_eq!(config.history_depth, 4);
        assert_eq!(config.bin_count, 8);
    }

    #[test]
    fn synthetic_periodic_trace_has_high_binning_ais() {
        let observer = Observer::default();
        let trace = build_trace_with_pattern(128, 1);
        let metrics = observer.observe(&trace);
        assert!(metrics.ais_binning > 0.85);
        assert!(metrics.ais_binning < 1.1);
    }

    #[test]
    fn ksg_cross_check_ratio_is_in_expected_band_on_continuous_trace() {
        let observer = Observer::new(ObserverConfig {
            history_depth: 4,
            bin_count: 8,
            ksg_k: 3,
        });
        let trace = build_continuous_trace(512);
        let metrics = observer.observe(&trace);
        assert!(metrics.ais_binning > 0.0);
        assert!(metrics.ais_ksg > 0.0);
        assert!(
            metrics.ais_ksg_ratio >= 0.67,
            "ratio: {}",
            metrics.ais_ksg_ratio
        );
        assert!(
            metrics.ais_ksg_ratio <= 0.87,
            "ratio: {}",
            metrics.ais_ksg_ratio
        );
    }

    #[test]
    fn observer_signature_is_read_only_over_state_trace() {
        let method: fn(&Observer, &StateTrace) -> super::ObserverMetrics = Observer::observe;
        let observer = Observer::default();
        let trace = build_trace_with_pattern(32, 2);
        let metrics = method(&observer, &trace);
        assert!(metrics.tc >= 0.0);
    }
}
