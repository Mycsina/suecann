use crate::evaluator;
use crate::wann::RustWannNetwork;
use numpy::{PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

#[pyclass]
#[derive(Clone)]
pub struct PyWannNetwork {
    pub inner: RustWannNetwork,
}

#[pymethods]
impl PyWannNetwork {
    #[getter]
    fn num_nodes(&self) -> usize {
        self.inner.num_nodes
    }

    #[getter]
    fn node_activations(&self) -> Vec<u8> {
        self.inner.node_activations.clone()
    }

    #[getter]
    fn node_aggregations(&self) -> Vec<u8> {
        self.inner.node_aggregations.clone()
    }

    #[getter]
    fn topological_order(&self) -> Vec<usize> {
        self.inner.topological_order.clone()
    }

    #[getter]
    fn node_ptrs(&self) -> Vec<usize> {
        self.inner.node_ptrs.clone()
    }

    #[getter]
    fn incoming_srcs(&self) -> Vec<usize> {
        self.inner.incoming_srcs.clone()
    }

    #[getter]
    fn incoming_signs(&self) -> Vec<i8> {
        self.inner.incoming_signs.clone()
    }

    #[staticmethod]
    #[allow(clippy::too_many_arguments)]
    pub fn from_genome(
        node_ids: Vec<usize>,
        _node_types: Vec<u8>,
        node_activations_input: Vec<u8>,
        node_aggregations_input: Vec<u8>,
        conn_srcs: Vec<usize>,
        conn_dsts: Vec<usize>,
        conn_signs: Vec<i8>,
        conn_enableds: Vec<bool>,
    ) -> PyResult<Self> {
        let inner = RustWannNetwork::new(
            &node_ids,
            &node_activations_input,
            &node_aggregations_input,
            &conn_srcs,
            &conn_dsts,
            &conn_signs,
            &conn_enableds,
        );
        Ok(PyWannNetwork { inner })
    }

    fn forward(&self, inputs: Vec<f64>, shared_weight: f64) -> PyResult<Vec<f64>> {
        if inputs.len() != crate::constants::INPUT_COUNT {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "Inputs length must be {}",
                crate::constants::INPUT_COUNT
            )));
        }
        let mut scratchpad = vec![0.0f64; self.inner.num_nodes];
        let mut inputs_arr = [0.0f64; crate::constants::INPUT_COUNT];
        inputs_arr.copy_from_slice(&inputs);
        self.inner
            .forward(&inputs_arr, shared_weight, &mut scratchpad);

        if self.inner.num_nodes < crate::constants::FIRST_HIDDEN_ID {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "Network num_nodes is {}, must be at least {} to have outputs",
                self.inner.num_nodes,
                crate::constants::FIRST_HIDDEN_ID
            )));
        }

        let out_start = crate::constants::OUTPUT_START;
        let out_end = out_start + crate::constants::OUTPUT_COUNT;
        Ok(scratchpad[out_start..out_end].to_vec())
    }

    fn forward_weight_sweep(
        &self,
        inputs: Vec<f64>,
        weights: Vec<f64>,
    ) -> PyResult<(Vec<Vec<f64>>, Vec<f64>)> {
        if inputs.len() != crate::constants::INPUT_COUNT {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "Inputs length must be {}",
                crate::constants::INPUT_COUNT
            )));
        }
        if self.inner.num_nodes < crate::constants::FIRST_HIDDEN_ID {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "Network num_nodes is {}, must be at least {} to have outputs",
                self.inner.num_nodes,
                crate::constants::FIRST_HIDDEN_ID
            )));
        }
        let mut scratchpad = vec![0.0f64; self.inner.num_nodes];
        let mut inputs_arr = [0.0f64; crate::constants::INPUT_COUNT];
        inputs_arr.copy_from_slice(&inputs);

        let mut all_outputs = Vec::with_capacity(weights.len());
        let mut mean_output = vec![0.0f64; crate::constants::OUTPUT_COUNT];

        let out_start = crate::constants::OUTPUT_START;
        let out_end = out_start + crate::constants::OUTPUT_COUNT;

        for &w in &weights {
            self.inner.forward(&inputs_arr, w, &mut scratchpad);
            let out = scratchpad[out_start..out_end].to_vec();
            for i in 0..crate::constants::OUTPUT_COUNT {
                mean_output[i] += out[i];
            }
            all_outputs.push(out);
        }

        if !weights.is_empty() {
            let n = weights.len() as f64;
            for i in 0..crate::constants::OUTPUT_COUNT {
                mean_output[i] /= n;
            }
        }

        Ok((all_outputs, mean_output))
    }
}

#[derive(FromPyObject)]
pub struct PyCompatibilityGenome {
    pub conn_innovations: Vec<usize>,
    pub conn_signs: Vec<i8>,
    pub conn_enableds: Vec<bool>,
    pub node_ids: Vec<usize>,
    pub node_types: Vec<u8>,
    pub node_activations: Vec<u8>,
    pub node_aggregations: Vec<u8>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JsonGenome {
    pub next_innovation: usize,
    pub node_ids: Vec<i32>,
    pub node_types: Vec<i32>,
    pub node_acts: Vec<i32>,
    pub node_aggs: Vec<i32>,
    pub conn_innovs: Vec<i32>,
    pub conn_srcs: Vec<i32>,
    pub conn_dsts: Vec<i32>,
    pub conn_signs: Vec<i32>,
    pub conn_enabled: Vec<i32>,
}

fn compatibility_distance_internal(
    a: &PyCompatibilityGenome,
    b: &PyCompatibilityGenome,
    c1: f64,
    c2: f64,
    c3: f64,
) -> f64 {
    let mut conns_a: Vec<(usize, i8, bool)> = a
        .conn_innovations
        .iter()
        .cloned()
        .zip(a.conn_signs.iter().cloned())
        .zip(a.conn_enableds.iter().cloned())
        .map(|((i, s), e)| (i, s, e))
        .collect();
    conns_a.sort_by_key(|c| c.0);

    let mut conns_b: Vec<(usize, i8, bool)> = b
        .conn_innovations
        .iter()
        .cloned()
        .zip(b.conn_signs.iter().cloned())
        .zip(b.conn_enableds.iter().cloned())
        .map(|((i, s), e)| (i, s, e))
        .collect();
    conns_b.sort_by_key(|c| c.0);

    if conns_a.is_empty() && conns_b.is_empty() {
        return 0.0;
    }

    let max_innov_a = conns_a.last().map(|c| c.0).unwrap_or(0);
    let max_innov_b = conns_b.last().map(|c| c.0).unwrap_or(0);

    let mut i_a = 0;
    let mut i_b = 0;
    let mut shared = 0;
    let mut disjoint = 0;
    let mut excess = 0;
    let mut sign_diff_total = 0.0;

    while i_a < conns_a.len() && i_b < conns_b.len() {
        let innov_a = conns_a[i_a].0;
        let innov_b = conns_b[i_b].0;
        if innov_a == innov_b {
            shared += 1;
            if conns_a[i_a].1 != conns_b[i_b].1 {
                sign_diff_total += 1.0;
            }
            if conns_a[i_a].2 != conns_b[i_b].2 {
                sign_diff_total += 0.5;
            }
            i_a += 1;
            i_b += 1;
        } else if innov_a < innov_b {
            if innov_a <= max_innov_b {
                disjoint += 1;
            } else {
                excess += 1;
            }
            i_a += 1;
        } else {
            if innov_b <= max_innov_a {
                disjoint += 1;
            } else {
                excess += 1;
            }
            i_b += 1;
        }
    }

    while i_a < conns_a.len() {
        if conns_a[i_a].0 <= max_innov_b {
            disjoint += 1;
        } else {
            excess += 1;
        }
        i_a += 1;
    }

    while i_b < conns_b.len() {
        if conns_b[i_b].0 <= max_innov_a {
            disjoint += 1;
        } else {
            excess += 1;
        }
        i_b += 1;
    }

    let n_conns = conns_a.len().max(conns_b.len()).max(1) as f64;
    let avg_sign_diff = if shared > 0 {
        sign_diff_total / (shared as f64)
    } else {
        0.0
    };

    // 2. Symmetric hidden node comparison using union of IDs
    use std::collections::HashMap;
    let nodes_a_map: HashMap<usize, (u8, u8)> = a
        .node_ids
        .iter()
        .cloned()
        .zip(a.node_activations.iter().cloned())
        .zip(a.node_aggregations.iter().cloned())
        .zip(a.node_types.iter().cloned())
        .filter(|(((_, _), _), t)| *t == 2) // HIDDEN NodeType
        .map(|(((id, act), agg), _)| (id, (act, agg)))
        .collect();

    let nodes_b_map: HashMap<usize, (u8, u8)> = b
        .node_ids
        .iter()
        .cloned()
        .zip(b.node_activations.iter().cloned())
        .zip(b.node_aggregations.iter().cloned())
        .zip(b.node_types.iter().cloned())
        .filter(|(((_, _), _), t)| *t == 2) // HIDDEN NodeType
        .map(|(((id, act), agg), _)| (id, (act, agg)))
        .collect();

    let all_ids: std::collections::HashSet<usize> = nodes_a_map
        .keys()
        .copied()
        .chain(nodes_b_map.keys().copied())
        .collect();

    let mut node_diff = 0.0;
    for nid in all_ids {
        match (nodes_a_map.get(&nid), nodes_b_map.get(&nid)) {
            (Some(&(act_a, agg_a)), Some(&(act_b, agg_b))) => {
                if act_a != act_b {
                    node_diff += 0.5;
                }
                if agg_a != agg_b {
                    node_diff += 0.5;
                }
            }
            _ => {
                node_diff += 1.0;
            }
        }
    }

    c1 * (excess as f64) / n_conns
        + c2 * (disjoint as f64) / n_conns
        + c3 * avg_sign_diff
        + c3 * node_diff
}

#[pyfunction]
pub fn batch_compatibility_distances(
    rep: PyCompatibilityGenome,
    others: Vec<PyCompatibilityGenome>,
    c1: f64,
    c2: f64,
    c3: f64,
) -> PyResult<Vec<f64>> {
    let distances = others
        .iter()
        .map(|other| compatibility_distance_internal(&rep, other, c1, c2, c3))
        .collect();
    Ok(distances)
}

#[pyfunction]
#[pyo3(signature = (
    genomes,
    deals,
    sweep_weights,
    baseline_bot_type,
    partner_bot_type,
    opp1_bot_type,
    opp2_bot_type,
    hof_genomes,
    base_seed
))]
#[allow(clippy::too_many_arguments)]
pub fn evaluate_wann_population(
    py: Python,
    genomes: Vec<PyWannNetwork>,
    deals: Vec<super::matchup::PyDeal>,
    sweep_weights: Vec<f64>,
    baseline_bot_type: i32,
    partner_bot_type: i32,
    opp1_bot_type: i32,
    opp2_bot_type: i32,
    hof_genomes: Vec<PyWannNetwork>,
    base_seed: u64,
) -> PyResult<Vec<f64>> {
    // 1. Extract Python structures to pure Rust types while holding the GIL.
    let rust_genomes: Vec<RustWannNetwork> = genomes.iter().map(|g| g.inner.clone()).collect();
    let rust_hof: Vec<RustWannNetwork> = hof_genomes.iter().map(|g| g.inner.clone()).collect();
    let rust_deals: Vec<evaluator::EvaluatorDeal> = deals.iter().map(|d| d.to_rust()).collect();

    // 2. Allow other threads to execute Python code while we simulate inside Rayon.
    let max_nodes = rust_genomes
        .iter()
        .map(|g| g.num_nodes)
        .chain(rust_hof.iter().map(|g| g.num_nodes))
        .max()
        .unwrap_or(crate::constants::FIRST_HIDDEN_ID);

    let results = py.allow_threads(|| {
        rust_genomes
            .into_par_iter()
            .enumerate()
            .map(|(i, candidate)| {
                let mut scratchpad = vec![0.0f64; max_nodes];
                let (delta, _behavior) = evaluator::evaluate_genome_delta(
                    &candidate,
                    baseline_bot_type,
                    partner_bot_type,
                    opp1_bot_type,
                    opp2_bot_type,
                    &rust_hof,
                    &sweep_weights,
                    &rust_deals,
                    base_seed + (i as u64),
                    &mut scratchpad,
                );
                delta
            })
            .collect::<Vec<f64>>()
    });

    Ok(results)
}

#[pyfunction]
pub fn evaluate_wann_accuracy(
    py: Python,
    genomes: Vec<PyWannNetwork>,
    states: PyReadonlyArray2<f64>,
    intents: PyReadonlyArray1<u8>,
    legal_masks: PyReadonlyArray1<u8>,
    sweep_weights: Vec<f64>,
) -> PyResult<Vec<f64>> {
    let rust_genomes: Vec<RustWannNetwork> = genomes.iter().map(|g| g.inner.clone()).collect();
    let states_array = states.as_array();
    let intents_array = intents.as_array();
    let legal_masks_array = legal_masks.as_array();

    let num_states = intents_array.len();

    let max_nodes = rust_genomes
        .iter()
        .map(|g| g.num_nodes)
        .max()
        .unwrap_or(crate::constants::FIRST_HIDDEN_ID);

    let accuracies = py.allow_threads(|| {
        rust_genomes
            .into_par_iter()
            .map(|candidate| {
                let mut scratchpad = vec![0.0f64; max_nodes];
                let mut correct = 0;

                for idx in 0..num_states {
                    let mut inputs = [0.0f64; crate::constants::INPUT_COUNT];
                    for i in 0..crate::constants::INPUT_COUNT {
                        inputs[i] = states_array[[idx, i]];
                    }
                    let target_intent = intents_array[idx] as usize;
                    let mask = legal_masks_array[idx];

                    let mut total_outputs = [0.0f64; crate::constants::OUTPUT_COUNT];
                    for &w in &sweep_weights {
                        candidate.forward(&inputs, w, &mut scratchpad);
                        for i in 0..crate::constants::OUTPUT_COUNT {
                            total_outputs[i] += scratchpad[crate::constants::OUTPUT_START + i];
                        }
                    }

                    // Apply legal mask and find argmax with baseline offset for EQUITY_BUILDER (index 3)
                    let mut best_intent = 0;
                    let mut max_val = f64::NEG_INFINITY;
                    for i in 0..crate::constants::OUTPUT_COUNT {
                        let is_legal = (mask & (1 << i)) != 0;
                        let val = if i == 3 {
                            total_outputs[i] - 0.25 * (sweep_weights.len() as f64)
                        } else {
                            total_outputs[i]
                        };
                        if is_legal && val > max_val {
                            max_val = val;
                            best_intent = i;
                        }
                    }

                    if best_intent == target_intent {
                        correct += 1;
                    }
                }

                correct as f64 / num_states as f64
            })
            .collect::<Vec<f64>>()
    });

    Ok(accuracies)
}

#[pyfunction]
pub fn pareto_rank_rust(fitnesses: Vec<f64>, complexities: Vec<f64>) -> PyResult<Vec<f64>> {
    let n = fitnesses.len();
    if n == 0 {
        return Ok(Vec::new());
    }

    // Convert complexity to simplicity (lower complexity = higher simplicity)
    let max_complexity = complexities.iter().fold(0.0f64, |a, &b| a.max(b));
    let simplicities: Vec<f64> = complexities.iter().map(|&c| max_complexity - c).collect();

    // Compute domination count and dominated-by sets
    let mut domination_count = vec![0; n];
    let mut dominated_by = vec![Vec::new(); n];

    for i in 0..n {
        let f_i = fitnesses[i];
        let s_i = simplicities[i];
        for j in (i + 1)..n {
            let f_j = fitnesses[j];
            let s_j = simplicities[j];

            // i dominates j if better on both objectives
            let i_dom_j = f_i >= f_j && s_i >= s_j && (f_i > f_j || s_i > s_j);
            let j_dom_i = f_j >= f_i && s_j >= s_i && (f_j > f_i || s_j > s_i);

            if i_dom_j {
                dominated_by[i].push(j);
                domination_count[j] += 1;
            } else if j_dom_i {
                dominated_by[j].push(i);
                domination_count[i] += 1;
            }
        }
    }

    // Assign Pareto front levels (lower level = better)
    let mut levels = vec![0; n];
    let mut current_front = Vec::new();
    for i in 0..n {
        if domination_count[i] == 0 {
            current_front.push(i);
        }
    }
    let mut level = 0;

    while !current_front.is_empty() {
        for &i in &current_front {
            levels[i] = level;
        }
        let mut next_front = Vec::new();
        for &i in &current_front {
            for &j in &dominated_by[i] {
                domination_count[j] -= 1;
                if domination_count[j] == 0 {
                    next_front.push(j);
                }
            }
        }
        current_front = next_front;
        level += 1;
    }

    // Min-Max normalize fitnesses to [0, 1] for fair tie-breaking
    let mut min_fit = fitnesses[0];
    let mut max_fit = fitnesses[0];
    for &f in &fitnesses {
        if f < min_fit {
            min_fit = f;
        }
        if f > max_fit {
            max_fit = f;
        }
    }
    let fit_range = max_fit - min_fit;
    let perf_scores: Vec<f64> = fitnesses
        .iter()
        .map(|&f| {
            if fit_range > 0.0 {
                (f - min_fit) / fit_range
            } else {
                1.0
            }
        })
        .collect();

    // Convert levels to scores (lower level = higher score)
    let max_level = levels.iter().max().copied().unwrap_or(0);
    let mut scores = Vec::with_capacity(n);
    for i in 0..n {
        let base_score = (max_level - levels[i]) as f64;
        scores.push(base_score + 0.5 * perf_scores[i]);
    }

    // Normalize final scores to [0, 1]
    let mut min_score = scores[0];
    let mut max_score = scores[0];
    for &s in &scores {
        if s < min_score {
            min_score = s;
        }
        if s > max_score {
            max_score = s;
        }
    }
    let score_range = max_score - min_score;
    let final_scores: Vec<f64> = scores
        .iter()
        .map(|&s| {
            if score_range > 0.0 {
                (s - min_score) / score_range
            } else {
                1.0
            }
        })
        .collect();

    Ok(final_scores)
}

#[pyfunction]
pub fn load_genome(_py: Python, filepath: &str) -> PyResult<PyWannNetwork> {
    use npyz::NpyFile;
    use std::fs::File;
    use std::io::BufReader;
    use zip::ZipArchive;

    if filepath.ends_with(".json") {
        let file = File::open(filepath)
            .map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))?;
        let reader = BufReader::new(file);
        let json_genome: JsonGenome = serde_json::from_reader(reader)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;

        let node_ids: Vec<usize> = json_genome.node_ids.iter().map(|&x| x as usize).collect();
        let node_activations: Vec<u8> = json_genome.node_acts.iter().map(|&x| x as u8).collect();
        let node_aggregations: Vec<u8> = json_genome.node_aggs.iter().map(|&x| x as u8).collect();
        let conn_srcs: Vec<usize> = json_genome.conn_srcs.iter().map(|&x| x as usize).collect();
        let conn_dsts: Vec<usize> = json_genome.conn_dsts.iter().map(|&x| x as usize).collect();
        let conn_signs: Vec<i8> = json_genome.conn_signs.iter().map(|&x| x as i8).collect();
        let conn_enableds: Vec<bool> = json_genome.conn_enabled.iter().map(|&x| x != 0).collect();

        PyWannNetwork::from_genome(
            node_ids,
            vec![],
            node_activations,
            node_aggregations,
            conn_srcs,
            conn_dsts,
            conn_signs,
            conn_enableds,
        )
    } else {
        let file = File::open(filepath)
            .map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))?;
        let reader = BufReader::new(file);
        let mut archive = ZipArchive::new(reader)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;

        let read_npy_i32 = |archive: &mut ZipArchive<BufReader<File>>,
                            name: &str|
         -> Result<Vec<i32>, Box<dyn std::error::Error>> {
            let mut f = archive.by_name(name)?;
            let npy = NpyFile::new(&mut f)?;
            let data: Vec<i32> = npy.into_vec()?;
            Ok(data)
        };

        let node_ids_i32 = read_npy_i32(&mut archive, "node_ids.npy")
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
        let node_acts_i32 = read_npy_i32(&mut archive, "node_acts.npy")
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
        let node_aggs_i32 = read_npy_i32(&mut archive, "node_aggs.npy")
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;

        let conn_srcs_i32 = read_npy_i32(&mut archive, "conn_srcs.npy")
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
        let conn_dsts_i32 = read_npy_i32(&mut archive, "conn_dsts.npy")
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
        let conn_signs_i32 = read_npy_i32(&mut archive, "conn_signs.npy")
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
        let conn_enabled_i32 = read_npy_i32(&mut archive, "conn_enabled.npy")
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;

        let node_ids: Vec<usize> = node_ids_i32.iter().map(|&x| x as usize).collect();
        let node_activations: Vec<u8> = node_acts_i32.iter().map(|&x| x as u8).collect();
        let node_aggregations: Vec<u8> = node_aggs_i32.iter().map(|&x| x as u8).collect();

        let conn_srcs: Vec<usize> = conn_srcs_i32.iter().map(|&x| x as usize).collect();
        let conn_dsts: Vec<usize> = conn_dsts_i32.iter().map(|&x| x as usize).collect();
        let conn_signs: Vec<i8> = conn_signs_i32.iter().map(|&x| x as i8).collect();
        let conn_enableds: Vec<bool> = conn_enabled_i32.iter().map(|&x| x != 0).collect();

        PyWannNetwork::from_genome(
            node_ids,
            vec![],
            node_activations,
            node_aggregations,
            conn_srcs,
            conn_dsts,
            conn_signs,
            conn_enableds,
        )
    }
}
