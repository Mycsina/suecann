/// WANN network implementation in Rust.
/// Uses Compressed Sparse Row (CSR) format for connection graph lookups
/// to ensure cache-locality, and runs inference on a pre-allocated scratchpad.

#[derive(Clone, Debug)]
pub struct RustWannNetwork {
    pub num_nodes: usize,
    pub node_activations: Vec<u8>,
    pub node_aggregations: Vec<u8>,
    pub topological_order: Vec<usize>,

    // CSR Format
    pub node_ptrs: Vec<usize>,     // Length: num_nodes + 1
    pub incoming_srcs: Vec<usize>, // Contiguous source node IDs
    pub incoming_signs: Vec<i8>,   // Contiguous connection signs (+1 / -1)
}

impl RustWannNetwork {
    /// Build a WANN network from raw node/connection vectors.
    /// Computes topological order and builds CSR sparse row pointers.
    pub fn new(
        node_ids: &[usize],
        node_activations_input: &[u8],
        node_aggregations_input: &[u8],
        conn_srcs: &[usize],
        conn_dsts: &[usize],
        conn_signs: &[i8],
        conn_enableds: &[bool],
    ) -> Self {
        use std::collections::{HashMap, HashSet};

        let num_nodes = node_ids.iter().max().copied().unwrap_or(0) + 1;

        // Filter enabled connections
        let enabled_conns: Vec<(usize, usize, i8)> = conn_srcs
            .iter()
            .cloned()
            .zip(conn_dsts.iter().cloned())
            .zip(conn_signs.iter().cloned())
            .zip(conn_enableds.iter().cloned())
            .filter(|(((_, _), _), enabled)| *enabled)
            .map(|(((src, dst), sign), _)| (src, dst, sign))
            .collect();

        // Priority for topological sort: inputs(0..21), bias(21), hidden(27+), outputs(22..26)
        let mut hidden_ids: Vec<usize> =
            node_ids.iter().cloned().filter(|&nid| nid >= 27).collect();
        hidden_ids.sort_unstable();

        let get_priority = |nid: usize| -> usize {
            if nid < 21 {
                nid
            } else if nid == 21 {
                21
            } else if nid >= 27 {
                match hidden_ids.binary_search(&nid) {
                    Ok(idx) => 22 + idx,
                    Err(_) => 999999,
                }
            } else if nid >= 22 && nid <= 26 {
                22 + hidden_ids.len() + (nid - 22)
            } else {
                999999
            }
        };

        let node_set: HashSet<usize> = node_ids.iter().cloned().collect();
        let mut adj: HashMap<usize, Vec<usize>> = HashMap::new();
        let mut in_degree: HashMap<usize, usize> = HashMap::new();

        for &nid in node_ids {
            adj.insert(nid, Vec::new());
            in_degree.insert(nid, 0);
        }

        for &(src, dst, _) in &enabled_conns {
            if node_set.contains(&src) && node_set.contains(&dst) {
                adj.get_mut(&src).unwrap().push(dst);
                *in_degree.get_mut(&dst).unwrap() += 1;
            }
        }

        // Kahn's algorithm with priority tiebreaker
        let mut queue: Vec<usize> = node_ids
            .iter()
            .cloned()
            .filter(|nid| in_degree[nid] == 0)
            .collect();
        queue.sort_by_key(|&n| get_priority(n));

        let mut topological_order: Vec<usize> = Vec::with_capacity(node_ids.len());
        let mut visited: HashSet<usize> = HashSet::new();

        while !queue.is_empty() {
            let nid = queue.remove(0);
            if visited.contains(&nid) {
                continue;
            }
            visited.insert(nid);
            topological_order.push(nid);
            if let Some(neighbors) = adj.get(&nid) {
                for &neighbor in neighbors {
                    let deg = in_degree.get_mut(&neighbor).unwrap();
                    *deg -= 1;
                    if *deg == 0 {
                        queue.push(neighbor);
                        queue.sort_by_key(|&n| get_priority(n));
                    }
                }
            }
        }

        let mut remaining: Vec<usize> = node_ids
            .iter()
            .cloned()
            .filter(|nid| !visited.contains(nid))
            .collect();
        remaining.sort_by_key(|&n| get_priority(n));
        topological_order.extend(remaining);

        // Build activation/aggregation arrays
        let mut node_activations = vec![0u8; num_nodes];
        let mut node_aggregations = vec![0u8; num_nodes];
        for i in 0..node_ids.len() {
            let nid = node_ids[i];
            if nid < num_nodes {
                node_activations[nid] = node_activations_input[i];
                node_aggregations[nid] = node_aggregations_input[i];
            }
        }

        // Build CSR format
        let mut incoming: Vec<Vec<(usize, i8)>> = vec![Vec::new(); num_nodes];
        for &(src, dst, sign) in &enabled_conns {
            if dst < num_nodes {
                incoming[dst].push((src, sign));
            }
        }

        let mut node_ptrs = vec![0usize; num_nodes + 1];
        let mut incoming_srcs = Vec::new();
        let mut incoming_signs = Vec::new();

        for nid in 0..num_nodes {
            for &(src, sign) in &incoming[nid] {
                incoming_srcs.push(src);
                incoming_signs.push(sign);
            }
            node_ptrs[nid + 1] = incoming_srcs.len();
        }

        Self {
            num_nodes,
            node_activations,
            node_aggregations,
            topological_order,
            node_ptrs,
            incoming_srcs,
            incoming_signs,
        }
    }

    /// Zero-allocation forward pass.
    /// Evaluates the network using the provided input belief state and shared weight.
    /// The scratchpad must be pre-allocated to self.num_nodes.
    /// The output intents will rest in scratchpad[22..27].
    pub fn forward(&self, inputs: &[f64; 21], weight: f64, scratchpad: &mut [f64]) {
        // 1. Copy inputs into scratchpad[0..21] and set bias scratchpad[21] = 1.0
        for i in 0..21 {
            scratchpad[i] = inputs[i].clamp(0.0, 1.0);
        }
        scratchpad[21] = 1.0;

        // Reset the rest of the nodes (outputs and hiddens)
        for i in 22..self.num_nodes {
            scratchpad[i] = 0.0;
        }

        // 2. Evaluate all nodes in topological order
        for &nid in &self.topological_order {
            if nid < 22 {
                continue; // input or bias, already set
            }

            let start = self.node_ptrs[nid];
            let end = self.node_ptrs[nid + 1];
            if start == end {
                scratchpad[nid] = 0.0;
                continue;
            }

            let agg_fn = self.node_aggregations[nid];
            let mut sum_val = 0.0;
            let mut min_val = f64::INFINITY;
            let mut max_val = f64::NEG_INFINITY;

            for idx in start..end {
                let src = self.incoming_srcs[idx];
                let sign = self.incoming_signs[idx];
                let src_val = scratchpad[src];

                // Apply sign inversion
                let val = if sign == -1 {
                    1.0 - src_val.clamp(0.0, 1.0)
                } else {
                    src_val
                };

                let signal = val * weight;
                match agg_fn {
                    0 => {
                        sum_val += signal;
                    } // SUM
                    1 => {
                        if signal < min_val {
                            min_val = signal;
                        }
                    } // MIN
                    2 => {
                        if signal > max_val {
                            max_val = signal;
                        }
                    } // MAX
                    _ => {}
                }
            }

            let agg_val = match agg_fn {
                0 => sum_val,
                1 => min_val,
                2 => max_val,
                _ => 0.0,
            };

            // Activate and clamp
            let act_fn = self.node_activations[nid];
            let activated = match act_fn {
                0 => agg_val,                       // IDENTITY
                1 => 1.0 - agg_val.clamp(0.0, 1.0), // NOT
                2 => {
                    // THRESHOLD
                    if weight >= 0.0 {
                        if agg_val > 0.5 * weight {
                            1.0
                        } else {
                            0.0
                        }
                    } else {
                        if agg_val < 0.5 * weight {
                            1.0
                        } else {
                            0.0
                        }
                    }
                }
                _ => 0.0,
            };

            scratchpad[nid] = activated.clamp(0.0, 1.0);
        }
    }
}
