use serde::{Deserialize, Serialize};
use std::collections::{BinaryHeap, HashSet};

pub use sueca_solver::constants::{
    BIAS_ID, FIRST_HIDDEN_ID, INPUT_COUNT, INPUT_START, OUTPUT_COUNT, OUTPUT_START,
};

#[allow(clippy::upper_case_acronyms)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum NodeType {
    INPUT = 0,
    BIAS = 1,
    HIDDEN = 2,
    OUTPUT = 3,
}

#[allow(clippy::upper_case_acronyms)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ActivationFn {
    IDENTITY = 0,
    NOT = 1,
    THRESHOLD = 2,
}

#[allow(clippy::upper_case_acronyms)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AggregationFn {
    SUM = 0,
    MIN = 1,
    MAX = 2,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NodeGene {
    pub id: usize,
    pub node_type: NodeType,
    pub activation_fn: ActivationFn,
    pub aggregation_fn: AggregationFn,
}

impl NodeGene {
    pub fn make(
        id: usize,
        node_type: NodeType,
        activation_fn: ActivationFn,
        aggregation_fn: AggregationFn,
    ) -> Self {
        Self {
            id,
            node_type,
            activation_fn,
            aggregation_fn,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConnGene {
    pub innovation: usize,
    pub src: usize,
    pub dst: usize,
    pub sign: i8,
    pub enabled: bool,
}

impl ConnGene {
    pub fn make(innovation: usize, src: usize, dst: usize, sign: i8, enabled: bool) -> Self {
        assert!(sign == -1 || sign == 1, "Sign must be +1 or -1");
        Self {
            innovation,
            src,
            dst,
            sign,
            enabled,
        }
    }
}

/// Genome with sorted-Vec storage. Nodes sorted by id, connections by innovation.
/// Binary search replaces HashMap lookups; crossover uses two-pointer merge.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Genome {
    pub node_genes: Vec<NodeGene>, // sorted by id
    pub conn_genes: Vec<ConnGene>, // sorted by innovation
    pub next_innovation: usize,
}

// --- Private helpers ---

fn node_idx(nodes: &[NodeGene], id: usize) -> Result<usize, usize> {
    nodes.binary_search_by_key(&id, |n| n.id)
}

fn conn_idx(conns: &[ConnGene], innovation: usize) -> Result<usize, usize> {
    conns.binary_search_by_key(&innovation, |c| c.innovation)
}

impl Genome {
    pub fn new(
        node_genes: Option<Vec<NodeGene>>,
        conn_genes: Option<Vec<ConnGene>>,
        next_innovation: usize,
    ) -> Self {
        let nodes = if let Some(ng_list) = node_genes {
            let mut v = ng_list;
            v.sort_by_key(|n| n.id);
            v.dedup_by_key(|n| n.id);
            v
        } else {
            let mut v = Vec::with_capacity(INPUT_COUNT + 1 + OUTPUT_COUNT);
            for i in 0..INPUT_COUNT {
                v.push(NodeGene::make(
                    INPUT_START + i,
                    NodeType::INPUT,
                    ActivationFn::IDENTITY,
                    AggregationFn::SUM,
                ));
            }
            v.push(NodeGene::make(
                BIAS_ID,
                NodeType::BIAS,
                ActivationFn::IDENTITY,
                AggregationFn::SUM,
            ));
            for i in 0..OUTPUT_COUNT {
                v.push(NodeGene::make(
                    OUTPUT_START + i,
                    NodeType::OUTPUT,
                    ActivationFn::IDENTITY,
                    AggregationFn::SUM,
                ));
            }
            v
        };

        let conns = if let Some(cg_list) = conn_genes {
            let mut v = cg_list;
            v.sort_by_key(|c| c.innovation);
            v.dedup_by_key(|c| c.innovation);
            v
        } else {
            Vec::new()
        };

        let mut next_inno = next_innovation;
        if let Some(last) = conns.last() {
            if next_inno <= last.innovation {
                next_inno = last.innovation + 1;
            }
        }

        Self {
            node_genes: nodes,
            conn_genes: conns,
            next_innovation: next_inno,
        }
    }

    pub fn initial() -> Self {
        Self::new(None, None, 0)
    }

    // --- Node accessors ---

    pub fn get_node(&self, id: usize) -> Option<&NodeGene> {
        node_idx(&self.node_genes, id)
            .ok()
            .map(|i| &self.node_genes[i])
    }

    pub fn get_node_mut(&mut self, id: usize) -> Option<&mut NodeGene> {
        node_idx(&self.node_genes, id)
            .ok()
            .map(|i| &mut self.node_genes[i])
    }

    pub fn has_node(&self, id: usize) -> bool {
        node_idx(&self.node_genes, id).is_ok()
    }

    pub fn node_ids(&self) -> HashSet<usize> {
        self.node_genes.iter().map(|n| n.id).collect()
    }

    pub fn hidden_ids(&self) -> Vec<usize> {
        self.node_genes
            .iter()
            .filter(|n| n.node_type == NodeType::HIDDEN)
            .map(|n| n.id)
            .collect()
    }

    pub fn add_node(&mut self, node: NodeGene) {
        match node_idx(&self.node_genes, node.id) {
            Ok(i) => self.node_genes[i] = node,
            Err(i) => self.node_genes.insert(i, node),
        }
    }

    // --- Connection accessors ---

    pub fn get_conn_by_inno(&self, innovation: usize) -> Option<&ConnGene> {
        conn_idx(&self.conn_genes, innovation)
            .ok()
            .map(|i| &self.conn_genes[i])
    }

    pub fn get_conn_mut(&mut self, innovation: usize) -> Option<&mut ConnGene> {
        conn_idx(&self.conn_genes, innovation)
            .ok()
            .map(|i| &mut self.conn_genes[i])
    }

    pub fn add_connection(&mut self, conn: ConnGene) {
        let inno = conn.innovation;
        match conn_idx(&self.conn_genes, inno) {
            Ok(i) => self.conn_genes[i] = conn,
            Err(i) => self.conn_genes.insert(i, conn),
        }
        if inno >= self.next_innovation {
            self.next_innovation = inno + 1;
        }
    }

    #[allow(dead_code)]
    pub fn has_connection(&self, src: usize, dst: usize) -> bool {
        self.conn_genes.iter().any(|c| c.src == src && c.dst == dst)
    }

    #[allow(dead_code)]
    pub fn get_connection(&self, src: usize, dst: usize) -> Option<&ConnGene> {
        self.conn_genes
            .iter()
            .find(|c| c.src == src && c.dst == dst)
    }

    #[allow(dead_code)]
    pub fn enabled_connections(&self) -> Vec<&ConnGene> {
        self.conn_genes.iter().filter(|c| c.enabled).collect()
    }

    pub fn num_enabled(&self) -> usize {
        self.conn_genes.iter().filter(|c| c.enabled).count()
    }

    #[allow(dead_code)]
    pub fn num_nodes(&self) -> usize {
        self.node_genes.len()
    }

    pub fn copy(&self) -> Self {
        Self {
            node_genes: self.node_genes.clone(),
            conn_genes: self.conn_genes.clone(),
            next_innovation: self.next_innovation,
        }
    }

    /// Two-pointer crossover. O(n + m) time, zero heap allocations beyond the result.
    pub fn crossover_with(
        &self,
        other: &Genome,
        self_is_fitter: bool,
        rng: &mut impl rand::Rng,
    ) -> Genome {
        let fitter = if self_is_fitter { self } else { other };
        let lesser = if self_is_fitter { other } else { self };

        // Node genes: merge two sorted arrays
        let mut child_nodes = Vec::with_capacity(self.node_genes.len().max(other.node_genes.len()));
        let (mut i, mut j) = (0, 0);
        while i < self.node_genes.len() && j < other.node_genes.len() {
            match self.node_genes[i].id.cmp(&other.node_genes[j].id) {
                std::cmp::Ordering::Equal => {
                    child_nodes.push(if rng.gen_bool(0.5) {
                        self.node_genes[i].clone()
                    } else {
                        other.node_genes[j].clone()
                    });
                    i += 1;
                    j += 1;
                }
                std::cmp::Ordering::Less => {
                    child_nodes.push(self.node_genes[i].clone());
                    i += 1;
                }
                std::cmp::Ordering::Greater => {
                    child_nodes.push(other.node_genes[j].clone());
                    j += 1;
                }
            }
        }
        child_nodes.extend_from_slice(&self.node_genes[i..]);
        for n in &other.node_genes[j..] {
            if child_nodes.last().map(|x| x.id) != Some(n.id) {
                child_nodes.push(n.clone());
            }
        }

        // Connection genes: two-pointer merge on innovation
        let mut child_conns = Vec::with_capacity(fitter.conn_genes.len());
        let (mut i, mut j) = (0usize, 0usize);
        let a = &fitter.conn_genes;
        let b = &lesser.conn_genes;
        while i < a.len() && j < b.len() {
            match a[i].innovation.cmp(&b[j].innovation) {
                std::cmp::Ordering::Equal => {
                    child_conns.push(if rng.gen_bool(0.5) {
                        a[i].clone()
                    } else {
                        b[j].clone()
                    });
                    i += 1;
                    j += 1;
                }
                std::cmp::Ordering::Less => {
                    child_conns.push(a[i].clone());
                    i += 1;
                }
                std::cmp::Ordering::Greater => {
                    j += 1; // skip lesser-only genes
                }
            }
        }
        // Remaining fitter genes
        child_conns.extend_from_slice(&a[i..]);

        Genome {
            node_genes: child_nodes,
            conn_genes: child_conns,
            next_innovation: fitter.next_innovation.max(lesser.next_innovation),
        }
    }

    pub fn to_rust_wann(&self) -> crate::wann_network::RustWannNetwork {
        let num_nodes = self.node_genes.last().map(|n| n.id).unwrap_or(0) + 1;

        let mut node_ids = Vec::with_capacity(self.node_genes.len());
        let mut node_activations = Vec::with_capacity(self.node_genes.len());
        let mut node_aggregations = Vec::with_capacity(self.node_genes.len());
        let mut n_idx = 0;
        for nid in 0..num_nodes {
            if n_idx < self.node_genes.len() && self.node_genes[n_idx].id == nid {
                node_ids.push(nid);
                node_activations.push(self.node_genes[n_idx].activation_fn as u8);
                node_aggregations.push(self.node_genes[n_idx].aggregation_fn as u8);
                n_idx += 1;
            }
        }

        let enabled: Vec<&ConnGene> = self.conn_genes.iter().filter(|c| c.enabled).collect();
        let conn_srcs: Vec<usize> = enabled.iter().map(|c| c.src).collect();
        let conn_dsts: Vec<usize> = enabled.iter().map(|c| c.dst).collect();
        let conn_signs: Vec<i8> = enabled.iter().map(|c| c.sign).collect();
        let conn_enableds: Vec<bool> = vec![true; enabled.len()];

        crate::wann_network::RustWannNetwork::new(
            &node_ids,
            &node_activations,
            &node_aggregations,
            &conn_srcs,
            &conn_dsts,
            &conn_signs,
            &conn_enableds,
        )
    }

    pub fn topological_order(&self) -> Vec<usize> {
        let node_set: HashSet<usize> = self.node_genes.iter().map(|n| n.id).collect();
        let conns: Vec<&ConnGene> = self.conn_genes.iter().collect();

        let mut adj: std::collections::HashMap<usize, Vec<usize>> =
            node_set.iter().map(|&nid| (nid, Vec::new())).collect();
        let mut in_degree: std::collections::HashMap<usize, usize> =
            node_set.iter().map(|&nid| (nid, 0)).collect();

        for c in &conns {
            if c.enabled && node_set.contains(&c.src) && node_set.contains(&c.dst) {
                adj.entry(c.src).or_default().push(c.dst);
                *in_degree.entry(c.dst).or_default() += 1;
            }
        }

        let hidden_ids = self.hidden_ids();
        let mut priority = std::collections::HashMap::new();
        let mut pri = 0;
        for i in 0..INPUT_COUNT {
            priority.insert(INPUT_START + i, pri);
            pri += 1;
        }
        priority.insert(BIAS_ID, pri);
        pri += 1;
        for &hid in &hidden_ids {
            priority.insert(hid, pri);
            pri += 1;
        }
        for i in 0..OUTPUT_COUNT {
            priority.insert(OUTPUT_START + i, pri);
            pri += 1;
        }

        use std::cmp::Reverse;
        let mut heap: BinaryHeap<Reverse<(usize, usize)>> = node_set
            .iter()
            .copied()
            .filter(|nid| *in_degree.get(nid).unwrap_or(&0) == 0)
            .map(|nid| Reverse((*priority.get(&nid).unwrap_or(&999), nid)))
            .collect();

        let mut order = Vec::with_capacity(node_set.len());
        let mut visited = HashSet::new();

        while let Some(Reverse((_, nid))) = heap.pop() {
            if !visited.insert(nid) {
                continue;
            }
            order.push(nid);
            if let Some(neighbors) = adj.get(&nid) {
                for &neighbor in neighbors {
                    let degree = in_degree.get_mut(&neighbor).unwrap();
                    *degree -= 1;
                    if *degree == 0 {
                        heap.push(Reverse((
                            *priority.get(&neighbor).unwrap_or(&999),
                            neighbor,
                        )));
                    }
                }
            }
        }

        let mut remaining: Vec<usize> = node_set
            .iter()
            .copied()
            .filter(|nid| !visited.contains(nid))
            .collect();
        remaining.sort_by_key(|n| *priority.get(n).unwrap_or(&999));
        order.extend(remaining);
        order
    }

    pub fn calculate_complexity(&self) -> f64 {
        let mut adj_forward: std::collections::HashMap<usize, Vec<usize>> =
            std::collections::HashMap::new();
        let mut adj_backward: std::collections::HashMap<usize, Vec<usize>> =
            std::collections::HashMap::new();

        for conn in &self.conn_genes {
            if conn.enabled {
                adj_forward.entry(conn.src).or_default().push(conn.dst);
                adj_backward.entry(conn.dst).or_default().push(conn.src);
            }
        }

        let mut visited_forward = HashSet::new();
        let mut stack = Vec::new();
        for i in 0..=BIAS_ID {
            if self.has_node(i) {
                stack.push(i);
                visited_forward.insert(i);
            }
        }
        while let Some(node) = stack.pop() {
            if let Some(neighbors) = adj_forward.get(&node) {
                for &neighbor in neighbors {
                    if visited_forward.insert(neighbor) {
                        stack.push(neighbor);
                    }
                }
            }
        }

        let mut visited_backward = HashSet::new();
        for i in OUTPUT_START..(OUTPUT_START + OUTPUT_COUNT) {
            if self.has_node(i) {
                stack.push(i);
                visited_backward.insert(i);
            }
        }
        while let Some(node) = stack.pop() {
            if let Some(neighbors) = adj_backward.get(&node) {
                for &neighbor in neighbors {
                    if visited_backward.insert(neighbor) {
                        stack.push(neighbor);
                    }
                }
            }
        }

        let mut reachable_hidden = 0;
        let mut unreachable_hidden = 0;
        for n in &self.node_genes {
            if n.node_type == NodeType::HIDDEN {
                if visited_forward.contains(&n.id) && visited_backward.contains(&n.id) {
                    reachable_hidden += 1;
                } else {
                    unreachable_hidden += 1;
                }
            }
        }

        let mut count_functional = 0;
        let mut count_disabled = 0;
        for conn in &self.conn_genes {
            if conn.enabled {
                if visited_forward.contains(&conn.src)
                    && visited_backward.contains(&conn.src)
                    && visited_forward.contains(&conn.dst)
                    && visited_backward.contains(&conn.dst)
                {
                    count_functional += 1;
                }
            } else {
                count_disabled += 1;
            }
        }

        let penalty = 100.0;
        let beta = 0.05;
        (count_functional as f64)
            + 0.5 * (reachable_hidden as f64)
            + penalty * (unreachable_hidden as f64)
            + beta * (count_disabled as f64)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JsonGenomeJoint {
    pub lead: Option<JsonGenome>,
    pub follow: Option<JsonGenome>,
}

// Flat structure for JSON serialization compatibility with Python
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

impl JsonGenome {
    pub fn from_genome(genome: &Genome) -> Self {
        let node_ids: Vec<i32> = genome.node_genes.iter().map(|n| n.id as i32).collect();
        let node_types: Vec<i32> = genome
            .node_genes
            .iter()
            .map(|n| n.node_type as i32)
            .collect();
        let node_acts: Vec<i32> = genome
            .node_genes
            .iter()
            .map(|n| n.activation_fn as i32)
            .collect();
        let node_aggs: Vec<i32> = genome
            .node_genes
            .iter()
            .map(|n| n.aggregation_fn as i32)
            .collect();

        let conn_innovs: Vec<i32> = genome
            .conn_genes
            .iter()
            .map(|c| c.innovation as i32)
            .collect();
        let conn_srcs: Vec<i32> = genome.conn_genes.iter().map(|c| c.src as i32).collect();
        let conn_dsts: Vec<i32> = genome.conn_genes.iter().map(|c| c.dst as i32).collect();
        let conn_signs: Vec<i32> = genome.conn_genes.iter().map(|c| c.sign as i32).collect();
        let conn_enabled: Vec<i32> = genome
            .conn_genes
            .iter()
            .map(|c| if c.enabled { 1 } else { 0 })
            .collect();

        Self {
            next_innovation: genome.next_innovation,
            node_ids,
            node_types,
            node_acts,
            node_aggs,
            conn_innovs,
            conn_srcs,
            conn_dsts,
            conn_signs,
            conn_enabled,
        }
    }

    #[allow(dead_code)]
    pub fn to_genome(&self) -> Genome {
        let node_genes: Vec<NodeGene> = (0..self.node_ids.len())
            .map(|i| {
                let node_type = match self.node_types[i] {
                    0 => NodeType::INPUT,
                    1 => NodeType::BIAS,
                    2 => NodeType::HIDDEN,
                    3 => NodeType::OUTPUT,
                    _ => panic!("Unknown node type: {}", self.node_types[i]),
                };
                let activation_fn = match self.node_acts[i] {
                    0 => ActivationFn::IDENTITY,
                    1 => ActivationFn::NOT,
                    2 => ActivationFn::THRESHOLD,
                    _ => panic!("Unknown activation: {}", self.node_acts[i]),
                };
                let aggregation_fn = match self.node_aggs[i] {
                    0 => AggregationFn::SUM,
                    1 => AggregationFn::MIN,
                    2 => AggregationFn::MAX,
                    _ => panic!("Unknown aggregation: {}", self.node_aggs[i]),
                };
                NodeGene::make(
                    self.node_ids[i] as usize,
                    node_type,
                    activation_fn,
                    aggregation_fn,
                )
            })
            .collect();

        let conn_genes: Vec<ConnGene> = (0..self.conn_innovs.len())
            .map(|i| {
                ConnGene::make(
                    self.conn_innovs[i] as usize,
                    self.conn_srcs[i] as usize,
                    self.conn_dsts[i] as usize,
                    self.conn_signs[i] as i8,
                    self.conn_enabled[i] != 0,
                )
            })
            .collect();

        Genome::new(Some(node_genes), Some(conn_genes), self.next_innovation)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn random_genome() -> Genome {
        let mut g = Genome::initial();
        g.add_node(NodeGene::make(
            FIRST_HIDDEN_ID,
            NodeType::HIDDEN,
            ActivationFn::IDENTITY,
            AggregationFn::SUM,
        ));
        g.add_connection(ConnGene::make(1, 0, OUTPUT_START, 1, true));
        g.add_connection(ConnGene::make(2, 1, OUTPUT_START + 1, -1, true));
        g.add_connection(ConnGene::make(3, 5, FIRST_HIDDEN_ID, 1, true));
        g.add_connection(ConnGene::make(
            4,
            FIRST_HIDDEN_ID,
            OUTPUT_START + 2,
            1,
            true,
        ));
        g
    }

    #[test]
    fn test_topological_order_all_nodes_once() {
        let g = random_genome();
        let order = g.topological_order();
        let node_ids = g.node_ids();
        assert_eq!(order.len(), node_ids.len());
        for &nid in &node_ids {
            assert!(order.contains(&nid), "Node {} missing from order", nid);
        }
    }

    #[test]
    fn test_topological_order_edges_forward() {
        let g = random_genome();
        let order = g.topological_order();
        let pos: std::collections::HashMap<usize, usize> =
            order.iter().enumerate().map(|(i, &n)| (n, i)).collect();
        for c in &g.conn_genes {
            if c.enabled {
                let src_pos = pos[&c.src];
                let dst_pos = pos[&c.dst];
                assert!(
                    src_pos < dst_pos,
                    "Edge {} -> {} goes backwards",
                    c.src,
                    c.dst
                );
            }
        }
    }

    #[test]
    fn test_json_round_trip() {
        let g = random_genome();
        let json = JsonGenome::from_genome(&g);
        let g2 = json.to_genome();
        assert_eq!(g.node_ids(), g2.node_ids());
        assert_eq!(g.conn_genes.len(), g2.conn_genes.len());
        for c in &g.conn_genes {
            let c2 = g2.get_conn_by_inno(c.innovation).unwrap();
            assert_eq!(c.src, c2.src);
            assert_eq!(c.dst, c2.dst);
            assert_eq!(c.sign, c2.sign);
            assert_eq!(c.enabled, c2.enabled);
        }
    }

    #[test]
    fn test_csr_round_trip() {
        let g = random_genome();
        let network = g.to_rust_wann();
        assert!(!network.topological_order.is_empty());
        assert_eq!(network.node_ptrs.len(), network.num_nodes + 1);
        let total_incoming = network.node_ptrs[network.num_nodes];
        assert_eq!(network.incoming_srcs.len(), total_incoming);
        assert_eq!(network.incoming_signs.len(), total_incoming);
        let n_enabled = g.num_enabled();
        assert_eq!(total_incoming, n_enabled);
    }

    #[test]
    fn test_crossover_bounds() {
        let g1 = random_genome();
        let mut g2 = Genome::initial();
        g2.add_node(NodeGene::make(
            FIRST_HIDDEN_ID + 1,
            NodeType::HIDDEN,
            ActivationFn::NOT,
            AggregationFn::MAX,
        ));
        g2.add_connection(ConnGene::make(5, BIAS_ID, OUTPUT_START, 1, true));
        g2.add_connection(ConnGene::make(6, 0, FIRST_HIDDEN_ID + 1, -1, true));

        let child = g1.crossover_with(&g2, true, &mut rand::thread_rng());

        let union_ids: HashSet<usize> = g1.node_ids().union(&g2.node_ids()).copied().collect();
        assert!(child.node_ids().is_subset(&union_ids));
        let max_conns = g1.conn_genes.len().max(g2.conn_genes.len());
        assert!(child.conn_genes.len() <= max_conns);
    }

    #[test]
    fn test_forward_weighted_equivalence() {
        let g = random_genome();
        let net = g.to_rust_wann();

        let mut scratch_forward = vec![0.0; net.num_nodes];
        let mut scratch_weighted = vec![0.0; net.num_nodes];

        let mut inputs = [0.0; INPUT_COUNT];
        for i in 0..INPUT_COUNT {
            inputs[i] = (i as f64) / (INPUT_COUNT as f64);
        }

        // Run forward with uniform weight W = 1.0
        net.forward(&inputs, 1.0, &mut scratch_forward);

        // Run forward_weighted with weights all set to 1.0
        let weights = vec![1.0; net.incoming_srcs.len()];
        net.forward_weighted(&inputs, &weights, &mut scratch_weighted);

        // They must produce identical values on all output and hidden nodes
        for i in 0..net.num_nodes {
            assert!(
                (scratch_forward[i] - scratch_weighted[i]).abs() < 1e-9,
                "Mismatch at node {}: forward={}, forward_weighted={}",
                i,
                scratch_forward[i],
                scratch_weighted[i]
            );
        }
    }
}
