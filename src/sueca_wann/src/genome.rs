use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};

pub const INPUT_START: usize = 0;
pub const INPUT_COUNT: usize = 21;
pub const BIAS_ID: usize = INPUT_START + INPUT_COUNT; // 21
pub const OUTPUT_START: usize = BIAS_ID + 1; // 22
pub const OUTPUT_COUNT: usize = 5;
pub const FIRST_HIDDEN_ID: usize = OUTPUT_START + OUTPUT_COUNT; // 27

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum NodeType {
    INPUT = 0,
    BIAS = 1,
    HIDDEN = 2,
    OUTPUT = 3,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ActivationFn {
    IDENTITY = 0,
    NOT = 1,
    THRESHOLD = 2,
}

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
    pub sign: i8, // +1 or -1
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

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Genome {
    pub node_genes: HashMap<usize, NodeGene>,
    pub conn_genes: HashMap<usize, ConnGene>,
    pub next_innovation: usize,
}

impl Genome {
    pub fn new(
        node_genes: Option<Vec<NodeGene>>,
        conn_genes: Option<Vec<ConnGene>>,
        next_innovation: usize,
    ) -> Self {
        let mut nodes = HashMap::new();
        let mut conns = HashMap::new();

        if let Some(ng_list) = node_genes {
            for ng in ng_list {
                nodes.insert(ng.id, ng);
            }
        } else {
            // initial node genes
            for i in 0..INPUT_COUNT {
                nodes.insert(
                    INPUT_START + i,
                    NodeGene::make(
                        INPUT_START + i,
                        NodeType::INPUT,
                        ActivationFn::IDENTITY,
                        AggregationFn::SUM,
                    ),
                );
            }
            nodes.insert(
                BIAS_ID,
                NodeGene::make(
                    BIAS_ID,
                    NodeType::BIAS,
                    ActivationFn::IDENTITY,
                    AggregationFn::SUM,
                ),
            );
            for i in 0..OUTPUT_COUNT {
                nodes.insert(
                    OUTPUT_START + i,
                    NodeGene::make(
                        OUTPUT_START + i,
                        NodeType::OUTPUT,
                        ActivationFn::IDENTITY,
                        AggregationFn::SUM,
                    ),
                );
            }
        }

        if let Some(cg_list) = conn_genes {
            for cg in cg_list {
                conns.insert(cg.innovation, cg);
            }
        }

        let mut next_inno = next_innovation;
        if !conns.is_empty() {
            let max_inno = *conns.keys().max().unwrap();
            if next_inno <= max_inno {
                next_inno = max_inno + 1;
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

    pub fn node_ids(&self) -> HashSet<usize> {
        self.node_genes.keys().copied().collect()
    }

    pub fn hidden_ids(&self) -> Vec<usize> {
        let mut ids: Vec<usize> = self
            .node_genes
            .values()
            .filter(|n| n.node_type == NodeType::HIDDEN)
            .map(|n| n.id)
            .collect();
        ids.sort_unstable();
        ids
    }

    pub fn add_node(&mut self, node: NodeGene) {
        self.node_genes.insert(node.id, node);
    }

    pub fn add_connection(&mut self, conn: ConnGene) {
        let inno = conn.innovation;
        self.conn_genes.insert(inno, conn);
        if inno >= self.next_innovation {
            self.next_innovation = inno + 1;
        }
    }

    pub fn has_connection(&self, src: usize, dst: usize) -> bool {
        self.conn_genes
            .values()
            .any(|c| c.src == src && c.dst == dst)
    }

    pub fn get_connection(&self, src: usize, dst: usize) -> Option<&ConnGene> {
        self.conn_genes
            .values()
            .find(|c| c.src == src && c.dst == dst)
    }

    pub fn enabled_connections(&self) -> Vec<&ConnGene> {
        self.conn_genes.values().filter(|c| c.enabled).collect()
    }

    pub fn num_enabled(&self) -> usize {
        self.conn_genes.values().filter(|c| c.enabled).count()
    }

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

    pub fn to_rust_wann(&self) -> sueca_solver::wann::RustWannNetwork {
        let num_nodes = self.node_genes.keys().max().copied().unwrap_or(0) + 1;

        let mut node_ids = Vec::new();
        let mut node_activations = Vec::new();
        let mut node_aggregations = Vec::new();
        for nid in 0..num_nodes {
            if let Some(ng) = self.node_genes.get(&nid) {
                node_ids.push(nid);
                node_activations.push(ng.activation_fn as u8);
                node_aggregations.push(ng.aggregation_fn as u8);
            }
        }

        let enabled: Vec<&ConnGene> = self.enabled_connections();
        let conn_srcs: Vec<usize> = enabled.iter().map(|c| c.src).collect();
        let conn_dsts: Vec<usize> = enabled.iter().map(|c| c.dst).collect();
        let conn_signs: Vec<i8> = enabled.iter().map(|c| c.sign).collect();
        let conn_enableds: Vec<bool> = vec![true; enabled.len()];

        sueca_solver::wann::RustWannNetwork::new(
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
        let node_ids = self.node_ids();
        let conns = self.conn_genes.values().collect::<Vec<_>>();

        // Build adjacency and in-degree
        let mut adj: HashMap<usize, Vec<usize>> =
            node_ids.iter().map(|&nid| (nid, Vec::new())).collect();
        let mut in_degree: HashMap<usize, usize> = node_ids.iter().map(|&nid| (nid, 0)).collect();

        for c in conns {
            if c.enabled && node_ids.contains(&c.src) && node_ids.contains(&c.dst) {
                adj.entry(c.src).or_default().push(c.dst);
                *in_degree.entry(c.dst).or_default() += 1;
            }
        }

        let hidden_ids = self.hidden_ids();

        // Priority list
        let mut priority_list = Vec::new();
        for i in 0..INPUT_COUNT {
            priority_list.push(INPUT_START + i);
        }
        priority_list.push(BIAS_ID);
        priority_list.extend(hidden_ids);
        for i in 0..OUTPUT_COUNT {
            priority_list.push(OUTPUT_START + i);
        }

        let mut priority = HashMap::new();
        for (i, &nid) in priority_list.iter().enumerate() {
            priority.insert(nid, i);
        }

        // Kahn's algorithm with priority tiebreaker
        let mut queue: Vec<usize> = node_ids
            .iter()
            .copied()
            .filter(|nid| *in_degree.get(nid).unwrap_or(&0) == 0)
            .collect();

        // sort queue by priority ascending
        queue.sort_by_key(|n| *priority.get(n).unwrap_or(&999));

        let mut order = Vec::new();
        let mut visited = HashSet::new();

        while !queue.is_empty() {
            let nid = queue.remove(0);
            if visited.contains(&nid) {
                continue;
            }
            visited.insert(nid);
            order.push(nid);

            if let Some(neighbors) = adj.get(&nid) {
                for &neighbor in neighbors {
                    let degree = in_degree.get_mut(&neighbor).unwrap();
                    *degree -= 1;
                    if *degree == 0 {
                        queue.push(neighbor);
                        queue.sort_by_key(|n| *priority.get(n).unwrap_or(&999));
                    }
                }
            }
        }

        // Any remaining nodes not in adjacency graph
        let mut remaining: Vec<usize> = node_ids
            .iter()
            .copied()
            .filter(|nid| !visited.contains(nid))
            .collect();
        remaining.sort_by_key(|n| *priority.get(n).unwrap_or(&999));
        order.extend(remaining);

        order
    }
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

#[cfg(test)]
mod tests {
    use super::*;

    fn random_genome() -> Genome {
        let mut g = Genome::initial();
        // Add a hidden node
        g.add_node(NodeGene::make(
            27,
            NodeType::HIDDEN,
            ActivationFn::IDENTITY,
            AggregationFn::SUM,
        ));
        // Add some connections
        g.add_connection(ConnGene::make(1, 0, 22, 1, true));
        g.add_connection(ConnGene::make(2, 1, 23, -1, true));
        g.add_connection(ConnGene::make(3, 5, 27, 1, true));
        g.add_connection(ConnGene::make(4, 27, 24, 1, true));
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

        for c in g.conn_genes.values() {
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
        for (inno, c) in &g.conn_genes {
            let c2 = g2.conn_genes.get(inno).unwrap();
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

        // Topological order has all nodes
        assert!(!network.topological_order.is_empty());
        // Node ptrs have correct length
        assert_eq!(network.node_ptrs.len(), network.num_nodes + 1);
        // Incoming arrays match ptrs
        let total_incoming = network.node_ptrs[network.num_nodes];
        assert_eq!(network.incoming_srcs.len(), total_incoming);
        assert_eq!(network.incoming_signs.len(), total_incoming);
        // Number of incoming entries matches number of enabled connections
        let n_enabled = g.num_enabled();
        assert_eq!(total_incoming, n_enabled);
    }

    #[test]
    fn test_crossover_bounds() {
        let g1 = random_genome();
        let mut g2 = Genome::initial();
        g2.add_node(NodeGene::make(
            28,
            NodeType::HIDDEN,
            ActivationFn::NOT,
            AggregationFn::MAX,
        ));
        g2.add_connection(ConnGene::make(5, 21, 22, 1, true));
        g2.add_connection(ConnGene::make(6, 0, 28, -1, true));

        let child = crate::population::crossover(
            &g1,
            &g2,
            1.0, // g1 fitter
            0.5, // g2 less fit
            &mut rand::thread_rng(),
        );

        // Child should have at most the union of nodes
        let union_ids: std::collections::HashSet<usize> =
            g1.node_ids().union(&g2.node_ids()).copied().collect();
        assert!(child.node_ids().is_subset(&union_ids));

        // Child connections bounded by max of parents
        let max_conns = g1.conn_genes.len().max(g2.conn_genes.len());
        assert!(child.conn_genes.len() <= max_conns);
    }
}

impl JsonGenome {
    pub fn from_genome(genome: &Genome) -> Self {
        let mut node_ids = Vec::new();
        let mut node_types = Vec::new();
        let mut node_acts = Vec::new();
        let mut node_aggs = Vec::new();

        let mut sorted_nodes: Vec<&NodeGene> = genome.node_genes.values().collect();
        sorted_nodes.sort_by_key(|n| n.id);

        for n in sorted_nodes {
            node_ids.push(n.id as i32);
            node_types.push(n.node_type as i32);
            node_acts.push(n.activation_fn as i32);
            node_aggs.push(n.aggregation_fn as i32);
        }

        let mut conn_innovs = Vec::new();
        let mut conn_srcs = Vec::new();
        let mut conn_dsts = Vec::new();
        let mut conn_signs = Vec::new();
        let mut conn_enabled = Vec::new();

        let mut sorted_conns: Vec<&ConnGene> = genome.conn_genes.values().collect();
        sorted_conns.sort_by_key(|c| c.innovation);

        for c in sorted_conns {
            conn_innovs.push(c.innovation as i32);
            conn_srcs.push(c.src as i32);
            conn_dsts.push(c.dst as i32);
            conn_signs.push(c.sign as i32);
            conn_enabled.push(if c.enabled { 1 } else { 0 });
        }

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

    pub fn to_genome(&self) -> Genome {
        let mut node_genes = Vec::new();
        for i in 0..self.node_ids.len() {
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
                _ => panic!("Unknown activation function: {}", self.node_acts[i]),
            };
            let aggregation_fn = match self.node_aggs[i] {
                0 => AggregationFn::SUM,
                1 => AggregationFn::MIN,
                2 => AggregationFn::MAX,
                _ => panic!("Unknown aggregation function: {}", self.node_aggs[i]),
            };
            node_genes.push(NodeGene::make(
                self.node_ids[i] as usize,
                node_type,
                activation_fn,
                aggregation_fn,
            ));
        }

        let mut conn_genes = Vec::new();
        for i in 0..self.conn_innovs.len() {
            conn_genes.push(ConnGene::make(
                self.conn_innovs[i] as usize,
                self.conn_srcs[i] as usize,
                self.conn_dsts[i] as usize,
                self.conn_signs[i] as i8,
                self.conn_enabled[i] != 0,
            ));
        }

        Genome::new(Some(node_genes), Some(conn_genes), self.next_innovation)
    }
}
