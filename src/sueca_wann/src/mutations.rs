use crate::genome::{
    ActivationFn, AggregationFn, ConnGene, Genome, NodeGene, NodeType, FIRST_HIDDEN_ID,
};
use rand::Rng;
use std::collections::{HashMap, HashSet};

pub struct InnovationRegistry {
    pub next_innovation: usize,
    pub conns: HashMap<(usize, usize), usize>,
}

impl InnovationRegistry {
    pub fn new(next_innovation: usize) -> Self {
        Self {
            next_innovation,
            conns: HashMap::new(),
        }
    }

    pub fn get_or_create(&mut self, src: usize, dst: usize) -> usize {
        if let Some(&inno) = self.conns.get(&(src, dst)) {
            inno
        } else {
            let inno = self.next_innovation;
            self.conns.insert((src, dst), inno);
            self.next_innovation += 1;
            inno
        }
    }
}

fn pick_random_connection<R: Rng>(genome: &Genome, rng: &mut R) -> Option<ConnGene> {
    let conns: Vec<ConnGene> = genome.conn_genes.values().cloned().collect();
    if conns.is_empty() {
        return None;
    }
    let idx = rng.gen_range(0..conns.len());
    Some(conns[idx].clone())
}

fn pick_random_enabled_connection<R: Rng>(genome: &Genome, rng: &mut R) -> Option<ConnGene> {
    let enabled: Vec<ConnGene> = genome
        .conn_genes
        .values()
        .filter(|c| c.enabled)
        .cloned()
        .collect();
    if enabled.is_empty() {
        return None;
    }
    let idx = rng.gen_range(0..enabled.len());
    Some(enabled[idx].clone())
}

fn pick_random_node<R: Rng>(
    genome: &Genome,
    rng: &mut R,
    exclude_inputs: bool,
    exclude_bias: bool,
    exclude_outputs: bool,
) -> Option<NodeGene> {
    let mut candidates: Vec<NodeGene> = genome.node_genes.values().cloned().collect();
    if exclude_inputs {
        candidates.retain(|n| n.node_type != NodeType::INPUT);
    }
    if exclude_bias {
        candidates.retain(|n| n.node_type != NodeType::BIAS);
    }
    if exclude_outputs {
        candidates.retain(|n| n.node_type != NodeType::OUTPUT);
    }
    if candidates.is_empty() {
        return None;
    }
    let idx = rng.gen_range(0..candidates.len());
    Some(candidates[idx].clone())
}

pub fn mutate_add_node<R: Rng>(
    genome: &mut Genome,
    registry: &mut InnovationRegistry,
    rng: &mut R,
) -> bool {
    let conn = match pick_random_enabled_connection(genome, rng) {
        Some(c) => c,
        None => return false,
    };

    // Disable the old connection
    if let Some(c) = genome.conn_genes.get_mut(&conn.innovation) {
        c.enabled = false;
    }

    // Create a new hidden node
    let existing_ids = genome.node_ids();
    let max_id = existing_ids
        .iter()
        .max()
        .copied()
        .unwrap_or(FIRST_HIDDEN_ID - 1);
    let node_id = max_id + 1;

    let all_activations = [
        ActivationFn::IDENTITY,
        ActivationFn::NOT,
        ActivationFn::THRESHOLD,
    ];
    let all_aggregations = [AggregationFn::SUM, AggregationFn::MIN, AggregationFn::MAX];

    let act = all_activations[rng.gen_range(0..all_activations.len())];
    let agg = all_aggregations[rng.gen_range(0..all_aggregations.len())];

    let new_node = NodeGene::make(node_id, NodeType::HIDDEN, act, agg);
    genome.add_node(new_node);

    // Add src -> new_node (sign = +1)
    let inno1 = registry.get_or_create(conn.src, node_id);
    genome.add_connection(ConnGene::make(inno1, conn.src, node_id, 1, true));

    // Add new_node -> dst (sign = original_sign)
    let inno2 = registry.get_or_create(node_id, conn.dst);
    genome.add_connection(ConnGene::make(inno2, node_id, conn.dst, conn.sign, true));

    genome.next_innovation = registry.next_innovation;

    true
}

pub fn mutate_add_connection<R: Rng>(
    genome: &mut Genome,
    registry: &mut InnovationRegistry,
    rng: &mut R,
) -> bool {
    let order = genome.topological_order();
    let mut depth = HashMap::new();
    for (i, &nid) in order.iter().enumerate() {
        depth.insert(nid, i);
    }

    let existing: HashSet<(usize, usize)> =
        genome.conn_genes.values().map(|c| (c.src, c.dst)).collect();

    let mut sources = Vec::new();
    let mut destinations = Vec::new();

    for &nid in &order {
        if let Some(node) = genome.node_genes.get(&nid) {
            if node.node_type != NodeType::OUTPUT {
                sources.push(nid);
            }
            if node.node_type != NodeType::INPUT && node.node_type != NodeType::BIAS {
                destinations.push(nid);
            }
        }
    }

    let mut candidates = Vec::new();
    for &src in &sources {
        let src_depth = depth[&src];
        for &dst in &destinations {
            if src_depth >= depth[&dst] {
                continue;
            }
            if !existing.contains(&(src, dst)) {
                candidates.push((src, dst));
            }
        }
    }

    if candidates.is_empty() {
        return false;
    }

    let idx = rng.gen_range(0..candidates.len());
    let (src, dst) = candidates[idx];
    let sign = if rng.gen_bool(0.5) { 1 } else { -1 };

    let inno = registry.get_or_create(src, dst);
    genome.add_connection(ConnGene::make(inno, src, dst, sign, true));

    genome.next_innovation = registry.next_innovation;

    true
}

pub fn mutate_toggle_connection<R: Rng>(genome: &mut Genome, rng: &mut R) -> bool {
    let conn = match pick_random_connection(genome, rng) {
        Some(c) => c,
        None => return false,
    };

    if let Some(c) = genome.conn_genes.get_mut(&conn.innovation) {
        c.enabled = !c.enabled;
        true
    } else {
        false
    }
}

pub fn mutate_flip_sign<R: Rng>(genome: &mut Genome, rng: &mut R) -> bool {
    let conn = match pick_random_connection(genome, rng) {
        Some(c) => c,
        None => return false,
    };

    if let Some(c) = genome.conn_genes.get_mut(&conn.innovation) {
        c.sign = -c.sign;
        true
    } else {
        false
    }
}

pub fn mutate_change_activation<R: Rng>(genome: &mut Genome, rng: &mut R) -> bool {
    let node = match pick_random_node(genome, rng, true, true, true) {
        Some(n) => n,
        None => return false,
    };

    let all_activations = [
        ActivationFn::IDENTITY,
        ActivationFn::NOT,
        ActivationFn::THRESHOLD,
    ];
    let current = node.activation_fn;
    let choices: Vec<ActivationFn> = all_activations
        .iter()
        .copied()
        .filter(|&a| a != current)
        .collect();
    if choices.is_empty() {
        return false;
    }

    let new_fn = choices[rng.gen_range(0..choices.len())];
    if let Some(n) = genome.node_genes.get_mut(&node.id) {
        n.activation_fn = new_fn;
        true
    } else {
        false
    }
}

pub fn mutate_change_aggregation<R: Rng>(genome: &mut Genome, rng: &mut R) -> bool {
    let node = match pick_random_node(genome, rng, true, true, true) {
        Some(n) => n,
        None => return false,
    };

    let all_aggregations = [AggregationFn::SUM, AggregationFn::MIN, AggregationFn::MAX];
    let current = node.aggregation_fn;
    let choices: Vec<AggregationFn> = all_aggregations
        .iter()
        .copied()
        .filter(|&a| a != current)
        .collect();
    if choices.is_empty() {
        return false;
    }

    let new_fn = choices[rng.gen_range(0..choices.len())];
    if let Some(n) = genome.node_genes.get_mut(&node.id) {
        n.aggregation_fn = new_fn;
        true
    } else {
        false
    }
}

pub fn apply_mutations<R: Rng>(
    genome: &mut Genome,
    registry: &mut InnovationRegistry,
    rng: &mut R,
    p_add_node: f64,
    p_add_conn: f64,
    p_toggle_conn: f64,
    p_flip_sign: f64,
    p_change_act: f64,
    p_change_agg: f64,
) -> usize {
    let mut n = 0;
    if rng.gen_bool(p_add_node) {
        if mutate_add_node(genome, registry, rng) {
            n += 1;
        }
    }
    if rng.gen_bool(p_add_conn) {
        if mutate_add_connection(genome, registry, rng) {
            n += 1;
        }
    }
    if rng.gen_bool(p_toggle_conn) {
        if mutate_toggle_connection(genome, rng) {
            n += 1;
        }
    }
    if rng.gen_bool(p_flip_sign) {
        if mutate_flip_sign(genome, rng) {
            n += 1;
        }
    }
    if rng.gen_bool(p_change_act) {
        if mutate_change_activation(genome, rng) {
            n += 1;
        }
    }
    if rng.gen_bool(p_change_agg) {
        if mutate_change_aggregation(genome, rng) {
            n += 1;
        }
    }
    n
}
