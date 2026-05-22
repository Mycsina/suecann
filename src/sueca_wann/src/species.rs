use crate::genome::{Genome, NodeType};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Species {
    pub id: usize,
    pub representative: Genome,
    pub members: Vec<usize>, // indices into the population array
    pub best_fitness: f64,
    pub generations_no_improvement: usize,
    pub created_at_gen: usize,
}

impl Species {
    pub fn new(id: usize, representative: Genome, created_at_gen: usize) -> Self {
        Self {
            id,
            representative,
            members: Vec::new(),
            best_fitness: f64::NEG_INFINITY,
            generations_no_improvement: 0,
            created_at_gen,
        }
    }

    pub fn update_best(&mut self, fitness: f64) -> bool {
        if fitness > self.best_fitness {
            self.best_fitness = fitness;
            self.generations_no_improvement = 0;
            true
        } else {
            false
        }
    }

    pub fn increment_stagnation(&mut self) {
        self.generations_no_improvement += 1;
    }
}

pub fn compatibility_distance(
    genome_a: &Genome,
    genome_b: &Genome,
    c1: f64,
    c2: f64,
    c3: f64,
) -> f64 {
    let conn_innov_a: HashSet<usize> = genome_a.conn_genes.keys().copied().collect();
    let conn_innov_b: HashSet<usize> = genome_b.conn_genes.keys().copied().collect();

    if conn_innov_a.is_empty() && conn_innov_b.is_empty() {
        return 0.0;
    }

    let max_innov_a = conn_innov_a.iter().max().copied().unwrap_or(0);
    let max_innov_b = conn_innov_b.iter().max().copied().unwrap_or(0);

    let mut shared = 0;
    let mut disjoint = 0;
    let mut excess = 0;
    let mut sign_diff_total = 0.0;

    // We can iterate over the union/intersection of connection innovations.
    // To make it efficient and simple:
    for &innov in conn_innov_a.union(&conn_innov_b) {
        let has_a = conn_innov_a.contains(&innov);
        let has_b = conn_innov_b.contains(&innov);

        if has_a && has_b {
            shared += 1;
            let ca = &genome_a.conn_genes[&innov];
            let cb = &genome_b.conn_genes[&innov];
            if ca.sign != cb.sign {
                sign_diff_total += 1.0;
            }
            if ca.enabled != cb.enabled {
                sign_diff_total += 0.5;
            }
        } else if has_a {
            if innov < max_innov_b {
                disjoint += 1;
            } else {
                excess += 1;
            }
        } else {
            if innov < max_innov_a {
                disjoint += 1;
            } else {
                excess += 1;
            }
        }
    }

    let n_conns = conn_innov_a.len().max(conn_innov_b.len()).max(1) as f64;
    let avg_sign_diff = if shared > 0 {
        sign_diff_total / shared as f64
    } else {
        0.0
    };

    // Symmetric hidden node comparison: union of both genomes' hidden node IDs.
    let mut node_diff = 0.0;
    let hidden_a: HashMap<usize, &crate::genome::NodeGene> = genome_a
        .node_genes
        .iter()
        .filter(|(_, ng)| ng.node_type == NodeType::HIDDEN)
        .map(|(&id, ng)| (id, ng))
        .collect();
    let hidden_b: HashMap<usize, &crate::genome::NodeGene> = genome_b
        .node_genes
        .iter()
        .filter(|(_, ng)| ng.node_type == NodeType::HIDDEN)
        .map(|(&id, ng)| (id, ng))
        .collect();

    let all_hidden_ids: HashSet<usize> = hidden_a
        .keys()
        .copied()
        .chain(hidden_b.keys().copied())
        .collect();

    for nid in all_hidden_ids {
        match (hidden_a.get(&nid), hidden_b.get(&nid)) {
            (Some(ng_a), Some(ng_b)) => {
                if ng_a.activation_fn != ng_b.activation_fn {
                    node_diff += 0.5;
                }
                if ng_a.aggregation_fn != ng_b.aggregation_fn {
                    node_diff += 0.5;
                }
            }
            _ => {
                // Disjoint/excess hidden node
                node_diff += 1.0;
            }
        }
    }

    c1 * excess as f64 / n_conns
        + c2 * disjoint as f64 / n_conns
        + c3 * avg_sign_diff
        + c3 * node_diff
}

pub fn speciate(
    genomes: &[Genome],
    species_list: &mut Vec<Species>,
    threshold: f64,
    mut next_species_id: usize,
    created_at_gen: usize,
    c1: f64,
    c2: f64,
    c3: f64,
) -> usize {
    // Reset all species members
    for sp in species_list.iter_mut() {
        sp.members.clear();
    }

    let mut unassigned: Vec<usize> = (0..genomes.len()).collect();

    // Assign to existing species
    for sp in species_list.iter_mut() {
        if unassigned.is_empty() {
            break;
        }

        let mut still_unassigned = Vec::new();
        for &idx in &unassigned {
            let dist = compatibility_distance(&sp.representative, &genomes[idx], c1, c2, c3);
            if dist < threshold {
                sp.members.push(idx);
            } else {
                still_unassigned.push(idx);
            }
        }
        unassigned = still_unassigned;
    }

    // Create new species for remaining genomes
    while !unassigned.is_empty() {
        let idx = unassigned[0];
        let mut new_sp = Species::new(next_species_id, genomes[idx].copy(), created_at_gen);
        next_species_id += 1;
        new_sp.members.push(idx);

        unassigned.remove(0);

        let mut still_unassigned = Vec::new();
        for &other_idx in &unassigned {
            let dist =
                compatibility_distance(&new_sp.representative, &genomes[other_idx], c1, c2, c3);
            if dist < threshold {
                new_sp.members.push(other_idx);
            } else {
                still_unassigned.push(other_idx);
            }
        }
        unassigned = still_unassigned;

        species_list.push(new_sp);
    }

    next_species_id
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::genome::{ActivationFn, AggregationFn, ConnGene, NodeGene, NodeType};

    fn make_genome(conns: Vec<(usize, usize, i8)>) -> Genome {
        let mut g = Genome::initial();
        for (i, (src, dst, sign)) in conns.into_iter().enumerate() {
            g.add_connection(ConnGene::make(i, src, dst, sign, true));
        }
        g
    }

    fn make_genome_with_hidden(
        conns: Vec<(usize, usize, i8)>,
        hidden_nodes: Vec<(usize, ActivationFn, AggregationFn)>,
    ) -> Genome {
        let mut g = make_genome(conns);
        for (id, act, agg) in hidden_nodes {
            g.add_node(NodeGene::make(id, NodeType::HIDDEN, act, agg));
        }
        g
    }

    #[test]
    fn test_compatibility_identity() {
        let g = make_genome(vec![(0, 22, 1), (1, 23, -1)]);
        let dist = compatibility_distance(&g, &g, 1.0, 1.0, 0.4);
        assert!(
            dist.abs() < 1e-9,
            "Distance to self should be 0, got {}",
            dist
        );
    }

    #[test]
    fn test_compatibility_symmetry() {
        let a = make_genome(vec![(0, 22, 1), (5, 23, -1)]);
        let b = make_genome(vec![(0, 22, 1), (21, 24, 1)]);
        let d_ab = compatibility_distance(&a, &b, 1.0, 1.0, 0.4);
        let d_ba = compatibility_distance(&b, &a, 1.0, 1.0, 0.4);
        assert!(
            (d_ab - d_ba).abs() < 1e-9,
            "Distance should be symmetric: {} vs {}",
            d_ab,
            d_ba
        );
    }

    #[test]
    fn test_compatibility_symmetry_with_hidden_nodes() {
        let a = make_genome_with_hidden(
            vec![(0, 27, 1), (27, 22, 1)],
            vec![(27, ActivationFn::IDENTITY, AggregationFn::SUM)],
        );
        let b = make_genome_with_hidden(
            vec![(0, 28, 1), (28, 22, 1)],
            vec![(28, ActivationFn::NOT, AggregationFn::MAX)],
        );
        let d_ab = compatibility_distance(&a, &b, 1.0, 1.0, 0.4);
        let d_ba = compatibility_distance(&b, &a, 1.0, 1.0, 0.4);
        assert!(
            (d_ab - d_ba).abs() < 1e-9,
            "Distance with hidden nodes should be symmetric: {} vs {}",
            d_ab,
            d_ba
        );
    }

    #[test]
    fn test_compatibility_triangle_inequality() {
        let a = make_genome(vec![(0, 22, 1), (1, 23, -1)]);
        let b = make_genome(vec![(0, 22, 1), (21, 24, 1)]);
        let c = make_genome(vec![(5, 25, 1), (6, 26, -1)]);
        let d_ab = compatibility_distance(&a, &b, 1.0, 1.0, 0.4);
        let d_bc = compatibility_distance(&b, &c, 1.0, 1.0, 0.4);
        let d_ac = compatibility_distance(&a, &c, 1.0, 1.0, 0.4);
        assert!(
            d_ab + d_bc + 1e-9 >= d_ac,
            "Triangle inequality violated: {} + {} < {}",
            d_ab,
            d_bc,
            d_ac
        );
    }

    #[test]
    fn test_disjoint_hidden_node_penalized() {
        // Identical genomes except one has an extra hidden node
        let a = make_genome_with_hidden(
            vec![(0, 22, 1)],
            vec![(27, ActivationFn::IDENTITY, AggregationFn::SUM)],
        );
        let b = make_genome(vec![(0, 22, 1)]);
        let d = compatibility_distance(&a, &b, 1.0, 1.0, 0.4);
        // Disjoint hidden node: 1.0 * c3 = 0.4
        assert!(d > 0.0, "Disjoint hidden node should increase distance");
    }

    #[test]
    fn test_speciation_preserves_all_genomes() {
        let genomes = vec![
            make_genome(vec![(0, 22, 1)]),
            make_genome(vec![(0, 22, 1), (1, 23, -1)]),
            make_genome(vec![(5, 24, 1)]),
        ];
        let mut species_list = Vec::new();
        let next_id = speciate(&genomes, &mut species_list, 0.5, 0, 0, 1.0, 1.0, 0.4);
        assert!(next_id > 0);
        let total_assigned: usize = species_list.iter().map(|s| s.members.len()).sum();
        assert_eq!(total_assigned, genomes.len());
    }
}
