use crate::genome::{Genome, NodeType};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

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
    let len_a = genome_a.conn_genes.len();
    let len_b = genome_b.conn_genes.len();

    if len_a == 0 && len_b == 0 {
        return 0.0;
    }

    // Compute max innovations
    let max_innov_a = genome_a
        .conn_genes
        .last()
        .map(|c| c.innovation)
        .unwrap_or(0);
    let max_innov_b = genome_b
        .conn_genes
        .last()
        .map(|c| c.innovation)
        .unwrap_or(0);

    let mut shared = 0;
    let mut disjoint = 0;
    let mut excess = 0;
    let mut sign_diff_total = 0.0;

    // Process genes from A
    for ca in &genome_a.conn_genes {
        let innov = ca.innovation;
        if let Some(cb) = genome_b.get_conn_by_inno(innov) {
            shared += 1;
            if ca.sign != cb.sign {
                sign_diff_total += 1.0;
            }
            if ca.enabled != cb.enabled {
                sign_diff_total += 0.5;
            }
        } else if innov < max_innov_b {
            disjoint += 1;
        } else {
            excess += 1;
        }
    }

    // Process genes in B not seen in A
    for cb in &genome_b.conn_genes {
        if genome_a.get_conn_by_inno(cb.innovation).is_none() {
            if cb.innovation < max_innov_a {
                disjoint += 1;
            } else {
                excess += 1;
            }
        }
    }

    let n_conns = len_a.max(len_b).max(1) as f64;
    let avg_sign_diff = if shared > 0 {
        sign_diff_total / shared as f64
    } else {
        0.0
    };

    // Symmetric hidden node comparison — zero allocation.
    let mut node_diff = 0.0;
    for ng_a in &genome_a.node_genes {
        if ng_a.node_type != NodeType::HIDDEN {
            continue;
        }
        if let Some(ng_b) = genome_b.get_node(ng_a.id) {
            if ng_b.node_type == NodeType::HIDDEN {
                if ng_a.activation_fn != ng_b.activation_fn {
                    node_diff += 0.5;
                }
                if ng_a.aggregation_fn != ng_b.aggregation_fn {
                    node_diff += 0.5;
                }
                continue;
            }
        }
        node_diff += 1.0;
    }
    for ng_b in &genome_b.node_genes {
        if ng_b.node_type != NodeType::HIDDEN {
            continue;
        }
        if let Some(ng_a) = genome_a.get_node(ng_b.id) {
            if ng_a.node_type == NodeType::HIDDEN {
                continue;
            }
        }
        node_diff += 1.0;
    }

    c1 * excess as f64 / n_conns
        + c2 * disjoint as f64 / n_conns
        + c3 * avg_sign_diff
        + c3 * node_diff
}

#[allow(clippy::too_many_arguments)]
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

    // Assign to existing species — compute distances in parallel
    for sp in species_list.iter_mut() {
        if unassigned.is_empty() {
            break;
        }

        // Parallel distance computation for all unassigned genomes
        let rep = &sp.representative;
        let results: Vec<(usize, f64)> = unassigned
            .par_iter()
            .map(|&idx| (idx, compatibility_distance(rep, &genomes[idx], c1, c2, c3)))
            .collect();

        let mut still_unassigned = Vec::new();
        for (idx, dist) in results {
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

        let rep = &new_sp.representative;
        let results: Vec<(usize, f64)> = unassigned
            .par_iter()
            .map(|&other_idx| {
                (
                    other_idx,
                    compatibility_distance(rep, &genomes[other_idx], c1, c2, c3),
                )
            })
            .collect();

        let mut still_unassigned = Vec::new();
        for (other_idx, dist) in results {
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
    use crate::genome::{
        ActivationFn, AggregationFn, ConnGene, NodeGene, NodeType, BIAS_ID, FIRST_HIDDEN_ID,
        OUTPUT_START,
    };

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
        let g = make_genome(vec![(0, OUTPUT_START, 1), (1, OUTPUT_START + 1, -1)]);
        let dist = compatibility_distance(&g, &g, 1.0, 1.0, 0.4);
        assert!(
            dist.abs() < 1e-9,
            "Distance to self should be 0, got {}",
            dist
        );
    }

    #[test]
    fn test_compatibility_symmetry() {
        let a = make_genome(vec![(0, OUTPUT_START, 1), (5, OUTPUT_START + 1, -1)]);
        let b = make_genome(vec![(0, OUTPUT_START, 1), (BIAS_ID, OUTPUT_START + 2, 1)]);
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
            vec![(0, FIRST_HIDDEN_ID, 1), (FIRST_HIDDEN_ID, OUTPUT_START, 1)],
            vec![(FIRST_HIDDEN_ID, ActivationFn::IDENTITY, AggregationFn::SUM)],
        );
        let b = make_genome_with_hidden(
            vec![
                (0, FIRST_HIDDEN_ID + 1, 1),
                (FIRST_HIDDEN_ID + 1, OUTPUT_START, 1),
            ],
            vec![(FIRST_HIDDEN_ID + 1, ActivationFn::NOT, AggregationFn::MAX)],
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
        let a = make_genome(vec![(0, OUTPUT_START, 1), (1, OUTPUT_START + 1, -1)]);
        let b = make_genome(vec![(0, OUTPUT_START, 1), (BIAS_ID, OUTPUT_START + 2, 1)]);
        let c = make_genome(vec![(5, OUTPUT_START + 2, 1), (6, OUTPUT_START + 3, -1)]);
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
            vec![(0, OUTPUT_START, 1)],
            vec![(FIRST_HIDDEN_ID, ActivationFn::IDENTITY, AggregationFn::SUM)],
        );
        let b = make_genome(vec![(0, OUTPUT_START, 1)]);
        let d = compatibility_distance(&a, &b, 1.0, 1.0, 0.4);
        // Disjoint hidden node: 1.0 * c3 = 0.4
        assert!(d > 0.0, "Disjoint hidden node should increase distance");
    }

    #[test]
    fn test_speciation_preserves_all_genomes() {
        let genomes = vec![
            make_genome(vec![(0, OUTPUT_START, 1)]),
            make_genome(vec![(0, OUTPUT_START, 1), (1, OUTPUT_START + 1, -1)]),
            make_genome(vec![(5, OUTPUT_START + 2, 1)]),
        ];
        let mut species_list = Vec::new();
        let next_id = speciate(&genomes, &mut species_list, 0.5, 0, 0, 1.0, 1.0, 0.4);
        assert!(next_id > 0);
        let total_assigned: usize = species_list.iter().map(|s| s.members.len()).sum();
        assert_eq!(total_assigned, genomes.len());
    }
}
