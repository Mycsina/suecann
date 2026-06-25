//! Genome pruning: compact a trained champion by removing connections (and the
//! nodes they leave orphaned) that do not contribute to its card-match accuracy.
//!
//! Two stages:
//!   1. **Behavioural** — for each enabled connection, tentatively disable it
//!      and re-score the brain on its own dataset split (lead/follow). If the
//!      card-match score stays within `tolerance` of the original, the
//!      connection is redundant and is kept disabled.
//!   2. **Structural** — [`Genome::prune_structural`] then deletes every
//!      disabled connection and any hidden node left without both an incoming
//!      and an outgoing edge.
//!
//! Behavioural pruning uses Phase-0 card-match as a fast, stationary retention
//! metric; the resulting genome should still be verified with a game-strength
//! benchmark (the metric the champion was ultimately selected on).

use crate::dataset::ExpertDataset;
use crate::genome::{Genome, JsonGenome, JsonGenomeJoint};
use crate::wann_network::RustWannNetwork;
use sueca_solver::constants::{INPUT_COUNT, OUTPUT_COUNT, OUTPUT_START};
use sueca_solver::heuristic::{outputs_to_knobs, resolve_card_phi_utility_ctx};

/// Summary of one brain's pruning.
#[derive(Debug, Clone, Copy)]
pub struct PruneReport {
    pub conns_before: usize,
    pub conns_after: usize,
    pub nodes_before: usize,
    pub nodes_after: usize,
    pub card_match_before: f64,
    pub card_match_after: f64,
}

/// Card-match accuracy of `net` on a subset of dataset states (a brain's split).
/// Mirrors the inner loop of `train::evaluate_phase0` but restricted to `idx`.
pub fn card_match_on(
    net: &RustWannNetwork,
    dataset: &ExpertDataset,
    idx: &[usize],
    sweep: &[f64],
) -> f64 {
    if idx.is_empty() {
        return 0.0;
    }
    let nw = sweep.len();
    let mut scratch = vec![0.0f64; net.num_nodes * nw];
    let mut correct = 0.0f64;
    let mut inputs = [0.0f64; INPUT_COUNT];
    for &i in idx {
        inputs.copy_from_slice(&dataset.states[i * INPUT_COUNT..(i + 1) * INPUT_COUNT]);
        net.forward_batched(&inputs, sweep, &mut scratch);
        let mut mean_outputs = [0.0f64; OUTPUT_COUNT];
        for k in 0..OUTPUT_COUNT {
            let mut s = 0.0f64;
            for w in 0..nw {
                s += scratch[(OUTPUT_START + k) * nw + w];
            }
            mean_outputs[k] = s / nw as f64;
        }
        let knobs = outputs_to_knobs(&mean_outputs);
        let ctx = dataset.ctx(i);
        let card = resolve_card_phi_utility_ctx(&knobs, &ctx);
        if (dataset.best_cards[i] >> card) & 1 == 1 {
            correct += 1.0;
        }
    }
    correct / idx.len() as f64
}

/// Behaviourally prune one brain against its dataset split, then compact
/// structurally. A connection is kept disabled iff removing it does not drop
/// card-match below `baseline - tol`. Returns the pruned genome + a report.
pub fn prune_brain_behavioral(
    genome: &Genome,
    dataset: &ExpertDataset,
    idx: &[usize],
    sweep: &[f64],
    tol: f64,
    passes: usize,
) -> (Genome, PruneReport) {
    let conns_before = genome.num_enabled();
    let nodes_before = genome.node_genes.len();
    let score_before = card_match_on(&genome.to_rust_wann(), dataset, idx, sweep);
    let floor = score_before - tol;

    let mut g = genome.copy();
    for _ in 0..passes {
        // Snapshot enabled innovations; disabling keeps vector indices valid.
        let innovs: Vec<usize> = g
            .conn_genes
            .iter()
            .filter(|c| c.enabled)
            .map(|c| c.innovation)
            .collect();
        let mut removed_any = false;
        for inno in innovs {
            let ci = g
                .conn_genes
                .binary_search_by_key(&inno, |c| c.innovation)
                .expect("innovation present");
            if !g.conn_genes[ci].enabled {
                continue;
            }
            g.conn_genes[ci].enabled = false;
            let score = card_match_on(&g.to_rust_wann(), dataset, idx, sweep);
            if score >= floor {
                removed_any = true; // keep disabled
            } else {
                g.conn_genes[ci].enabled = true; // it mattered — restore
            }
        }
        if !removed_any {
            break;
        }
    }

    g.prune_structural();
    let conns_after = g.num_enabled();
    let nodes_after = g.node_genes.len();
    let score_after = card_match_on(&g.to_rust_wann(), dataset, idx, sweep);

    (
        g,
        PruneReport {
            conns_before,
            conns_after,
            nodes_before,
            nodes_after,
            card_match_before: score_before,
            card_match_after: score_after,
        },
    )
}

/// CLI entry point: load a joint (lead+follow) champion, prune each brain
/// against its own dataset split, and write `<genome_path>_pruned.json`.
pub fn run_prune(
    genome_path: &str,
    dataset_path: &str,
    tolerance: f64,
    passes: usize,
    sweep: &[f64],
) -> Result<(), Box<dyn std::error::Error>> {
    let (lead_opt, follow_opt) = crate::compile_rules::load_genome(genome_path)?;
    let dataset = crate::dataset::load_expert_dataset(dataset_path)?;
    let (lead_idx, follow_idx) = dataset.split_lead_follow();

    println!(
        "Pruning sweep={:?} tol={tolerance} passes={passes}  | dataset={} ({} lead / {} follow states)",
        sweep,
        dataset_path,
        lead_idx.len(),
        follow_idx.len()
    );

    let mut lead_out: Option<Genome> = None;
    let mut follow_out: Option<Genome> = None;

    for (label, brain, idx, sink) in [
        ("Lead", lead_opt, &lead_idx, &mut lead_out),
        ("Follow", follow_opt, &follow_idx, &mut follow_out),
    ] {
        if let Some(g) = brain {
            let (pruned, rep) = prune_brain_behavioral(&g, &dataset, idx, sweep, tolerance, passes);
            println!(
                "  {label}: conns {} → {} | nodes {} → {} | card-match {:.4} → {:.4} (Δ {:+.4})",
                rep.conns_before,
                rep.conns_after,
                rep.nodes_before,
                rep.nodes_after,
                rep.card_match_before,
                rep.card_match_after,
                rep.card_match_after - rep.card_match_before
            );
            *sink = Some(pruned);
        }
    }

    let joint = JsonGenomeJoint {
        lead: lead_out.as_ref().map(JsonGenome::from_genome),
        follow: follow_out.as_ref().map(JsonGenome::from_genome),
    };
    let out_path = format!("{genome_path}_pruned.json");
    std::fs::write(&out_path, serde_json::to_string_pretty(&joint)?)?;
    println!("Wrote pruned genome → {out_path}");
    Ok(())
}
