//! Game-delta-gated pruning for trained joint champions.
//!
//! The previous pruning experiment used Phase-0 card-match as a proxy and found
//! it over-pruned: card-match stayed flat while benchmark strength dropped. This
//! module gates removals directly on fixed-deal game-point delta, the metric that
//! matters for champion strength.

use crate::evaluator::{
    evaluate_genome_delta, get_bot_from_type, play_game_sim, rotate_hands, EvaluatorDeal,
    WannBehavior,
};
use crate::genome::{Genome, JsonGenome, JsonGenomeJoint};
use crate::train::generate_deals_rust;
use crate::wann_network::RustWannNetwork;
use rayon::prelude::*;

/// Summary of joint pruning.
#[derive(Debug, Clone, Copy)]
pub struct PruneReport {
    pub lead_conns_before: usize,
    pub lead_conns_after: usize,
    pub follow_conns_before: usize,
    pub follow_conns_after: usize,
    pub lead_nodes_before: usize,
    pub lead_nodes_after: usize,
    pub follow_nodes_before: usize,
    pub follow_nodes_after: usize,
    pub delta_before: f64,
    pub delta_after: f64,
}

fn precompute_baseline_scores(
    deals: &[EvaluatorDeal],
    sweep: &[f64],
    base_seed: u64,
    max_nodes: usize,
) -> Vec<f64> {
    let bot_type = 2; // HeuristicBot, matching Phase-1 delta baseline.
    (0..deals.len() * 4)
        .into_par_iter()
        .map(|idx| {
            let deal_idx = idx / 4;
            let rot = idx % 4;
            let deal = &deals[deal_idx];
            let baseline = get_bot_from_type(bot_type, &[], &[], sweep);
            let opp1 = get_bot_from_type(bot_type, &[], &[], sweep);
            let partner = get_bot_from_type(bot_type, &[], &[], sweep);
            let opp2 = get_bot_from_type(bot_type, &[], &[], sweep);
            let rotated_hands = rotate_hands(&deal.hands, rot);
            let adj_first = rot as u8;
            let evaluation_seed = base_seed ^ ((deal_idx as u64) << 32) ^ (rot as u64);
            let bots = [baseline, opp1, partner, opp2];
            let mut scratchpad = vec![0.0f64; max_nodes * sweep.len()];
            let mut behavior = WannBehavior::default();
            let result = play_game_sim(
                rotated_hands,
                deal.trump,
                adj_first,
                &bots,
                evaluation_seed,
                &mut scratchpad,
                &mut behavior,
            );
            result.team_02_score as f64
        })
        .collect()
}

fn score_joint_delta(
    lead: &Genome,
    follow: &Genome,
    deals: &[EvaluatorDeal],
    baseline_scores: &[f64],
    sweep: &[f64],
    base_seed: u64,
) -> f64 {
    let lead_net: RustWannNetwork = lead.to_rust_wann();
    let follow_net: RustWannNetwork = follow.to_rust_wann();
    let max_nodes = lead_net.num_nodes.max(follow_net.num_nodes);
    let mut scratchpad = vec![0.0f64; max_nodes * sweep.len()];
    let (delta, _) = evaluate_genome_delta(
        &lead_net,
        &follow_net,
        2,
        2,
        2,
        &[],
        &[],
        sweep,
        deals,
        base_seed,
        baseline_scores,
        &mut scratchpad,
    );
    delta
}

fn prune_brain_connections(
    label: &str,
    target: &mut Genome,
    other: &Genome,
    target_is_lead: bool,
    deals: &[EvaluatorDeal],
    baseline_scores: &[f64],
    sweep: &[f64],
    base_seed: u64,
    floor: f64,
) -> usize {
    let innovs: Vec<usize> = target
        .conn_genes
        .iter()
        .filter(|c| c.enabled)
        .map(|c| c.innovation)
        .collect();
    let mut removed = 0usize;
    for inno in innovs {
        let ci = target
            .conn_genes
            .binary_search_by_key(&inno, |c| c.innovation)
            .expect("innovation present");
        if !target.conn_genes[ci].enabled {
            continue;
        }
        target.conn_genes[ci].enabled = false;
        let delta = if target_is_lead {
            score_joint_delta(target, other, deals, baseline_scores, sweep, base_seed)
        } else {
            score_joint_delta(other, target, deals, baseline_scores, sweep, base_seed)
        };
        if delta >= floor {
            removed += 1;
        } else {
            target.conn_genes[ci].enabled = true;
        }
    }
    println!("    {label}: removed {removed} connections this pass");
    removed
}

/// Prune a joint champion using fixed-deal game-point delta as the retention
/// gate. `tolerance` is in average game points: with tolerance 0.25, a removal is
/// kept if the pair stays within 0.25 points/game of the original fixed-yardstick
/// delta.
pub fn prune_joint_game_delta(
    lead: &Genome,
    follow: &Genome,
    deals: usize,
    seed: u64,
    tolerance: f64,
    passes: usize,
    sweep: &[f64],
) -> (Genome, Genome, PruneReport) {
    let mut lead_g = lead.copy();
    let mut follow_g = follow.copy();
    let lead_conns_before = lead_g.num_enabled();
    let follow_conns_before = follow_g.num_enabled();
    let lead_nodes_before = lead_g.node_genes.len();
    let follow_nodes_before = follow_g.node_genes.len();

    let eval_deals = generate_deals_rust(0, deals, seed);
    let max_nodes = lead.to_rust_wann().num_nodes.max(follow.to_rust_wann().num_nodes);
    let baseline_scores = precompute_baseline_scores(&eval_deals, sweep, seed, max_nodes);
    let delta_before = score_joint_delta(&lead_g, &follow_g, &eval_deals, &baseline_scores, sweep, seed);
    let floor = delta_before - tolerance;

    println!(
        "Pruning on fixed game delta: deals={} seed={} sweep={:?} tol={:.3} passes={} | baseline delta {:.4} floor {:.4}",
        deals, seed, sweep, tolerance, passes, delta_before, floor
    );

    for pass in 0..passes {
        println!("  pass {}/{}", pass + 1, passes);
        let removed_lead = prune_brain_connections(
            "Lead",
            &mut lead_g,
            &follow_g,
            true,
            &eval_deals,
            &baseline_scores,
            sweep,
            seed,
            floor,
        );
        let removed_follow = prune_brain_connections(
            "Follow",
            &mut follow_g,
            &lead_g,
            false,
            &eval_deals,
            &baseline_scores,
            sweep,
            seed,
            floor,
        );
        if removed_lead + removed_follow == 0 {
            break;
        }
    }

    lead_g.prune_structural();
    follow_g.prune_structural();
    let delta_after = score_joint_delta(&lead_g, &follow_g, &eval_deals, &baseline_scores, sweep, seed);
    let report = PruneReport {
        lead_conns_before,
        lead_conns_after: lead_g.num_enabled(),
        follow_conns_before,
        follow_conns_after: follow_g.num_enabled(),
        lead_nodes_before,
        lead_nodes_after: lead_g.node_genes.len(),
        follow_nodes_before,
        follow_nodes_after: follow_g.node_genes.len(),
        delta_before,
        delta_after,
    };
    (lead_g, follow_g, report)
}

/// CLI entry point: load a joint (lead+follow) champion, game-delta-prune it,
/// and write `<genome_path>_pruned.json`.
pub fn run_prune(
    genome_path: &str,
    deals: usize,
    seed: u64,
    tolerance: f64,
    passes: usize,
    sweep: &[f64],
) -> Result<(), Box<dyn std::error::Error>> {
    let (lead_opt, follow_opt) = crate::compile_rules::load_genome(genome_path)?;
    let lead = lead_opt.ok_or("joint genome has no lead brain")?;
    let follow = follow_opt.ok_or("joint genome has no follow brain")?;
    let (lead_out, follow_out, rep) =
        prune_joint_game_delta(&lead, &follow, deals, seed, tolerance, passes, sweep);

    println!(
        "  Lead:   conns {} → {} | nodes {} → {}",
        rep.lead_conns_before, rep.lead_conns_after, rep.lead_nodes_before, rep.lead_nodes_after
    );
    println!(
        "  Follow: conns {} → {} | nodes {} → {}",
        rep.follow_conns_before,
        rep.follow_conns_after,
        rep.follow_nodes_before,
        rep.follow_nodes_after
    );
    println!(
        "  delta:  {:.4} → {:.4} (Δ {:+.4})",
        rep.delta_before,
        rep.delta_after,
        rep.delta_after - rep.delta_before
    );

    let joint = JsonGenomeJoint {
        lead: Some(JsonGenome::from_genome(&lead_out)),
        follow: Some(JsonGenome::from_genome(&follow_out)),
    };
    let out_path = format!("{genome_path}_pruned.json");
    std::fs::write(&out_path, serde_json::to_string_pretty(&joint)?)?;
    println!("Wrote pruned genome → {out_path}");
    Ok(())
}
