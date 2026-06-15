//! Runtime data snapshots for checkpoint inspection and resume fidelity.
//!
//! At each checkpoint, the training loop collects lightweight snapshots of all
//! complex runtime systems (tabu lists, innovation registries, species state,
//! MAP-Elites grid, population distribution) and dispatches a background thread
//! to write them as human-readable JSON files under `data/`. The training loop
//! never blocks on I/O.

use crate::map_elites::MapElitesCellSnapshot;
use crate::mutations::InnovationRegistryState;
use crate::population::Population;
use crate::species::Species;
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::PathBuf;

// ---------------------------------------------------------------------------
// Snapshot types — all owned, no references, suitable for sending across threads
// ---------------------------------------------------------------------------

/// Complete snapshot of all runtime systems at a checkpoint.
/// All fields are owned plain data — no locks, no references.
/// This is what gets sent to the background writer thread.
#[derive(Debug, Clone, Serialize)]
pub struct RuntimeDataSnapshot {
    pub lead_tabu: Vec<(usize, usize)>,
    pub follow_tabu: Vec<(usize, usize)>,
    pub lead_innov: InnovationRegistryState,
    pub follow_innov: InnovationRegistryState,
    pub lead_species: Vec<SpeciesSummary>,
    pub follow_species: Vec<SpeciesSummary>,
    pub lead_map_elites: Vec<Vec<Option<MapElitesCellSnapshot>>>,
    pub follow_map_elites: Vec<Vec<Option<MapElitesCellSnapshot>>>,
    pub lead_population: Vec<GenomeSnapshot>,
    pub follow_population: Vec<GenomeSnapshot>,
}

/// What we get back when loading from disk.
/// Every field is optional for graceful degradation with old/partial checkpoints.
#[derive(Debug, Clone, Default, Deserialize)]
pub struct LoadedRuntimeData {
    pub lead_tabu: Option<Vec<(usize, usize)>>,
    pub follow_tabu: Option<Vec<(usize, usize)>>,
    pub lead_innov: Option<InnovationRegistryState>,
    pub follow_innov: Option<InnovationRegistryState>,
}

// ---------------------------------------------------------------------------
// Lightweight per-species summary (no full representative genome)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpeciesSummary {
    pub id: usize,
    pub size: usize,
    pub best_fitness: f64,
    pub stagnation: usize,
    pub created_at_gen: usize,
    pub representative_conn_count: usize,
    pub representative_node_count: usize,
    pub representative_enabled_count: usize,
}

// ---------------------------------------------------------------------------
// Per-genome population snapshot
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GenomeSnapshot {
    pub index: usize,
    pub species_id: Option<usize>,
    pub fitness: f64,
    pub complexity: f64,
    pub n_connections: usize,
    pub n_enabled: usize,
    pub n_nodes: usize,
}

// ---------------------------------------------------------------------------
// Extraction helpers — convert from runtime types to snapshot types
// ---------------------------------------------------------------------------

pub fn extract_species_summary(species_list: &[Species]) -> Vec<SpeciesSummary> {
    species_list
        .iter()
        .map(|sp| SpeciesSummary {
            id: sp.id,
            size: sp.members.len(),
            best_fitness: sp.best_fitness,
            stagnation: sp.generations_no_improvement,
            created_at_gen: sp.created_at_gen,
            representative_conn_count: sp.representative.conn_genes.len(),
            representative_node_count: sp.representative.node_genes.len(),
            representative_enabled_count: sp.representative.num_enabled(),
        })
        .collect()
}

pub fn extract_population_snapshot(
    pop: &Population,
    species_list: &[Species],
) -> Vec<GenomeSnapshot> {
    // Build a genome_index → species_id lookup
    let mut species_map = std::collections::HashMap::new();
    for sp in species_list {
        for &member_idx in &sp.members {
            species_map.insert(member_idx, sp.id);
        }
    }

    pop.genomes
        .iter()
        .enumerate()
        .map(|(i, g)| GenomeSnapshot {
            index: i,
            species_id: species_map.get(&i).copied(),
            fitness: pop.fitnesses.get(i).copied().unwrap_or(0.0),
            complexity: g.calculate_complexity(),
            n_connections: g.conn_genes.len(),
            n_enabled: g.num_enabled(),
            n_nodes: g.node_genes.len(),
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Save — non-blocking via background thread
// ---------------------------------------------------------------------------

/// Write all runtime data files under `data_dir`. This function is designed to
/// be called from `std::thread::spawn` so the training loop never blocks on I/O.
///
/// All data is owned — no references back to the training state.
pub fn save_runtime_data(data_dir: PathBuf, snapshot: &RuntimeDataSnapshot) -> Result<(), String> {
    fs::create_dir_all(&data_dir).map_err(|e| format!("mkdir data/: {e}"))?;

    // Tabu lists — arrays of [src, dst] pairs
    write_json(&data_dir.join("tabu_lead.json"), &snapshot.lead_tabu)?;
    write_json(&data_dir.join("tabu_follow.json"), &snapshot.follow_tabu)?;

    // Innovation registries — {pairs: [[src, dst, innovation]], next_innovation}
    write_json(
        &data_dir.join("innovation_registry_lead.json"),
        &snapshot.lead_innov,
    )?;
    write_json(
        &data_dir.join("innovation_registry_follow.json"),
        &snapshot.follow_innov,
    )?;

    // Species summaries
    write_json(&data_dir.join("species_lead.json"), &snapshot.lead_species)?;
    write_json(
        &data_dir.join("species_follow.json"),
        &snapshot.follow_species,
    )?;

    // MAP-Elites grid occupancy
    write_json(
        &data_dir.join("map_elites_lead.json"),
        &snapshot.lead_map_elites,
    )?;
    write_json(
        &data_dir.join("map_elites_follow.json"),
        &snapshot.follow_map_elites,
    )?;

    // Population snapshot
    write_json(
        &data_dir.join("population_snapshot.json"),
        &serde_json::json!({
            "lead": snapshot.lead_population,
            "follow": snapshot.follow_population,
        }),
    )?;

    Ok(())
}

fn write_json<T: Serialize>(path: &PathBuf, value: &T) -> Result<(), String> {
    let json = serde_json::to_string_pretty(value).map_err(|e| format!("serialize: {e}"))?;
    fs::write(path, json).map_err(|e| format!("write {}: {e}", path.display()))?;
    Ok(())
}

// ---------------------------------------------------------------------------
// Load — for resume
// ---------------------------------------------------------------------------

/// Load runtime data from a `data/` directory. Returns defaults (empty Options)
/// if the directory doesn't exist or individual files are missing — this makes
/// old checkpoints without `data/` work transparently.
pub fn load_runtime_data(data_dir: &std::path::Path) -> Result<LoadedRuntimeData, String> {
    if !data_dir.exists() {
        return Ok(LoadedRuntimeData::default());
    }

    let data = LoadedRuntimeData {
        lead_tabu: read_json(data_dir.join("tabu_lead.json")).ok(),
        follow_tabu: read_json(data_dir.join("tabu_follow.json")).ok(),
        lead_innov: read_json(data_dir.join("innovation_registry_lead.json")).ok(),
        follow_innov: read_json(data_dir.join("innovation_registry_follow.json")).ok(),
    };

    Ok(data)
}

fn read_json<T: serde::de::DeserializeOwned>(path: PathBuf) -> Result<T, String> {
    let raw = fs::read_to_string(&path)
        .map_err(|e| format!("read {}: {e}", path.display()))?;
    serde_json::from_str(&raw).map_err(|e| format!("parse {}: {e}", path.display()))
}
