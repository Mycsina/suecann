use serde::{Deserialize, Serialize};
use std::fs;
use std::path::Path;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PopulationConfig {
    pub pop_size: usize,
    pub generations: usize,
    pub elitism: usize,
    pub pareto_complexity_prob: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvaluationConfig {
    pub n_deals: usize,
    pub sweep_weights: Vec<f64>,
    pub seed: u64,
}

fn default_min_species_size() -> usize {
    3
}

fn default_max_species() -> usize {
    20
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpeciesConfig {
    pub compatibility_threshold: f64,
    pub stagnation_limit: usize,
    pub c_excess: f64,
    pub c_disjoint: f64,
    pub c_mismatch: f64,
    #[serde(default = "default_min_species_size")]
    pub min_species_size: usize,
    /// Hard cap on number of active species. When exceeded, the smallest
    /// species are merged into their closest larger neighbour after
    /// speciation completes. Prevents species proliferation from bloating
    /// O(P · Sp · E) compatibility checks.
    #[serde(default = "default_max_species")]
    pub max_species: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MutationConfig {
    pub p_add_node: f64,
    pub p_add_conn: f64,
    pub p_toggle_conn: f64,
    pub p_flip_sign: f64,
    pub p_change_act: f64,
    pub p_change_agg: f64,
    pub p_crossover: f64,
}

fn default_min_gens_per_phase() -> usize {
    20
}
fn default_adaptive_window() -> usize {
    10
}
fn default_phase0_dataset() -> String {
    "expert_states_w50_d2.npz".to_string()
}
fn default_pfs_sample_size() -> usize {
    100
}
fn default_class_balance_target() -> usize {
    30000
}
fn default_soft_balance_min_ratio() -> f64 {
    0.20
}
fn default_use_class_weighting() -> bool {
    true
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CurriculumConfig {
    pub phase0_gens: usize,
    pub bulking_gens: usize,
    #[serde(default = "default_min_gens_per_phase")]
    pub min_gens_per_phase: usize,
    #[serde(default = "default_adaptive_window")]
    pub adaptive_window: usize,
    #[serde(default = "default_phase0_dataset")]
    pub phase0_dataset: String,
    /// Number of expert states sampled for PFS-NEAT mutation validation.
    /// Reduced from the original 1000 — 100 catches catastrophic mutations
    /// while cutting validation cost by 10×.
    #[serde(default = "default_pfs_sample_size")]
    pub pfs_sample_size: usize,
    /// Target total states for class-balanced dataset generation.
    /// Reduced from 100K — capacity-limited WANNs cannot absorb datasets
    /// that large, and quality filtering matters more than volume.
    #[serde(default = "default_class_balance_target")]
    pub class_balance_target: usize,
    /// Minimum fraction of total states any single intent class must have.
    /// 0.20 means no class falls below 20% of total. Soft balance avoids
    /// the brutal marginal cost of perfect 33.3% balance.
    #[serde(default = "default_soft_balance_min_ratio")]
    pub soft_balance_min_ratio: f64,
    /// Whether to apply inverse-frequency class weighting in Phase 0
    /// fitness. Compensates for imbalanced datasets without requiring
    /// perfect balance from the generator.
    #[serde(default = "default_use_class_weighting")]
    pub use_class_weighting: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HallOfFameConfig {
    pub hof_size: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OutputConfig {
    pub checkpoint_dir: String,
    pub stats_file: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Config {
    pub population: PopulationConfig,
    pub evaluation: EvaluationConfig,
    pub species: SpeciesConfig,
    pub mutation: MutationConfig,
    pub curriculum: CurriculumConfig,
    pub hall_of_fame: HallOfFameConfig,
    pub output: OutputConfig,
}

impl Config {
    pub fn load_from_file<P: AsRef<Path>>(path: P) -> Result<Self, Box<dyn std::error::Error>> {
        let contents = fs::read_to_string(path)?;
        let config: Config = toml::from_str(&contents)?;
        Ok(config)
    }
}
