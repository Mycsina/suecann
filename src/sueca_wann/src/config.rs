use serde::{Deserialize, Serialize};
use std::fs;
use std::path::Path;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PopulationConfig {
    pub pop_size: usize,
    pub generations: usize,
    pub elitism: usize,
    pub seed_fraction: f64,
    pub pareto_complexity_prob: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvaluationConfig {
    pub n_deals: usize,
    pub curriculum_gens: usize,
    pub sweep_weights: Vec<f64>,
    pub seed: u64,
}

fn default_min_species_size() -> usize {
    3
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

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CurriculumConfig {
    pub phase0_gens: usize,
    pub bulking_gens: usize,
    pub phase0_threshold: f64,
    pub phase1_threshold: f64,
    pub phase2_hof_min: usize,
    pub min_gens_per_phase: usize,
    pub adaptive_window: usize,
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
