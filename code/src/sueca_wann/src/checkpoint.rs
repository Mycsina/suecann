use crate::genome::Genome;
use crate::hall_of_fame::HallOfFame;
use crate::map_elites::MapElitesArchive;
use crate::species::Species;
use serde::{Deserialize, Serialize};
use std::fs::File;
use std::io::{BufReader, BufWriter};
use std::path::Path;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BrainTrainingState {
    pub genomes: Vec<Genome>,
    pub species: Vec<Species>,
    pub hof: HallOfFame,
    pub map_elites: MapElitesArchive,
    pub next_species_id: usize,
    pub global_best_fitness: f64,
    pub global_best_genome: Option<Genome>,
    pub generations_since_improvement: usize,
    pub next_innovation: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingState {
    pub generation: usize,
    pub current_phase: usize,
    pub lead: BrainTrainingState,
    pub follow: BrainTrainingState,
}

impl TrainingState {
    pub fn save_to_file<P: AsRef<Path>>(&self, path: P) -> Result<(), Box<dyn std::error::Error>> {
        let file = File::create(path)?;
        let writer = BufWriter::new(file);
        bincode::serialize_into(writer, self)?;
        Ok(())
    }

    pub fn load_from_file<P: AsRef<Path>>(path: P) -> Result<Self, Box<dyn std::error::Error>> {
        let file = File::open(path)?;
        let reader = BufReader::new(file);
        let state: TrainingState = bincode::deserialize_from(reader)?;
        Ok(state)
    }
}
