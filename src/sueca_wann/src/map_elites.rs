use crate::genome::Genome;
use crate::hall_of_fame::HoFEntry;
use rand::Rng;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MapElitesArchive {
    pub grid: Vec<Vec<Option<HoFEntry>>>,
}

impl MapElitesArchive {
    pub fn new() -> Self {
        Self {
            grid: vec![vec![None; 10]; 10],
        }
    }

    pub fn add(
        &mut self,
        genome: &Genome,
        fitness: f64,
        generation: usize,
        descriptor1: f64,
        descriptor2: f64,
    ) -> bool {
        // Map descriptor1 (ratio) to [0, 9]
        let idx1 = (descriptor1 * 10.0).floor().clamp(0.0, 9.0) as usize;
        // Map descriptor2 (aggression) to [0, 9]
        let idx2 = (descriptor2 * 10.0).floor().clamp(0.0, 9.0) as usize;

        let entry = HoFEntry {
            genome: genome.copy(),
            fitness,
            generation,
        };

        if let Some(existing) = &self.grid[idx1][idx2] {
            if fitness > existing.fitness {
                self.grid[idx1][idx2] = Some(entry);
                true
            } else {
                false
            }
        } else {
            self.grid[idx1][idx2] = Some(entry);
            true
        }
    }

    pub fn sample_random<R: Rng>(&self, rng: &mut R) -> Option<Genome> {
        let mut occupied = Vec::new();
        for row in &self.grid {
            for cell in row {
                if let Some(entry) = cell {
                    occupied.push(&entry.genome);
                }
            }
        }
        if occupied.is_empty() {
            None
        } else {
            let idx = rng.gen_range(0..occupied.len());
            Some(occupied[idx].copy())
        }
    }
}

impl Default for MapElitesArchive {
    fn default() -> Self {
        Self::new()
    }
}
