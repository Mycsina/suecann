use crate::genome::{Genome, JsonGenome};
use rand::seq::SliceRandom;
use rand::Rng;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HoFEntry {
    pub genome: Genome,
    pub fitness: f64,
    pub generation: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HallOfFame {
    pub max_size: usize,
    pub entries: Vec<HoFEntry>,
}

impl HallOfFame {
    pub fn new(max_size: usize) -> Self {
        Self {
            max_size,
            entries: Vec::new(),
        }
    }

    pub fn add(&mut self, genome: &Genome, fitness: f64, generation: usize) {
        let entry = HoFEntry {
            genome: genome.copy(),
            fitness,
            generation,
        };
        self.entries.push(entry);
        // Sort descending by fitness
        self.entries
            .sort_by(|a, b| b.fitness.partial_cmp(&a.fitness).unwrap());
        // Trim to max_size
        if self.entries.len() > self.max_size {
            self.entries.truncate(self.max_size);
        }
    }

    pub fn sample<R: Rng>(&self, rng: &mut R, n: usize) -> Vec<Genome> {
        if self.entries.is_empty() {
            return Vec::new();
        }
        let sample_size = n.min(self.entries.len());
        let mut indices: Vec<usize> = (0..self.entries.len()).collect();
        indices.shuffle(rng);
        indices[0..sample_size]
            .iter()
            .map(|&i| self.entries[i].genome.copy())
            .collect()
    }

    pub fn best(&self) -> Option<&HoFEntry> {
        self.entries.first()
    }

    pub fn clear(&mut self) {
        self.entries.clear();
    }
}

// JSON representation helper for compatibility
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JsonHoFEntry {
    pub fitness: f64,
    pub generation: usize,
    pub genome: JsonGenome,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JsonHallOfFame {
    pub max_size: usize,
    pub entries: Vec<JsonHoFEntry>,
}

impl JsonHallOfFame {
    pub fn from_hof(hof: &HallOfFame) -> Self {
        let entries = hof
            .entries
            .iter()
            .map(|e| JsonHoFEntry {
                fitness: e.fitness,
                generation: e.generation,
                genome: JsonGenome::from_genome(&e.genome),
            })
            .collect();
        Self {
            max_size: hof.max_size,
            entries,
        }
    }

    pub fn to_hof(&self) -> HallOfFame {
        let entries = self
            .entries
            .iter()
            .map(|e| HoFEntry {
                genome: e.genome.to_genome(),
                fitness: e.fitness,
                generation: e.generation,
            })
            .collect();
        HallOfFame {
            max_size: self.max_size,
            entries,
        }
    }
}
