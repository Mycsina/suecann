use crate::genome::{INPUT_COUNT, OUTPUT_COUNT};
use npyz::NpyFile;
use std::fs::File;
use std::io::BufReader;
use std::path::Path;
use zip::ZipArchive;

pub struct ExpertDataset {
    pub states: Vec<f64>, // flat array of states, shape (N, INPUT_COUNT)
    pub num_states: usize,
    pub intents: Vec<u8>,
    pub legal_masks: Vec<u8>,
}

pub fn load_expert_dataset<P: AsRef<Path>>(
    path: P,
) -> Result<ExpertDataset, Box<dyn std::error::Error>> {
    if !path.as_ref().exists() {
        println!(
            "  >>> Dataset file {:?} not found, using mock dataset",
            path.as_ref()
        );
        let num_states = 100;
        let states = vec![0.0; num_states * INPUT_COUNT];
        let intents = vec![0; num_states];
        let legal_masks = vec![(1 << OUTPUT_COUNT) - 1; num_states]; // all intents legal
        return Ok(ExpertDataset {
            states,
            num_states,
            intents,
            legal_masks,
        });
    }

    let file = File::open(path)?;
    let reader = BufReader::new(file);
    let mut archive = ZipArchive::new(reader)?;

    // Read states.npy
    let states: Vec<f64> = {
        let mut states_file = archive.by_name("states.npy")?;
        let states_reader = NpyFile::new(&mut states_file)?;
        let states_f32: Vec<f32> = states_reader.into_vec()?;
        states_f32.into_iter().map(|v| v as f64).collect()
    };

    // Read intents.npy
    let intents: Vec<u8> = {
        let mut intents_file = archive.by_name("intents.npy")?;
        let intents_reader = NpyFile::new(&mut intents_file)?;
        intents_reader.into_vec()?
    };

    // Read legal_masks.npy
    let legal_masks: Vec<u8> = {
        let mut legal_masks_file = archive.by_name("legal_masks.npy")?;
        let legal_masks_reader = NpyFile::new(&mut legal_masks_file)?;
        legal_masks_reader.into_vec()?
    };

    let num_states = intents.len();
    assert_eq!(
        states.len(),
        num_states * INPUT_COUNT,
        "States length mismatch"
    );
    assert_eq!(legal_masks.len(), num_states, "Legal masks length mismatch");

    Ok(ExpertDataset {
        states,
        num_states,
        intents,
        legal_masks,
    })
}
