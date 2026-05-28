use crate::genome::INPUT_COUNT;
use npyz::NpyFile;
use std::fs::File;
use std::io::BufReader;
use std::path::Path;
use zip::ZipArchive;

pub struct ExpertDataset {
    pub states: Vec<f64>, // flat array of states, shape (N, INPUT_COUNT)
    pub num_states: usize,
    pub intents: Vec<u8>,
    #[allow(dead_code)]
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
        let legal_masks = vec![0x0F; num_states];
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

    // Detect input dimension from the file: old datasets may have 21 features,
    // current code expects INPUT_COUNT (30). Auto-pad with zeros if needed.
    let file_input_count = states.len() / num_states;
    let states = if file_input_count < INPUT_COUNT {
        println!(
            "  >>> Dataset has {} features per state, padding to {} with zeros.",
            file_input_count, INPUT_COUNT
        );
        let mut padded = Vec::with_capacity(num_states * INPUT_COUNT);
        for i in 0..num_states {
            let start = i * file_input_count;
            let end = start + file_input_count;
            padded.extend_from_slice(&states[start..end]);
            padded.resize(padded.len() + (INPUT_COUNT - file_input_count), 0.0);
        }
        padded
    } else if file_input_count > INPUT_COUNT {
        panic!(
            "Dataset has {} features per state but code expects {}. Regenerate the dataset.",
            file_input_count, INPUT_COUNT
        );
    } else {
        states
    };

    assert_eq!(
        states.len(),
        num_states * INPUT_COUNT,
        "States length mismatch after padding"
    );

    Ok(ExpertDataset {
        states,
        num_states,
        intents,
        legal_masks,
    })
}
