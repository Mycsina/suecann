use crate::genome::{INPUT_COUNT, OUTPUT_COUNT};
use npyz::NpyFile;
use std::fs::File;
use std::io::BufReader;
use std::path::Path;
use zip::ZipArchive;

pub struct ExpertDataset {
    pub states: Vec<f64>, // flat array of states, shape (N, INPUT_COUNT)
    pub num_states: usize,
    pub soft_intents: Vec<f32>, // flat array of shape (N, OUTPUT_COUNT)
    #[allow(dead_code)]
    pub legal_masks: Vec<u8>,
}

pub fn load_expert_dataset<P: AsRef<Path>>(
    path: P,
) -> Result<ExpertDataset, Box<dyn std::error::Error>> {
    let path_display = path.as_ref().display().to_string();

    if !path.as_ref().exists() {
        println!(
            "  >>> Dataset file {} not found, using mock dataset",
            path_display
        );
        let num_states = 100;
        let states = vec![0.0; num_states * INPUT_COUNT];
        let soft_intents = vec![0.33f32; num_states * OUTPUT_COUNT as usize];
        let legal_masks = vec![0x0F; num_states];
        return Ok(ExpertDataset {
            states,
            num_states,
            soft_intents,
            legal_masks,
        });
    }

    let file = File::open(&path_display)?;
    let reader = BufReader::new(file);
    let mut archive = ZipArchive::new(reader)?;

    // Read states.npy
    let states: Vec<f64> = {
        let mut states_file = archive.by_name("states.npy")?;
        let states_reader = NpyFile::new(&mut states_file)?;
        let states_f32: Vec<f32> = states_reader.into_vec()?;
        states_f32.into_iter().map(|v| v as f64).collect()
    };

    // Read intents.npy (expects 3-intent format: MAX_FORCE, EFFICIENT_WIN, EQUITY_BUILDER)
    let soft_intents: Vec<f32> = {
        let mut intents_file = archive.by_name("intents.npy")?;
        let intents_reader = NpyFile::new(&mut intents_file)?;
        let shape = intents_reader.shape().to_vec();

        if shape.len() == 1 || (shape.len() == 2 && shape[1] == 1) {
            // Legacy u8 hard-intent format — remap to 3-output soft targets
            let intents_u8: Vec<u8> = intents_reader.into_vec()?;
            let mut soft = Vec::with_capacity(intents_u8.len() * OUTPUT_COUNT as usize);
            for &val in &intents_u8 {
                let v = val as usize;
                let mut vec = [0.0f32; 3];
                // 0=MAX_FORCE stays, 1=MIN_FORCE→EFFICIENT_WIN, 2=EFFICIENT_WIN→EFFICIENT_WIN, 3=EQUITY_BUILDER→2
                if v == 0 {
                    vec[0] = 1.0;
                } else if v <= 2 {
                    vec[1] = 1.0;  // MIN_FORCE or EFFICIENT_WIN → new EFFICIENT_WIN
                } else if v == 3 {
                    vec[2] = 1.0;  // EQUITY_BUILDER stays
                }
                soft.extend_from_slice(&vec);
            }
            soft
        } else if shape.len() == 2 && shape[1] == OUTPUT_COUNT as u64 {
            intents_reader.into_vec()?
        } else {
            return Err(format!(
                "Unexpected intents shape: {:?} (expected {} intents per state). \
                 Migrate legacy datasets with: python scripts/migrate_intents.py <file>",
                shape, OUTPUT_COUNT
            ).into());
        }
    };

    // Read legal_masks.npy
    let legal_masks: Vec<u8> = {
        let mut legal_masks_file = archive.by_name("legal_masks.npy")?;
        let legal_masks_reader = NpyFile::new(&mut legal_masks_file)?;
        legal_masks_reader.into_vec()?
    };

    let num_states = legal_masks.len();

    // Validate that the dataset has exactly the expected number of features.
    // No zero-padding — stale datasets must be regenerated.
    let file_input_count = states.len() / num_states;
    if file_input_count != INPUT_COUNT {
        return Err(format!(
            "Dataset '{}' has {} features per state, but code expects {}. \
             Regenerate the dataset with: cargo run --release -- generate-dataset",
            path_display, file_input_count, INPUT_COUNT
        )
        .into());
    }

    assert_eq!(
        states.len(),
        num_states * INPUT_COUNT,
        "States length mismatch"
    );

    Ok(ExpertDataset {
        states,
        num_states,
        soft_intents,
        legal_masks,
    })
}
