use crate::genome::INPUT_COUNT;
use npyz::NpyFile;
use std::fs::File;
use std::io::BufReader;
use std::path::Path;
use zip::ZipArchive;

pub struct ExpertDataset {
    pub states: Vec<f64>, // flat array of states, shape (N, INPUT_COUNT)
    pub num_states: usize,
    pub soft_intents: Vec<f32>, // flat array of shape (N, 4)
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
        let soft_intents = vec![0.25f32; num_states * 4];
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

    // Read intents.npy (dynamically handle legacy u8 vs new f32 soft targets)
    let soft_intents: Vec<f32> = {
        let mut intents_file = archive.by_name("intents.npy")?;
        let intents_reader = NpyFile::new(&mut intents_file)?;
        let shape = intents_reader.shape().to_vec();

        if shape.len() == 1 || (shape.len() == 2 && shape[1] == 1) {
            let intents_u8: Vec<u8> = intents_reader.into_vec()?;
            let mut soft = Vec::with_capacity(intents_u8.len() * 4);
            for &val in &intents_u8 {
                let mut vec = [0.0f32; 4];
                if (val as usize) < 4 {
                    vec[val as usize] = 1.0;
                }
                soft.extend_from_slice(&vec);
            }
            soft
        } else if shape.len() == 2 && shape[1] == 4 {
            intents_reader.into_vec()?
        } else {
            return Err(format!("Unexpected intents shape: {:?}", shape).into());
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
