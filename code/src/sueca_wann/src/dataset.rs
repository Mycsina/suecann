// ===========================================================================
// Stage B expert dataset loader (2026-06-19 resolver/action-space overhaul)
// ===========================================================================
//
// Loads the card-match dataset produced by `dataset_gen`: beliefs + a teacher-
// best-cards bitmask + the compact `PhiCtx` needed to resolve the WANN's 6
// knobs to a card. Phase-0 fitness is `1` iff the resolver's card is in the
// mask. The old 3-intent soft-label format is no longer supported — regenerate.

use crate::genome::INPUT_COUNT;
use npyz::NpyFile;
use std::fs::File;
use std::io::BufReader;
use std::path::Path;
use sueca_solver::constants::PHI_FEATURE_COUNT;
use sueca_solver::heuristic::PhiCtx;
use zip::ZipArchive;

/// The dataset version this loader accepts. Must match `dataset_gen::DATASET_VERSION`.
const EXPECTED_VERSION: u32 = 2;

pub struct ExpertDataset {
    /// Flat belief states, shape (N, INPUT_COUNT) as f64.
    pub states: Vec<f64>,
    pub num_states: usize,
    /// Teacher-best-cards bitmask per state (ties within one stderr).
    pub best_cards: Vec<u64>,
    // PhiCtx flat arrays (compact context for on-the-fly φ resolution).
    pub ctx_trump: Vec<u8>,
    pub ctx_hand: Vec<u64>,
    pub ctx_trick: Vec<u8>, // len N*4
    pub ctx_trick_len: Vec<u8>,
}

impl ExpertDataset {
    /// Reconstruct the `PhiCtx` for state `i` (used by the Phase-0 fitness kernel).
    #[inline]
    pub fn ctx(&self, i: usize) -> PhiCtx {
        let base = i * 4;
        PhiCtx {
            trump: self.ctx_trump[i],
            hand: self.ctx_hand[i],
            trick_cards: [
                self.ctx_trick[base],
                self.ctx_trick[base + 1],
                self.ctx_trick[base + 2],
                self.ctx_trick[base + 3],
            ],
            trick_len: self.ctx_trick_len[i],
        }
    }

    /// Split indices into lead / follow by the `AmILeading` belief flag
    /// (index `BeliefFeature::AmILeading` = 5). Each brain trains only on its
    /// split, exactly as in the prior pipeline.
    pub fn split_lead_follow(&self) -> (Vec<usize>, Vec<usize>) {
        use crate::constants::BeliefFeature;
        let lead = BeliefFeature::AmILeading as usize;
        let mut li = Vec::new();
        let mut fi = Vec::new();
        for i in 0..self.num_states {
            let is_lead = (self.states[i * INPUT_COUNT + lead] - 1.0).abs() < 1e-9;
            if is_lead {
                li.push(i);
            } else {
                fi.push(i);
            }
        }
        (li, fi)
    }
}

// Number of φ knobs the WANN emits — referenced here so a stale dataset that
// pre-dates the 6-knob layout is caught by the version gate rather than
// silently mis-shaped.
#[allow(dead_code)]
const _PHI_KNOB_COUNT: usize = PHI_FEATURE_COUNT;

pub fn load_expert_dataset<P: AsRef<Path>>(
    path: P,
) -> Result<ExpertDataset, Box<dyn std::error::Error>> {
    let path_display = path.as_ref().display().to_string();

    if !path.as_ref().exists() {
        // Mock dataset so training plumbing still compiles/runs without a real file.
        eprintln!(
            "  >>> Dataset file {} not found, using mock (empty) dataset",
            path_display
        );
        return Ok(empty_mock());
    }

    let file = File::open(&path_display)?;
    let reader = BufReader::new(file);
    let mut archive = ZipArchive::new(reader)?;

    // Version gate — reject stale datasets from the old intent format.
    let version: u32 = {
        let mut vf = archive.by_name("version.npy")?;
        let v: Vec<u32> = NpyFile::new(&mut vf)?.into_vec()?;
        v.into_iter().next().unwrap_or(0)
    };
    if version != EXPECTED_VERSION {
        return Err(format!(
            "Dataset '{}' is version {}, but this build expects version {} (Stage B \
             card-match format). Regenerate with: cargo run --release -- generate-dataset",
            path_display, version, EXPECTED_VERSION
        )
        .into());
    }

    let states: Vec<f64> = {
        let mut sf = archive.by_name("states.npy")?;
        let v: Vec<f32> = NpyFile::new(&mut sf)?.into_vec()?;
        v.into_iter().map(|x| x as f64).collect()
    };
    let best_cards: Vec<u64> = {
        let mut bf = archive.by_name("best_cards.npy")?;
        NpyFile::new(&mut bf)?.into_vec()?
    };
    let ctx_trump: Vec<u8> = {
        let mut tf = archive.by_name("ctx_trump.npy")?;
        NpyFile::new(&mut tf)?.into_vec()?
    };
    let ctx_hand: Vec<u64> = {
        let mut hf = archive.by_name("ctx_hand.npy")?;
        NpyFile::new(&mut hf)?.into_vec()?
    };
    let ctx_trick: Vec<u8> = {
        let mut xf = archive.by_name("ctx_trick.npy")?;
        NpyFile::new(&mut xf)?.into_vec()?
    };
    let ctx_trick_len: Vec<u8> = {
        let mut lf = archive.by_name("ctx_trick_len.npy")?;
        NpyFile::new(&mut lf)?.into_vec()?
    };

    let num_states = best_cards.len();
    let file_input_count = states.len() / num_states;
    if file_input_count != INPUT_COUNT {
        return Err(format!(
            "Dataset '{}' has {} features per state, but code expects {}. \
             Regenerate the dataset with: cargo run --release -- generate-dataset",
            path_display, file_input_count, INPUT_COUNT
        )
        .into());
    }
    assert_eq!(ctx_trick.len(), num_states * 4, "ctx_trick length mismatch");
    assert_eq!(ctx_hand.len(), num_states, "ctx_hand length mismatch");

    Ok(ExpertDataset {
        states,
        num_states,
        best_cards,
        ctx_trump,
        ctx_hand,
        ctx_trick,
        ctx_trick_len,
    })
}

fn empty_mock() -> ExpertDataset {
    ExpertDataset {
        states: Vec::new(),
        num_states: 0,
        best_cards: Vec::new(),
        ctx_trump: Vec::new(),
        ctx_hand: Vec::new(),
        ctx_trick: Vec::new(),
        ctx_trick_len: Vec::new(),
    }
}
