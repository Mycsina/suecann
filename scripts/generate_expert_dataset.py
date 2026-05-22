#!/usr/bin/env python3
"""Generate a balanced expert dataset of [21-dim Belief State, Target Intent, Legal Mask] triples.

Saves the dataset as a self-documenting .npz file: expert_states_w50_d2.npz
Class balance is strictly enforced: exactly 10,000 samples per intent.
Delegates the heavy simulations, PIMC solves, and state encoding entirely to Rust.
"""

from __future__ import annotations

import os
import sys
import numpy as np

# Ensure project root is in python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import sueca_solver


def main():
    n_worlds = 50
    search_depth = 2
    target_count = 10000
    seed = 12345

    print("Generating balanced expert dataset in Rust...")
    print(f"Goal: {target_count} states per intent. Total: {target_count * 5} states.")

    # Call the high-performance Rust implementation
    states_flat, intents, legal_masks = sueca_solver.generate_expert_dataset_rust(
        n_worlds, search_depth, target_count, seed
    )

    # Reshape states flat list/array to (N, 21)
    states_arr = np.array(states_flat, dtype=np.float32).reshape(-1, 21)
    intents_arr = np.array(intents, dtype=np.uint8)
    masks_arr = np.array(legal_masks, dtype=np.uint8)

    out_filename = "expert_states_w50_d2.npz"
    np.savez_compressed(
        out_filename,
        states=states_arr,
        intents=intents_arr,
        legal_masks=masks_arr,
        n_worlds=n_worlds,
        search_depth=search_depth,
    )

    print(f"\nDataset generation complete! Saved to {out_filename}")
    print(f"Final shape of states: {states_arr.shape}")
    print(f"Final shape of intents: {intents_arr.shape}")
    print(f"Final shape of legal masks: {masks_arr.shape}")
    print(f"Final intent distribution: {np.bincount(intents_arr)}")


if __name__ == "__main__":
    main()
