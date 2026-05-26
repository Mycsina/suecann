#!/usr/bin/env python3
"""Generate a balanced expert dataset of [21-dim Belief State, Target Intent, Legal Mask] triples.

Saves the dataset as a self-documenting .npz file: expert_states_w50_d2.npz
Class balance is strictly enforced: exactly 10,000 samples per intent.
Delegates the heavy simulations, PIMC solves, and state encoding entirely to Rust.
"""

import os
import sys
import numpy as np

# Ensure project root is in python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import sueca_solver


import argparse


def main():
    parser = argparse.ArgumentParser(description="Generate a balanced expert dataset.")
    parser.add_argument(
        "--n-worlds", type=int, default=50, help="Number of worlds for PIMC."
    )
    parser.add_argument(
        "--search-depth", type=int, default=2, help="Search depth for PIMC."
    )
    parser.add_argument(
        "--target-count", type=int, default=10000, help="Number of states per intent."
    )
    parser.add_argument("--seed", type=int, default=12345, help="Random seed.")
    parser.add_argument(
        "--output",
        type=str,
        default="expert_states_w50_d2.npz",
        help="Output filename.",
    )
    args = parser.parse_args()

    n_worlds = args.n_worlds
    search_depth = args.search_depth
    target_count = args.target_count
    seed = args.seed
    out_filename = args.output

    print("Generating balanced expert dataset in Rust...")
    print(
        f"Goal: {target_count} states per intent. Total: {target_count * sueca_solver.OUTPUT_COUNT} states."
    )

    # Call the high-performance Rust implementation
    states_flat, intents, legal_masks = sueca_solver.generate_expert_dataset_rust(
        n_worlds, search_depth, target_count, seed
    )

    # Reshape states flat list/array to (N, INPUT_COUNT)
    states_arr = np.array(states_flat, dtype=np.float32).reshape(
        -1, sueca_solver.INPUT_COUNT
    )
    intents_arr = np.array(intents, dtype=np.uint8)
    masks_arr = np.array(legal_masks, dtype=np.uint8)

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
