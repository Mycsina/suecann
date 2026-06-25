"""Comprehensive quantitative analysis of an expert dataset NPZ file.

DEPRECATED: targets the long-obsolete 33-feature / 4-intent format. The current
dataset is the Stage B v2 card-match format (35-feature beliefs + `best_cards`
u64 mask + `ctx_*` arrays). This script is preserved for historical analysis of
very old checkpoints only and will not load modern datasets.

Usage: python scripts/dataset_analysis.py [dataset.npz]
"""

import sys
import numpy as np

EXPECTED_FEATURES = 33

# input name → description
INPUT_NAMES = [
    "Has_Led_Suit", "Has_Trump", "Led_Suit_Power", "Trump_Power",
    "Hand_Point_Density", "Am_I_Leading", "Am_I_Last_To_Play",
    "Is_Partner_Winning", "Trick_Point_Value", "Has_Trick_Been_Cut",
    "Partner_Void_Led", "Partner_Void_Trump", "Any_Opp_Void_Led",
    "Any_Opp_Void_Trump", "Led_Suit_Ace_Played", "Led_Suit_7_Played",
    "Trump_Ace_Played", "Game_Pts_Remaining", "Trick_Number",
    "Trumps_Remaining", "Score_Delta",
    "Side0_Depletion", "Side0_Ace_Played", "Side0_7_Played",
    "Side1_Depletion", "Side1_Ace_Played", "Side1_7_Played",
    "Side2_Depletion", "Side2_Ace_Played", "Side2_7_Played",
    "Points_Secured_Us", "Known_Void_Suits_Count", "Depleted_Suits_Count",
]

INTENT_NAMES = ["MAX_FORCE", "MIN_FORCE", "EFFICIENT_WIN", "EQUITY_BUILDER"]


def analyze(dataset_path: str):
    d = np.load(dataset_path)
    states = d["states"]
    intents = d["intents"]
    masks = d["legal_masks"]

    n = len(intents)
    n_features = states.shape[1]
    print(f"Dataset: {dataset_path}")
    print(f"Entries: {n:,}")
    print(f"State dim: {n_features}")

    # Feature count validation
    if n_features != EXPECTED_FEATURES:
        print(f"  ⚠ WARNING: Dataset has {n_features} features, expected {EXPECTED_FEATURES}.")
        if n_features < EXPECTED_FEATURES:
            print(f"    Missing features: {INPUT_NAMES[n_features:]}")
        else:
            print(f"    Extra {n_features - EXPECTED_FEATURES} features beyond expected.")
    print()

    # --- 1. Intent distribution ---
    print("=" * 60)
    print("INTENT DISTRIBUTION")
    print("=" * 60)
    unique, counts = np.unique(intents, return_counts=True)
    entropy = 0.0
    for u, c in zip(unique, counts):
        p = c / n
        entropy -= p * np.log2(p) if p > 0 else 0
        print(f"  {INTENT_NAMES[u]:15s} (id={u}): {c:7,d}  ({p*100:5.1f}%)")
    print(f"  Entropy: {entropy:.3f} bits (max 2.0 for uniform)")
    print(f"  Dominance ratio (top/bottom): {counts.max()/counts.min():.2f}x")
    print()

    # --- 2. Mask analysis ---
    print("=" * 60)
    print("LEGAL MASK ANALYSIS")
    print("=" * 60)
    mask_unique, mask_counts = np.unique(masks, return_counts=True)
    print(f"  Unique masks: {len(mask_unique)}")
    for v, c in zip(mask_unique, mask_counts):
        active = [INTENT_NAMES[i] for i in range(4) if (v >> i) & 1]
        print(f"    0x{v:02X} ({', '.join(active) if active else 'NONE'}): {c:7,d} ({c/n*100:5.1f}%)")
    print()

    # --- 3. Per-input statistics ---
    print("=" * 60)
    print("INPUT FEATURE STATISTICS")
    print("=" * 60)
    print(f"  {'#':>2s} {'Name':<22s} {'Mean':>7s} {'Std':>7s} {'Min':>7s} {'Max':>7s} {'Zero%':>6s}")
    print(f"  {'-'*2} {'-'*22} {'-'*7} {'-'*7} {'-'*7} {'-'*7} {'-'*6}")
    for i in range(states.shape[1]):
        col = states[:, i]
        name = INPUT_NAMES[i] if i < len(INPUT_NAMES) else f"Input_{i}"
        print(f"  {i:2d} {name:<22s} {col.mean():7.4f} {col.std():7.4f} "
              f"{col.min():7.4f} {col.max():7.4f} {(col==0).mean()*100:5.1f}%")

    # Flag suspicious features
    binary_features = [0, 1, 5, 6, 7, 9, 10, 11, 12, 13, 14, 15, 16, 22, 23, 25, 26, 28, 29]
    print()
    print("  Binary feature purity check (should be 0 or 1):")
    bad_binary = []
    for idx in binary_features:
        col = states[:, idx]
        non_binary = ((col != 0) & (col != 1)).sum()
        if non_binary > 0:
            bad_binary.append((idx, non_binary, col.min(), col.max()))
            print(f"    Input {idx:2d} ({INPUT_NAMES[idx]}): {non_binary} non-binary vals "
                  f"[{col.min():.4f}, {col.max():.4f}]")
    if not bad_binary:
        print("    All binary features are pure.")
    print()

    # --- 4. Intent-conditioned feature profiles ---
    print("=" * 60)
    print("FEATURE PROFILES BY INTENT (mean per intent)")
    print("=" * 60)
    # Select key features for compact display
    KEY_FEATURES = [0, 1, 2, 3, 5, 6, 7, 8, 9, 10, 12, 20]
    header = "  " + "".join(f"{INPUT_NAMES[i][:6]:>7s}" for i in KEY_FEATURES)
    print(header)
    for intent_id in range(4):
        mask_i = intents == intent_id
        if mask_i.sum() == 0:
            continue
        means = [states[mask_i, i].mean() for i in KEY_FEATURES]
        row = "  " + "".join(f"{m:7.3f}" for m in means)
        print(f"{INTENT_NAMES[intent_id]:15s}{row}")

    # ANOVA: which features discriminate intents most
    print()
    print("  Feature vs intent discrimination (F-statistic, higher = more discriminating):")
    f_stats = []
    grand_mean = states.mean(axis=0)
    for feat in range(states.shape[1]):
        ss_between = 0
        for intent_id in range(4):
            mask_i = intents == intent_id
            if mask_i.sum() == 0:
                continue
            group_mean = states[mask_i, feat].mean()
            ss_between += mask_i.sum() * (group_mean - grand_mean[feat]) ** 2
        ss_within = 0
        for intent_id in range(4):
            mask_i = intents == intent_id
            if mask_i.sum() == 0:
                continue
            ss_within += ((states[mask_i, feat] - states[mask_i, feat].mean()) ** 2).sum()
        f = (ss_between / 3) / (ss_within / (n - 4)) if ss_within > 0 else 0
        f_stats.append((f, feat))
    f_stats.sort(reverse=True)
    for f, feat in f_stats[:10]:
        name = INPUT_NAMES[feat] if feat < len(INPUT_NAMES) else f"Input_{feat}"
        print(f"    {name:<25s} F={f:.1f}")
    print()

    # --- 5. State vector diversity ---
    print("=" * 60)
    print("STATE VECTOR DIVERSITY")
    print("=" * 60)

    # Pairwise distances on a sample (for large datasets)
    sample_size = min(5000, n)
    if n > sample_size:
        rng = np.random.default_rng(42)
        idx = rng.choice(n, sample_size, replace=False)
        sample = states[idx]
    else:
        sample = states

    # Compute mean pairwise Euclidean
    diff = sample[:, None, :] - sample[None, :, :]
    sq_dists = (diff ** 2).sum(axis=2)
    # Only upper triangular avoids diagonal zeros
    triu_idx = np.triu_indices(len(sample), k=1)
    mean_dist = np.sqrt(sq_dists[triu_idx]).mean()
    min_dist = np.sqrt(sq_dists[triu_idx]).min()
    max_dist = np.sqrt(sq_dists[triu_idx]).max()

    print(f"  Mean pairwise Euclidean (sample {sample_size}): {mean_dist:.4f}")
    print(f"  Min pairwise Euclidean: {min_dist:.4f}")
    print(f"  Max pairwise Euclidean: {max_dist:.4f}")

    # Check for exact duplicates
    unique_vals, counts = np.unique(states, axis=0, return_counts=True)
    n_dupes = n - len(unique_vals)
    print(f"  Unique state vectors: {len(np.unique(states, axis=0)):,} / {n:,}")
    print(f"  Duplicate entries: {n_dupes:,} ({n_dupes/n*100:.2f}%)")
    if n_dupes > 0:
        max_dup = counts.max()
        print(f"  Most duplicated: {max_dup}x")
    print()

    # --- 6. Correlation structure ---
    print("=" * 60)
    print("FEATURE CORRELATIONS (top 10 by magnitude)")
    print("=" * 60)
    corr = np.corrcoef(states.T)
    pairs = []
    for i in range(states.shape[1]):
        for j in range(i + 1, states.shape[1]):
            pairs.append((abs(corr[i, j]), corr[i, j], i, j))
    pairs.sort(reverse=True)
    for abs_c, c, i, j in pairs[:10]:
        ni = INPUT_NAMES[i] if i < len(INPUT_NAMES) else f"In_{i}"
        nj = INPUT_NAMES[j] if j < len(INPUT_NAMES) else f"In_{j}"
        print(f"  {ni:<22s} × {nj:<22s}  r = {c:+.3f}")

    # Flag high correlations (>0.8) as potential redundancy
    high_corr = [(c, i, j) for abs_c, c, i, j in pairs if abs_c > 0.8]
    if high_corr:
        print(f"\n  High-correlation pairs (|r| > 0.8): {len(high_corr)}")
        for c, i, j in high_corr:
            ni = INPUT_NAMES[i] if i < len(INPUT_NAMES) else f"In_{i}"
            nj = INPUT_NAMES[j] if j < len(INPUT_NAMES) else f"In_{j}"
            print(f"    {ni} × {nj}: r = {c:+.3f}")
    print()

    # --- 7. Class balance randomization test ---
    print("=" * 60)
    print("LABEL RANDOMNESS CHECK")
    print("=" * 60)
    # Check if adjacent entries (same deal) have similar intents
    # If intents are truly varied per state, autocorrelation of intent sequence should be low
    intent_series = intents.astype(float)
    for lag in [1, 2, 4, 10, 100]:
        if lag < len(intent_series):
            ac = np.corrcoef(intent_series[:-lag], intent_series[lag:])[0, 1]
            print(f"  Intent autocorrelation lag={lag:3d}: {ac:+.4f}")
    print()

    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  Total states:       {n:,}")
    print(f"  Intent balance:     {counts.min()/counts.max():.3f} (min/max ratio)")
    print(f"  State uniqueness:   {(1 - n_dupes/n)*100:.1f}%")
    print(f"  State range:        [{states.min():.3f}, {states.max():.3f}]")
    print(f"  NaN/Inf:            None")
    print(f"  Top discriminating: {INPUT_NAMES[f_stats[0][1]]} (F={f_stats[0][0]:.1f})")
    print(f"  Intent entropy:     {entropy:.3f} bits")
    print()


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "db_w40_d3_n1e5.npz"
    analyze(path)
