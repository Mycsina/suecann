"""Quantitative analysis of expert dataset NPZ files."""
import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

INTENT_NAMES = {
    0: "MAX_FORCE",
    1: "MIN_FORCE",
    2: "EFFICIENT_WIN",
    3: "EQUITY_BUILDER",
}

FEATURE_NAMES = [
    "Has_Led_Suit", "Has_Trump", "Led_Suit_Power", "Trump_Power",
    "Hand_Point_Density", "Am_I_Leading", "Am_I_Last_To_Play",
    "Is_Partner_Winning", "Trick_Point_Value", "Has_Trick_Been_Cut",
    "Partner_Void_Led", "Partner_Void_Trump", "Any_Opp_Void_Led",
    "Any_Opp_Void_Trump", "Led_Suit_Ace_Played", "Led_Suit_7_Played",
    "Trump_Ace_Played", "Game_Pts_Remaining", "Trick_Number",
    "Trumps_Remaining", "Score_Delta", "Side0_Depletion",
    "Side0_Ace_Played", "Side0_7_Played", "Side1_Depletion",
    "Side1_Ace_Played", "Side1_7_Played", "Side2_Depletion",
    "Side2_Ace_Played", "Side2_7_Played",
]

BINARY_FEATURES = {
    0, 1, 5, 6, 7, 9, 10, 11, 12, 13, 14, 15, 16,
    22, 23, 25, 26, 28, 29,
}


def load_dataset(path: str) -> dict:
    data = np.load(path)
    return {
        "states": data["states"].astype(np.float64),
        "intents": data["intents"],
        "masks": data["legal_masks"],
    }


def analyze(ds: dict, name: str) -> dict:
    states = ds["states"]
    intents = ds["intents"]
    masks = ds["masks"]
    n_states, n_features = states.shape

    report = {}

    # --- Basic counts ---
    report["n_states"] = n_states
    report["n_features"] = n_features

    # --- Intent distribution ---
    unique, counts = np.unique(intents, return_counts=True)
    intent_dist = {int(k): int(v) for k, v in zip(unique, counts)}
    report["intent_dist"] = intent_dist
    report["intent_pct"] = {
        k: round(100 * v / n_states, 2) for k, v in intent_dist.items()
    }

    # --- Mask analysis ---
    unique_masks, mask_counts = np.unique(masks, return_counts=True)
    report["mask_values"] = {
        int(k): int(v) for k, v in zip(unique_masks, mask_counts)
    }
    report["all_masks_15"] = bool(np.all(masks == 15))

    # --- Data quality ---
    report["has_nan"] = bool(np.any(np.isnan(states)))
    report["has_inf"] = bool(np.any(np.isinf(states)))
    report["min_vals"] = states.min(axis=0).tolist()
    report["max_vals"] = states.max(axis=0).tolist()
    report["features_outside_01"] = int(
        np.sum((states < 0) | (states > 1))
    )

    # Check binary features are exactly 0 or 1
    binary_violations = 0
    for feat_idx in BINARY_FEATURES:
        col = states[:, feat_idx]
        violations = np.sum((col != 0) & (col != 1))
        if violations > 0:
            binary_violations += violations
    report["binary_feature_violations"] = int(binary_violations)

    # --- Duplicates ---
    # Round to 4 decimals for fuzzy duplicate detection
    rounded = np.round(states, 4)
    _, unique_idx = np.unique(rounded, axis=0, return_index=True)
    n_duplicates = n_states - len(unique_idx)
    report["n_duplicates"] = int(n_duplicates)
    report["duplicate_pct"] = round(100 * n_duplicates / n_states, 2)

    # --- Per-intent feature means ---
    per_intent_means = {}
    for intent in sorted(unique):
        mask = intents == intent
        per_intent_means[int(intent)] = states[mask].mean(axis=0).tolist()
    report["per_intent_means"] = per_intent_means

    # --- Feature variance (information content) ---
    report["feature_var"] = states.var(axis=0).tolist()
    dead_features = [
        FEATURE_NAMES[i]
        for i in range(n_features)
        if states[:, i].var() == 0
    ]
    report["dead_features"] = dead_features

    return report


def print_report(report: dict, name: str):
    print(f"\n{'='*70}")
    print(f"  Dataset: {name}")
    print(f"{'='*70}")
    print(f"  States:     {report['n_states']:,}")
    print(f"  Features:   {report['n_features']}")

    print(f"\n  Intent Distribution:")
    for intent in sorted(report["intent_dist"]):
        count = report["intent_dist"][intent]
        pct = report["intent_pct"][intent]
        bar = "█" * int(pct / 2)
        print(f"    {INTENT_NAMES[intent]:<16} {count:>8,}  ({pct:5.1f}%)  {bar}")

    print(f"\n  Mask Values:")
    for mask_val, count in sorted(report["mask_values"].items()):
        print(f"    0x{mask_val:02X} ({mask_val:>3})  {count:>8,}")
    if report["all_masks_15"]:
        print(f"    ⚠ ALL masks are 0x0F — likely HARDCODED BUG (fix not compiled)")

    print(f"\n  Data Quality:")
    print(f"    NaN values:         {report['has_nan']}")
    print(f"    Inf values:         {report['has_inf']}")
    print(f"    Values ∉ [0,1]:     {report['features_outside_01']}")
    print(f"    Binary violations:  {report['binary_feature_violations']}")
    print(f"    Dead features:      {report['dead_features'] or 'none'}")
    print(f"    Duplicates:         {report['n_duplicates']:,} ({report['duplicate_pct']}%)")

    # Highlight key features
    print(f"\n  Feature Variance (top 10 most informative):")
    vars_with_idx = sorted(
        enumerate(report["feature_var"]), key=lambda x: -x[1]
    )[:10]
    for idx, var in vars_with_idx:
        marker = " [BIN]" if idx in BINARY_FEATURES else ""
        print(f"    [{idx:2d}] {FEATURE_NAMES[idx]:<28} var={var:.4f}{marker}")

    print(f"\n  Feature Variance (zero-variance features):")
    zero_vars = [
        (i, report["feature_var"][i])
        for i in range(report["n_features"])
        if report["feature_var"][i] == 0
    ]
    if zero_vars:
        for idx, var in zero_vars:
            print(f"    [{idx:2d}] {FEATURE_NAMES[idx]:<28} var={var:.4f}")
    else:
        print("    none")

    # Per-intent distinguishing features
    print(f"\n  Per-Intent Distinguishing Features (z-score deviation from global mean):")
    global_means = np.array([
        report["per_intent_means"][i]
        for i in sorted(report["per_intent_means"])
    ]).mean(axis=0)

    for intent in sorted(report["per_intent_means"]):
        means = np.array(report["per_intent_means"][intent])
        # Find features where this intent deviates most
        diff = means - global_means
        top = np.argsort(-np.abs(diff))[:5]
        print(f"    {INTENT_NAMES[intent]:<16} ", end="")
        parts = []
        for idx in top:
            direction = "↑" if diff[idx] > 0 else "↓"
            parts.append(f"{FEATURE_NAMES[idx]} {direction}{abs(diff[idx]):.3f}")
        print(" | ".join(parts))


def plot_analysis(ds: dict, report: dict, name: str, out_dir: str):
    """Generate diagnostic plots."""
    states = ds["states"]
    intents = ds["intents"]
    n_features = states.shape[1]
    intent_labels = [INTENT_NAMES[i] for i in range(4)]
    colors = ["#e74c3c", "#3498db", "#2ecc71", "#f39c12"]

    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    fig.suptitle(f"Dataset Analysis: {name}", fontsize=14, fontweight="bold")

    # 1. Intent distribution pie
    ax = axes[0, 0]
    counts = [report["intent_dist"].get(i, 0) for i in range(4)]
    wedges, texts, autotexts = ax.pie(
        counts, labels=None, colors=colors, autopct="%1.1f%%",
        startangle=90, pctdistance=0.6,
    )
    ax.set_title("Intent Distribution")

    # 2. Intent distribution bar
    ax = axes[0, 1]
    bars = ax.bar(intent_labels, counts, color=colors, edgecolor="white")
    for bar, count in zip(bars, counts):
        ax.text(
            bar.get_x() + bar.get_width() / 2, bar.get_height() + max(counts) * 0.01,
            f"{count:,}\n({100*count/report['n_states']:.1f}%)",
            ha="center", va="bottom", fontsize=9,
        )
    ax.set_title("Intent Counts")
    ax.set_ylabel("States")
    ax.tick_params(axis="x", rotation=15)

    # 3. Feature variance
    ax = axes[0, 2]
    vars_arr = np.array(report["feature_var"])
    bar_colors = [
        "#95a5a6" if v == 0 else ("#34495e" if i in BINARY_FEATURES else "#3498db")
        for i, v in enumerate(vars_arr)
    ]
    ax.barh(range(n_features), vars_arr, color=bar_colors, edgecolor="white", height=0.7)
    ax.set_yticks(range(n_features))
    ax.set_yticklabels([f"[{i}] {FEATURE_NAMES[i]}" for i in range(n_features)], fontsize=6)
    ax.set_xlabel("Variance")
    ax.set_title("Feature Variance")
    ax.invert_yaxis()

    # 4. Per-intent feature heatmap (deviation from global mean)
    ax = axes[1, 0]
    global_means = np.array([
        report["per_intent_means"][i] for i in range(4)
    ]).mean(axis=0)
    heatmap_data = np.array([
        np.array(report["per_intent_means"][i]) - global_means
        for i in range(4)
    ])
    vmax = max(abs(heatmap_data.min()), abs(heatmap_data.max()))
    im = ax.imshow(heatmap_data, cmap="RdBu_r", aspect="auto", vmin=-vmax, vmax=vmax)
    ax.set_yticks(range(4))
    ax.set_yticklabels(intent_labels)
    ax.set_xticks(range(n_features))
    ax.set_xticklabels([f"{i}" for i in range(n_features)], fontsize=5)
    ax.set_title("Per-Intent Feature Deviation from Mean")
    ax.set_xlabel("Feature Index")
    plt.colorbar(im, ax=ax, shrink=0.8)

    # 5. Pairwise feature correlations (top 12 by variance)
    ax = axes[1, 1]
    top_features = np.argsort(-np.array(report["feature_var"]))[:12]
    top_states = states[:, top_features]
    corr = np.corrcoef(top_states.T)
    im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(range(12))
    ax.set_yticks(range(12))
    ax.set_xticklabels([FEATURE_NAMES[i][:12] for i in top_features], fontsize=5, rotation=45)
    ax.set_yticklabels([FEATURE_NAMES[i][:12] for i in top_features], fontsize=5)
    ax.set_title("Feature Correlations (top 12 by variance)")
    plt.colorbar(im, ax=ax, shrink=0.8)

    # 6. Mask distribution
    ax = axes[1, 2]
    mask_vals = list(report["mask_values"].keys())
    mask_counts = list(report["mask_values"].values())
    ax.bar([f"0x{m:02X}" for m in mask_vals], mask_counts, color="#9b59b6", edgecolor="white")
    ax.set_title("Legal Mask Distribution")
    ax.set_ylabel("Count")
    for i, count in enumerate(mask_counts):
        ax.text(i, count + max(mask_counts) * 0.01, f"{count:,}", ha="center", fontsize=9)

    plt.tight_layout()
    out_path = Path(out_dir) / f"dataset_analysis_{Path(name).stem}.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  Plot saved to: {out_path}")

    # 7. Second figure: per-intent feature distribution for key features
    fig2, axes2 = plt.subplots(2, 4, figsize=(20, 10))
    fig2.suptitle(f"Per-Intent Feature Distributions: {name}", fontsize=14, fontweight="bold")

    key_features = [2, 3, 4, 17, 18, 19, 20, 5]  # Non-binary features of interest
    for ax, feat_idx in zip(axes2.flat, key_features):
        for intent in range(4):
            mask_i = intents == intent
            if mask_i.sum() > 0:
                ax.hist(
                    states[mask_i, feat_idx], bins=30, alpha=0.5,
                    color=colors[intent], label=INTENT_NAMES[intent], density=True,
                )
        ax.set_title(f"[{feat_idx}] {FEATURE_NAMES[feat_idx]}")
        ax.legend(fontsize=6, loc="upper right")

    plt.tight_layout()
    out_path2 = Path(out_dir) / f"dataset_distributions_{Path(name).stem}.png"
    fig2.savefig(out_path2, dpi=150, bbox_inches="tight")
    plt.close(fig2)
    print(f"  Plot saved to: {out_path2}")


def main():
    parser = argparse.ArgumentParser(description="Analyze expert dataset")
    parser.add_argument("npz_path", type=str, help="Path to .npz dataset file")
    parser.add_argument(
        "--out-dir", type=str,
        default="/home/mycsina/Projects/Uni/CAA/project/checkpoints/production/2026-05-28-2",
        help="Output directory for plots",
    )
    args = parser.parse_args()

    path = Path(args.npz_path)
    if not path.exists():
        print(f"ERROR: {path} not found")
        sys.exit(1)

    ds = load_dataset(str(path))
    report = analyze(ds, path.name)
    print_report(report, path.name)
    plot_analysis(ds, report, path.name, args.out_dir)


if __name__ == "__main__":
    main()
