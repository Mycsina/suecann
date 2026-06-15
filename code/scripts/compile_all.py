#!/usr/bin/env python3
import subprocess
from pathlib import Path

def main():
    checkpoints_dir = Path("checkpoints")
    if not checkpoints_dir.is_dir():
        print("Error: checkpoints/ directory not found.")
        return

    # Find all subdirectories
    runs = sorted([d for d in checkpoints_dir.iterdir() if d.is_dir()])
    print(f"Discovered {len(runs)} checkpoint directories.")

    for run in runs:
        print(f"\nProcessing run: {run.name}")
        
        # 1. Look for the best genome final JSON
        genome_path = run / "genomes" / "best_genome_final.json"
        if not genome_path.exists():
            genome_path = run / "best_genome_final.json"
            
        if not genome_path.exists():
            # Let's try to look for any best_genome_*.json
            candidate_genomes = sorted(list(run.glob("**/best_genome_*.json")))
            if candidate_genomes:
                genome_path = candidate_genomes[-1]
            else:
                print(f"  [Warning] No genome file found for {run.name}. Skipping rule compilation and network plotting.")
                genome_path = None

        if genome_path:
            print(f"  Found genome: {genome_path}")
            # Compile rules (which also exports DOT and renders PNG if dot is available)
            cmd_rules = [
                "./target/release/sueca_wann",
                "compile-rules",
                "--genome", str(genome_path),
                "--output-dir", str(run)
            ]
            print(f"  Running: {' '.join(cmd_rules)}")
            res = subprocess.run(cmd_rules, capture_output=True, text=True)
            if res.returncode == 0:
                print(f"  Successfully compiled rules and topology graphs for {run.name}.")
            else:
                print(f"  [Error] Failed to compile rules: {res.stderr}")

        # 2. Run plot_training.py on training_stats.csv
        stats_path = run / "training_stats.csv"
        if stats_path.exists():
            cmd_plot = [
                "uv", "run", "python", "scripts/plot_training.py",
                "--stats", str(stats_path),
                "--out-dir", str(run)
            ]
            print(f"  Running: {' '.join(cmd_plot)}")
            res = subprocess.run(cmd_plot, capture_output=True, text=True)
            if res.returncode == 0:
                print(f"  Successfully plotted training stats for {run.name}.")
            else:
                print(f"  [Error] Failed to plot training stats: {res.stderr}")
        else:
            print(f"  No training_stats.csv found for {run.name}. Skipping training plots.")

    # 3. Finally, run compare_runs.py to get the progression plot
    print("\nRunning compare_runs.py for all runs...")
    cmd_compare = ["uv", "run", "python", "scripts/compare_runs.py"]
    res = subprocess.run(cmd_compare, capture_output=True, text=True)
    if res.returncode == 0:
        print("Successfully generated checkpoints/run_comparison.png comparison plot.")
        print(res.stdout)
    else:
        print(f"[Error] Failed to run compare_runs.py: {res.stderr}")

if __name__ == "__main__":
    main()
