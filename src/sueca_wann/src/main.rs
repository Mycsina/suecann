#![allow(clippy::needless_range_loop)]

use sueca_wann::{benchmark, checkpoint, compile_rules, config, constants, dataset, dataset_gen, evaluator, genome, hall_of_fame, map_elites, mutations, optimize, population, species, train, wann_network};

use clap::{Parser, Subcommand};

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    #[command(subcommand)]
    command: Option<Command>,
}

#[derive(Subcommand, Debug)]
enum Command {
    /// Run the WANN evolution training loop
    Train {
        #[arg(short, long, default_value = "configs/default.toml")]
        config: String,
        #[arg(short, long)]
        resume: bool,
    },
    /// Run round-robin tournament benchmark
    Benchmark {
        #[arg(long, default_value = "200")]
        deals: usize,
        #[arg(long)]
        genome: Option<String>,
        #[arg(long, default_value = "42")]
        seed: u64,
        #[arg(long)]
        output_dir: Option<String>,
        #[arg(long, default_value = "-2.0,-1.0,-0.5,0.5,1.0,2.0", value_delimiter = ',')]
        weights: Vec<f64>,
    },
    /// Compile a genome into human-readable rules
    CompileRules {
        #[arg(long)]
        genome: String,
        #[arg(long, default_value = "1.0")]
        weight: f64,
        #[arg(long, default_value = "checkpoints")]
        output_dir: String,
    },
    /// Generate PIMC expert dataset for Phase 0 pretraining
    GenerateDataset {
        #[arg(long, default_value = "80")]
        n_worlds: usize,
        #[arg(long, default_value = "4")]
        search_depth: u8,
        #[arg(long, default_value = "10000")]
        target_count: usize,
        #[arg(long, default_value = "12345")]
        seed: u64,
        #[arg(long, default_value = "expert_states.npz")]
        output: String,
        /// Minimum fraction of total states each intent class must have (0.0-1.0)
        #[arg(long, default_value = "0.20")]
        soft_balance_min_ratio: f64,
        /// Disable early termination and futility in PIMC for label diffing
        #[arg(long, default_value_t = false)]
        diff_mode: bool,
        /// Use exactly this many worlds per PIMC call (diff mode only)
        #[arg(long)]
        fixed_worlds: Option<usize>,
        /// Resume from a checkpoint file (output path with .checkpoint extension)
        #[arg(long, default_value_t = false)]
        resume: bool,
    },
    /// Optimize independent continuous weights using Differential Evolution
    OptimizeWeights {
        #[arg(long)]
        genome: String,
        #[arg(long, default_value = "200")]
        deals: usize,
        #[arg(long, default_value = "50")]
        generations: usize,
        #[arg(long, default_value = "42")]
        seed: u64,
    },
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();

    match args.command.unwrap_or(Command::Train {
        config: "configs/default.toml".into(),
        resume: false,
    }) {
        Command::Train { config, resume } => {
            let cfg = config::Config::load_from_file(&config)?;
            train::train(cfg, resume)?;
        }
        Command::Benchmark {
            deals,
            genome,
            seed,
            output_dir,
            weights,
        } => {
            run_benchmark(deals, genome, seed, output_dir, weights);
        }
        Command::CompileRules {
            genome,
            weight,
            output_dir,
        } => {
            run_compile_rules(&genome, weight, &output_dir);
        }
        Command::GenerateDataset {
            n_worlds,
            search_depth,
            target_count,
            seed,
            output,
            soft_balance_min_ratio,
            resume,
            diff_mode,
            fixed_worlds,
        } => {
            let config = dataset_gen::DatasetConfig {
                n_worlds,
                search_depth,
                target_total: target_count,
                seed,
                output_path: output,
                soft_balance_min_ratio,
                diff_mode,
                fixed_worlds,
            };
            dataset_gen::generate_dataset(&config, resume);
        }
        Command::OptimizeWeights {
            genome,
            deals,
            generations,
            seed,
        } => {
            optimize::run_weight_optimization(&genome, deals, generations, seed)?;
        }
    }


    Ok(())
}

fn run_benchmark(
    n_deals: usize,
    genome_path: Option<String>,
    seed: u64,
    output_dir: Option<String>,
    sweep_weights: Vec<f64>,
) {
    // Resolve genome path
    let genome_path = genome_path.unwrap_or_else(|| {
        let checkpoints = std::path::Path::new("checkpoints");
        if let Ok(entries) = std::fs::read_dir(checkpoints) {
            let mut dirs: Vec<_> = entries
                .filter_map(|e| e.ok())
                .filter(|e| e.path().is_dir())
                .collect();
            dirs.sort_by_key(|d| d.file_name());
            for d in dirs.iter().rev() {
                for name in &["best_genome_final.json", "genomes/best_genome_final.json"] {
                    let candidate = d.path().join(name);
                    if candidate.exists() {
                        println!("Auto-detected genome: {}", candidate.display());
                        return candidate.to_string_lossy().to_string();
                    }
                }
            }
        }
        eprintln!("Error: No genome specified and none found in checkpoints/");
        std::process::exit(1);
    });

    let output_dir = output_dir.unwrap_or_else(|| {
        std::path::Path::new(&genome_path)
            .parent()
            .unwrap_or(std::path::Path::new("."))
            .to_string_lossy()
            .to_string()
    });
    std::fs::create_dir_all(&output_dir).ok();

    let opt_weights_path = std::path::Path::new(&genome_path)
        .parent()
        .unwrap_or(std::path::Path::new("."))
        .join("optimized_weights.json");

    let mut opt_lead = None;
    let mut opt_follow = None;

    if opt_weights_path.exists() {
        println!("Detected optimized weights file: {}", opt_weights_path.display());
        if let Ok(file) = std::fs::File::open(&opt_weights_path) {
            if let Ok(report) = serde_json::from_reader::<_, sueca_wann::optimize::OptimizedWeightsReport>(file) {
                println!("Loaded optimized weights (best fitness: {:.4})", report.best_fitness);
                opt_lead = Some(report.lead_weights);
                opt_follow = Some(report.follow_weights);
            }
        }
    }

    let mut bots = vec![
        benchmark::BotEntry {
            name: "RandomBot".into(),
            bot_type: 0,
            genome_path: None,
            lead_weights: None,
            follow_weights: None,
        },
        benchmark::BotEntry {
            name: "OldHeuristicBot".into(),
            bot_type: 1,
            genome_path: None,
            lead_weights: None,
            follow_weights: None,
        },
        benchmark::BotEntry {
            name: "EliteHeuristicBot".into(),
            bot_type: 3,
            genome_path: None,
            lead_weights: None,
            follow_weights: None,
        },
        benchmark::BotEntry {
            name: "WANN (Champion)".into(),
            bot_type: 10,
            genome_path: Some(genome_path.clone()),
            lead_weights: None,
            follow_weights: None,
        },
    ];

    if let (Some(lead_w), Some(follow_w)) = (opt_lead, opt_follow) {
        bots.push(benchmark::BotEntry {
            name: "WANN (Optimized)".into(),
            bot_type: 100,
            genome_path: Some(genome_path),
            lead_weights: Some(lead_w),
            follow_weights: Some(follow_w),
        });
    }

    let config = benchmark::TournamentConfig {
        n_deals,
        base_seed: seed,
        sweep_weights,
    };

    println!(
        "Starting Sueca Benchmarking Tournament (N={} deals)...",
        n_deals
    );
    let (win_rates, ci_margins, pts) = benchmark::run_tournament(&bots, &config);

    println!("\n{}", "=".repeat(80));
    println!("{:^80}", "SUECA BOT TOURNAMENT BENCHMARK RESULTS");
    println!("{}", "=".repeat(80));

    let bot_names: Vec<String> = bots.iter().map(|b| b.name.clone()).collect();

    // Header
    print!("{:<25} |", "Candidate / Opponent");
    for name in &bot_names {
        print!(" {:^12} |", &name[..name.len().min(12)]);
    }
    println!();
    println!("{}", "-".repeat(80));

    for i in 0..bot_names.len() {
        print!("{:<25} |", bot_names[i]);
        for j in 0..bot_names.len() {
            if i == j {
                print!(" {:^12} |", "50.0% (Ref)");
            } else {
                print!(
                    " {:5.1}% ±{:4.1}% |",
                    win_rates[i][j] * 100.0,
                    ci_margins[i][j] * 100.0,
                );
            }
        }
        println!();
    }
    println!("{}", "=".repeat(80));

    let csv_path = format!("{}/tournament_report.csv", output_dir);
    benchmark::save_tournament_csv(&win_rates, &ci_margins, &pts, &bot_names, &csv_path);

    println!("\nBenchmark completed successfully.");
}

fn run_compile_rules(genome_path: &str, weight: f64, output_dir: &str) {
    std::fs::create_dir_all(output_dir).ok();
    match compile_rules::load_genome(genome_path) {
        Ok((lead_opt, follow_opt)) => {
            if let Some(lead) = lead_opt {
                println!("Compiling Lead Brain rules...");
                let rules = compile_rules::compile_rules(&lead, weight, output_dir, "lead");
                println!("{}", rules);
            }
            if let Some(follow) = follow_opt {
                println!("\nCompiling Follow Brain rules...");
                let rules = compile_rules::compile_rules(&follow, weight, output_dir, "follow");
                println!("{}", rules);
            }
        }
        Err(e) => {
            eprintln!("Error loading genome: {}", e);
            std::process::exit(1);
        }
    }
}
