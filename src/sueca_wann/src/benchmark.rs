use crate::evaluator::{EvaluatorDeal, WannBehavior};
use crate::genome::Genome;
use crate::wann_network::RustWannNetwork;
use rayon::prelude::*;

pub struct TournamentConfig {
    pub n_deals: usize,
    pub base_seed: u64,
    pub sweep_weights: Vec<f64>,
}

pub struct BotEntry {
    pub name: String,
    pub bot_type: i32,
    pub genome_path: Option<String>,
}

pub struct MatchupResult {
    pub wins_a: usize,
    pub wins_b: usize,
    pub ties: usize,
    pub total_games: usize,
    pub avg_pts_a: f64,
    pub avg_pts_b: f64,
}

pub fn generate_deals(n_deals: usize, gen: u64, base_seed: u64) -> Vec<EvaluatorDeal> {
    crate::train::generate_deals_rust(gen as usize, n_deals, base_seed)
}

pub fn load_genome(path: &str) -> Genome {
    use crate::genome::JsonGenome;
    use std::io::BufReader;
    let file = std::fs::File::open(path).unwrap_or_else(|e| {
        eprintln!("Error: Cannot open genome file '{}': {}", path, e);
        std::process::exit(1);
    });
    let reader = BufReader::new(file);
    let jg: JsonGenome = serde_json::from_reader(reader).unwrap_or_else(|e| {
        eprintln!("Error: Invalid JSON in genome file '{}': {}", path, e);
        std::process::exit(1);
    });
    jg.to_genome()
}

pub fn run_matchup(
    bot_a_network: Option<&RustWannNetwork>,
    bot_a_type: i32,
    bot_b_network: Option<&RustWannNetwork>,
    bot_b_type: i32,
    deals: &[EvaluatorDeal],
    sweep_weights: &[f64],
    base_seed: u64,
) -> MatchupResult {
    use std::sync::Mutex;

    let num_threads = rayon::current_num_threads();
    let scratchpads: Vec<Mutex<Vec<f64>>> = (0..num_threads)
        .map(|_| Mutex::new(vec![0.0f64; 4096]))
        .collect();

    // Build a combined network list for bot type resolution
    let mut all_nets = Vec::new();
    if let Some(n) = bot_a_network {
        all_nets.push(n.clone());
    }
    if let Some(n) = bot_b_network {
        all_nets.push(n.clone());
    }

    let results: Vec<(f64, f64)> = deals
        .par_iter()
        .map(|deal| {
            let tid = rayon::current_thread_index().unwrap_or(0);
            let mut scratchpad = scratchpads[tid % num_threads].lock().unwrap();
            let mut total_a = 0.0f64;
            let mut total_b = 0.0f64;

            for rot in 0..2 {
                let swapped = rot == 1;
                let rotated_hands = crate::evaluator::rotate_hands(&deal.hands, rot * 2);
                let game_seed = base_seed + deal.seed + (rot as u64) * 10000;

                let (a_bot, b_bot) = if !swapped {
                    (
                        crate::evaluator::get_bot_from_type(bot_a_type, &all_nets, sweep_weights),
                        crate::evaluator::get_bot_from_type(bot_b_type, &all_nets, sweep_weights),
                    )
                } else {
                    (
                        crate::evaluator::get_bot_from_type(bot_b_type, &all_nets, sweep_weights),
                        crate::evaluator::get_bot_from_type(bot_a_type, &all_nets, sweep_weights),
                    )
                };

                let bots: [crate::evaluator::SimulatorBot; 4] =
                    [a_bot.clone(), b_bot.clone(), a_bot.clone(), b_bot.clone()];

                let mut behavior = WannBehavior::default();
                let result = crate::evaluator::play_game_sim(
                    rotated_hands,
                    deal.trump,
                    0,
                    &bots,
                    game_seed,
                    &mut scratchpad,
                    &mut behavior,
                );

                if !swapped {
                    total_a += result.team_02_score as f64;
                    total_b += result.team_13_score as f64;
                } else {
                    total_b += result.team_02_score as f64;
                    total_a += result.team_13_score as f64;
                }
            }

            (total_a, total_b)
        })
        .collect();

    let mut wins_a = 0usize;
    let mut wins_b = 0usize;
    let mut ties = 0usize;
    let mut sum_pts_a = 0.0;
    let mut sum_pts_b = 0.0;

    for (pts_a, pts_b) in &results {
        let a_per_game = *pts_a / 2.0;
        let b_per_game = *pts_b / 2.0;
        sum_pts_a += a_per_game;
        sum_pts_b += b_per_game;
        if a_per_game > 60.0 {
            wins_a += 1;
        } else if b_per_game > 60.0 {
            wins_b += 1;
        } else {
            ties += 1;
        }
    }

    let total_games = deals.len() * 2;
    MatchupResult {
        wins_a,
        wins_b,
        ties,
        total_games,
        avg_pts_a: sum_pts_a / total_games as f64,
        avg_pts_b: sum_pts_b / total_games as f64,
    }
}

pub type TournamentResult = (Vec<Vec<f64>>, Vec<Vec<f64>>, Vec<Vec<f64>>);

pub fn run_tournament(bots: &[BotEntry], config: &TournamentConfig) -> TournamentResult {
    let n = bots.len();
    let mut win_rate_matrix = vec![vec![0.5f64; n]; n];
    let mut ci_matrix = vec![vec![0.0f64; n]; n];
    let mut pts_matrix = vec![vec![60.0f64; n]; n];

    let deals = generate_deals(config.n_deals, 0, config.base_seed * 1000);

    // Load genomes
    let mut networks: Vec<Option<RustWannNetwork>> = Vec::new();
    for bot in bots {
        if let Some(ref path) = bot.genome_path {
            let genome = load_genome(path);
            networks.push(Some(genome.to_rust_wann()));
        } else {
            networks.push(None);
        }
    }

    let total_matchups = n * (n - 1) / 2;
    let mut matchup_idx = 0;

    for i in 0..n {
        for j in (i + 1)..n {
            matchup_idx += 1;
            println!(
                "[{}/{}] {} vs {}...",
                matchup_idx, total_matchups, bots[i].name, bots[j].name
            );

            let result = run_matchup(
                networks[i].as_ref(),
                bots[i].bot_type,
                networks[j].as_ref(),
                bots[j].bot_type,
                &deals,
                &config.sweep_weights,
                config.base_seed,
            );

            let wr_a =
                (result.wins_a as f64 + 0.5 * result.ties as f64) / result.total_games as f64;
            let wr_b =
                (result.wins_b as f64 + 0.5 * result.ties as f64) / result.total_games as f64;
            let ci_a = 1.96 * (wr_a * (1.0 - wr_a) / result.total_games as f64).sqrt();
            let ci_b = 1.96 * (wr_b * (1.0 - wr_b) / result.total_games as f64).sqrt();

            println!(
                "  -> {}: {:.1}% ± {:.1}% | {}: {:.1}% ± {:.1}%",
                bots[i].name,
                wr_a * 100.0,
                ci_a * 100.0,
                bots[j].name,
                wr_b * 100.0,
                ci_b * 100.0
            );
            println!(
                "  -> Card Pts: {} = {:.1} vs {} = {:.1}",
                bots[i].name, result.avg_pts_a, bots[j].name, result.avg_pts_b
            );

            win_rate_matrix[i][j] = wr_a;
            ci_matrix[i][j] = ci_a;
            pts_matrix[i][j] = result.avg_pts_a;
            win_rate_matrix[j][i] = wr_b;
            ci_matrix[j][i] = ci_b;
            pts_matrix[j][i] = result.avg_pts_b;
        }
    }

    (win_rate_matrix, ci_matrix, pts_matrix)
}

pub fn save_tournament_csv(
    win_rates: &[Vec<f64>],
    ci_margins: &[Vec<f64>],
    pts: &[Vec<f64>],
    bot_names: &[String],
    path: &str,
) {
    let mut wtr = csv::Writer::from_path(path).unwrap();
    wtr.write_record([
        "Candidate Bot",
        "Opponent Bot",
        "Win Rate (%)",
        "CI Margin (%)",
        "Avg Card Pts",
    ])
    .unwrap();
    let n = bot_names.len();
    for i in 0..n {
        for j in 0..n {
            wtr.write_record([
                &bot_names[i],
                &bot_names[j],
                &format!("{:.2}", win_rates[i][j] * 100.0),
                &format!("{:.2}", ci_margins[i][j] * 100.0),
                &format!("{:.2}", pts[i][j]),
            ])
            .unwrap();
        }
    }
    wtr.flush().unwrap();
    println!("Saved CSV report to {}", path);
}
