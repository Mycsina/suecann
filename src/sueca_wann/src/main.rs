mod checkpoint;
mod config;
mod dataset;
mod genome;
mod hall_of_fame;
mod mutations;
mod population;
mod species;
mod train;

use clap::Parser;

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    #[arg(short, long, default_value = "configs/default.toml")]
    config: String,

    #[arg(short, long)]
    resume: bool,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();
    let config = config::Config::load_from_file(&args.config)?;
    train::train(config, args.resume)?;
    Ok(())
}
