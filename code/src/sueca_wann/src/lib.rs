#![allow(clippy::needless_range_loop)]

pub mod constants;
pub mod genome;
pub mod wann_network;
pub mod evaluator;

#[cfg(feature = "training")]
pub mod benchmark;
#[cfg(feature = "training")]
pub mod checkpoint;
#[cfg(feature = "training")]
pub mod compile_rules;
#[cfg(feature = "training")]
pub mod config;
#[cfg(feature = "training")]
pub mod dataset;
#[cfg(feature = "training")]
pub mod dataset_gen;
#[cfg(feature = "training")]
pub mod hall_of_fame;
#[cfg(feature = "training")]
pub mod map_elites;
#[cfg(feature = "training")]
pub mod mutations;
#[cfg(feature = "training")]
pub mod optimize;
#[cfg(feature = "training")]
pub mod prune;
#[cfg(feature = "training")]
pub mod population;
#[cfg(feature = "training")]
pub mod runtime_data;
#[cfg(feature = "training")]
pub mod species;
#[cfg(feature = "training")]
pub mod train;
