pub mod belief;
pub mod engine;
pub mod evaluator;
pub mod heuristic;
pub mod pimc;
pub mod rng;
pub mod search;
pub mod simulator;
pub mod wann;

#[cfg(feature = "python")]
pub mod py_bindings;

#[cfg(feature = "python")]
use pyo3::pymodule;

#[cfg(feature = "python")]
#[pymodule]
fn sueca_solver(
    m: &pyo3::prelude::Bound<'_, pyo3::prelude::PyModule>,
) -> pyo3::prelude::PyResult<()> {
    use pyo3::prelude::*;
    crate::search::init_zobrist();

    // Register PIMC classes/functions
    m.add_class::<py_bindings::pimc::SuecaState>()?;
    m.add_function(wrap_pyfunction!(py_bindings::pimc::solve, m)?)?;
    m.add_function(wrap_pyfunction!(py_bindings::pimc::solve_batch, m)?)?;

    // Register WANN classes/functions
    m.add_class::<py_bindings::wann::PyWannNetwork>()?;
    m.add_function(wrap_pyfunction!(
        py_bindings::wann::evaluate_wann_population,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        py_bindings::wann::evaluate_wann_accuracy,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        py_bindings::wann::batch_compatibility_distances,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(py_bindings::wann::pareto_rank_rust, m)?)?;
    m.add_function(wrap_pyfunction!(py_bindings::wann::load_genome, m)?)?;

    // Register Matchup/SNR functions
    m.add_function(wrap_pyfunction!(py_bindings::matchup::run_matchup_rust, m)?)?;
    m.add_function(wrap_pyfunction!(
        py_bindings::matchup::run_snr_matchup_rust,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        py_bindings::matchup::generate_expert_dataset_rust,
        m
    )?)?;

    Ok(())
}
