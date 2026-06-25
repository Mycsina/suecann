use crate::genome::{ActivationFn, AggregationFn, ConnGene, Genome, NodeType};
use crate::wann_network::RustWannNetwork;
use std::collections::{HashMap, HashSet};
use std::process::Command;
use sueca_solver::constants::{BIAS_ID, FIRST_HIDDEN_ID, INPUT_COUNT, OUTPUT_COUNT, OUTPUT_START};

pub fn load_genome(
    path: &str,
) -> Result<(Option<Genome>, Option<Genome>), Box<dyn std::error::Error>> {
    use crate::genome::{JsonGenome, JsonGenomeJoint};
    let content = std::fs::read_to_string(path)?;
    if let Ok(joint) = serde_json::from_str::<JsonGenomeJoint>(&content) {
        let lead = joint.lead.map(|jg| jg.to_genome());
        let follow = joint.follow.map(|jg| jg.to_genome());
        Ok((lead, follow))
    } else if let Ok(single) = serde_json::from_str::<JsonGenome>(&content) {
        let genome = single.to_genome();
        Ok((Some(genome.copy()), Some(genome)))
    } else {
        Err("Invalid WANN genome format: neither single nor joint genome".into())
    }
}

fn get_reachable_nodes(genome: &Genome) -> HashSet<usize> {
    let mut reachable: HashSet<usize> = (OUTPUT_START..OUTPUT_START + OUTPUT_COUNT).collect();
    let mut queue: Vec<usize> = reachable.iter().cloned().collect();

    let mut incoming: HashMap<usize, Vec<usize>> = HashMap::new();
    for c in &genome.conn_genes {
        if c.enabled {
            incoming.entry(c.dst).or_default().push(c.src);
        }
    }

    while let Some(curr) = queue.pop() {
        if let Some(srcs) = incoming.get(&curr) {
            for &src in srcs {
                if reachable.insert(src) {
                    queue.push(src);
                }
            }
        }
    }
    reachable
}

fn compute_depths(genome: &Genome, reachable: &HashSet<usize>) -> HashMap<usize, usize> {
    let order = genome.topological_order();
    let mut depths = HashMap::new();

    for nid in 0..BIAS_ID {
        depths.insert(nid, 0);
    }
    depths.insert(BIAS_ID, 0);

    for &nid in &order {
        if nid < FIRST_HIDDEN_ID && !(OUTPUT_START..OUTPUT_START + OUTPUT_COUNT).contains(&nid) {
            depths.insert(nid, 0);
            continue;
        }
        if !reachable.contains(&nid) {
            continue;
        }

        let incoming: Vec<&ConnGene> = genome
            .conn_genes
            .iter()
            .filter(|c| c.enabled && c.dst == nid)
            .collect();

        if incoming.is_empty() {
            depths.insert(nid, 1);
        } else {
            let max_d = incoming
                .iter()
                .map(|c| depths.get(&c.src).copied().unwrap_or(0))
                .max()
                .unwrap_or(0);
            depths.insert(nid, 1 + max_d);
        }
    }

    let max_hidden = reachable
        .iter()
        .filter(|&&n| n >= FIRST_HIDDEN_ID)
        .map(|n| depths.get(n).copied().unwrap_or(0))
        .max()
        .unwrap_or(0);

    for nid in OUTPUT_START..OUTPUT_START + OUTPUT_COUNT {
        depths.insert(nid, max_hidden + 1);
    }

    depths
}

/// Compute the provably-constant value (at the given shared weight) of every node
/// whose output does not depend on any *variable* input feature — i.e. it is fed
/// only by BIAS or by empty/other-constant aggregations. Inputs 0..INPUT_COUNT are
/// always variable, BIAS is the sole constant *source* (=1.0), and a node with no
/// enabled incoming edges evaluates to 0.0 (matching the runtime `start == end`
/// branch). The aggregation/activation/clamp math mirrors `wann_network::forward`
/// exactly, so a folded constant is bit-identical to what the network computes.
///
/// These nodes are dead weight in the compiled rules (e.g. `hidden_46 = 0.0`,
/// `hidden_44 = NOT(hidden_46) = 1.0`): folding inlines their value and drops them
/// from the human-readable rule listing.
fn fold_constants(genome: &Genome, weight: f64) -> HashMap<usize, f64> {
    let mut consts: HashMap<usize, f64> = HashMap::new();
    consts.insert(BIAS_ID, 1.0);

    let mut incoming: HashMap<usize, Vec<(usize, i8)>> = HashMap::new();
    for c in &genome.conn_genes {
        if c.enabled {
            incoming.entry(c.dst).or_default().push((c.src, c.sign));
        }
    }

    for nid in genome.topological_order() {
        if nid <= BIAS_ID {
            continue; // inputs are variable; BIAS already inserted
        }
        let ng = match genome.get_node(nid) {
            Some(n) => n,
            None => continue,
        };
        let conns = match incoming.get(&nid) {
            Some(v) if !v.is_empty() => v,
            _ => {
                consts.insert(nid, 0.0); // runtime: no incoming -> 0.0
                continue;
            }
        };
        if !conns.iter().all(|(src, _)| consts.contains_key(src)) {
            continue; // depends transitively on a variable input
        }
        let signal = |src: usize, sign: i8| -> f64 {
            let v = consts[&src];
            let v = if sign == -1 { 1.0 - v } else { v };
            v * weight
        };
        let agg_val = match ng.aggregation_fn {
            AggregationFn::SUM => conns.iter().map(|&(s, sg)| signal(s, sg)).sum(),
            AggregationFn::MIN => conns
                .iter()
                .map(|&(s, sg)| signal(s, sg))
                .fold(f64::INFINITY, f64::min),
            AggregationFn::MAX => conns
                .iter()
                .map(|&(s, sg)| signal(s, sg))
                .fold(f64::NEG_INFINITY, f64::max),
        };
        let activated = match ng.activation_fn {
            ActivationFn::IDENTITY => agg_val,
            ActivationFn::NOT => 1.0 - agg_val.clamp(0.0, 1.0),
            ActivationFn::THRESHOLD => {
                let t = 0.5 * weight;
                let fires = if weight >= 0.0 { agg_val > t } else { agg_val < t };
                if fires {
                    1.0
                } else {
                    0.0
                }
            }
        };
        consts.insert(nid, activated.clamp(0.0, 1.0));
    }
    consts
}

/// A non-constant node is a *pure alias* of its single source when it has exactly
/// one enabled incoming edge and IDENTITY activation: a single-operand aggregation
/// is degenerate (`SUM(x)=MIN(x)=MAX(x)=x·W`) and IDENTITY passes it through, so at
/// **W=1** the node equals its source (or `1−source` if the edge sign is −1). Such
/// nodes are renames that inflate apparent depth/gate-count; we collapse them.
///
/// Returns `node -> (ultimate_non_alias_source, negated)`. The negation parity is
/// accumulated along the chain (NOT∘NOT cancels). Resolution runs in topological
/// order, so an alias of an alias is fully flattened. Only valid at W=1, so the map
/// is empty for any other weight (callers then keep the explicit alias steps).
fn compute_aliases(genome: &Genome, consts: &HashMap<usize, f64>, weight: f64) -> HashMap<usize, (usize, bool)> {
    let mut aliases: HashMap<usize, (usize, bool)> = HashMap::new();
    if (weight - 1.0).abs() > 1e-9 {
        return aliases;
    }
    let mut incoming: HashMap<usize, Vec<(usize, i8)>> = HashMap::new();
    for c in &genome.conn_genes {
        if c.enabled {
            incoming.entry(c.dst).or_default().push((c.src, c.sign));
        }
    }
    for nid in genome.topological_order() {
        if nid < FIRST_HIDDEN_ID || consts.contains_key(&nid) {
            continue;
        }
        let ng = match genome.get_node(nid) {
            Some(n) => n,
            None => continue,
        };
        if ng.activation_fn != ActivationFn::IDENTITY {
            continue;
        }
        let conns = match incoming.get(&nid) {
            Some(v) if v.len() == 1 => v,
            _ => continue,
        };
        let (src, sign) = conns[0];
        if consts.contains_key(&src) {
            continue; // a const single-source IDENTITY node would already be a const itself
        }
        let neg = sign == -1;
        let resolved = match aliases.get(&src) {
            Some(&(rs, rneg)) => (rs, neg ^ rneg),
            None => (src, neg),
        };
        aliases.insert(nid, resolved);
    }
    aliases
}

/// Resolve a connection source through the alias map. Returns the rendered source
/// string (feature/hidden/constant) with the effective sign already applied.
fn render_src(
    src: usize,
    sign: i8,
    consts: &HashMap<usize, f64>,
    aliases: &HashMap<usize, (usize, bool)>,
) -> String {
    let base_neg = sign == -1;
    // Folded constant source -> inline its numeric value (sign applied).
    if let Some(&v) = consts.get(&src) {
        let v = if base_neg { 1.0 - v } else { v };
        return fmt_const(v);
    }
    let (target, neg) = match aliases.get(&src) {
        Some(&(t, n)) => (t, base_neg ^ n),
        None => (src, base_neg),
    };
    let feature_names = crate::constants::FEATURE_NAMES;
    let s = if target == BIAS_ID {
        "1.0".to_string()
    } else if target < INPUT_COUNT {
        feature_names[target].to_string()
    } else {
        format!("hidden_{}", target)
    };
    if neg {
        format!("NOT({})", s)
    } else {
        s
    }
}

/// Pretty-print a folded constant: `0`, `1`, or up to 4 decimals without trailing zeros.
fn fmt_const(v: f64) -> String {
    if (v - v.round()).abs() < 1e-9 {
        format!("{}", v.round() as i64)
    } else {
        let s = format!("{:.4}", v);
        s.trim_end_matches('0').trim_end_matches('.').to_string()
    }
}

pub fn export_topology(genome: &Genome, output_dir: &str, _wann: &RustWannNetwork, prefix: &str) {
    let reachable = get_reachable_nodes(genome);
    let depths = compute_depths(genome, &reachable);
    let max_depth = depths.values().max().copied().unwrap_or(1);

    let mut dot = String::new();
    dot.push_str("// Evolved WANN Topology\n");
    dot.push_str("digraph {\n");
    dot.push_str("    nodesep=0.4 overlap=false rankdir=LR ranksep=1.2 splines=true\n");

    let feature_names = crate::constants::FEATURE_NAMES;
    let output_names = crate::constants::OUTPUT_NAMES;

    // Nodes
    for ng in &genome.node_genes {
        if !reachable.contains(&ng.id) {
            continue;
        }
        match ng.node_type {
            NodeType::INPUT => {
                let name = if ng.id < INPUT_COUNT {
                    feature_names[ng.id]
                } else {
                    "?"
                };
                dot.push_str(&format!(
                    "    {} [label=\"{}\\n(I{})\" color=\"#3182bd\" fillcolor=\"#deebf7\" \
                     fontname=Helvetica shape=ellipse style=filled]\n",
                    ng.id, name, ng.id
                ));
            }
            NodeType::BIAS => {
                dot.push_str(&format!(
                    "    {} [label=\"BIAS\\n({})\" color=\"#636363\" fillcolor=\"#f0f0f0\" \
                     fontname=Helvetica shape=ellipse style=filled]\n",
                    ng.id, BIAS_ID
                ));
            }
            NodeType::OUTPUT => {
                let out_idx = ng.id - OUTPUT_START;
                let name = if out_idx < OUTPUT_COUNT {
                    output_names[out_idx]
                } else {
                    "?"
                };
                dot.push_str(&format!(
                    "    {} [label=\"{}\\n(O{})\" color=\"#de2d26\" fillcolor=\"#fee0d2\" \
                     fontname=\"Helvetica-Bold\" shape=box style=\"filled,bold\"]\n",
                    ng.id, name, ng.id
                ));
            }
            NodeType::HIDDEN => {
                let agg_name = format!("{:?}", ng.aggregation_fn);
                let act_name = format!("{:?}", ng.activation_fn);
                dot.push_str(&format!(
                    "    {} [label=\"H{}\\n{}\\n{}\" color=\"#e6550d\" fillcolor=\"#fee6ce\" \
                     fontname=Helvetica shape=box style=filled]\n",
                    ng.id, ng.id, agg_name, act_name
                ));
            }
        }
    }

    // Edges. Skip any edge touching an unreachable node: such endpoints are
    // never declared in the node loop above, so graphviz would otherwise
    // auto-create a bare numbered circle for them (the orphan-node clutter).
    // Disabled edges between two reachable nodes are kept (dotted), as they
    // are genuinely part of the genome.
    for c in &genome.conn_genes {
        if !reachable.contains(&c.src) || !reachable.contains(&c.dst) {
            continue;
        }
        if !c.enabled {
            dot.push_str(&format!(
                "    {} -> {} [arrowsize=0.5 color=\"#d9d9d9\" style=dotted]\n",
                c.src, c.dst
            ));
        } else if c.sign == 1 {
            dot.push_str(&format!(
                "    {} -> {} [arrowsize=0.8 color=\"#31a354\" penwidth=2.0 style=solid]\n",
                c.src, c.dst
            ));
        } else {
            dot.push_str(&format!(
                "    {} -> {} [arrowsize=0.8 color=\"#de2d26\" penwidth=2.0 style=dashed]\n",
                c.src, c.dst
            ));
        }
    }

    // Rank groupings by depth
    for d in 0..=max_depth {
        dot.push_str("    {\n        rank=same\n");
        for (&nid, &depth) in &depths {
            if depth == d && reachable.contains(&nid) {
                dot.push_str(&format!("        {}\n", nid));
            }
        }
        dot.push_str("    }\n");
    }

    dot.push_str("}\n");

    // Write .dot file
    let dot_name = if prefix.is_empty() {
        "topology_graph.dot".to_string()
    } else {
        format!("topology_graph_{}.dot", prefix)
    };
    let png_name = if prefix.is_empty() {
        "topology_graph.png".to_string()
    } else {
        format!("topology_graph_{}.png", prefix)
    };
    let dot_path = format!("{}/{}", output_dir, dot_name);
    std::fs::write(&dot_path, &dot).unwrap();
    println!("Saved raw dot source to {}", dot_path);

    // Try rendering with Graphviz
    let png_path = format!("{}/{}", output_dir, png_name);
    match Command::new("dot")
        .args(["-Kdot", "-Tpng", "-o", &png_path, &dot_path])
        .output()
    {
        Ok(out) if out.status.success() => {
            println!("Rendered network topology to {}", png_path);
        }
        _ => {
            println!("Note: Graphviz 'dot' not found. The .dot file can be viewed at viz-js.com");
        }
    }
}

fn format_node_rhs(
    genome: &Genome,
    nid: usize,
    weight: f64,
    consts: &HashMap<usize, f64>,
    aliases: &HashMap<usize, (usize, bool)>,
    _reachable: &HashSet<usize>,
    _indent: &str,
) -> String {
    // A constant node folds to its numeric value (e.g. `EFFICIENT_WIN = 0`).
    if let Some(&v) = consts.get(&nid) {
        return fmt_const(v);
    }

    let ng = match genome.get_node(nid) {
        Some(n) => n,
        None => return "0.0".to_string(),
    };

    let conns: Vec<&ConnGene> = genome
        .conn_genes
        .iter()
        .filter(|c| c.enabled && c.dst == nid)
        .collect();

    if conns.is_empty() {
        return "0.0".to_string();
    }

    let signals: Vec<String> = conns
        .iter()
        .map(|c| render_src(c.src, c.sign, consts, aliases))
        .collect();

    // A single-operand aggregation is degenerate: SUM(x)=MIN(x)=MAX(x)=x, so drop
    // the AND()/OR() wrapper and the parens for one operand.
    let agg_str = if signals.len() == 1 {
        signals[0].clone()
    } else {
        match ng.aggregation_fn {
            AggregationFn::SUM => format!("({})", signals.join(" + ")),
            AggregationFn::MIN => format!("AND({})", signals.join(", ")),
            AggregationFn::MAX => format!("OR({})", signals.join(", ")),
        }
    };

    match ng.activation_fn {
        ActivationFn::IDENTITY => agg_str,
        ActivationFn::NOT => format!("NOT({})", agg_str),
        ActivationFn::THRESHOLD => {
            let threshold = 0.5 * weight;
            if weight >= 0.0 {
                format!("THRESHOLD({} > {:.4})", agg_str, threshold)
            } else {
                format!("THRESHOLD({} < {:.4})", agg_str, threshold)
            }
        }
    }
}

pub fn compile_rules(genome: &Genome, weight: f64, output_dir: &str, prefix: &str) -> String {
    let reachable = get_reachable_nodes(genome);
    let consts = fold_constants(genome, weight);
    let aliases = compute_aliases(genome, &consts, weight);
    let order = genome.topological_order();
    let feature_names = crate::constants::FEATURE_NAMES;
    let output_names = crate::constants::OUTPUT_NAMES;

    let active_nodes: Vec<usize> = order
        .iter()
        .cloned()
        .filter(|nid| reachable.contains(nid) && *nid >= OUTPUT_START)
        .collect();

    let wann = genome.to_rust_wann();

    let mut out = String::new();
    out.push_str(&format!(
        "=== Evolved Sueca WANN Strategy Rules (W = {}) ===\n\n",
        weight
    ));

    out.push_str(&format!(
        "Loaded genome with {} nodes and {} connections.\n",
        genome.node_genes.len(),
        genome.conn_genes.len()
    ));
    out.push_str(&format!(
        "Topology: {} total nodes, {} enabled connections ({:?})\n\n",
        wann.num_nodes,
        wann.incoming_srcs.len(),
        genome.conn_genes.iter().filter(|c| c.enabled).count()
    ));

    out.push_str("Active Inputs Referenced:\n");
    let refs: Vec<usize> = reachable
        .iter()
        .cloned()
        .filter(|&n| n <= BIAS_ID)
        .collect();
    let mut sorted_refs = refs.clone();
    sorted_refs.sort();
    for &r in &sorted_refs {
        if r == BIAS_ID {
            out.push_str(&format!("  BIAS (Node {}) = 1.0\n", BIAS_ID));
        } else {
            out.push_str(&format!("  {} (Node {})\n", feature_names[r], r));
        }
    }
    out.push('\n');

    // Live hidden gates = reachable hidden nodes that are neither folded constants
    // nor pure aliases (both are inlined away).
    let hidden_nodes: Vec<usize> = active_nodes
        .iter()
        .cloned()
        .filter(|&n| n >= FIRST_HIDDEN_ID && !consts.contains_key(&n) && !aliases.contains_key(&n))
        .collect();

    if !hidden_nodes.is_empty() {
        out.push_str("Active Hidden Logic Steps:\n");
        for &nid in &hidden_nodes {
            let rhs = format_node_rhs(genome, nid, weight, &consts, &aliases, &reachable, "  ");
            out.push_str(&format!("  hidden_{} = {}\n", nid, rhs));
        }
        out.push('\n');
    }

    out.push_str("Decision Rules for φ-Utility Knobs (WANN outputs, [-1,1] after [0,1]→2x-1 remap):\n");
    for out_idx in 0..OUTPUT_COUNT {
        let nid = OUTPUT_START + out_idx;
        let name = if out_idx < OUTPUT_COUNT {
            output_names[out_idx]
        } else {
            "?"
        };
        if reachable.contains(&nid) {
            let rhs = format_node_rhs(genome, nid, weight, &consts, &aliases, &reachable, "  ");
            out.push_str(&format!("  {} = {}\n", name, rhs));
        } else {
            out.push_str(&format!("  {} = 0.0 (Inactive)\n", name));
        }
    }

    // ── Rule-complexity metrics (on the collapsed rule DAG) ────────────────────
    // Measured over the *printed* structure after constant-folding AND alias
    // inlining — i.e. what a human actually reads. A folded-constant source
    // contributes nothing (it's a literal number); every other edge into a
    // printed node is a "live connection", and one whose resolved source is a
    // variable input feature is a "literal" (a feature mention in the rules).
    let hidden_set: HashSet<usize> = hidden_nodes.iter().cloned().collect();
    let printed_outputs: Vec<usize> = (OUTPUT_START..OUTPUT_START + OUTPUT_COUNT)
        .filter(|n| reachable.contains(n))
        .collect();
    // Resolve a source through const/alias maps: None = folded constant (drops out),
    // Some(target) = a variable input feature or a live gate.
    let resolve = |src: usize| -> Option<usize> {
        if consts.contains_key(&src) {
            None
        } else if let Some(&(t, _)) = aliases.get(&src) {
            Some(t)
        } else {
            Some(src)
        }
    };
    let folded_hidden = active_nodes
        .iter()
        .filter(|&&n| n >= FIRST_HIDDEN_ID && consts.contains_key(&n))
        .count();
    let aliased_hidden = active_nodes
        .iter()
        .filter(|&&n| n >= FIRST_HIDDEN_ID && aliases.contains_key(&n))
        .count();
    let enabled_conns = genome.conn_genes.iter().filter(|c| c.enabled).count();
    let printed: HashSet<usize> = hidden_set.iter().cloned().chain(printed_outputs.iter().cloned()).collect();
    let mut live_conns = 0usize;
    let mut total_literals = 0usize;
    for c in &genome.conn_genes {
        if !c.enabled || !printed.contains(&c.dst) {
            continue;
        }
        if let Some(t) = resolve(c.src) {
            live_conns += 1;
            if t < INPUT_COUNT {
                total_literals += 1;
            }
        }
    }
    // Longest path through live gates only (inputs/constants/aliases are depth-0 leaves).
    let mut depth: HashMap<usize, usize> = HashMap::new();
    let mut max_depth = 0usize;
    for &nid in &order {
        if !printed.contains(&nid) {
            continue;
        }
        let d = genome
            .conn_genes
            .iter()
            .filter(|c| c.enabled && c.dst == nid)
            .filter_map(|c| resolve(c.src))
            .filter(|t| hidden_set.contains(t))
            .map(|t| depth.get(&t).copied().unwrap_or(0))
            .max()
            .unwrap_or(0)
            + 1;
        depth.insert(nid, d);
        max_depth = max_depth.max(d);
    }
    out.push('\n');
    out.push_str("Rule Complexity (after constant-folding + alias inlining):\n");
    out.push_str(&format!(
        "  Active hidden gates:  {}  (folded constants: {}, inlined aliases: {})\n",
        hidden_nodes.len(),
        folded_hidden,
        aliased_hidden
    ));
    out.push_str(&format!(
        "  Live connections:     {}  (of {} enabled)\n",
        live_conns, enabled_conns
    ));
    out.push_str(&format!("  Max logic depth:      {}\n", max_depth));
    out.push_str(&format!("  Total input literals: {}\n", total_literals));

    let rules_name = if prefix.is_empty() {
        "compiled_rules.txt".to_string()
    } else {
        format!("compiled_rules_{}.txt", prefix)
    };
    let rules_path = format!("{}/{}", output_dir, rules_name);
    std::fs::write(&rules_path, &out).unwrap();
    println!("Saved text rules to {}", rules_path);

    // Generate topology visualization
    export_topology(genome, output_dir, &wann, prefix);

    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::genome::{ActivationFn, AggregationFn, ConnGene, Genome, NodeGene, NodeType};
    use rand::{Rng, SeedableRng};
    use rand_pcg::Pcg64;

    /// Build a base genome (35 inputs + bias + 3 outputs) augmented with `hidden`
    /// nodes and `conns` connections.
    fn build(hidden: Vec<NodeGene>, conns: Vec<ConnGene>) -> Genome {
        let mut g = Genome::initial();
        let mut nodes = g.node_genes.clone();
        nodes.extend(hidden);
        let next = conns.iter().map(|c| c.innovation).max().unwrap_or(0) + 1;
        g = Genome::new(Some(nodes), Some(conns), next);
        g
    }

    fn hid(id: usize, act: ActivationFn, agg: AggregationFn) -> NodeGene {
        NodeGene::make(id, NodeType::HIDDEN, act, agg)
    }

    #[test]
    fn fold_detects_bias_only_constants_not_input_dependent_nodes() {
        // h0: fed only by BIAS -> constant 1.0
        // h1: fed by input feature 1 -> NOT constant
        let h0 = FIRST_HIDDEN_ID;
        let h1 = FIRST_HIDDEN_ID + 1;
        let hidden = vec![
            hid(h0, ActivationFn::IDENTITY, AggregationFn::SUM),
            hid(h1, ActivationFn::THRESHOLD, AggregationFn::SUM),
        ];
        let conns = vec![
            ConnGene::make(0, BIAS_ID, h0, 1, true),
            ConnGene::make(1, 1, h1, 1, true),
        ];
        let g = build(hidden, conns);
        let consts = fold_constants(&g, 1.0);
        assert_eq!(consts.get(&BIAS_ID), Some(&1.0));
        assert_eq!(consts.get(&h0), Some(&1.0)); // IDENTITY(BIAS) = 1.0
        assert!(!consts.contains_key(&h1)); // depends on a variable input
    }

    #[test]
    fn fold_handles_no_input_node_as_zero() {
        // h0 has no enabled incoming -> runtime yields 0.0
        let h0 = FIRST_HIDDEN_ID;
        let hidden = vec![hid(h0, ActivationFn::IDENTITY, AggregationFn::SUM)];
        let g = build(hidden, vec![]);
        let consts = fold_constants(&g, 1.0);
        assert_eq!(consts.get(&h0), Some(&0.0));
    }

    #[test]
    fn aliases_flatten_chains_with_negation_parity() {
        // h0 = IDENTITY(input0)        -> alias (input0, +)
        // h1 = IDENTITY(NOT h0)        -> alias (input0, negated)
        // h2 = THRESHOLD(in1, in2)     -> real gate (not alias)
        let h0 = FIRST_HIDDEN_ID;
        let h1 = FIRST_HIDDEN_ID + 1;
        let h2 = FIRST_HIDDEN_ID + 2;
        let hidden = vec![
            hid(h0, ActivationFn::IDENTITY, AggregationFn::SUM),
            hid(h1, ActivationFn::IDENTITY, AggregationFn::MAX),
            hid(h2, ActivationFn::THRESHOLD, AggregationFn::SUM),
        ];
        let conns = vec![
            ConnGene::make(0, 0, h0, 1, true),
            ConnGene::make(1, h0, h1, -1, true),
            ConnGene::make(2, 1, h2, 1, true),
            ConnGene::make(3, 2, h2, 1, true),
        ];
        let g = build(hidden, conns);
        let consts = fold_constants(&g, 1.0);
        let aliases = compute_aliases(&g, &consts, 1.0);
        assert_eq!(aliases.get(&h0), Some(&(0usize, false)));
        assert_eq!(aliases.get(&h1), Some(&(0usize, true))); // parity flipped
        assert!(!aliases.contains_key(&h2)); // 2 inputs -> not an alias
    }

    #[test]
    fn aliases_disabled_off_unit_weight() {
        // alias equivalence only holds at W=1; map must be empty otherwise.
        let h0 = FIRST_HIDDEN_ID;
        let hidden = vec![hid(h0, ActivationFn::IDENTITY, AggregationFn::SUM)];
        let conns = vec![ConnGene::make(0, 0, h0, 1, true)];
        let g = build(hidden, conns);
        let consts = fold_constants(&g, 2.0);
        assert!(compute_aliases(&g, &consts, 2.0).is_empty());
    }

    /// Empirical soundness on the real champion: every folded constant must be
    /// invariant across random belief states, and every alias node must equal its
    /// resolved source (with parity) — proving the rule transforms are exact.
    /// `#[ignore]` because it depends on the checked-out champion genome file.
    #[test]
    #[ignore]
    fn champion_fold_and_alias_are_behavior_preserving() {
        let path = "checkpoints/production/2026-06-14-2/genomes/best_genome_final.json";
        if !std::path::Path::new(path).exists() {
            eprintln!("skipping: champion genome not found at {path}");
            return;
        }
        let (lead, follow) = load_genome(path).expect("load champion");
        for (label, genome) in [("lead", lead), ("follow", follow)] {
            let genome = genome.expect("brain present");
            let consts = fold_constants(&genome, 1.0);
            let aliases = compute_aliases(&genome, &consts, 1.0);
            let wann = genome.to_rust_wann();
            let mut rng = Pcg64::seed_from_u64(0xA11A5);
            let mut scratch = vec![0.0f64; wann.num_nodes];
            for _ in 0..2000 {
                let mut inputs = [0.0f64; INPUT_COUNT];
                for v in inputs.iter_mut() {
                    *v = rng.gen_range(0.0..=1.0);
                }
                wann.forward(&inputs, 1.0, &mut scratch);
                for (&nid, &val) in &consts {
                    if nid <= BIAS_ID {
                        continue;
                    }
                    assert!(
                        (scratch[nid] - val).abs() < 1e-9,
                        "{label}: const node {nid} = {} but runtime {}",
                        val,
                        scratch[nid]
                    );
                }
                for (&nid, &(target, neg)) in &aliases {
                    let expected = if neg { 1.0 - scratch[target] } else { scratch[target] };
                    assert!(
                        (scratch[nid] - expected).abs() < 1e-9,
                        "{label}: alias node {nid} = {} but resolved {}",
                        scratch[nid],
                        expected
                    );
                }
            }
        }
    }
}
