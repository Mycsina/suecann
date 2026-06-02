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
                     fontname=Helvetica-Bold shape=box style=\"filled,bold\"]\n",
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

    // Edges
    for c in &genome.conn_genes {
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
    _reachable: &HashSet<usize>,
    _indent: &str,
) -> String {
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

    let feature_names = crate::constants::FEATURE_NAMES;
    let signals: Vec<String> = conns
        .iter()
        .map(|c| {
            let src_str = if c.src == BIAS_ID {
                "1.0".to_string()
            } else if c.src < INPUT_COUNT {
                feature_names[c.src].to_string()
            } else {
                format!("hidden_{}", c.src)
            };
            if c.sign == -1 {
                format!("NOT({})", src_str)
            } else {
                src_str
            }
        })
        .collect();

    let agg_str = match ng.aggregation_fn {
        AggregationFn::SUM => {
            if signals.len() > 1 {
                format!("({})", signals.join(" + "))
            } else {
                signals.join(" + ")
            }
        }
        AggregationFn::MIN => format!("AND({})", signals.join(", ")),
        AggregationFn::MAX => format!("OR({})", signals.join(", ")),
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

    let hidden_nodes: Vec<usize> = active_nodes
        .iter()
        .cloned()
        .filter(|&n| n >= FIRST_HIDDEN_ID)
        .collect();

    if !hidden_nodes.is_empty() {
        out.push_str("Active Hidden Logic Steps:\n");
        for &nid in &hidden_nodes {
            let rhs = format_node_rhs(genome, nid, weight, &reachable, "  ");
            out.push_str(&format!("  hidden_{} = {}\n", nid, rhs));
        }
        out.push('\n');
    }

    out.push_str("Decision Rules for Play Intents:\n");
    for out_idx in 0..OUTPUT_COUNT {
        let nid = OUTPUT_START + out_idx;
        let name = if out_idx < OUTPUT_COUNT {
            output_names[out_idx]
        } else {
            "?"
        };
        if reachable.contains(&nid) {
            let rhs = format_node_rhs(genome, nid, weight, &reachable, "  ");
            out.push_str(&format!("  {} = {}\n", name, rhs));
        } else {
            out.push_str(&format!("  {} = 0.0 (Inactive)\n", name));
        }
    }

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
