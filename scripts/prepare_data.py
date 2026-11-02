
from pathlib import Path
import argparse
import re

import networkx as nx
import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]

ORF_PATTERN = re.compile(
    r"Y[A-P][LR]\d{3}[CW](?:-[A-Z])?",
    re.IGNORECASE,
)

ERAD_SEEDS = {"HRD1", "HRD3", "DER1", "UBC7", "CUE1"}

EXPECTED = {
    "full_nodes": 6600,
    "full_edges": 104188,
    "component_nodes": 5716,
    "component_edges": 104138,
    "eligible": 4439,
}


def resolve_path(path):
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def find_column(df, candidates):
    def normalise(value):
        return re.sub(r"[^a-z0-9]", "", str(value).lower())

    available = {
        normalise(column): column
        for column in df.columns
    }

    for candidate in candidates:
        key = normalise(candidate)
        if key in available:
            return available[key]

    raise ValueError(
        f"Cannot identify column from {candidates}.\n"
        f"Available columns: {df.columns.tolist()}"
    )


# ============================================================
# 1. Protein universe
# ============================================================

def load_protein_info(path):
    """
    Load the STRING protein universe and preferred gene names.

    Using protein.info avoids losing isolated proteins when
    applying the confidence threshold to the edge list.
    """
    print(f"[1/6] Loading protein universe: {path}")

    df = pd.read_csv(path, sep="\t", compression="infer")
    df.columns = df.columns.str.lstrip("#")

    id_col = find_column(df, ["string_protein_id", "string_id", "protein_id"])
    name_col = find_column(df, ["preferred_name", "preferredname"])

    df = df[[id_col, name_col]].copy()
    df.columns = ["string_id", "preferred_name"]

    df = df.dropna(subset=["string_id"])
    df["string_id"] = df["string_id"].astype(str).str.strip()
    df["preferred_name"] = df["preferred_name"].fillna("").astype(str).str.strip()

    df = df.drop_duplicates(subset=["string_id"])

    universe = set(df["string_id"])
    preferred_names = dict(zip(df["string_id"], df["preferred_name"]))

    print(f"    Protein universe: {len(universe):,} nodes")

    return universe, preferred_names


# ============================================================
# 2. Network construction
# ============================================================

def load_network(path, threshold, protein_universe):
    """
    Load STRING functional associations.

    Preserve the full protein universe, including proteins
    without edges satisfying the selected threshold.
    """

    print(f"[2/6] Loading STRING edges: {path}")

    df = pd.read_csv(path, compression="infer")

    source_col = find_column(
        df,
        [
            "protein1", "protein_1", "protein_a",
            "source", "source_id", "source_protein",
            "string_id_a", "node1", "node_a", "u",
        ],
    )

    target_col = find_column(
        df,
        [
            "protein2", "protein_2", "protein_b",
            "target", "target_id", "target_protein",
            "string_id_b", "node2", "node_b", "v",
        ],
    )

    score_col = find_column(
        df,
        [
            "combined_score", "combinedscore",
            "confidence", "confidence_score",
            "score", "weight",
        ],
    )

    df = df[[source_col, target_col, score_col]].copy()
    df.columns = ["source", "target", "score"]

    df = df.dropna(subset=["source", "target", "score"])

    df["source"] = df["source"].astype(str).str.strip()
    df["target"] = df["target"].astype(str).str.strip()
    df["score"] = pd.to_numeric(df["score"], errors="coerce")

    df = df.dropna(subset=["score"])

    # Handle scores stored either in [0, 1000] or [0, 1].
    if not df.empty and df["score"].max() <= 1.0:
        df["score"] *= 1000.0

    if ((df["score"] < 0) | (df["score"] > 1000)).any():
        raise ValueError("Invalid STRING confidence scores.")

    pre_threshold_nodes = set(df["source"]) | set(df["target"])
    print(
        f"    Pre-threshold nodes: "
        f"{len(pre_threshold_nodes):,}"
    )

    # Retain confidence >= threshold.
    df = df[df["score"] >= threshold].copy()

    # Remove self-loops.
    df = df[df["source"] != df["target"]].copy()

    # Canonical ordering for undirected edges.
    swap = df["source"] > df["target"]

    df["u"] = df["source"].where(~swap, df["target"])
    df["v"] = df["target"].where(~swap, df["source"])

    # Merge reciprocal and duplicate edges.
    df = df.groupby(["u", "v"], as_index=False)["score"].max()

    df["weight"] = df["score"] / 1000.0

    graph = nx.from_pandas_edgelist(
        df,
        source="u",
        target="v",
        edge_attr="weight",
        create_using=nx.Graph(),
    )

    connected_nodes = graph.number_of_nodes()

    # Keep all proteins from STRING protein.info.
    # Also preserve any edge endpoints absent from protein.info.
    graph.add_nodes_from(protein_universe)

    unmatched_nodes = set(graph.nodes) - protein_universe

    print(f"    Nodes incident to retained edges: {connected_nodes:,}")
    print(
        f"    Full graph: "
        f"{graph.number_of_nodes():,} nodes, "
        f"{graph.number_of_edges():,} edges"
    )
    print(f"    Isolated nodes: {nx.number_of_isolates(graph):,}")

    if unmatched_nodes:
        print(
            f"    WARNING: {len(unmatched_nodes)} edge nodes "
            "are missing from protein.info"
        )

    return graph


# ============================================================
# 3. Protein ID and essentiality mapping
# ============================================================

def load_id_mapping(path):
    df = pd.read_csv(path)
    id_col = find_column(
        df,
        [
            "string_id", "string_protein_id",
            "protein_id", "stringid",
        ],
    )
    orf_col = find_column(df, ["orf", "orf_id", "systematic_name"],)

    df = df[[id_col, orf_col]].dropna()

    return dict(zip(
        df[id_col].astype(str).str.strip(),
        df[orf_col].astype(str).str.upper().str.strip(),
    ))


def get_orf(string_id, mapping):
    orf = mapping.get(string_id)

    if orf and ORF_PATTERN.fullmatch(orf):
        return orf

    # Standard STRING yeast IDs often contain the ORF directly.
    suffix = string_id.split(".")[-1].upper()

    if ORF_PATTERN.fullmatch(suffix):
        return suffix

    return None


def load_essential_orfs(path):
    """
    Read ORFs from a headerless essential_proteins.csv.

    Scan all cells because the CSV may contain multiple
    annotation columns without a header.
    """

    df = pd.read_csv(path, header=None, dtype=str)

    essential = set()

    for value in df.to_numpy().ravel():
        if pd.isna(value):
            continue

        value = str(value).strip().upper()

        if ORF_PATTERN.fullmatch(value):
            essential.add(value)

    print(f"    Essential ORFs loaded: {len(essential):,}")

    return essential


# ============================================================
# 4. Structural node features
# ============================================================

def compute_node_features(graph, nodes, config):
    """
    Construct topology-only node features.

    No GO annotations or biological labels enter the GNN input.
    """

    print("[4/6] Computing structural node features")

    degree = dict(graph.degree())
    strength = dict(graph.degree(weight="weight"))
    clustering = nx.clustering(graph)
    core = nx.core_number(graph)

    features = pd.DataFrame({
        "string_id": nodes,
        "degree": [degree[v] for v in nodes],
        "strength": [strength[v] for v in nodes],
        "clustering_coefficient": [
            clustering[v] for v in nodes
        ],
        "core_number": [core[v] for v in nodes],
    })

    columns = []
    feature_names = []

    if config.get("log_degree", True):
        columns.append(np.log1p(features["degree"].to_numpy()))
        feature_names.append("log_degree")

    if config.get("log_strength", True):
        columns.append(np.log1p(features["strength"].to_numpy()))
        feature_names.append("log_strength")

    if config.get("clustering_coefficient", True):
        columns.append(features["clustering_coefficient"].to_numpy())
        feature_names.append("clustering")

    if config.get("core_number", True):
        columns.append(features["core_number"].to_numpy())
        feature_names.append("core_number")

    if not columns:
        raise ValueError("No node features selected.")

    x = np.column_stack(columns).astype(np.float32)

    if config.get("standardize", True):
        x = StandardScaler().fit_transform(x)
        x = x.astype(np.float32)

    print(f"    Feature matrix: {x.shape}")
    print(f"    Features: {', '.join(feature_names)}")

    return x, features


# ============================================================
# 5. Eligibility and candidate metadata
# ============================================================

def prepare_eligibility(
    features,
    mapping,
    preferred_names,
    essential,
    anchor,
    extra_excluded_orfs,
):
    print("[5/6] Preparing eligibility metadata")

    features = features.copy()
    features.insert(0, "node_index", np.arange(len(features)))

    features["orf"] = [
        get_orf(node, mapping)
        for node in features["string_id"]
    ]

    features["preferred_name"] = [
        preferred_names.get(node, "")
        for node in features["string_id"]
    ]

    features["is_essential"] = features["orf"].isin(essential)
    features["is_anchor"] = (features["string_id"] == anchor)

    extra_excluded_orfs = {
        str(orf).strip().upper()
        for orf in extra_excluded_orfs
    }

    features["is_extra_excluded"] = features["orf"].isin(extra_excluded_orfs)
    features["is_eligible"] = (
        features["orf"].notna()
        & ~features["is_essential"]
        & ~features["is_anchor"]
        & ~features["is_extra_excluded"]
    )

    print("\nEligibility breakdown")
    print("-" * 48)

    print(f"    Component nodes: {len(features):,}")
    print(f"    Missing ORF: {features['orf'].isna().sum():,}")
    print(f"    Essential: {features['is_essential'].sum():,}")
    print(f"    Anchor: {features['is_anchor'].sum():,}")
    print(
        f"    Extra excluded: "
        f"{features['is_extra_excluded'].sum():,}"
    )
    print(
        f"    Eligible: "
        f"{features['is_eligible'].sum():,}"
    )

    # Diagnostic only: ERAD seeds are NOT automatically excluded.
    print("\nERAD seed diagnostics")
    print("-" * 48)

    seed_rows = features[
        features["preferred_name"]
        .str.upper()
        .isin(ERAD_SEEDS)
    ]

    if seed_rows.empty:
        print("    No ERAD seed preferred names matched.")
    else:
        print(
            seed_rows[
                [
                    "preferred_name",
                    "orf",
                    "is_essential",
                    "is_eligible",
                ]
            ].to_string(index=False)
        )

    # Explain the remaining mismatch without altering data.
    expected = EXPECTED["eligible"]
    observed = int(features["is_eligible"].sum())

    if observed != expected:
        print(
            f"\n    NOTE: Eligible count differs from "
            f"Group Report by {observed - expected:+d}."
        )
        print(
            "    Check original candidate exclusion rules "
            "before changing the eligibility mask."
        )

    return features


# ============================================================
# 6. PyTorch Geometric-compatible tensors
# ============================================================

def build_graph_tensors(graph, nodes, x, anchor, features):
    index = {node: i for i, node in enumerate(nodes)}
    edges = sorted(
        (
            index[u],
            index[v],
            float(data["weight"]),
        )
        for u, v, data in graph.edges(data=True)
    )

    source = np.array([e[0] for e in edges], dtype=np.int64)
    target = np.array([e[1] for e in edges], dtype=np.int64)
    weights = np.array([e[2] for e in edges], dtype=np.float32)

    # Represent each undirected edge in both directions.
    edge_index = np.stack([
        np.concatenate([source, target]),
        np.concatenate([target, source]),
    ])

    edge_weight = np.concatenate([weights, weights])

    tensors = {
        "x": torch.from_numpy(x),
        "edge_index": torch.from_numpy(edge_index),
        "edge_weight": torch.from_numpy(edge_weight),
        "anchor_index": index[anchor],
        "num_nodes": len(nodes),
        "eligible_mask": torch.tensor(
            features["is_eligible"].to_numpy(),
            dtype=torch.bool,
        ),
    }

    return tensors


# ============================================================
# 7. Sanity checks
# ============================================================

def print_sanity_checks(observed):
    print("\nSanity checks")
    print("-" * 60)

    for key, expected in EXPECTED.items():
        actual = observed[key]
        status = "PASS" if actual == expected else "CHECK"

        print(
            f"    {status:5s} "
            f"{key:18s} "
            f"observed={actual:,} "
            f"expected={expected:,}"
        )


# ============================================================
# Main
# ============================================================

def main(config_path):
    with open(config_path, "r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    seed = config.get("seed", 3888)

    np.random.seed(seed)
    torch.manual_seed(seed)

    data_cfg = config["data"]

    # 1. Complete protein universe
    protein_info_path = resolve_path(
        data_cfg.get(
            "protein_info",
            "data/raw/4932.protein.info.v12.0.txt.gz",
        )
    )

    universe, preferred_names = load_protein_info(
        protein_info_path
    )

    # 2. Weighted functional network
    graph = load_network(
        resolve_path(data_cfg["edges"]),
        data_cfg["confidence_threshold"],
        universe,
    )

    # 3. Resolve HLJ1 and its connected component
    print("[3/6] Resolving HLJ1 component")

    mapping = load_id_mapping(resolve_path(data_cfg["id_mapping"]))
    essential = load_essential_orfs(resolve_path(data_cfg["essential_proteins"]))

    anchor_orf = data_cfg["anchor_orf"].upper()

    anchor_candidates = [
        node for node in graph.nodes
        if get_orf(node, mapping) == anchor_orf
    ]

    if len(anchor_candidates) != 1:
        raise ValueError(
            f"Expected exactly one HLJ1 node: "
            f"{anchor_candidates}"
        )

    anchor = anchor_candidates[0]
    component = nx.node_connected_component(graph, anchor)
    graph_h = graph.subgraph(component).copy()

    print(f"    HLJ1 STRING ID: {anchor}")
    print(
        f"    HLJ1 component: "
        f"{graph_h.number_of_nodes():,} nodes, "
        f"{graph_h.number_of_edges():,} edges"
    )

    # Deterministic node ordering
    nodes = sorted(graph_h.nodes())

    # 4. Structural features
    x, features = compute_node_features(
        graph_h,
        nodes,
        config["features"],
    )

    # 5. Eligibility metadata
    features = prepare_eligibility(
        features,
        mapping,
        preferred_names,
        essential,
        anchor,
        data_cfg.get("extra_excluded_orfs", []),
    )

    # 6. Construct and save GNN tensors
    print("\n[6/6] Saving processed outputs")

    tensors = build_graph_tensors(
        graph_h,
        nodes,
        x,
        anchor,
        features,
    )

    graph_path = resolve_path(data_cfg["output_graph"])
    nodes_path = resolve_path(data_cfg["output_nodes"])

    graph_path.parent.mkdir(parents=True, exist_ok=True)
    nodes_path.parent.mkdir(parents=True, exist_ok=True)

    torch.save(tensors, graph_path)
    features.to_csv(nodes_path, index=False)

    print(f"    Graph saved: {graph_path}")
    print(f"    Protein metadata saved: {nodes_path}")

    observed = {
        "full_nodes": graph.number_of_nodes(),
        "full_edges": graph.number_of_edges(),
        "component_nodes": graph_h.number_of_nodes(),
        "component_edges": graph_h.number_of_edges(),
        "eligible": int(features["is_eligible"].sum()),
    }

    print_sanity_checks(observed)

    print("\nNetwork preparation complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "configs/default.yaml")

    args = parser.parse_args()

    main(args.config)