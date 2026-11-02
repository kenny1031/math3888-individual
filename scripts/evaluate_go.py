
"""Evaluate GNN rankings using GO annotations and matched nulls."""

from pathlib import Path
import argparse
import sys

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import torch
import yaml
from statsmodels.stats.multitest import multipletests


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from evaluation.go import build_go_labels


TRACKS = ["Local", "Cross", "Boundary"]


def load_network(graph):
    edges = graph["edge_index"].cpu().numpy()
    weights = graph["edge_weight"].cpu().numpy()

    mask = edges[0] < edges[1]

    G = nx.Graph()
    G.add_nodes_from(range(graph["num_nodes"]))

    G.add_weighted_edges_from((
        int(u), int(v), float(w))
        for u, v, w in zip(
            edges[0, mask],
            edges[1, mask],
            weights[mask],
        )
    )

    return G


def compute_tracks(G, anchor, seeds):
    """Louvain co-membership frequency with HLJ1."""
    counts = np.zeros(G.number_of_nodes(), dtype=int)

    for seed in seeds:
        communities = nx.community.louvain_communities(
            G,
            weight="weight",
            resolution=1.0,
            seed=seed,
        )

        anchor_community = next(
            community
            for community in communities
            if anchor in community
        )

        counts[list(anchor_community)] += 1

    frequency = counts / len(seeds)

    track = np.where(
        frequency >= 0.8,
        "Local",
        np.where(frequency <= 0.2, "Cross", "Boundary"),
    )

    return pd.DataFrame({
        "node_index": np.arange(G.number_of_nodes()),
        "membership_frequency": frequency,
        "track": track,
    })


def matched_null_test(
    population,
    selected,
    degree_bins=8,
    B=2000,
    seed=3888,
):
    """
    Degree- and distance-stratified null sampling.

    This is an explicitly defined exploratory null.
    It is not guaranteed to reproduce the exact
    Group Report randomisation implementation.
    """

    rng = np.random.default_rng(seed)

    population = population.copy()
    population["degree_bin"] = pd.qcut(
        population["degree"].rank(method="first"),
        q=min(degree_bins, len(population)),
        labels=False,
        duplicates="drop",
    )
    
    population["stratum"] = list(zip(
        population["distance"].astype(int),
        population["degree_bin"].astype(int),
    ))

    selected_indices = set(selected["node_index"])
    top = population[population["node_index"].isin(selected_indices)]
    observed = int(top["go_positive"].sum())
    pools = {
        key: group["go_positive"].to_numpy(dtype=int)
        for key, group in population.groupby("stratum")
    }
    selected_counts = top["stratum"].value_counts()
    null_hits = np.zeros(B, dtype=int)

    for b in range(B):
        hits = 0
        for stratum, count in selected_counts.items():
            pool = pools[stratum]
            sampled = rng.choice(
                pool,
                size=int(count),
                replace=False,
            )

            hits += int(sampled.sum())

        null_hits[b] = hits

    p = (1 + np.count_nonzero(null_hits >= observed)) / (B + 1)

    null_mean = float(null_hits.mean())

    return {
        "observed_hits": observed,
        "null_mean": null_mean,
        "matched_fe": (
            observed / null_mean
            if null_mean > 0 else np.nan
        ),
        "p_value": p,
    }


def main(seed):
    with open(
        ROOT / "configs/default.yaml",
        encoding="utf-8",
    ) as f:
        config = yaml.safe_load(f)

    cfg = config["evaluation"]

    graph = torch.load(
        ROOT / config["data"]["output_graph"],
        map_location="cpu",
        weights_only=True,
    )

    nodes = pd.read_csv(
        ROOT / config["data"]["output_nodes"]
    ).sort_values("node_index").reset_index(drop=True)

    if len(nodes) != graph["num_nodes"]:
        raise ValueError("Graph/node metadata size mismatch.")

    nodes["is_eligible"] = nodes["is_eligible"].astype(str).str.lower() == "true"

    output_dir = ROOT / "outputs"
    figure_dir = ROOT / "figures"
    output_dir.mkdir(exist_ok=True)
    figure_dir.mkdir(exist_ok=True)

    print("[1/4] Preparing GO annotations")

    labels = build_go_labels(
        gaf_path=ROOT / cfg["gaf"],
        obo_path=ROOT / cfg["obo"],
        nodes=nodes,
        roots=cfg["go_roots"],
        include_part_of=cfg.get("include_part_of", False),
        exclude_evidence=cfg.get("exclude_evidence", []),
    )

    labels.to_csv(
        ROOT / "data/processed/go_labels.csv",
        index=False,
    )

    print("\n[2/4] Preparing Louvain tracks")

    G = load_network(graph)
    anchor = int(graph["anchor_index"])

    track_path = ROOT / "data/processed/louvain_tracks.csv"

    if track_path.exists():
        tracks = pd.read_csv(track_path)
        print("    Using existing Louvain tracks.")
    else:
        print("    Computing weighted Louvain ensemble.")
        tracks = compute_tracks(G, anchor, cfg["louvain_seeds"])
        tracks.to_csv(track_path, index=False)

    if (len(tracks) != len(nodes)
        or tracks["node_index"].nunique() != len(nodes)):
        raise ValueError("Louvain track indices do not match graph.")

    nodes = (
        nodes.merge(
            labels[["node_index", "go_positive"]],
            on="node_index",
            validate="one_to_one",
        )
        .merge(
            tracks[["node_index", "track"]],
            on="node_index",
            validate="one_to_one",
        )
    )

    distances = nx.single_source_shortest_path_length(G, anchor)
    nodes["distance"] = nodes["node_index"].map(distances)
    eligible = nodes[nodes["is_eligible"]].copy()

    print("\nEligible track summary:")
    summary = eligible.groupby("track").agg(
        proteins=("node_index", "size"),
        go_positive=("go_positive", "sum"),
    )

    print(summary.to_string())

    expected = {
        "Local": (120, 59),
        "Cross": (3985, 83),
        "Boundary": (334, 18),
    }

    for track, (n_expected, go_expected) in expected.items():
        if track not in summary.index:
            print(f"    CHECK: {track} missing.")
            continue

        row = summary.loc[track]

        if (
            row["proteins"] != n_expected
            or row["go_positive"] != go_expected
        ):
            print(
                f"    CHECK {track}: does not match "
                "Group Report counts."
            )

    print("\n[3/4] Evaluating trained GNN rankings")

    files = sorted(output_dir.glob(f"gnn_rankings_k*_seed{seed}.csv"))

    if not files:
        raise FileNotFoundError("No GNN ranking CSVs found.")

    results = []

    for file in files:
        ranking = pd.read_csv(file)
        ranking = ranking.merge(
            eligible[[
                "node_index",
                "track",
                "degree",
                "distance",
                "go_positive",
            ]],
            on="node_index",
            how="inner",
            validate="one_to_one",
        )

        if ranking.empty:
            continue

        method = file.stem.replace("gnn_rankings_", "")

        for track in TRACKS:
            pool = ranking[ranking["track"] == track].copy()

            if len(pool) < 20:
                continue

            pool = pool.sort_values(
                ["gnn_cosine", "string_id"],
                ascending=[False, True],
            )

            top20 = pool.head(20)

            null = matched_null_test(
                population=pool,
                selected=top20,
                degree_bins=cfg["degree_bins"],
                B=cfg["null_randomisations"],
                seed=seed,
            )

            results.append({
                "method": method,
                "track": track,
                "hits_at_20": null["observed_hits"],
                "precision_at_20": (
                    null["observed_hits"] / 20
                ),
                "null_mean": null["null_mean"],
                "matched_fe": null["matched_fe"],
                "p_value": null["p_value"],
            })

    results = pd.DataFrame(results)

    if results.empty:
        raise RuntimeError("No evaluation results generated.")

    # BH adjustment across all tests in this experiment.
    results["bh_q"] = multipletests(
        results["p_value"],
        method="fdr_bh",
    )[1]

    results = results.sort_values(["track", "method"])
    results.to_csv(
        output_dir / "gnn_go_evaluation.csv",
        index=False,
    )

    print(results[[
        "method", "track", "hits_at_20",
        "precision_at_20", "matched_fe",
        "p_value", "bh_q",
    ]].to_string(index=False))

    print("\n[4/4] Generating figure")

    pivot = results.pivot(
        index="method",
        columns="track",
        values="precision_at_20",
    )

    pivot = pivot.reindex(columns=TRACKS)

    ax = pivot.plot.bar(figsize=(9, 4.5), rot=0)
    ax.set_xlabel("GNN configuration")
    ax.set_ylabel("GO Precision@20")
    ax.set_ylim(0, 1)
    ax.set_title("GO evaluation across diffusion depths")
    ax.grid(axis="y", alpha=0.25)

    fig = ax.get_figure()
    fig.tight_layout()

    fig.savefig(
        figure_dir / "gnn_go_precision_at_20.pdf",
        bbox_inches="tight",
    )
    plt.close(fig)

    print("\nSaved:")
    print("    data/processed/go_labels.csv")
    print("    data/processed/louvain_tracks.csv")
    print("    outputs/gnn_go_evaluation.csv")
    print("    figures/gnn_go_precision_at_20.pdf")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=3888)

    args = parser.parse_args()
    main(args.seed)
