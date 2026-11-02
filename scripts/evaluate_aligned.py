"""Aligned GO evaluation: Group Report baselines vs Diffusion GNN."""
from pathlib import Path
import re
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.stats.multitest import multipletests


ROOT = Path(__file__).resolve().parents[1]
REF = ROOT / "data/metadata/group_reference"
OUTPUT = ROOT / "outputs"
FIGURES = ROOT / "figures"

SEED = 3888
B = 2000
TOP_K = 20

TRACKS = ["Local", "Cross", "Boundary"]

EXPECTED = {
    "Local": (120, 59),
    "Cross": (3985, 83),
    "Boundary": (334, 18),
}

BASELINES = {
    "Degree": "degree",
    "Node2Vec": "N",
    "RWR": "R",
    "CADTP": "cadtp_score",
}


def parse_bool(series):
    values = series.astype("string").str.strip().str.lower()
    result = values.map({
        "true": True,
        "false": False,
        "1": True,
        "0": False,
    })

    if result.isna().any():
        raise ValueError(f"Invalid boolean values: {series.name}")

    return result.astype(bool)


def load_reference():
    """Load original candidate tracks, eligibility and GO labels."""

    print("[1/5] Loading Group Report reference data")

    scores = pd.read_csv(REF / "cadtp_all_scores_refactor.csv")
    labels = pd.read_csv(REF / "go_biological_process_labels_refactor.csv")

    if "orf" not in labels.columns:
        unnamed = [
            c for c in labels.columns
            if c.startswith("Unnamed:")
        ]
        if len(unnamed) == 1:
            labels = labels.rename(columns={unnamed[0]: "orf"})

    if "orf" not in labels or "proteostasis_any" not in labels:
        raise ValueError(f"Unexpected GO label columns: {labels.columns.tolist()}")

    scores["eligible"] = parse_bool(scores["eligible"])
    scores = scores[scores["eligible"]].copy()

    scores["orf"] = scores["orf"].str.upper().str.strip()
    labels["orf"] = labels["orf"].str.upper().str.strip()
    labels["go_positive"] = parse_bool(labels["proteostasis_any"]).astype(int)

    if scores["string_id"].duplicated().any():
        raise ValueError("Duplicate STRING IDs.")

    if labels["orf"].duplicated().any():
        raise ValueError("Duplicate ORFs in GO labels.")

    data = scores.merge(
        labels[["orf", "go_positive"]],
        on="orf",
        how="left",
        validate="many_to_one",
    )

    if data["go_positive"].isna().any():
        raise ValueError("Some eligible proteins are missing GO labels.")

    data["go_positive"] = data["go_positive"].astype(int)

    print("\nReference validation:")
    summary = data.groupby("track").agg(
        proteins=("string_id", "size"),
        go_positive=("go_positive", "sum"),
    )

    print(summary.to_string())

    for track, expected in EXPECTED.items():
        actual = (
            int(summary.loc[track, "proteins"]),
            int(summary.loc[track, "go_positive"]),
        )

        if actual != expected:
            raise ValueError(f"{track} mismatch: {actual} vs {expected}")

    if len(data) != 4439:
        raise ValueError("Expected exactly 4,439 eligible proteins.")

    print("    PASS: Group Report evaluation universe reproduced")

    return data


def attach_gnn_scores(data):
    """Recalculate HLJ1 cosine scores from complete embeddings."""

    print("\n[2/5] Loading saved GNN embeddings")

    nodes = pd.read_csv(ROOT / "data/processed/protein_ids.csv").sort_values("node_index")

    if not np.array_equal(
        nodes["node_index"].to_numpy(),
        np.arange(len(nodes)),
    ):
        raise ValueError("Unexpected node-index ordering.")

    anchor_rows = nodes[nodes["string_id"] == "4932.YMR161W"]

    if len(anchor_rows) != 1:
        raise ValueError("HLJ1 anchor could not be resolved.")

    anchor = int(anchor_rows["node_index"].iloc[0])
    embedding_files = sorted(OUTPUT.glob(f"gnn_embeddings_k*_seed{SEED}.npz"))

    if not embedding_files:
        raise FileNotFoundError("No saved GNN embeddings found.")

    result = data.copy()

    for path in embedding_files:
        match = re.fullmatch(
            r"gnn_embeddings_k(\d+)_seed(\d+)\.npz",
            path.name,
        )

        if match is None:
            continue

        k = int(match.group(1))
        method = f"GNN_K{k}"

        with np.load(path, allow_pickle=False) as archive:
            z = archive["embeddings"].astype(np.float64)

        if z.shape[0] != len(nodes):
            raise ValueError(f"Embedding dimension mismatch: {path}")

        norms = np.linalg.norm(z, axis=1, keepdims=True)

        if np.any(norms < 1e-12):
            raise ValueError(f"Zero-norm embeddings: {path}")

        z = z / norms

        similarities = z @ z[anchor]

        score_map = pd.Series(
            similarities,
            index=nodes["string_id"].to_numpy(),
        )

        result[method] = result["string_id"].map(score_map)

        if result[method].isna().any():
            raise ValueError(f"Missing scores for {method}")

        print(f"    Loaded {method}: {z.shape}")

    return result


def make_strata(pool):
    """Exact degree and distance bins from Group Report."""

    pool = pool.copy()

    pool["degree_bin"] = pd.cut(
        pool["degree"],
        bins=[-np.inf, 5, 20, 60, np.inf],
        labels=["0-5", "6-20", "21-60", "61+"],
    )

    pool["distance_bin"] = (
        pool["h_hops"]
        .clip(upper=4)
        .astype(int)
        .astype(str)
    )

    pool["stratum"] = (
        pool["distance_bin"]
        + "_"
        + pool["degree_bin"].astype(str)
    )

    if pool["stratum"].isna().any():
        raise ValueError("Missing matching strata.")

    return pool


def matched_null(pool, top, rng):
    """Sample matched candidates within the same track."""

    quotas = top["stratum"].value_counts()

    groups = {
        name: group["go_positive"].to_numpy(dtype=int)
        for name, group in pool.groupby("stratum")
    }

    observed = int(top["go_positive"].sum())
    null_hits = np.zeros(B, dtype=int)

    for b in range(B):
        hits = 0

        for stratum, count in quotas.items():
            values = groups[stratum]

            if count > len(values):
                raise ValueError("Insufficient matched candidates.")

            sampled = rng.choice(
                values,
                size=int(count),
                replace=False,
            )

            hits += int(sampled.sum())

        null_hits[b] = hits

    null_mean = float(null_hits.mean())

    return {
        "hits_at_20": observed,
        "precision_at_20": observed / TOP_K,
        "null_mean": null_mean,
        "matched_fe": (
            observed / null_mean
            if null_mean > 0 else np.nan
        ),
        "p_value": (
            1 + np.count_nonzero(null_hits >= observed)
        ) / (B + 1),
    }


def evaluate(data):
    print("\n[3/5] Running matched GO evaluation")

    gnn_methods = sorted(
        [c for c in data.columns if re.fullmatch(r"GNN_K\d+", c)],
        key=lambda name: int(name.split("K")[1]),
    )

    methods = {**BASELINES}
    methods.update({name: name for name in gnn_methods})

    results = []
    top_candidates = []

    test_index = 0

    for track in TRACKS:
        pool = make_strata(
            data[data["track"] == track]
        )

        for method, score_column in methods.items():
            top = pool.sort_values(
                [score_column, "orf", "string_id"],
                ascending=[False, True, True],
                kind="mergesort",
            ).head(TOP_K)

            # Deterministic, independent RNG for each test.
            rng = np.random.default_rng(SEED + test_index)
            test_index += 1

            statistics = matched_null(pool, top, rng)

            results.append({
                "method": method,
                "track": track,
                **statistics,
            })

            for rank, row in enumerate(
                top.itertuples(index=False), start=1
            ):
                top_candidates.append({
                    "method": method,
                    "track": track,
                    "rank": rank,
                    "string_id": row.string_id,
                    "orf": row.orf,
                    "preferred_name": row.preferred_name,
                    "go_positive": row.go_positive,
                })

    results = pd.DataFrame(results)

    # One explicitly defined family across all methods/tracks.
    results["bh_q"] = multipletests(
        results["p_value"],
        method="fdr_bh",
    )[1]

    return results, pd.DataFrame(top_candidates)


def print_results(results):
    print("\n[4/5] Evaluation results")

    print(
        results[
            [
                "method",
                "track",
                "hits_at_20",
                "precision_at_20",
                "matched_fe",
                "p_value",
                "bh_q",
            ]
        ].to_string(index=False)
    )

    print("\nBaseline sanity checks:")

    expected_hits = {
        ("Local", "Degree"): 18,
        ("Local", "Node2Vec"): 15,
        ("Local", "RWR"): 16,
        ("Local", "CADTP"): 16,
        ("Cross", "Degree"): 0,
        ("Cross", "Node2Vec"): 12,
        ("Cross", "RWR"): 4,
        ("Cross", "CADTP"): 9,
    }

    for (track, method), expected in expected_hits.items():
        observed = int(
            results.loc[
                (results["track"] == track)
                & (results["method"] == method),
                "hits_at_20",
            ].iloc[0]
        )

        status = "PASS" if observed == expected else "CHECK"

        print(
            f"    {status}: {track} {method} "
            f"observed={observed}, expected={expected}"
        )

    if any(
            int(
                results.loc[
                    (results["track"] == track)
                    & (results["method"] == method),
                    "hits_at_20",
                ].iloc[0]
            ) != expected
            for (track, method), expected in expected_hits.items()
    ):
        print(
            "\nWARNING: Two Cross-track baseline results differ "
            "from the published Group Report. "
            "Reference counts and GO labels are aligned. "
            "Preserving observed results without modification."
        )


def plot_gnn_results(results):
    print("\n[5/5] Generating comparison figure")

    gnn = results[
        results["method"].str.startswith("GNN_K")
    ].copy()

    gnn["K"] = (
        gnn["method"].str.extract(r"GNN_K(\d+)")[0].astype(int)
    )

    fig, ax = plt.subplots(figsize=(7.5, 4.5))

    for track in TRACKS:
        subset = gnn[
            gnn["track"] == track
        ].sort_values("K")

        ax.plot(
            subset["K"],
            100 * subset["precision_at_20"],
            marker="o",
            label=track,
        )

    ax.set_xlabel("Propagation steps K")
    ax.set_ylabel("GO Precision@20 (%)")
    ax.set_title("Diffusion depth and GO relevance")
    ax.set_xticks(sorted(gnn["K"].unique()))
    ax.set_ylim(0, 100)
    ax.legend()
    ax.grid(alpha=0.25)

    fig.tight_layout()
    fig.savefig(
        FIGURES / "aligned_go_precision_at_20.pdf",
        bbox_inches="tight",
    )
    plt.close(fig)


def main():
    OUTPUT.mkdir(exist_ok=True)
    FIGURES.mkdir(exist_ok=True)

    data = load_reference()
    data = attach_gnn_scores(data)

    results, top_candidates = evaluate(data)

    results.to_csv(
        OUTPUT / "aligned_go_evaluation.csv",
        index=False,
    )

    top_candidates.to_csv(
        OUTPUT / "aligned_top20_candidates.csv",
        index=False,
    )

    print_results(results)
    plot_gnn_results(results)

    print("\nAligned evaluation complete.")
    print("    outputs/aligned_go_evaluation.csv")
    print("    outputs/aligned_top20_candidates.csv")
    print("    figures/aligned_go_precision_at_20.pdf")


if __name__ == "__main__":
    main()
