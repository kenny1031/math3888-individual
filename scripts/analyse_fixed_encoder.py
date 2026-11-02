"""Fixed-encoder PPR diffusion: convergence and ranking stability."""
from pathlib import Path
import sys
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from models.diffusion import (
    build_operators,
    solve_fixed_point,
    dirichlet_energy,
)
from models.diffusion_gnn import DiffusionGNN


SEED = 3888
TRAINED_K = 4
ALPHA = 0.85
STEPS = [0, 1, 2, 4, 8, 16, 32, 64]
TRACKS = ["Local", "Cross", "Boundary"]


def cosine_scores(H, anchor):
    norms = np.linalg.norm(H, axis=1, keepdims=True)
    Z = H / np.maximum(norms, 1e-12)
    return Z @ Z[anchor], Z


def top20(scores, indices, ids):
    candidates = sorted(
        indices,
        key=lambda i: (-float(scores[i]), str(ids[i])),
    )
    return set(ids[i] for i in candidates[:20])


def jaccard(a, b):
    union = a | b
    return len(a & b) / len(union) if union else 1.0


def main():
    graph = torch.load(
        ROOT / "data/processed/graph.pt",
        map_location="cpu",
        weights_only=True,
    )

    checkpoint = torch.load(
        ROOT / f"outputs/gnn_checkpoint_k{TRAINED_K}_seed{SEED}.pt",
        map_location="cpu",
        weights_only=True,
    )

    cfg = checkpoint["gnn_config"]

    model = DiffusionGNN(
        in_channels=graph["x"].shape[1],
        hidden_dim=cfg["hidden_dim"],
        embedding_dim=cfg["embedding_dim"],
        propagation_steps=TRAINED_K,
        continuation=cfg["continuation"],
        dropout=cfg["dropout"],
        logit_scale=cfg.get("logit_scale", 5.0),
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # FIX encoder parameters; no gradient updates.
    with torch.no_grad():
        H0 = model.encoder(graph["x"]).numpy().astype(np.float64)

    _, _, S = build_operators(
        graph["edge_index"],
        graph["edge_weight"],
        graph["num_nodes"],
    )

    H_star = solve_fixed_point(S, H0, ALPHA)

    denominator = np.linalg.norm(H0 - H_star, "fro")
    if denominator < 1e-12:
        raise ValueError("Initial representation already at fixed point.")

    nodes = pd.read_csv(
        ROOT / "data/processed/protein_ids.csv"
    ).sort_values("node_index").reset_index(drop=True)

    ids = nodes["string_id"].astype(str).to_numpy()
    anchor = int(graph["anchor_index"])

    reference = pd.read_csv(ROOT / "data/metadata/group_reference/cadtp_all_scores_refactor.csv")

    eligible = (
        reference["eligible"]
        .astype(str).str.lower().eq("true")
    )

    reference = reference.loc[eligible].copy()
    index_of = {sid: i for i, sid in enumerate(ids)}

    if not reference["string_id"].isin(index_of).all():
        raise ValueError("Reference proteins missing from the graph.")

    groups = {
        "All": np.array(
            [index_of[sid] for sid in reference["string_id"]],
            dtype=int,
        )
    }

    for track in TRACKS:
        subset = reference[reference["track"] == track]
        groups[track] = np.array(
            [index_of[sid] for sid in subset["string_id"]],
            dtype=int,
        )

    star_scores, _ = cosine_scores(H_star, anchor)
    star_top = {
        track: top20(star_scores, indices, ids)
        for track, indices in groups.items()
    }

    records = []
    top_records = []

    H = H0.copy()
    k_previous = 0

    for k in STEPS:
        for _ in range(k_previous, k):
            H = ALPHA * (S @ H) + (1 - ALPHA) * H0

        k_previous = k

        relative_error = np.linalg.norm(H - H_star, "fro") / denominator

        energy = dirichlet_energy(S, H)
        scores, Z = cosine_scores(H, anchor)

        row = {
            "K": k,
            "relative_error": relative_error,
            "theoretical_bound": ALPHA ** k,
            "dirichlet_energy": energy,
        }

        for track, indices in groups.items():
            chosen = top20(scores, indices, ids)

            row[f"jaccard_{track.lower()}"] = jaccard(
                chosen, star_top[track]
            )

            ordered = sorted(
                indices,
                key=lambda i: (-float(scores[i]), str(ids[i])),
            )[:20]

            for rank, idx in enumerate(ordered, start=1):
                top_records.append({
                    "K": k,
                    "track": track,
                    "rank": rank,
                    "string_id": ids[idx],
                    "cosine": scores[idx],
                })

        records.append(row)

    df = pd.DataFrame(records)

    output = ROOT / "outputs"
    figures = ROOT / "figures"
    output.mkdir(exist_ok=True)
    figures.mkdir(exist_ok=True)

    tag = f"k{TRAINED_K}_seed{SEED}"

    df.to_csv(
        output / f"fixed_encoder_convergence_{tag}.csv",
        index=False,
    )

    pd.DataFrame(top_records).to_csv(
        output / f"fixed_encoder_top20_{tag}.csv",
        index=False,
    )

    print("\nFixed-encoder results:")
    print(df.to_string(index=False, float_format=lambda x: f"{x:.6g}"))

    fig, ax = plt.subplots(figsize=(7, 4))

    ax.semilogy(
        df["K"], df["relative_error"],
        "o-", label="Observed relative error",
    )
    ax.semilogy(
        df["K"], df["theoretical_bound"],
        "--", label="Theoretical upper bound",
    )

    ax.set_xlabel("Propagation steps K")
    ax.set_ylabel("Relative Frobenius error")
    ax.set_title("Fixed-encoder PPR convergence")
    ax.grid(alpha=0.25)
    ax.legend()

    fig.tight_layout()
    fig.savefig(
        figures / f"fixed_encoder_convergence_{tag}.pdf",
        bbox_inches="tight",
    )
    plt.close(fig)

    df_plot = df[df["K"] <= 16].copy()

    fig, ax = plt.subplots(figsize=(7.5, 4.5))

    for track in ["All", *TRACKS]:
        column = f"jaccard_{track.lower()}"

        ax.plot(
            df_plot["K"],
            df_plot[column],
            marker="o",
            markersize=5,
            linewidth=1.8,
            label=track,
        )

    ax.set_xlabel("Propagation steps $K$")
    ax.set_ylabel("Top-20 Jaccard similarity")
    ax.set_title("Protein ranking stability under PPR diffusion")

    ax.set_xlim(0, 16)
    ax.set_ylim(0, 1.05)
    ax.set_xticks([0, 1, 2, 4, 8, 16])
    ax.set_yticks(np.arange(0, 1.01, 0.2))

    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False, loc="lower right")

    fig.tight_layout()
    fig.savefig(
        figures / f"fixed_encoder_ranking_stability_{tag}.pdf",
        bbox_inches="tight",
    )
    plt.close(fig)

    # Check consistency against saved K=4 GNN embeddings.
    saved_path = output / f"gnn_embeddings_{tag}.npz"

    if saved_path.exists():
        with np.load(saved_path, allow_pickle=False) as archive:
            saved_z = archive["embeddings"]

        H4 = H0.copy()
        for _ in range(TRAINED_K):
            H4 = ALPHA * (S @ H4) + (1 - ALPHA) * H0

        _, calculated_z = cosine_scores(H4, anchor)

        print(
            "\nMax absolute difference from saved K=4 embeddings:",
            np.max(np.abs(calculated_z - saved_z)),
        )

    print("\nSaved convergence and ranking stability results.")


if __name__ == "__main__":
    main()
