
"""Train a self-supervised diffusion-based GNN."""

from pathlib import Path
import argparse
import random
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import yaml

from torch.nn import functional as F
from torch_geometric.utils import negative_sampling


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from models.diffusion_gnn import DiffusionGNN


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_graph(path: Path):
    return torch.load(
        path,
        map_location="cpu",
        weights_only=True,
    )


def sample_training_pairs(
    edge_index: torch.Tensor,
    positive_edges: torch.Tensor,
    num_nodes: int,
    batch_size: int,
    negative_ratio: float,
):
    """
    Sample positive edges and unobserved node pairs.

    All sampling is performed on CPU for compatibility.
    """

    n_positive = min(
        batch_size,
        positive_edges.shape[1],
    )

    indices = torch.randperm(
        positive_edges.shape[1]
    )[:n_positive]

    pos_batch = positive_edges[:, indices]

    n_negative = max(
        1,
        round(n_positive * negative_ratio),
    )

    # Excludes observed edges from negative samples.
    neg_batch = negative_sampling(
        edge_index=edge_index,
        num_nodes=num_nodes,
        num_neg_samples=n_negative,
        method="sparse",
        force_undirected=False,
    )

    if neg_batch.shape[1] == 0:
        raise RuntimeError("No negative edges were sampled.")

    return pos_batch, neg_batch


def reconstruction_loss(
    model,
    z,
    pos_pairs,
    neg_pairs,
):
    pos_logits = model.decode(z, pos_pairs)
    neg_logits = model.decode(z, neg_pairs)

    pos_loss = F.binary_cross_entropy_with_logits(
        pos_logits,
        torch.ones_like(pos_logits),
    )

    neg_loss = F.binary_cross_entropy_with_logits(
        neg_logits,
        torch.zeros_like(neg_logits),
    )

    loss = pos_loss + neg_loss

    with torch.no_grad():
        pos_prob = torch.sigmoid(pos_logits).mean().item()
        neg_prob = torch.sigmoid(neg_logits).mean().item()

    return loss, pos_prob, neg_prob


def save_rankings(
    z,
    graph,
    output_path: Path,
):
    """
    Rank eligible proteins by cosine similarity to HLJ1.
    GO labels are not used.
    """

    nodes = pd.read_csv(
        ROOT / "data/processed/protein_ids.csv"
    )

    nodes = nodes.sort_values(
        "node_index"
    ).reset_index(drop=True)

    if len(nodes) != z.shape[0]:
        raise ValueError("Protein metadata and embeddings do not align.")

    if not np.array_equal(
        nodes["node_index"].to_numpy(),
        np.arange(len(nodes)),
    ):
        raise ValueError("Unexpected node-index ordering.")

    anchor_index = int(graph["anchor_index"])

    # z is already L2-normalised.
    scores = (
        z @ z[anchor_index]
    ).detach().cpu().numpy()

    ranking = nodes[
        [
            "node_index",
            "string_id",
            "orf",
            "preferred_name",
            "is_eligible",
        ]
    ].copy()

    ranking["gnn_cosine"] = scores

    eligible = graph["eligible_mask"].cpu().numpy()

    ranking = ranking.loc[eligible].copy()

    ranking = ranking.sort_values(
        ["gnn_cosine", "string_id"],
        ascending=[False, True],
    ).reset_index(drop=True)

    ranking.insert(
        0,
        "rank",
        np.arange(1, len(ranking) + 1),
    )

    ranking.to_csv(output_path, index=False)

    return ranking


def train(args):
    config_path = (
        args.config
        if args.config.is_absolute()
        else ROOT / args.config
    )

    with open(config_path, encoding="utf-8") as f:
        config = yaml.safe_load(f)

    cfg = config["gnn"]

    seed = args.seed if args.seed is not None else config["seed"]

    steps = (
        args.steps
        if args.steps is not None
        else cfg["propagation_steps"]
    )

    epochs = (
        args.epochs
        if args.epochs is not None
        else cfg["epochs"]
    )

    if epochs < 1:
        raise ValueError("epochs must be positive.")

    set_seed(seed)

    device = torch.device(cfg.get("device", "cpu"))

    output_dir = ROOT / "outputs"
    figure_dir = ROOT / "figures"

    output_dir.mkdir(exist_ok=True)
    figure_dir.mkdir(exist_ok=True)

    tag = f"k{steps}_seed{seed}"

    print("[1/4] Loading graph")

    graph = load_graph(
        ROOT / config["data"]["output_graph"]
    )

    x = graph["x"].to(device)
    edge_index = graph["edge_index"].to(device)
    edge_weight = graph["edge_weight"].to(device)

    num_nodes = int(graph["num_nodes"])

    # Each undirected edge appears twice in edge_index.
    # Keep a single orientation for positive training pairs.
    positive_mask = (
        graph["edge_index"][0]
        < graph["edge_index"][1]
    )

    positive_edges = graph["edge_index"][
        :, positive_mask
    ]

    print(f"    Nodes: {num_nodes:,}")
    print(f"    Undirected edges: {positive_edges.shape[1]:,}")
    print(f"    Feature dimension: {x.shape[1]}")
    print(f"    Device: {device}")

    print("\n[2/4] Initialising diffusion GNN")

    model = DiffusionGNN(
        in_channels=x.shape[1],
        hidden_dim=cfg["hidden_dim"],
        embedding_dim=cfg["embedding_dim"],
        propagation_steps=steps,
        continuation=cfg["continuation"],
        dropout=cfg["dropout"],
        logit_scale=cfg.get("logit_scale", 5.0),
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg["learning_rate"],
        weight_decay=cfg["weight_decay"],
    )

    print(
        f"    Trainable parameters: "
        f"{sum(p.numel() for p in model.parameters()):,}"
    )
    print(f"    Propagation steps: {steps}")
    print(f"    Continuation: {cfg['continuation']}")
    print(f"    Epochs: {epochs}")

    history = []

    print("\n[3/4] Training")

    for epoch in range(1, epochs + 1):
        model.train()

        pos_pairs, neg_pairs = sample_training_pairs(
            edge_index=graph["edge_index"],
            positive_edges=positive_edges,
            num_nodes=num_nodes,
            batch_size=cfg["positive_batch_size"],
            negative_ratio=cfg["negative_ratio"],
        )

        pos_pairs = pos_pairs.to(device)
        neg_pairs = neg_pairs.to(device)

        optimizer.zero_grad(set_to_none=True)

        z = model(
            x,
            edge_index,
            edge_weight,
        )

        loss, pos_prob, neg_prob = reconstruction_loss(
            model,
            z,
            pos_pairs,
            neg_pairs,
        )

        if not torch.isfinite(loss):
            raise RuntimeError(
                f"Non-finite loss at epoch {epoch}"
            )

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=1.0,
        )

        optimizer.step()

        history.append({
            "epoch": epoch,
            "loss": loss.item(),
            "positive_probability": pos_prob,
            "negative_probability": neg_prob,
        })

        if epoch == 1 or epoch % 10 == 0:
            print(
                f"    Epoch {epoch:3d}/{epochs} | "
                f"Loss: {loss.item():.5f} | "
                f"Positive: {pos_prob:.3f} | "
                f"Negative: {neg_prob:.3f}"
            )

    print("\n[4/4] Saving results")

    model.eval()

    with torch.no_grad():
        final_z = model(
            x,
            edge_index,
            edge_weight,
        ).cpu()

    # Save complete embeddings in the same node order
    # as data/processed/protein_ids.csv.
    np.savez_compressed(
        output_dir / f"gnn_embeddings_{tag}.npz",
        embeddings=final_z.numpy(),
    )

    ranking = save_rankings(
        final_z,
        graph,
        output_dir / f"gnn_rankings_{tag}.csv",
    )

    # Save model for reproducibility.
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "seed": seed,
            "propagation_steps": steps,
            "gnn_config": cfg,
        },
        output_dir / f"gnn_checkpoint_{tag}.pt",
    )

    history_df = pd.DataFrame(history)

    history_df.to_csv(
        output_dir / f"gnn_training_{tag}.csv",
        index=False,
    )

    fig, ax = plt.subplots(figsize=(7, 4))

    ax.plot(
        history_df["epoch"],
        history_df["loss"],
    )

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Reconstruction loss")
    ax.set_title(
        f"Diffusion GNN training (K={steps})"
    )

    ax.grid(alpha=0.25)

    fig.tight_layout()

    fig.savefig(
        figure_dir / f"gnn_training_{tag}.pdf",
        bbox_inches="tight",
    )

    plt.close(fig)

    print("\nTop 20 eligible proteins:")
    print(
        ranking[
            ["rank", "preferred_name", "orf", "gnn_cosine"]
        ].head(20).to_string(index=False)
    )

    print(f"\nFinal training loss: {history[-1]['loss']:.5f}")
    print(f"Results saved with tag: {tag}")
    print("Training complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)

    train(parser.parse_args())
