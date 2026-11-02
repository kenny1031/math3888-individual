
"""Numerical experiments for graph diffusion."""

from pathlib import Path
import argparse
import json
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy.sparse.linalg import eigsh

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from models.diffusion import (
    build_operators,
    diffuse,
    solve_fixed_point,
    dirichlet_energy,
    spectral_response,
)


def run_experiment(alpha=0.85, max_steps=64):
    graph_path = ROOT / "data/processed/graph.pt"
    figure_dir = ROOT / "figures"
    output_dir = ROOT / "outputs"

    figure_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("[1/5] Loading HLJ1 graph")

    graph = torch.load(
        graph_path,
        map_location="cpu",
        weights_only=True,
    )

    H0 = graph["x"].numpy().astype(np.float64)

    W, P, S = build_operators(
        graph["edge_index"],
        graph["edge_weight"],
        graph["num_nodes"],
    )

    n = S.shape[0]

    print(f"    Nodes: {n:,}")
    print(f"    Undirected edges: {W.nnz // 2:,}")
    print(f"    Node features: {H0.shape[1]}")
    print(f"    Alpha: {alpha}")

    # --------------------------------------------------
    # Operator checks
    # --------------------------------------------------
    print("\n[2/5] Checking diffusion operators")

    row_sums = np.asarray(P.sum(axis=1)).ravel()

    stochastic_error = np.max(np.abs(row_sums - 1.0))

    print(f"    P row-sum error: {stochastic_error:.3e}")

    lambda_max = float(
        eigsh(S, k=1, which="LA", return_eigenvectors=False)[0]
    )
    lambda_min = float(
        eigsh(S, k=1, which="SA", return_eigenvectors=False)[0]
    )

    print(f"    Smallest eigenvalue: {lambda_min:.8f}")
    print(f"    Largest eigenvalue:  {lambda_max:.8f}")

    assert stochastic_error < 1e-10
    assert lambda_min >= -1.0 - 1e-6
    assert lambda_max <= 1.0 + 1e-6

    # --------------------------------------------------
    # Fixed-point solution
    # --------------------------------------------------
    print("\n[3/5] Solving diffusion fixed point")

    H_star = solve_fixed_point(S, H0, alpha=alpha)
    residual = H_star - alpha * (S @ H_star) - (1.0 - alpha) * H0
    residual_norm = np.linalg.norm(residual, ord="fro")

    print(f"    Fixed-point residual: {residual_norm:.3e}")

    # --------------------------------------------------
    # Convergence experiment
    # --------------------------------------------------
    print("\n[4/5] Testing theoretical convergence bound")

    check_steps = sorted(set([0, 1, 2, 4, 8, 16, 32, 64, max_steps]))
    check_steps = [k for k in check_steps if k <= max_steps]

    initial_error = np.linalg.norm(H0 - H_star, ord="fro")

    rows = []

    for k in check_steps:
        Hk = diffuse(S, H0, alpha=alpha, steps=k)
        observed = np.linalg.norm(Hk - H_star, ord="fro")
        bound = (alpha ** k) * initial_error
        energy = dirichlet_energy(S, Hk)

        rows.append({
            "steps": k,
            "observed_error": observed,
            "theoretical_bound": bound,
            "dirichlet_energy": energy,
        })

        print(
            f"    k={k:3d} | "
            f"error={observed:.6e} | "
            f"bound={bound:.6e}"
        )

    results = pd.DataFrame(rows)

    # The bound is exact in real arithmetic. Allow a small
    # tolerance for the numerical fixed-point solution.
    tolerance = 1e-8

    if np.any(
        results["observed_error"].to_numpy()
        > results["theoretical_bound"].to_numpy() + tolerance
    ):
        print("    WARNING: Numerical bound check exceeded tolerance.")
    else:
        print("    PASS: Observed errors satisfy the convergence bound.")

    results.to_csv(
        output_dir / "diffusion_convergence.csv",
        index=False,
    )

    # Figure 1: convergence
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.semilogy(
        results["steps"],
        results["observed_error"],
        "o-",
        label="Observed error",
    )

    ax.semilogy(
        results["steps"],
        results["theoretical_bound"],
        "--",
        label="Theoretical upper bound",
    )

    ax.set_xlabel("Propagation steps")
    ax.set_ylabel(r"$\|H^{(k)}-H^*\|_F$")
    ax.set_title("Convergence of PPR-based graph diffusion")
    ax.legend()
    ax.grid(alpha=0.25)

    fig.tight_layout()
    fig.savefig(
        figure_dir / "diffusion_convergence.pdf",
        bbox_inches="tight",
    )
    plt.close(fig)

    # --------------------------------------------------
    # Spectral experiment
    # --------------------------------------------------
    print("\n[5/5] Plotting spectral filter")

    lam = np.linspace(-1.0, 1.0, 500)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for a in [0.5, 0.7, 0.85, 0.95]:
        response = spectral_response(lam, a)
        ax.plot(lam, response, label=fr"$\alpha={a}$")

    ax.axvline(
        lambda_min,
        linestyle=":",
        alpha=0.5,
        label="Observed minimum eigenvalue",
    )

    ax.set_xlabel(r"Eigenvalue $\lambda$")
    ax.set_ylabel(r"$g_\alpha(\lambda)$")
    ax.set_title("Spectral response of restart diffusion")
    ax.legend()
    ax.grid(alpha=0.25)

    fig.tight_layout()
    fig.savefig(
        figure_dir / "diffusion_spectral_response.pdf",
        bbox_inches="tight",
    )
    plt.close(fig)

    summary = {
        "num_nodes": n,
        "num_edges": W.nnz // 2,
        "alpha": alpha,
        "lambda_min": lambda_min,
        "lambda_max": lambda_max,
        "stochastic_error": float(stochastic_error),
        "fixed_point_residual": float(residual_norm),
    }

    (output_dir / "diffusion_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    print("\nExperiment complete.")
    print("    figures/diffusion_convergence.pdf")
    print("    figures/diffusion_spectral_response.pdf")
    print("    outputs/diffusion_convergence.csv")
    print("    outputs/diffusion_summary.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--alpha", type=float, default=0.85)
    parser.add_argument("--max-steps", type=int, default=64)

    args = parser.parse_args()

    run_experiment(alpha=args.alpha, max_steps=args.max_steps)
