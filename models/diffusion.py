"""Graph diffusion operators for the HLJ1 network."""
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import cg


def build_operators(edge_index, edge_weight, num_nodes):
    """
    Construct:
        W: weighted adjacency
        P: row-stochastic random-walk operator
        S: symmetric GNN propagation operator

    Input edge_index contains both directions of each
    undirected STRING edge, without self-loops.
    """
    row, col = edge_index.cpu().numpy()
    weights = edge_weight.cpu().numpy().astype(np.float64)

    n = int(num_nodes)
    W = sparse.coo_matrix(
        (weights, (row, col)),
        shape=(n, n),
        dtype=np.float64,
    ).tocsr()
    W.sum_duplicates()

    if np.any(W.diagonal() != 0):
        raise ValueError("Input graph already contains self-loops.")

    asymmetry = W - W.T
    if asymmetry.nnz and np.max(np.abs(asymmetry.data)) > 1e-10:
        raise ValueError("Expected an undirected weighted graph.")

    # Random-walk operator: P = D^{-1} W
    degree = np.asarray(W.sum(axis=1)).ravel()

    if np.any(degree <= 0):
        raise ValueError("Graph contains isolated vertices.")

    P = sparse.diags(1.0 / degree) @ W

    # GNN operator: S = D_tilde^{-1/2}(W+I)D_tilde^{-1/2}
    W_tilde = W + sparse.eye(n, format="csr")

    degree_tilde = np.asarray(W_tilde.sum(axis=1)).ravel()
    D_inv_sqrt = sparse.diags(1.0 / np.sqrt(degree_tilde))

    S = (D_inv_sqrt @ W_tilde @ D_inv_sqrt).tocsr()

    return W, P.tocsr(), S


def diffuse(S, H0, alpha=0.85, steps=16):
    """
    H^(k+1) = alpha*S*H^(k) + (1-alpha)*H0

    Initially H^(0) = H0.
    """
    if not 0 < alpha < 1:
        raise ValueError("alpha must lie in (0, 1).")

    H0 = np.asarray(H0, dtype=np.float64)
    H = H0.copy()

    for _ in range(steps):
        H = alpha * (S @ H) + (1.0 - alpha) * H0

    return H


def solve_fixed_point(S, H0, alpha=0.85):
    """
    Solve:
        (I-alpha*S) H* = (1-alpha) H0

    Uses conjugate gradient because I-alpha*S
    is symmetric positive definite.
    """
    if not 0 < alpha < 1:
        raise ValueError("alpha must lie in (0, 1).")

    H0 = np.asarray(H0, dtype=np.float64)

    n = S.shape[0]
    A = sparse.eye(n, format="csr") - alpha * S

    H_star = np.empty_like(H0)

    for j in range(H0.shape[1]):
        rhs = (1.0 - alpha) * H0[:, j]

        solution, info = cg(
            A,
            rhs,
            rtol=1e-12,
            atol=0.0,
            maxiter=5000,
        )

        if info != 0:
            raise RuntimeError(
                f"Conjugate gradient failed for column {j}: {info}"
            )

        H_star[:, j] = solution

    return H_star


def dirichlet_energy(S, H):
    """
    Normalised graph Dirichlet energy:

        E(H) = tr(H^T (I-S) H)
    """
    H = np.asarray(H, dtype=np.float64)

    return float(np.sum(H * (H - S @ H)))


def spectral_response(eigenvalues, alpha):
    """
    PPR spectral filter:

        g_alpha(lambda) = (1-alpha)/(1-alpha*lambda)
    """
    eigenvalues = np.asarray(eigenvalues)

    return (1.0 - alpha) / (1.0 - alpha * eigenvalues)