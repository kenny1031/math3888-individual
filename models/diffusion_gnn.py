"""Self-supervised PPR-based GNN for protein representation learning."""
import torch
from torch import nn
from torch.nn import functional as F
from torch_geometric.nn import APPNP


class DiffusionGNN(nn.Module):
    """
    MLP encoder + weighted APPNP propagation.

    H0 = MLP(X)
    H(k+1) = alpha * S @ H(k) + (1-alpha) * H0

    Output embeddings are L2-normalised for ranking and
    inner-product edge reconstruction.
    """
    def __init__(
        self,
        in_channels: int,
        hidden_dim: int = 64,
        embedding_dim: int = 64,
        propagation_steps: int = 8,
        continuation: float = 0.85,
        dropout: float = 0.15,
        logit_scale: float = 5.0,
    ):
        super().__init__()

        if not 0.0 < continuation < 1.0:
            raise ValueError("continuation must be between 0 and 1.")

        if propagation_steps < 0:
            raise ValueError("propagation_steps must be non-negative.")

        self.propagation_steps = propagation_steps
        self.continuation = continuation
        self.logit_scale = logit_scale

        # Learnable node feature transformation.
        self.encoder = nn.Sequential(
            nn.Linear(in_channels, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embedding_dim),
        )

        # In PyG, APPNP alpha is the RESTART probability.
        # Our mathematical alpha is the CONTINUATION probability.
        if propagation_steps > 0:
            self.propagation = APPNP(
                K=propagation_steps,
                alpha=1.0 - continuation,
                dropout=0.0,
                cached=True,
                add_self_loops=True,
                normalize=True,
            )
        else:
            self.propagation = None

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor,
    ) -> torch.Tensor:
        h0 = self.encoder(x)
        if self.propagation is None:
            h = h0
        else:
            h = self.propagation(
                h0,
                edge_index,
                edge_weight=edge_weight,
            )

        # Normalisation is applied AFTER diffusion.
        z = F.normalize(h, p=2, dim=-1, eps=1e-8)

        return z

    def decode(
        self,
        z: torch.Tensor,
        pairs: torch.Tensor,
    ) -> torch.Tensor:
        """Return edge logits for node pairs."""
        u, v = pairs
        return self.logit_scale * (z[u] * z[v]).sum(dim=-1)