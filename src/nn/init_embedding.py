import torch
import torch.nn as nn
from torch_geometric.data import Batch


def get_init_embedding(mode: str):
    # assert mode in ["momd", "somd", "sosd"]

    if mode in ["momd", "MoMd"]:
        emb_cls = MOMDInitEmbedding
    elif mode in ["somd", "SoMd"]:
        emb_cls = SOMDInitEmbedding
    elif mode in ["sosd", "SoSd"]:
        emb_cls = SOSDInitEmbedding
    else:
        raise ValueError(f"Invalid mode: {mode}")
    return emb_cls


class MOMDInitEmbedding(nn.Module):
    """
    Multi Origin Multi Destination (MOMD) Init Embedding
    """

    def __init__(self, node_dim: int, edge_dim: int, embed_dim: int):
        super().__init__()
        self.node_emb = nn.Linear(node_dim, embed_dim)
        self.edge_emb = nn.Linear(edge_dim, embed_dim)

    def forward(self, batch: Batch) -> torch.Tensor:
        nf, ef = batch.x, batch.edge_attr
        node_emb = self.node_emb(nf)
        edge_emb = self.edge_emb(ef)
        return node_emb, edge_emb


class SOMDInitEmbedding(nn.Module):

    """
    Single Origin Multi Destination (SOMD) Init Embedding

    Assume the first node is the origin
    and remaining nodes are the destinations.
    """

    def __init__(self, node_dim: int, edge_dim: int, embed_dim: int):
        super().__init__()
        self.node_emb = nn.Linear(node_dim, embed_dim)
        self.edge_emb = nn.Linear(edge_dim, embed_dim)
        self.origin_emb = nn.Parameter(torch.randn(embed_dim))

    def forward(self, batch: Batch) -> torch.Tensor:
        nf, ef = batch.x, batch.edge_attr
        node_emb = self.node_emb(nf)  # [Total #. nodes, embed_dim]
        edge_emb = self.edge_emb(ef)

        assert node_emb.shape[0] % batch.num_graphs == 0
        n = node_emb.shape[0] // batch.num_graphs

        # Assume the first node is the origin
        node_emb[::n] += self.origin_emb

        return node_emb, edge_emb


class SOSDInitEmbedding(nn.Module):

    """
    Single Origin Single Destination (SOSD) Init Embedding

    Assume the first node is the origin
    and remaining nodes are the destinations.
    """

    def __init__(self, node_dim: int, edge_dim: int, embed_dim: int):
        super().__init__()
        self.node_emb = nn.Linear(node_dim, embed_dim)
        self.edge_emb = nn.Linear(edge_dim, embed_dim)
        self.origin_emb = nn.Parameter(torch.randn(embed_dim))
        self.destination_emb = nn.Parameter(torch.randn(embed_dim))

    def forward(self, batch: Batch) -> torch.Tensor:
        nf, ef = batch.x, batch.edge_attr
        node_emb = self.node_emb(nf)  # [Total #. nodes, embed_dim]
        edge_emb = self.edge_emb(ef)

        assert node_emb.shape[0] % batch.num_graphs == 0
        n = node_emb.shape[0] // batch.num_graphs

        # Assume the first node is the origin
        node_emb[::n] += self.origin_emb

        # Assume the last node is the destination
        node_emb[n - 1 :: n, :] += self.destination_emb

        return node_emb, edge_emb


if __name__ == "__main__":
    problem_type = "sosd"
    node_dim = 2
    edge_dim = 3
    model_dim = 4
    init = get_init_embedding(problem_type)(node_dim, edge_dim, model_dim)
    print(init)
