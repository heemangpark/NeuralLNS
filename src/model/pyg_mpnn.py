import torch
import torch.nn as nn
from torch_geometric.data import Batch

from src.nn.pyg_mpnn import MPLayer


class MPNN(nn.Module):
    def __init__(
            self,
            node_dim: int,
            edge_dim: int,
            model_dim: int,
            num_layers: int,
            act: str = "ReLU",
            node_aggr: str = "add",
            residual: bool = True,
    ):
        super().__init__()
        self.node_enc = nn.Linear(node_dim, model_dim)
        self.edge_enc = nn.Linear(edge_dim, model_dim)

        self.num_layers = num_layers
        if self.num_layers == 0:  # Layer sharing
            self.graph_conv = MPLayer(model_dim, act, residual, node_aggr)
        else:
            self.graph_convs = nn.ModuleList([MPLayer(model_dim, act, residual, node_aggr)])

        self.dec = nn.Linear(2 * model_dim, edge_dim)
        self.model_dim = model_dim

    def forward(self, batch: Batch) -> torch.Tensor:
        nf, ef = batch.x, batch.edge_attr
        nf, ef = self.node_enc(nf), self.edge_enc(ef)

        # Path convolution with layer sharing
        if self.num_layers <= 0:
            raise NotImplementedError("Layer sharing is not implemented yet.")
        else:
            for graph_conv in self.graph_convs:
                nf, ef = graph_conv(nf, ef, batch.edge_index)

        return nf, ef
