import torch.nn as nn
from torch_geometric.data import Batch

from src.nn.pyg_mpnn import MPLayer


class MPNN(nn.Module):
    def __init__(
            self,
            n_enc_dim: int,
            e_enc_dim: int,
            model_dim: int,
            num_layers: int,
            node_aggr: str = "add",
            act: str = "ReLU",
            residual: bool = True,
    ):
        super().__init__()
        self.node_enc = nn.Linear(n_enc_dim, model_dim)
        self.edge_enc = nn.Linear(e_enc_dim, model_dim)

        self.num_layers = num_layers
        if self.num_layers == 0:  # Layer sharing
            self.graph_conv = MPLayer(node_aggr, model_dim, act, residual)
        else:
            self.graph_convs = nn.ModuleList([MPLayer(node_aggr, model_dim, act, residual)])

        self.dec = nn.Linear(2 * model_dim, e_enc_dim)
        self.model_dim = model_dim

    def forward(self, batch: Batch):
        nf, e_id, ef = batch.x, batch.edge_attr, batch.edge_index
        nf, ef = self.node_enc(nf), self.edge_enc(ef)

        if self.num_layers <= 0:
            raise NotImplementedError("Layer sharing is not implemented yet.")
        else:
            for graph_conv in self.graph_convs:
                nf = graph_conv(nf, e_id, ef)

        return self.dec(nf)
