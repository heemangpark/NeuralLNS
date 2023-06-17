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
        self.dec = nn.Linear(2 * model_dim, e_enc_dim)

        self.graph_convs = nn.ModuleList([MPLayer(node_aggr, model_dim, act, residual)
                                          for _ in range(num_layers)])

        self.residual = residual

    def forward(self, batch: Batch):
        nf, e_id, ef = batch.x, batch.edge_index, batch.edge_attr
        nf, ef = self.node_enc(nf), self.edge_enc(ef)

        for graph_conv in self.graph_convs:
            nf = graph_conv(nf, e_id, ef)

        return self.dec(nf)
