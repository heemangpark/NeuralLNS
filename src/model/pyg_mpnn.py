import torch.nn as nn
from torch_geometric.data import Batch

from src.nn.pyg_mp_layer import MPLayer


class MPNN(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.node_enc = nn.Linear(config.n_enc_dim, config.model_dim)
        self.edge_enc = nn.Linear(config.e_enc_dim, config.model_dim)
        self.dec = nn.Linear(config.model_dim, 1)

        self.graph_convs = nn.ModuleList([MPLayer(config.node_aggr, config.model_dim, config.act, config.residual)
                                          for _ in range(config.num_layers)])

        self.to(config.device)

    def forward(self, batch: Batch):
        B = batch.num_graphs

        nf, e_id, ef = batch.x, batch.edge_index, batch.edge_attr
        nf, ef = self.node_enc(nf), self.edge_enc(ef)

        for graph_conv in self.graph_convs:
            nf = graph_conv(nf, e_id, ef)

        return self.dec(nf).view(B, -1, 1)
