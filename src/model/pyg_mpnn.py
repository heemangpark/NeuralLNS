import torch.nn as nn
from einops import rearrange
from lion_pytorch import Lion
from torch_geometric.data import Batch
from torch_geometric.nn.models import MLP

from src.nn.pyg_mp_layer import MPLayer


class MPNN(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dim = config.model_dim

        self.node_enc = nn.Linear(config.n_enc_dim, config.model_dim)
        self.edge_enc = nn.Linear(config.e_enc_dim, config.model_dim)
        self.dec = nn.Linear(config.model_dim, 1)

        self.graph_convs = nn.ModuleList([MPLayer(config.node_aggr, config.model_dim, config.act, config.residual)
                                          for _ in range(config.num_layers)])

        self.mlp = MLP([2 * config.model_dim, config.model_dim, config.model_dim, 1])
        self.mlp.reset_parameters()

        self.loss = getattr(nn, config.loss)()
        self.optimizer = Lion(self.parameters(), lr=config.optimizer.lr, weight_decay=config.optimizer.wd)

    def forward(self, batch: Batch):
        nf, e_id, ef = batch.x, batch.edge_index, batch.edge_attr
        nf, ef = self.node_enc(nf), self.edge_enc(ef)

        for graph_conv in self.graph_convs:
            nf = graph_conv(nf, e_id, ef)

        b_nf = rearrange(nf, '(N C) L -> N C L', N=len(batch))
        cat_b_nf = rearrange(b_nf, 'N (C1 C2) L -> N C2 (C1 L)', C1=2)

        pred = self.mlp(rearrange(cat_b_nf, 'N C L -> (N C) L'))
        y_hat = rearrange(pred, '(N1 N2) C -> N1 (N2 C)', N1=len(batch))
        label = rearrange(batch.y, '(N C) -> N C', N=len(batch))

        loss = self.loss(y_hat, label)
        loss.backward()

        self.optimizer.step()
        self.optimizer.zero_grad()

        return loss.item()
