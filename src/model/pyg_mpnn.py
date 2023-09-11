import omegaconf
import torch
import torch.nn as nn
from einops import rearrange
from lion_pytorch import Lion
from torch.nn import GRU
from torch_geometric.data import Batch
from torch_geometric.nn.models import MLP

from src.nn.pyg_mp_layer import MPLayer


class MPNN(nn.Module):
    def __init__(self, config: omegaconf.dictconfig.DictConfig, num_layer: int):
        super().__init__()
        self.node_enc = nn.Linear(config.n_enc_dim, config.model_dim)
        self.edge_enc = nn.Linear(config.e_enc_dim, config.model_dim)

        self.graph_convs = nn.ModuleList(
            [MPLayer(config.node_aggr, config.model_dim, config.residual) for _ in range(num_layer)]
        )

        self.readout = getattr(torch, config.read_aggr)

        if config.h_scope == 'F':
            in_size = config.model_dim
        elif config.h_scope == 'T':
            in_size = (len(self.graph_convs) + 1) * config.model_dim

        if config.regressor == 'MLP':
            hidden_size = in_size // 2
            out_size = 1
            self.reg = MLP([in_size, hidden_size, hidden_size, out_size])

        elif config.regressor == 'RNN':
            self.reg = GRU(input_size=config.model_dim,
                           hidden_size=config.model_dim,
                           num_layers=num_layer)

        self.loss = getattr(nn, config.loss)()

        self.optimizer = Lion(self.parameters(), lr=config.optimizer.lr, weight_decay=config.optimizer.wd)

    def forward(self, batch: Batch, config: omegaconf.dictconfig.DictConfig, test: bool = False):
        nf_list = []
        nf, e_id, ef = batch.x, batch.edge_index, batch.edge_attr[:, :2]
        nf, ef = self.node_enc(nf), self.edge_enc(ef)
        nf_list.append(nf)

        for graph_conv in self.graph_convs:
            nf = graph_conv(nf, e_id, ef)
            nf_list.append(nf)

        if config.h_scope == 'F':
            if batch.batch is None:
                nf = rearrange(nf, '(N C) L -> N C L', N=1)
            else:
                nf = rearrange(nf, '(N C) L -> N C L', N=len(batch))
            h = self.readout(nf, dim=1)

        elif config.h_scope == 'T':
            nfs = torch.stack(nf_list)
            if batch.batch is None:
                nfs = rearrange(nfs, 'N (C1 C2) L -> N C1 C2 L', C1=1)
            else:
                nfs = rearrange(nfs, 'N (C1 C2) L -> N C1 C2 L', C1=len(batch))
            h = self.readout(nfs, dim=2)
            h = rearrange(h, 'T N L -> N (T L)')

        pred = self.reg(h)
        label = batch.y.view(-1, 1)

        if test:
            return pred
            # return nf_list
        else:
            loss = self.loss(pred, label)
            loss.backward()

            self.optimizer.step()
            self.optimizer.zero_grad()

            return loss.item()


class PathMPNN(nn.Module):
    def __init__(self, config, mpnn_type: str, readout_type: str):
        super().__init__()
        self.node_enc = nn.Linear(config.model.n_enc_dim, config.model.model_dim)
        self.edge_enc = nn.Linear(config.model.e_enc_dim, config.model.model_dim)

        self.graph_convs = nn.ModuleList(
            [MPLayer(config.model.node_aggr, config.model.model_dim, config.model.residual)
             for _ in range(config.model.num_layers)])
        self.dec = nn.Linear(config.model.model_dim, config.model.e_enc_dim)

        self.mpnn_type = mpnn_type
        self.readout_type = readout_type

        if (self.mpnn_type == 'nodewise') & (self.readout_type == 'mlp'):
            self.mlp = MLP([64, 32, 32, 1])
        elif (self.mpnn_type == 'pathwise') & (self.readout_type == 'mlp'):
            self.mlp = MLP([10, 5, 5, 1])

        self.loss = getattr(nn, config.model.loss)()

        self.optimizer = Lion(self.parameters(), lr=config.optimizer.lr, weight_decay=config.optimizer.wd)

    def forward(self, batch: Batch, test: bool = False):
        nf, e_id, ef = batch.x, batch.edge_index, batch.edge_attr
        nf, ef = self.node_enc(nf), self.edge_enc(ef)

        for graph_conv in self.graph_convs:
            nf = graph_conv(nf, e_id, ef)

        feat = self.dec(nf).squeeze(-1)
        feat = rearrange(feat, '(N C) -> N C', N=len(batch))

        if self.mpnn_type == 'nodewise':
            if (self.readout_type == 'sum') or (self.readout_type == 'mean'):
                y_hat = getattr(torch, self.readout_type)(feat, dim=-1)
            elif (self.readout_type == 'max') or (self.readout_type == 'min'):
                y_hat = getattr(torch, self.readout_type)(feat, dim=-1)[0]
            else:
                y_hat = self.mlp(feat).squeeze(-1)
        else:
            sch = rearrange(batch.schedule, '(N C) L -> N C L', N=len(batch))
            sch_feat = torch.stack([f[s] for f, s in zip(feat, sch)]).view(len(batch), -1)
            if (self.readout_type == 'sum') or (self.readout_type == 'mean'):
                y_hat = getattr(torch, self.readout_type)(sch_feat, dim=-1)
            elif (self.readout_type == 'max') or (self.readout_type == 'min'):
                y_hat = getattr(torch, self.readout_type)(sch_feat, dim=-1)[0]
            else:
                y_hat = self.mlp(sch_feat).squeeze(-1)

        if test:
            return y_hat.item()
        else:
            loss = self.loss(y_hat, batch.y)
            loss.backward()

            self.optimizer.step()
            self.optimizer.zero_grad()

            return loss.item()
