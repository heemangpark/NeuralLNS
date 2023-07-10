import torch
import torch.nn as nn
from einops import rearrange
from lion_pytorch import Lion
from torch_geometric.data import Batch
from torch_geometric.nn.models import MLP

from src.nn.pyg_mp_layer import MPLayer


class MPNN(nn.Module):
    def __init__(self, config, edge_type: int):
        super().__init__()
        model_config = config.model

        self.node_enc = nn.Linear(model_config.n_enc_dim, model_config.model_dim)
        self.edge_enc = nn.Linear(edge_type, model_config.model_dim)

        self.graph_convs = nn.ModuleList(
            [MPLayer(model_config.node_aggr, model_config.model_dim, model_config.act, model_config.residual)
             for _ in range(model_config.num_layers)])

        self.mlp = MLP([model_config.model_dim * 2, model_config.model_dim, model_config.model_dim // 2, 1])
        self.mlp.reset_parameters()

        self.loss = getattr(nn, model_config.loss)()
        self.optimizer = Lion(self.parameters(), lr=model_config.optimizer.lr, weight_decay=model_config.optimizer.wd)

        self.to(config.device)

    def forward(self, batch: Batch, type: None):
        num_object = batch[0].y.shape[0]
        nf, e_id = batch.x, batch.edge_index

        if type == 'ones':
            ef = torch.ones_like(batch.edge_attr)[:, 0].view(-1, 1)
        elif type == 'A':
            ef = batch.edge_attr[:, 0].view(-1, 1)
        elif type == 'M':
            ef = batch.edge_attr[:, 1].view(-1, 1)
        elif type == 'AP':
            ef = torch.cat((batch.edge_attr[:, 0].view(-1, 1), batch.edge_attr[:, 2].view(-1, 1)), dim=-1)
        elif type == 'MP':
            ef = torch.cat((batch.edge_attr[:, 1].view(-1, 1), batch.edge_attr[:, 2].view(-1, 1)), dim=-1)
        else:
            ef = batch.edge_attr

        nf, ef = self.node_enc(nf), self.edge_enc(ef)

        for graph_conv in self.graph_convs:
            nf = graph_conv(nf, e_id, ef)

        nf = rearrange(nf, '(N C) L -> N C L', N=len(batch))
        agent_nf, task_nf = nf[:, -num_object * 2:-num_object, :], nf[:, -num_object:, :]
        nf = rearrange(torch.cat((agent_nf, task_nf), -1), 'N C L -> (N C) L')

        y_hat = self.mlp(nf)
        label = batch.y.view(-1, 1)

        loss = self.loss(y_hat, label)
        loss.backward()

        self.optimizer.step()
        self.optimizer.zero_grad()

        return loss.item()


class PathMPNN(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.node_enc = nn.Linear(config.model.n_enc_dim, config.model.model_dim)
        self.edge_enc = nn.Linear(config.model.e_enc_dim, config.model.model_dim)

        self.graph_convs = nn.ModuleList(
            [MPLayer(config.model.node_aggr, config.model.model_dim, config.model.act, config.model.residual)
             for _ in range(config.model.num_layers)])

        self.mlp = MLP([config.model.model_dim * 2, config.model.model_dim, config.model.model_dim // 2, 1])
        self.mlp.reset_parameters()

        self.loss = getattr(nn, config.model.loss)()
        self.optimizer = Lion(self.parameters(), lr=config.model.optimizer.lr, weight_decay=config.model.optimizer.wd)


    def forward(self, batch: Batch):
        nf, e_id, ef = batch.x, batch.edge_index, batch.edge_attr

        nf, ef = self.node_enc(nf), self.edge_enc(ef)

        for graph_conv in self.graph_convs:
            nf = graph_conv(nf, e_id, ef)
        nf = rearrange(nf, '(N C) L -> N C L', N=len(batch))
        sch = rearrange(batch.schedule, '(N C) L -> N C L', N=len(batch))

        sch_nf = torch.cat([rearrange(n[s], 'N C L -> 1 N (C L)') for n, s in zip(nf, sch)])
        sch_nf = rearrange(sch_nf, 'N C L -> (N C) L')

        y_hat = self.mlp(sch_nf)

        loss = self.loss(y_hat, batch.y)
        loss.backward()

        self.optimizer.step()
        self.optimizer.zero_grad()

        return loss.item()
