from math import sqrt
from typing import Any

import torch
import torch.nn as nn
from einops import repeat, rearrange
from lion_pytorch import Lion
from torch_geometric.data import Batch
from torch_geometric.nn import MessagePassing
from torch_geometric.nn.models import MLP


def path_aggr(t1, t2, mode: str):
    if mode == "max":
        "performs max pooling"
        aggr, _ = torch.max(torch.stack([t1, t2], 0), 0)
    elif mode == "mean":
        "performs mean pooling"
        aggr = torch.mean(torch.stack([t1, t2], 0), 0)
    elif mode == "add":
        aggr = t1 + t2
    else:
        raise NotImplementedError(f"Mode {mode} is not implemented.")

    return aggr


class MOMDInitEmbedding(nn.Module):
    def __init__(self, node_dim: int, edge_dim: int, embed_dim: int):
        super().__init__()
        self.node_emb = nn.Linear(node_dim, embed_dim)
        self.edge_emb = nn.Linear(edge_dim, embed_dim)

    def forward(self, batch: Batch) -> tuple[Any, Any]:
        nf, ef = batch.x, batch.edge_attr
        node_emb = self.node_emb(nf)
        edge_emb = self.edge_emb(ef)
        return node_emb, edge_emb


class MPLayer(MessagePassing):
    def __init__(self, dim: int, act: str, residual: bool = True, node_aggr: str = "add"):
        super().__init__(aggr=node_aggr)

        self.node_model = nn.Sequential(nn.Linear(2 * dim, dim), getattr(nn, act)())
        self.residual = residual

    def forward(self, x, edge_index, edge_attr):
        return self.propagate(edge_index, x=x, edge_attr=edge_attr)

    def update(self, aggr_msg: torch.tensor, x: torch.tensor):
        unf = self.node_model(torch.cat([x, aggr_msg], dim=-1))
        return unf


class PathConv(nn.Module):
    def __init__(self, dim: int, act: str, aggr: str = "max"):
        super().__init__()

        self.dim = dim
        self.W_hz = nn.Linear(dim, dim)
        self.W_zz = nn.Linear(3 * dim, dim)
        self.act = getattr(nn, act)()
        self.aggr = aggr

    def forward(self, node_emb, path_emb):
        """
        Args:
            node_emb: (torch.tensor) node embedding [batch x num. nodes, dim]
            path_emb: (torch.tensor) path embedding  [batch, num. nodes, num nodes, dim]

        Returns:
            path_emb: Updated path embedding [batch, num. nodes, num nodes, dim]
        """

        n = path_emb.shape[1]
        z = self.W_hz(path_emb)

        # Summation
        if self.aggr == "sum":
            z_conv = torch.einsum("bikd,bkjd->bijd", z, z) / sqrt(self.dim)
        elif self.aggr == "max":
            # Max
            z_ik = repeat(z, "b i j d -> b i k j d", k=n)
            z_kj = repeat(z, "b i j d -> b k j i d", k=n)
            z_conv, _ = (z_ik + z_kj).max(dim=3)
        else:
            raise NotImplementedError(f"Mode {self.aggr} is not implemented.")

        z = path_aggr(z_conv, z, mode=self.aggr)

        node_emb = rearrange(node_emb, "(b n) d -> b n d", n=n)
        n_i = repeat(node_emb, "b n d -> b n i d", i=n)
        n_j = repeat(node_emb, "b n d -> b i n d", i=n)
        z = torch.cat([n_i, z, n_j], dim=-1)
        z = self.act(self.W_zz(z))
        return z


class PathConvModule(nn.Module):
    def __init__(
            self,
            dim: int,
            act: str,
            path_aggr: str = "max",
            node_aggr: str = "add",
            residual: bool = True,
    ):
        super().__init__()
        self.pathConv = PathConv(dim, act, aggr=path_aggr)
        self.graphConv = MPLayer(dim, act, node_aggr=node_aggr, residual=residual)
        self.residual = residual

    def forward(self, node_emb, path_emb, edge_index):
        u_path_emb = self.pathConv(node_emb, path_emb)
        u_node_emb = self.graphConv(node_emb, edge_index, u_path_emb)

        if self.residual:
            node_emb = u_node_emb + node_emb
            path_emb = u_path_emb + path_emb
        else:
            node_emb = u_node_emb
            path_emb = u_path_emb

        return node_emb, path_emb


class PathInit(nn.Module):
    def __init__(self, d_edge: int, d_model: int):
        super().__init__()

        self.d_edge = d_edge
        self.d_model = d_model
        self.self_loop_token = nn.Parameter(torch.randn(d_model))
        self.not_conn_token = nn.Parameter(torch.randn(d_model))
        self.edge_enc = nn.Linear(d_edge, d_model)

    def forward(
            self,
            batch,
            batch_size: int,
            num_nodes: int,
            edge_index: torch.Tensor,
            edge_attr: torch.Tensor,
    ):
        b, n = batch_size, num_nodes
        path_emb = torch.ones(b, n, n, self.d_model, device=batch.device) * self.not_conn_token

        edge_batches = batch[edge_index[0]]
        source_nodes = edge_index[0] % n
        target_nodes = edge_index[1] % n

        path_emb[edge_batches, source_nodes, target_nodes] = self.edge_enc(edge_attr)

        # self-loop
        diag_idx = torch.arange(n)
        path_emb[:, diag_idx, diag_idx] = self.self_loop_token
        return path_emb


class PathGNN(nn.Module):
    def __init__(self, config, readout_dir: int, readout_type: str):
        super().__init__()
        self.init_emb = MOMDInitEmbedding(config.model.n_enc_dim, config.model.e_enc_dim, config.model.model_dim)
        self.path_init = PathInit(config.model.model_dim, config.model.model_dim)

        self.num_layers = config.model.num_layers
        self.path_convs = nn.ModuleList(
            [
                PathConvModule(config.model.model_dim, config.model.act,
                               config.model.path_aggr, config.model.node_aggr, config.model.residual)
                for _ in range(config.model.num_layers)
            ]
        )

        self.dec = nn.Linear(config.model.model_dim, config.model.e_enc_dim)

        self.readout_dir = readout_dir
        self.readout_type = readout_type
        if self.readout_type == 'mlp':
            self.mlp = MLP([5, 2, 2, 1])

        self.loss = getattr(nn, config.model.loss)()
        self.optimizer = Lion(self.parameters(), lr=config.model.optimizer.lr, weight_decay=config.model.optimizer.wd)

    def forward(self, batch: Batch, test: bool = False) -> torch.Tensor:
        nf, ef = self.init_emb(batch)
        n = batch.batch.shape[0] // len(batch)
        pf = self.path_init(batch.batch, len(batch), n, batch.edge_index, ef)

        for path_conv in self.path_convs:
            nf, pf = path_conv(nf, pf, batch.edge_index)

        feat = self.dec(pf).squeeze(-1)
        sch = rearrange(batch.schedule, '(N C) L -> N C L', N=len(batch))

        sch_pf = []
        if self.readout_dir == 1:  # A -> T
            for p, s in zip(feat, sch):
                sch_pf_per_batch = torch.stack([p[tuple(_s)] for _s in s])
                sch_pf.append(sch_pf_per_batch)  # batch x num_sch
        elif self.readout_dir == 2:  # A <-> T
            for p, s in zip(feat, sch):
                rev_s = torch.stack((s[:, 1], s[:, 0]), dim=-1)
                batch_sch_pf = torch.stack([p[tuple(_s)] for _s in s])
                batch_rev_sch_pf = torch.stack([p[tuple(_s)] for _s in rev_s])
                sch_pf.append(torch.stack((batch_sch_pf, batch_rev_sch_pf), dim=-1))  # batch x num_sch x 2
                # TODO: pre-readout not implemented yet

        final_pf = torch.stack(sch_pf)
        if (self.readout_type == 'sum') or (self.readout_type == 'mean'):
            y_hat = getattr(torch, self.readout_type)(final_pf, dim=-1)
        elif (self.readout_type == 'max') or (self.readout_type == 'min'):
            y_hat = getattr(torch, self.readout_type)(final_pf, dim=-1)[0]
        else:
            y_hat = self.mlp(final_pf).squeeze(-1)

        if test:
            return y_hat

        else:
            loss = self.loss(y_hat, batch.y)
            loss.backward()
            nn.utils.clip_grad_norm_(self.parameters(), 1.0)
            self.optimizer.step()
            self.optimizer.zero_grad()

            return loss.item()
