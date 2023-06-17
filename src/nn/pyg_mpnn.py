import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing


class MPLayer(MessagePassing):
    def __init__(self, node_aggr: str, dim: int, act: str, residual: bool):
        super().__init__(aggr=node_aggr)

        self.update_func = nn.Sequential(nn.Linear(2 * dim, dim), getattr(nn, act)())
        self.residual = residual

    def forward(self, x, edge_index, edge_attr):
        nf = x
        u_nf = self.propagate(x=x, edge_index=edge_index, edge_attr=edge_attr)
        if self.residual:
            return nf + u_nf
        else:
            return u_nf

    def update(self, aggr_msg: torch.tensor, x: torch.tensor):
        return self.update_func(torch.cat([x, aggr_msg], dim=-1))
