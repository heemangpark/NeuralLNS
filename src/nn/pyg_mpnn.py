import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing


class MPLayer(MessagePassing):
    def __init__(self, node_aggr: str, dim: int, act: str, residual: bool):
        super().__init__(aggr=node_aggr)

        self.node_model = nn.Sequential(nn.Linear(2 * dim, dim), getattr(nn, act)())
        self.residual = residual

    def forward(self, x, edge_index, edge_attr):
        return self.propagate(x=x, edge_index=edge_index, edge_attr=edge_attr)

    def update(self, aggr_msg: torch.tensor, x: torch.tensor):
        unf = self.node_model(torch.cat([x, aggr_msg], dim=-1))
        return unf
