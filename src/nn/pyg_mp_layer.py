import torch
from torch_geometric.nn import MessagePassing
from torch_geometric.nn.models import MLP


class MPLayer(MessagePassing):
    def __init__(self, node_aggr: str, dim: int, residual: bool):
        super(MPLayer, self).__init__(aggr=node_aggr)

        self.message_func = MLP([3 * dim, dim])
        self.update_func = MLP([2 * dim, dim])
        self.residual = residual

    def forward(self, x, edge_index, edge_attr):
        nf = x
        u_nf = self.propagate(x=nf, edge_index=edge_index, edge_attr=edge_attr)
        if self.residual:
            return nf + u_nf
        else:
            return u_nf

    def message(self, x_i, x_j, edge_attr):
        return self.message_func(torch.cat([x_i, x_j, edge_attr], dim=-1))

    def update(self, aggr_msg: torch.tensor, x: torch.tensor):
        return self.update_func(torch.cat([x, aggr_msg], dim=-1))
