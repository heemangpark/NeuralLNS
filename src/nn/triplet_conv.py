import torch
import torch.nn as nn
from einops import rearrange, repeat

from src.nn.mpnn import MPLayer


class TripletConv(nn.Module):
    def __init__(self, dim: int, act: str, aggr: str = "max", sub_dim: int = 8):
        super().__init__()

        self.node_proj = nn.Linear(dim, sub_dim)
        self.edge_proj = nn.Linear(dim, sub_dim)
        self.triplet_func = nn.Sequential(nn.Linear(6 * sub_dim, sub_dim), getattr(nn, act)())
        self.out_func = nn.Sequential(nn.Linear(sub_dim, dim), getattr(nn, act)())
        self.aggr = aggr

    def forward(self, node_emb, path_emb):
        """
        Args:
            node_emb: (torch.tensor) node embedding [batch x num nodes, dim]
            path_emb: (torch.tensor) path embedding  [batch, num nodes, num nodes, dim]

        Returns:
            path_emb: Updated path embedding [batch, num. nodes, num nodes, dim]
        """

        num_nodes = path_emb.shape[1]

        node_emb = self.node_proj(node_emb)
        node_emb = rearrange(node_emb, "(b i) d -> b i d", i=num_nodes)
        path_emb = self.edge_proj(path_emb)

        n_i = repeat(node_emb, "b i d -> b i j k d", j=num_nodes, k=num_nodes)
        n_j = repeat(node_emb, "b i d -> b j i k d", j=num_nodes, k=num_nodes)
        n_k = repeat(node_emb, "b i d -> b k j i d", j=num_nodes, k=num_nodes)

        ef_ik = repeat(path_emb, "b i j d -> b i k j d", k=num_nodes)
        ef_kj = repeat(path_emb, "b i j d -> b k j i d", k=num_nodes)
        ef_ij = repeat(path_emb, "b i j d -> b i j k d", k=num_nodes)

        # t_ijk = [batch, num_nodes, num_nodes, num_nodes, 6 * sub_dim]
        t_ijk = torch.cat([n_i, n_j, n_k, ef_ik, ef_kj, ef_ij], dim=-1)

        # t_ijk = [batch, num_nodes, num_nodes, num_nodes, sub_dim]
        t_ijk = self.triplet_func(t_ijk)

        if self.aggr == "max":
            t_ij, _ = torch.max(t_ijk, dim=-2)

        out = self.out_func(t_ij)  # [batch, num_nodes, num_nodes, dim]
        return out


class TripletConvModule(nn.Module):
    def __init__(
        self,
        dim: int,
        act: str,
        sub_dim: int = 8,
        path_aggr: str = "max",
        node_aggr: str = "add",
        residual: bool = True,
    ):
        super().__init__()
        self.pathConv = TripletConv(dim, act, aggr=path_aggr, sub_dim=sub_dim)
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
