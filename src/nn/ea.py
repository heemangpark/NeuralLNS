import torch
import torch.nn as nn
from einops import rearrange

from src.nn.mpnn import MPLayer


class EdgeAttention(nn.Module):
    """
    Edge attention module proposed from "Systematic Generalization with Edge Transformers"
    https://arxiv.org/pdf/2112.00578.pdf
    """

    def __init__(self, embed_dim: int):
        super().__init__()

        self.embed_dim = embed_dim
        self.Wq = nn.Linear(embed_dim, embed_dim)
        self.Wk = nn.Linear(embed_dim, embed_dim)
        self.Wv1 = nn.Linear(embed_dim, embed_dim)
        self.Wv2 = nn.Linear(embed_dim, embed_dim)
        self.Wo = nn.Linear(embed_dim, embed_dim)

    def forward(self, ef: torch.Tensor):
        """
        ef: (batch_size, num_nodes, num_nodes, embed_dim)
        """

        v_il = rearrange(self.Wv1(ef), "b i l d -> b i l 1 d")
        v_li = rearrange(self.Wv2(ef), "b l j d -> b 1 l j d")
        v_ilj = v_il * v_li  # [b, n, n, n, d]

        q_il = rearrange(self.Wq(ef), "b i l d -> b i l 1 d")
        k_lj = rearrange(self.Wk(ef), "b l j d -> b 1 l j d")
        qk = (q_il * k_lj).sum(dim=-1, keepdim=True) / self.embed_dim ** 0.5
        a_ilj = torch.softmax(qk, dim=-2)  # softmax over l

        out = self.Wo((v_ilj * a_ilj).sum(dim=-2))  # ef'_ij
        return out


class EdgeAttentionModule(nn.Module):
    def __init__(
            self,
            dim: int,
            act: str,
            node_aggr: str = "add",
            residual: bool = True,
    ):
        super().__init__()
        self.pathConv = EdgeAttention(dim)
        self.graphConv = MPLayer(dim, act, node_aggr=node_aggr, residual=residual)
        self.residual = residual

    def forward(self, path_emb):
        u_path_emb = self.pathConv(path_emb)
        # u_node_emb = self.graphConv(node_emb, edge_index, u_path_emb)

        if self.residual:
            # node_emb = u_node_emb + node_emb
            path_emb = u_path_emb + path_emb
        else:
            # node_emb = u_node_emb
            path_emb = u_path_emb

        return path_emb


if __name__ == "__main__":
    batch = 2
    num_nodes = 3
    embed_dim = 4

    ef = torch.randn(batch, num_nodes, num_nodes, embed_dim)

    Wq = nn.Linear(embed_dim, embed_dim)
    Wk = nn.Linear(embed_dim, embed_dim)
    Wv1 = nn.Linear(embed_dim, embed_dim)
    Wv2 = nn.Linear(embed_dim, embed_dim)

    v1, v2 = Wv1(ef), Wv2(ef)
    q, k = Wq(ef), Wk(ef)

    v_ilj = torch.zeros(batch, num_nodes, num_nodes, num_nodes, embed_dim)
    for i in range(num_nodes):
        for l in range(num_nodes):
            for j in range(num_nodes):
                v_ilj[:, i, l, j] = v1[:, i, l] * v2[:, l, j]

    # Rearrange dimensions of ef using einops
    v_il = rearrange(v1, "b i l d -> b i l 1 d")
    v_lj = rearrange(v2, "b l j d -> b 1 l j d")
    v_ilj2 = v_il * v_lj
    assert torch.allclose(v_ilj, v_ilj2)

    qk = torch.zeros(batch, num_nodes, num_nodes, num_nodes, embed_dim)
    for i in range(num_nodes):
        for j in range(num_nodes):
            for l in range(num_nodes):
                qk[:, i, l, j] = q[:, i, l] * k[:, l, j]
    qk = qk.sum(dim=-1)

    q_il = rearrange(q, "b i l d -> b i l 1 d")
    k_lj = rearrange(k, "b l j d -> b 1 l j d")
    qk2 = (q_il * k_lj).sum(dim=-1)

    assert torch.allclose(qk, qk2)

    ea = EdgeAttention(embed_dim)
    uef = ea(ef)
    print(uef.shape)
