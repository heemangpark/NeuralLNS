import torch
import torch.nn as nn
from einops import rearrange, repeat


class RelationalAttention(nn.Module):
    def __init__(self, embed_dim: int):
        super().__init__()

        self.embed_dim = embed_dim
        self.Wqn = nn.Linear(embed_dim, embed_dim)
        self.Wkn = nn.Linear(embed_dim, embed_dim)
        self.Wvn = nn.Linear(embed_dim, embed_dim)
        self.Wqe = nn.Linear(embed_dim, embed_dim)
        self.Wke = nn.Linear(embed_dim, embed_dim)
        self.Wve = nn.Linear(embed_dim, embed_dim)

        self.m_func = nn.Sequential(
            nn.Linear(embed_dim * 4, embed_dim),
            nn.ReLU(),
        )

        self.W5 = nn.Linear(embed_dim, embed_dim)
        self.LN1 = nn.LayerNorm(embed_dim)

        self.e_func = nn.Sequential(
            nn.Linear(embed_dim, embed_dim), nn.ReLU(), nn.Linear(embed_dim, embed_dim)
        )  # W6, W7

        self.LN2 = nn.LayerNorm(embed_dim)

    def forward(
            self,
            x: torch.Tensor,
            ef: torch.Tensor,
    ):
        """
        x: (batch_size, num_nodes, embed_dim)
        ef: (batch_size, num_nodes, num_nodes, embed_dim)
        """
        num_nodes = x.shape[1]

        # Update node embedding
        qn, kn, vn = self.Wqn(x), self.Wkn(x), self.Wvn(x)
        qe, ke, ve = self.Wqe(ef), self.Wke(ef), self.Wve(ef)

        q = repeat(qn, "b n d -> b n m d", m=num_nodes) + qe
        k = repeat(kn, "b n d -> b n m d", m=num_nodes) + ke
        v = repeat(vn, "b n d -> b n m d", m=num_nodes) + ve

        z = torch.einsum("b n m d, b n m d -> b n m", q, k)  # [B, N, N]
        z = torch.softmax(z / (self.embed_dim ** 0.5), dim=-1)

        # Update edge embedding
        x = (z[..., None] * v).sum(dim=-2)  # [B, N, D]

        e_ij = ef
        e_ji = rearrange(ef, "b i j d -> b j i d")
        n_i = repeat(x, "b n d -> b n m d", m=num_nodes)
        n_j = repeat(x, "b n d -> b m n d", m=num_nodes)

        m = self.m_func(torch.cat([n_i, n_j, e_ij, e_ji], dim=-1))
        u = self.LN1(self.W5(m) + ef)
        e = self.LN2(self.e_func(u) + u)

        return x, e
