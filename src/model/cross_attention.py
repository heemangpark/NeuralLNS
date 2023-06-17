import torch.nn as nn
from torch.nn import MultiheadAttention


class CrossAttention(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, batch_first: bool = True):
        super(CrossAttention, self).__init__()

        self.attn_layer = MultiheadAttention(embed_dim=embed_dim,
                                             num_heads=num_heads,
                                             batch_first=batch_first)

        self.WQ = nn.Linear(embed_dim, embed_dim)
        self.WK = nn.Linear(embed_dim, embed_dim)
        self.WV = nn.Linear(embed_dim, embed_dim)

        self.dec = nn.Linear(embed_dim, 1)

    def forward(self, query, key, value):
        q = self.WQ(query)
        k = self.WK(key)
        v = self.WV(value)
        attn, _ = self.attn_layer(q, k, v, need_weights=False)

        return self.dec(attn)
