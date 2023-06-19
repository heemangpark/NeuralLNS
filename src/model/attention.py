import torch.nn as nn
from torch.nn import MultiheadAttention


class MultiHeadCrossAttention(nn.Module):
    def __init__(self, config):
        super(MultiHeadCrossAttention, self).__init__()

        self.attn_layer = MultiheadAttention(embed_dim=config.embed_dim,
                                             num_heads=config.num_heads,
                                             batch_first=config.batch_first)

        self.WQ = nn.Linear(config.embed_dim, config.embed_dim)
        self.WK = nn.Linear(config.embed_dim, config.embed_dim)
        self.WV = nn.Linear(config.embed_dim, config.embed_dim)

        self.dec = nn.Linear(config.embed_dim, 1)

    def forward(self, query, key, value):
        q = self.WQ(query)
        k = self.WK(key)
        v = self.WV(value)
        attn, _ = self.attn_layer(q, k, v, need_weights=False)

        return self.dec(attn)
