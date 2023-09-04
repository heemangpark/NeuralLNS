import torch
import torch.nn as nn


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
