import torch
import torch.nn as nn
from einops import rearrange
from lion_pytorch import Lion
from torch_geometric.data import Batch

from src.nn.decoding import get_path_feat_convertor
from src.nn.init_embedding import get_init_embedding
from src.nn.path_init import PathInit
from src.nn.triplet_conv import TripletConvModule


class TripletGNN(nn.Module):
    def __init__(
            self,
            config,
            problem_type: str = "momd",
            sub_dim: int = 8,
            act: str = "ReLU",
            path_aggr: str = "max",
            node_aggr: str = "add",
            residual: bool = True,
    ):
        super().__init__()
        node_dim = config.model.n_enc_dim
        edge_dim = config.model.n_enc_dim
        model_dim = config.model.model_dim
        num_layers = config.model.num_layers

        self.init_emb = get_init_embedding(problem_type)(node_dim, edge_dim, model_dim)
        self.path_init = PathInit(model_dim, model_dim)

        self.num_layers = num_layers
        self.path_convs = nn.ModuleList(
            [
                TripletConvModule(model_dim, act, sub_dim, path_aggr, node_aggr, residual)
                for _ in range(num_layers)
            ]
        )
        self.path_feat_convertor = get_path_feat_convertor(problem_type)
        self.dec = nn.Linear(model_dim, edge_dim)

        self.loss = getattr(nn, 'L1Loss')()
        self.optimizer = Lion(self.parameters(), lr=1e-4, weight_decay=1e-2)

    def forward(self, batch: Batch, test: bool = False) -> torch.Tensor:
        nf, ef = self.init_emb(batch)
        n = nf.shape[0] // batch.num_graphs
        pf = self.path_init(batch.batch, batch.num_graphs, n, batch.edge_index, ef)

        for path_conv in self.path_convs:
            nf, pf = path_conv(nf, pf, batch.edge_index)

        # Manipulate tensors for supporting various types of shortest path (-like) problems
        pf = self.path_feat_convertor(pf)
        pf = self.dec(pf)

        # convert into MAPF format
        sch_pf = []
        sch = rearrange(batch.schedule, '(N C) L -> N C L', N=len(batch))
        for p, s in zip(pf, sch):
            sch_pf_per_batch = torch.stack([p[tuple(_s)] for _s in s]).view(-1)
            sch_pf.append(sch_pf_per_batch)  # batch x num_sch
        final_pf = torch.stack(sch_pf)
        pred = getattr(torch, 'mean')(final_pf, dim=-1)

        if test:
            return pred

        else:
            loss = self.loss(pred, batch.y)
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()

            return loss.item()
