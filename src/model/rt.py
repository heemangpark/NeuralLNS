import torch
import torch.nn as nn
from einops import rearrange
from lion_pytorch import Lion
from torch_geometric.data import Batch

from src.nn.decoding import get_path_feat_convertor
from src.nn.init_embedding import get_init_embedding
from src.nn.path_init import PathInit
from src.nn.ra import RelationalAttention


class RelationalTransformer(nn.Module):
    def __init__(
            self,
            config,
            problem_type: str = "momd",
    ):
        super().__init__()
        node_dim = config.model.n_enc_dim
        edge_dim = config.model.e_enc_dim
        model_dim = config.model.model_dim
        num_layers = config.model.num_layers
        self.init_emb = get_init_embedding(problem_type)(node_dim, edge_dim, model_dim)
        self.path_init = PathInit(model_dim, model_dim)
        self.path_convs = nn.ModuleList([RelationalAttention(model_dim) for _ in range(num_layers)])
        self.path_feat_convertor = get_path_feat_convertor(problem_type)
        self.dec = nn.Linear(model_dim, 1)

        self.loss = getattr(nn, config.model.loss)()
        self.optimizer = Lion(self.parameters(), lr=config.model.optimizer.lr, weight_decay=config.model.optimizer.wd)

    def forward(self, batch: Batch, test: bool = False) -> torch.Tensor:
        if test:
            with torch.no_grad():
                nf, ef = self.init_emb(batch)
                n = nf.shape[0] // batch.num_graphs
                pf = self.path_init(batch.batch, batch.num_graphs, n, batch.edge_index, ef)

                nf = rearrange(nf, "(b i) d -> b i d", i=n)
                for path_conv in self.path_convs:
                    nf, pf = path_conv(nf, pf)

                # Manipulate tensors for supporting various types of shortest path (-like) problems
                pf = self.path_feat_convertor(pf)
                pf = self.dec(pf)

                # convert into MAPF format
                sch_pf = []
                sch = rearrange(batch.schedule, '(N C) L -> N C L', N=len(batch))
                for p, s in zip(pf, sch):
                    sch_pf_per_batch = torch.stack([p[tuple(_s)] for _s in s]).view(-1)
                    sch_pf.append(sch_pf_per_batch)
                final_pf = torch.stack(sch_pf)
                pred = getattr(torch, 'sum')(final_pf, dim=-1)

                return pred.cpu().detach().numpy()

        else:
            nf, ef = self.init_emb(batch)
            n = nf.shape[0] // batch.num_graphs
            pf = self.path_init(batch.batch, batch.num_graphs, n, batch.edge_index, ef)

            nf = rearrange(nf, "(b i) d -> b i d", i=n)
            for path_conv in self.path_convs:
                nf, pf = path_conv(nf, pf)

            # Manipulate tensors for supporting various types of shortest path (-like) problems
            pf = self.path_feat_convertor(pf)
            pf = self.dec(pf)

            # convert into MAPF format
            sch_pf = []
            sch = rearrange(batch.schedule, '(N C) L -> N C L', N=len(batch))
            for p, s in zip(pf, sch):
                sch_pf_per_batch = torch.stack([p[tuple(_s)] for _s in s]).view(-1)
                sch_pf.append(sch_pf_per_batch)
            final_pf = torch.stack(sch_pf)
            pred = getattr(torch, 'sum')(final_pf, dim=-1)

            loss = self.loss(pred, batch.y)
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()

            return loss.item()
