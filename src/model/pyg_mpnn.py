import torch
import torch.nn as nn
from einops import rearrange
from lion_pytorch import Lion
from torch_geometric.data import Batch
from torch_geometric.nn.models import MLP

from src.nn.pyg_mp_layer import MPLayer


class MPNN(nn.Module):
    def __init__(self, config):
        super().__init__()
        model_config = config.model

        self.node_enc = nn.Linear(model_config.n_enc_dim, model_config.model_dim)
        self.edge_enc = nn.Linear(model_config.e_enc_dim, model_config.model_dim)

        self.graph_convs = nn.ModuleList(
            [MPLayer(model_config.node_aggr, model_config.model_dim, model_config.act, model_config.residual)
             for _ in range(model_config.num_layers)])

        self.mlp = MLP([model_config.model_dim * 2, model_config.model_dim, model_config.model_dim // 2, 1])
        self.mlp.reset_parameters()

        self.loss = getattr(nn, model_config.loss)()
        self.optimizer = Lion(self.parameters(), lr=model_config.optimizer.lr, weight_decay=model_config.optimizer.wd)

        self.to(config.device)

    def forward(self, batch: Batch):
        device = batch.batch.device

        nf, e_id, ef = batch.x, batch.edge_index, batch.edge_attr
        nf, ef = self.node_enc(nf), self.edge_enc(ef)

        for graph_conv in self.graph_convs:
            nf = graph_conv(nf, e_id, ef)

        nf = rearrange(nf, '(N C) L -> N C L', N=len(batch))
        agent_nf, task_nf = nf[:, -10:-5, :], nf[:, -5:, :]
        nf = rearrange(torch.cat((agent_nf, task_nf), -1), 'N C L -> (N C) L')

        y_hat = self.mlp(nf)
        label = torch.Tensor(batch.y).view(-1, 1).to(device)

        # num_nodes = batch.ptr.cpu().numpy()
        # total_types = []
        # for s, e in zip(num_nodes[:-1], num_nodes[1:]):
        #     total_types.extend(batch.type[s:e].cpu().numpy())
        #
        # nf = rearrange(nf[np.array(total_types) != -1], '(N C) L -> N C L', N=len(batch))
        # nf_idx = rearrange(batch.type[batch.type != -1], '(N C) -> N C', N=len(batch)).cpu().numpy()
        # for i in range(len(batch)):
        #     nf_idx[i] += i * len(batch)
        #
        # sort_keys = []
        # for i in range(len(batch) * 10):
        #     sort_keys.append(tuple(np.argwhere(nf_idx == i)[0]))
        #
        # temp_tensor = torch.zeros_like(torch.cat((nf[sort_keys[0]], nf[sort_keys[0]])).unsqueeze(0))
        # for i in range(len(batch)):
        #     for j in range(5):
        #         feat = torch.cat((nf[sort_keys[i + j]], nf[sort_keys[i + j + 5]])).unsqueeze(0)
        #         temp_tensor = torch.cat((temp_tensor, feat))
        # final_nf = temp_tensor[1:]

        # b_nf = rearrange(nf, '(N C) L -> N C L', N=len(batch))
        # cat_b_nf = rearrange(b_nf, 'N (C1 C2) L -> N C2 (C1 L)', C1=2)
        #
        # pred = self.mlp(rearrange(cat_b_nf, 'N C L -> (N C) L'))
        # y_hat = rearrange(pred, '(N1 N2) C -> N1 (N2 C)', N1=len(batch))
        # label = rearrange(batch.y, '(N C) -> N C', N=len(batch))

        loss = self.loss(y_hat, label)
        loss.backward()

        self.optimizer.step()
        self.optimizer.zero_grad()

        return loss.item()
