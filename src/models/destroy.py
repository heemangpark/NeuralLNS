import copy

import dgl
import torch
from lion_pytorch import Lion
from torch import nn as nn

from src.models.mpnn import MPNN


def destroyGraph(graph, destroy, device):
    discon, con = torch.LongTensor([0]).to(device), torch.LongTensor([1]).to(device)
    g = copy.deepcopy(graph)

    to_con_f = []
    to_con_t = []
    to_discon_f = []
    to_discon_t = []

    for d in destroy:
        n_id = g.nodes()[g.ndata['idx'] == d].item()
        prev_id, next_id = n_id - 1, n_id + 1

        if (next_id == g.number_of_nodes()) or (g.ndata['type'][next_id] == 1):
            to_discon_f.extend([prev_id, n_id])
            to_discon_t.extend([n_id, prev_id])
        else:
            if (torch.Tensor(destroy) == g.ndata['idx'][next_id].item()).sum() > 0:
                pass
            else:
                to_con_f.extend([prev_id, next_id])
                to_con_t.extend([next_id, prev_id])
            to_discon_f.extend([prev_id, n_id, next_id, n_id])
            to_discon_t.extend([n_id, prev_id, n_id, next_id])

    g.edges[to_con_f, to_con_t].data['connected'] *= 0
    g.edges[to_con_f, to_con_t].data['connected'] += con

    g.edges[to_discon_f, to_discon_t].data['connected'] *= 0
    g.edges[to_discon_f, to_discon_t].data['connected'] += discon

    return g


def destroyBatchGraph(graphs: list, destroy, device=None):
    discon, con = torch.LongTensor([0]).to(device), torch.LongTensor([1]).to(device)
    to_con_f = []
    to_con_t = []
    to_discon_f = []
    to_discon_t = []
    node_cnt = 0

    for _g, _destroy in zip(graphs, destroy):
        g = copy.deepcopy(_g)
        for d in _destroy:
            n_id = g.nodes()[g.ndata['idx'] == d].item()
            prev_id = n_id - 1
            next_id = n_id + 1

            n_id_b = n_id + node_cnt
            prev_id_b = prev_id + node_cnt
            next_id_b = next_id + node_cnt

            if (next_id == g.number_of_nodes()) or (g.ndata['type'][next_id] == 1):
                to_discon_f.extend([prev_id_b, n_id_b])
                to_discon_t.extend([n_id_b, prev_id_b])
            else:
                if (torch.Tensor(_destroy) == g.ndata['idx'][next_id].item()).sum() > 0:
                    pass
                else:
                    to_con_f.extend([prev_id_b, next_id_b])
                    to_con_t.extend([next_id_b, prev_id_b])
                to_discon_f.extend([prev_id_b, n_id_b, next_id_b, n_id_b])
                to_discon_t.extend([n_id_b, prev_id_b, n_id_b, next_id_b])

        node_cnt += g.number_of_nodes()

    batched_g = dgl.batch(graphs)

    batched_g.edges[to_con_f, to_con_t].data['connected'] *= 0
    batched_g.edges[to_con_f, to_con_t].data['connected'] += con

    batched_g.edges[to_discon_f, to_discon_t].data['connected'] *= 0
    batched_g.edges[to_discon_f, to_discon_t].data['connected'] += discon

    return batched_g


class Destroy(nn.Module):
    def __init__(self, cfg: dict):
        super(Destroy, self).__init__()

        aggr, delta, dim, num_layers, readout, _ = cfg.model.values()
        lr, wd = cfg.optimizer.values()

        self.gnn = MPNN(in_dim=dim, out_dim=dim, embedding_dim=dim, n_layers=num_layers,
                        aggr=aggr, delta=delta, residual=True)
        self.readout = getattr(torch, readout)
        self.Wxz = nn.Linear(2, dim)
        self.Why = nn.Linear(dim, 1)

        self.loss = nn.L1Loss()
        self.optimizer = Lion(self.parameters(), lr=lr, weight_decay=wd)

        self.to(cfg.device)

    def forward(self, graphs: dgl.DGLGraph, y: torch.Tensor):
        # destroy_num = len(destroys[0])
        # nf = self.node_W(graphs.ndata['coord'])
        # next_nf = self.gnn(graphs, nf)
        # ef = self._get_edge_embedding(graphs, next_nf)
        #
        # unbatched_graphs = dgl.unbatch(graphs)
        # gs_to_destroy = []
        # destroy_list = []
        # for g, destroy in zip(unbatched_graphs, destroys):
        #     gs_to_destroy.extend([g] * len(destroy))
        #     destroy_list.extend(list(destroys[0].keys()))
        # destroyedGraph = destroyBatchGraph(gs_to_destroy, destroy_list, device)
        #
        # des_src, des_dst = [], []
        # node_cnt = 0
        # for i, dg in enumerate(dgl.unbatch(destroyedGraph)):
        #     des_src.extend([dg.edges()[0][dg.edata['connected'] == 1] + node_cnt])
        #     des_dst.extend([dg.edges()[1][dg.edata['connected'] == 1] + node_cnt])
        #     if i % destroy_num == 0 and i > 0:
        #         node_cnt += dg.number_of_nodes()
        #
        # src_idx = torch.cat(des_src)
        # dst_idx = torch.cat(des_dst)
        # mask = graphs.edge_ids(src_idx, dst_idx)
        # input_ef = ef[mask].reshape(batch_num, len(destroys[0]), -1, ef.shape[-1])
        # input_ef = torch.sum(input_ef, -2)
        # pred = self.mlp(input_ef)
        # pred = pred.squeeze(-1)
        #
        # x1, x2 = pred[:, :5], pred[:, 5:]
        # sign = torch.ones(batch_num, 5).to(self.device)

        # cost = torch.Tensor([list(d.values()) for d in destroys]).to(device)
        # baseline = torch.tile(torch.mean(cost, dim=-1).view(-1, 1), dims=(1, 10)).detach()
        # loss = self.loss(pred.log(), cost)
        # loss = torch.mean(-(cost - baseline) * torch.log(pred + 1e-5))
        # loss = torch.mean(-cost * torch.log(pred + 1e-5))
        b, k = y.shape
        x = graphs.ndata['coord']
        z = self.Wxz(x)
        z_p = self.gnn(graphs, z).view(b, -1, k, z.shape[-1])
        h = self.readout(z_p, dim=1)
        y_hat = self.Why(h).view(b, k)

        loss = self.loss(y_hat, y)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def act(self, graph: dgl.DGLHeteroGraph, candidates: list):
        b = len(candidates)
        graphs = dgl.batch([dgl.node_subgraph(graph, list(set(range(graph.num_nodes())) - set(c))) for c in candidates])
        x = graphs.ndata['coord']
        z = self.Wxz(x)
        z_p = self.gnn(graphs, z).view(b, -1, z.shape[-1])
        h = self.readout(z_p, dim=1)
        pred = self.Why(h)

        return list(candidates[torch.argmax(pred).item()])
