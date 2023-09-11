import dgl
import torch
import torch.nn as nn


class MPNNLayer(nn.Module):
    def __init__(self, in_dim, out_dim, aggr, delta):
        super(MPNNLayer, self).__init__()
        self.node_W = nn.Sequential(nn.Linear(out_dim + in_dim, out_dim, bias=False), nn.LeakyReLU())
        self.edge_W = nn.Sequential(nn.Linear(in_dim * 2 + 1, out_dim, bias=False), nn.LeakyReLU())
        self.aggr = aggr
        self.delta = delta

    def forward(self, g: dgl.DGLGraph, node_feat):
        g.ndata['nf'] = node_feat
        g.update_all(message_func=self.message_func,
                     reduce_func=self.reduce_func,
                     apply_node_func=self.apply_node_func)
        node_feat_p = g.ndata.pop('nf_p')
        g.ndata.pop('nf')

        return node_feat_p

    def message_func(self, edges):
        u, v = edges.src['nf'], edges.dst['nf']
        e_u_v = edges.data['dist'].view(-1, 1)
        edge_mask = edges.data['dist'] < self.delta
        feature = torch.concat([u, e_u_v, v], -1)
        msg = self.edge_W(feature)

        return {'msg': msg, 'mask': edge_mask}

    def reduce_func(self, nodes):
        mask = nodes.mailbox['mask'].unsqueeze(-1)
        message = nodes.mailbox['msg']
        masked_msg = message * mask
        if self.aggr == 'max':
            msg = masked_msg.max(1).values
        elif self.aggr == 'min':
            masked_msg += (masked_msg == 0) * 99999
            msg = masked_msg.min(1).values
        elif self.aggr == 'sum':
            msg = masked_msg.sum(1)
        elif self.aggr == 'mean':
            msg = masked_msg.sum(1) / mask.sum(1)
        else:
            raise NotImplementedError

        return {'aggr_msg': msg}

    def apply_node_func(self, nodes):
        node_feat = torch.concat([nodes.data['nf'], nodes.data['aggr_msg']], -1)
        node_feat_p = self.node_W(node_feat)

        return {'nf_p': node_feat_p}


class MPNN(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, embedding_dim: int, n_layers: int,
                 aggr: str, delta: int, residual: bool):
        super(MPNN, self).__init__()

        ins = [in_dim] + [embedding_dim] * (n_layers - 1)
        outs = [embedding_dim] * (n_layers - 1) + [out_dim]

        self.layers = nn.ModuleList([MPNNLayer(i, o, aggr, delta) for i, o in zip(ins, outs)])
        self.is_residual = residual

    def forward(self, graph: dgl.DGLGraph, node_feat: torch.Tensor):
        if self.is_residual:
            for layer in self.layers:
                node_feat_p = layer(graph, node_feat)
                node_feat += node_feat_p
        else:
            for layer in self.layers:
                node_feat = layer(graph, node_feat)

        return node_feat


class CompleteEdges(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(CompleteEdges, self).__init__()
        self.edge_W = nn.Sequential(nn.Linear(in_dim, out_dim, bias=False), nn.LeakyReLU())

    def forward(self, graph: dgl.DGLGraph, node_feat):
        graph.ndata['nf'] = node_feat
        graph.apply_edges(self.message_func)
        msg = graph.edata.pop('msg')
        graph.ndata.pop('nf')

        return msg

    def message_func(self, edges):
        u, v = edges.src['nf'], edges.dst['nf']
        feature = torch.concat([u, v], -1)
        msg = self.edge_W(feature)

        return {'msg': msg}
