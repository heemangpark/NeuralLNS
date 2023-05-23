import dgl
import torch
import torch.nn as nn


class MPLayers(nn.Module):
    def __init__(self, in_dim, out_dim, aggr: str = 'min'):
        super(MPLayers, self).__init__()
        self.node_W = nn.Sequential(nn.Linear(out_dim + in_dim, out_dim, bias=False), nn.LeakyReLU())
        self.edge_W = nn.Sequential(nn.Linear(in_dim * 2 + 1, out_dim, bias=False), nn.LeakyReLU())
        self.aggr = aggr

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
        feature = torch.concat([u, e_u_v, v], -1)
        msg = self.edge_W(feature)

        return {'msg': msg}

    def reduce_func(self, nodes):
        if self.aggr == 'sum':
            msg = nodes.mailbox['msg'].sum(1).values
        elif self.aggr == 'mean':
            msg = nodes.mailbox['msg'].mean(1).values
        elif self.aggr == 'max':
            msg = nodes.mailbox['msg'].max(1).values
        elif self.aggr == 'min':
            msg = nodes.mailbox['msg'].min(1).values
        else:
            raise NotImplementedError

        return {'aggr_msg': msg}

    def apply_node_func(self, nodes):
        node_feat = torch.concat([nodes.data['nf'], nodes.data['aggr_msg']], -1)
        node_feat_p = self.node_W(node_feat)

        return {'nf_p': node_feat_p}


class MPGNN(nn.Module):
    def __init__(self, in_dim, out_dim, embedding_dim, n_layers, aggr: str, residual: bool):
        super(MPGNN, self).__init__()
        self.is_residual = residual

        ins = [in_dim] + [embedding_dim] * (n_layers - 1)
        outs = [embedding_dim] * (n_layers - 1) + [out_dim]

        gnn_layers = []
        for i, o in zip(ins, outs):
            gnn_layers.append(MPLayers(i, o, aggr))
        self.gnn_layers = nn.ModuleList(gnn_layers)

    def forward(self, graph: dgl.DGLGraph, node_feat: torch.Tensor):
        node_feat_p = node_feat
        for layer in self.gnn_layers:
            node_feat_p = layer(graph, node_feat_p)

        if self.is_residual:
            return node_feat + node_feat_p
        else:
            return node_feat_p


class FC_Edges(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(FC_Edges, self).__init__()
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
