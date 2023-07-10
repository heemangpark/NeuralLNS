import random
from copy import deepcopy

import dgl
import networkx as nx
import numpy as np
import torch


# def grid_to_dgl(world, rand_coord=True, four_dir=True):
#     world = deepcopy(world)
#     m, n = world.shape
#     g = dgl.graph(list())
#     g.add_nodes(m * n)
#
#     if rand_coord:
#         # node position
#         rand_interval_x = torch.rand(m - 1) + .5
#         rand_x = [rand_interval_x[:i].sum() for i in range(m)]
#         rand_x = torch.Tensor(rand_x)
#         rand_x /= rand_x.sum()
#
#         rand_interval_y = torch.rand(n - 1) + .5
#         rand_y = [rand_interval_y[:i].sum() for i in range(n)]
#         rand_y = torch.Tensor(rand_y)
#         rand_y /= rand_y.sum()
#     else:
#         rand_x = torch.Tensor([0 + i / (n - 1) for i in range(m)])
#         rand_y = torch.Tensor([0 + i / (m - 1) for i in range(n)])
#
#     xs = rand_x.repeat(n, 1).reshape(-1)
#     ys = rand_y.flip(-1).repeat(m, 1).T.reshape(-1)
#
#     g.ndata['loc'] = torch.stack([xs, ys], -1)
#     g.ndata['type'] = torch.Tensor(world.reshape(-1, 1))
#
#     # add edge
#     matrix = np.arange(m * n).reshape(m, -1)
#     v_from = matrix[:-1].reshape(-1)
#     v_to = matrix[1:].reshape(-1)
#
#     g.add_edges(v_from, v_to)
#     g.add_edges(v_to, v_from)
#
#     h_from = matrix[:, :-1].reshape(-1)
#     h_to = matrix[:, 1:].reshape(-1)
#
#     g.add_edges(h_from, h_to)
#     g.add_edges(h_to, h_from)
#
#     if not four_dir:
#         dig_from = matrix[:-1, :-1].reshape(-1)
#         dig_to = matrix[1:, 1:].reshape(-1)
#         g.add_edges(dig_from, dig_to)
#         g.add_edges(dig_to, dig_from)
#
#         ddig_from = matrix[1:, :-1].reshape(-1)
#         ddig_to = matrix[:-1, 1:].reshape(-1)
#         g.add_edges(ddig_from, ddig_to)
#         g.add_edges(ddig_to, ddig_from)
#
#     # compute ef
#     g.apply_edges(lambda edges: {'dist': ((edges.src['loc'] - edges.dst['loc']) ** 2).sum(-1).reshape(-1, 1) ** .5})
#
#     # remove obstacle
#     obs_idx = world.reshape(-1).nonzero()[0]
#     g.remove_nodes(obs_idx)
#
#     return g


def valid_graph(size: int, obs: int, fixed: bool):
    if fixed:
        num_obs = int(size ** 2 * obs / 100)
        temp = np.array([False for _ in range(size ** 2 - num_obs)] + [True for _ in range(num_obs)])
        random.shuffle(temp)
        temp = temp.reshape(size, size)
        instance = np.ones((size, size)) * temp
    else:
        instance = np.zeros((size, size))
        obstacle = np.random.random((size, size)) <= obs / 100
        instance[obstacle] = 1

    graph = generate_2d_graph(instance)
    components = [c for c in nx.connected_components(graph)]

    while len(components) != 1:
        if fixed:
            num_obs = int(size ** 2 * obs / 100)
            temp = np.array([False for _ in range(size ** 2 - num_obs)] + [True for _ in range(num_obs)])
            random.shuffle(temp)
            temp = temp.reshape(size, size)
            instance = np.ones((size, size)) * temp
        else:
            instance = np.zeros((size, size))
            obstacle = np.random.random((size, size)) <= obs / 100
            instance[obstacle] = 1
        graph = generate_2d_graph(instance)
        components = [c for c in nx.connected_components(graph)]

    return instance, graph


def generate_2d_graph(instance):
    instance = deepcopy(instance)
    m, n = instance.shape[0], instance.shape[1]
    g = nx.grid_2d_graph(m, n)

    for r, c in zip(instance.nonzero()[0], instance.nonzero()[1]):
        g.remove_node((r, c))

    return g


def sch_to_dgl(assign_idx, coord_schedule, size):
    # loc_idx = dict(zip(list(tuple(v) for v in nx.get_node_attributes(grid_graph, 'loc').values()),
    #                    nx.get_node_attributes(grid_graph, 'loc').keys()))
    coords = [item for sublist in coord_schedule for item in sublist]
    norm_coords = [[c[0] / size, c[1] / size] for c in coords]
    sch_nx = nx.complete_graph(len(norm_coords))
    nx.set_node_attributes(sch_nx, dict(zip(sch_nx.nodes, norm_coords)), 'coord')

    types = []
    for c in coord_schedule:
        types.extend([1] + [2 for _ in range(len(c) - 1)])
    nx.set_node_attributes(sch_nx, dict(zip(sch_nx.nodes, types)), 'type')

    graph_assign_id = []
    for idx in assign_idx:
        graph_assign_id.extend([-1] + idx)
    nx.set_node_attributes(sch_nx, dict(zip(sch_nx.nodes, graph_assign_id)), 'idx')
    nx.set_node_attributes(sch_nx, dict(zip(sch_nx.nodes, range(sch_nx.number_of_nodes()))), 'graph_idx')
    # astar = [nx.astar_path_length(grid_graph, loc_idx[tuple(coords[i])],
    #                               loc_idx[tuple(coords[j])]) for i, j in sch_nx.edges]
    norm_dist = [np.abs(coords[i][0] - coords[j][0]) + np.abs(coords[i][1] - coords[j][1]) / size
                 for i, j in sch_nx.edges]
    nx.set_edge_attributes(sch_nx, dict(zip(sch_nx.edges, norm_dist)), 'dist')
    nx.set_edge_attributes(sch_nx, dict(zip(sch_nx.edges, [0] * sch_nx.number_of_edges())), 'connected')

    start_node_idx = 0
    for schedule in coord_schedule:
        n_schedule = len(schedule)
        node_indices = range(start_node_idx, start_node_idx + n_schedule)
        for i, j in zip(node_indices[:-1], node_indices[1:]):
            sch_nx.edges[i, j]['connected'] = 1
        start_node_idx += n_schedule

    graph = dgl.from_networkx(sch_nx.to_directed(),
                              node_attrs=['coord', 'type', 'idx', 'graph_idx'],
                              edge_attrs=['dist', 'connected'])
    graph.edata['dist'] = graph.edata['dist'].to(torch.float32)

    return graph


def partition_8():
    obs = [(0, 3), (1, 3), (2, 3),
           (4, 0), (4, 1), (4, 2),
           (5, 4), (6, 4), (7, 4),
           (3, 5), (3, 6), (3, 7)]

    grid = np.zeros((8, 8))
    graph = nx.grid_2d_graph(8, 8)

    for o in obs:
        grid[o] = 1
        graph.remove_node(o)

    return grid, graph
