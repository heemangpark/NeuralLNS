import os
import pickle
import random
import sys
from pathlib import Path

import numpy as np
from tqdm import trange

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from utils.graph import valid_graph
from utils.seed import seed_everything

curr_path = os.path.realpath(__file__)
scenario_dir = os.path.join(Path(curr_path).parent.parent, 'datas/scenarios')


def save_scenarios(itrs: int, size: int, obs: int, a: int, t: int,
                   seed: int, train: bool):
    seed_everything(seed)

    if train:
        dir = scenario_dir + '/{}{}{}_{}_{}/'.format(size, size, obs, a, t)
    else:
        dir = scenario_dir + '/{}{}{}_{}_{}_eval/'.format(size, size, obs, a, t)

    grid, graph = valid_graph(size, obs)

    for itr in trange(itrs):
        grid_idx = list(range(len(graph)))
        a_idx = random.sample(grid_idx, a)
        grid_idx = list(set(grid_idx) - set(a_idx))
        t_idx = random.sample(grid_idx, t)

        a_coord = np.array(graph.nodes())[a_idx].tolist()
        t_coord = np.array(graph.nodes())[t_idx].tolist()
        data = [grid, graph, a_coord, t_coord]

        try:
            if not os.path.exists(dir):
                os.makedirs(dir)
        except OSError:
            print("Error: Cannot create the directory.")
        with open(dir + 'scenario_{}.pkl'.format(itr), 'wb') as f:
            pickle.dump(data, f)


# def task_only_scenarios(itrs: int,
#                         size: int,
#                         obs: int,
#                         t: int,
#                         seed: int):
#     seed_everything(seed)
#
#     dir = scenario_dir + '/test_destroy/'
#     map, map_graph = valid_graph(size, obs)
#
#     for itr in trange(itrs):
#         empty_idx = list(range(len(map_graph)))
#         task_graph_id, task_coord = [], []
#         for i in range(t):
#             task_idx = random.sample(empty_idx, 1)[0]
#             empty_idx.remove(task_idx)
#             t_g_id = list(map_graph.nodes())[task_idx]
#             t_coord = map_graph.nodes[t_g_id]['loc']
#             task_graph_id.append(t_g_id)
#             task_coord.append(t_coord)
#
#         y = torch.LongTensor([nx.astar_path_length(map_graph, i, j)
#                               for i in task_graph_id for j in task_graph_id]).view(10, 10)
#
#         x = nx.complete_graph(t, nx.DiGraph)
#         nx.set_node_attributes(x, dict(zip(x.nodes(), task_coord)), name='coord')
#         nx.set_edge_attributes(x, dict(zip(x.edges(), [y[r.item(), c.item()].item()
#                                                        for r, c in zip(y.nonzero(as_tuple=True)[0],
#                                                                        y.nonzero(as_tuple=True)[1])])), name='astar')
#         x = dgl.from_networkx(x, node_attrs=['coord'], edge_attrs=['astar'])
#         x.edata['astar'] = x.edata['astar'].float()
#
#         try:
#             if not os.path.exists(dir):
#                 os.makedirs(dir)
#         except OSError:
#             print("Error: Cannot create the directory.")
#
#         with open(dir + '{}.pkl'.format(itr), 'wb') as f:
#             pickle.dump([x, y], f)


def load_scenarios(dir):
    dir = scenario_dir + '/' + dir
    with open(dir, 'rb') as f:
        data = pickle.load(f)

    return data


if __name__ == "__main__":
    save_scenarios(itrs=100, size=32, obs=20, a=4, t=20, seed=42, train=True)
    save_scenarios(itrs=20, size=32, obs=20, a=4, t=20, seed=43, train=False)
