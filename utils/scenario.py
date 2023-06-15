import os
import pickle
import random
import sys
from pathlib import Path

import numpy as np
import torch
from tqdm import trange

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from utils.graph import valid_graph
from utils.seed import seed_everything

curr_path = os.path.realpath(__file__)
scenario_dir = os.path.join(Path(curr_path).parent.parent, 'datas/scenarios/')


def save_scenarios(itrs: int, size: int, obs: int, a: int, t: int, seed: int):
    seed_everything(seed)

    data_list = []

    num_train = int(itrs * .6)
    num_val = int(itrs * .2)
    num_test = int(itrs * .2)

    data_dir = scenario_dir + '{}_{}_{}_{}_{}/'.format(size, size, obs, a, t)
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    for _ in trange(itrs):
        grid, graph = valid_graph(size, obs)

        grid_idx = list(range(len(graph)))
        a_idx = random.sample(grid_idx, a)
        grid_idx = list(set(grid_idx) - set(a_idx))
        t_idx = random.sample(grid_idx, t)

        a_coord = np.array(graph.nodes())[a_idx].tolist()
        t_coord = np.array(graph.nodes())[t_idx].tolist()

        # type = dict(zip(graph.nodes(), ['B' for _ in range(graph.number_of_nodes())]))
        # for _a in a_coord:
        #     type[tuple(_a)] = 'A'
        # for _t in t_coord:
        #     type[tuple(_t)] = 'T'
        # nx.set_node_attributes(G=graph, values=type, name='type')

        data_list.append([grid, graph, a_coord, t_coord])
    random.shuffle(data_list)

    torch.save(data_list[:num_train], data_dir + 'train.pt')
    torch.save(data_list[num_train:num_train + num_val], data_dir + 'val.pt')
    torch.save(data_list[num_train + num_val: num_train + num_val + num_test], data_dir + 'test.pt')


def load_scenarios(dir):
    with open(scenario_dir + dir, 'rb') as f:
        data = pickle.load(f)

    return data


if __name__ == "__main__":
    save_scenarios(itrs=10000, size=8, obs=20, a=5, t=5, seed=42)
    save_scenarios(itrs=10000, size=16, obs=20, a=5, t=20, seed=42)
    save_scenarios(itrs=10000, size=32, obs=20, a=5, t=50, seed=42)
