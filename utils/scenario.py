import os
import pickle
import random
import sys
from pathlib import Path

import networkx as nx
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
        dir = scenario_dir + '/{}{}{}_{}_{}_train/'.format(size, size, obs, a, t)
    else:
        dir = scenario_dir + '/{}{}{}_{}_{}_eval/'.format(size, size, obs, a, t)

    for itr in trange(itrs):
        grid, graph = valid_graph(size, obs)

        grid_idx = list(range(len(graph)))
        a_idx = random.sample(grid_idx, a)
        grid_idx = list(set(grid_idx) - set(a_idx))
        t_idx = random.sample(grid_idx, t)

        a_coord = np.array(graph.nodes())[a_idx].tolist()
        t_coord = np.array(graph.nodes())[t_idx].tolist()

        type = dict(zip(graph.nodes(), ['B' for _ in range(graph.number_of_nodes())]))
        for _a in a_coord:
            type[tuple(_a)] = 'A'
        for _t in t_coord:
            type[tuple(_t)] = 'T'
        nx.set_node_attributes(G=graph, values=type, name='type')
        data = [grid, graph, a_coord, t_coord]

        try:
            if not os.path.exists(dir):
                os.makedirs(dir)
        except OSError:
            print("Error: Cannot create the directory.")
        with open(dir + 'scenario_{}.pkl'.format(itr), 'wb') as f:
            pickle.dump(data, f)


def load_scenarios(dir):
    dir = scenario_dir + '/' + dir
    with open(dir, 'rb') as f:
        data = pickle.load(f)

    return data


if __name__ == "__main__":
    save_scenarios(itrs=10000, size=8, obs=20, a=5, t=5, seed=42, train=True)
    save_scenarios(itrs=2000, size=8, obs=20, a=5, t=5, seed=24, train=False)
