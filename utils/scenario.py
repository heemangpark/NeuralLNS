import os
import pickle
import random
import sys
from pathlib import Path

import networkx as nx
import numpy as np
import torch
from tqdm import trange

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from utils.graph import valid_graph, partition_8
from utils.seed import seed_everything
from utils.solver import one_step_solver

curr_path = os.path.realpath(__file__)
home_dir = Path(curr_path).parent.parent

EMPTY, OBS, AGENT, TASK = 0, 1, 2, 3


def only_for_pathgnn(itrs: int, size: int, obs: int, a: int, t: int, seed: int):
    seed_everything(seed)
    data_list = []

    num_train = int(itrs * .6)
    num_val = int(itrs * .2)
    num_test = int(itrs * .2)

    data_dir = os.path.join(home_dir, 'datas/scenarios/') + 'pathgnn/'
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    for _ in trange(itrs):
        while True:
            grid, _ = valid_graph(size=size, obs=obs, fixed=True)
            graph = nx.grid_2d_graph(size, size)
            total_coord = [(i, j) for i in range(size) for j in range(size)]

            obs_coord = [(i, j) for i, j in zip(grid.nonzero()[0], grid.nonzero()[1])]
            empty_coord = list(set(total_coord) - set(obs_coord))

            a_coord = random.sample(empty_coord, a)
            empty_coord = list(set(empty_coord) - set(a_coord))

            t_coord = random.sample(empty_coord, t)
            empty_coord = list(set(empty_coord) - set(t_coord))

            schedule = [(total_coord.index(a), total_coord.index(t)) for a, t in zip(a_coord, t_coord)]

            type_list = []
            for n in graph.nodes():
                if n in obs_coord:
                    type_list.append(OBS)
                elif n in a_coord:
                    type_list.append(AGENT)
                elif n in t_coord:
                    type_list.append(TASK)
                else:
                    type_list.append(EMPTY)

            nx.set_node_attributes(graph, dict(zip(graph.nodes(), type_list)), 'type')

            cost = one_step_solver(grid, a_coord, t_coord, os.path.join(home_dir, 'PBS/pyg/'), '_')
            if cost == 'retry':
                itrs += 1
            else:
                break

        data_list.append([grid, graph, a_coord, t_coord, schedule, cost])

    random.shuffle(data_list)
    torch.save(data_list[:num_train], data_dir + 'train.pt')
    torch.save(data_list[num_train:num_train + num_val], data_dir + 'val.pt')
    torch.save(data_list[num_train + num_val: num_train + num_val + num_test], data_dir + 'test.pt')


def save_random_scenarios(itrs: int, size: int, obs: int, a: int, t: int, seed: int):
    seed_everything(seed)

    data_list = []

    num_train = int(itrs * .6)
    num_val = int(itrs * .2)
    num_test = int(itrs * .2)

    data_dir = os.path.join(home_dir, 'datas/scenarios/') + '{}_{}_{}_{}_{}/'.format(size, size, obs, a, t)
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    for _ in trange(itrs):
        while True:
            grid, graph = valid_graph(size=size, obs=obs, fixed=False)

            grid_idx = list(range(len(graph)))
            a_idx = random.sample(grid_idx, a)
            grid_idx = list(set(grid_idx) - set(a_idx))
            t_idx = random.sample(grid_idx, t)

            a_coord = np.array(graph.nodes())[a_idx].tolist()
            t_coord = np.array(graph.nodes())[t_idx].tolist()

            cost = one_step_solver(grid, a_coord, t_coord, os.path.join(home_dir, 'PBS/pyg/'), '_')
            if cost == 'retry':
                itrs += 1
            else:
                break

        data_list.append([grid, graph, a_coord, t_coord, cost])
    random.shuffle(data_list)

    torch.save(data_list[:num_train], data_dir + 'train.pt')
    torch.save(data_list[num_train:num_train + num_val], data_dir + 'val.pt')
    torch.save(data_list[num_train + num_val: num_train + num_val + num_test], data_dir + 'test.pt')


def save_partition_scenarios(itrs: int, size: int, a: int, t: int, seed: int):
    seed_everything(seed)

    data_list = []

    num_train = int(itrs * .6)
    num_val = int(itrs * .2)
    num_test = int(itrs * .2)

    data_dir = os.path.join(home_dir, 'datas/scenarios/') + '{}_{}_partition_{}_{}/'.format(size, size, a, t)
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    grid, graph = partition_8()

    for _ in trange(itrs):
        while True:
            grid_idx = list(range(len(graph)))
            a_idx = random.sample(grid_idx, a)
            grid_idx = list(set(grid_idx) - set(a_idx))
            t_idx = random.sample(grid_idx, t)

            a_coord = np.array(graph.nodes())[a_idx].tolist()
            t_coord = np.array(graph.nodes())[t_idx].tolist()

            cost = one_step_solver(grid, a_coord, t_coord, os.path.join(home_dir, 'PBS/pyg/'), '_')
            if cost == 'retry':
                itrs += 1
            else:
                break

        data_list.append([grid, graph, a_coord, t_coord, cost])
    random.shuffle(data_list)

    torch.save(data_list[:num_train], data_dir + 'train.pt')
    torch.save(data_list[num_train:num_train + num_val], data_dir + 'val.pt')
    torch.save(data_list[num_train + num_val: num_train + num_val + num_test], data_dir + 'test.pt')


def load_scenarios(dir):
    with open(os.path.join(home_dir, 'datas/scenarios/') + dir, 'rb') as f:
        data = pickle.load(f)

    return data


if __name__ == "__main__":
    only_for_pathgnn(itrs=10000, size=8, obs=20, a=5, t=5, seed=42)
    # save_random_scenarios(itrs=100000, size=16, obs=20, a=20, t=20, seed=42)
    # save_partition_scenarios(itrs=100000, size=16, a=20, t=20, seed=42)
