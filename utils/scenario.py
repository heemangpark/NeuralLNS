import os
import random
import sys
from pathlib import Path

import networkx as nx
import numpy as np
import torch
from tqdm import trange

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from utils.graph import valid_graph, special_case
from utils.seed import seed_everything
from utils.solver import one_step_solver, solver

curr_path = os.path.realpath(__file__)
home_dir = Path(curr_path).parent.parent
EMPTY, OBS, AGENT, TASK = 0, 1, 2, 3


def implicit(itrs: int, size: int, obs: int, a: int, t: int, seed: int):
    seed_everything(seed)
    data = []
    num_train = int(itrs * .6)
    num_val = int(itrs * .2)
    num_test = int(itrs * .2)

    data_dir = os.path.join(home_dir, 'datas/scenarios/test/') + '{}_{}_{}_{}_{}/'.format(size, size, obs, a, t)
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
                pass
            else:
                x = a_coord + t_coord

                edge_index = []
                for i in range(len(x)):
                    for j in range(i, len(x)):
                        if i == j:
                            pass
                        else:
                            edge_index += [[i, j], [j, i]]

                A, M, P = [[] for _ in range(3)]
                for i, j in edge_index:
                    astar = nx.astar_path_length(graph, tuple(x[i]), tuple(x[j]))
                    manhattan = sum(abs(np.array(x[i]) - np.array(x[j])))
                    proxy = (astar - manhattan)
                    A.append(astar)
                    M.append(manhattan)
                    P.append(proxy)

                edge_index = np.array(edge_index).transpose(-1, 0)

                ef_1, ef_2, ef_3 = np.array(A), np.array(M), np.array(P)
                ef = np.stack((ef_1, ef_2, ef_3), axis=-1)

                data.append([x, edge_index, ef, sum(cost)])
                break

    random.shuffle(data)
    torch.save(data[:num_train], data_dir + 'train.pt')
    torch.save(data[num_train:num_train + num_val], data_dir + 'val.pt')
    torch.save(data[num_train + num_val: num_train + num_val + num_test], data_dir + 'test.pt')


def explicit(itrs: int, size: int, obs: int, a: int, t: int, seed: int, hard: bool = False):
    seed_everything(seed)

    data_list, coord_data, type_data, e_id_data, sch_data, cost_data = [[] for _ in range(6)]

    num_train = int(itrs * .6)
    num_val = int(itrs * .2)
    num_test = int(itrs * .2)

    data_dir = os.path.join(home_dir, 'data/32_single/')
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    for _ in trange(itrs):
        while True:
            grid, _ = valid_graph(size=size, obs=obs, fixed=True)
            graph = nx.grid_2d_graph(size, size)

            coords = list(graph.nodes())
            obs_coord = [(i, j) for i, j in zip(grid.nonzero()[0], grid.nonzero()[1])]
            empty_coord = list(set(coords) - set(obs_coord))
            a_coord = random.sample(empty_coord, a)
            empty_coord = list(set(empty_coord) - set(a_coord))
            t_coord = random.sample(empty_coord, t)
            empty_coord = list(set(empty_coord) - set(t_coord))

            types = []
            for n in graph.nodes():
                if n in obs_coord:
                    types.append(OBS)
                elif n in a_coord:
                    types.append(AGENT)
                elif n in t_coord:
                    types.append(TASK)
                elif n in empty_coord:
                    types.append(EMPTY)

            row = np.array([(g, g + 1) for g in range(grid.shape[0] - 1)])
            col = row * grid.shape[0]
            edge_index = []
            for g in range(grid.shape[0]):
                edge_index += (row + g * grid.shape[0]).tolist() * 2 + (col + g).tolist() * 2

            schedule = [(coords.index(a), coords.index(t)) for a, t in zip(a_coord, t_coord)]

            cost = one_step_solver(grid, a_coord, t_coord, os.path.join(home_dir, 'PBS/pyg/'), '32_single')
            if cost == 'retry':
                itrs += 1
            else:
                if hard:
                    naive_cost = sum([abs(a[0] - t[0]) + abs(a[1] - t[1]) for a, t in zip(a_coord, t_coord)])
                    if abs(sum(cost) - naive_cost) <= 10:
                        itrs += 1
                    else:
                        data_list.append([coords, types, edge_index, schedule, sum(cost)])
                        break
                else:
                    data_list.append([coords, types, edge_index, schedule, sum(cost)])
                break

    random.shuffle(data_list)
    torch.save(data_list[:num_train], data_dir + 'train.pt')
    torch.save(data_list[num_train:num_train + num_val], data_dir + 'val.pt')
    torch.save(data_list[num_train + num_val: num_train + num_val + num_test], data_dir + 'test.pt')


def explicit_sequential(itrs: int, size: int, obs: int, a: int, t: int, seed: int):
    seed_everything(seed)
    data_list = []

    if obs == 0:
        data_dir = os.path.join(home_dir, 'data/sparse/')
    elif obs == 10:
        data_dir = os.path.join(home_dir, 'data/medium/')
    elif obs == 20:
        data_dir = os.path.join(home_dir, 'data/dense/')

    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    for _ in trange(itrs):
        while True:
            grid, _ = valid_graph(size=size, obs=obs, fixed=True)
            graph = nx.grid_2d_graph(size, size)

            coords = list(graph.nodes())
            obs_coord = [(i, j) for i, j in zip(grid.nonzero()[0], grid.nonzero()[1])]
            empty_coord = list(set(coords) - set(obs_coord))
            a_coord = random.sample(empty_coord, a)
            empty_coord = list(set(empty_coord) - set(a_coord))
            t_coord = random.sample(empty_coord, t)
            empty_coord = list(set(empty_coord) - set(t_coord))

            types = []
            for n in graph.nodes():
                if n in obs_coord:
                    types.append(OBS)
                elif n in a_coord:
                    types.append(AGENT)
                elif n in t_coord:
                    types.append(TASK)
                elif n in empty_coord:
                    types.append(EMPTY)

            row = np.array([(g, g + 1) for g in range(grid.shape[0] - 1)])
            col = row * grid.shape[0]
            edge_index = []
            for g in range(grid.shape[0]):
                edge_index += (row + g * grid.shape[0]).tolist() * 2 + (col + g).tolist() * 2

            num_seq = t // a
            if num_seq == 3:
                seq = [(coords.index(a), coords.index(t1), coords.index(t2), coords.index(t3))
                       for a, t1, t2, t3 in
                       zip(a_coord, t_coord[:len(a_coord)],
                           t_coord[len(a_coord):2 * len(a_coord)],
                           t_coord[2 * len(a_coord):3 * len(a_coord)]
                           )
                       ]
            elif num_seq == 4:
                seq = [(coords.index(a), coords.index(t1), coords.index(t2), coords.index(t3), coords.index(t4))
                       for a, t1, t2, t3, t4 in
                       zip(a_coord, t_coord[:len(a_coord)],
                           t_coord[len(a_coord):2 * len(a_coord)],
                           t_coord[2 * len(a_coord):3 * len(a_coord)],
                           t_coord[3 * len(a_coord):4 * len(a_coord)]
                           )
                       ]
            elif num_seq == 5:
                seq = [(coords.index(a), coords.index(t1), coords.index(t2),
                        coords.index(t3), coords.index(t4), coords.index(t5))
                       for a, t1, t2, t3, t4, t5 in
                       zip(a_coord, t_coord[:len(a_coord)],
                           t_coord[len(a_coord):2 * len(a_coord)],
                           t_coord[2 * len(a_coord):3 * len(a_coord)],
                           t_coord[3 * len(a_coord):4 * len(a_coord)],
                           t_coord[4 * len(a_coord):5 * len(a_coord)]
                           )
                       ]
            elif num_seq == 6:
                seq = [(coords.index(a), coords.index(t1), coords.index(t2),
                        coords.index(t3), coords.index(t4), coords.index(t5), coords.index(t6))
                       for a, t1, t2, t3, t4, t5, t6 in
                       zip(a_coord, t_coord[:len(a_coord)],
                           t_coord[len(a_coord):2 * len(a_coord)],
                           t_coord[2 * len(a_coord):3 * len(a_coord)],
                           t_coord[3 * len(a_coord):4 * len(a_coord)],
                           t_coord[4 * len(a_coord):5 * len(a_coord)],
                           t_coord[5 * len(a_coord):6 * len(a_coord)]
                           )
                       ]

            cost, path = solver(grid, a_coord, [[list(coords[t_id]) for t_id in s[1:]] for s in seq],
                                os.path.join(home_dir, 'PBS/pyg/'), '{}{}_{}_{}'.format(size, size, a, t))

            if cost == 'retry':
                itrs += 1
            else:
                data_list.append([coords, types, edge_index, seq, cost])
                break

    random.shuffle(data_list)
    torch.save(data_list, data_dir + '{}{}_{}_{}.pt'.format(size, size, a, t))


def specific(itrs: int, size: int, a: int, t: int, seed: int):
    seed_everything(seed)

    data_list = []

    num_train = int(itrs * .6)
    num_val = int(itrs * .2)
    num_test = int(itrs * .2)

    data_dir = os.path.join(home_dir, 'datas/scenarios/') + '{}_{}_partition_{}_{}/'.format(size, size, a, t)
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    grid, graph = special_case()

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


def explicit_test_only(itrs: int, size: int, obs: int, a: int, t: int, seed: int, hard: bool = False):
    seed_everything(seed)

    data_list, coord_data, type_data, e_id_data, sch_data, cost_data = [[] for _ in range(6)]

    data_dir = os.path.join(home_dir, 'data/single/')
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    for _ in trange(itrs):
        while True:
            grid, _ = valid_graph(size=size, obs=obs, fixed=True)
            graph = nx.grid_2d_graph(size, size)

            coords = list(graph.nodes())
            obs_coord = [(i, j) for i, j in zip(grid.nonzero()[0], grid.nonzero()[1])]
            empty_coord = list(set(coords) - set(obs_coord))
            a_coord = random.sample(empty_coord, a)
            empty_coord = list(set(empty_coord) - set(a_coord))
            t_coord = random.sample(empty_coord, t)
            empty_coord = list(set(empty_coord) - set(t_coord))

            types = []
            for n in graph.nodes():
                if n in obs_coord:
                    types.append(OBS)
                elif n in a_coord:
                    types.append(AGENT)
                elif n in t_coord:
                    types.append(TASK)
                elif n in empty_coord:
                    types.append(EMPTY)

            row = np.array([(g, g + 1) for g in range(grid.shape[0] - 1)])
            col = row * grid.shape[0]
            edge_index = []
            for g in range(grid.shape[0]):
                edge_index += (row + g * grid.shape[0]).tolist() * 2 + (col + g).tolist() * 2

            schedule = [(coords.index(a), coords.index(t)) for a, t in zip(a_coord, t_coord)]

            cost = one_step_solver(grid, a_coord, t_coord, os.path.join(home_dir, 'PBS/pyg/'), '_')
            if cost == 'retry':
                itrs += 1
            else:
                if hard:
                    naive_cost = sum([abs(a[0] - t[0]) + abs(a[1] - t[1]) for a, t in zip(a_coord, t_coord)])
                    if abs(sum(cost) - naive_cost) <= 10:
                        itrs += 1
                    else:
                        data_list.append([coords, types, edge_index, schedule, sum(cost)])
                        break
                else:
                    data_list.append([coords, types, edge_index, schedule, sum(cost)])
                break
    random.shuffle(data_list)

    if obs == 0:
        torch.save(data_list, data_dir + 'sparse_{}{}_{}_{}.pt'.format(size, size, a, t))
    elif obs == 10:
        torch.save(data_list, data_dir + 'medium_{}{}_{}_{}.pt'.format(size, size, a, t))
    elif obs == 20:
        torch.save(data_list, data_dir + 'dense_{}{}_{}_{}.pt'.format(size, size, a, t))


if __name__ == "__main__":
    # explicit_sequential(itrs=10000, size=8, obs=10, a=5, t=15, seed=42)
    # explicit_sequential(itrs=10000, size=8, obs=10, a=5, t=20, seed=42)
    # explicit_sequential(itrs=10000, size=8, obs=10, a=5, t=25, seed=42)
    # explicit_sequential(itrs=10000, size=8, obs=10, a=5, t=30, seed=42)
    #
    # explicit_sequential(itrs=10000, size=16, obs=10, a=5, t=15, seed=42)
    # explicit_sequential(itrs=10000, size=16, obs=10, a=5, t=20, seed=42)
    # explicit_sequential(itrs=10000, size=16, obs=10, a=5, t=25, seed=42)
    # explicit_sequential(itrs=10000, size=16, obs=10, a=5, t=30, seed=42)

    explicit_test_only(itrs=1000, size=8, obs=0, a=5, t=5, seed=42)
    explicit_test_only(itrs=1000, size=8, obs=10, a=5, t=5, seed=42)
    explicit_test_only(itrs=1000, size=8, obs=20, a=5, t=5, seed=42)

    explicit_test_only(itrs=1000, size=8, obs=0, a=10, t=10, seed=42)
    explicit_test_only(itrs=1000, size=8, obs=10, a=10, t=10, seed=42)
    explicit_test_only(itrs=1000, size=8, obs=20, a=10, t=10, seed=42)

    explicit_test_only(itrs=1000, size=8, obs=0, a=15, t=15, seed=42)
    explicit_test_only(itrs=1000, size=8, obs=10, a=15, t=15, seed=42)
    explicit_test_only(itrs=1000, size=8, obs=20, a=15, t=15, seed=42)
