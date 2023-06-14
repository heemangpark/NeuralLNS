import copy
import os
import pickle
import random
import shutil
import sys

import networkx as nx
import numpy as np
import torch
from matplotlib import pyplot as plt
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from src.heuristics.hungarian import hungarian
from src.heuristics.regret import f_ijk
from src.heuristics.shaw import removal
from utils.scenario import load_scenarios
from utils.seed import seed_everything
from utils.solver import solver


def lns_itr_test(cfg):
    seed_everything(cfg.seed)

    for itrs in [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]:
        gap = []

        for t in tqdm(list(random.sample(list(range(cfg.num_data)), k=100))):
            grid, grid_graph, a_coord, t_coord = load_scenarios(
                '{}{}{}_{}_{}/scenario_{}.pkl'.format(cfg.map_size, cfg.map_size, cfg.obs_ratio,
                                                      cfg.num_agent, cfg.num_task, t))
            assign_idx, assign_coord = hungarian(a_coord, t_coord)

            actual_init_cost, _ = solver(grid, a_coord, assign_coord, save_dir=cfg.save_dir, exp_name='init' + str(t))
            prev_cost = sum([sum(t) for t in [[abs(a[0] - b[0]) + abs(a[1] - b[1])
                                               for a, b, in zip(sch[:-1], sch[1:])] for sch in assign_coord]])

            for itr in range(itrs):
                temp_assign_idx = copy.deepcopy(assign_idx)
                removal_idx = removal(assign_idx, t_coord)
                removed = [False for _ in removal_idx]
                for schedule in temp_assign_idx:
                    for i, r in enumerate(removal_idx):
                        if removed[i]:
                            continue
                        if r in schedule:
                            schedule.remove(r)
                            removed[i] = True

                while len(removal_idx) != 0:
                    f_val = f_ijk(a_coord, t_coord, temp_assign_idx, removal_idx)
                    regrets = np.stack(list(f_val.values()))
                    argmin_regret = np.argmin(regrets, axis=None)
                    min_regret_idx = np.unravel_index(argmin_regret, regrets.shape)
                    r_idx, insertion_edge_idx = min_regret_idx
                    re_ins = removal_idx[r_idx]
                    ag_idx = 0
                    while True:
                        ag_schedule = assign_idx[ag_idx]
                        if insertion_edge_idx - (len(ag_schedule) + 1) < 0:
                            ins_pos = insertion_edge_idx
                            break
                        else:
                            insertion_edge_idx -= (len(ag_schedule) + 1)
                            ag_idx += 1

                    temp_assign_idx[ag_idx].insert(ins_pos, re_ins)
                    removal_idx.remove(re_ins)
                    assign_coord = [np.array(t_coord)[schedule].tolist() for schedule in temp_assign_idx]
                    est_cost = sum([sum(t) for t in [[abs(a[0] - b[0]) + abs(a[1] - b[1])
                                                      for a, b, in zip(sch[:-1], sch[1:])] for sch in assign_coord]])

                if est_cost < prev_cost:
                    prev_cost = est_cost
                    assign_idx = copy.deepcopy(temp_assign_idx)

            actual_final_cost, _ = solver(grid, a_coord, assign_coord, save_dir=cfg.save_dir, exp_name='fin' + str(t))

            if os.path.exists(cfg.save_dir):
                shutil.rmtree(cfg.save_dir)

            perf = (actual_init_cost - actual_final_cost) / actual_init_cost * 100
            gap.append(perf)

        plt.plot(gap)
        plt.title('{:.4f}'.format(np.mean(gap)))
        plt.savefig('itrs_{}.png'.format(itrs))
        plt.clf()


def pyg_8820_5_5():
    data_list = []

    for s_id in range(10000):
        with open('datas/scenarios/8820_5_5_train/scenario_{}.pkl'.format(s_id), 'rb') as f:
            grid, graph, a_coord, t_coord = pickle.load(f)

        x = torch.cat((torch.FloatTensor(a_coord), torch.FloatTensor(t_coord))) / grid.shape[0]

        edge_index = torch.LongTensor([[r, c, c, r] for r in range(len(a_coord))
                                       for c in range(len(a_coord), len(a_coord) + len(t_coord))]).view(-1, 2)

        edge_attr = torch.Tensor([[nx.astar_path_length(graph, tuple(_a), tuple(_t)) / grid.shape[0]] * 2
                                  for _a in a_coord for _t in t_coord]).flatten()

        data_list.append(Data(x=x, edge_index=edge_index, edge_attr=edge_attr))

    torch.save(data_list, 'datas/pyg/8820_5_5.pt')


def main():
    data_list = torch.load('datas/pyg/8820_5_5.pt')
    loader = DataLoader(data_list)
    debug = 1


if __name__ == '__main__':
    # temp(OmegaConf.load('config/lns_itr_test.yaml'))
    # pyg_8820_5_5()
    main()
