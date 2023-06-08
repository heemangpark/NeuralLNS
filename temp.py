import copy
import os
import random
import shutil
import sys

import numpy as np
from matplotlib import pyplot as plt
from omegaconf import OmegaConf
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from src.heuristics.hungarian import hungarian
from src.heuristics.regret import f_ijk
from src.heuristics.shaw import removal
from utils.scenario import load_scenarios
from utils.seed import seed_everything
from utils.solver import solver


def temp(cfg: dict):
    for num in [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]:
        seed_everything(cfg.seed)
        total_idx = list(range(cfg.num_data))
        test_idx = copy.deepcopy(total_idx)
        t_id = list(random.sample(test_idx, k=num))
        gap = []

        for t_id in tqdm(t_id):
            grid, grid_graph, a_coord, t_coord = load_scenarios(
                '{}{}{}_{}_{}/scenario_{}.pkl'.format(cfg.map_size, cfg.map_size, cfg.obs_ratio,
                                                      cfg.num_agent, cfg.num_task, t_id))
            assign_idx, assign_coord = hungarian(a_coord, t_coord)

            actual_init_cost, _ = solver(grid, a_coord, assign_coord, save_dir=cfg.save_dir,
                                         exp_name='init' + str(t_id))
            est_init_cost = sum(
                [sum(t) for t in [[abs(a[0] - b[0]) + abs(a[1] - b[1]) for a, b, in zip(sch[:-1], sch[1:])]
                                  for sch in assign_coord]])

            prev_cost = est_init_cost
            for itr in range(cfg.itrs):
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

                if (est_cost < prev_cost) or (itr == (cfg.itrs - 1)):
                    prev_cost = est_cost
                    assign_idx = temp_assign_idx

            actual_final_cost, _ = solver(grid, a_coord, assign_coord, save_dir=cfg.save_dir,
                                          exp_name='fin' + str(t_id))
            if os.path.exists(cfg.save_dir):
                shutil.rmtree(cfg.save_dir)

            perf = (actual_init_cost - actual_final_cost) / actual_init_cost * 100
            gap.append(perf)

        plt.plot(gap)
        plt.title('{:.4f}'.format(np.mean(gap)))
        plt.savefig('itrs_{}.png'.format(cfg.itrs))
        plt.clf()


if __name__ == '__main__':
    temp(OmegaConf.load('config/temp.yaml'))
