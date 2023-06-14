import copy
import os
import pickle
import random
import shutil
import sys
from datetime import datetime
from itertools import combinations
from pathlib import Path

import dgl
import numpy as np
import torch
import wandb
from omegaconf import OmegaConf
from tqdm import trange, tqdm

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1, 2, 3"
from src.heuristics.hungarian import hungarian
from src.heuristics.regret import f_ijk
from src.heuristics.shaw import removal
from src.models.destroy import Destroy
from utils.graph import sch_to_dgl
from utils.scenario import load_scenarios
from utils.seed import seed_everything
from utils.solver import solver


def train_data(cfg: dict):
    seed_everything(cfg.seed)
    for exp_num in trange(cfg.num_data):
        grid, grid_graph, a_coord, t_coord = load_scenarios('{}{}{}_{}_{}_train/scenario_{}.pkl'
                                                            .format(cfg.map_size, cfg.map_size, cfg.obs_ratio,
                                                                    cfg.num_agent, cfg.num_task, exp_num))
        assign_idx, assign_coord = hungarian(a_coord, t_coord)
        schedule = [[a] + t for a, t in zip(a_coord, assign_coord)]
        sch_graph = sch_to_dgl(assign_idx, schedule, grid.shape[0])

        init_cost, _ = solver(grid, a_coord, assign_coord, save_dir=cfg.save_dir, exp_name='init')
        if init_cost == 'error':
            return 'abandon_seed'

        full_set = list(combinations(range(cfg.num_task), 3))
        random.shuffle(full_set)
        destroy_set = full_set[:cfg.destroy_per_map]

        cost_dict = {}
        for D in destroy_set:
            temp_assign_idx = copy.deepcopy(assign_idx)
            removal_idx = list(D)
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

                # get min regret
                regrets = np.stack(list(f_val.values()))
                argmin_regret = np.argmin(regrets, axis=None)
                min_regret_idx = np.unravel_index(argmin_regret, regrets.shape)

                r_idx, insertion_edge_idx = min_regret_idx
                re_ins = removal_idx[r_idx]

                # get insertion agent index and location
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

                cost, _, time_log = solver(grid, a_coord, assign_coord,
                                           save_dir=cfg.save_dir, exp_name=str(exp_num), ret_log=True)

            if cost == 'error':
                pass
            else:
                decrement = init_cost - cost
                cost_dict[D] = decrement

        total_data = [sch_graph, cost_dict]
        # with open('datas/{}/train/{}.pkl'.format(cfg.map_size, exp_num), 'wb') as f:
        #     pickle.dump(total_data, f)


def eval_data(cfg: dict):
    seed_everything(cfg.seed)
    for eval_id in trange(cfg.num_data):
        grid, map_graph, a_coord, t_coord = load_scenarios('{}{}{}_{}_{}_eval/scenario_{}.pkl'
                                                           .format(cfg.map_size, cfg.map_size, cfg.obs_ratio,
                                                                   cfg.num_agent, cfg.num_task, eval_id))
        assign_idx, assign_coord = hungarian(a_coord, t_coord)
        schedule = [[a] + t for a, t in zip(a_coord, assign_coord)]
        sch_graph = sch_to_dgl(assign_idx, schedule, grid.shape[0])

        init_cost, _, time_log = solver(grid, a_coord, assign_coord,
                                        save_dir=cfg.save_dir, exp_name='init', ret_log=True)
        if init_cost == 'error':
            return 'abandon_seed'

        temp_assign_idx = copy.deepcopy(assign_idx)
        removal_idx = removal(assign_idx, t_coord, N=2, time_log=time_log)
        if removal_idx == 'stop':
            return 'stop'

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

            # get min regret
            regrets = np.stack(list(f_val.values()))
            argmin_regret = np.argmin(regrets, axis=None)
            min_regret_idx = np.unravel_index(argmin_regret, regrets.shape)

            r_idx, insertion_edge_idx = min_regret_idx
            re_ins = removal_idx[r_idx]

            # get insertion agent index and location
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
            assign_pos = [np.array(t_coord)[schedule].tolist() for schedule in temp_assign_idx]

        cost, _, time_log = solver(grid, a_coord, assign_pos,
                                   save_dir=cfg.save_dir, exp_name=str(eval_id), ret_log=True)

        # with open('datas/{}/eval/{}.pkl'.format(cfg.map_size, eval_id), 'wb') as f:
        #     pickle.dump([grid, map_graph, a_coord, t_coord, assign_idx, assign_coord,
        #                  schedule, sch_graph, init_cost, cost], f)


def train(cfg: dict):
    seed_everything(cfg.seed)
    date = datetime.now().strftime("%m%d_%H%M%S")

    h_params = dict(gnn=cfg.model.type, gnn_aggregator=cfg.model.aggr,
                    learning_rate=cfg.optimizer.lr, weight_decay=cfg.optimizer.weight_decay)
    config_dict = dict(yaml='../config/train.yaml', params=h_params)

    if cfg.wandb:
        wandb.init(project='NeuralLNS', name=date, config=config_dict)

    model = Destroy(cfg)
    train_idx = list(range(cfg.num_train))

    for e in trange(cfg.epochs):
        random.shuffle(train_idx)
        epoch_loss = 0

        for b_id in range(cfg.num_train // cfg.batch_size):
            flags = [random.choice([1, -1]) for _ in range(cfg.batch_size)]
            graphs = []

            train_id = train_idx[b_id * cfg.batch_size: (b_id + 1) * cfg.batch_size]
            for t_id, flag in zip(train_id, flags):
                with open('datas/32/train/{}.pkl'.format(t_id), 'rb') as f:
                    graph, destroy = pickle.load(f)
                d_sorted = sorted(destroy.items(), key=lambda x: x[1], reverse=True)
                destroy = dict((d_sorted[0], d_sorted[1])) if flag == 1 else dict((d_sorted[1], d_sorted[0]))
                graphs += [dgl.node_subgraph(graph, list(set(range(graph.num_nodes())) - set(d_key)))
                           for d_key in destroy.keys()]

            graphs = dgl.batch(graphs).to(cfg.device)
            targets = torch.Tensor(flags).to(cfg.device)

            batch_loss = model(graphs, targets)
            # print(batch_loss)
            epoch_loss += batch_loss

        epoch_loss /= (cfg.num_train // cfg.batch_size)

        if cfg.wandb:
            wandb.log({'epoch_loss': epoch_loss})

        if (e + 1) % 10 == 0:
            dir = 'datas/models/{}/'.format(date)
            if not os.path.exists(dir):
                os.makedirs(dir)
            torch.save(model.state_dict(), dir + '{}.pt'.format(e + 1))

            if cfg.val:
                temp = Destroy(cfg)
                temp.load_state_dict(torch.load(dir + '{}.pt'.format(e + 1)))
                temp.eval()
                PN = ['P', 'N']
                correct = 0

                for v_id in range(cfg.num_val):
                    random.shuffle(PN)
                    with open('datas/32/val/{}.pkl'.format(v_id), 'rb') as f:
                        graph, destroy = pickle.load(f)
                    d_sorted = sorted(destroy.items(), key=lambda x: x[1], reverse=True)
                    if PN[0] == 'P':
                        destroy = dict((d_sorted[0], d_sorted[1]))
                    else:
                        destroy = dict((d_sorted[1], d_sorted[0]))
                    graphs = dgl.batch([dgl.node_subgraph(graph, list(set(range(graph.num_nodes())) - set(d_key)))
                                        for d_key in destroy.keys()]).to(cfg.device)
                    val_res = temp.val(graphs)

                    if val_res == PN[0]:
                        correct += 1

                if cfg.wandb:
                    wandb.log({'val_result': correct / cfg.num_val})


def eval(cfg: dict):
    seed_everything(cfg.seed)
    model = Destroy(cfg)
    model.load_state_dict(torch.load(cfg.dir))
    model.eval()

    eval_dir = 'datas/eval_data_32/'
    map_index = list(range(cfg.num_eval))

    total_model_perf = 0
    total_baseline_perf = 0

    for map_id in tqdm(map_index):
        solver_dir = os.path.join(Path(os.path.realpath(__file__)).parent.parent, 'PBS/pbs')
        save_dir = os.path.join(Path(os.path.realpath(__file__)).parent.parent, 'PBS/{}/'.
                                format(datetime.now().strftime("%m%d_%H%M%S")))
        lns_dir = [solver_dir, save_dir, 'eval', map_id]

        try:
            if not os.path.exists(lns_dir[1]):
                os.makedirs(lns_dir[1])
        except OSError:
            print("Error: Cannot create the directory.")

        with open(eval_dir + 'eval_data{}.pkl'.format(map_id), 'rb') as f:
            grid, map_graph, a_coord, t_coord, \
            assign_idx, assign_coord, schedule, sch_graph, init_cost, lns_cost = pickle.load(f)

        temp_assign_idx = copy.deepcopy(assign_idx)
        temp_graph = copy.deepcopy(sch_graph).to(cfg.device)
        num_tasks = (temp_graph.ndata['type'] == 2).sum().item()
        destroys = [c for c in combinations(range(num_tasks), 3)]
        candidates = random.sample(destroys, cfg.cand_size)
        removal_idx = model.act(temp_graph, candidates)

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

            # get min regret
            regrets = np.stack(list(f_val.values()))
            argmin_regret = np.argmin(regrets, axis=None)
            min_regret_idx = np.unravel_index(argmin_regret, regrets.shape)

            r_idx, insertion_edge_idx = min_regret_idx
            re_ins = removal_idx[r_idx]

            # get insertion agent index and location
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
            assign_pos = [np.array(t_coord)[schedule].tolist() for schedule in temp_assign_idx]

            cost, _, time_log = solver(grid, a_coord, assign_pos,
                                       save_dir=cfg.save_dir, exp_name='eval', ret_log=True)

        model_perf = (init_cost - cost) / init_cost * 100
        baseline_perf = (init_cost - lns_cost) / init_cost * 100
        total_model_perf += model_perf
        total_baseline_perf += baseline_perf

        try:
            if os.path.exists(lns_dir[1]):
                shutil.rmtree(lns_dir[1])
        except OSError:
            print("Error: Cannot remove the directory.")

    print('Model Performance: {:.4f} || Model Performance: {:.4f}'
          .format(total_model_perf / cfg.num_eval, total_baseline_perf / cfg.num_eval))


def _getConfigPath(func):
    def wrapper(cfg_mode):
        cfg_path = 'config/{}.yaml'.format(cfg_mode)
        func(cfg_mode, cfg_path)

    return wrapper


@_getConfigPath
def run(cfg_mode, cfg_path):
    cfg = OmegaConf.load(cfg_path)
    globals()[cfg_mode](cfg)
