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
from omegaconf import OmegaConf
from tqdm import trange, tqdm

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from src.heuristics.hungarian import hungarian
from src.heuristics.regret import f_ijk
from src.heuristics.shaw import removal
from src.models.destroy_edgewise import DestroyEdgewise
from src.models.destroy_edgewise import TestDestroy
from utils.graph import convert_to_nx
from utils.scenario import load_scenarios
from utils.seed import seed_everything
from utils.solver import solver


def train_data(cfg, process_num: int = 0):
    seed_everything(cfg.seed)

    num_data_per_process = cfg.num_data // cfg.n_processes
    if process_num == 0:
        exp_num_range = trange(num_data_per_process * process_num, num_data_per_process * (process_num + 1))
    else:
        exp_num_range = range(num_data_per_process * process_num, num_data_per_process * (process_num + 1))
    for exp_num in exp_num_range:
        # total_data
        scenario = load_scenarios('{}{}{}_{}_{}/scenario_{}.pkl'
                                  .format(cfg.map_size, cfg.map_size,
                                          cfg.obs_ratio,
                                          cfg.num_agent,
                                          cfg.num_task,
                                          exp_num))

        info = {'grid': scenario[0], 'graph': scenario[1], 'agents': scenario[2],
                'tasks': [t[0] for t in scenario[3]]}
        assign_id, assign_pos = hungarian(info['graph'], info['agents'], info['tasks'])
        info['lns'] = assign_id, assign_pos

        coordination = [[a.tolist()] + t for a, t in zip(info['agents'], assign_pos)]
        init_graph = convert_to_nx(assign_id, coordination, info['grid'].shape[0])
        info['init_cost'], _ = solver(
            info['grid'],
            info['agents'],
            assign_pos,
            solver_dir=cfg.solver_dir,
            save_dir=cfg.save_dir,
            exp_name='init')
        if info['init_cost'] == 'error':
            return 'abandon_cfg.seed'

        # data = [init_graph]
        assign_idx, assign_pos = info['lns']
        pre_cost = info['init_cost']

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
                f_val = f_ijk(temp_assign_idx, info['agents'], removal_idx, info['tasks'], info['graph'])

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

                assign_pos = [np.array(info['tasks'])[schedule].tolist() for schedule in temp_assign_idx]

                cost, _, time_log = solver(
                    info['grid'],
                    info['agents'],
                    assign_pos,
                    solver_dir=cfg.solver_dir,
                    save_dir=cfg.save_dir,
                    exp_name=str(exp_num),
                    ret_log=True
                )

            if cost == 'error':
                pass
            else:
                decrement = pre_cost - cost  # decrement
                cost_dict[D] = decrement

        total_data = [init_graph, cost_dict]
        with open('data/train_data/train_data{}.pkl'.format(exp_num), 'wb') as f:
            pickle.dump(total_data, f)


def eval_data(cfg):
    seed_everything(cfg.seed)

    for exp_num in trange(cfg.num_data):
        scenario = load_scenarios('{}{}{}_{}_{}_eval/scenario_{}.pkl'
                                  .format(cfg.map_size, cfg.map_size,
                                          cfg.obs_ratio,
                                          cfg.num_agent,
                                          cfg.num_task,
                                          exp_num))

        info = {'grid': scenario[0], 'graph': scenario[1], 'agents': scenario[2],
                'tasks': [t[0] for t in scenario[3]]}
        assign_id, assign_pos = hungarian(info['graph'], info['agents'], info['tasks'])
        info['lns'] = assign_id, assign_pos

        coordination = [[a.tolist()] + t for a, t in zip(info['agents'], assign_pos)]
        init_graph = convert_to_nx(assign_id, coordination, info['grid'].shape[0])
        info['init_cost'], _, time_log = solver(
            info['grid'],
            info['agents'],
            assign_pos,
            solver_dir=cfg.solver_dir,
            save_dir=cfg.save_dir,
            exp_name='init',
            ret_log=True)
        if info['init_cost'] == 'error':
            return 'abandon_cfg.seed'

        assign_idx, assign_pos = info['lns']
        pre_cost = info['init_cost']

        data = [info, init_graph]

        for _ in range(100):
            temp_assign_idx = copy.deepcopy(assign_idx)
            removal_idx = removal(
                assign_idx,
                info['tasks'],
                info['graph'],
                N=2,
                time_log=time_log
            )
            if removal_idx == 'stop':
                return 'stop'

            # remove 'removal_idx'
            removed = [False for _ in removal_idx]
            for schedule in temp_assign_idx:
                for i, r in enumerate(removal_idx):
                    if removed[i]:
                        continue
                    if r in schedule:
                        schedule.remove(r)
                        removed[i] = True

            while len(removal_idx) != 0:
                f_val = f_ijk(temp_assign_idx, info['agents'], removal_idx, info['tasks'], info['graph'])

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

                assign_pos = [np.array(info['tasks'])[schedule].tolist() for schedule in temp_assign_idx]

                cost, _, time_log = solver(
                    info['grid'],
                    info['agents'],
                    assign_pos,
                    ret_log=True,
                    solver_dir=cfg.solver_dir,
                    save_dir=cfg.save_dir,
                    exp_name=str(exp_num)
                )

            if cost == 'error':
                pass
            else:
                if cost < pre_cost:
                    pre_cost = cost
                    assign_idx = temp_assign_idx

            data.append(pre_cost)
        with open('datas/eval_data/eval_data{}.pkl'.format(exp_num), 'wb') as f:
            pickle.dump(data, f)


def train(cfg: dict):
    date = datetime.now().strftime("%m%d_%H%M%S")
    seed_everything(cfg.seed)

    hyperparameter_defaults = dict(
        batch_size=cfg.data_size // cfg.batch_num,
        aggregator=cfg.model.aggr,
        learning_rate=cfg.optimizer.lr,
        weight_decay=cfg.optimizer.weight_decay
    )

    config_dictionary = dict(
        yaml='../config/train.yaml',
        params=hyperparameter_defaults,
    )

    if cfg.wandb:
        import wandb
        wandb.init(project='NeuralLNS', name=date, config=config_dictionary)

    model = DestroyEdgewise(device=cfg.device,
                            lr=cfg.optimizer.lr,
                            weight_decay=cfg.optimizer.weight_decay,
                            aggr=cfg.model.aggr)

    batch_size = cfg.data_size // cfg.batch_num
    data_idx = list(range(cfg.data_size))

    for e in trange(cfg.epochs):
        random.shuffle(data_idx)
        epoch_loss = 0

        for b in range(cfg.batch_num):
            batch_graph, batch_destroy = [], []

            for d_id in data_idx[b * batch_size: (b + 1) * batch_size]:
                with open('datas/train_data/train_data{}.pkl'.format(d_id), 'rb') as f:
                    graph, destroy = pickle.load(f)
                    if cfg.method == 'topK':
                        destroy = dict(sorted(destroy.items(), key=lambda x: abs(x[1]), reverse=True)[:10])
                    elif cfg.method == 'randomK':
                        random_key = list(destroy.keys())
                        random.shuffle(random_key)
                        random_key = random_key[:10]
                        destroy = dict(zip(random_key, [destroy[k] for k in random_key]))
                    batch_graph.append(graph)
                    batch_destroy.append(destroy)
            batch_graph = dgl.batch(batch_graph).to(cfg.device)

            batch_loss = model.learn(batch_graph, batch_destroy, batch_size, device=cfg.device)
            epoch_loss += batch_loss
        epoch_loss /= cfg.batch_num

        if cfg.wandb:
            wandb.log({'epoch_loss': epoch_loss})

        if (e + 1) % 10 == 0:
            dir = 'datas/trained/models/{}/'.format(date)
            if not os.path.exists(dir):
                os.makedirs(dir)
            torch.save(model.state_dict(), dir + '{}_{}.pt'.format(cfg.method, e + 1))


def eval(cfg: dict):
    seed_everything(cfg.seed)

    " Load model "
    model = DestroyEdgewise(device=cfg.device)
    model.load_state_dict(torch.load(cfg.dir))
    model.eval()

    map_index = list(range(len(os.listdir('datas/eval_data/550/'))))
    random.shuffle(map_index)
    map_index = map_index[:cfg.eval_num]

    entire_model_perf = 0
    entire_baseline_perf = 0

    for map_id in tqdm(map_index):
        " EECBS solver directory setup "
        solver_dir = os.path.join(Path(os.path.realpath(__file__)).parent.parent, 'PBS/pbs')
        save_dir = os.path.join(Path(os.path.realpath(__file__)).parent.parent, 'PBS/{}/'.
                                format(datetime.now().strftime("%m%d_%H%M%S")))
        lns_dir = [solver_dir, save_dir, 'eval_{}'.format(cfg.type), map_id]

        try:
            if not os.path.exists(lns_dir[1]):
                os.makedirs(lns_dir[1])
        except OSError:
            print("Error: Cannot create the directory.")

        " Load initial solution "
        with open('datas/eval_data/550/eval_data{}.pkl'.format(map_id), 'rb') as f:
            F = pickle.load(f)
            info, graph = F[0], F[1]
            graph = dgl.from_networkx(
                graph,
                node_attrs=['coord', 'type', 'idx', 'graph_id'],
                edge_attrs=['dist', 'connected']
            ).to(cfg.device)
            graph.edata['dist'] = graph.edata['dist'].to(torch.float32)
            lns = ((F[2] - F[-1]) / F[2]) * 100

        " LNS procedure "
        assign_idx, assign_pos = info['lns']
        pre_cost = info['init_cost']
        results = [pre_cost]

        for _ in range(100):
            if cfg.type == 'neural':
                temp_assign_idx = copy.deepcopy(assign_idx)
                temp_graph = copy.deepcopy(graph)
                num_tasks = (temp_graph.ndata['type'] == 2).sum().item()
                destroyCand = [c for c in combinations(range(num_tasks), 3)]
                candDestroy = random.sample(destroyCand, cfg.cand_size)
                removal_idx = model.act(temp_graph, candDestroy, 'greedy', cfg.device)
                removal_idx = list(removal_idx)

            elif cfg.type == 'heuristic':
                time_log = None
                temp_assign_idx = copy.deepcopy(assign_idx)
                removal_idx = removal(
                    assign_idx,
                    info['tasks'],
                    info['graph'],
                    N=2,
                    time_log=time_log
                )
                if removal_idx == 'stop':
                    return 'stop'

            # remove 'removal_idx'
            removed = [False for _ in removal_idx]
            for schedule in temp_assign_idx:
                for i, r in enumerate(removal_idx):
                    if removed[i]:
                        continue
                    if r in schedule:
                        schedule.remove(r)
                        removed[i] = True

            while len(removal_idx) != 0:
                f_val = f_ijk(temp_assign_idx, info['agents'], removal_idx, info['tasks'], info['graph'])

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

                assign_pos = [np.array(info['tasks'])[schedule].tolist() for schedule in temp_assign_idx]

                cost, _, time_log = solver(
                    info['grid'],
                    info['agents'],
                    assign_pos,
                    ret_log=True,
                    solver_dir=cfg.solver_dir,
                    save_dir=cfg.save_dir,
                    exp_name='eval'
                )

            if cost == 'error':
                pass

            else:
                if cost < pre_cost:
                    pre_cost = cost
                    assign_idx = temp_assign_idx
                    results.append(pre_cost)
                    if cfg.type == 'neural':
                        coordination = [[a.tolist()] + t for a, t in zip(info['agents'], assign_pos)]
                        next_nx_graph = convert_to_nx(assign_idx, coordination, info['grid'].shape[0])
                        next_graph = dgl.from_networkx(
                            next_nx_graph,
                            node_attrs=['coord', 'type', 'idx', 'graph_id'],
                            edge_attrs=['dist', 'connected']
                        ).to(cfg.device)
                        next_graph.edata['dist'] = next_graph.edata['dist'].to(torch.float32)
                        graph = next_graph
                    elif cfg.type == 'heuristic':
                        pass
                    else:
                        raise NotImplementedError('EVALUATION SUPPORTS NEURAL OR HEURISTIC ONLY')

                elif cost >= pre_cost:
                    results.append(pre_cost)

        each_map_perf = (results[0] - results[-1]) / results[0]
        entire_model_perf += each_map_perf
        entire_baseline_perf += lns

        try:
            if os.path.exists(lns_dir[1]):
                shutil.rmtree(lns_dir[1])
        except OSError:
            print("Error: Cannot remove the directory.")

    print(entire_model_perf / cfg.eval_num, entire_baseline_perf / cfg.eval_num)


def test_destroy(cfg: dict):
    date = datetime.now().strftime("%m%d_%H%M%S")
    seed_everything(cfg.seed)

    hyperparameter_defaults = dict(
        batch_size=cfg.data_size // cfg.batch_num,
        aggregator=cfg.model.aggr,
        learning_rate=cfg.optimizer.lr,
        weight_decay=cfg.optimizer.weight_decay
    )

    config_dictionary = dict(
        yaml='../config/test_destroy.yaml',
        params=hyperparameter_defaults,
    )

    if cfg.wandb:
        import wandb
        wandb.init(project='NeuralLNS', name=date, config=config_dictionary)

    model = TestDestroy(device=cfg.device,
                        lr=cfg.optimizer.lr,
                        weight_decay=cfg.optimizer.weight_decay,
                        aggr=cfg.model.aggr)

    train_size = int(cfg.data_size * .7)
    eval_size = int(cfg.data_size * .3)

    batch_size = cfg.data_size // cfg.batch_num
    train_idx = list(range(cfg.data_size))
    eval_idx = list(range(train_size, cfg.data_size))

    for _ in trange(cfg.epochs):
        random.shuffle(train_idx)
        epoch_loss = 0

        for b in range(cfg.batch_num):
            batch_graphs, batch_target = [], []

            for d_id in train_idx[b * batch_size: (b + 1) * batch_size]:
                with open('datas/scenarios/test_destroy/{}.pkl'.format(d_id), 'rb') as f:
                    graphs, targets = pickle.load(f)
                batch_graphs.append(graphs)
                batch_target.append(targets)

            batch_graphs = dgl.batch(batch_graphs)
            batch_loss = model(batch_graphs, batch_target)
            epoch_loss += batch_loss

        epoch_loss /= cfg.batch_num

        if cfg.wandb:
            wandb.log({'epoch_loss': epoch_loss})

    torch.save(model.state_dict(), 'datas/trained/models/test_destroy.pt')

    "evaluation"
    model.eval()
    score = 0
    for e_id in eval_idx:
        with open('datas/scenarios/test_destroy/{}.pkl'.format(e_id), 'rb') as f:
            graph, target = pickle.load(f)
        y_hat = model._eval(graph).cpu()
        score += torch.mean(torch.abs(y_hat - target.view(-1)[target.view(-1).nonzero()]))
    print(score / eval_size)


def _getConfigPath(func):
    def wrapper(cfg_mode):
        cfg_path = 'config/{}.yaml'.format(cfg_mode)
        func(cfg_mode, cfg_path)

    return wrapper


@_getConfigPath
def run(cfg_mode, cfg_path):
    cfg = OmegaConf.load(cfg_path)
    globals()[cfg_mode](cfg)
