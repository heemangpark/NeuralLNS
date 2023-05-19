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
import yaml
from tqdm import trange

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from src.models.destroyEdgewise import DestroyEdgewise
from src.heuristics.regret import f_ijk, get_regret
from src.heuristics.shaw import removal
from utils.graph import convert_to_nx
from utils.seed import seed_everything
from utils.solver import solver, assignment_to_id, to_solver


def train(cfg: dict):
    batch_num = cfg['batch_num']
    data_size = cfg['data_size']
    device = cfg['device']
    epochs = cfg['epochs']
    method = cfg['method']
    seed = cfg['seed']
    wandb = cfg['wandb']

    seed_everything(seed)

    date = datetime.now().strftime("%m%d_%H%M%S")

    if wandb:
        import wandb
        wandb.init(
            project='NLNS-destroy',
            name=date,
            config={
                'method': method,
                'loss type': 'REINFORCE'
            }
        )

    model = DestroyEdgewise(device=device)

    batchSize = data_size // batch_num
    data_idx = list(range(data_size))

    for e in trange(epochs):
        random.shuffle(data_idx)
        epoch_loss = 0

        for b in range(batch_num):
            batch_graph, batch_destroy = [], []

            for d_id in data_idx[b * batchSize: (b + 1) * batchSize]:
                with open('train_data/dataDestroy_{}.pkl'.format(d_id), 'rb') as f:
                    graph, destroy = pickle.load(f)
                    if method == 'topK':
                        destroy = dict(sorted(destroy.items(), key=lambda x: abs(x[1]), reverse=True)[:10])
                    elif method == 'randomK':
                        randomKey = list(destroy.keys())
                        random.shuffle(randomKey)
                        randomKey = randomKey[:10]
                        destroy = dict(zip(randomKey, [destroy[k] for k in randomKey]))
                    batch_graph.append(graph)
                    batch_destroy.append(destroy)
            batch_graph = dgl.batch(batch_graph).to(device)

            batch_loss = model.learn(batch_graph, batch_destroy, batch_num, device=device)
            epoch_loss += batch_loss
        epoch_loss /= batch_num

        if wandb:
            wandb.log({'epoch_loss': epoch_loss})

        if (e + 1) % 10 == 0:
            dir = 'data/trained/models/{}/'.format(date)
            if not os.path.exists(dir):
                os.makedirs(dir)
            torch.save(model.state_dict(), dir + '{}_{}.pt'.format(method, e + 1))


def eval(cfg: dict):
    cand_size = cfg['cand_size']
    device = cfg['device']
    dir = cfg['dir']
    eval_num = cfg['eval_num']
    seed = cfg['seed']
    type = cfg['type']  # neural, heuristic

    seed_everything(seed)

    " Load model "
    model = DestroyEdgewise(device=device)
    model.load_state_dict(torch.load(dir))
    model.eval()

    map_index = random.choices(np.arange(len(os.listdir('../data/eval_data/550/'))), eval_num)
    entire_eval_perf = 0

    for map_id in map_index:
        " EECBS solver directory setup "
        solver_dir = os.path.join(Path(os.path.realpath(__file__)).parent.parent, 'PBS/pbs')
        save_dir = os.path.join(Path(os.path.realpath(__file__)).parent.parent, 'PBS/{}/'.
                                format(datetime.now().strftime("%m%d_%H%M%S")))
        lns_dir = [solver_dir, save_dir, 'eval_{}'.format(type), map_id]

        try:
            if not os.path.exists(lns_dir[1]):
                os.makedirs(lns_dir[1])
        except OSError:
            print("Error: Cannot create the directory.")

        " Load initial solution "
        with open('data/eval_data/evalData_{}.pkl'.format(map_id), 'rb') as f:
            info, graph, lns = pickle.load(f)

        " LNS procedure "

        task_idx, assign = copy.deepcopy(info['lns'])
        pre_cost = info['init_cost']
        results = [pre_cost]

        for itr in range(100):
            if type == 'neural':
                temp_assign = copy.deepcopy(assign)
                temp_graph = copy.deepcopy(graph)

                num_tasks = len([i for i in temp_graph.nodes() if temp_graph.ndata['type'][i] == 2])

                destroyCand = [c for c in combinations(range(num_tasks), 3)]
                candDestroy = random.sample(destroyCand, cand_size)
                removal_idx = model.act(temp_graph, candDestroy, 'greedy', device)
                removal_idx = list(removal_idx)

            elif type == 'heuristic':
                time_log = None
                temp_assign = copy.deepcopy(assign)
                removal_idx = removal(
                    task_idx,
                    info['tasks'],
                    info['graph'],
                    N=2,
                    time_log=time_log
                )

            else:
                raise NotImplementedError('EVALUATION SUPPORTS NEURAL OR HEURISTIC ONLY')

            for i, t in enumerate(temp_assign.values()):
                for r in removal_idx:
                    if {r: info['tasks'][r]} in t:
                        temp_assign[i].remove({r: info['tasks'][r]})

            while len(removal_idx) != 0:
                f_val = f_ijk(temp_assign, info['agents'], removal_idx, info['tasks'], info['graph'])
                regret = get_regret(f_val)
                regret = dict(sorted(regret.items(), key=lambda x: x[1][0], reverse=True))
                re_ins = list(regret.keys())[0]
                re_a, re_j = regret[re_ins][1], regret[re_ins][2]
                removal_idx.remove(re_ins)
                to_insert = {re_ins: info['tasks'][re_ins]}
                temp_assign[re_a].insert(re_j, to_insert)

            cost, _, time_log = solver(
                info['grid'],
                info['agents'],
                to_solver(info['tasks'], temp_assign),
                ret_log=True,
                dir=lns_dir
            )

            if cost == 'error':
                pass

            else:
                if cost < pre_cost:
                    pre_cost = cost
                    assign = temp_assign
                    results.append(pre_cost)
                    task_idx = assignment_to_id(len(info['agents']), assign)
                    if type == 'neural':
                        coordination = [[a] for a in info['agents'].tolist()]
                        for i, coords in enumerate(assign.values()):
                            temp_schedule = [list(c.values())[0][0] for c in coords]
                            coordination[i].extend(temp_schedule)
                        next_nx_graph = convert_to_nx(task_idx, coordination, info['grid'].shape[0])
                        next_graph = dgl.from_networkx(
                            next_nx_graph,
                            node_attrs=['coord', 'type', 'idx', 'graph_id'],
                            edge_attrs=['dist', 'connected']
                        ).to(device)
                        next_graph.edata['dist'] = next_graph.edata['dist'].to(torch.float32)
                        graph = next_graph
                    elif type == 'heuristic':
                        pass
                    else:
                        raise NotImplementedError('EVALUATION SUPPORTS NEURAL OR HEURISTIC ONLY')

                elif cost >= pre_cost:
                    results.append(pre_cost)

        each_map_perf = (results[0] - results[-1]) / results[0]
        entire_eval_perf += each_map_perf

        try:
            if os.path.exists(lns_dir[1]):
                shutil.rmtree(lns_dir[1])
        except OSError:
            print("Error: Cannot remove the directory.")

    return entire_eval_perf / eval_num


def run(mode):
    with open('config/{}.yaml'.format(mode)) as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
    if mode == 'train':
        train(cfg)
    elif mode == 'eval':
        eval(cfg)
    else:
        raise NotImplementedError('RUN SUPPORTS train OR eval ONLY')

    # for evalItr in range(10):
    #     manager = multiprocessing.Manager()
    #     output = manager.dict()
    #     process = []
    #
    #     for mapID in mapIndex[evalItr * 10: evalItr * 10 + 10]:
    #         p = Process(target=multiEval, args=(*['', '', '', ''] + [mapID],))
    #         process.append(p)
    #         p.start()
    #
    #     for proc in process:
    #         proc.join()
    #     nlnsList = list(output.values())
    #     totalList.extend(nlnsList)
    #
    # print('NLNS: {:.4f}'.format(np.mean(totalList)))
