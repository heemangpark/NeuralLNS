import copy
import math
import os
import pickle
import random
import sys
from pathlib import Path

import numpy as np
import torch
from einops import rearrange
from matplotlib import pyplot as plt
from omegaconf import OmegaConf
from torch_geometric.data import Data, Batch
from tqdm import trange

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from src.model.rt import RelationalTransformer
from utils.seed import seed_everything
from utils.solver import solver


def test_sequential(device: str, map_size):
    seed_everything(seed=42)
    config = OmegaConf.load('config/experiment/rt.yaml')
    test_data = torch.load('data/{}_multi/3.pt'.format(map_size), map_location=device)

    pred_list = []
    label_list = [td[-1].item() for td in test_data]
    # label_list.pop(4642)
    label_list = np.array(label_list)
    label_list = (label_list - label_list.mean()) / (label_list.std() + 1e-5)
    label_list = label_list.tolist()

    batch_size = 10
    for b_id in trange(len(test_data) // batch_size):
        batch_data = []
        for td in test_data[b_id * batch_size: (b_id + 1) * batch_size]:
            coords, types, edge_index, schedule, _ = td

            agent_task_list = [[] for _ in range(4)]
            for idx in range(4):
                agent_task_list[idx].extend([sch[idx] for sch in schedule])

            nf_coord = torch.Tensor(coords)
            nf_coord = (nf_coord - nf_coord.min()) / (nf_coord.max() + nf_coord.min())
            edge_index = torch.LongTensor(edge_index).transpose(-1, 0)

            mask_type = [copy.deepcopy(types) for _ in range(3)]
            for a, t1, t2, t3 in zip(agent_task_list[0], agent_task_list[1], agent_task_list[2], agent_task_list[3]):
                mask_type[0][t2] = 0  # t2, t3 -> 0
                mask_type[0][t3] = 0

                mask_type[1][a] = 0  # a -> 0 | t1 -> 2 | t3 -> 0
                mask_type[1][t1] = 2
                mask_type[1][t3] = 0

                mask_type[2][a] = 0  # a, t1 -> 0 | t2 -> 2
                mask_type[2][t1] = 0
                mask_type[2][t2] = 2

            nf_1 = torch.cat((nf_coord, torch.eye(4)[mask_type[0]]), -1)
            nf_2 = torch.cat((nf_coord, torch.eye(4)[mask_type[1]]), -1)
            nf_3 = torch.cat((nf_coord, torch.eye(4)[mask_type[2]]), -1)

            sch_1 = [(sch[0], sch[1]) for sch in schedule]
            sch_2 = [(sch[1], sch[2]) for sch in schedule]
            sch_3 = [(sch[2], sch[3]) for sch in schedule]

            batch_data += [Data(x=nf_1, edge_index=edge_index, edge_attr=torch.ones(edge_index.shape[-1], 1),
                                schedule=torch.LongTensor(sch_1)),
                           Data(x=nf_2, edge_index=edge_index, edge_attr=torch.ones(edge_index.shape[-1], 1),
                                schedule=torch.LongTensor(sch_2)),
                           Data(x=nf_3, edge_index=edge_index, edge_attr=torch.ones(edge_index.shape[-1], 1),
                                schedule=torch.LongTensor(sch_3))]

        data = Batch.from_data_list(batch_data).to(device)

        gnn = RelationalTransformer(config).to(device)
        gnn.load_state_dict(torch.load('data/models/rt_8_sum.pt'))
        gnn.eval()

        pred = gnn(data, test=True)
        pred = rearrange(pred, '(N C) -> N C', N=batch_size)
        pred_list.extend(pred.sum(-1))

    plt.clf()
    # pred_list.pop(4642)
    plt.plot(pred_list, label_list, 'b.')
    criterion = range(math.floor(min(pred_list + label_list)), math.ceil(max(pred_list + label_list)))
    plt.plot(criterion, criterion, 'r--')
    plt.xlabel('preds')
    plt.ylabel('labels')
    plt.title('rt_sequential')
    plt.show()


def convert_to_data(coords, types, e_id, sch):
    m_sch = copy.deepcopy(sch)
    data = []

    nf_coord = torch.Tensor(coords)
    nf_coord = (nf_coord - nf_coord.min()) / (nf_coord.max() + nf_coord.min())
    e_id = torch.LongTensor(e_id).transpose(-1, 0)
    ef = torch.ones(e_id.shape[-1], 1)

    sketch_types = [t if t in [0, 1] else 0 for t in types]

    max_len = max([len(s) for s in m_sch])
    for s in m_sch:
        if len(s) == max_len:
            pass
        else:
            while True:
                s.append(s[-1])
                if len(s) == max_len:
                    break
    split = [[] for _ in range(max_len - 1)]
    for idx, p in enumerate(split):
        p += [(s[idx], s[idx + 1]) for s in m_sch]

    num_map = len(split)

    complete_types = [None for _ in range(num_map)]
    for s_id, s in enumerate(split):
        agent_id, task_id = [_s[0] for _s in s], [_s[1] for _s in s]
        temp = copy.deepcopy(sketch_types)
        for a, t in zip(agent_id, task_id):
            temp[t] = 3
            temp[a] = 2
        complete_types[s_id] = temp

    for c_id, ct in enumerate(complete_types):
        data.append(Data(x=torch.cat((nf_coord, torch.eye(4)[ct]), -1),
                         edge_index=e_id, edge_attr=ef,
                         schedule=torch.LongTensor(split[c_id])))

    return data


def NLNS(map_data, prev_sch, device, batch_size=100):
    # load NN model
    config = OmegaConf.load('config/experiment/rt.yaml')
    gnn = RelationalTransformer(config).to(device)
    gnn.load_state_dict(torch.load('data/models/rt_8_sum.pt'))
    gnn.eval()

    _d = []
    data_idx = []
    modified_schedule_set = []

    for _ in range(batch_size):
        # destroy idx
        num_tasks = []
        for p_id, ps in enumerate(prev_sch):
            num_tasks.extend([p_id for _ in range(len(ps) - 1)])
        from_a = sorted(random.sample(num_tasks, k=3))
        from_s = []
        for f_a in list(set(from_a)):
            from_s.extend(random.sample(prev_sch[f_a][1:], k=from_a.count(f_a)))

        # modify schedule
        modified_schedule = copy.deepcopy(prev_sch)
        for a_id, t_id in zip(from_a, from_s):
            modified_schedule[a_id].remove(t_id)
        for fs in from_s:
            to_a = random.choice(range(len(prev_sch)))
            to_s = random.choice(range(1, [len(m_s) for m_s in modified_schedule][to_a] + 1))
            modified_schedule[to_a].insert(to_s, fs)
        modified_schedule_set.append(modified_schedule)

        data = convert_to_data(map_data[0], map_data[1], map_data[2], modified_schedule)
        _d.extend(data)
        data_idx.append(len(data))
    _d = Batch.from_data_list(data_list=_d).to(device)

    pred = gnn(_d, test=True)
    count = 0
    batch_pred = []
    for d_id in data_idx:
        next_count = count + d_id
        batch_pred.append(sum(pred[count:next_count]))
        count = next_count

    return modified_schedule_set[batch_pred.index(min(batch_pred))]


def random_LNS(map_data, prev_sch, batch_size, dc):
    modified_schedule_set = []

    while True:
        for step_count in range(batch_size):
            # destroy idx
            num_tasks = []
            for p_id, ps in enumerate(prev_sch):
                num_tasks.extend([p_id for _ in range(len(ps) - 1)])
            from_a = sorted(random.sample(num_tasks, k=3))
            from_s = []
            for f_a in list(set(from_a)):
                from_s.extend(random.sample(prev_sch[f_a][1:], k=from_a.count(f_a)))

            # modify schedule
            modified_schedule = copy.deepcopy(prev_sch)
            for a_id, t_id in zip(from_a, from_s):
                modified_schedule[a_id].remove(t_id)
            for fs in from_s:
                to_a = random.choice(range(len(prev_sch)))
                to_s = random.choice(range(1, [len(m_s) for m_s in modified_schedule][to_a] + 1))
                modified_schedule[to_a].insert(to_s, fs)
            modified_schedule_set.append(modified_schedule)

        random_idx = random.choice(range(batch_size))
        cost = solver(map_data[0],
                      [list(map_data[1][s[0]]) for s in modified_schedule_set[random_idx]],
                      [[list(map_data[1][_ts]) for _ts in ts] for ts in
                       [s[1:] if len(s) != 1 else [] for s in modified_schedule_set[random_idx]]],
                      os.path.join(Path(os.path.realpath(__file__)).parent, 'PBS/pyg/'), 'random_lns_' + dc)[0]
        if cost != 'retry':
            break

    return modified_schedule_set[random_idx], cost


def LNS(map_data, prev_sch, batch_size, dc):
    modified_schedule_set = []
    cost_set = []

    for step_count in range(batch_size):
        while True:
            # destroy idx
            num_tasks = []
            for p_id, ps in enumerate(prev_sch):
                num_tasks.extend([p_id for _ in range(len(ps) - 1)])
            from_a = sorted(random.sample(num_tasks, k=3))
            from_s = []
            for f_a in list(set(from_a)):
                from_s.extend(random.sample(prev_sch[f_a][1:], k=from_a.count(f_a)))

            # modify schedule
            modified_schedule = copy.deepcopy(prev_sch)
            for a_id, t_id in zip(from_a, from_s):
                modified_schedule[a_id].remove(t_id)
            for fs in from_s:
                to_a = random.choice(range(len(prev_sch)))
                to_s = random.choice(range(1, [len(m_s) for m_s in modified_schedule][to_a] + 1))
                modified_schedule[to_a].insert(to_s, fs)

            # calculation through solver
            cost = solver(map_data[0],
                          [list(map_data[1][s[0]]) for s in modified_schedule],
                          [[list(map_data[1][_ts]) for _ts in ts] for ts in
                           [s[1:] if len(s) != 1 else [] for s in modified_schedule]],
                          os.path.join(Path(os.path.realpath(__file__)).parent, 'PBS/pyg/'), 'heu_lns_' + dc)[0]
            if cost != 'retry':
                break
            else:
                modified_schedule_set.append(modified_schedule)
                step_count -= 1

        cost_set.append(cost)

    return modified_schedule_set[cost_set.index(min(cost_set))], min(cost_set)


def run_nn(data, device, sample_per_steps=100, config=''):
    seed_everything()

    coords, types, edge_index, schedule, _ = data
    grid = np.array([t if t in [0, 1] else 0 for t in types]).reshape(8, 8)
    map_data = coords, types, edge_index
    schedule = [list(s) for s in schedule]

    cost = solver(grid,
                  [list(coords[s[0]]) for s in schedule],
                  [[list(coords[_ts]) for _ts in ts] for ts in [s[1:] if len(s) != 1 else [] for s in schedule]],
                  os.path.join(Path(os.path.realpath(__file__)).parent, 'PBS/pyg/'), 'nn_' + config)[0]
    output = [cost]

    for itr in range(1, 301):
        while True:
            next_schedule = NLNS(map_data, schedule, device, sample_per_steps)
            next_cost = solver(grid,
                               [list(coords[s[0]]) for s in next_schedule],
                               [[list(coords[_ts]) for _ts in ts] for ts in
                                [s[1:] if len(s) != 1 else [] for s in next_schedule]],
                               os.path.join(Path(os.path.realpath(__file__)).parent, 'PBS/pyg/'), 'nn_' + config)[0]
            if next_cost != 'retry':
                break

        if cost > next_cost:
            schedule = next_schedule
            cost = next_cost
        else:
            pass
        output.append(cost)

    return output


def run_heu(data, sample_per_steps: int = 100, ran=False, config=''):
    seed_everything()

    coords, types, _, schedule, _ = data
    grid = np.array([t if t in [0, 1] else 0 for t in types]).reshape(8, 8)
    map_data = grid, coords, types
    schedule = [list(s) for s in schedule]

    cost = solver(grid,
                  [list(coords[s[0]]) for s in schedule],
                  [[list(coords[_ts]) for _ts in ts] for ts in [s[1:] if len(s) != 1 else [] for s in schedule]],
                  os.path.join(Path(os.path.realpath(__file__)).parent, 'PBS/pyg/'), 'heu_' + config)[0]
    output = [cost]

    for itr in range(1, 301):
        if ran:
            next_schedule, next_cost = random_LNS(map_data, schedule, sample_per_steps, config)
        else:
            next_schedule, next_cost = LNS(map_data, schedule, sample_per_steps, config)

        if cost > next_cost:
            schedule = next_schedule
            cost = next_cost
        else:
            pass
        output.append(cost)

    return output


if __name__ == '__main__':
    seed_everything()
    mode = 'exp'
    # mode = 'plot'

    data_config = '88_5_20'
    cuda = 'cuda:3'

    if mode == 'exp':
        from tqdm import tqdm

        datadir = torch.load('data/{}.pt'.format(data_config), map_location=torch.device('cpu'))
        map_id = random.sample(range(10000), 100)

        nn_res, heu_res, r_res = [], [], []

        for m_id in tqdm(map_id):
            # nn = run_nn(data=datadir[m_id], device=cuda, config=data_config)
            # nn_res.append(nn)
            heu = run_heu(data=datadir[m_id], config=data_config)
            heu_res.append(heu)
            # r = run_heu(data=datadir[m_id], ran=True, config=data_config)
            # r_res.append(r)

        # with open('nn_' + data_config, 'wb') as fp:
        #     pickle.dump(nn_res, fp)
        with open('heu_' + data_config, 'wb') as fp:
            pickle.dump(heu_res, fp)
        # with open('r_' + data_config, 'wb') as fp:
        #     pickle.dump(r_res, fp)

    elif mode == 'plot':
        with open('nn_' + data_config, 'rb') as fp:
            nn = pickle.load(fp)
        with open('heu_' + data_config, 'rb') as fp:
            heu = pickle.load(fp)
        with open('r_' + data_config, 'rb') as fp:
            r = pickle.load(fp)

        nn_sig = np.array(nn).mean(0) + np.array(nn).std(0) * .2, np.array(nn).mean(0) - np.array(nn).std(0) * .2
        heu_sig = np.array(heu).mean(0) + np.array(heu).std(0) * .2, np.array(heu).mean(0) - np.array(heu).std(0) * .2
        r_sig = np.array(r).mean(0) + np.array(r).std(0) * .2, np.array(r).mean(0) - np.array(r).std(0) * .2

        init_cost = np.array(nn).mean(0)[0]

        plt.clf()
        plt.plot(range(0, 301), np.array(nn).mean(0), 'r-', label='NN')
        plt.plot(range(0, 301), np.array(heu).mean(0), 'b-', label='Heuristic')
        plt.plot(range(0, 301), np.array(r).mean(0), 'g-', label='Random')
        plt.axhline(init_cost * 2 / 3, color='lightgrey', linestyle='--')
        plt.axhline(init_cost * 1 / 2, color='lightgrey', linestyle='--')
        plt.fill_between(range(0, 301), nn_sig[0], nn_sig[1], color='lightcoral', alpha=.5)
        plt.fill_between(range(0, 301), heu_sig[0], heu_sig[1], color='lightskyblue', alpha=.5)
        plt.fill_between(range(0, 301), r_sig[0], r_sig[1], color='limegreen', alpha=.5)
        plt.legend()
        plt.xlabel('Iteration Step')
        plt.ylabel('Schedule Cost')
        plt.title(data_config)
        plt.show()
