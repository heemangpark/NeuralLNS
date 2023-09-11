import copy
import os
import pickle
import random
import sys
import time
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


def divide_map(a_t_ratio, schedule, coords, edge_index, types):
    agent_task_list = [[] for _ in range(a_t_ratio + 1)]
    for idx in range(a_t_ratio + 1):
        agent_task_list[idx].extend([sch[idx] for sch in schedule])

    nf_coord = torch.Tensor(coords)
    nf_coord = (nf_coord - nf_coord.min()) / (nf_coord.max() + nf_coord.min())
    edge_index = torch.LongTensor(edge_index).transpose(-1, 0)

    mask_type = [copy.deepcopy(types) for _ in range(a_t_ratio)]

    if a_t_ratio == 3:
        for a, t1, t2, t3 in zip(agent_task_list[0], agent_task_list[1],
                                 agent_task_list[2], agent_task_list[3]):
            mask_type[0][t2] = 0
            mask_type[0][t3] = 0

            mask_type[1][a] = 0
            mask_type[1][t1] = 2
            mask_type[1][t3] = 0

            mask_type[2][a] = 0
            mask_type[2][t1] = 0
            mask_type[2][t2] = 2

        nf_1 = torch.cat((nf_coord, torch.eye(4)[mask_type[0]]), -1)
        nf_2 = torch.cat((nf_coord, torch.eye(4)[mask_type[1]]), -1)
        nf_3 = torch.cat((nf_coord, torch.eye(4)[mask_type[2]]), -1)

        sch_1 = [(sch[0], sch[1]) for sch in schedule]
        sch_2 = [(sch[1], sch[2]) for sch in schedule]
        sch_3 = [(sch[2], sch[3]) for sch in schedule]

        map_data = [Data(x=nf_1, edge_index=edge_index, edge_attr=torch.ones(edge_index.shape[-1], 1),
                         schedule=torch.LongTensor(sch_1)),
                    Data(x=nf_2, edge_index=edge_index, edge_attr=torch.ones(edge_index.shape[-1], 1),
                         schedule=torch.LongTensor(sch_2)),
                    Data(x=nf_3, edge_index=edge_index, edge_attr=torch.ones(edge_index.shape[-1], 1),
                         schedule=torch.LongTensor(sch_3))]

    elif a_t_ratio == 4:
        for a, t1, t2, t3, t4 in zip(agent_task_list[0], agent_task_list[1],
                                     agent_task_list[2], agent_task_list[3], agent_task_list[4]):
            mask_type[0][t2] = 0
            mask_type[0][t3] = 0
            mask_type[0][t4] = 0

            mask_type[1][a] = 0
            mask_type[1][t1] = 2
            mask_type[1][t3] = 0
            mask_type[1][t4] = 0

            mask_type[2][a] = 0
            mask_type[2][t1] = 0
            mask_type[2][t2] = 2
            mask_type[2][t4] = 0

            mask_type[3][a] = 0
            mask_type[3][t1] = 0
            mask_type[3][t2] = 0
            mask_type[3][t3] = 2

        nf_1 = torch.cat((nf_coord, torch.eye(4)[mask_type[0]]), -1)
        nf_2 = torch.cat((nf_coord, torch.eye(4)[mask_type[1]]), -1)
        nf_3 = torch.cat((nf_coord, torch.eye(4)[mask_type[2]]), -1)
        nf_4 = torch.cat((nf_coord, torch.eye(4)[mask_type[3]]), -1)

        sch_1 = [(sch[0], sch[1]) for sch in schedule]
        sch_2 = [(sch[1], sch[2]) for sch in schedule]
        sch_3 = [(sch[2], sch[3]) for sch in schedule]
        sch_4 = [(sch[3], sch[4]) for sch in schedule]

        map_data = [Data(x=nf_1, edge_index=edge_index, edge_attr=torch.ones(edge_index.shape[-1], 1),
                         schedule=torch.LongTensor(sch_1)),
                    Data(x=nf_2, edge_index=edge_index, edge_attr=torch.ones(edge_index.shape[-1], 1),
                         schedule=torch.LongTensor(sch_2)),
                    Data(x=nf_3, edge_index=edge_index, edge_attr=torch.ones(edge_index.shape[-1], 1),
                         schedule=torch.LongTensor(sch_3)),
                    Data(x=nf_4, edge_index=edge_index, edge_attr=torch.ones(edge_index.shape[-1], 1),
                         schedule=torch.LongTensor(sch_4))]

    elif a_t_ratio == 5:
        for a, t1, t2, t3, t4, t5 in zip(agent_task_list[0], agent_task_list[1], agent_task_list[2],
                                         agent_task_list[3], agent_task_list[4], agent_task_list[5]):
            mask_type[0][t2] = 0
            mask_type[0][t3] = 0
            mask_type[0][t4] = 0
            mask_type[0][t5] = 0

            mask_type[1][a] = 0
            mask_type[1][t1] = 2
            mask_type[1][t3] = 0
            mask_type[1][t4] = 0
            mask_type[1][t5] = 0

            mask_type[2][a] = 0
            mask_type[2][t1] = 0
            mask_type[2][t2] = 2
            mask_type[2][t4] = 0
            mask_type[2][t5] = 0

            mask_type[3][a] = 0
            mask_type[3][t1] = 0
            mask_type[3][t2] = 0
            mask_type[3][t3] = 2
            mask_type[3][t5] = 0

            mask_type[4][a] = 0
            mask_type[4][t1] = 0
            mask_type[4][t2] = 0
            mask_type[4][t3] = 0
            mask_type[4][t4] = 2

        nf_1 = torch.cat((nf_coord, torch.eye(4)[mask_type[0]]), -1)
        nf_2 = torch.cat((nf_coord, torch.eye(4)[mask_type[1]]), -1)
        nf_3 = torch.cat((nf_coord, torch.eye(4)[mask_type[2]]), -1)
        nf_4 = torch.cat((nf_coord, torch.eye(4)[mask_type[3]]), -1)
        nf_5 = torch.cat((nf_coord, torch.eye(4)[mask_type[4]]), -1)

        sch_1 = [(sch[0], sch[1]) for sch in schedule]
        sch_2 = [(sch[1], sch[2]) for sch in schedule]
        sch_3 = [(sch[2], sch[3]) for sch in schedule]
        sch_4 = [(sch[3], sch[4]) for sch in schedule]
        sch_5 = [(sch[4], sch[5]) for sch in schedule]

        map_data = [Data(x=nf_1, edge_index=edge_index, edge_attr=torch.ones(edge_index.shape[-1], 1),
                         schedule=torch.LongTensor(sch_1)),
                    Data(x=nf_2, edge_index=edge_index, edge_attr=torch.ones(edge_index.shape[-1], 1),
                         schedule=torch.LongTensor(sch_2)),
                    Data(x=nf_3, edge_index=edge_index, edge_attr=torch.ones(edge_index.shape[-1], 1),
                         schedule=torch.LongTensor(sch_3)),
                    Data(x=nf_4, edge_index=edge_index, edge_attr=torch.ones(edge_index.shape[-1], 1),
                         schedule=torch.LongTensor(sch_4)),
                    Data(x=nf_5, edge_index=edge_index, edge_attr=torch.ones(edge_index.shape[-1], 1),
                         schedule=torch.LongTensor(sch_5))]

    elif a_t_ratio == 6:
        for a, t1, t2, t3, t4, t5, t6 in zip(agent_task_list[0], agent_task_list[1], agent_task_list[2],
                                             agent_task_list[3], agent_task_list[4], agent_task_list[5],
                                             agent_task_list[6]):
            mask_type[0][t2] = 0
            mask_type[0][t3] = 0
            mask_type[0][t4] = 0
            mask_type[0][t5] = 0
            mask_type[0][t6] = 0

            mask_type[1][a] = 0
            mask_type[1][t1] = 2
            mask_type[1][t3] = 0
            mask_type[1][t4] = 0
            mask_type[1][t5] = 0
            mask_type[1][t6] = 0

            mask_type[2][a] = 0
            mask_type[2][t1] = 0
            mask_type[2][t2] = 2
            mask_type[2][t4] = 0
            mask_type[2][t5] = 0
            mask_type[2][t6] = 0

            mask_type[3][a] = 0
            mask_type[3][t1] = 0
            mask_type[3][t2] = 0
            mask_type[3][t3] = 2
            mask_type[3][t5] = 0
            mask_type[3][t6] = 0

            mask_type[4][a] = 0
            mask_type[4][t1] = 0
            mask_type[4][t2] = 0
            mask_type[4][t3] = 0
            mask_type[4][t4] = 2
            mask_type[4][t6] = 0

            mask_type[5][a] = 0
            mask_type[5][t1] = 0
            mask_type[5][t2] = 0
            mask_type[5][t3] = 0
            mask_type[5][t4] = 0
            mask_type[5][t5] = 2

        nf_1 = torch.cat((nf_coord, torch.eye(4)[mask_type[0]]), -1)
        nf_2 = torch.cat((nf_coord, torch.eye(4)[mask_type[1]]), -1)
        nf_3 = torch.cat((nf_coord, torch.eye(4)[mask_type[2]]), -1)
        nf_4 = torch.cat((nf_coord, torch.eye(4)[mask_type[3]]), -1)
        nf_5 = torch.cat((nf_coord, torch.eye(4)[mask_type[4]]), -1)
        nf_6 = torch.cat((nf_coord, torch.eye(4)[mask_type[5]]), -1)

        sch_1 = [(sch[0], sch[1]) for sch in schedule]
        sch_2 = [(sch[1], sch[2]) for sch in schedule]
        sch_3 = [(sch[2], sch[3]) for sch in schedule]
        sch_4 = [(sch[3], sch[4]) for sch in schedule]
        sch_5 = [(sch[4], sch[5]) for sch in schedule]
        sch_6 = [(sch[5], sch[6]) for sch in schedule]

        map_data = [Data(x=nf_1, edge_index=edge_index, edge_attr=torch.ones(edge_index.shape[-1], 1),
                         schedule=torch.LongTensor(sch_1)),
                    Data(x=nf_2, edge_index=edge_index, edge_attr=torch.ones(edge_index.shape[-1], 1),
                         schedule=torch.LongTensor(sch_2)),
                    Data(x=nf_3, edge_index=edge_index, edge_attr=torch.ones(edge_index.shape[-1], 1),
                         schedule=torch.LongTensor(sch_3)),
                    Data(x=nf_4, edge_index=edge_index, edge_attr=torch.ones(edge_index.shape[-1], 1),
                         schedule=torch.LongTensor(sch_4)),
                    Data(x=nf_5, edge_index=edge_index, edge_attr=torch.ones(edge_index.shape[-1], 1),
                         schedule=torch.LongTensor(sch_5)),
                    Data(x=nf_6, edge_index=edge_index, edge_attr=torch.ones(edge_index.shape[-1], 1),
                         schedule=torch.LongTensor(sch_6))]

    else:
        raise NotImplementedError('Supports Agent Task Ratio from 3~6 only')

    return map_data


def test_pred(obs_dens, map_con, device, single=False):
    seed_everything()
    config = OmegaConf.load('config/experiment/rt.yaml')
    if not single:
        test_data = torch.load('data/{}/{}.pt'.format(obs_dens, map_con), map_location=device)
    else:
        test_data = torch.load('data/single/{}_{}.pt'.format(obs_dens, map_con), map_location=device)

    gnn = RelationalTransformer(config).to(device)
    gnn.load_state_dict(torch.load('data/models/rt_8_sum.pt'))
    gnn.eval()

    pred_list = []
    label_list = [td[-1].item() for td in test_data]
    label_list = np.array(label_list)
    label_list = (label_list - label_list.mean()) / (label_list.std() + 1e-5)
    label_list = label_list.tolist()

    batch_size = 100
    for b_id in trange(len(test_data) // batch_size):
        batch_data = []

        for td in test_data[b_id * batch_size: (b_id + 1) * batch_size]:
            coords, types, edge_index, schedule, _ = td
            map_data = divide_map(int(map_con.split('_')[2]) // int(map_con.split('_')[1]),
                                  schedule, coords, edge_index, types)
            batch_data += map_data

        data = Batch.from_data_list(batch_data).to(device)
        pred = gnn(data, test=True)
        pred = rearrange(pred, '(N C) -> N C', N=batch_size)
        pred_list.extend(pred.sum(-1))

    return pred_list, label_list


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
                modified_schedule_set.append(modified_schedule)
                cost_set.append(cost)
                break
            else:
                step_count -= 1

    return modified_schedule_set[cost_set.index(min(cost_set))], min(cost_set)


def run_nn(data, device, sample_per_steps=50, config='', max_time=120, time_step=3):
    time_list = list(range(time_step, max_time + 1, time_step))
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

    total_time = 0
    while True:
        while True:
            algo_start = time.time()
            next_schedule = NLNS(map_data, schedule, device, sample_per_steps)
            algo_end = time.time()
            algo_spend = algo_end - algo_start
            total_time += algo_spend

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

        if total_time >= time_list[0]:
            output.append(cost)
            time_list.pop(0)

            if len(time_list) == 0:
                return output


def run_heu(data, sample_per_steps: int = 100, ran=False, config='', max_time=120, time_step=3):
    time_list = list(range(time_step, max_time + 1, time_step))
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

    total_time = 0
    while True:
        if ran:
            algo_start = time.time()
            next_schedule, next_cost = random_LNS(map_data, schedule, sample_per_steps, config)
            algo_end = time.time()
            algo_spend = algo_end - algo_start
            total_time += algo_spend
        else:
            algo_start = time.time()
            next_schedule, next_cost = LNS(map_data, schedule, sample_per_steps, config)
            algo_end = time.time()
            algo_spend = algo_end - algo_start
            total_time += algo_spend

        if cost > next_cost:
            schedule = next_schedule
            cost = next_cost
        else:
            pass

        if total_time >= time_list[0]:
            output.append(cost)
            time_list.pop(0)

            if len(time_list) == 0:
                return output


if __name__ == '__main__':
    seed_everything()

    # mode = 'exp'
    # mode = 'plot'
    mode = 'test_pred'

    cuda = 'cuda:3'

    obs_density = ['sparse', 'medium', 'dense']
    map_config = ['88_5_15', '88_5_20', '88_5_25', '88_5_30']
    # map_config = ['1616_5_15', '1616_5_20', '1616_5_25', '1616_5_30']
    # map_config = ['88_5_5', '88_10_10', '88_15_15']

    if mode == 'exp':
        from tqdm import tqdm

        for od in obs_density:
            for mc in map_config:

                datadir = torch.load('data/{}/{}.pt'.format(od, mc))
                map_id = random.sample(range(10000), 5)

                nn_res, heu_res, r_res = [], [], []

                for m_id in tqdm(map_id):
                    nn = run_nn(data=datadir[m_id], device=cuda, config=od + '_' + mc)
                    nn_res.append(nn)
                    heu = run_heu(data=datadir[m_id], config=od + '_' + mc)
                    heu_res.append(heu)
                    r = run_heu(data=datadir[m_id], ran=True, config=od + '_' + mc)
                    r_res.append(r)

                with open('result/nn' + '_' + od + '_' + mc, 'wb') as fp:
                    pickle.dump(nn_res, fp)
                with open('result/heu' + '_' + od + '_' + mc, 'wb') as fp:
                    pickle.dump(heu_res, fp)
                with open('result/r' + '_' + od + '_' + mc, 'wb') as fp:
                    pickle.dump(r_res, fp)

    elif mode == 'plot':
        for od in obs_density:
            for mc in map_config:
                with open('result/nn' + '_' + od + '_' + mc, 'rb') as fp:
                    nn = pickle.load(fp)
                with open('result/heu' + '_' + od + '_' + mc, 'rb') as fp:
                    heu = pickle.load(fp)
                with open('result/r' + '_' + od + '_' + mc, 'rb') as fp:
                    r = pickle.load(fp)

                nn_sig = np.array(nn).mean(0) + np.array(nn).std(0) * .2, np.array(nn).mean(0) - np.array(nn).std(
                    0) * .2
                heu_sig = np.array(heu).mean(0) + np.array(heu).std(0) * .2, np.array(heu).mean(0) - np.array(heu).std(
                    0) * .2
                r_sig = np.array(r).mean(0) + np.array(r).std(0) * .2, np.array(r).mean(0) - np.array(r).std(0) * .2

                init_cost = np.array(nn).mean(0)[0]

                plt.clf()
                plt.plot(range(0, 121, 3), np.array(nn).mean(0), 'r-', label='NN')
                plt.plot(range(0, 121, 3), np.array(heu).mean(0), 'b-', label='Heuristic')
                plt.plot(range(0, 121, 3), np.array(r).mean(0), 'g-', label='Random')
                plt.fill_between(range(0, 121, 3), nn_sig[0], nn_sig[1], color='lightcoral', alpha=.5)
                plt.fill_between(range(0, 121, 3), heu_sig[0], heu_sig[1], color='lightskyblue', alpha=.5)
                plt.fill_between(range(0, 121, 3), r_sig[0], r_sig[1], color='limegreen', alpha=.5)
                plt.legend()
                plt.xlabel('Computation Time(s)')
                plt.ylabel('Schedule Cost')
                plt.title(od + '_' + mc)
                plt.show()

    else:
        plt.clf()
        plot_idx = 0
        for od in obs_density:
            for mc in map_config:
                plot_idx += 1
                plt.subplot(len(obs_density), len(map_config), plot_idx)
                x, y = test_pred(od, mc, cuda, single=False)
                plt.plot(x, y, 'b.', alpha=.5)
                plt.xlabel('preds')
                plt.ylabel('labels')
                plt.title('{}_{}'.format(od, mc))
        plt.tight_layout()
        plt.show()
