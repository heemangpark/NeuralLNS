import copy
import os
import random
import shutil
import sys

import networkx as nx
import numpy as np
import torch
from matplotlib import pyplot as plt
from torch_geometric.data import Data, HeteroData
from torch_geometric.loader import DataLoader
from tqdm import tqdm, trange

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from src.heuristic.hungarian import hungarian
from src.heuristic.regret import f_ijk
from src.heuristic.shaw import removal
from src.model.cross_attention import CrossAttention
from src.model.pyg_mpnn import MPNN
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


def pyg(graph_type: str):
    if graph_type == 'homo':
        for data_type in ['train', 'val', 'test']:
            data_list_A, data_list_M, data_list_P = [], [], []
            scenarios = torch.load('datas/scenarios/8_8_20_5_5/{}.pt'.format(data_type))

            for scen in tqdm(scenarios):
                grid, graph, a_coord, t_coord = scen

                x = torch.cat((torch.FloatTensor(a_coord), torch.FloatTensor(t_coord))) / grid.shape[0]

                src, dst = [], []
                for a_id in range(len(a_coord)):
                    for t_id in range(len(a_coord), len(a_coord) + len(t_coord)):
                        src.extend([a_id, t_id])
                        dst.extend([t_id, a_id])
                edge_index = torch.LongTensor([src, dst])

                A, M, P = [], [], []
                for _a in a_coord:
                    for _t in t_coord:
                        astar = nx.astar_path_length(graph, tuple(_a), tuple(_t)) / grid.shape[0]
                        man = sum(abs(np.array(_a) - np.array(_t))) / grid.shape[0]
                        proxy = astar - man
                        A.extend([astar] * 2)
                        M.extend([man] * 2)
                        P.extend([proxy] * 2)

                edge_attr_A = torch.FloatTensor(A).view(-1, 1)
                edge_attr_M = torch.FloatTensor(M).view(-1, 1)
                edge_attr_P = torch.FloatTensor(P).view(-1, 1)

                data_list_A.append(Data(x=x, edge_index=edge_index, edge_attr=edge_attr_A))
                data_list_M.append(Data(x=x, edge_index=edge_index, edge_attr=edge_attr_M))
                data_list_P.append(Data(x=x, edge_index=edge_index, edge_attr=edge_attr_P))

            torch.save(data_list_A, 'datas/pyg/8_8_20_5_5/{}/A.pt'.format(data_type))
            torch.save(data_list_M, 'datas/pyg/8_8_20_5_5/{}/M.pt'.format(data_type))
            torch.save(data_list_P, 'datas/pyg/8_8_20_5_5/{}/P.pt'.format(data_type))

    elif graph_type == 'hetero':
        for data_type in ['train', 'val', 'test']:
            data_list = []
            scenarios = torch.load('datas/scenarios/8_8_20_5_5/{}.pt'.format(data_type))

            for scen in tqdm(scenarios):
                grid, graph, a_coord, t_coord = scen

                src, dst = [], []
                for a_id in range(len(a_coord)):
                    for t_id in range(len(a_coord), len(a_coord) + len(t_coord)):
                        src.extend([a_id, t_id])
                        dst.extend([t_id, a_id])
                edge_index = torch.LongTensor([src, dst])

                A, M, P = [], [], []
                for _a in a_coord:
                    for _t in t_coord:
                        astar = nx.astar_path_length(graph, tuple(_a), tuple(_t)) / grid.shape[0]
                        man = sum(abs(np.array(_a) - np.array(_t))) / grid.shape[0]
                        proxy = astar - man
                        A.extend([astar] * 2)
                        M.extend([man] * 2)
                        P.extend([proxy] * 2)

                edge_attr_1 = torch.FloatTensor(A).view(-1, 1)
                edge_attr_2 = torch.FloatTensor(M).view(-1, 1)
                edge_attr_3 = torch.FloatTensor(P).view(-1, 1)

                data = HeteroData()
                data['agent'].x = torch.FloatTensor(a_coord) / grid.shape[0]
                data['task'].x = torch.FloatTensor(t_coord) / grid.shape[0]

                data['agent', 'astar', 'task'].edge_index = edge_index
                data['task', 'astar', 'agent'].edge_index = edge_index
                data['agent', 'astar', 'task'].edge_attr = edge_attr_1
                data['task', 'astar', 'agent'].edge_attr = edge_attr_1

                data['agent', 'man', 'task'].edge_index = edge_index
                data['task', 'man', 'agent'].edge_index = edge_index
                data['agent', 'man', 'task'].edge_attr = edge_attr_2
                data['task', 'man', 'agent'].edge_attr = edge_attr_2

                data['agent', 'proxy', 'task'].edge_index = edge_index
                data['task', 'proxy', 'agent'].edge_index = edge_index
                data['agent', 'proxy', 'task'].edge_attr = edge_attr_3
                data['task', 'proxy', 'agent'].edge_attr = edge_attr_3

                data_list.append(data)

            torch.save(data_list, 'datas/pyg/8_8_20_5_5/{}/hetero.pt'.format(data_type))

    else:
        raise ValueError('supports only homogeneous and heterogeneous')


def run():
    seed_everything(seed=42)
    train_data = torch.load('datas/pyg/8_8_20_5_5/train/P.pt')
    train_loader = DataLoader(train_data, batch_size=100, shuffle=True)

    gnn = MPNN(n_enc_dim=2, e_enc_dim=1, model_dim=32, num_layers=3)  # config/model/gnn.yaml
    attn = CrossAttention(embed_dim=32, num_heads=4)  # config/model/attention.yaml

    # Training Loop
    for e in trange(100):  # config/main/temp.yaml_epochs
        for batch in train_loader:
            hidden = gnn(batch)
            hidden_a, hidden_t = hidden[:, :5, :], hidden[:, 5:, :]

            output = torch.bmm(hidden_a, hidden_t.transpose(1, -1))
            # output = attn(a_emb, t_emb, t_emb)

            print(output)


if __name__ == '__main__':
    # lns_itr_test(OmegaConf.load('config/lns_itr_test.yaml'))
    # pyg(graph_type='homo')
    # pyg(graph_type='hetero')
    run()
