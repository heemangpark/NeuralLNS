import copy
import os
import random
import shutil
import sys
from datetime import datetime

import networkx as nx
import numpy as np
import torch
from matplotlib import pyplot as plt
from omegaconf import OmegaConf
from torch_geometric.data import Data, HeteroData
from torch_geometric.loader import DataLoader
from tqdm import tqdm, trange

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from src.heuristic.hungarian import hungarian
from src.heuristic.regret import f_ijk
from src.heuristic.shaw import removal
from src.model.pyg_mpnn import MPNN
from utils.scenario import load_scenarios
from utils.seed import seed_everything
from utils.solver import solver


def lns_itr_test(config):
    seed_everything(config.seed)

    for itrs in [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]:
        gap = []

        for t in tqdm(list(random.sample(list(range(config.num_data)), k=100))):
            grid, grid_graph, a_coord, t_coord = load_scenarios(
                '{}{}{}_{}_{}/scenario_{}.pkl'.format(config.map_size, config.map_size, config.obs_ratio,
                                                      config.num_agent, config.num_task, t))
            assign_idx, assign_coord = hungarian(a_coord, t_coord)

            actual_init_cost, _ = solver(grid, a_coord, assign_coord, save_dir=config.save_dir,
                                         exp_name='init' + str(t))
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

            actual_final_cost, _ = solver(grid, a_coord, assign_coord, save_dir=config.save_dir,
                                          exp_name='fin' + str(t))

            if os.path.exists(config.save_dir):
                shutil.rmtree(config.save_dir)

            perf = (actual_init_cost - actual_final_cost) / actual_init_cost * 100
            gap.append(perf)

        plt.plot(gap)
        plt.title('{:.4f}'.format(np.mean(gap)))
        plt.savefig('itrs_{}.png'.format(itrs))
        plt.clf()


def pyg_data(scen_config: str, graph_type: str = 'homo'):
    seed_everything(seed=42)
    if graph_type == 'homo':
        for data_type in ['train', 'val', 'test']:

            scenarios = torch.load('datas/scenarios/{}/{}.pt'.format(scen_config, data_type))
            save_dir = 'datas/pyg/{}/'.format(scen_config)
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)

            data_list = []
            for scen in tqdm(scenarios):
                grid, graph, a_coord, t_coord, y = scen
                node_list = a_coord + t_coord

                # node feature
                x = torch.FloatTensor(node_list) / grid.shape[0]

                # edge index
                edge_index = []
                for i in range(len(node_list)):
                    for j in range(i, len(node_list)):
                        if i == j:
                            pass
                        else:
                            edge_index.append([i, j])

                # edge features
                A, M, O = [], [], []
                for i, j in edge_index:
                    astar = nx.astar_path_length(graph, tuple(node_list[i]), tuple(node_list[j])) / grid.shape[0]
                    manhattan = sum(abs(np.array(node_list[i]) - np.array(node_list[j]))) / grid.shape[0]
                    obstacle = (astar - manhattan) / grid.shape[0]
                    A.append(astar)
                    M.append(manhattan)
                    O.append(obstacle)

                edge_attr = torch.cat((torch.FloatTensor(A).view(-1, 1),
                                       torch.FloatTensor(M).view(-1, 1),
                                       torch.FloatTensor(O).view(-1, 1)), dim=-1)

                edge_index = torch.LongTensor(edge_index).transpose(-1, 0)

                # label data
                y = torch.FloatTensor(y)

                data_list.append(Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y))

            torch.save(data_list, save_dir + '{}.pt'.format(data_type))

    elif graph_type == 'hetero':
        for data_type in ['train', 'val', 'test']:

            data_list = []
            scenarios = torch.load('datas/scenarios/{}/{}.pt'.format(scen_config, data_type))

            for scen in tqdm(scenarios):
                grid, graph, a_coord, t_coord, y = scen

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

                data = HeteroData()
                data['agent'].x = torch.FloatTensor(a_coord) / grid.shape[0]
                data['task'].x = torch.FloatTensor(t_coord) / grid.shape[0]

                data['agent', 'astar', 'task'].edge_index = edge_index
                data['agent', 'astar', 'task'].edge_attr = torch.FloatTensor(A).view(-1, 1)
                data['agent', 'astar', 'task'].y = torch.Tensor(y)

                data['task', 'astar', 'agent'].edge_index = edge_index
                data['task', 'astar', 'agent'].edge_attr = torch.FloatTensor(A).view(-1, 1)
                data['task', 'astar', 'agent'].y = torch.Tensor(y)

                data['agent', 'man', 'task'].edge_index = edge_index
                data['agent', 'man', 'task'].edge_attr = torch.FloatTensor(M).view(-1, 1)
                data['agent', 'man', 'task'].y = torch.Tensor(y)

                data['task', 'man', 'agent'].edge_index = edge_index
                data['task', 'man', 'agent'].edge_attr = torch.FloatTensor(M).view(-1, 1)
                data['task', 'man', 'agent'].y = torch.Tensor(y)

                data['agent', 'proxy', 'task'].edge_index = edge_index
                data['agent', 'proxy', 'task'].edge_attr = torch.FloatTensor(P).view(-1, 1)
                data['agent', 'proxy', 'task'].y = torch.Tensor(y)

                data['task', 'proxy', 'agent'].edge_index = edge_index
                data['task', 'proxy', 'agent'].edge_attr = torch.FloatTensor(P).view(-1, 1)
                data['task', 'proxy', 'agent'].y = torch.Tensor(y)

                data_list.append(data)
            torch.save(data_list, 'datas/pyg/{}/{}/hetero.pt'.format(scen_config, data_type))

    else:
        raise ValueError('supports only homogeneous and heterogeneous graphs')


def run(device: str, edge_type: str, logging: bool = False):
    seed_everything(seed=42)
    configs = OmegaConf.load('config/experiment/edge_test.yaml')

    date = datetime.now().strftime("%m%d_%H%M%S")
    model_dir = 'datas/models/{}_edge_test/'.format(date)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    train_data = torch.load('datas/pyg/{}/train.pt'.format(configs.map), map_location=device)
    train_loader = DataLoader(train_data, batch_size=configs.batch_size, shuffle=True)
    val_data = torch.load('datas/pyg/{}/val.pt'.format(configs.map), map_location=device)
    val_loader = DataLoader(val_data, batch_size=configs.batch_size, shuffle=True)

    if logging:
        import wandb
        wandb.init(project='NeuralLNS', name='{}_{}'.format(edge_type, date))

    GNN = MPNN(configs, edge_type).to(device)

    for e in trange(100):
        train_loss, num_batch = 0, 0

        for train in train_loader:
            batch_loss = GNN(train, edge_type)
            train_loss += batch_loss
            num_batch += 1
        train_loss /= num_batch

        if logging:
            wandb.log({'train_loss': train_loss})

        if (e + 1) % 10 == 0:
            torch.save(GNN.state_dict(), model_dir + '{}_{}.pt'.format(edge_type, e + 1))
            val_GNN = MPNN(configs, edge_type).to(device)
            val_GNN.load_state_dict(torch.load(model_dir + '{}_{}.pt'.format(edge_type, e + 1)))
            val_GNN.eval()

            val_loss, num_batch = 0, 0
            for val in val_loader:
                batch_loss = val_GNN(val, edge_type)
                val_loss += batch_loss
                num_batch += 1
            val_loss /= num_batch

            if logging:
                wandb.log({'val_loss': val_loss})


if __name__ == '__main__':
    # pyg_data('16_16_20_20_20')
    # pyg_data('16_16_partition_20_20')
    run(device='cuda:3', edge_type='AMP', logging=False)
