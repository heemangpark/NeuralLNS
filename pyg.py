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
            ################################################################
            scenarios = scenarios[:len(scenarios) // 10]
            ################################################################
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


def run(logging: bool = False):
    seed_everything(seed=42)
    configs = OmegaConf.load('config/experiment/edge_test.yaml')

    date = datetime.now().strftime("%m%d_%H%M%S")
    model_dir = 'datas/models/{}_edge_test/'.format(date)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    train_data = torch.load('datas/pyg/{}/train.pt'.format(configs.map), map_location=configs.device)
    train_loader = DataLoader(train_data, batch_size=configs.batch_size, shuffle=True)
    val_data = torch.load('datas/pyg/{}/val.pt'.format(configs.map), map_location=configs.device)
    val_loader = DataLoader(val_data, batch_size=configs.batch_size, shuffle=True)

    if logging:
        import wandb
        wandb.init(project='NeuralLNS', name=date)

    GNN_ones, GNN_A, GNN_M = [MPNN(configs, edge_type=1) for _ in range(3)]
    GNN_AP, GNN_MP = [MPNN(configs, edge_type=2) for _ in range(2)]
    GNN_AMP = MPNN(configs, edge_type=3)

    for e in trange(100):
        num_batch, e_ones, e_A, e_M, e_AP, e_MP, e_AMP = [0 for _ in range(7)]

        for train in train_loader:
            b_ones = GNN_ones(train, type='ones')
            b_A = GNN_A(train, type='A')
            b_M = GNN_M(train, type='M')
            b_AP = GNN_AP(train, type='AP')
            b_MP = GNN_MP(train, type='MP')
            b_AMP = GNN_AMP(train, type='AMP')
            e_ones += b_ones
            e_A += b_A
            e_M += b_M
            e_AP += b_AP
            e_MP += b_MP
            e_AMP += b_AMP
            num_batch += 1

        e_ones /= num_batch
        e_A /= num_batch
        e_M /= num_batch
        e_AP /= num_batch
        e_MP /= num_batch
        e_AMP /= num_batch

        if logging:
            wandb.log({'one_train_loss': e_ones,
                       'A_train_loss': e_A,
                       'M_train_loss': e_M,
                       'AP_train_loss': e_AP,
                       'MP_train_loss': e_MP,
                       'AMP_train_loss': e_AMP})

        if (e + 1) % 10 == 0:
            torch.save(GNN_ones.state_dict(), model_dir + 'ones_{}.pt'.format(e + 1))
            torch.save(GNN_A.state_dict(), model_dir + 'A_{}.pt'.format(e + 1))
            torch.save(GNN_M.state_dict(), model_dir + 'M_{}.pt'.format(e + 1))
            torch.save(GNN_AP.state_dict(), model_dir + 'AP_{}.pt'.format(e + 1))
            torch.save(GNN_MP.state_dict(), model_dir + 'MP_{}.pt'.format(e + 1))
            torch.save(GNN_AMP.state_dict(), model_dir + 'AMP_{}.pt'.format(e + 1))

            val_GNN_ones, val_GNN_A, val_GNN_M = [MPNN(configs, edge_type=1) for _ in range(3)]
            val_GNN_AP, val_GNN_MP = [MPNN(configs, edge_type=2) for _ in range(2)]
            val_GNN_AMP = MPNN(configs, edge_type=3)

            val_GNN_ones.load_state_dict(torch.load(model_dir + 'ones_{}.pt'.format(e + 1)))
            val_GNN_A.load_state_dict(torch.load(model_dir + 'A_{}.pt'.format(e + 1)))
            val_GNN_M.load_state_dict(torch.load(model_dir + 'M_{}.pt'.format(e + 1)))
            val_GNN_AP.load_state_dict(torch.load(model_dir + 'AP_{}.pt'.format(e + 1)))
            val_GNN_MP.load_state_dict(torch.load(model_dir + 'MP_{}.pt'.format(e + 1)))
            val_GNN_AMP.load_state_dict(torch.load(model_dir + 'AMP_{}.pt'.format(e + 1)))

            val_GNN_ones.eval()
            val_GNN_A.eval()
            val_GNN_M.eval()
            val_GNN_AP.eval()
            val_GNN_MP.eval()
            val_GNN_AMP.eval()

            num_batch, v_ones, v_A, v_M, v_AP, v_MP, v_AMP = [0 for _ in range(7)]
            for val in val_loader:
                v_b_ones = val_GNN_ones(val, type='ones')
                v_b_A = val_GNN_A(val, type='A')
                v_b_M = val_GNN_M(val, type='M')
                v_b_AP = val_GNN_AP(val, type='AP')
                v_b_MP = val_GNN_MP(val, type='MP')
                v_b_AMP = val_GNN_AMP(val, type='AMP')

                v_ones += v_b_ones
                v_A += v_b_A
                v_M += v_b_M
                v_AP += v_b_AP
                v_MP += v_b_MP
                v_AMP += v_b_AMP
                num_batch += 1

            v_ones /= num_batch
            v_A /= num_batch
            v_M /= num_batch
            v_AP /= num_batch
            v_MP /= num_batch
            v_AMP /= num_batch

            if logging:
                wandb.log({'ones_val_loss': v_ones,
                           'A_val_loss': v_A,
                           'M_val_loss': v_M,
                           'AP_val_loss': v_AP,
                           'MP_val_loss': v_MP,
                           'AMP_val_loss': v_AMP})


if __name__ == '__main__':
    # pyg_data('16_16_20_20_20')
    # pyg_data('16_16_partition_20_20')
    run(logging=True)

    # configs = OmegaConf.load('config/experiment/edge_test.yaml')
    # test_data = torch.load('datas/pyg/16_16_20_20_20/test.pt', map_location='cuda:3')
    # test_loader = DataLoader(test_data, batch_size=1000, shuffle=True)
    # GNN_A, GNN_M = MPNN(configs, edge_type=1), MPNN(configs, edge_type=1)
    # GNN_A.load_state_dict(torch.load('datas/models/0705_163312_edge_test/A_100.pt'))
    # GNN_M.load_state_dict(torch.load('datas/models/0705_163312_edge_test/M_100.pt'))
    # GNN_A.eval(), GNN_M.eval()
    # for test in test_loader:
    #     A, M = GNN_A(test, type='A'), GNN_M(test, type='M')
