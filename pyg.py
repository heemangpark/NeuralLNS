import argparse
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
from src.model.attention import MultiHeadCrossAttention
from src.model.pyg_mpnn import MPNN
from utils.prefetch_loader import PrefetchLoader
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


def pyg_data(graph_type: str):
    if graph_type == 'homo':
        for data_type in ['train', 'val', 'test']:
            data_list_A, data_list_M, data_list_P = [], [], []
            scenarios = torch.load('datas/scenarios/8_8_20_5_5/{}.pt'.format(data_type))

            for scen in tqdm(scenarios):
                grid, graph, a_coord, t_coord, y = scen

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

                data_A = Data(x=x, edge_index=edge_index, edge_attr=edge_attr_A, y=torch.Tensor(y))
                data_A.pin_memory()
                data_A.to('cuda', non_blocking=True)
                data_list_A.append(data_A)

                data_M = Data(x=x, edge_index=edge_index, edge_attr=edge_attr_M, y=torch.Tensor(y))
                data_M.pin_memory()
                data_M.to('cuda', non_blocking=True)
                data_list_M.append(data_M)

                data_P = Data(x=x, edge_index=edge_index, edge_attr=edge_attr_P, y=torch.Tensor(y))
                data_P.pin_memory()
                data_P.to('cuda', non_blocking=True)
                data_list_P.append(data_P)

            torch.save(data_list_A, 'datas/pyg/8_8_20_5_5/{}/A.pt'.format(data_type))
            torch.save(data_list_M, 'datas/pyg/8_8_20_5_5/{}/M.pt'.format(data_type))
            torch.save(data_list_P, 'datas/pyg/8_8_20_5_5/{}/P.pt'.format(data_type))

    elif graph_type == 'hetero':
        for data_type in ['train', 'val', 'test']:
            data_list = []
            scenarios = torch.load('datas/scenarios/8_8_20_5_5/{}.pt'.format(data_type))

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

                edge_attr_1 = torch.FloatTensor(A).view(-1, 1)
                edge_attr_2 = torch.FloatTensor(M).view(-1, 1)
                edge_attr_3 = torch.FloatTensor(P).view(-1, 1)

                data = HeteroData()
                data['agent'].x = torch.FloatTensor(a_coord) / grid.shape[0]
                data['task'].x = torch.FloatTensor(t_coord) / grid.shape[0]

                data['agent', 'astar', 'task'].edge_index = edge_index
                data['agent', 'astar', 'task'].edge_attr = edge_attr_1
                data['agent', 'astar', 'task'].y = torch.Tensor(y)

                data['task', 'astar', 'agent'].edge_index = edge_index
                data['task', 'astar', 'agent'].edge_attr = edge_attr_1
                data['task', 'astar', 'agent'].y = torch.Tensor(y)

                data['agent', 'man', 'task'].edge_index = edge_index
                data['agent', 'man', 'task'].edge_attr = edge_attr_2
                data['agent', 'man', 'task'].y = torch.Tensor(y)

                data['task', 'man', 'agent'].edge_index = edge_index
                data['task', 'man', 'agent'].edge_attr = edge_attr_2
                data['task', 'man', 'agent'].y = torch.Tensor(y)

                data['agent', 'proxy', 'task'].edge_index = edge_index
                data['agent', 'proxy', 'task'].edge_attr = edge_attr_3
                data['agent', 'proxy', 'task'].y = torch.Tensor(y)

                data['task', 'proxy', 'agent'].edge_index = edge_index
                data['task', 'proxy', 'agent'].edge_attr = edge_attr_3
                data['task', 'proxy', 'agent'].y = torch.Tensor(y)

                data.pin_memory()
                data.to('cuda', non_blocking=True)

                data_list.append(data)

            torch.save(data_list, 'datas/pyg/8_8_20_5_5/{}/hetero.pt'.format(data_type))

    else:
        raise ValueError('supports only homogeneous and heterogeneous graphs')


def run(exp_type: str):
    seed_everything(seed=42)
    date = datetime.now().strftime("%m%d_%H%M%S")
    exp_config = OmegaConf.load('config/experiment/pyg_{}.yaml'.format(exp_type))

    train_data = torch.load('datas/pyg/8_8_20_5_5/train/{}.pt'.format(exp_config.edge_type))
    val_data = torch.load('datas/pyg/8_8_20_5_5/val/{}.pt'.format(exp_config.edge_type))
    # test_data = torch.load('datas/pyg/8_8_20_5_5/test/{}.pt'.format(exp_config.edge_type))

    train_loader = PrefetchLoader(loader=DataLoader(train_data, batch_size=exp_config.batch_size, shuffle=True),
                                  device=exp_config.device)
    val_loader = PrefetchLoader(loader=DataLoader(val_data, batch_size=exp_config.batch_size, shuffle=True),
                                device=exp_config.device)
    # test_loader = PrefetchLoader(loader=DataLoader(test_data, batch_size=exp_config.batch_size, shuffle=True),
    #                              device=exp_config.device)

    gnn_config = OmegaConf.load('config/model/mpnn.yaml')
    attn_config = OmegaConf.load('config/model/attention.yaml')

    if exp_config.wandb:
        import wandb
        wandb_config = dict(exp_setup=exp_config, params=gnn_config)
        wandb.init(project='NeuralLNS', name=date, config=wandb_config)

    gnn = MPNN(gnn_config).to(exp_config.device)
    attn = MultiHeadCrossAttention(attn_config).to(exp_config.device)

    for e in trange(exp_config.epochs):
        epoch_loss, num_batch = 0, 0

        for batch in train_loader:
            batch_loss = gnn(batch)
            epoch_loss += batch_loss
            num_batch += 1
        epoch_loss /= num_batch

        if exp_config.wandb:
            wandb.log({'epoch_loss': epoch_loss})

        if (e + 1) % 10 == 0:
            torch.save(gnn.state_dict(), 'pyg_{}_{}.pt'.format(exp_config.edge_type, e + 1))

            val_gnn = MPNN(gnn_config)
            val_gnn.load_state_dict(torch.load('pyg_{}_{}.pt'.format(exp_config.edge_type, e + 1)))
            val_gnn.eval()

            val_loss, num_batch = 0, 0
            for batch in val_loader:
                val_batch_loss = val_gnn(batch)
                val_loss += val_batch_loss
                num_batch += 1
            val_loss /= num_batch

            if exp_config.wandb:
                wandb.log({'val_loss': val_loss})


if __name__ == '__main__':
    # pyg_data(graph_type='homo')
    # pyg_data(graph_type='hetero')
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_type')
    args = parser.parse_args()
    run(args.exp_type)
