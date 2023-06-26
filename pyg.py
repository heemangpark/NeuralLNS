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


def pyg_data(graph_type: str, scen_config: str):
    seed_everything(seed=42)
    if graph_type == 'homo':
        for data_type in ['train', 'val', 'test']:

            data_list_A, data_list_M, data_list_P = [], [], []
            scenarios = torch.load('datas/scenarios/{}/{}.pt'.format(scen_config, data_type))
            #  scen_config  8_8_20_5_5  16_16_20_10_10  32_32_20_10_10  8_8_partition_5_5

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

                data_list_A.append(Data(x=x,
                                        edge_index=edge_index,
                                        edge_attr=torch.FloatTensor(A).view(-1, 1),
                                        y=torch.Tensor(y)))
                data_list_M.append(Data(x=x,
                                        edge_index=edge_index,
                                        edge_attr=torch.FloatTensor(M).view(-1, 1),
                                        y=torch.Tensor(y)))
                data_list_P.append(Data(x=x,
                                        edge_index=edge_index,
                                        edge_attr=torch.FloatTensor(P).view(-1, 1),
                                        y=torch.Tensor(y)))

            torch.save(data_list_A, 'datas/pyg/{}/{}/A.pt'.format(scen_config, data_type))
            torch.save(data_list_M, 'datas/pyg/{}/{}/M.pt'.format(scen_config, data_type))
            torch.save(data_list_P, 'datas/pyg/{}/{}/P.pt'.format(scen_config, data_type))

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


def run(exp_type: str, logging: bool):
    seed_everything(seed=42)

    exp_config = OmegaConf.load('config/experiment/pyg_{}.yaml'.format(exp_type))
    gnn_config = OmegaConf.load('config/model/mpnn.yaml')
    attn_config = OmegaConf.load('config/model/attention.yaml')

    date = datetime.now().strftime("%m%d_%H%M%S")
    model_dir = 'datas/models/{}_{}/'.format(date, exp_type)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    with open(model_dir + 'config.txt', 'w') as file:
        file.write('EXP SETUP: ' + str(exp_config) + '\n' +
                   'GNN SETUP: ' + str(gnn_config))

    train_data = torch.load('datas/pyg/{}/train/{}.pt'.format(exp_config.map, exp_config.edge_type),
                            map_location=exp_config.device)
    val_data = torch.load('datas/pyg/{}/val/{}.pt'.format(exp_config.map, exp_config.edge_type),
                          map_location=exp_config.device)
    # test_data = torch.load('datas/pyg/{}/test/{}.pt'.format(exp_config.map, exp_config.edge_type),
    #                        map_location=exp_config.device)

    train_loader = DataLoader(train_data, batch_size=exp_config.batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=exp_config.batch_size, shuffle=True)
    # test_loader = DataLoader(test_data, batch_size=exp_config.batch_size, shuffle=True)

    if logging:
        import wandb
        wandb_config = dict(exp_setup=exp_config, params=gnn_config)
        wandb.init(project='NeuralLNS', name=exp_type, config=wandb_config)

    gnn = MPNN(gnn_config).to(exp_config.device)
    attn = MultiHeadCrossAttention(attn_config).to(exp_config.device)

    for e in trange(exp_config.epochs):
        epoch_loss, num_batch = 0, 0

        for tr in train_loader:
            batch_loss = gnn(tr.to(exp_config.device))
            epoch_loss += batch_loss
            num_batch += 1
        epoch_loss /= num_batch

        if logging:
            wandb.log({'epoch_loss': epoch_loss})

        if (e + 1) % 10 == 0:
            torch.save(gnn.state_dict(), model_dir + '{}_{}.pt'.format(exp_config.edge_type, e + 1))

            val_gnn = MPNN(gnn_config).to(exp_config.device)
            val_gnn.load_state_dict(torch.load(model_dir + '{}_{}.pt'.format(exp_config.edge_type, e + 1)))
            val_gnn.eval()

            val_loss, num_batch = 0, 0
            for val in val_loader:
                val_batch_loss = val_gnn(val.to(exp_config.device))
                val_loss += val_batch_loss
                num_batch += 1
            val_loss /= num_batch

            if logging:
                wandb.log({'val_loss': val_loss})


if __name__ == '__main__':
    import multiprocessing

    torch.multiprocessing.set_start_method('spawn')
    process = []
    edge_type = ['A', 'M', 'P']

    for e_id in edge_type:
        p = multiprocessing.Process(target=run, args=(e_id, True,))
        p.start()
        process.append(p)

    for p in process:
        p.join()

    # pyg_data(graph_type='homo', scen_config='8_8_20_5_5')
    # pyg_data(graph_type='homo', scen_config='16_16_20_10_10')
    # pyg_data(graph_type='homo', scen_config='32_32_20_10_10')
    # pyg_data(graph_type='homo', scen_config='8_8_partition_5_5')

    # gnn_config = OmegaConf.load('config/model/mpnn.yaml')
    # exp_config = OmegaConf.load('config/experiment/pyg_P.yaml')
    #
    # val_data = torch.load('datas/pyg/{}/val/{}.pt'.format(exp_config.map, exp_config.edge_type),
    #                       map_location=exp_config.device)
    # val_loader = DataLoader(val_data, batch_size=exp_config.batch_size, shuffle=True)
    #
    # val_gnn = MPNN(gnn_config).to(exp_config.device)
    # val_gnn.load_state_dict(torch.load('datas/models/0626_113229/P_100.pt'))
    # val_gnn.eval()
    #
    # val_loss, num_batch = 0, 0
    # for val in val_loader:
    #     val_batch_loss = val_gnn(val.to(exp_config.device))
    #     val_loss += val_batch_loss
    #     num_batch += 1
    # val_loss /= num_batch
