import os
import sys
from datetime import datetime

import networkx as nx
import numpy as np
import torch
import torch_geometric as pyg
from omegaconf import OmegaConf
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from tqdm import tqdm, trange

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from src.model.attention import MultiHeadCrossAttention
from src.model.pyg_mpnn import MPNN
from utils.seed import seed_everything


def generate_pathgnn_data():
    seed_everything(seed=42)
    for data_type in ['train', 'val', 'test']:

        data_list_A, data_list_M, data_list_P = [], [], []
        scenarios = torch.load('datas/scenarios/8_8_20_5_5/{}.pt'.format(data_type))

        save_dir = 'datas/pyg/8_8_20_5_5/pathgnn/'
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        for scen in tqdm(scenarios):
            grid, graph, a_coord, t_coord, y = scen

            pyg_graph = pyg.utils.from_networkx(graph, group_node_attrs='', group_edge_attrs='')

            coords = torch.Tensor(list(graph.nodes())) / grid.shape[0]


            x = torch.cat((coords, types), -1)

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

        torch.save(data_list_A, save_dir + 'A.pt')
        torch.save(data_list_M, save_dir + 'M.pt')
        torch.save(data_list_P, save_dir + 'P.pt')


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
    # import multiprocessing
    #
    # torch.multiprocessing.set_start_method('spawn')
    # process = []
    # edge_type = ['A', 'M', 'P']
    #
    # for e_id in edge_type:
    #     p = multiprocessing.Process(target=run, args=(e_id, True,))
    #     p.start()
    #     process.append(p)
    #
    # for p in process:
    #     p.join()

    generate_pathgnn_data()
