import os
import sys
from datetime import datetime

import networkx as nx
import numpy as np
import torch
from omegaconf import OmegaConf
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from tqdm import tqdm, trange

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from src.model.pathgnn import PathGNN
from src.model.pyg_mpnn import PathMPNN
from utils.seed import seed_everything


def generate_pathgnn_data():
    seed_everything(seed=42)
    for data_type in ['train', 'val', 'test']:
        scenarios = torch.load('datas/scenarios/pathgnn/{}.pt'.format(data_type))

        save_dir = 'datas/pyg/pathgnn/'
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        data_list = []
        for scen in tqdm(scenarios):
            grid, graph, a_coord, t_coord, schedule, y = scen

            nf_coord = torch.Tensor(list(graph.nodes())) / grid.shape[0]
            nf_type = torch.eye(4)[list(nx.get_node_attributes(graph, 'type').values())]  # TODO: positional encoding

            nf = torch.cat((nf_coord, nf_type), -1)
            ef = torch.ones(graph.number_of_edges(), 1)  # TODO: edge feature

            row = np.array([(g, g + 1) for g in range(grid.shape[0] - 1)])
            col = row * grid.shape[0]
            edge_index = []
            for g in range(grid.shape[0]):
                edge_index += (row + g * grid.shape[0]).tolist() + (col + g).tolist()
            edge_index = torch.LongTensor(edge_index).transpose(-1, 0)

            data = Data(x=nf, edge_index=edge_index, edge_attr=ef,
                        schedule=torch.LongTensor(schedule), y=torch.Tensor(y).view(-1, 1))
            data_list.append(data)

        torch.save(data_list, save_dir + '{}.pt'.format(data_type))


def run(device: str, gnn_type: str, logging: bool):
    seed_everything(seed=42)
    config = OmegaConf.load('config/experiment/FW.yaml')

    date = datetime.now().strftime("%m%d_%H%M%S")
    model_dir = 'datas/models/{}_pathgnn/'.format(date)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    train_data = torch.load('datas/pyg/pathgnn/train.pt', map_location=device)
    train_loader = DataLoader(train_data, batch_size=config.batch_size, shuffle=True)

    val_data = torch.load('datas/pyg/pathgnn/val.pt', map_location=device)
    val_loader = DataLoader(val_data, batch_size=config.batch_size, shuffle=True)

    if logging:
        import wandb
        wandb.init(project='FW', name=gnn_type, config=dict(setup=config))

    if gnn_type == 'mpnn':
        gnn = PathMPNN(config).to(device)
    elif gnn_type == 'pathgnn':
        gnn = PathGNN(config).to(device)
    else:
        raise NotImplementedError('{} is an unsupported model'.format(gnn_type))

    for e in trange(100):
        train_loss, num_batch = 0, 0

        for tr in train_loader:
            loss_per_batch = gnn(tr)
            train_loss += loss_per_batch
            num_batch += 1
        train_loss /= num_batch

        if logging:
            wandb.log({'train_loss': train_loss})

        if (e + 1) % 10 == 0:
            torch.save(gnn.state_dict(), model_dir + '{}_{}.pt'.format(gnn_type, e + 1))
            if gnn_type == 'mpnn':
                val_gnn = PathMPNN(config).to(device)
            elif gnn_type == 'pathgnn':
                val_gnn = PathGNN(config).to(device)
            else:
                raise NotImplementedError('{} is an unsupported model'.format(gnn_type))
            val_gnn.load_state_dict(torch.load(model_dir + '{}_{}.pt'.format(gnn_type, e + 1)))
            val_gnn.eval()

            val_loss, num_batch = 0, 0
            for val in val_loader:
                loss_per_batch = val_gnn(val)
                val_gnn += loss_per_batch
                num_batch += 1
            val_loss /= num_batch

            if logging:
                wandb.log({'val_loss': val_loss})


if __name__ == '__main__':
    # generate_pathgnn_data()
    run(device='cuda:2', gnn_type='mpnn', logging=True)
    run(device='cuda:3', gnn_type='pathgnn', logging=True)

    "TESTING"
    seed_everything(seed=42)
    config = OmegaConf.load('config/experiment/FW.yaml')
    mpnn, pathgnn = PathMPNN(config), PathGNN(config)
    mpnn.load_state_dict(torch.load('datas/models/0710_154603_pathgnn/mpnn_100.pt'))
    pathgnn.load_state_dict(torch.load('datas/models/0710_154603_pathgnn/pathgnn_100.pt'))
    mpnn.eval(), pathgnn.eval()

    test_data = torch.load('datas/pyg/pathgnn/test.pt', map_location='cuda:1')
    test_loader = DataLoader(test_data, batch_size=config.batch_size, shuffle=True)

    for test in test_loader:
        mpnn_loss, pathgnn_loss = mpnn(test), pathgnn(test)
        print(mpnn_loss, pathgnn_loss)
