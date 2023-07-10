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


def run(logging: bool):
    seed_everything(seed=42)
    config = OmegaConf.load('config/experiment/pathgnn.yaml')

    date = datetime.now().strftime("%m%d_%H%M%S")
    model_dir = 'datas/models/{}_pathgnn/'.format(date)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    train_data = torch.load('datas/pyg/pathgnn/train.pt', map_location=config.device)
    train_loader = DataLoader(train_data, batch_size=config.batch_size, shuffle=True)

    val_data = torch.load('datas/pyg/pathgnn/val.pt', map_location=config.device)
    val_loader = DataLoader(val_data, batch_size=config.batch_size, shuffle=True)

    if logging:
        import wandb
        wandb.init(project='FW', name=date, config=dict(setup=config))

    mpnn = PathMPNN(config)
    pathgnn = PathGNN(config)

    for e in trange(100):
        mpnn_train_loss, pathgnn_train_loss, num_batch = 0, 0, 0

        for tr in train_loader:
            mpnn_batch_loss = mpnn(tr)
            mpnn_train_loss += mpnn_batch_loss

            pathgnn_batch_loss = pathgnn(tr)
            pathgnn_train_loss += pathgnn_batch_loss

            num_batch += 1

        mpnn_train_loss /= num_batch
        pathgnn_train_loss /= num_batch

        if logging:
            wandb.log({'mpnn_train_loss': mpnn_train_loss,
                       'pathgnn_train_loss': pathgnn_train_loss})

        if (e + 1) % 10 == 0:
            torch.save(mpnn.state_dict(), model_dir + 'mpnn_{}.pt'.format(e + 1))
            torch.save(pathgnn.state_dict(), model_dir + 'pathgnn_{}.pt'.format(e + 1))

            val_mpnn, val_pathgnn = PathMPNN(config), PathGNN(config)
            val_mpnn.load_state_dict(torch.load(model_dir + 'mpnn_{}.pt'.format(e + 1)))
            val_pathgnn.load_state_dict(torch.load(model_dir + 'pathgnn_{}.pt'.format(e + 1)))
            val_mpnn.eval(), val_pathgnn.eval()

            mpnn_val_loss, pathgnn_val_loss, num_batch = 0, 0, 0
            for val in val_loader:
                mpnn_val_batch_loss = val_mpnn(val)
                mpnn_val_loss += mpnn_val_batch_loss

                pathgnn_val_batch_loss = val_pathgnn(val)
                pathgnn_val_loss += pathgnn_val_batch_loss

                num_batch += 1
            mpnn_val_loss /= num_batch
            pathgnn_val_loss /= num_batch

            if logging:
                wandb.log({'mpnn_val_loss': mpnn_val_loss,
                           'pathgnn_val_loss': pathgnn_val_loss})


if __name__ == '__main__':
    # generate_pathgnn_data()
    run(logging=True)
