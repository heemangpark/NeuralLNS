import os
import sys
from datetime import datetime

import networkx as nx
import torch
import torch_geometric as pyg
from omegaconf import OmegaConf
from torch_geometric.loader import DataLoader
from tqdm import tqdm, trange

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from src.model.pyg_mpnn import MPNN
from utils.seed import seed_everything


def generate_pathgnn_data():
    seed_everything(seed=42)
    for data_type in ['train', 'val', 'test']:
        scenarios = torch.load('datas/scenarios/8_8_20_5_5/{}.pt'.format(data_type))

        save_dir = 'datas/pyg/8_8_20_5_5/pathgnn/'
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        data_list = []
        for scen in tqdm(scenarios):
            grid, graph, a_coord, t_coord, y = scen

            coords = torch.Tensor(list(graph.nodes())) / grid.shape[0]
            types = torch.eye(3)[list(nx.get_node_attributes(graph, 'type').values())]

            nf = torch.cat((coords, types), -1)
            ef = torch.ones(graph.number_of_edges() * 2, 1)

            pyg_graph = pyg.utils.from_networkx(graph)
            pyg_graph.x, pyg_graph.edge_attr, pyg_graph.y = nf, ef, y
            data_list.append(pyg_graph)

        torch.save(data_list, save_dir + '{}.pt'.format(data_type))


def run(logging: bool):
    seed_everything(seed=42)

    exp_config = OmegaConf.load('config/experiment/pathgnn.yaml')
    gnn_config = OmegaConf.load('config/model/mpnn.yaml')

    date = datetime.now().strftime("%m%d_%H%M%S")
    model_dir = 'datas/models/{}_pathgnn/'.format(date)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    with open(model_dir + 'config.txt', 'w') as file:
        file.write('EXP SETUP: ' + str(exp_config) + '\n' +
                   'GNN SETUP: ' + str(gnn_config))

    train_data = torch.load('datas/pyg/{}/pathgnn/train.pt'.format(exp_config.map), map_location=exp_config.device)
    train_loader = DataLoader(train_data, batch_size=exp_config.batch_size, shuffle=True)

    val_data = torch.load('datas/pyg/{}/pathgnn/val.pt'.format(exp_config.map), map_location=exp_config.device)
    val_loader = DataLoader(val_data, batch_size=exp_config.batch_size, shuffle=True)

    if logging:
        import wandb
        wandb_config = dict(exp_setup=exp_config, params=gnn_config)
        wandb.init(project='FW', name=date, config=wandb_config)

    gnn = MPNN(gnn_config).to(exp_config.device)

    for e in trange(100):
        epoch_loss, num_batch = 0, 0

        for tr in train_loader:
            batch_loss = gnn(tr)
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
    generate_pathgnn_data()
    run(logging=False)
