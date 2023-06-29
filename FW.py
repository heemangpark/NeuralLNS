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
from src.model.pathgnn import PathGNN
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

            one_hot = []  # EMPTY, AGENT, TASK one-hot
            for t in nx.get_node_attributes(graph, 'type').values():
                if t in range(5):
                    one_hot.append(1)
                elif t in range(5, 10):
                    one_hot.append(2)
                else:
                    one_hot.append(0)
            types = torch.eye(3)[one_hot]

            nf = torch.cat((coords, types), -1)
            ef = torch.ones(graph.number_of_edges() * 2, 1)

            pyg_graph = pyg.utils.from_networkx(graph)
            pyg_graph.x, pyg_graph.edge_attr, pyg_graph.y = nf, ef, y
            data_list.append(pyg_graph)

        torch.save(data_list, save_dir + '{}.pt'.format(data_type))


def run(logging: bool):
    seed_everything(seed=42)
    config = OmegaConf.load('config/experiment/pathgnn.yaml')

    date = datetime.now().strftime("%m%d_%H%M%S")
    model_dir = 'datas/models/{}_pathgnn/'.format(date)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    with open(model_dir + 'config.txt', 'w') as file:
        file.write('EXP SETUP: ' + str(config))

    train_data = torch.load('datas/pyg/{}/pathgnn/train.pt'.format(config.map), map_location=config.device)
    train_loader = DataLoader(train_data, batch_size=config.batch_size, shuffle=True)

    val_data = torch.load('datas/pyg/{}/pathgnn/val.pt'.format(config.map), map_location=config.device)
    val_loader = DataLoader(val_data, batch_size=config.batch_size, shuffle=True)

    if logging:
        import wandb
        wandb.init(project='FW', name=date, config=dict(setup=config))

    mpnn = MPNN(config)
    pathgnn = PathGNN(config)

    for e in trange(100):
        mpnn_epoch_loss, pathgnn_epoch_loss, num_batch = 0, 0, 0

        for tr in train_loader:
            mpnn_batch_loss, pathgnn_batch_loss = mpnn(tr), pathgnn(tr)
            mpnn_epoch_loss += mpnn_batch_loss
            pathgnn_epoch_loss += pathgnn_batch_loss
            num_batch += 1
        mpnn_epoch_loss /= num_batch
        pathgnn_epoch_loss /= num_batch

        if logging:
            wandb.log({'mpnn_train_loss': mpnn_epoch_loss,
                       'pathgnn_train_loss': pathgnn_epoch_loss})

        if (e + 1) % 10 == 0:
            torch.save(mpnn.state_dict(), model_dir + 'mpnn.pt')
            torch.save(pathgnn.state_dict(), model_dir + 'pathgnn.pt')

            val_mpnn, val_pathgnn = MPNN(config), PathGNN(config)
            val_mpnn.load_state_dict(torch.load(model_dir + 'mpnn.pt'))
            val_pathgnn.load_state_dict(torch.load(model_dir + 'pathgnn.pt'))
            val_mpnn.eval(), val_pathgnn.eval()

            val_mpnn_loss, val_pathgnn_loss, num_batch = 0, 0
            for val in val_loader:
                val_mpnn_batch_loss = val_mpnn(val)
                val_pathgnn_batch_loss = val_pathgnn(val)
                val_mpnn_loss += val_mpnn_batch_loss
                val_pathgnn_loss += val_pathgnn_batch_loss
                num_batch += 1
            val_mpnn_loss /= num_batch
            val_pathgnn_loss /= num_batch

            if logging:
                wandb.log({'mpnn_val_loss': val_mpnn_loss,
                           'pathgnn_val_loss': val_pathgnn_loss})


if __name__ == '__main__':
    run(logging=False)
