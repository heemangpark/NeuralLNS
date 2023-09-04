import math
import os
import sys

import numpy as np
import torch
from matplotlib import pyplot as plt
from omegaconf import OmegaConf
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from tqdm import tqdm, trange

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from src.model.triplet import TripletGNN
from src.model.et import EdgeTransformer
from src.model.rt import RelationalTransformer
from utils.seed import seed_everything


def generate_pathgnn_data():
    seed_everything(seed=42)
    for data_type in ['test']:
        scenarios = torch.load('data/32_single/{}.pt'.format(data_type))

        save_dir = 'data/32_single/'
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        # standardize label data
        label = np.array([scen[-1] for scen in scenarios])
        label = (label - label.mean()) / (label.std() + 1e-5)

        data_list = []
        for scen, y in zip(tqdm(scenarios), label):
            coords, types, edge_index, schedule, _ = scen

            nf_coord = torch.Tensor(coords)
            nf_coord = (nf_coord - nf_coord.min()) / (nf_coord.max() + nf_coord.min())
            nf_type = torch.eye(4)[types]
            nf = torch.cat((nf_coord, nf_type), -1)

            edge_index = torch.LongTensor(edge_index).transpose(-1, 0)

            data = Data(x=nf, edge_index=edge_index, edge_attr=torch.ones(edge_index.shape[-1], 1),
                        schedule=torch.LongTensor(schedule), y=torch.Tensor([y]))

            data_list.append(data)

        torch.save(data_list, save_dir + '{}.pt'.format(data_type))


def gnn_train(device: str, logging: bool, type: str):
    seed_everything(seed=42)
    config = OmegaConf.load('config/experiment/{}.yaml'.format(type))

    if logging:
        model_dir = 'datas/models/{}_{}layers_sum/'.format(type, config.model.num_layers)
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)

    train_data = torch.load('datas/pyg/pathgnn_hard/train.pt', map_location=device)
    val_data = torch.load('datas/pyg/pathgnn_hard/val.pt', map_location=device)

    if logging:
        import wandb
        wandb.init(project='FW', name='{}_{}layers_sum'.format(type, config.model.num_layers))

    if type == 'triplet':
        gnn = TripletGNN(config).to(device)
    elif type == 'et':
        gnn = EdgeTransformer(config).to(device)
    elif type == 'rt':
        gnn = RelationalTransformer(config).to(device)
    else:
        raise NotImplementedError('Unknown GNN Type')

    for e in trange(100):
        train_loader = DataLoader(train_data, batch_size=config.batch_size, shuffle=True)
        train_loss = 0

        for tr in train_loader:
            batch_loss = gnn(tr)
            train_loss += (batch_loss / len(train_loader))

        if logging:
            wandb.log({'train_loss': train_loss})

        if (e + 1) % 10 == 0:
            torch.save(gnn.state_dict(), model_dir + '{}.pt'.format(e + 101))

            if type == 'triplet':
                val_gnn = TripletGNN(config).to(device)
            elif type == 'et':
                val_gnn = EdgeTransformer(config).to(device)
            elif type == 'rt':
                val_gnn = RelationalTransformer(config).to(device)
            else:
                raise NotImplementedError('Unknown GNN Type')

            val_gnn.load_state_dict(torch.load(model_dir + '{}.pt'.format(e + 101)))
            val_gnn.eval()

            val_loader = DataLoader(val_data, batch_size=config.batch_size, shuffle=True)
            val_loss = 0
            for val in val_loader:
                batch_loss = val_gnn(val)
                val_loss += (batch_loss / len(val_loader))

            if logging:
                wandb.log({'val_loss': val_loss})


def gnn_test(device: str):
    seed_everything(seed=43)
    config = OmegaConf.load('config/experiment/rt.yaml')
    test_data = torch.load('data/32_single/test.pt', map_location=device)
    test_loader = DataLoader(test_data, batch_size=10, shuffle=True)

    gnn = RelationalTransformer(config).to(device)
    gnn.load_state_dict(torch.load('data/models/rt_8_sum.pt'))
    gnn.eval()

    preds = []
    labels = []
    for test in tqdm(test_loader):
        preds.extend(gnn(test, test=True).tolist())
        labels.extend(test.y.cpu().detach())

    plt.clf()
    plt.plot(preds, labels, 'b.')
    criterion = range(math.floor(min(preds + labels)), math.ceil(max(preds + labels)))
    plt.plot(criterion, criterion, 'r--')
    plt.xlabel('preds')
    plt.ylabel('labels')
    plt.title('rt_{}layers'.format(config.model.num_layers))
    plt.show()


if __name__ == '__main__':
    # generate_pathgnn_data()
    # gnn_train(device='cuda:0', logging=True, type='rt')
    gnn_test(device='cuda:3')
