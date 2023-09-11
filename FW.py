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
from src.model.pathgnn import PathGNN
from src.model.pyg_mpnn import PathMPNN
from utils.seed import seed_everything


def generate_pathgnn_data():
    seed_everything(seed=42)
    for data_type in ['train', 'val', 'test']:
        scenarios = torch.load('datas/scenarios/pathgnn_hard/{}.pt'.format(data_type))

        save_dir = 'datas/pyg/pathgnn_hard/'
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


def train_pathgnn(device: str, logging: bool, readout_dir: int, readout_type: str):
    seed_everything(seed=42)
    config = OmegaConf.load('config/experiment/FW.yaml')

    if logging:
        model_dir = 'datas/models/pathgnn_hard_{}_{}layers/'.format(readout_type, config.model.num_layers)
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)

    train_data = torch.load('datas/pyg/pathgnn_hard/train.pt', map_location=device)
    val_data = torch.load('datas/pyg/pathgnn_hard/val.pt', map_location=device)

    if logging:
        import wandb
        wandb.init(project='FW', name='pathgnn_hard_{}_{}layers'.format(readout_type, config.model.num_layers))

    gnn = PathGNN(config, readout_dir, readout_type).to(device)
    gnn.load_state_dict(torch.load(model_dir + 'pathgnn_60.pt'))

    for e in trange(100):
        train_loader = DataLoader(train_data, batch_size=config.batch_size, shuffle=True)
        train_loss = 0

        for tr in train_loader:
            batch_loss = gnn(tr)
            train_loss += (batch_loss / len(train_loader))

        if logging:
            wandb.log({'train_loss': train_loss})

        if (e + 1) % 10 == 0:
            torch.save(gnn.state_dict(), model_dir + 'pathgnn_{}.pt'.format(e + 61))

            val_gnn = PathGNN(config, readout_dir, readout_type).to(device)
            val_gnn.load_state_dict(torch.load(model_dir + 'pathgnn_{}.pt'.format(e + 61)))
            val_gnn.eval()

            val_loader = DataLoader(val_data, batch_size=config.batch_size, shuffle=True)
            val_loss = 0
            for val in val_loader:
                batch_loss = val_gnn(val)
                val_loss += (batch_loss / len(val_loader))

            if logging:
                wandb.log({'val_loss': val_loss})


def train_mpnn(device: str, logging: bool, mpnn_type: str, readout_type: str):
    seed_everything(seed=42)
    config = OmegaConf.load('config/experiment/FW.yaml')

    if logging:
        model_dir = 'datas/models/mpnn_{}_{}/'.format(mpnn_type, readout_type)
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)

    train_data = torch.load('datas/pyg/pathgnn/train.pt', map_location=device)
    val_data = torch.load('datas/pyg/pathgnn/val.pt', map_location=device)

    if logging:
        import wandb
        wandb.init(project='FW', name='mpnn_{}_{}'.format(mpnn_type, readout_type))

    gnn = PathMPNN(config, mpnn_type, readout_type).to(device)

    for e in trange(100):
        train_loader = DataLoader(train_data, batch_size=config.batch_size, shuffle=True)
        train_loss = 0

        for tr in train_loader:
            batch_loss = gnn(tr)
            train_loss += (batch_loss / len(train_loader))

        if logging:
            wandb.log({'train_loss': train_loss})

        if (e + 1) % 10 == 0:
            torch.save(gnn.state_dict(), model_dir + 'mpnn_{}.pt'.format(e + 1))

            val_gnn = PathMPNN(config, mpnn_type, readout_type).to(device)
            val_gnn.load_state_dict(torch.load(model_dir + 'mpnn_{}.pt'.format(e + 1)))
            val_gnn.eval()

            val_loader = DataLoader(val_data, batch_size=config.batch_size, shuffle=True)
            val_loss = 0
            for val in val_loader:
                batch_loss = val_gnn(val)
                val_loss += (batch_loss / len(val_loader))

            if logging:
                wandb.log({'val_loss': val_loss})


def test_pathgnn(readout_dir: int, readout_type: list):
    seed_everything(seed=43)
    config = OmegaConf.load('config/experiment/FW.yaml')
    test_data = torch.load('datas/pyg/pathgnn_hard/test.pt', map_location='cuda:0')
    test_loader = DataLoader(test_data, shuffle=True)

    for rt in readout_type:
        model_dir = 'datas/models/pathgnn_hard_{}_{}layers/'.format(rt, config.model.num_layers)
        gnn = PathGNN(config, readout_dir, rt).to('cuda:0')
        gnn.load_state_dict(torch.load(model_dir + 'pathgnn_160.pt'))
        gnn.eval()

        preds = []
        labels = []
        for test in tqdm(test_loader):
            preds.append(gnn(test, test=True))
            labels.append(test.y.item())

        plt.clf()
        plt.plot(preds, labels, 'b.')
        criterion = range(math.floor(min(preds + labels)), math.ceil(max(preds + labels)))
        plt.plot(criterion, criterion, 'r--')
        plt.xlabel('preds')
        plt.ylabel('labels')
        plt.title('pathgnn_{}_{}layers'.format(rt, config.model.num_layers))
        plt.show()


def test_mpnn(mpnn_type: str, readout_type):
    seed_everything(seed=42)
    config = OmegaConf.load('config/experiment/FW.yaml')

    if isinstance(readout_type, list):
        for rt in readout_type:

            model_dir = 'datas/models/mpnn_{}_{}/'.format(mpnn_type, rt)

            gnn = PathMPNN(config, mpnn_type, rt).to('cuda:0')
            gnn.load_state_dict(torch.load(model_dir + 'mpnn_100.pt'))
            gnn.eval()

            test_data = torch.load('datas/pyg/pathgnn/test.pt', map_location='cuda:0')
            test_loader = DataLoader(test_data, shuffle=True)

            preds = []
            labels = []
            for test in tqdm(test_loader):
                preds.append(gnn(test, test=True))
                labels.append(test.y.item())

            plt.clf()
            plt.plot(preds, labels, 'b.')
            criterion = range(math.floor(min(preds + labels)), math.ceil(max(preds + labels)))
            plt.plot(criterion, criterion, 'r--')
            plt.xlabel('preds')
            plt.ylabel('labels')
            plt.title('mpnn_{}_{}'.format(mpnn_type, rt))
            plt.show()

    elif isinstance(readout_type, str):
        model_dir = 'datas/models/mpnn_{}_{}/'.format(mpnn_type, readout_type)

        gnn = PathMPNN(config, mpnn_type, readout_type).to('cuda:0')
        gnn.load_state_dict(torch.load(model_dir + 'mpnn_100.pt'))
        gnn.eval()

        test_data = torch.load('datas/pyg/pathgnn/test.pt', map_location='cuda:0')
        test_loader = DataLoader(test_data, shuffle=True)

        preds = []
        labels = []
        for test in tqdm(test_loader):
            preds.append(gnn(test, test=True))
            labels.append(test.y.item())

        plt.clf()
        plt.plot(preds, labels, 'b.')
        criterion = range(math.floor(min(preds + labels)), math.ceil(max(preds + labels)))
        plt.plot(criterion, criterion, 'r--')
        plt.xlabel('preds')
        plt.ylabel('labels')
        plt.title('mpnn_{}_{}'.format(mpnn_type, readout_type))
        plt.show()


if __name__ == '__main__':
    # generate_pathgnn_data()
    # train_pathgnn(device='cuda:3', logging=True, readout_dir=1, readout_type='mean')
    # train_mpnn(device='cuda:3', logging=True, mpnn_type='pathwise', readout_type='mlp')
    test_pathgnn(readout_dir=1, readout_type=['mean'])
    # test_mpnn(mpnn_type='pathwise', readout_type=['sum', 'mean', 'max', 'min', 'mlp'])
