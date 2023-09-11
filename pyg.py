import copy
import math
import os
import random
import shutil
import sys

import numpy as np
import torch
from matplotlib import pyplot as plt
from omegaconf import OmegaConf
from torch_geometric.data import Data
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


def pyg_data(norm: bool, scen_config: str):
    seed_everything(seed=42)
    for data_type in ['train', 'val', 'test']:
        scenarios = torch.load('datas/scenarios/test/{}/{}.pt'.format(scen_config, data_type))

        if norm:
            data_list = []

            norm_dir = 'datas/pyg/norm_test/{}/'.format(scen_config)
            if not os.path.exists(norm_dir):
                os.makedirs(norm_dir)

            nf_list, e_id_list, ef_list, y_list = [[] for _ in range(4)]
            for scen in scenarios:
                nf_list.append(torch.Tensor(scen[0]))
                e_id_list.append(torch.LongTensor(scen[1]))
                ef_list.append(torch.Tensor(scen[2]))
                y_list.append(scen[3])

            nf = torch.stack(nf_list)
            e_id = torch.stack(e_id_list)
            ef = torch.stack(ef_list)
            ys = torch.Tensor(y_list)

            norm_nf = (nf - nf.mean()) / (nf.std() + 1e-5)
            norm_ef_0 = (ef[:, :, 0] - ef[:, :, 0].mean()) / (ef[:, :, 0].std() + 1e-5)
            norm_ef_1 = (ef[:, :, 1] - ef[:, :, 1].mean()) / (ef[:, :, 1].std() + 1e-5)
            norm_ef_2 = (ef[:, :, 2] - ef[:, :, 2].mean()) / (ef[:, :, 2].std() + 1e-5)
            norm_ef = torch.stack([norm_ef_0, norm_ef_1, norm_ef_2], dim=-1)
            norm_y = (ys - ys.mean()) / (ys.std() + 1e-5)

            for x, edge_index, edge_attr, y in zip(tqdm(norm_nf), e_id, norm_ef, norm_y):
                data_list.append(Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y))

            torch.save(data_list, norm_dir + '{}.pt'.format(data_type))

        else:
            data_list = []

            save_dir = 'datas/pyg/test/{}/'.format(scen_config)
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)

            for scen in tqdm(scenarios):
                nf, edge_index, edge_attr, y = scen
                x = torch.FloatTensor(nf)
                edge_attr = torch.Tensor(edge_attr)
                edge_index = torch.LongTensor(edge_index)
                y = torch.Tensor(y)
                data_list.append(Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y))

            torch.save(data_list, save_dir + '{}.pt'.format(data_type))


def run(device: str, logging: bool, num_layer: int):
    seed_everything(seed=42)
    config = OmegaConf.load('config/experiment/edge_test.yaml')

    if logging:
        # date = datetime.now().strftime("%m%d_%H%M%S")
        model_dir = 'datas/models/norm_NL_{}_HS_F/'.format(num_layer)
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)

    train_data = torch.load('datas/pyg/norm_test/{}/train.pt'.format(config.map), map_location=device)
    train_loader = DataLoader(train_data, batch_size=config.batch_size, shuffle=True)
    val_data = torch.load('datas/pyg/norm_test/{}/val.pt'.format(config.map), map_location=device)
    val_loader = DataLoader(val_data, batch_size=config.batch_size, shuffle=True)

    if logging:
        import wandb
        wandb.init(project='NeuralLNS', name='norm_NL_{}_HS_F'.format(num_layer))

    GNN = MPNN(config, num_layer).to(device)

    for e in trange(100):
        train_loss = 0
        for train in train_loader:
            batch_loss = GNN(train, config)
            train_loss += batch_loss / len(train_loader)

        if logging:
            wandb.log({'train_loss': train_loss})

        if (e + 1) % 10 == 0:
            torch.save(GNN.state_dict(), model_dir + '{}.pt'.format(e + 1))
            val_GNN = MPNN(config, num_layer).to(device)
            val_GNN.load_state_dict(torch.load(model_dir + '{}.pt'.format(e + 1)))
            val_GNN.eval()

            val_loss = 0
            for val in val_loader:
                batch_loss = val_GNN(val, config)
                val_loss += batch_loss / len(val_loader)

            if logging:
                wandb.log({'val_loss': val_loss})


def test(device: str, model_dir: str, num_layer: int, title: str):
    seed_everything(seed=44)
    config = OmegaConf.load('config/experiment/edge_test.yaml')
    test_data = torch.load('datas/pyg/norm_test/{}/test.pt'.format(config.map), map_location=device)
    test_loader = DataLoader(test_data, shuffle=True)

    GNN = MPNN(config, num_layer).to(device)
    GNN.load_state_dict(torch.load('datas/models/' + model_dir))
    GNN.eval()

    preds, labels = [], []
    for test in tqdm(test_loader):
        preds.append(GNN(test, config, test=True).item())
        labels.append(test.y.item())

    plt.clf()
    plt.plot(preds, labels, 'b.')
    criterion = range(math.floor(min(preds + labels)), math.ceil(max(preds + labels)))
    plt.plot(criterion, criterion, 'r--')
    plt.xlabel('preds')
    plt.ylabel('labels')
    plt.title(title)
    plt.show()


def layer_test(device: str, model_dir: str):
    import torch.nn as nn
    seed_everything(seed=43)
    lin = nn.Linear(32, 2, bias=False)
    lin.weight = nn.Parameter(torch.randn(2, 32), requires_grad=False)
    lin.to(device)

    config = OmegaConf.load('config/experiment/edge_test.yaml')
    test_data = torch.load('datas/pyg/test/{}/test.pt'.format(config.map), map_location=device)

    GNN = MPNN(config).to(device)
    GNN.load_state_dict(torch.load('datas/models/' + model_dir))
    GNN.eval()

    layer_h = GNN(test_data[0], config, test=True)
    for id, h in enumerate([lin(h) for h in layer_h]):
        hx, hy = h[:, 0].cpu().detach().numpy(), h[:, 1].cpu().detach().numpy()
        plt.clf()
        plt.plot(hx, hy, 'b.')
        plt.title("layer_{}".format(id))
        plt.show()


if __name__ == '__main__':
    # pyg_data(norm=True, scen_config='8_8_20_5_5')
    # run(device='cuda:0', logging=True, num_layer=1)
    NL = 1
    test(device='cuda:0', model_dir='norm_NL_{}_HS_F/100.pt'.format(NL), num_layer=NL,
         title='norm_NL_{}_HS_F'.format(NL))
    # layer_test(device='cuda:1', model_dir='NL_8_GIN/100.pt')
