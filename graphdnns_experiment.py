import numpy as np
import torch
import torch_geometric
import torch.nn as nn
import torch.nn.functional as F
# from torch.utils.data import TensorDataset, DataLoader
from torch_geometric.data.dataloader import DataLoader
# from torch_geometric.data import DataLoader

from rdkit import Chem
from rdkit.Chem.AllChem import GetMorganFingerprintAsBitVect
from rdkit.DataStructs.cDataStructs import ConvertToNumpyArray

from utils.data import load_alvadesc_data, load_descriptors
from models.regressors.GraphDnns import NeuralFP, MLP_Regressor
from models.preprocessors.column_selectors import make_col_selector

from sklearn.model_selection import train_test_split
import pandas as pd
import os

import re

import matplotlib.pyplot as plt


def get_atom_features(mol):
    atomic_number = []
    num_hs = []

    for atom in mol.GetAtoms():
        atomic_number.append(atom.GetAtomicNum())
        num_hs.append(atom.GetTotalNumHs(includeNeighbors=True))

    return torch.tensor([atomic_number, num_hs]).t()


def get_edge_index(mol):
    row, col = [], []

    for bond in mol.GetBonds():
        start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        row += [start, end]
        col += [end, start]

    return torch.tensor([row, col], dtype=torch.long)


def prepare_dataloader(mol_list, batch_size):
    data_list = []

    for i, mol in enumerate(mol_list):
        x = get_atom_features(mol)
        edge_index = get_edge_index(mol)

        data = torch_geometric.data.data.Data(x=x, edge_index=edge_index)
        data_list.append(data)

    return DataLoader(data_list, batch_size=batch_size, shuffle=False), data_list


def train_step(batch, labels, reg, opt):
    opt.zero_grad()
    out = reg(batch)
    loss = F.mse_loss(torch.flatten(out), labels.to(torch.float), reduction='mean')
    loss.backward()
    opt.step()
    return loss


def test_step(batch, labels, reg):
    out = reg(batch)
    loss = F.mse_loss(torch.flatten(out), labels.to(torch.float), reduction='mean')
    return loss


def train_fn(train_loader, train_labels_loader, reg, opt):
    reg.train()
    total_loss = 0
    for idx, (batch, labels) in enumerate(zip(train_loader, train_labels_loader)):
        loss = train_step(batch, labels, reg, opt)
        total_loss += loss.item()

    # opt.zero_grad()
    # opt.step()
    # torch.nn.utils.clip_grad_norm_(reg.parameters(), 1)
    return total_loss / len(train_loader)


def test_fn(test_loader, test_labels_loader, reg):
    reg.eval()
    total_loss = 0
    with torch.no_grad():
        for idx, (batch, labels) in enumerate(zip(test_loader, test_labels_loader)):
            loss = test_step(batch, labels, reg)
            total_loss += loss.item()

    total_loss /= len(test_loader)

    return total_loss


if __name__ == '__main__':

    ############################################
    ## GRAPH CONVOLUTIONS
    ############################################

    batch_size = 64

    with open('data/alvadesc/fingerprints/smiles.txt') as f:
        smiles = f.read().splitlines()

    mol_list = [Chem.MolFromSmiles(smi) for smi in smiles]
    # dloader, dlist = prepare_dataloader(mol_list, batch_size)

    # for batch in dloader:
    #     break

    neural_fp = NeuralFP(atom_features=2, fp_size=2214)


    # fps = neural_fp(batch)

    # # ############################################
    # # ## LEARNING FGPS THROUGH BACKPROPAGATION
    # # ############################################

    # Retrieve smiles
    def sorted_alphanumeric(data):
        convert = lambda text: int(text) if text.isdigit() else text.lower()
        alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
        return sorted(data, key=alphanum_key)


    filenames = sorted_alphanumeric(os.listdir('../../extra/sdfs/'))
    pids = [f.strip('.sdf') for f in filenames]
    df_mols = pd.DataFrame({'pid': pids, 'mol': mol_list})
    df_mols['pid'] = df_mols['pid'].astype('int')

    # Load fgps
    fgp = load_alvadesc_data(split_as_np=False)

    # Get the target
    df_rts = pd.DataFrame({'pid': fgp['pid'], 'rt': fgp['rt']})
    df_rts['pid'] = df_rts['pid'].astype('int')
    df_rts['rt'] = df_rts['rt'].astype('float32')

    # Merge both
    df_mols_rts = pd.merge(df_mols, df_rts, on='pid')

    # Get final input and target
    X = df_mols_rts['mol'].values
    y = df_mols_rts['rt'].values.astype('float32').flatten()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    train_loader, _ = prepare_dataloader(X_train, batch_size)
    test_loader, _ = prepare_dataloader(X_test, batch_size)

    train_labels_loader = torch.utils.data.DataLoader(y_train, batch_size)
    # valid_labels_loader = torch.utils.data.DataLoader(valid.y, batch_size=batch_size)
    test_labels_loader = torch.utils.data.DataLoader(y_test, batch_size)

    reg = MLP_Regressor(neural_fp, atom_features=2, fp_size=2214, hidden_size=100)
    optimizer = torch.optim.SGD(reg.parameters(), lr=0.001, weight_decay=0.001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=100)

    total_epochs = 100  # 1000
    train_loss_values = test_loss_values = []
    for epoch in range(1, total_epochs + 1):
        train_loss = train_fn(train_loader, train_labels_loader, reg, opt=optimizer)
        train_loss_values.append(train_loss)
        test_loss = test_fn(test_loader, test_labels_loader, reg)
        test_loss_values.append(test_loss)
        scheduler.step(test_loss)  # valid_loss)

        if epoch % 10 == 0:
            print(f'Epoch:{epoch}, Train loss: {train_loss}, Test loss: {test_loss}')  # , Valid loss: {valid_loss}')

    plt.plot(train_loss_values)
    plt.plot(test_loss_values)
    plt.show()
