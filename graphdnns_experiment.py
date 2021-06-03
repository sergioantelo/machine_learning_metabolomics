import numpy as np
import tensorflow
import torch
#FIXME: library not working/properly installed (maybe use a different one)
import torch_geometric
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
#from torch_geometric.data.dataloader import DataLoader #changed

from rdkit import Chem
from rdkit.Chem.AllChem import GetMorganFingerprintAsBitVect
from rdkit.DataStructs.cDataStructs import ConvertToNumpyArray

from utils.data import load_alvadesc_data, load_descriptors
from models.regressors import NeuralFP, MLPRegressor
from models.preprocessors.column_selectors import make_col_selector


from sklearn.model_selection import train_test_split
import pandas as pd


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


def prepare_dataloader(mol_list):
    data_list = []

    for i, mol in enumerate(mol_list):

        x = get_atom_features(mol)
        edge_index = get_edge_index(mol)

        #FIXME: line throwing the error 
        data = torch_geometric.data.data.Data(x=x, edge_index=edge_index)
        #data = TensorDataset(torch.from_numpy(np.asarray(x)), torch.from_numpy(np.asarray(tensorflow.transpose(edge_index))))
        data_list.append(data)

    return DataLoader(data_list, batch_size=3, shuffle=False), data_list


# def train_step(batch, labels, reg):
#     out = reg(batch)
#     loss = F.mse_loss(out, labels.to(torch.float), reduction='mean')
#     loss.backward()
#     return loss


# def test_step(batch, labels, reg):
#     out = reg(batch)
#     loss = F.mse_loss(out, labels.to(torch.float), reduction='mean')
#     return loss


# def train_fn(train_loader, train_labels_loader, reg, opt):
#     reg.train()
#     total_loss = 0
#     for idx, (batch, labels) in enumerate(zip(train_loader, train_labels_loader)):
#         loss = train_step(batch, labels, reg)
#         total_loss += loss.item()

#     torch.nn.utils.clip_grad_norm_(reg.parameters(), 1)    
#     opt.step()
#     opt.zero_grad()
#     return total_loss/len(train_loader)


# def test_fn(test_loader, test_labels_loader, reg):
#     reg.eval()
#     total_loss = 0
#     with torch.no_grad():
#         for idx, (batch, labels) in enumerate(zip(test_loader, test_labels_loader)):
#             loss = test_step(batch, labels, reg)
#             total_loss += loss.item()
    
#     total_loss /= len(test_loader)
        
#     return total_loss


if __name__ == '__main__':

    ############################################
    ## GRAPH CONVOLUTIONS
    ############################################

    #TODO: test if it works
    with open('data/alvadesc/fingerprints/smiles.txt') as f:
        smiles_list = f.read().splitlines()

    # smiles_list = ['Cc1cc(c(C)n1c2ccc(F)cc2)S(=O)(=O)NCC(=O)N',
    # 'CN(CC(=O)N)S(=O)(=O)c1c(C)n(c(C)c1S(=O)(=O)N(C)CC(=O)N)c2ccc(F)cc2',
    # 'Fc1ccc(cc1)n2cc(COC(=O)CBr)nn2',
    # 'CCOC(=O)COCc1cn(nn1)c2ccc(F)cc2',
    # 'COC(=O)COCc1cn(nn1)c2ccc(F)cc2',
    # 'Fc1ccc(cc1)n2cc(COCC(=O)OCc3cn(nn3)c4ccc(F)cc4)nn2']

    mol_list = [Chem.MolFromSmiles(smi) for smi in smiles_list]

    dloader, dlist = prepare_dataloader(mol_list)
    print(dlist)

    #FIXME: not working due to Data format
    for batch in dloader:
        break

    print(batch)

    #As soon as the batch is load correctly, this should work
    # neural_fp = NeuralFP(atom_features=2, fp_size=2214)
    # fps = neural_fp(batch) 
    # print(fps.shape)


    # ############################################
    # ## LEARNING FGPS THROUGH BACKPROPAGATION
    # ############################################

    # common_cols = ['pid', 'rt']

    # #Load fgps and descriptors
    # fgp = load_alvadesc_data(split_as_np=False)
    # descriptors = load_descriptors(split_as_np=False)
    # descriptors = descriptors.drop_duplicates()

    # descriptors_fgp = pd.merge(descriptors, fgp, on=common_cols)

    # def get_feature_names(x):
    #     return x.drop(common_cols, axis=1).columns

    # #Get fgps and descriptors' features
    # X_fgp = descriptors_fgp[get_feature_names(fgp)].values.astype('float32')
    # X_desc = descriptors_fgp[get_feature_names(descriptors)].values.astype('float32')
    # X = np.concatenate([X_desc, X_fgp], axis=1)

    # #Get the target
    # y = descriptors_fgp['rt'].values.astype('float32').flatten()

    # X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2) #del

    # batch_size = 64

    # train_loader, _ = prepare_dataloader(list(X_train), batch_size=batch_size)
    # test_loader, _ = prepare_dataloader(X_test, batch_size)

    # train_labels_loader = torch.utils.data.DataLoader(y_train, batch_size=batch_size)
    # #valid_labels_loader = torch.utils.data.DataLoader(valid.y, batch_size=batch_size)
    # test_labels_loader = torch.utils.data.DataLoader(y_test, batch_size=batch_size)

    # reg = MLPRegressor(atom_features=2, fp_size=2048, hidden_size=100)
    # optimizer = torch.optim.SGD(reg.parameters(), lr=0.001, weight_decay=0.001)
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=100)

    # total_epochs = 1000
    # for epoch in range(1, total_epochs+1):
    #     train_loss = train_fn(train_loader, train_labels_loader, reg, opt=optimizer)
    #     #valid_loss = valid_fn(valid_loader, valid_labels_loader, reg)
    #     scheduler.step(train_loss)#valid_loss)

    #     if epoch % 10 == 0:
    #         print(f'Epoch:{epoch}, Train loss: {train_loss}')#, Valid loss: {valid_loss}')


