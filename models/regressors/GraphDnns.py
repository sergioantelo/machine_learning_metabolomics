import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import MessagePassing
from torch_scatter import scatter_add
from torch_geometric.utils import add_self_loops, degree

class NeuralLoop(MessagePassing):
    def __init__(self, atom_features, fp_size):
        super(NeuralLoop, self).__init__(aggr='add')
        self.H = nn.Linear(atom_features, atom_features)
        self.W = nn.Linear(atom_features, fp_size)
        
    def forward(self, x, edge_index):
        # x shape: [Number of atoms in molecule, Number of atom features]; [N, in_channels]
        # edge_index shape: [2, E]; E is the number of edges
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
        return self.propagate(edge_index, size=(x.size(0), x.size(0)), x=x)
    
    def message(self, x_j, edge_index, size):
        # We simply sum all the neighbouring nodes (including self-loops)
        # This is done implicitly by PyTorch-Geometric :)
        return x_j 
    
    def update(self, v):
        
        updated_atom_features = self.H(v.float()).sigmoid()
        updated_fingerprint = self.W(updated_atom_features).softmax(dim=-1)
        
        return updated_atom_features, updated_fingerprint # shape [N, atom_features]
    
class NeuralFP(nn.Module):
    def __init__(self, atom_features=2, fp_size=2214):
        super(NeuralFP, self).__init__()
        
        self.atom_features = 2
        self.fp_size = 2214
        
        self.loop1 = NeuralLoop(atom_features=atom_features, fp_size=fp_size)
        self.loop2 = NeuralLoop(atom_features=atom_features, fp_size=fp_size)
        self.loops = nn.ModuleList([self.loop1, self.loop2])
        
    def forward(self, data):
        fingerprint = torch.zeros((data.batch.shape[0], self.fp_size), dtype=torch.float)
        
        out = data.x
        for idx, loop in enumerate(self.loops):
            updated_atom_features, updated_fingerprint = loop(out, data.edge_index)
            out = updated_atom_features
            fingerprint += updated_fingerprint
            
        return scatter_add(fingerprint, data.batch, dim=0)


class MLP_Regressor(nn.Module):
    def __init__(self, neural_fp, atom_features=2, fp_size=2048, hidden_size=100):
        super(MLP_Regressor, self).__init__()
        self.neural_fp = neural_fp
        self.lin1 = nn.Linear(fp_size, hidden_size)
        self.lin2 = nn.Linear(hidden_size, 1)
        self.dropout = nn.Dropout(p=0.3)
    
    def forward(self, batch):
        fp = self.neural_fp(batch)
        hidden = F.relu(self.dropout(self.lin1(fp)))
        return fp.t()#out