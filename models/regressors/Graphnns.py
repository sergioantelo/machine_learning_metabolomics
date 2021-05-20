# Check HMDB webpage: https://hmdb.ca/structures/search/metabolites/structure
# Check RDKIT webpage: https://www.rdkit.org/docs/GettingStartedInPython.html
# Check RDKIT documentation: http://www.rdkit.org/RDKit_Docs.2012_12_1.pdf

import numpy as np

from rdkit import Chem
from rdkit.Chem import rdmolops
import igraph
from igraph import plot
from igraph import GraphBase

# Function to convert molecule to graph, and add some node and edge attributes.
def mol2graph(mol):
    admatrix = rdmolops.GetAdjacencyMatrix(mol)
    bondidxs = [(b.GetBeginAtomIdx(), b.GetEndAtomIdx())
                for b in mol.GetBonds()]
    adlist = np.ndarray.tolist(admatrix)
    graph = igraph.Graph()
    g = graph.Adjacency(adlist).as_undirected()
    for idx in g.vs.indices:
        g.vs[idx]["AtomicNum"] = mol.GetAtomWithIdx(idx).GetAtomicNum()
        g.vs[idx]["AtomicSymbole"] = mol.GetAtomWithIdx(idx).GetSymbol()
    for bd in bondidxs:
        btype = mol.GetBondBetweenAtoms(bd[0], bd[1]).GetBondTypeAsDouble()
        g.es[g.get_eid(bd[0], bd[1])]["BondType"] = btype
        print(bd, mol.GetBondBetweenAtoms(bd[0], bd[1]).GetBondTypeAsDouble())
    return g


# Load molecule from Smiles and convert it to graph
m = Chem.MolFromSmiles('Cc1ccccc1')
gr = mol2graph(m)
plot(gr)

# Add new options for plotting
# layout = gr.layout_graphopt()
# color_dict = {"C": "gray", "N": "blue", "O":  "white"}
# my_plot = igraph.Plot()
# my_plot.add(gr, layout=layout, bbox=(400, 400),
#             margin=90,
#             vertex_color=[color_dict[atom] for atom in gr.vs["AtomicSymbole"]],
#             vertex_size=[v.degree()*10 for v in gr.vs])
# my_plot.show()
