import hashlib
import os
import pickle
import time
from functools import wraps

import dgl
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from dgllife.model.model_zoo import AttentiveFPPredictor
from dgllife.utils import AttentiveFPAtomFeaturizer
from dgllife.utils import AttentiveFPBondFeaturizer
from dgllife.utils import CanonicalAtomFeaturizer
from dgllife.utils import CanonicalBondFeaturizer
from dgllife.utils import mol_to_bigraph
from rdkit import Chem
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import FunctionTransformer
from sklearn.preprocessing import RobustScaler
from torch.utils.data import DataLoader

from utils.data import load_alvadesc_data


def load_mols_df():
    pids = []
    mss = []
    for file in os.listdir('data/sdfs'):
        with Chem.SDMolSupplier(os.path.join('data/sdfs', file)) as suppl:
            pid = int(file.strip('.sdf'))
            ms = [x for x in suppl if x is not None]
            assert len(ms) == 1, 'Unexpected number of molecules (!= 1)'
            pids.append(pid)
            mss.append(ms[0])
    df_mols = pd.DataFrame({'pid': pids, 'mol': mss})

    fgp = load_alvadesc_data(split_as_np=False)
    df_rts = pd.DataFrame({'pid': fgp['pid'], 'rt': fgp['rt']})
    df_rts['pid'] = df_rts['pid'].astype('int')
    df_rts['rt'] = df_rts['rt'].astype('float32')

    return pd.merge(df_mols, df_rts, on='pid')


def memorize(fun):
    """ Memoization decorator, intended to cache the results for the graph building function to disk. np.arrays
    or pandas dataframes are hex-hashed. The other arguments are expected to be convertable to strings
    """
    CACHE_PATH = '.cache'
    if not os.path.exists(CACHE_PATH):
        os.makedirs(CACHE_PATH)

    def hash_argument(arg):
        if hasattr(arg, '__name__'):
            return arg.__name__
        if isinstance(arg, pd.DataFrame):
            # Python's hash function is randomized for security reasons. Hence, use hashlib.
            # Use only first 10 characters to avoid "name too long errors"
            return hashlib.sha1(arg.values).hexdigest()[:10]
        if isinstance(arg, np.ndarray):
            try:
                return hashlib.sha1(arg).hexdigest()[:10]
            except ValueError:
                # In case the numpy array is not C-ordered, fix this
                return hashlib.sha1(arg.copy(order='C')).hexdigest()[:10]
        return str(arg)[:10]

    @wraps(fun)
    def new_fun(*args, **kwargs):
        string_args = ''
        if len(args) > 0:
            string_args += '_' + '_'.join([hash_argument(arg) for arg in args])
        if len(kwargs) > 0:
            string_args += '_' + '_'.join([(str(k)[:10] + hash_argument(v)) for k, v in kwargs.items()])

        filename = os.path.join(CACHE_PATH, '.cache_{}{}.pickle'.format(fun.__name__, string_args))

        if os.path.exists(filename):
            with open(filename, 'rb') as file:
                result = pickle.load(file)
        else:
            result = fun(*args, **kwargs)
            with open(filename, 'wb') as file:
                pickle.dump(result, file)
        return result
    return new_fun


def get_atom_featurizer(atom_featurizer):
    if atom_featurizer == 'canonical':
       return CanonicalAtomFeaturizer()
    elif atom_featurizer == 'attentive_featurizer':
        return AttentiveFPAtomFeaturizer()
    else:
        raise ValueError('Invalid atom featurizer')


def get_bond_featurizer(bond_featurizer, self_loop):
    if bond_featurizer == 'canonical':
       return CanonicalBondFeaturizer(self_loop=self_loop)
    elif bond_featurizer == 'attentive_featurizer':
        return AttentiveFPBondFeaturizer(self_loop=self_loop)
    else:
        raise ValueError('Invalid bond featurizer')


def get_transformer(transformer):
    if transformer == 'none':
        return FunctionTransformer()
    elif transformer == 'robust':
        return RobustScaler()
    else:
        raise ValueError('Invalid transformer')


@memorize
def build_graph_and_transform_target(train, test, atom_alg, bond_alg, transformer_alg, self_loop):
    (X_train, y_train) = train
    (X_test, y_test) = test

    atom_featurizer = get_atom_featurizer(atom_alg)
    bond_featurizer = get_bond_featurizer(bond_alg, self_loop)
    transformer = get_transformer(transformer_alg)

    y_train = y_train.reshape(-1, 1)
    y_test = y_test.reshape(-1, 1)
    if transformer is not None:
        y_train = transformer.fit_transform(y_train)
        y_test = transformer.transform(y_test)

    def featurize(x, y):
        # each item is a duple of type (graph(x), y)
        return (
            mol_to_bigraph(x, node_featurizer=atom_featurizer,
                        edge_featurizer=bond_featurizer,
                        add_self_loop=self_loop),
            y
        )

    train = [featurize(x_i, y_i) for x_i, y_i in zip(X_train, y_train)]
    test = [featurize(x_i, y_i) for x_i, y_i in zip(X_test, y_test)]
    return train, test, transformer



def collate_molgraphs(data):
    assert len(data[0]) == 2, 'ooops'
    graphs, labels = map(list, zip(*data))

    bg = dgl.batch(graphs)
    bg.set_n_initializer(dgl.init.zero_initializer)
    bg.set_e_initializer(dgl.init.zero_initializer)
    labels = torch.tensor(labels, dtype=torch.float32).reshape(-1, 1)
    masks = torch.ones(labels.shape)
    return bg, labels, masks


def to_cuda(bg, labels, masks):
    if torch.cuda.is_available():
        bg = bg.to(torch.device('cuda:0'))
        labels = labels.to('cuda:0')
        masks = masks.to('cuda:0')
    return bg, labels, masks


def train_step(reg, bg, labels, masks, loss_criterion, optimizer):
    optimizer.zero_grad()
    prediction = reg(bg, bg.ndata['h'], bg.edata['e'])
    loss = (loss_criterion(prediction, labels, reduction='none') * (masks != 0).float()).mean()
    loss.backward()
    optimizer.step()
    return loss.data.item()


def eval_step(reg, bg, labels, masks, loss_criterion, transformer):
    """ Compute loss_criterion and the absolute error after undoing the transformation from transformer """
    prediction = reg(bg, bg.ndata['h'], bg.edata['e'])
    loss = (loss_criterion(prediction, labels, reduction='none') * (masks != 0).float()).mean().item()

    prediction = prediction.cpu().numpy().reshape(-1, 1)
    labels = labels.cpu().numpy().reshape(-1, 1)
    if transformer is not None:
        abs_errors = np.abs(
            transformer.inverse_transform(prediction) - transformer.inverse_transform(labels)
        )
    else:
        abs_errors = np.abs(prediction - labels)
    return loss, abs_errors


if __name__ == '__main__':
    batch_size = 256
    fp_size = 1024
    total_epochs = 40
    self_loop = True
    graph_feat_size = 128
    num_layers = 2
    dropout = 0.2
    learning_rate = 1e-3
    SEED = 129767345
    #########################
    rt_scaler = 'robust'
    atom_featurizer = 'canonical'
    bond_featurizer = 'canonical'
    #######################

    df_mols_rts = load_mols_df()
    X = df_mols_rts['mol'].values
    y = df_mols_rts['rt'].values.astype('float32').flatten()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=SEED)

    print('Building graphs...', end='')
    start = time.time()
    train, test, transformer = build_graph_and_transform_target(
        (X_train, y_train),
        (X_test, y_test),
        atom_alg=atom_featurizer,
        bond_alg=bond_featurizer,
        transformer_alg=rt_scaler,
        self_loop=self_loop
    )
    print(f'Done! (Ellapsed: {time.time() - start})')

    train_loader = DataLoader(train, batch_size=batch_size, shuffle=True, collate_fn=collate_molgraphs)
    test_loader = DataLoader(test, batch_size=batch_size, shuffle=False, collate_fn=collate_molgraphs)

    # Get a sample of the graph to know the node_feat_size and the edge_feat_size
    graph, y, masks = next(iter(test_loader))
    reg = AttentiveFPPredictor(node_feat_size=graph.ndata['h'].shape[1],
                               edge_feat_size=graph.edata['e'].shape[1],
                               graph_feat_size=graph_feat_size,
                               num_layers=num_layers,
                               dropout=dropout)

    if torch.cuda.is_available():
        print('using CUDA!')
        reg = reg.cuda()

    optimizer = torch.optim.Adam(reg.parameters(), lr=learning_rate)

    train_losses = []
    test_losses = []
    maes = []
    medaes = []
    loss_criterion = F.smooth_l1_loss


    for epoch in range(1, total_epochs + 1):
        reg.train()
        epoch_losses = []
        for batch_id, (bg, labels, masks) in enumerate(train_loader):
            bg, labels, masks = to_cuda(bg, labels, masks)
            loss = train_step(reg, bg, labels, masks, loss_criterion, optimizer)
            epoch_losses.append(loss)
        train_losses.append(np.mean(epoch_losses))

        reg.eval()
        with torch.no_grad():
            epoch_losses = []
            abs_errors = []
            for batch_id, (bg, labels, masks) in enumerate(test_loader):
                bg, labels, masks = to_cuda(bg, labels, masks)
                loss, absolute_errors = eval_step(reg, bg, labels, masks, loss_criterion, transformer)
                epoch_losses.append(loss)
                abs_errors.append(absolute_errors)

        test_losses.append(np.mean(epoch_losses))
        maes.append(
            np.mean(np.concatenate(abs_errors))
        )
        medaes.append(
            np.median(np.concatenate(abs_errors))
        )

        if epoch % 1 == 0:
            print(f'Epoch:{epoch}, Train loss: {train_losses[-1]}, Test loss: {test_losses[-1]}, Test MEDAE: {medaes[-1]}, Test MAE: {maes[-1]}' )

    losses = pd.DataFrame({
        'epoch': np.arange(len(train_losses)),
        'train_loss': train_losses,
        'test_loss': test_losses,
        'medae': medaes,
        'mae': maes
    })
    losses.index = losses.epoch
    # TODO: change name depending on arguments
    losses.to_csv('losses.csv', index=False)
    print('Done')

    import matplotlib.pyplot as plt
    losses[['train_loss', 'test_loss']].plot()
    plt.show()