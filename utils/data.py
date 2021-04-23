import os

import numpy as np
import pandas as pd
import pickle


def load_descriptors(n=None, split_as_np=True):
    target = 'rt'
    filename = 'data/alvadesc/descriptors/SMRT_descriptors.pkl'
    csv_filename = 'data/alvadesc/descriptors/SMRT_descriptors.csv'
    if os.path.exists(filename):
        with open(filename, 'rb') as fd:
            descriptors = pickle.load(fd)
    else:
        sample = pd.read_csv(csv_filename, nrows=100)
        float_cols = [c for c in sample if sample[c].dtype == "float64"]
        float32_cols = {c: np.float32 for c in float_cols}
        descriptors = pd.read_csv(csv_filename, engine='c', dtype=float32_cols)
        with open(filename, 'wb') as fd:
            pickle.dump(descriptors, fd)
    descriptors = descriptors.rename(columns={"pubchem": "pid"})
    if n:
        descriptors = descriptors.sample(n)
    if split_as_np:
        return (
            descriptors.drop(columns=['pid', target], axis=1).to_numpy().astype('float32'),
            descriptors[target].values.astype('float32').reshape(-1, 1)
        )
    else:
        return descriptors


def load_alvadesc_data(n=None, split_as_np=True):
    target = 'rt'
    filename = 'data/alvadesc/fingerprints/fingerprint.pkl'
    csv_filename = 'data/alvadesc/fingerprints/fingerprint.csv'
    if os.path.exists(filename):
        with open(filename, 'rb') as fd:
            fgp = pickle.load(fd)
    else:
        sample = pd.read_csv(csv_filename, nrows=100)
        float_cols = [c for c in sample if sample[c].dtype == "float64"]
        float32_cols = {c: np.float32 for c in float_cols}
        fgp = pd.read_csv(csv_filename, engine='c', dtype=float32_cols)
        with open(filename, 'wb') as fd:
            pickle.dump(fgp, fd)
    fgp = fgp.rename(columns={"pubchem": "pid"})
    if n:
        fgp = fgp.sample(n)
    if split_as_np:
        return (
            fgp.drop(columns=['pid', target], axis=1).to_numpy().astype('float32'),
            fgp[target].values.astype('float32').reshape(-1, 1)
        )
    else:
        return fgp


def is_non_retained(y):
    return (y < 300).astype('int')


def is_binary_feature(x):
    ux = np.unique(x)
    if len(ux) == 1:
        return ux == 0 or ux == 1
    if len(ux) == 2:
        return np.all(np.sort(ux) == np.array([0, 1]))
    else:
        return False