import numpy as np
import pandas as pd
from sklearn.model_selection import RepeatedStratifiedKFold


def stratify_y(y, n_bins=6):
    ps = np.linspace(0, 1, n_bins)
    # make sure last is 1, to avoid rounding issues
    ps[-1] = 1
    quantiles = np.quantile(y, ps)
    cuts = pd.cut(y, quantiles, include_lowest=True)
    codes, _ = cuts.factorize()
    return codes


class RegressionStratifiedKFold:
    """This class implements Repeated K-fold with stratification for regression. To that end, the targe y is binned
    in several classes (as indicated by n_bins). """
    def __init__(self, n_splits, n_repeats=1, n_bins=6, random_seed=None):
        """
        :param n_training: total number of training points in the whole dataset
        :param n_split_training: number of training points in a single split. Should be the same order of the required
        number of molecules requested to the user in CEU-Mass
        :param n_repeats:
        :param n_bins:
        """
        self.n_splits = n_splits
        self.n_repeats = n_repeats
        self.n_bins = n_bins
        self._base_kfold = RepeatedStratifiedKFold(
            n_splits=self.n_splits, n_repeats=self.n_repeats, random_state=random_seed
        )

    def split(self, X, y, groups=None):
        y_class = stratify_y(y, self.n_bins)
        for train_indices, test_indices in self._base_kfold.split(X, y_class):
            yield train_indices, test_indices

    def get_n_splits(self, X, y, groups=None):
        y_class = stratify_y(y, self.n_bins)
        return self._base_kfold.get_n_splits(X, y_class, groups)

