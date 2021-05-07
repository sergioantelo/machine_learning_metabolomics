import numpy as np

def make_col_selector(first_half_size, return_first_half, extra_cols_list=None):
    def splitter(X):
        assert X.ndim == 2, 'X ndim should be 2'
        if return_first_half:
            if extra_cols_list:
                assert not np.any([i in np.arange(first_half_size) for i in extra_cols_list])
                return np.concatenate([X[:, :first_half_size], X[:, extra_cols_list]], axis=1)
            else:
                return X[:, :first_half_size]
        else:
            if extra_cols_list:
                assert not np.any([i in np.arange(first_half_size, X.shape[1]) for i in extra_cols_list])
                return np.concatenate([X[:, first_half_size:], X[:, extra_cols_list]], axis=1)
            else:
                return X[:, first_half_size:]
    return splitter


