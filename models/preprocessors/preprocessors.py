import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_regression
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from models.preprocessors.threshold_selectors import CorThreshold
from models.preprocessors.column_selectors import make_col_selector


class Preprocessor(BaseEstimator, TransformerMixin):
    def __init__(self, desc_col_selector, fgp_col_selector, p=0.9, cor_th=0.9, k='all'):
        self.desc_col_selector = desc_col_selector
        self.fgp_col_selector = fgp_col_selector
        self.p = p
        self.cor_th = cor_th
        self.k = k

    def _init_hidden_models(self):
        # TODO: not included here the non-retained predictor for simplicity

        # Try out MultipleImputer (similar to MICE when sample_posterior = True and random_state = number)
        # Try out PCA and supress SelectKBest
        self._desc_pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('imputation', SimpleImputer(missing_values=np.nan, strategy='median', add_indicator=True)),
            ('var_threshold', VarianceThreshold()),
            ('cor_selector', CorThreshold(threshold=self.cor_th)),
            ('f_selector', SelectKBest(score_func=f_regression, k=self.k))
        ])
        self._fgp_vs = VarianceThreshold(threshold=self.p * (1 - self.p))

    def fit(self, X, y=None):
        self._init_hidden_models()
        X_desc = self.desc_col_selector(X)
        X_fgp = self.fgp_col_selector(X)

        self._desc_pipeline.fit(X_desc, y)
        self._fgp_vs.fit(X_fgp)
        return self

    def transform(self, X, y=None):
        X_desc = self.desc_col_selector(X)
        X_fgp = self.fgp_col_selector(X)

        X_desc_proc = self._desc_pipeline.transform(X_desc)
        X_fgp_proc = self._fgp_vs.transform(X_fgp)
        new_X = np.concatenate([X_desc_proc, X_fgp_proc], axis=1)

        self.transformed_desc_col_selector = make_col_selector(X_desc_proc.shape[1], return_first_half=True)
        self.transformed_fgp_col_selector = make_col_selector(X_desc_proc.shape[1], return_first_half=False)

        return new_X