import copy
import inspect
from sklearn.base import BaseEstimator, RegressorMixin
from catboost import CatBoostRegressor, Pool

class WeightedCatBoostRegressor(BaseEstimator, RegressorMixin):
    def __init__(self,
                 iterations=None,
                 learning_rate=None,
                 depth=None,
                 l2_leaf_reg=None,
                 random_strength=None,
                 model_size_reg=None,
                 loss_function='RMSE',
                 verbose=False,
                 weight_function=None,
                 eval_metric=None):
        super().__init__()
        args, _, _, values = inspect.getargvalues(inspect.currentframe())
        values.pop("self")
        for arg, val in values.items():
            setattr(self, arg, val)

    def _catboost_params(self):
        params = copy.copy(self.get_params())
        params.pop('weight_function')
        return params

    def fit(self, X, y):
        self._model = CatBoostRegressor(**self._catboost_params())
        if self.weight_function:
            self._model.fit(
                Pool(data=X, label=y, weight=
                self.weight_function(y))
            )
        else:
            self._model.fit(X, y)
        return self

    def predict(self, X):
        return self._model.predict(X)
