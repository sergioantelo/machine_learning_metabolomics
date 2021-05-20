import inspect

from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.compose import TransformedTargetRegressor
from sklearn.preprocessing import QuantileTransformer
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from torch.optim.swa_utils import AveragedModel, SWALR

from utils.train.torchutils import EarlyStopping
from utils.train.model_selection import stratify_y


class DnnModel(nn.Module):
    def __init__(self, n_features):
        super().__init__()
        sizes = [1512, 128]
        dropouts = [0.5, 0.1]
        self.l1 = nn.Linear(n_features, sizes[0])
        nn.init.zeros_(self.l1.bias)
        self.d1 = nn.Dropout(dropouts[0])
        self.l2 = nn.Linear(sizes[0], sizes[1])
        nn.init.zeros_(self.l2.bias)
        self.d2 = nn.Dropout(dropouts[1])
        self.l_out = nn.Linear(sizes[1], 1)
        nn.init.zeros_(self.l_out.bias)

    def forward(self, x):
        x = self.d1(F.gelu(self.l1(x))) #reason? RELU, ELU, SELU, Sigmoid, TanH?
        x = self.d2(F.gelu(self.l2(x)))
        return self.l_out(x)


class SkDnnModel(BaseEstimator, RegressorMixin):
    def __init__(self, n_features):
        self.n_features = n_features
        # Some constants to make the algorithm work
        self._batch_size = 64
        self._T0 = 30
        self._n_epochs = 2 * self._T0
        self._swa_epochs = 20
        self._lr = 3e-4
        self._device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(self._device)

    def _init_hidden_model(self):
        self._model = DnnModel(self.n_features).to(self._device)
        min_lr = 0.1 * self._lr
        self._optimizer = torch.optim.Adam(self._model.parameters(), lr=self._lr) #reason? SGD better but slower, or AdaBound, Adam variant using dynamic bounds
                                                                                  #on learning rates to achieve a smooth transition to SGD (as fast as Adama and 
                                                                                  #as good as SGD)
        self._scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self._optimizer, T_0=self._T0, T_mult=1, eta_min=min_lr
        ) #reason? why Adam and then SWA?
        self._swa_model = AveragedModel(self._model) 
        self._swa_scheduler = SWALR(self._optimizer, swa_lr=min_lr)

    def fit(self, X, y):
        self._init_hidden_model()
        dataset = TensorDataset(torch.from_numpy(X), torch.from_numpy(y).view(-1, 1))
        data_loader = DataLoader(dataset, batch_size=self._batch_size, shuffle=True) #batches?

        self._model.train()
        train_iters = len(data_loader)
        for epoch in range(self._n_epochs):
            for i, (xb, yb) in enumerate(data_loader): #?
                self._batch_step(xb, yb)
                self._scheduler.step(epoch + i / train_iters)

        self._swa_model.train()
        for epoch in range(self._swa_epochs):
            for xb, yb in data_loader:
                self._batch_step(xb, yb)
            self._swa_model.update_parameters(self._model)
            self._swa_scheduler.step()

        return self

    def _batch_step(self, xb, yb):
        self._optimizer.zero_grad()
        pred = self._model(xb.to(self._device))
        loss = F.l1_loss(pred, target=yb.to(self._device))
        loss.backward()
        self._optimizer.step()

    def predict(self, X):
        self._model.eval()
        self._swa_model.eval()
        with torch.no_grad():
            return self._swa_model(torch.from_numpy(X).to(self._device)).cpu().numpy().flatten()


class SkTransformedDnnModel(BaseEstimator, RegressorMixin):
    def __init__(self, n_features, hidden_sizes=[2048, 1024, 512, 128], dropout=0.2,
                 use_bn=False, use_wn=False, lr=1e-3, max_epochs=10000, batch_size=64,
                 patience=5, train_split=0.1,
                 optimizer=torch.optim.Adam, lr_scheduler=None,
                 weight_decay=0, selector=None):
        super().__init__()
        args, _, _, values = inspect.getargvalues(inspect.currentframe())
        values.pop("self")
        for arg, val in values.items():
            setattr(self, arg, val)

    def _select_features(self, X):
        if self.selector is not None:
            X_ = self.selector(X)
            return X_
        else:
            return X

    def fit(self, X, y):
        self._model = TransformedTargetRegressor(
            regressor=SkDnnModel(self.n_features),
            transformer=QuantileTransformer(n_quantiles=1000, output_distribution='normal')
        )
        self._model.fit(self._select_features(X), y)
        return self

    def predict(self, X):
        return self._model.predict(self._select_features(X))

