"""Un MLP PyTorch compatible avec GridSearchCV de scikit-learn.

Le petit adaptateur ci-dessous rend explicites fit/predict et la boucle
d'apprentissage. Les prétraitements restent à l'extérieur, dans les pipelines.
"""

from copy import deepcopy
import random

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.utils.validation import check_is_fitted


def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    # Les petits tableaux sont plus rapides avec peu de threads.
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)


class MLP(nn.Module):
    def __init__(self, inputs, outputs, hidden=(64, 32), dropout=0.):
        super().__init__()
        layers = []
        previous = inputs
        for width in hidden:
            layers.extend([nn.Linear(previous, width), nn.ReLU()])
            if dropout:
                layers.append(nn.Dropout(dropout))
            previous = width
        # Pas de softmax ici : CrossEntropyLoss attend des logits bruts.
        layers.append(nn.Linear(previous, outputs))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class _TorchEstimator(BaseEstimator):
    def __init__(self, hidden=(64, 32), lr=.001, optimizer="adam", epochs=30,
                 batch_size=512, dropout=0., weight_decay=0., patience=None,
                 random_state=42):
        # Garder les paramètres inchangés permet à sklearn.clone de fonctionner.
        self.hidden = hidden
        self.lr = lr
        self.optimizer = optimizer
        self.epochs = epochs
        self.batch_size = batch_size
        self.dropout = dropout
        self.weight_decay = weight_decay
        self.patience = patience
        self.random_state = random_state

    def fit(self, X, y, validation_data=None, curve_sets=None, record_states=False):
        """Apprendre sur X/y ; seul validation_data peut arrêter l'entraînement.

        curve_sets sert uniquement à enregistrer des mesures. Lors du calcul des
        courbes de test après sélection, aucun de ces scores ne pilote le modèle.
        """
        seed_everything(self.random_state)
        X = np.asarray(X, dtype=np.float32)
        self.n_features_in_ = X.shape[1]
        self.is_classification_ = isinstance(self, TorchClassifier)
        if self.is_classification_:
            self.classes_ = np.unique(y)
            target = np.searchsorted(self.classes_, y)
            outputs = len(self.classes_)
            criterion = nn.CrossEntropyLoss()
        else:
            # Standardiser la cible dans fit évite une fuite entre folds CV.
            self.y_mean_ = float(np.mean(y))
            self.y_scale_ = max(float(np.std(y)), 1e-8)
            target = (np.asarray(y) - self.y_mean_) / self.y_scale_
            outputs = 1
            criterion = nn.MSELoss()
        self.model_ = MLP(X.shape[1], outputs, self.hidden, self.dropout)
        optimizers = {"adam": torch.optim.Adam, "sgd": torch.optim.SGD}
        kwargs = {"momentum": .9} if self.optimizer == "sgd" else {}
        optimizer = optimizers[self.optimizer](self.model_.parameters(), lr=self.lr,
                                               weight_decay=self.weight_decay, **kwargs)
        yt = torch.as_tensor(target, dtype=torch.long if self.is_classification_
                             else torch.float32)
        loader = DataLoader(TensorDataset(torch.from_numpy(X), yt),
                            batch_size=self.batch_size, shuffle=True,
                            generator=torch.Generator().manual_seed(self.random_state))
        self.history_ = []
        self.epoch_states_ = []
        best_loss, stale, best_state, best_epoch = float("inf"), 0, None, 0
        for epoch in range(1, self.epochs + 1):
            self.model_.train()
            for xb, yb in loader:
                optimizer.zero_grad()
                pred = self.model_(xb)
                if not self.is_classification_:
                    pred = pred.squeeze(1)
                loss = criterion(pred, yb)
                loss.backward()
                optimizer.step()
            row = {"epoch": epoch}
            for name, (features, labels) in (curve_sets or {}).items():
                row.update({f"{name}_{key}": value for key, value in
                            self._measure(features, labels).items()})
            if validation_data is not None:
                metrics = self._measure(*validation_data)
                row.update({f"validation_{key}": value for key, value in metrics.items()})
                if metrics["loss"] < best_loss - 1e-7:
                    best_loss, stale = metrics["loss"], 0
                    best_state, best_epoch = deepcopy(self.model_.state_dict()), epoch
                else:
                    stale += 1
            self.history_.append(row)
            if record_states:
                self.epoch_states_.append(deepcopy(self.model_.state_dict()))
            if self.patience is not None and stale >= self.patience:
                break
        self.epochs_run_ = len(self.history_)
        self.selected_epoch_ = self.epochs_run_
        if self.patience is not None and best_state is not None:
            self.model_.load_state_dict(best_state)
            self.selected_epoch_ = best_epoch
        return self

    def _logits(self, X):
        check_is_fitted(self, "model_")
        self.model_.eval()  # Dropout désactivé pour une comparaison équitable.
        with torch.no_grad():
            return self.model_(torch.as_tensor(np.asarray(X, dtype=np.float32))).numpy()

    def predict(self, X):
        logits = self._logits(X)
        if self.is_classification_:
            return self.classes_[logits.argmax(axis=1)]
        return logits[:, 0] * self.y_scale_ + self.y_mean_

    def _measure(self, X, y):
        logits = self._logits(X)
        if self.is_classification_:
            target = np.searchsorted(self.classes_, y)
            loss = nn.functional.cross_entropy(torch.from_numpy(logits),
                                                torch.as_tensor(target)).item()
            return {"loss": loss,
                    "accuracy": float(np.mean(self.classes_[logits.argmax(1)] == y))}
        prediction = logits[:, 0] * self.y_scale_ + self.y_mean_
        mse = float(np.mean((prediction - np.asarray(y)) ** 2))
        return {"loss": mse, "rmse": float(np.sqrt(mse))}


class TorchRegressor(RegressorMixin, _TorchEstimator):
    """Régression : sortie continue et erreur quadratique."""


class TorchClassifier(ClassifierMixin, _TorchEstimator):
    """Classification multiclasse : logits et entropie croisée."""

    def predict_proba(self, X):
        return torch.softmax(torch.from_numpy(self._logits(X)), dim=1).numpy()
