"""A scikit-learn compatible PyTorch MLP.

Implemented as a first-class registry entry rather than a special case in the RL core or
the executor: it exposes ``fit``/``predict``/``predict_proba`` and the constructor
parameters scikit-learn's ``clone`` expects, so it is swappable like any other model.

Two deliberate behaviours:

* CPU is always a valid device, and a CUDA request on a CPU-only machine degrades to CPU
  with a warning rather than raising (spec §27).
* Training holds out an internal validation slice for early stopping and restores the best
  weights, which stops the network from overfitting small tabular datasets.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.base import BaseEstimator

from rl_automl.core.errors import ModelFitError
from rl_automl.core.logging import get_logger

logger = get_logger("execution.torch_models")


def resolve_device(requested: str | None) -> str:
    """Pick a torch device string. Never fails: CPU is the fallback."""
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - guarded by registry availability
        raise ModelFitError(
            "the 'rl' extra is required for the MLP model: pip install rl-automl[rl]"
        ) from exc

    if requested in (None, "", "auto"):
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested.startswith("cuda") and not torch.cuda.is_available():
        logger.warning(
            "CUDA requested but unavailable; falling back to CPU",
            extra={"context": {"requested": requested}},
        )
        return "cpu"
    if requested == "cpu" or requested.startswith("cuda"):
        return requested
    return "cpu"


class _TorchBase(BaseEstimator):
    """Shared training loop. Subclasses only decide the output layer and loss."""

    _estimator_type = "regressor"

    def __init__(
        self,
        hidden_sizes: tuple[int, ...] = (128, 64),
        learning_rate: float = 1e-3,
        max_epochs: int = 80,
        batch_size: int = 256,
        dropout: float = 0.1,
        weight_decay: float = 1e-4,
        patience: int = 12,
        validation_fraction: float = 0.15,
        random_state: int = 0,
        device: str | None = None,
        verbose: bool = False,
    ) -> None:
        self.hidden_sizes = hidden_sizes
        self.learning_rate = learning_rate
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.dropout = dropout
        self.weight_decay = weight_decay
        self.patience = patience
        self.validation_fraction = validation_fraction
        self.random_state = random_state
        self.device = device
        self.verbose = verbose

    # -- construction ------------------------------------------------------------

    def _build_module(self, n_features: int, n_outputs: int) -> Any:
        import torch.nn as nn

        layers: list[Any] = []
        previous = n_features
        for size in tuple(self.hidden_sizes):
            layers.append(nn.Linear(previous, int(size)))
            layers.append(nn.ReLU())
            if self.dropout and self.dropout > 0:
                layers.append(nn.Dropout(float(self.dropout)))
            previous = int(size)
        layers.append(nn.Linear(previous, n_outputs))
        return nn.Sequential(*layers)

    def _loss_function(self) -> Any:
        raise NotImplementedError

    def _prepare_targets(self, y: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def _output_dim(self, y: np.ndarray) -> int:
        raise NotImplementedError

    # -- training ----------------------------------------------------------------

    def fit(self, X: Any, y: Any) -> _TorchBase:
        import torch

        matrix = _dense(X)
        if matrix.ndim != 2:
            raise ModelFitError("MLP expects a 2-D feature matrix")
        if not np.all(np.isfinite(matrix)):
            raise ModelFitError(
                "MLP received non-finite features; imputation and scaling must run first"
            )

        targets = np.asarray(y)
        if targets.ndim != 1:
            targets = targets.reshape(-1)
        if len(targets) != len(matrix):
            raise ModelFitError("feature and target lengths differ")

        self.n_features_in_ = int(matrix.shape[1])
        self.classes_ = self._resolve_classes(targets)
        encoded = self._prepare_targets(targets)
        n_outputs = self._output_dim(targets)

        device = resolve_device(self.device)
        torch.manual_seed(int(self.random_state))
        np.random.seed(int(self.random_state))

        self.module_ = self._build_module(self.n_features_in_, n_outputs).to(device)
        optimizer = torch.optim.Adam(
            self.module_.parameters(),
            lr=float(self.learning_rate),
            weight_decay=float(self.weight_decay),
        )
        loss_fn = self._loss_function()

        n_rows = len(matrix)
        train_idx, val_idx = _train_validation_split(
            n_rows, float(self.validation_fraction), int(self.random_state)
        )
        if len(train_idx) == 0:
            raise ModelFitError("no rows available for training after the internal split")

        features = torch.as_tensor(matrix, dtype=torch.float32).to(device)
        labels = torch.as_tensor(encoded).to(device)

        batch_size = max(1, min(int(self.batch_size), len(train_idx)))
        generator = torch.Generator().manual_seed(int(self.random_state))
        train_index = torch.as_tensor(train_idx).to(device)

        best_loss = float("inf")
        best_state: dict[str, Any] | None = None
        epochs_without_improvement = 0
        self.loss_curve_: list[float] = []

        for _epoch in range(int(self.max_epochs)):
            self.module_.train()
            permutation = torch.randperm(len(train_idx), generator=generator).to(device)
            epoch_loss = 0.0
            n_batches = 0

            for start in range(0, len(permutation), batch_size):
                batch = permutation[start : start + batch_size]
                rows = train_index[batch]
                optimizer.zero_grad()
                output = self.module_(features[rows])
                loss = loss_fn(output, labels[rows])
                loss.backward()
                optimizer.step()
                epoch_loss += float(loss.detach())
                n_batches += 1

            train_loss = epoch_loss / max(n_batches, 1)
            monitor_loss = self._validation_loss(
                features, labels, val_idx, loss_fn, train_loss, device
            )
            self.loss_curve_.append(monitor_loss)

            if monitor_loss < best_loss - 1e-6:
                best_loss = monitor_loss
                best_state = {
                    key: value.detach().clone() for key, value in self.module_.state_dict().items()
                }
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= int(self.patience):
                    break

        if best_state is not None:
            self.module_.load_state_dict(best_state)
        self.module_.eval()
        self.best_loss_ = best_loss
        self.n_iter_ = len(self.loss_curve_)
        self.device_ = device
        return self

    def _validation_loss(
        self,
        features: Any,
        labels: Any,
        val_idx: np.ndarray,
        loss_fn: Any,
        fallback: float,
        device: str,
    ) -> float:
        import torch

        if len(val_idx) == 0:
            return fallback
        self.module_.eval()
        with torch.no_grad():
            rows = torch.as_tensor(val_idx).to(device)
            output = self.module_(features[rows])
            return float(loss_fn(output, labels[rows]))

    # -- inference ---------------------------------------------------------------

    def _forward(self, X: Any) -> np.ndarray:
        import torch

        self._check_fitted()
        matrix = _dense(X)
        device = getattr(self, "device_", "cpu")
        model = self.module_.to(device)
        model.eval()

        outputs: list[np.ndarray] = []
        batch_size = max(1, int(self.batch_size))
        with torch.no_grad():
            for start in range(0, len(matrix), batch_size):
                chunk = torch.as_tensor(matrix[start : start + batch_size], dtype=torch.float32)
                outputs.append(model(chunk.to(device)).cpu().numpy())
        return np.vstack(outputs) if outputs else np.empty((0, 0))

    def _check_fitted(self) -> None:
        if not hasattr(self, "module_"):
            raise ModelFitError("this MLP instance has not been fitted")

    def to_cpu(self) -> _TorchBase:
        """Move the network to CPU. Used when loading an artifact for inference."""
        if hasattr(self, "module_"):
            self.module_ = self.module_.to("cpu")
            self.device_ = "cpu"
        return self

    @property
    def n_parameters(self) -> int:
        if not hasattr(self, "module_"):
            return 0
        return int(sum(p.numel() for p in self.module_.parameters()))

    def get_feature_names_out(self, input_features: Any = None) -> np.ndarray:
        n = int(getattr(self, "n_features_in_", 0))
        return np.asarray([f"x{i}" for i in range(n)], dtype=object)


class TorchMLPClassifier(_TorchBase):
    _estimator_type = "classifier"

    def _resolve_classes(self, y: np.ndarray) -> np.ndarray:
        classes = np.unique(y)
        if len(classes) < 2:
            raise ModelFitError(
                "the training split contains a single class; a classifier cannot be fitted"
            )
        if len(classes) > 100:
            raise ModelFitError(f"too many classes for an MLP ({len(classes)})")
        return classes

    def _output_dim(self, y: np.ndarray) -> int:
        classes = np.unique(y)
        return 1 if len(classes) == 2 else len(classes)

    def _prepare_targets(self, y: np.ndarray) -> np.ndarray:
        classes = np.unique(y)
        indices = np.searchsorted(classes, y)
        if len(classes) == 2:
            return indices.astype(np.float32).reshape(-1, 1)
        return indices.astype(np.int64)

    def _loss_function(self) -> Any:
        import torch.nn as nn

        if len(self.classes_) == 2:
            return nn.BCEWithLogitsLoss()
        return nn.CrossEntropyLoss()

    def predict_proba(self, X: Any) -> np.ndarray:
        logits = self._forward(X)
        if logits.shape[1] == 1:
            positive = 1.0 / (1.0 + np.exp(-logits[:, 0]))
            return np.column_stack([1.0 - positive, positive])
        shifted = logits - logits.max(axis=1, keepdims=True)
        exponentiated = np.exp(shifted)
        return exponentiated / exponentiated.sum(axis=1, keepdims=True)

    def predict(self, X: Any) -> np.ndarray:
        probabilities = self.predict_proba(X)
        return self.classes_[np.argmax(probabilities, axis=1)]


class TorchMLPRegressor(_TorchBase):
    _estimator_type = "regressor"

    def _resolve_classes(self, y: np.ndarray) -> None:
        return None

    def _output_dim(self, y: np.ndarray) -> int:
        return 1

    def _prepare_targets(self, y: np.ndarray) -> np.ndarray:
        return np.asarray(y, dtype=np.float32).reshape(-1, 1)

    def _loss_function(self) -> Any:
        import torch.nn as nn

        return nn.MSELoss()

    def predict(self, X: Any) -> np.ndarray:
        return self._forward(X).reshape(-1)


def _dense(X: Any) -> np.ndarray:
    array = X.toarray() if hasattr(X, "toarray") else np.asarray(X)
    return np.asarray(array, dtype=np.float32)


def _train_validation_split(
    n_rows: int, validation_fraction: float, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    if validation_fraction <= 0 or n_rows < 40:
        return np.arange(n_rows), np.empty(0, dtype=np.int64)

    rng = np.random.default_rng(seed)
    permutation = rng.permutation(n_rows)
    n_val = max(1, min(round(n_rows * validation_fraction)), n_rows - 1)
    return np.sort(permutation[n_val:]), np.sort(permutation[:n_val])


__all__ = ["TorchMLPClassifier", "TorchMLPRegressor", "resolve_device"]
