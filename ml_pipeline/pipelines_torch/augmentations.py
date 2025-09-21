"""Tabular augmentation utilities with official repo integrations."""

from __future__ import annotations

import importlib
from typing import Dict, Iterable, Optional, Sequence, Tuple

import numpy as np
from sklearn.neighbors import NearestNeighbors


def _try_import_module(candidates: Iterable[str]):
    for name in candidates:
        if not name:
            continue
        try:
            return importlib.import_module(name)
        except ImportError:
            continue
    return None


class MGS_GRFAugmentor:
    """Wrapper around the official ``artefactory/mgs-grf`` implementation."""

    def __init__(self, **kwargs) -> None:
        module = _try_import_module(("mgs_grf", "MGS_GRF", "third_party.mgs_grf"))
        if module is None:
            raise ImportError(
                "mgs-grf is not installed. Install the official repo via "
                "`pip install git+https://github.com/artefactory/mgs-grf`."
            )
        self._module = module
        self._kwargs = kwargs

    def fit_resample(self, X: np.ndarray, y: np.ndarray, target_class: Optional[int] = None, n_samples: Optional[int] = None):
        sampler = getattr(self._module, "MGS_GRF", None)
        if sampler is None:
            raise AttributeError("The installed mgs-grf package does not expose `MGS_GRF`. Check the upstream repository.")
        sampler = sampler(**self._kwargs)
        return sampler.fit_resample(X, y, target_class=target_class, n_samples=n_samples)


class TabEBMGenerator:
    """Wrapper for the official ``andreimargeloiu/TabEBM`` sampler."""

    def __init__(self, **kwargs) -> None:
        module = _try_import_module(("TabEBM", "tabebm", "tabebm.api"))
        if module is None:
            raise ImportError(
                "TabEBM is not installed. Install it with `pip install git+https://github.com/andreimargeloiu/TabEBM`."
            )
        self._module = module
        self._kwargs = kwargs
        self._model = None

    def fit(self, X: np.ndarray, y: np.ndarray, **fit_kwargs):
        fit_api = getattr(self._module, "fit_from_numpy", None)
        if fit_api is None:
            raise AttributeError("The TabEBM package must expose `fit_from_numpy`. Please check the upstream version.")
        args = {**self._kwargs, **fit_kwargs}
        self._model = fit_api(X, y, **args)
        return self

    def sample(self, n_per_class: int, **sample_kwargs) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("Call `fit` before sampling with TabEBM.")
        sample_api = getattr(self._module, "sample_numpy", None)
        if sample_api is None:
            raise AttributeError("The TabEBM package must expose `sample_numpy`. Please check the upstream version.")
        return sample_api(self._model, n_per_class=n_per_class, **sample_kwargs)


def _prepare_indices(n_features: int, categorical_indices: Optional[Sequence[int]]) -> Tuple[np.ndarray, np.ndarray]:
    categorical_indices = np.array(sorted(categorical_indices or []), dtype=int)
    mask = np.ones(n_features, dtype=bool)
    mask[categorical_indices] = False
    continuous = np.arange(n_features)[mask]
    return continuous, categorical_indices


class SimplicialSMOTE:
    """Implementation of Simplicial SMOTE (Kachan et al., 2025)."""

    def __init__(self, k_neighbors: int = 5, simplex_dim: int = 3, random_state: Optional[int] = None, categorical_indices: Optional[Sequence[int]] = None):
        if simplex_dim < 2:
            raise ValueError("simplex_dim must be at least 2")
        self.k_neighbors = k_neighbors
        self.simplex_dim = simplex_dim
        self.random_state = random_state
        self.categorical_indices = tuple(categorical_indices or [])

    def _generate_single(self, X: np.ndarray, idx: int, nn: NearestNeighbors, cont_idx: np.ndarray, cat_idx: np.ndarray) -> np.ndarray:
        neighbors = nn.kneighbors(X[idx][None, :], return_distance=False)[0]
        if neighbors.size < self.simplex_dim:
            choices = np.random.choice(neighbors, size=self.simplex_dim, replace=True)
        else:
            choices = np.random.choice(neighbors, size=self.simplex_dim, replace=False)
        simplex = X[choices]
        weights = np.random.dirichlet(np.ones(self.simplex_dim))

        sample = np.zeros_like(simplex[0])
        if cont_idx.size:
            sample[cont_idx] = np.dot(weights, simplex[:, cont_idx])
        if cat_idx.size:
            cat_values = simplex[:, cat_idx]
            # majority vote with weights
            for j, feature in enumerate(cat_idx):
                values, counts = np.unique(cat_values[:, j], return_counts=True)
                sample[feature] = values[np.argmax(counts)]
        return sample

    def fit_resample(self, X: np.ndarray, y: np.ndarray, sampling_strategy: Optional[Dict[int, int]] = None):
        rng = np.random.default_rng(self.random_state)
        X = np.asarray(X)
        y = np.asarray(y)
        classes, counts = np.unique(y, return_counts=True)
        majority = counts.max()

        if sampling_strategy is None:
            sampling_strategy = {cls: majority - cnt for cls, cnt in zip(classes, counts) if cnt < majority}

        cont_idx, cat_idx = _prepare_indices(X.shape[1], self.categorical_indices)
        X_new = [X]
        y_new = [y]

        for cls, n_add in sampling_strategy.items():
            if n_add <= 0:
                continue
            indices = np.where(y == cls)[0]
            if indices.size == 0:
                continue
            nn = NearestNeighbors(n_neighbors=min(len(indices), self.k_neighbors), metric="euclidean")
            nn.fit(X[indices][:, cont_idx] if cont_idx.size else X[indices])

            samples = []
            for _ in range(n_add):
                idx = rng.choice(indices)
                samples.append(self._generate_single(X, idx, nn, cont_idx, cat_idx))
            X_new.append(np.vstack(samples))
            y_new.append(np.full(n_add, cls))

        return np.vstack(X_new), np.concatenate(y_new)


class MEBSMOTE:
    """Minimum Enclosing Ball SMOTE (Shangguan et al., 2024)."""

    def __init__(self, k_neighbors: int = 5, random_state: Optional[int] = None, categorical_indices: Optional[Sequence[int]] = None):
        self.k_neighbors = k_neighbors
        self.random_state = random_state
        self.categorical_indices = tuple(categorical_indices or [])

    def _generate(self, X: np.ndarray, idx: int, nn: NearestNeighbors, cont_idx: np.ndarray, cat_idx: np.ndarray) -> np.ndarray:
        neighbors = nn.kneighbors(X[idx][None, :], return_distance=False)[0]
        points = X[neighbors]
        sample = X[idx].copy()

        if cont_idx.size:
            cont_points = points[:, cont_idx]
            center = cont_points.mean(axis=0)
            radius = np.linalg.norm(cont_points - center, axis=1).mean()
            direction = center - sample[cont_idx]
            alpha = np.random.rand()
            sample[cont_idx] = sample[cont_idx] + alpha * direction
            if radius > 0:
                noise = np.random.normal(scale=0.1 * radius, size=cont_idx.size)
                sample[cont_idx] += noise

        if cat_idx.size:
            cat_points = points[:, cat_idx]
            for j, feature in enumerate(cat_idx):
                values, counts = np.unique(cat_points[:, j], return_counts=True)
                sample[feature] = values[np.argmax(counts)]

        return sample

    def fit_resample(self, X: np.ndarray, y: np.ndarray, sampling_strategy: Optional[Dict[int, int]] = None):
        rng = np.random.default_rng(self.random_state)
        X = np.asarray(X)
        y = np.asarray(y)
        classes, counts = np.unique(y, return_counts=True)
        majority = counts.max()

        if sampling_strategy is None:
            sampling_strategy = {cls: majority - cnt for cls, cnt in zip(classes, counts) if cnt < majority}

        cont_idx, cat_idx = _prepare_indices(X.shape[1], self.categorical_indices)
        X_new = [X]
        y_new = [y]

        for cls, n_add in sampling_strategy.items():
            if n_add <= 0:
                continue
            indices = np.where(y == cls)[0]
            if indices.size == 0:
                continue
            nn = NearestNeighbors(n_neighbors=min(len(indices), self.k_neighbors), metric="euclidean")
            nn.fit(X[indices][:, cont_idx] if cont_idx.size else X[indices])

            samples = []
            for _ in range(n_add):
                idx = rng.choice(indices)
                samples.append(self._generate(X, idx, nn, cont_idx, cat_idx))
            X_new.append(np.vstack(samples))
            y_new.append(np.full(n_add, cls))

        return np.vstack(X_new), np.concatenate(y_new)


def none_augmentation(X: np.ndarray, y: np.ndarray, **_):
    return X, y


def simplicial_smote_augmentation(X: np.ndarray, y: np.ndarray, **kwargs):
    sampler = SimplicialSMOTE(**kwargs)
    return sampler.fit_resample(X, y)


def meb_smote_augmentation(X: np.ndarray, y: np.ndarray, **kwargs):
    sampler = MEBSMOTE(**kwargs)
    return sampler.fit_resample(X, y)


AUGMENTATION_REGISTRY: Dict[str, callable] = {
    "none": none_augmentation,
    "simplicial_smote": simplicial_smote_augmentation,
    "meb_smote": meb_smote_augmentation,
}


__all__ = [
    "MGS_GRFAugmentor",
    "TabEBMGenerator",
    "SimplicialSMOTE",
    "MEBSMOTE",
    "AUGMENTATION_REGISTRY",
]

