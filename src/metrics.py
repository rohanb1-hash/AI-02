"""
Conventions
-----------
* A representation is a matrix ``H`` of shape ``(n_examples, n_features)``.
  Arrays with more than two dimensions (e.g. conv features ``N, C, H, W``) are
  flattened to ``(N, -1)``. NumPy arrays and PyTorch tensors are both accepted.
* All computation is done in float64 NumPy (accurate and deterministic; the
  float32 versions lose precision for rank / CKA on wide layers). Results are
  plain Python floats.
* Every metric that compares two representations requires the *same examples
  in the same order*; the feature dimensions may differ.
* For large layers, compute metrics on a fixed subset of examples
  (e.g. 2000-5000) and use the same subset for every model you compare.

Contents
--------
effective_rank, stable_rank
linear_cka, debiased_linear_cka
rdm, rsa
procrustes_distance
linear_probe
"""

from __future__ import annotations

import numpy as np
from scipy.stats import rankdata
from scipy.spatial.distance import pdist

__all__ = [
    "effective_rank",
    "stable_rank",
    "linear_cka",
    "debiased_linear_cka",
    "rdm",
    "rsa",
    "procrustes_distance",
    "linear_probe",
]

_EPS = 1e-12


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def _as_matrix(H, name: str = "H") -> np.ndarray:
    """Convert to a finite float64 matrix of shape (n_examples, n_features)."""
    if hasattr(H, "detach"):  # torch tensor, without importing torch
        H = H.detach().cpu().numpy()
    H = np.asarray(H, dtype=np.float64)
    if H.ndim < 2:
        raise ValueError(f"{name} must have at least 2 dimensions, got {H.ndim}.")
    if H.ndim > 2:
        H = H.reshape(H.shape[0], -1)
    if not np.all(np.isfinite(H)):
        raise ValueError(f"{name} contains NaN or inf.")
    return H


def _center(H: np.ndarray) -> np.ndarray:
    return H - H.mean(axis=0, keepdims=True)


def _pair(H1, H2):
    X = _as_matrix(H1, "H1")
    Y = _as_matrix(H2, "H2")
    if X.shape[0] != Y.shape[0]:
        raise ValueError(
            "Representations must contain the same examples "
            f"(got {X.shape[0]} and {Y.shape[0]} rows)."
        )
    return X, Y


# --------------------------------------------------------------------------
# rank metrics (single representation)
# --------------------------------------------------------------------------
def _singular_values(H, center: bool) -> np.ndarray:
    H = _as_matrix(H)
    if center:
        H = _center(H)
    return np.linalg.svd(H, compute_uv=False)


def effective_rank(H, center: bool = True, eps: float = _EPS) -> float:
    """Entropy-based effective rank (Roy & Vetterli, 2007).

    ``exp(-sum p_i log p_i)`` with ``p_i = s_i / sum(s)`` and ``s`` the
    singular values. Lies in ``[1, min(n, d)]`` (``0.0`` for an all-constant
    representation). A matrix with ``k`` equal non-zero singular values has
    effective rank exactly ``k``.

    ``center=True`` removes the per-feature mean first, so the metric measures
    variation across examples, not the shared offset.
    """
    s = _singular_values(H, center)
    total = s.sum()
    if total <= eps:
        return 0.0
    p = s / total
    p = p[p > 0]
    return float(np.exp(-(p * np.log(p)).sum()))


def stable_rank(H, center: bool = True, eps: float = _EPS) -> float:
    """Stable rank: ``||H||_F^2 / ||H||_2^2`` (sum of squared singular values
    over the largest squared singular value). Never exceeds the true rank and
    is dominated by the top direction, so it is less sensitive to a long tail
    of small singular values than the effective rank. ``0.0`` if constant.
    """
    s = _singular_values(H, center)
    if s.size == 0 or s[0] <= eps:
        return 0.0
    s2 = s**2
    return float(s2.sum() / s2[0])


# --------------------------------------------------------------------------
# CKA
# --------------------------------------------------------------------------
def _unit_centered(H: np.ndarray, eps: float):
    X = _center(H)
    norm = np.linalg.norm(X)
    if norm <= eps:
        return None
    return X / norm


def linear_cka(H1, H2, eps: float = _EPS) -> float:
    """Linear CKA (Kornblith et al., 2019). Range [0, 1]; ``nan`` if either
    representation is constant.

    Invariant to orthogonal transforms, isotropic scaling and translation of
    either representation; NOT invariant to general invertible linear maps.

    Note: this (biased) estimator is inflated when ``n_examples`` is not much
    larger than the feature dimension. Use ``debiased_linear_cka`` in that
    regime, or report both.
    """
    X, Y = _pair(H1, H2)
    X = _unit_centered(X, eps)
    Y = _unit_centered(Y, eps)
    if X is None or Y is None:
        return float("nan")

    n = X.shape[0]
    if n < max(X.shape[1], Y.shape[1]):
        # Gram form: <X^T Y, X^T Y>_F = <XX^T, YY^T>_F, cheaper when n < d.
        Kx, Ky = X @ X.T, Y @ Y.T
        num = np.sum(Kx * Ky)
        den = np.linalg.norm(Kx) * np.linalg.norm(Ky)
    else:
        num = np.sum((X.T @ Y) ** 2)
        den = np.linalg.norm(X.T @ X) * np.linalg.norm(Y.T @ Y)
    if den <= 0:
        return float("nan")
    return float(num / den)


def _unbiased_hsic(K: np.ndarray, L: np.ndarray) -> float:
    """Unbiased HSIC estimator (Song et al., 2012); needs n >= 4."""
    n = K.shape[0]
    Kt, Lt = K.copy(), L.copy()
    np.fill_diagonal(Kt, 0.0)
    np.fill_diagonal(Lt, 0.0)
    term1 = np.sum(Kt * Lt)
    term2 = Kt.sum() * Lt.sum() / ((n - 1) * (n - 2))
    term3 = 2.0 / (n - 2) * (Kt.sum(axis=1) @ Lt.sum(axis=1))
    return float((term1 + term2 - term3) / (n * (n - 3)))


def debiased_linear_cka(H1, H2, eps: float = _EPS) -> float:
    """Debiased linear CKA (unbiased HSIC; Song et al. 2012, as used by
    Nguyen et al. 2021). Close to 0 for independent representations even when
    ``n_examples < n_features``. Can be slightly negative or above 1 from
    estimator noise; ``nan`` if either self-HSIC is non-positive.
    Requires at least 4 examples.
    """
    X, Y = _pair(H1, H2)
    if X.shape[0] < 4:
        raise ValueError("debiased_linear_cka needs at least 4 examples.")
    X = _unit_centered(X, eps)
    Y = _unit_centered(Y, eps)
    if X is None or Y is None:
        return float("nan")

    K, L = X @ X.T, Y @ Y.T
    hxy, hxx, hyy = _unbiased_hsic(K, L), _unbiased_hsic(K, K), _unbiased_hsic(L, L)
    if hxx <= 0 or hyy <= 0:
        return float("nan")
    return float(hxy / np.sqrt(hxx * hyy))


# --------------------------------------------------------------------------
# RSA
# --------------------------------------------------------------------------
_DISTANCES = ("euclidean", "cosine", "correlation")


def rdm(H, distance: str = "euclidean") -> np.ndarray:
    """Representational dissimilarity matrix in condensed form.

    Returns the strict upper triangle (row-major, the same order as
    ``np.triu_indices(n, k=1)``) of the ``n x n`` matrix of pairwise
    distances between examples, as a 1D array of length ``n(n-1)/2``.
    ``distance`` is one of ``euclidean``, ``cosine``, ``correlation``.
    """
    if distance not in _DISTANCES:
        raise ValueError(f"distance must be one of {_DISTANCES}, got {distance!r}.")
    X = _as_matrix(H)
    d = pdist(X, metric=distance)
    if not np.all(np.isfinite(d)):
        raise ValueError(
            "Non-finite distances (zero-norm or constant examples with "
            f"distance={distance!r})."
        )
    return d


def rsa(H1, H2, distance: str = "euclidean", method: str = "spearman") -> float:
    """Representational similarity analysis score between two representations.

    Correlates the two RDMs over example pairs. ``method='spearman'`` (default,
    the standard choice: only the ordering of distances matters) or
    ``'pearson'``. ``nan`` if one RDM is constant. Needs at least 3 examples.
    """
    if method not in ("spearman", "pearson"):
        raise ValueError(f"method must be 'spearman' or 'pearson', got {method!r}.")
    X, Y = _pair(H1, H2)
    if X.shape[0] < 3:
        raise ValueError("rsa needs at least 3 examples.")

    d1, d2 = rdm(X, distance), rdm(Y, distance)
    if method == "spearman":
        d1, d2 = rankdata(d1), rankdata(d2)
    d1, d2 = d1 - d1.mean(), d2 - d2.mean()
    n1, n2 = np.linalg.norm(d1), np.linalg.norm(d2)
    if n1 == 0.0 or n2 == 0.0:
        return float("nan")
    return float(d1 @ d2 / (n1 * n2))


# --------------------------------------------------------------------------
# Procrustes
# --------------------------------------------------------------------------
def procrustes_distance(H1, H2, kind: str = "frobenius", eps: float = _EPS) -> float:
    """Orthogonal Procrustes distance between two representations.

    Both are centered and scaled to unit Frobenius norm, then the best
    orthogonal map (rotation or reflection) is fitted. This makes the distance
    symmetric and invariant to translation, isotropic scaling and orthogonal
    transforms. Feature dimensions may differ (the smaller one is zero-padded).

    ``kind='frobenius'``: ``min_R ||X R - Y||_F``, in ``[0, sqrt(2)]``.
    ``kind='angular'``: the angular (true-metric) version
    ``arccos(nuclear_norm(X^T Y))`` in ``[0, pi/2]`` (Williams et al., 2021).
    """
    if kind not in ("frobenius", "angular"):
        raise ValueError(f"kind must be 'frobenius' or 'angular', got {kind!r}.")
    X, Y = _pair(H1, H2)
    if X.shape[0] < 2:
        raise ValueError("At least two examples are required.")
    X = _unit_centered(X, eps)
    Y = _unit_centered(Y, eps)
    if X is None or Y is None:
        raise ValueError("A representation has zero or near-zero centered norm.")

    d = max(X.shape[1], Y.shape[1])
    Xp = np.pad(X, ((0, 0), (0, d - X.shape[1])))
    Yp = np.pad(Y, ((0, 0), (0, d - Y.shape[1])))

    U, _, Vt = np.linalg.svd(Xp.T @ Yp)
    R = U @ Vt
    r = float(np.linalg.norm(Xp @ R - Yp))  # explicit residual: accurate near 0
    if kind == "frobenius":
        return r
    return float(2.0 * np.arcsin(min(r / 2.0, 1.0)))


# --------------------------------------------------------------------------
# linear probe
# --------------------------------------------------------------------------
def linear_probe(
    H_train,
    y_train,
    H_val,
    y_val,
    H_test=None,
    y_test=None,
    Cs=(1e-3, 1e-2, 1e-1, 1.0, 10.0),
    seed: int = 0,
    max_iter: int = 2000,
) -> dict:
    """Logistic-regression probe on frozen features.

    Features are standardized with statistics from the *training* split only.
    The regularization strength ``C`` is chosen from ``Cs`` by validation
    log-loss; the test split (optional) is only scored once, with the chosen
    model. Majority-class baselines (class chosen on train) are returned so a
    probe score can always be read against chance.

    Returns a dict with keys: ``C``, ``val_accuracy``, ``val_log_loss``,
    ``val_majority_accuracy``, ``test_accuracy``, ``test_log_loss``,
    ``test_majority_accuracy`` (the test entries are ``None`` if no test split
    is given).
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, log_loss
    from sklearn.preprocessing import StandardScaler

    Xtr, Xva = _as_matrix(H_train, "H_train"), _as_matrix(H_val, "H_val")
    ytr, yva = np.asarray(y_train), np.asarray(y_val)
    if len(ytr) != len(Xtr) or len(yva) != len(Xva):
        raise ValueError("Features and labels must have the same number of rows.")
    if (H_test is None) != (y_test is None):
        raise ValueError("Provide both H_test and y_test, or neither.")

    scaler = StandardScaler().fit(Xtr)
    Xtr_s, Xva_s = scaler.transform(Xtr), scaler.transform(Xva)

    best = None
    for C in Cs:
        model = LogisticRegression(C=C, max_iter=max_iter, random_state=seed)
        model.fit(Xtr_s, ytr)
        loss = log_loss(yva, model.predict_proba(Xva_s), labels=model.classes_)
        if best is None or loss < best[0]:
            best = (loss, C, model)
    val_loss, best_C, model = best

    values, counts = np.unique(ytr, return_counts=True)
    majority = values[counts.argmax()]

    out = {
        "C": best_C,
        "val_accuracy": float(accuracy_score(yva, model.predict(Xva_s))),
        "val_log_loss": float(val_loss),
        "val_majority_accuracy": float(np.mean(yva == majority)),
        "test_accuracy": None,
        "test_log_loss": None,
        "test_majority_accuracy": None,
    }
    if H_test is not None:
        Xte = _as_matrix(H_test, "H_test")
        yte = np.asarray(y_test)
        if len(yte) != len(Xte):
            raise ValueError("H_test and y_test must have the same number of rows.")
        Xte_s = scaler.transform(Xte)
        out["test_accuracy"] = float(accuracy_score(yte, model.predict(Xte_s)))
        out["test_log_loss"] = float(
            log_loss(yte, model.predict_proba(Xte_s), labels=model.classes_)
        )
        out["test_majority_accuracy"] = float(np.mean(yte == majority))
    return out