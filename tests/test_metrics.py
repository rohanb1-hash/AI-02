"""Tests for src/metrics.py. All data is generated from fixed seeds."""

import numpy as np
import pytest
from scipy.stats import pearsonr, spearmanr

from src.metrics import (
    debiased_linear_cka,
    effective_rank,
    linear_cka,
    linear_probe,
    procrustes_distance,
    rdm,
    rsa,
    stable_rank,
)


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def make_with_singular_values(s, n=60, d=None, seed=0):
    """Centered (n, d) matrix whose singular values are exactly ``s``."""
    rng = np.random.default_rng(seed)
    k = len(s)
    d = d or k
    Q = rng.normal(size=(n, k))
    Q = Q - Q.mean(axis=0, keepdims=True)  # centered columns ...
    Q, _ = np.linalg.qr(Q)  # ... stay centered after orthonormalizing
    V, _ = np.linalg.qr(rng.normal(size=(d, k)))
    return Q @ np.diag(s) @ V.T


def random_orthogonal(d, seed=0):
    rng = np.random.default_rng(seed)
    Q, _ = np.linalg.qr(rng.normal(size=(d, d)))
    return Q


def reference_linear_cka(X, Y):
    """Direct feature-space formula (the one from the learning notebook)."""
    X = X - X.mean(0)
    Y = Y - Y.mean(0)
    num = np.linalg.norm(X.T @ Y) ** 2
    den = np.linalg.norm(X.T @ X) * np.linalg.norm(Y.T @ Y)
    return num / den


# --------------------------------------------------------------------------
# effective rank / stable rank
# --------------------------------------------------------------------------
@pytest.mark.parametrize("k", [1, 3, 8])
def test_equal_singular_values_give_rank_k(k):
    H = make_with_singular_values(np.full(k, 5.0), n=60, d=12)
    assert effective_rank(H) == pytest.approx(k, abs=1e-8)
    assert stable_rank(H) == pytest.approx(k, abs=1e-8)


def test_uneven_singular_values_match_hand_computation():
    s = np.array([10.0, 2.0, 1.0])
    H = make_with_singular_values(s)
    p = s / s.sum()
    expected_eff = np.exp(-(p * np.log(p)).sum())
    expected_stable = (s**2).sum() / s[0] ** 2  # 1.05
    assert effective_rank(H) == pytest.approx(expected_eff, rel=1e-8)
    assert stable_rank(H) == pytest.approx(expected_stable, rel=1e-8)


def test_one_dominant_direction_is_rank_one():
    H = make_with_singular_values(np.array([10.0, 1e-9, 1e-9]))
    assert effective_rank(H) == pytest.approx(1.0, abs=1e-6)
    assert stable_rank(H) == pytest.approx(1.0, abs=1e-6)


@pytest.mark.parametrize("fn", [effective_rank, stable_rank])
def test_rank_scale_and_translation_invariance(fn):
    rng = np.random.default_rng(1)
    H = rng.normal(size=(200, 15)) @ rng.normal(size=(15, 15))
    assert fn(3.7 * H) == pytest.approx(fn(H), rel=1e-9)
    assert fn(H + 42.0) == pytest.approx(fn(H), rel=1e-9)


@pytest.mark.parametrize("fn", [effective_rank, stable_rank])
def test_constant_representation_has_zero_rank(fn):
    assert fn(np.full((50, 8), 3.0)) == 0.0


def test_rank_bounds_for_random_gaussian():
    rng = np.random.default_rng(2)
    H = rng.normal(size=(2000, 10))
    er, sr = effective_rank(H), stable_rank(H)
    assert 9.5 < er <= 10.0
    assert 1.0 <= sr <= 10.0
    assert sr <= er  # stable rank is the smaller of the two here


def test_stable_rank_never_exceeds_matrix_rank():
    rng = np.random.default_rng(3)
    H = rng.normal(size=(100, 6)) @ rng.normal(size=(6, 30))  # rank 6
    assert stable_rank(H) <= np.linalg.matrix_rank(H) + 1e-9


def test_rank_flattens_higher_dimensional_input():
    rng = np.random.default_rng(4)
    H4 = rng.normal(size=(80, 3, 4, 4))
    assert effective_rank(H4) == pytest.approx(effective_rank(H4.reshape(80, -1)))


def test_rank_rejects_bad_input():
    with pytest.raises(ValueError):
        effective_rank(np.ones(10))
    with pytest.raises(ValueError):
        effective_rank(np.array([[1.0, np.nan], [2.0, 3.0]]))


# --------------------------------------------------------------------------
# linear CKA
# --------------------------------------------------------------------------
def test_cka_self_similarity_is_one():
    X = np.random.default_rng(5).normal(size=(100, 20))
    assert linear_cka(X, X) == pytest.approx(1.0, abs=1e-12)


def test_cka_invariances():
    rng = np.random.default_rng(6)
    X = rng.normal(size=(100, 20))
    Q = random_orthogonal(20, seed=6)
    assert linear_cka(X, 3.0 * X) == pytest.approx(1.0, abs=1e-12)
    assert linear_cka(X, X @ Q) == pytest.approx(1.0, abs=1e-12)
    assert linear_cka(X, X + 7.0) == pytest.approx(1.0, abs=1e-12)


def test_cka_is_symmetric():
    rng = np.random.default_rng(7)
    X, Y = rng.normal(size=(100, 20)), rng.normal(size=(100, 12))
    assert linear_cka(X, Y) == pytest.approx(linear_cka(Y, X), abs=1e-12)


def test_cka_independent_representations_are_dissimilar():
    rng = np.random.default_rng(8)
    X, Y = rng.normal(size=(2000, 10)), rng.normal(size=(2000, 10))
    assert linear_cka(X, Y) < 0.05


def test_cka_detects_shared_structure():
    rng = np.random.default_rng(9)
    X = rng.normal(size=(500, 20))
    Y = X @ rng.normal(size=(20, 30)) + 0.05 * rng.normal(size=(500, 30))
    assert linear_cka(X, Y) > 0.5


def test_cka_not_invariant_to_anisotropic_scaling():
    rng = np.random.default_rng(10)
    X = rng.normal(size=(500, 20))
    Xs = X.copy()
    Xs[:, 0] *= 10.0
    assert linear_cka(X, Xs) < 0.5


@pytest.mark.parametrize("n,d1,d2", [(300, 10, 6), (40, 100, 80), (60, 30, 90)])
def test_cka_matches_reference_in_both_regimes(n, d1, d2):
    # (300,10,6): feature-space path; (40,100,80) and (60,30,90): Gram path
    rng = np.random.default_rng(11)
    X, Y = rng.normal(size=(n, d1)), rng.normal(size=(n, d2))
    Y[:, : min(d1, d2)] += X[:, : min(d1, d2)]  # add some real similarity
    assert linear_cka(X, Y) == pytest.approx(reference_linear_cka(X, Y), rel=1e-9)


def test_cka_tiny_scale_is_not_mistaken_for_constant():
    X = np.random.default_rng(12).normal(size=(100, 10))
    assert linear_cka(1e-9 * X, 1e-9 * X) == pytest.approx(1.0, abs=1e-12)


def test_cka_constant_input_returns_nan():
    X = np.random.default_rng(13).normal(size=(50, 5))
    assert np.isnan(linear_cka(X, np.ones((50, 5))))


def test_cka_requires_same_examples():
    with pytest.raises(ValueError):
        linear_cka(np.ones((10, 3)), np.ones((11, 3)))


# --------------------------------------------------------------------------
# debiased CKA
# --------------------------------------------------------------------------
def test_debiased_cka_self_similarity_is_one():
    X = np.random.default_rng(14).normal(size=(100, 20))
    assert debiased_linear_cka(X, X) == pytest.approx(1.0, abs=1e-9)


def test_debiased_cka_invariances():
    rng = np.random.default_rng(15)
    X = rng.normal(size=(100, 20))
    Q = random_orthogonal(20, seed=15)
    Y = X + rng.normal(size=(100, 20))
    assert debiased_linear_cka(X @ Q, 2.0 * Y + 3.0) == pytest.approx(
        debiased_linear_cka(X, Y), abs=1e-9
    )


def test_biased_cka_is_inflated_when_n_less_than_d_but_debiased_is_not():
    rng = np.random.default_rng(16)
    X, Y = rng.normal(size=(50, 200)), rng.normal(size=(50, 200))  # independent
    assert linear_cka(X, Y) > 0.5  # the artefact
    assert abs(debiased_linear_cka(X, Y)) < 0.15  # corrected


def test_debiased_cka_needs_four_examples():
    with pytest.raises(ValueError):
        debiased_linear_cka(np.ones((3, 2)), np.ones((3, 2)))


# --------------------------------------------------------------------------
# RSA
# --------------------------------------------------------------------------
@pytest.mark.parametrize("method", ["spearman", "pearson"])
def test_rsa_invariances(method):
    rng = np.random.default_rng(17)
    X = rng.normal(size=(100, 20))
    Q = random_orthogonal(20, seed=17)
    assert rsa(X, X, method=method) == pytest.approx(1.0, abs=1e-12)
    assert rsa(X, X @ Q, method=method) == pytest.approx(1.0, abs=1e-9)
    assert rsa(X, 3.0 * X + 7.0, method=method) == pytest.approx(1.0, abs=1e-9)


def test_rsa_symmetric_and_handles_different_feature_dims():
    rng = np.random.default_rng(18)
    X, Y = rng.normal(size=(100, 20)), rng.normal(size=(100, 12))
    assert rsa(X, Y) == pytest.approx(rsa(Y, X), abs=1e-12)


def test_rsa_independent_representations_are_near_zero():
    rng = np.random.default_rng(19)
    X, Y = rng.normal(size=(200, 20)), rng.normal(size=(200, 20))
    assert abs(rsa(X, Y)) < 0.1


def test_rsa_matches_scipy_correlations():
    rng = np.random.default_rng(20)
    X = rng.normal(size=(60, 8))
    Y = X[:, :5] + 0.5 * rng.normal(size=(60, 5))
    d1, d2 = rdm(X), rdm(Y)
    assert rsa(X, Y, method="spearman") == pytest.approx(spearmanr(d1, d2)[0], abs=1e-12)
    assert rsa(X, Y, method="pearson") == pytest.approx(pearsonr(d1, d2)[0], abs=1e-12)


def test_rdm_order_matches_upper_triangle():
    from scipy.spatial.distance import cdist

    X = np.random.default_rng(21).normal(size=(12, 4))
    full = cdist(X, X)
    iu = np.triu_indices(12, k=1)
    np.testing.assert_allclose(rdm(X), full[iu], atol=1e-12)
    assert rdm(X).shape == (12 * 11 // 2,)


@pytest.mark.parametrize("distance", ["euclidean", "cosine", "correlation"])
def test_rsa_works_for_each_distance(distance):
    X = np.random.default_rng(22).normal(size=(50, 10))
    assert rsa(X, X, distance=distance) == pytest.approx(1.0, abs=1e-12)


def test_rsa_rejects_bad_arguments():
    X = np.random.default_rng(23).normal(size=(10, 3))
    with pytest.raises(ValueError):
        rsa(X, X, method="kendall")
    with pytest.raises(ValueError):
        rsa(X, X, distance="manhattan")
    with pytest.raises(ValueError):
        rsa(X[:2], X[:2])
    with pytest.raises(ValueError):
        rsa(X, X[:9])


def test_rsa_constant_rdm_returns_nan():
    rng = np.random.default_rng(24)
    X = rng.normal(size=(5, 3))
    same_points = np.ones((5, 3))  # all distances are 0
    assert np.isnan(rsa(X, same_points))


# --------------------------------------------------------------------------
# Procrustes
# --------------------------------------------------------------------------
def test_procrustes_identical_is_zero():
    X = np.random.default_rng(25).normal(size=(200, 16))
    assert procrustes_distance(X, X) == pytest.approx(0.0, abs=1e-10)
    assert procrustes_distance(X, X, kind="angular") == pytest.approx(0.0, abs=1e-10)


def test_procrustes_invariances():
    rng = np.random.default_rng(26)
    X = rng.normal(size=(200, 16))
    Y = X @ random_orthogonal(16, seed=26) + 5.0
    assert procrustes_distance(X, Y) == pytest.approx(0.0, abs=1e-10)
    assert procrustes_distance(X, 2.0 * X) == pytest.approx(0.0, abs=1e-10)  # scale


def test_procrustes_is_symmetric():
    rng = np.random.default_rng(27)
    X = rng.normal(size=(150, 10))
    Y = X.copy()
    Y[:, 0] *= 4.0
    Y += 0.1 * rng.normal(size=Y.shape)
    assert procrustes_distance(X, Y) == pytest.approx(procrustes_distance(Y, X), abs=1e-10)


def test_procrustes_detects_deformation_and_independence():
    rng = np.random.default_rng(28)
    X = rng.normal(size=(200, 16))
    Y = X.copy()
    Y[:, 0] *= 4.0
    assert procrustes_distance(X, Y) > 1e-3
    indep = procrustes_distance(X, rng.normal(size=(200, 16)))
    assert indep > 1.0
    assert indep <= np.sqrt(2) + 1e-9


def test_procrustes_angular_is_consistent_with_frobenius():
    rng = np.random.default_rng(29)
    X = rng.normal(size=(200, 12))
    Y = X + 0.5 * rng.normal(size=X.shape)
    r = procrustes_distance(X, Y)
    a = procrustes_distance(X, Y, kind="angular")
    assert np.cos(a) == pytest.approx(1 - r**2 / 2, abs=1e-12)
    assert 0.0 <= a <= np.pi / 2


def test_procrustes_different_feature_dims():
    rng = np.random.default_rng(30)
    X = rng.normal(size=(100, 8))
    Y = np.hstack([X @ random_orthogonal(8, seed=30), np.zeros((100, 5))])
    assert procrustes_distance(X, Y) == pytest.approx(0.0, abs=1e-10)
    assert procrustes_distance(Y, X) == pytest.approx(0.0, abs=1e-10)


def test_procrustes_rejects_bad_input():
    X = np.random.default_rng(31).normal(size=(10, 3))
    with pytest.raises(ValueError):
        procrustes_distance(X, X[:9])
    with pytest.raises(ValueError):
        procrustes_distance(X, np.ones((10, 3)))
    with pytest.raises(ValueError):
        procrustes_distance(X, X, kind="cosine")


# --------------------------------------------------------------------------
# linear probe
# --------------------------------------------------------------------------
@pytest.fixture(scope="module")
def probe_data():
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split

    X, y = make_classification(
        n_samples=2000, n_features=20, n_informative=10, n_redundant=5,
        n_classes=2, class_sep=1.0, flip_y=0.05, random_state=42,
    )
    Xtr, Xtmp, ytr, ytmp = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)
    Xva, Xte, yva, yte = train_test_split(Xtmp, ytmp, test_size=0.5, stratify=ytmp, random_state=42)
    return Xtr, ytr, Xva, yva, Xte, yte


def test_probe_beats_majority_baseline(probe_data):
    Xtr, ytr, Xva, yva, Xte, yte = probe_data
    out = linear_probe(Xtr, ytr, Xva, yva, Xte, yte)
    assert out["test_accuracy"] > out["test_majority_accuracy"] + 0.15
    assert out["val_accuracy"] > out["val_majority_accuracy"] + 0.15
    assert out["test_log_loss"] < np.log(2)  # better than predicting 50/50


def test_probe_at_chance_with_permuted_labels(probe_data):
    # sanity check from the plan: label permutation must destroy the signal
    Xtr, ytr, Xva, yva, Xte, yte = probe_data
    rng = np.random.default_rng(0)
    out = linear_probe(Xtr, rng.permutation(ytr), Xva, yva, Xte, yte)
    assert abs(out["test_accuracy"] - 0.5) < 0.1


def test_probe_chooses_C_from_grid_and_is_deterministic(probe_data):
    Xtr, ytr, Xva, yva, Xte, yte = probe_data
    Cs = (0.01, 0.1, 1.0)
    a = linear_probe(Xtr, ytr, Xva, yva, Xte, yte, Cs=Cs, seed=3)
    b = linear_probe(Xtr, ytr, Xva, yva, Xte, yte, Cs=Cs, seed=3)
    assert a["C"] in Cs
    assert a == b


def test_probe_test_split_is_optional(probe_data):
    Xtr, ytr, Xva, yva, _, _ = probe_data
    out = linear_probe(Xtr, ytr, Xva, yva)
    assert out["test_accuracy"] is None and out["test_log_loss"] is None


def test_probe_scaler_uses_only_training_statistics(probe_data):
    # shifting the val/test features by a constant must change the result,
    # i.e. val/test are NOT being re-standardized with their own statistics
    Xtr, ytr, Xva, yva, Xte, yte = probe_data
    base = linear_probe(Xtr, ytr, Xva, yva, Xte, yte, Cs=(1.0,))  # fixed C: same model
    shifted = linear_probe(Xtr, ytr, Xva + 5.0, yva, Xte + 5.0, yte, Cs=(1.0,))
    assert abs(shifted["val_log_loss"] - base["val_log_loss"]) > 1e-3
    assert abs(shifted["test_log_loss"] - base["test_log_loss"]) > 1e-3


def test_probe_rejects_mismatched_inputs(probe_data):
    Xtr, ytr, Xva, yva, Xte, yte = probe_data
    with pytest.raises(ValueError):
        linear_probe(Xtr, ytr[:-1], Xva, yva)
    with pytest.raises(ValueError):
        linear_probe(Xtr, ytr, Xva, yva, H_test=Xte)  # y_test missing


# --------------------------------------------------------------------------
# torch interop (skipped when torch is not installed)
# --------------------------------------------------------------------------
def test_torch_tensors_give_same_result_as_numpy():
    torch = pytest.importorskip("torch")
    rng = np.random.default_rng(32)
    X, Y = rng.normal(size=(100, 20)), rng.normal(size=(100, 12))
    tX, tY = torch.tensor(X, dtype=torch.float64), torch.tensor(Y, dtype=torch.float64)
    assert effective_rank(tX) == pytest.approx(effective_rank(X))
    assert linear_cka(tX, tY) == pytest.approx(linear_cka(X, Y))
    assert rsa(tX, tY) == pytest.approx(rsa(X, Y))
    assert procrustes_distance(tX, tX) == pytest.approx(0.0, abs=1e-10)