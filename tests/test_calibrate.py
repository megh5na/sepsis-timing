import numpy as np

from src.eval.calibrate import apply_platt, fit_platt, to_logit


def make(n=60000, K=6, a_true=0.5, b_true=-0.3, seed=0):
    rng = np.random.default_rng(seed)
    z = np.sort(rng.normal(-2.5, 1.5, (n, K)), axis=1)           # model logits, increasing in k
    p_true = 1 / (1 + np.exp(-(a_true * z + b_true)))            # truth is a squashed, shifted version
    y = (rng.random((n, K)) < p_true).astype(float)
    return 1 / (1 + np.exp(-z)), y, p_true


def test_shared_platt_recovers_the_true_miscalibration():
    F, y, p_true = make()
    par = fit_platt(F, y, np.ones_like(y), per_horizon=False)
    assert abs(par["a"][0] - 0.5) < 0.05 and abs(par["b"][0] + 0.3) < 0.1
    G = apply_platt(F, par)
    assert np.abs(G - p_true).mean() < np.abs(F - p_true).mean() / 3


def test_both_variants_keep_f_monotone_in_k():
    F, y, _ = make(K=8)
    for ph in (False, True):
        G = apply_platt(F, fit_platt(F, y, np.ones_like(y), per_horizon=ph))
        assert (G[:, 1:] >= G[:, :-1] - 1e-12).all()


def test_masked_terms_do_not_influence_the_fit():
    F, y, _ = make(n=20000)
    w = np.ones_like(y); w[:10000] = 0
    y_bad = y.copy(); y_bad[:10000] = 1 - y_bad[:10000]         # corrupt only the masked half
    a = fit_platt(F, y, w, False); b = fit_platt(F, y_bad, w, False)
    assert np.allclose(a["a"], b["a"]) and np.allclose(a["b"], b["b"])


def test_slope_is_positive_even_for_adversarial_labels():
    F, y, _ = make(n=5000)
    par = fit_platt(F, 1 - y, np.ones_like(y), per_horizon=True)
    assert (par["a"] > 0).all()
