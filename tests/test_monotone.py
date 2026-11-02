import inspect

import numpy as np
import pytest
import torch

from src.data.preprocess import build_multi_horizon_targets
from src.models import losses
from src.models.heads import IndependentHead, MonotoneHead
from src.models.model import SepsisModel


class TestMonotoneHead:
    @pytest.mark.parametrize("K", [1, 2, 6, 12, 18])
    def test_non_decreasing_on_random_input(self, K):
        torch.manual_seed(0)
        head = MonotoneHead(64, K)
        for scale in (0.1, 1.0, 10.0, 100.0):
            h = torch.randn(8, 30, 64) * scale
            F = head(h)
            assert F.shape == (8, 30, K)
            assert (F[..., 1:] >= F[..., :-1]).all()

    def test_non_decreasing_even_for_adversarial_weights(self):
        # make the unconstrained part wildly decreasing; the head must still be monotone
        head = MonotoneHead(64, 12)
        with torch.no_grad():
            head.linear.weight.copy_(torch.randn_like(head.linear.weight) * 5)
            head.linear.bias.copy_(torch.linspace(5, -5, 12))
        F = head(torch.randn(16, 20, 64) * 3)
        assert (F[..., 1:] >= F[..., :-1]).all()

    def test_probabilities_in_unit_interval(self):
        F = MonotoneHead(64, 12)(torch.randn(4, 10, 64) * 5)
        assert ((F >= 0) & (F <= 1)).all()

    def test_independent_head_can_violate_monotonicity(self):
        torch.manual_seed(0)
        F = IndependentHead(64, 12)(torch.randn(8, 30, 64))
        assert (F[..., 1:] < F[..., :-1]).any(), "A1 is expected to show coherence violations"

    def test_shapes_through_full_model(self):
        for head in ("monotone", "independent"):
            m = SepsisModel(head, K=12)
            F, logits = m(torch.randn(3, 7, 120))
            assert F.shape == logits.shape == (3, 7, 12)
            assert torch.allclose(F, torch.sigmoid(logits))

    def test_causal_model_ignores_future(self):
        m = SepsisModel("monotone", K=12).eval()
        Z = torch.randn(1, 20, 120)
        Z2 = Z.clone()
        Z2[:, 10:] = torch.randn(1, 10, 120) * 50
        with torch.no_grad():
            a, _ = m(Z)
            b, _ = m(Z2)
        assert torch.allclose(a[:, :10], b[:, :10], atol=1e-6)

    def test_encoder_init_identical_across_heads_for_same_seed(self):
        a = SepsisModel("monotone", 12, seed=7).encoder.state_dict()
        b = SepsisModel("independent", 12, seed=7).encoder.state_dict()
        c = SepsisModel("monotone", 12, seed=8).encoder.state_dict()
        assert all(torch.equal(a[k], b[k]) for k in a)
        assert not all(torch.equal(a[k], c[k]) for k in a)


class TestOnsetDistribution:
    def test_p_is_a_valid_distribution(self):
        from src.decision.stopping import onset_distribution
        F = MonotoneHead(64, 12)(torch.randn(5, 9, 64) * 3).detach().numpy()
        p, p_none = onset_distribution(F)
        assert (p >= 0).all() and (p_none >= 0).all()
        assert np.allclose(p.sum(-1) + p_none, 1.0, atol=1e-6)
        assert np.allclose(np.cumsum(p, -1), F, atol=1e-6)


class TestTargetsAndCensoring:
    def test_septic_targets_follow_spec(self):
        T, K = 30, 12
        y, m = build_multi_horizon_targets(T, 10, K)
        assert (m == 1).all(), "septic patients need no censoring mask"
        for t in range(T):
            for k in range(1, K + 1):
                assert y[t, k - 1] == (1.0 if 10 <= t + k else 0.0)

    def test_nonseptic_targets_all_zero(self):
        y, _ = build_multi_horizon_targets(20, None, 12)
        assert (y == 0).all()

    def test_censoring_boundary_is_zero_indexed(self):
        T, K = 20, 12
        _, m = build_multi_horizon_targets(T, None, K)
        for t in range(T):
            for k in range(1, K + 1):
                assert m[t, k - 1] == (0.0 if t + k >= T else 1.0)
        assert m[T - 1, 0] == 0.0       # last row, k=1 looks at unobserved hour T
        assert m[T - 2, 0] == 1.0       # second to last row, k=1 looks at observed hour T-1

    def test_short_stays_have_fully_masked_tails(self):
        _, m = build_multi_horizon_targets(5, None, 12)
        assert m[0].sum() == 4          # only k=1..4 stay inside rows 0..4


def _batch(T_list, K=12, seed=0):
    g = torch.Generator().manual_seed(seed)
    B, Tm = len(T_list), max(T_list)
    logits = torch.randn(B, Tm, K, generator=g)
    y = torch.zeros(B, Tm, K)
    mk = torch.zeros(B, Tm, K)
    lm = torch.zeros(B, Tm)
    for b, T in enumerate(T_list):
        yy, mm = build_multi_horizon_targets(T, 6 if b % 2 == 0 else None, K)
        y[b, :T], mk[b, :T], lm[b, :T] = torch.tensor(yy), torch.tensor(mm), 1.0
    return logits, y, mk, lm


class TestLoss:
    def test_matches_manual_bce(self):
        logits, y, mk, lm = _batch([9, 14, 6])
        F = torch.sigmoid(logits)
        manual = -(y * torch.log(F) + (1 - y) * torch.log(1 - F))
        w = mk * lm.unsqueeze(-1)
        expected = (manual * w).sum() / w.sum()
        assert torch.allclose(losses.masked_multihorizon_bce(logits, y, mk, lm), expected, atol=1e-5)

    def test_padding_invariance(self):
        lens = [9, 14, 6]
        logits, y, mk, lm = _batch(lens)
        padded = losses.masked_multihorizon_bce(logits, y, mk, lm)
        num = den = 0.0
        for b, T in enumerate(lens):
            w = mk[b:b + 1, :T]
            bce = torch.nn.functional.binary_cross_entropy_with_logits(logits[b:b + 1, :T], y[b:b + 1, :T], reduction="none")
            num += (bce * w).sum().item(); den += w.sum().item()
        assert padded.item() == pytest.approx(num / den, rel=1e-5)

    def test_padded_positions_get_no_gradient(self):
        logits, y, mk, lm = _batch([5, 12])
        logits = logits.clone().requires_grad_(True)
        losses.masked_multihorizon_bce(logits, y, mk, lm).backward()
        assert (logits.grad[0, 5:] == 0).all()

    def test_censored_terms_contribute_nothing(self):
        logits, y, mk, lm = _batch([8])
        l1 = losses.masked_multihorizon_bce(logits, y, mk, lm)
        logits2 = logits.clone()
        logits2[mk == 0] += 100.0                    # change only masked entries
        assert losses.masked_multihorizon_bce(logits2, y, mk, lm).item() == pytest.approx(l1.item())

    def test_normalised_by_active_entries_not_batch_size(self):
        logits, y, mk, lm = _batch([10])
        single = losses.masked_multihorizon_bce(logits, y, mk, lm)
        doubled = losses.masked_multihorizon_bce(torch.cat([logits, logits]), torch.cat([y, y]),
                                                 torch.cat([mk, mk]), torch.cat([lm, lm]))
        assert doubled.item() == pytest.approx(single.item(), rel=1e-6)

    def test_train_mask_restricts_hours(self):
        logits, y, mk, lm = _batch([12])
        tm = torch.zeros(1, 12); tm[:, :4] = 1
        got = losses.masked_multihorizon_bce(logits, y, mk, lm, tm)
        ref = losses.masked_multihorizon_bce(logits[:, :4], y[:, :4], mk[:, :4], lm[:, :4])
        assert got.item() == pytest.approx(ref.item(), rel=1e-6)

    def test_loss_module_does_not_touch_the_utility_function(self):
        src = inspect.getsource(losses)
        assert "import" in src
        code = [l for l in src.splitlines() if l.strip().startswith(("import", "from"))]
        assert not any("utility" in l or "decision" in l or "third_party" in l for l in code)
