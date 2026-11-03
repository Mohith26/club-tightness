import math
import torch
from club import CLUB, club_with_conditional
from gaussian import mi_to_rho, true_mi, oracle_club, sample


def test_rho_mi_roundtrip():
    for mi in [2.0, 6.0, 10.0]:
        assert abs(true_mi(mi_to_rho(mi, 20), 20) - mi) < 1e-9


def test_oracle_formula_matches_monte_carlo():
    g = torch.Generator().manual_seed(0)
    for mi in [2.0, 10.0]:
        rho = mi_to_rho(mi, 20)
        vals = []
        for _ in range(20):
            x, y = sample(rho, 20, 1024, g)
            vals.append(club_with_conditional(rho * x, 1 - rho ** 2, y).item())
        mc = sum(vals) / len(vals)
        assert abs(mc - oracle_club(rho, 20)) / oracle_club(rho, 20) < 0.02


def test_oracle_is_an_upper_bound():
    for mi in [0.5, 2.0, 6.0, 10.0, 20.0]:
        rho = mi_to_rho(mi, 20)
        assert oracle_club(rho, 20) >= true_mi(rho, 20)


def test_forward_matches_conditional_formula():
    torch.manual_seed(0)
    m = CLUB(5, 5, 16)
    x, y = torch.randn(32, 5), torch.randn(32, 5)
    mu, logvar = m.get_mu_logvar(x)
    assert torch.allclose(m(x, y), club_with_conditional(mu, logvar.exp(), y), atol=1e-5)


def test_sample_correlation():
    g = torch.Generator().manual_seed(1)
    rho = 0.7
    x, y = sample(rho, 4, 200000, g)
    r = (x * y).mean().item()
    assert abs(r - rho) < 0.01
