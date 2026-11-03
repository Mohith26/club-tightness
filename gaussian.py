"""The correlated-Gaussian benchmark from the CLUB paper (Section 4.1).

x, y in R^d, each coordinate pair (x_i, y_i) is bivariate normal with
correlation rho, independent across coordinates.
"""
import math
import torch


def mi_to_rho(mi, dim):
    return math.sqrt(1 - math.exp(-2 * mi / dim))


def true_mi(rho, dim):
    return -dim / 2 * math.log(1 - rho ** 2)


def oracle_club(rho, dim):
    """CLUB evaluated with the exact conditional p(y|x) = N(rho x, 1 - rho^2).

    E_p(x,y)[log p(y|x)] - E_p(x)p(y)[log p(y|x)]
      = sum_i [E(y - rho x)^2 under independence - under the joint] / (2 (1 - rho^2))
      = dim * ((1 + rho^2) - (1 - rho^2)) / (2 (1 - rho^2))
      = dim * rho^2 / (1 - rho^2)
    """
    return dim * rho ** 2 / (1 - rho ** 2)


def sample(rho, dim, n, generator=None):
    x = torch.randn(n, dim, generator=generator)
    y = rho * x + math.sqrt(1 - rho ** 2) * torch.randn(n, dim, generator=generator)
    return x, y
