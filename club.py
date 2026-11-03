"""CLUB and CLUBSample estimators (Cheng et al., ICML 2020), written to match
the formulas in https://github.com/Linear95/CLUB/blob/master/mi_estimators.py.

hidden_size follows the upstream convention: each of the two heads (mean and
log-variance) gets a single hidden layer of hidden_size // 2 ReLU units.
hidden_size=None gives a linear q(y|x) instead.
"""
import torch
import torch.nn as nn


def _head(x_dim, y_dim, hidden_size, tanh=False):
    if hidden_size is None:
        layers = [nn.Linear(x_dim, y_dim)]
    else:
        layers = [nn.Linear(x_dim, hidden_size // 2), nn.ReLU(), nn.Linear(hidden_size // 2, y_dim)]
    if tanh:
        layers.append(nn.Tanh())
    return nn.Sequential(*layers)


class CLUB(nn.Module):
    def __init__(self, x_dim, y_dim, hidden_size=15):
        super().__init__()
        self.p_mu = _head(x_dim, y_dim, hidden_size)
        # upstream bounds log-variance to [-1, 1] with a Tanh
        self.p_logvar = _head(x_dim, y_dim, hidden_size, tanh=True)

    def get_mu_logvar(self, x):
        return self.p_mu(x), self.p_logvar(x)

    def forward(self, x, y):
        mu, logvar = self.get_mu_logvar(x)
        positive = -((mu - y) ** 2) / 2.0 / logvar.exp()
        # all n*n pairs: mean over the y's in the batch for each x
        negative = -((y.unsqueeze(0) - mu.unsqueeze(1)) ** 2).mean(dim=1) / 2.0 / logvar.exp()
        return (positive.sum(-1) - negative.sum(-1)).mean()

    def loglikeli(self, x, y):
        # unnormalized log-likelihood (2x the Gaussian log-likelihood, up to a constant)
        mu, logvar = self.get_mu_logvar(x)
        return (-((mu - y) ** 2) / logvar.exp() - logvar).sum(1).mean(0)

    def learning_loss(self, x, y):
        return -self.loglikeli(x, y)


class CLUBSample(CLUB):
    def forward(self, x, y):
        mu, logvar = self.get_mu_logvar(x)
        perm = torch.randperm(x.shape[0])
        positive = -((mu - y) ** 2) / logvar.exp()
        negative = -((mu - y[perm]) ** 2) / logvar.exp()
        return (positive.sum(-1) - negative.sum(-1)).mean() / 2.0


def club_with_conditional(mu, var, y):
    """CLUB value for a given conditional mean/variance (same formula as CLUB.forward)."""
    positive = -((mu - y) ** 2) / 2.0 / var
    negative = -((y.unsqueeze(0) - mu.unsqueeze(1)) ** 2).mean(dim=1) / 2.0 / var
    return (positive.sum(-1) - negative.sum(-1)).mean()
