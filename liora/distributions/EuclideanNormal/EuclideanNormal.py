def add_distribution_args(parser):
    group = parser.add_argument_group('EuclideanNormal')
    group.add_argument('--layer', type=str, choices=['Vanilla', 'Geo'], default='Vanilla')
import torch
from torch.distributions import Normal

from ..utils import euclidean_kl_div


class Distribution():
    def __init__(self, mean, logvar) -> None:
        self.mean = mean
        self.logvar = logvar

        self.base = Normal(self.mean, (self.logvar * 0.5).exp())

    def log_prob(self, z):
        return self.base.log_prob(z)

    def rsample(self, N):
        return self.base.rsample([N])

    def sample(self, N):
        with torch.no_grad():
            return self.rsample(N)

    def kl_div(self, target_dist):
        return euclidean_kl_div(self.mean, self.logvar, target_dist.mean, target_dist.logvar)

from .arguments import add_distribution_args
from .layers import VanillaEncoderLayer, VanillaDecoderLayer
from .distribution import Distribution
from .prior import get_prior
import torch
from torch import nn
from torch.nn import functional as F


class VanillaEncoderLayer(nn.Module):
    def __init__(self, args, feature_dim) -> None:
        super().__init__()

        self.latent_dim = int(args.latent_dim * 2)
        self.feature_dim = feature_dim

        self.variational = nn.Linear(
            self.feature_dim,
            2 * self.latent_dim
        )

    def forward(self, feature):
        feature = self.variational(feature)
        mean, logvar = torch.split(
            feature,
            [self.latent_dim, self.latent_dim],
            dim=-1
        )

        return mean, logvar


class VanillaDecoderLayer(nn.Module):
    def __init__(self, args) -> None:
        super().__init__()

    def forward(self, z):
        return z

import torch

from .distribution import Distribution


def get_prior(args):
    mean = torch.zeros(
        [1, int(args.latent_dim * 2)], 
        device=args.device
    )
    covar = torch.zeros(
        [1, int(args.latent_dim * 2)], 
        device=args.device
    )

    prior = Distribution(mean, covar)
    return prior

