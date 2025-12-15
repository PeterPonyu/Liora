def add_distribution_args(parser):
    group = parser.add_argument_group('HWNormal')
    group.add_argument('--layer', type=str, choices=['Vanilla', 'Geo'], default='Vanilla')
import torch
import geoopt
from torch.nn import functional as F
from torch.distributions import Normal


class Distribution():
    def __init__(self, mean, sigma) -> None:
        self.mean = mean  # (1, *, 3)
        self.sigma = sigma  # (*, 2)

        self.latent_dim = 2
        self.base = Normal(
            torch.zeros([*self.sigma.shape[:-1], 2], device=self.mean.device),
            self.sigma
        )
        self.manifold = geoopt.manifolds.Lorentz()
        self.origin = self.manifold.origin(
            self.mean.size(),
            device=self.mean.device
        )

        self.kl_div = None

    def log_prob(self, z):  # (N, *, 2)
        u = self.manifold.logmap(self.mean, z)  # (N, *, 3)
        v = self.manifold.transp(self.mean, self.origin, u)
        log_prob_v = self.base.log_prob(v[..., 1:]).sum(dim=-1)  # (N, *)

        r = self.manifold.norm(u)  # (N, *)
        log_det = (self.latent_dim - 1) * (torch.sinh(r).log() - r.log())  # (N, *)

        log_prob_z = log_prob_v - log_det  # (N, *)
        return log_prob_z

    def rsample(self, N):
        v = self.base.rsample([N])  # (N, *, 2)
        v = F.pad(v, (1, 0))  # (N, *, 3)

        u = self.manifold.transp0(self.mean, v)  # (N, *, 3)
        z = self.manifold.expmap(self.mean, u)

        return z

    def sample(self, N):
        with torch.no_grad():
            return self.rsample(N)

from .arguments import add_distribution_args
from .layers import VanillaEncoderLayer, VanillaDecoderLayer
from .distribution import Distribution
from .prior import get_prior
import torch
import geoopt
from torch import nn
from torch.nn import functional as F


class VanillaEncoderLayer(nn.Module):
    def __init__(self, args, feature_dim) -> None:
        super().__init__()

        self.latent_dim = args.latent_dim
        self.feature_dim = feature_dim

        self.variational = nn.Linear(
            self.feature_dim,
            4 * self.latent_dim
        )
        self.manifold = geoopt.manifolds.Lorentz()

    def forward(self, feature):
        feature = self.variational(feature)
        mean, logsigma = torch.split(
            feature,
            [2 * self.latent_dim, 2 * self.latent_dim],
            dim=-1
        )

        mean = mean.view(*mean.shape[:-1], self.latent_dim, 2)
        mean = self.manifold.expmap0(F.pad(mean, (1, 0)))
        sigma = F.softplus(logsigma).view(*logsigma.shape[:-1], self.latent_dim, 2)

        return mean, sigma


class VanillaDecoderLayer(nn.Module):
    def __init__(self, args) -> None:
        super().__init__()
        self.manifold = geoopt.manifolds.Lorentz()

    def forward(self, z):
        z = self.manifold.logmap0(z)[..., 1:]
        z = z.reshape(*z.shape[:-2], -1)
        return z

import torch
import geoopt
from torch.nn import functional as F

from .distribution import Distribution


def get_prior(args):
    m = geoopt.manifolds.Lorentz()

    mean = torch.zeros(
        [1, args.latent_dim, 2], 
        device=args.device
    )
    mean = m.expmap0(F.pad(mean, (1, 0)))

    sigma = torch.ones(
        [1, args.latent_dim, 2], 
        device=args.device
    )

    prior = Distribution(mean, sigma)
    return prior

