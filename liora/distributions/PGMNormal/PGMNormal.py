def add_distribution_args(parser):
    group = parser.add_argument_group('PGMNormal')
    group.add_argument('--c', type=float, default=-1.)
    group.add_argument('--layer', type=str, choices=['Vanilla', 'Geo'], default='Vanilla')
import torch
from math import log, sqrt, pi
from torch.distributions import Normal, Gamma

from ..utils import euclidean_kl_div, gamma_kl_div


class Distribution():
    def __init__(self, means, log_gamma_square, c=-1) -> None:
        self.c = torch.tensor([c], device=means.device)
        self.alpha = means[..., 0]
        self.log_beta_square = means[..., 1]
        self.log_gamma_square = log_gamma_square

        self.normal_mu = self.alpha
        self.normal_logvar = self.log_beta_square + self.log_gamma_square
        self.base1 = Normal(
            self.normal_mu,
            (0.5 * self.normal_logvar).exp()
        )

        self.gamma_a = (-self.log_gamma_square).exp() / (4 * -self.c) + 1
        self.log_gamma_b = -self.normal_logvar - log(4 * -self.c)
        self.base2 = Gamma(
            self.gamma_a,
            self.log_gamma_b.exp()
        )

    def log_prob(self, z):
        target_mu, target_logvar = z[..., 0], z[..., 1]

        kl = euclidean_kl_div(
            sqrt(2 * -self.c) * target_mu,
            target_logvar,
            sqrt(2 * -self.c) * self.alpha,
            self.log_beta_square
        )

        log_prob = -kl / (2 * -self.c * self.log_gamma_square.exp()) + 1.5 * (target_logvar - self.log_beta_square)
        gamma_factor = (-self.log_gamma_square).exp() / (4 * -self.c)
        log_prob = log_prob - 0.5 * self.log_gamma_square
        log_prob = log_prob - torch.lgamma(gamma_factor)
        log_prob = log_prob - gamma_factor
        log_prob = log_prob - gamma_factor * ((4 * -self.c).log() + self.log_gamma_square)
        return log_prob

    def rsample(self, N):
        sample_mean = self.base1.rsample([N])
        sample_shape = torch.Size([N]) + self.gamma_a.shape
        sample_logvar = torch._standard_gamma(self.gamma_a[None].expand(sample_shape)).log() - self.log_gamma_b[None]
        return torch.stack([sample_mean, sample_logvar], dim=-1)

    def sample(self, N):
        with torch.no_grad():
            return self.rsample(N)

    def kl_div(self, target_dist):
        kl1 = euclidean_kl_div(
            self.normal_mu,
            self.normal_logvar,
            target_dist.normal_mu,
            target_dist.normal_logvar
        )
        kl2 = gamma_kl_div(
            self.gamma_a,
            self.log_gamma_b,
            target_dist.gamma_a,
            target_dist.log_gamma_b
        )
        return kl1 + kl2

from .arguments import add_distribution_args
from .layers import VanillaEncoderLayer, GeoEncoderLayer, VanillaDecoderLayer, GeoDecoderLayer
from .distribution import Distribution
from .prior import get_prior
import torch
import geoopt
from torch import nn
from torch.nn import functional as F


class EncoderLayer(nn.Module):
    def __init__(self, args, feature_dim) -> None:
        super().__init__()

        self.latent_dim = args.latent_dim
        self.feature_dim = feature_dim

        self.variational = nn.Linear(
            self.feature_dim,
            3 * self.latent_dim
        )

    def forward(self, feature):
        feature = self.variational(feature)
        alpha, beta, gamma = torch.split(
            feature,
            [self.latent_dim, self.latent_dim, self.latent_dim],
            dim=-1
        )

        return torch.stack([alpha, beta], dim=-1), gamma


class VanillaEncoderLayer(nn.Module):
    def __init__(self, args, feature_dim) -> None:
        super().__init__()
        
        self.encoder = EncoderLayer(args, feature_dim)

    def forward(self, feature):
        return self.encoder(feature)


class GeoEncoderLayer(nn.Module):
    def __init__(self, args, feature_dim) -> None:
        super().__init__()
        
        self.c = torch.tensor([args.c], device=args.device)
        self.encoder = EncoderLayer(args, feature_dim)
        self.manifold = geoopt.manifolds.Lorentz(-1 / args.c)

    def forward(self, feature):
        mean, gamma = self.encoder(feature)
        mean = self.manifold.expmap0(F.pad(mean, (1, 0)))
        mean = lorentz2halfplane(mean, self.c, log=torch.Tensor([True]))
        mean = torch.stack([
            mean[..., 0],
            mean[..., 1] * 2
        ], dim=-1)

        return mean, gamma


class VanillaDecoderLayer(nn.Module):
    def __init__(self, args) -> None:
        super().__init__()

    def forward(self, z):
        z = z.reshape(*z.shape[:-2], -1)
        return z


class GeoDecoderLayer(nn.Module):
    def __init__(self, args) -> None:
        super().__init__()

        self.c = torch.tensor([args.c], device=args.device)
        self.manifold = geoopt.manifolds.Lorentz(-1 / args.c)

    def forward(self, z):
        a, b = z[..., 0], (z[..., 1] * 0.5).exp()
        z = torch.stack([a, b], dim=-1)
        z = halfplane2lorentz(z, self.c)
        z = self.manifold.logmap0(z)[..., 1:]
        return z.reshape(*z.shape[:-2], -1)

import torch

from .distribution import Distribution


def get_prior(args):
    mean = torch.zeros(
        [1, args.latent_dim, 2], 
        device=args.device
    )
    covar = torch.zeros(
        [1, args.latent_dim], 
        device=args.device
    )

    prior = Distribution(mean, covar, args.c)
    return prior

