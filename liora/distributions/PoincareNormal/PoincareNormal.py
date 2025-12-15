def add_distribution_args(parser):
    group = parser.add_argument_group('PoincareNormal')
    group.add_argument('--layer', type=str, choices=['Vanilla', 'Geo'], default='Vanilla')

import torch

infty = torch.tensor(float('Inf'))

def diff(x):
    return x[:, 1:] - x[:, :-1]

class ARS():
    '''
    This class implements the Adaptive Rejection Sampling technique of Gilks and Wild '92.
    Where possible, naming convention has been borrowed from this paper.
    The PDF must be log-concave.
    Currently does not exploit lower hull described in paper- which is fine for drawing
    only small amount of samples at a time.
    '''

    def __init__(self, logpdf, grad_logpdf, device, xi, lb=-infty, ub=infty, use_lower=False, ns=50, **fargs):
        '''
        initialize the upper (and if needed lower) hulls with the specified params
        Parameters
        ==========
        f: function that computes log(f(u,...)), for given u, where f(u) is proportional to the
           density we want to sample from
        fprima:  d/du log(f(u,...))
        xi: ordered vector of starting points in wich log(f(u,...) is defined
            to initialize the hulls
        use_lower: True means the lower sqeezing will be used; which is more efficient
                   for drawing large numbers of samples
        lb: lower bound of the domain
        ub: upper bound of the domain
        ns: maximum number of points defining the hulls
        fargs: arguments for f and fprima
        '''
        self.device = device

        self.lb = lb
        self.ub = ub

        self.logpdf = logpdf
        self.grad_logpdf = grad_logpdf
        self.fargs = fargs

        #set limit on how many points to maintain on hull
        self.ns = ns
        self.xi = xi.to(self.device) # initialize x, the vector of absicassae at which the function h has been evaluated
        self.B, self.K = self.xi.size() # hull size
        self.h = torch.zeros(self.B, ns).to(self.device)
        self.hprime = torch.zeros(self.B, ns).to(self.device)
        self.x = torch.zeros(self.B, ns).to(self.device)
        self.h[:, :self.K] = self.logpdf(self.xi, **self.fargs)
        self.hprime[:, :self.K] = self.grad_logpdf(self.xi, **self.fargs)
        self.x[:, :self.K] = self.xi
        # Avoid under/overflow errors. the envelope and pdf are only
        # proportional to the true pdf, so can choose any constant of proportionality.
        self.offset = self.h.max(-1)[0].view(-1, 1)
        self.h = self.h - self.offset 

        # Derivative at first point in xi must be > 0
        # Derivative at last point in xi must be < 0
        if not (self.hprime[:, 0] > 0).all(): raise IOError('initial anchor points must span mode of PDF (left)')
        if not (self.hprime[:, self.K-1] < 0).all(): raise IOError('initial anchor points must span mode of PDF (right)')
        self.insert()


    def sample(self, shape=torch.Size()):
        '''
        Draw N samples and update upper and lower hulls accordingly
        '''
        shape = shape if isinstance(shape, torch.Size) else torch.Size([shape])
        samples = torch.ones(self.B, *shape).to(self.device)
        bool_mask = (torch.ones(self.B, *shape) == 1).to(self.device)
        count = 0
        while bool_mask.sum() != 0:
            count += 1
            xt, i = self.sampleUpper(shape)
            ht = self.logpdf(xt, **self.fargs)
            # hprimet = self.grad_logpdf(xt, **self.fargs)
            ht = ht - self.offset
            ut = self.h.gather(1, i) + (xt - self.x.gather(1, i)) * self.hprime.gather(1, i)

            # Accept sample?
            u = torch.rand(shape).to(self.device)
            accept = u < torch.exp(ht - ut)
            reject = ~accept
            samples[bool_mask * accept] = xt[bool_mask * accept]
            bool_mask[bool_mask * accept] = reject[bool_mask * accept]
            # Update hull with new function evaluations
            # if self.K < self.ns:
            #     nb_insert = self.ns - self.K
            #     self.insert(nb_insert, xt[:, :nb_insert], ht[:, :nb_insert], hprimet[:, :nb_insert])

        return samples.t().unsqueeze(-1)


    def insert(self, nbnew=0, xnew=None, hnew=None, hprimenew=None):
        '''
        Update hulls with new point(s) if none given, just recalculate hull from existing x,h,hprime
        # '''
        # if xnew is not None:
        #     self.x[:, self.K:self.K+nbnew] = xnew
        #     self.x, idx = self.x.sort()
        #     self.h[:, self.K:self.K+nbnew] = hnew
        #     self.h = self.h.gather(1, idx)
        #     self.hprime[:, self.K:self.K+nbnew] = hprimenew
        #     self.hprime = self.hprime.gather(1, idx)

        #     self.K += xnew.size(-1)

        self.z = torch.zeros(self.B, self.K + 1).to(self.device)
        self.z[:, 0] = self.lb; self.z[:, self.K] = self.ub
        self.z[:, 1:self.K] = (diff(self.h[:, :self.K]) - diff(self.x[:, :self.K] * self.hprime[:, :self.K])) / -diff(self.hprime[:, :self.K]) 
        idx = [0]+list(range(self.K))
        self.u = self.h[:, idx] + self.hprime[:, idx] * (self.z-self.x[:, idx])

        self.s = diff(torch.exp(self.u)) / self.hprime[:, :self.K]
        self.s[self.hprime[:, :self.K] == 0.] = 0. # should be 0 when gradient is 0
        self.cs = torch.cat((torch.zeros(self.B, 1).to(self.device), torch.cumsum(self.s, dim=-1)), dim=-1)
        self.cu = self.cs[:, -1]

    def sampleUpper(self, shape=torch.Size()):
        '''
        Return a single value randomly sampled from the upper hull and index of segment
        '''

        u = torch.rand(self.B, *shape).to(self.device)
        i = (self.cs/self.cu.unsqueeze(-1)).unsqueeze(-1) <= u.unsqueeze(1).expand(*self.cs.shape, *shape)
        idx = i.sum(1) - 1

        xt = self.x.gather(1, idx) + (-self.h.gather(1, idx) + torch.log(self.hprime.gather(1, idx)*(self.cu.unsqueeze(-1)*u - self.cs.gather(1, idx)) + 
        torch.exp(self.u.gather(1, idx)))) / self.hprime.gather(1, idx)

        return xt, idx

import torch
import geoopt

from .hyperbolic_radius import HyperbolicRadius
from .hyperbolic_uniform import HypersphericalUniform

MIN_NORM = 1e-15


def expmap_polar(c, x, u, r, dim: int = -1):
    m = geoopt.manifolds.PoincareBall(1.0)
    sqrt_c = c.sqrt()
    u_norm = u.norm(dim=dim, p=2, keepdim=True).clamp_min(MIN_NORM)
    second_term = (
        (sqrt_c / 2 * r).tanh()
        * u
        / (sqrt_c * u_norm)
    )

    gamma_1 = m.mobius_add(x, second_term)
    return gamma_1


class Distribution():
    def __init__(self, mean, sigma) -> None:
        self.mean = mean
        self.sigma = sigma  # .clamp(min=0.1, max=7.)

        self.manifold = geoopt.manifolds.PoincareBall(1.0)
        self.radius = HyperbolicRadius(2, self.manifold.c, self.sigma.view(-1, 1))
        self.direction = HypersphericalUniform(1, device=mean.device)
        self.kl_div = None

    def log_prob(self, z):
        mean = self.mean[None].expand(z.shape)
        radius_sq = self.manifold.dist(mean, z, keepdim=True).pow(2)
        log_prob_z = - radius_sq / 2 / self.sigma.pow(2)[None] - self.direction._log_normalizer() - self.radius.log_normalizer.view([*self.mean.shape[:-1], -1])
        log_prob_z = log_prob_z.sum(dim=-1)

        return log_prob_z

    def rsample(self, N):
        fixed_shape = self.mean.shape[:-1]
        mean = self.mean.view(-1, 2)
        shape = mean[None].expand([N, *mean.shape]).size()
        alpha = self.direction.sample(torch.Size([*shape[:-1]]))
        radius = self.radius.rsample(torch.Size([N]))
    
        z = expmap_polar(self.manifold.c, mean[None], alpha, radius)
        z = z.reshape([N, *fixed_shape, -1])
        return z

    def sample(self, N):
        with torch.no_grad():
            return self.rsample(N)

# https://github.com/emilemathieu/pvae/tree/master/pvae
import math
import torch
from torch.autograd import Function, grad
import torch.distributions as dist
from math import log, pi
# from pvae.utils import Constants, logsinh, log_sum_exp_signs, rexpand
from numbers import Number
from .ars import ARS
# from pvae.distributions.ars import ARS


def log_sum_exp_signs(value, signs, dim=0, keepdim=False):
    m, _ = torch.max(value, dim=dim, keepdim=True)
    value0 = value - m
    if keepdim is False:
        m = m.squeeze(dim)
    return m + torch.log(torch.sum(signs * torch.exp(value0), dim=dim, keepdim=keepdim))


def rexpand(A, *dimensions):
    """Expand tensor, adding new dimensions on right."""
    return A.view(A.shape + (1,)*len(dimensions)).expand(A.shape + tuple(dimensions))


def cdf_r(value, scale, c, dim):
    value = value.double()
    scale = scale.double()
    c = c.double()

    if dim == 2:
        return 1 / torch.erf(c.sqrt() * scale / math.sqrt(2)) * .5 * \
    (2 * torch.erf(c.sqrt() * scale / math.sqrt(2)) + torch.erf((value - c.sqrt() * scale.pow(2)) / math.sqrt(2) / scale) - \
        torch.erf((c.sqrt() * scale.pow(2) + value) / math.sqrt(2) / scale))
    else:
        device = value.device

        k_float = rexpand(torch.arange(dim), *value.size()).double().to(device)
        dim = torch.tensor(dim).to(device).double()

        s1 = torch.lgamma(dim) - torch.lgamma(k_float + 1) - torch.lgamma(dim - k_float) \
            + (dim - 1 - 2 * k_float).pow(2) * c * scale.pow(2) / 2 \
            + torch.log( \
                torch.erf((value - (dim - 1 - 2 * k_float) * c.sqrt() * scale.pow(2)) / scale / math.sqrt(2)) \
                + torch.erf((dim - 1 - 2 * k_float) * c.sqrt() * scale / math.sqrt(2)) \
                )
        s2 = torch.lgamma(dim) - torch.lgamma(k_float + 1) - torch.lgamma(dim - k_float) \
            + (dim - 1 - 2 * k_float).pow(2) * c * scale.pow(2) / 2 \
            + torch.log1p(torch.erf((dim - 1 - 2 * k_float) * c.sqrt() * scale / math.sqrt(2)))

        signs = torch.tensor([1., -1.]).double().to(device).repeat(((int(dim)+1) // 2)*2)[:int(dim)]
        signs = rexpand(signs, *value.size())

        S1 = log_sum_exp_signs(s1, signs, dim=0)
        S2 = log_sum_exp_signs(s2, signs, dim=0)

        output = torch.exp(S1 - S2)
        zero_value_idx = value == 0.
        output[zero_value_idx] = 0.
        return output  # .float()


def logsinh(x):
    # torch.log(sinh(x))
    return x + torch.log(1 - torch.exp(-2 * x)) - log(2.)


def logcosh(x):
    # torch.log(cosh(x))
    return x + torch.log(1 + torch.exp(-2 * x)) - log(2.)


def grad_cdf_value_scale(value, scale, c, dim):
    device = value.device

    dim = torch.tensor(int(dim)).to(device).double()

    signs = torch.tensor([1., -1.]).double().to(device).repeat(((int(dim)+1) // 2)*2)[:int(dim)]
    signs = rexpand(signs, *value.size())
    k_float = rexpand(torch.arange(dim), *value.size()).double().to(device)

    log_arg1 = (dim - 1 - 2 * k_float).pow(2) * c * scale * \
    (\
        torch.erf((value - (dim - 1 - 2 * k_float) * c.sqrt() * scale.pow(2)) / scale / math.sqrt(2)) \
        + torch.erf((dim - 1 - 2 * k_float) * c.sqrt() * scale / math.sqrt(2)) \
    )
    
    log_arg2 = math.sqrt(2 / math.pi) * ( \
        (dim - 1 - 2 * k_float) * c.sqrt() * torch.exp(-(dim - 1 - 2 * k_float).pow(2) * c * scale.pow(2) / 2) \
        - ((value / scale.pow(2) + (dim - 1 - 2 * k_float) * c.sqrt()) * torch.exp(-(value - (dim - 1 - 2 * k_float) * c.sqrt() * scale.pow(2)).pow(2) / (2 * scale.pow(2)))) \
        )

    log_arg = log_arg1 + log_arg2
    sign_log_arg = torch.sign(log_arg)

    s = torch.lgamma(dim) - torch.lgamma(k_float + 1) - torch.lgamma(dim - k_float) \
            + (dim - 1 - 2 * k_float).pow(2) * c * scale.pow(2) / 2 \
            + torch.log(sign_log_arg * log_arg)

    # log_grad_sum_sigma = log_sum_exp_signs(s, signs * sign_log_arg, dim=0)
    grad_sum_sigma = torch.sum(signs * sign_log_arg * torch.exp(s), dim=0)

    s1 = torch.lgamma(dim) - torch.lgamma(k_float + 1) - torch.lgamma(dim - k_float) \
        + (dim - 1 - 2 * k_float).pow(2) * c * scale.pow(2) / 2 \
        + torch.log( \
            torch.erf((value - (dim - 1 - 2 * k_float) * c.sqrt() * scale.pow(2)) / scale / math.sqrt(2)) \
            + torch.erf((dim - 1 - 2 * k_float) * c.sqrt() * scale / math.sqrt(2)) \
        )

    S1 = log_sum_exp_signs(s1, signs, dim=0)
    grad_log_cdf_scale = grad_sum_sigma / S1.exp()
    log_unormalised_prob = - value.pow(2) / (2 * scale.pow(2)) + (dim - 1) * logsinh(c.sqrt() * value) - (dim - 1) / 2 * c.log()
    
    with torch.autograd.enable_grad():
        # scale = scale.float()
        logZ = _log_normalizer_closed_grad.apply(scale, c, dim)
        grad_logZ_scale = grad(logZ, scale, grad_outputs=torch.ones_like(scale))

    grad_log_cdf_scale = - grad_logZ_scale[0] + 1 / scale + grad_log_cdf_scale  #.float()
    cdf = cdf_r(value.double(), scale.double(), c.double(), int(dim)).squeeze(0)
    grad_scale = cdf * grad_log_cdf_scale

    grad_value = (log_unormalised_prob - logZ).exp()
    return grad_value, grad_scale


class _log_normalizer_closed_grad(Function):
    @staticmethod 
    def forward(ctx, scale, c, dim):
        # scale = scale.double()
        # c = c.double()
        ctx.scale = scale.clone().detach()
        ctx.c = c.clone().detach()
        ctx.dim = dim

        device = scale.device
        output = .5 * (log(pi) - log(2.)) + scale.log() -(int(dim) - 1) * (c.log() / 2 + log(2.))
        dim = torch.tensor(int(dim), device=device)

        k_float = rexpand(torch.arange(int(dim)), *scale.size()).to(device)
        s = torch.lgamma(dim) - torch.lgamma(k_float + 1) - torch.lgamma(dim - k_float) \
            + (dim - 1 - 2 * k_float).pow(2) * c * scale.pow(2) / 2 \
            + torch.log1p(torch.erf((dim - 1 - 2 * k_float) * c.sqrt() * scale / math.sqrt(2)))
        signs = torch.tensor([1., -1.], device=device).repeat(((int(dim)+1) // 2)*2)[:int(dim)]
        signs = rexpand(signs, *scale.size())
        ctx.log_sum_term = log_sum_exp_signs(s, signs, dim=0)
        output = output + ctx.log_sum_term

        return output  # .float()

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.clone()

        device = grad_input.device
        scale = ctx.scale
        c = ctx.c
        dim = torch.tensor(int(ctx.dim), device=device)

        k_float = rexpand(torch.arange(int(dim)), *scale.size()).double().to(device)
        signs = torch.tensor([1., -1.]).double().to(device).repeat(((int(dim)+1) // 2)*2)[:int(dim)]
        signs = rexpand(signs, *scale.size())

        log_arg = (dim - 1 - 2 * k_float).pow(2) * c * scale * (1+torch.erf((dim - 1 - 2 * k_float) * c.sqrt() * scale / math.sqrt(2))) + \
            torch.exp(-(dim - 1 - 2 * k_float).pow(2) * c * scale.pow(2) / 2) * 2 / math.sqrt(math.pi) * (dim - 1 - 2 * k_float) * c.sqrt() / math.sqrt(2)
        log_arg_signs = torch.sign(log_arg)
        s = torch.lgamma(dim) - torch.lgamma(k_float + 1) - torch.lgamma(dim - k_float) \
            + (dim - 1 - 2 * k_float).pow(2) * c * scale.pow(2) / 2 \
            + torch.log(log_arg_signs * log_arg)
        log_grad_sum_sigma = log_sum_exp_signs(s, log_arg_signs * signs, dim=0)

        grad_scale = torch.exp(log_grad_sum_sigma - ctx.log_sum_term)
        grad_scale = 1 / ctx.scale + grad_scale

        grad_scale = (grad_input * grad_scale.double()).view(-1, *grad_input.shape).sum(0)
        # print(grad_scale.isnan().sum())
        return (grad_scale, None, None)


class impl_rsample(Function):
    @staticmethod
    def forward(ctx, value, scale, c, dim):
        ctx.scale = scale.clone().detach().double().requires_grad_(True)
        ctx.value = value.clone().detach().double().requires_grad_(True)
        ctx.c = c.clone().detach().double().requires_grad_(True)
        ctx.dim = dim
        return value

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.clone()
        grad_cdf_value, grad_cdf_scale = grad_cdf_value_scale(ctx.value, ctx.scale, ctx.c, ctx.dim)
        assert not torch.isnan(grad_cdf_value).any()
        assert not torch.isnan(grad_cdf_scale).any()
        grad_value_scale = -(grad_cdf_value).pow(-1) * grad_cdf_scale.expand(grad_input.shape)
        grad_scale = (grad_input * grad_value_scale).view(-1, *grad_cdf_scale.shape).sum(0)
        # grad_value_c = -(grad_cdf_value).pow(-1) * grad_cdf_c.expand(grad_input.shape)
        # grad_c = (grad_input * grad_value_c).view(-1, *grad_cdf_c.shape).sum(0)
        return (None, grad_scale, None, None)


class HyperbolicRadius(dist.Distribution):
    support = dist.constraints.positive
    has_rsample = True

    def __init__(self, dim, c, scale, ars=True, validate_args=False):
        self.dim = dim
        self.c = c
        self.scale = scale
        self.device = scale.device
        self.ars = ars
        if isinstance(scale, Number):
            batch_shape = torch.Size()
        else:
            batch_shape = self.scale.size()
        self.log_normalizer = self._log_normalizer()
        if torch.isnan(self.log_normalizer).any() or torch.isinf(self.log_normalizer).any():
            print('nan or inf in log_normalizer', torch.cat((self.log_normalizer, self.scale), dim=1))
            raise
        super(HyperbolicRadius, self).__init__(batch_shape, validate_args=False)

    def rsample(self, sample_shape=torch.Size()):
        value = self.sample(sample_shape)
        return impl_rsample.apply(value, self.scale, self.c, self.dim)

    def sample(self, sample_shape=torch.Size()):
        if sample_shape == torch.Size(): sample_shape=torch.Size([1])
        with torch.no_grad():
            mean = self.mean
            stddev = self.stddev
            if torch.isnan(stddev).any(): stddev[torch.isnan(stddev)] = self.scale[torch.isnan(stddev)]
            if torch.isnan(mean).any(): mean[torch.isnan(mean)] = ((self.dim - 1) * self.scale.pow(2) * self.c.sqrt())[torch.isnan(mean)]
            steps = torch.linspace(0.1, 3, 10).to(self.device)
            steps = torch.cat((-steps.flip(0), steps))
            xi = [mean + s * torch.min(stddev, .95 * mean / 3) for s in steps]
            xi = torch.cat(xi, dim=1)
            ars = ARS(self.log_prob, self.grad_log_prob, self.device, xi=xi, ns=20, lb=0)
            value = ars.sample(sample_shape)
        return value

    def __while_loop(self, logM, proposal, sample_shape):
        shape = self._extended_shape(sample_shape)
        r, bool_mask = torch.ones(shape).to(self.device), (torch.ones(shape) == 1).to(self.device)
        count = 0
        while bool_mask.sum() != 0:
            count += 1
            r_ = proposal.sample(sample_shape).to(self.device)
            u = torch.rand(shape).to(self.device)
            log_ratio = self.log_prob(r_) - proposal.log_prob(r_) - logM
            accept = log_ratio > torch.log(u)
            reject = 1 - accept
            r[bool_mask * accept] = r_[bool_mask * accept]
            bool_mask[bool_mask * accept] = reject[bool_mask * accept]
        return r

    def log_prob(self, value):
        res = - value.pow(2) / (2 * self.scale.pow(2)) + (self.dim - 1) * logsinh(self.c.sqrt() * value) \
            - (self.dim - 1) / 2 * self.c.log() - self.log_normalizer#.expand(value.shape)
        assert not torch.isnan(res).any()
        return res

    def grad_log_prob(self, value):
        res = - value / self.scale.pow(2) + (self.dim - 1) * self.c.sqrt() * torch.cosh(self.c.sqrt() * value) / torch.sinh(self.c.sqrt() * value) 
        return res

    def cdf(self, value):
        return cdf_r(value, self.scale, self.c, self.dim)

    @property
    def mean(self):
        c = self.c
        scale = self.scale
        dim = torch.tensor(int(self.dim), device=self.device)
        signs = torch.tensor([1., -1.], device=self.device).repeat(((self.dim+1) // 2) * 2)[:self.dim].unsqueeze(-1).unsqueeze(-1).expand(self.dim, *self.scale.size())
        
        k_float = rexpand(torch.arange(self.dim), *self.scale.size()).double().to(self.device)
        s2 = torch.lgamma(dim) - torch.lgamma(k_float + 1) - torch.lgamma(dim - k_float) \
                + (dim - 1 - 2 * k_float).pow(2) * c * scale.pow(2) / 2 \
                + torch.log1p(torch.erf((dim - 1 - 2 * k_float) * c.sqrt() * scale / math.sqrt(2)))
        S2 = log_sum_exp_signs(s2, signs, dim=0)

        log_arg = (dim - 1 - 2 * k_float) * c.sqrt() * scale.pow(2) * (1 + torch.erf((dim - 1 - 2 * k_float) * c.sqrt() * scale / math.sqrt(2))) + \
                torch.exp(-(dim - 1 - 2 * k_float).pow(2) * c * scale.pow(2) / 2) * scale * math.sqrt(2 / math.pi)
        log_arg_signs = torch.sign(log_arg)
        s1 = torch.lgamma(dim) - torch.lgamma(k_float + 1) - torch.lgamma(dim - k_float) \
                + (dim - 1 - 2 * k_float).pow(2) * c * scale.pow(2) / 2 \
                + torch.log(log_arg_signs * log_arg)
        S1 = log_sum_exp_signs(s1, signs * log_arg_signs, dim=0)

        output = torch.exp(S1 - S2)
        return output  # .float()

    @property
    def variance(self):
        c = self.c
        scale = self.scale
        dim = torch.tensor(int(self.dim), device=self.device)
        signs = torch.tensor([1., -1.], device=self.device).repeat(((int(dim)+1) // 2)*2)[:int(dim)].unsqueeze(-1).unsqueeze(-1).expand(int(dim), *self.scale.size())

        k_float = rexpand(torch.arange(self.dim), *self.scale.size()).double().to(self.device)
        s2 = torch.lgamma(dim) - torch.lgamma(k_float + 1) - torch.lgamma(dim - k_float) \
                + (dim - 1 - 2 * k_float).pow(2) * c * scale.pow(2) / 2 \
                + torch.log1p(torch.erf((dim - 1 - 2 * k_float) * c.sqrt() * scale / math.sqrt(2)))
        S2 = log_sum_exp_signs(s2, signs, dim=0)

        log_arg = (1 + (dim - 1 - 2 * k_float).pow(2) * c * scale.pow(2)) * (1 + torch.erf((dim - 1 - 2 * k_float) * c.sqrt() * scale / math.sqrt(2))) + \
               (dim - 1 - 2 * k_float) * c.sqrt() * torch.exp(-(dim - 1 - 2 * k_float).pow(2) * c * scale.pow(2) / 2) * scale * math.sqrt(2 / math.pi)
        log_arg_signs = torch.sign(log_arg)
        s1 = torch.lgamma(dim) - torch.lgamma(k_float + 1) - torch.lgamma(dim - k_float) \
                + (dim - 1 - 2 * k_float).pow(2) * c * scale.pow(2) / 2 \
                + 2 * scale.log() \
                + torch.log(log_arg_signs * log_arg)
        S1 = log_sum_exp_signs(s1, signs * log_arg_signs, dim=0)

        output = torch.exp(S1 - S2)
        output = output - self.mean.pow(2)
        return output

    @property
    def stddev(self): return self.variance.sqrt()

    def _log_normalizer(self): return _log_normalizer_closed_grad.apply(self.scale, self.c, self.dim)

import math
import torch
from torch.distributions.utils import _standard_normal

class HypersphericalUniform(torch.distributions.Distribution):
    """ source: https://github.com/nicola-decao/s-vae-pytorch/blob/master/hyperspherical_vae/distributions/von_mises_fisher.py """

    support = torch.distributions.constraints.real
    has_rsample = False
    _mean_carrier_measure = 0

    @property
    def dim(self):
        return self._dim
    
    def __init__(self, dim, device='cpu', validate_args=None):
        super(HypersphericalUniform, self).__init__(torch.Size([dim]), validate_args=False)
        self._dim = dim
        self._device = device

    def sample(self, shape=torch.Size()):
        with torch.no_grad():
            return self.rsample(shape)

    def rsample(self, sample_shape=torch.Size()):
        shape = torch.Size([*sample_shape, self._dim + 1])
        output = _standard_normal(shape, dtype=torch.float, device=self._device)

        return output / output.norm(dim=-1, keepdim=True)

    def log_prob(self, x):
        return - torch.ones(x.shape[:-1]).to(self._device) * self._log_normalizer()

    def _log_normalizer(self):
        return self._log_surface_area().to(self._device)

    def _log_surface_area(self):
        return math.log(2) + ((self._dim + 1) / 2) * math.log(math.pi) - torch.lgamma(
            torch.Tensor([(self._dim + 1) / 2]))

    def entropy(self):
        return self._log_surface_area()

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
            3 * self.latent_dim
        )
        self.manifold = geoopt.manifolds.PoincareBall(1.0)

    def forward(self, feature):
        feature = self.variational(feature)
        mean, logsigma = torch.split(
            feature,
            [2 * self.latent_dim, self.latent_dim],
            dim=-1
        )

        mean = mean.view(*mean.shape[:-1], self.latent_dim, 2)
        mean = self.manifold.expmap0(mean)
        sigma = F.softplus(logsigma)[..., None]

        return mean, sigma


class VanillaDecoderLayer(nn.Module):
    def __init__(self, args) -> None:
        super().__init__()

        self.manifold = geoopt.manifolds.PoincareBall(1.0)

    def forward(self, z):
        z = self.manifold.logmap0(z)
        z = z.reshape(*z.shape[:-2], -1)
        return z

import torch

from .distribution import Distribution


def get_prior(args):
    mean = torch.zeros(
        [1, args.latent_dim, 2], 
        device=args.device
    )
    sigma = torch.ones(
        [1, args.latent_dim, 1], 
        device=args.device
    )

    prior = Distribution(mean, sigma)
    return prior

