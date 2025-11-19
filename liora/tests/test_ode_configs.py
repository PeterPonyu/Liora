import torch
import pytest

from liora.module import VAE
from liora.ode_functions import create_ode_func
from liora.mixin import NODEMixin


class DummyNODE(NODEMixin):
    pass


def test_create_ode_funcs():
    f1 = create_ode_func('legacy', n_latent=4, n_hidden=8)
    f2 = create_ode_func('time_mlp', n_latent=4, n_hidden=8, time_cond='concat')
    f3 = create_ode_func('gru', n_latent=4, n_hidden=8)
    assert hasattr(f1, 'forward')
    assert hasattr(f2, 'forward')
    assert hasattr(f3, 'forward')


def _solve(helper, func, method='rk4', step_size='auto', rtol=None, atol=None):
    x0 = torch.randn(1, 4)
    t = torch.linspace(0, 1, 5)
    if hasattr(func, 'reset_hidden'):
        func.reset_hidden()
    z = helper.solve_ode(func, x0, t, method=method, step_size=step_size, rtol=rtol, atol=atol)
    assert z.shape[0] == t.shape[0]
    assert z.shape[1] == x0.shape[0]
    assert z.shape[2] == x0.shape[1]


def test_node_solve_fixed_and_adaptive():
    helper = DummyNODE()
    for ode_type in ['legacy', 'time_mlp', 'gru']:
        func = create_ode_func(ode_type, n_latent=4, n_hidden=8, time_cond='concat')
        # Fixed-step
        _solve(helper, func, method='rk4', step_size='auto')
        # Adaptive
        _solve(helper, func, method='dopri5', step_size=None, rtol=1e-5, atol=1e-7)


def test_vae_forward_across_odes():
    x = torch.randn(8, 20)
    # time_mlp with adaptive solver
    vae1 = VAE(20, 32, 6, 3, use_ode=True, ode_type='time_mlp', ode_time_cond='concat',
               ode_hidden_dim=16, ode_solver_method='dopri5', ode_rtol=1e-5, ode_atol=1e-7)
    out1 = vae1(x)
    assert isinstance(out1, tuple) and len(out1) == 16

    # gru with fixed step
    vae2 = VAE(20, 32, 6, 3, use_ode=True, ode_type='gru', ode_hidden_dim=12,
               ode_solver_method='rk4', ode_step_size='auto')
    out2 = vae2(x)
    assert isinstance(out2, tuple) and len(out2) == 16
