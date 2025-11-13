
"""Utility functions for Lorentz geometry."""

import torch

EPS = 1e-8
MAX_NORM = 15.0


def lorentzian_product(x, y, keepdim=False):
    """
    Lorentzian inner product: <x, y> = -x₀y₀ + x₁y₁ + ... + xₙyₙ
    
    Parameters
    ----------
    x, y : Tensor
        Points on Lorentz manifold
    keepdim : bool
        Keep dimension
        
    Returns
    -------
    Tensor
        Inner product
    """
    res = -x[..., 0] * y[..., 0] + torch.sum(x[..., 1:] * y[..., 1:], dim=-1)
    res = torch.clamp(res, min=-1e10, max=1e10)
    return res.unsqueeze(-1) if keepdim else res


def lorentz_distance(x, y, eps=EPS):
    """
    Hyperbolic distance on Lorentz manifold.
    
    d(x, y) = acosh(-<x, y>)
    
    Parameters
    ----------
    x, y : Tensor
        Points on hyperboloid
    eps : float
        Numerical stability constant
        
    Returns
    -------
    Tensor
        Hyperbolic distances
    """
    xy_inner = lorentzian_product(x, y)
    clamped = torch.clamp(-xy_inner, min=1.0 + eps, max=1e10)
    
    if torch.isnan(clamped).any() or torch.isinf(clamped).any():
        return torch.zeros_like(clamped).mean()
    
    # Stable acosh: for large x, acosh(x) ≈ log(2x)
    dist = torch.where(
        clamped > 1e4,
        torch.log(2 * clamped),
        torch.acosh(clamped)
    )
    
    return dist


def exp_map_at_origin(v_tangent, eps=EPS):
    """
    Exponential map at origin: tangent space → hyperboloid.
    
    exp₀(v) = (cosh(‖v‖), sinh(‖v‖) · v/‖v‖)
    
    Parameters
    ----------
    v_tangent : Tensor
        Tangent vectors at origin (first coordinate is 0)
    eps : float
        Numerical stability constant
        
    Returns
    -------
    Tensor
        Points on hyperboloid
    """
    v_spatial = v_tangent[..., 1:]
    v_norm = torch.clamp(torch.norm(v_spatial, p=2, dim=-1, keepdim=True), max=MAX_NORM)
    
    # Handle near-zero norms
    is_zero = v_norm < eps
    v_unit = torch.where(is_zero, torch.zeros_like(v_spatial), v_spatial / (v_norm + eps))
    
    # Hyperbolic functions
    x_coord = torch.cosh(v_norm)
    y_coords = torch.sinh(v_norm) * v_unit
    
    result = torch.cat([x_coord, y_coords], dim=-1)
    
    # Fallback to origin if invalid
    if torch.isnan(result).any() or torch.isinf(result).any():
        safe_point = torch.zeros_like(result)
        safe_point[..., 0] = 1.0
        return safe_point
    
    return result


def euclidean_distance(x, y):
    """
    Euclidean L2 distance.
    
    Parameters
    ----------
    x, y : Tensor
        Embedding vectors
        
    Returns
    -------
    Tensor
        L2 distances
    """
    return torch.norm(x - y, p=2, dim=-1)
    