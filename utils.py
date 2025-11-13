# ============================================================================
# utils.py - Lorentz Geometry Utilities
# ============================================================================

"""
Utility functions for Lorentz (hyperbolic) geometry.

The Lorentz model of hyperbolic space uses the hyperboloid:
    H^n = {(x₀, x₁, ..., xₙ) : <x, x> = -1, x₀ > 0}

where the Lorentzian inner product is:
    <x, y> = -x₀y₀ + Σᵢ xᵢyᵢ (i ≥ 1)

This provides negative curvature, ideal for tree-like hierarchical structures
common in biological data (cell lineages, differentiation hierarchies).
"""

import torch

EPS = 1e-8
MAX_NORM = 15.0


def lorentzian_product(
    x: torch.Tensor, 
    y: torch.Tensor, 
    keepdim: bool = False
) -> torch.Tensor:
    """
    Compute Lorentzian inner product.
    
    Defined as: <x, y> = -x₀·y₀ + Σᵢ≥₁ xᵢ·yᵢ
    For points on the hyperboloid, always <x, x> = -1.
    The temporal component (index 0) gets negated, spatial components summed.
    """
    # Negative product for first coordinate (temporal)
    # Positive product for spatial coordinates
    res = -x[..., 0] * y[..., 0] + torch.sum(x[..., 1:] * y[..., 1:], dim=-1)
    
    # Clamp to prevent numerical overflow
    res = torch.clamp(res, min=-1e10, max=1e10)
    
    return res.unsqueeze(-1) if keepdim else res


def lorentz_distance(
    x: torch.Tensor, 
    y: torch.Tensor, 
    eps: float = EPS
) -> torch.Tensor:
    """
    Compute hyperbolic distance on the Lorentz manifold.
    
    For points x, y on hyperboloid H^n:
        d(x, y) = acosh(-<x, y>)
    
    This distance is invariant under Lorentz transformations and gives
    logarithmic distance growth, natural for hierarchy visualization.

    Notes
    -----
    Uses numerically stable acosh:
    - For small x: acosh(x) ≈ sqrt(2(x-1))
    - For large x: acosh(x) ≈ log(2x)
    """
    # Compute Lorentzian product
    xy_inner = lorentzian_product(x, y)
    
    # Clamp for numerical stability: acosh requires input >= 1
    clamped = torch.clamp(-xy_inner, min=1.0 + eps, max=1e10)
    
    # Check for invalid values
    if torch.isnan(clamped).any() or torch.isinf(clamped).any():
        print("Warning: Invalid clamped values in lorentz_distance")
        return torch.zeros_like(clamped).mean()
    
    # Numerically stable acosh
    # For large values, use log approximation to avoid overflow
    dist = torch.where(
        clamped > 1e4,
        torch.log(2 * clamped),  # acosh(x) ≈ log(2x) for large x
        torch.acosh(clamped)      # Exact formula for moderate x
    )
    
    return dist


def exp_map_at_origin(
    v_tangent: torch.Tensor, 
    eps: float = EPS
) -> torch.Tensor:
    """
    Exponential map from tangent space at origin to hyperboloid.
    
    Maps tangent vectors v at the origin to points on the hyperboloid:
        exp₀(v) = (cosh(‖v‖), sinh(‖v‖) · v/‖v‖)
    
    This is the inverse of the logarithmic map and projects free tangent
    vectors into the constrained hyperbolic manifold.
    
    Notes
    -----
    Input should have v_tangent[..., 0] ≈ 0 (tangent space property).
    Uses cosh/sinh for numerically stable hyperbolic functions.
    Falls back to origin if computation produces NaN/Inf.
    """
    # Extract spatial components (skip temporal/0-th component)
    v_spatial = v_tangent[..., 1:]
    
    # Compute norm with clipping for stability
    v_norm = torch.clamp(
        torch.norm(v_spatial, p=2, dim=-1, keepdim=True), 
        max=MAX_NORM
    )
    
    # Handle near-zero norms (avoid division by zero)
    is_zero = v_norm < eps
    v_unit = torch.where(
        is_zero, 
        torch.zeros_like(v_spatial), 
        v_spatial / (v_norm + eps)
    )
    
    # Hyperbolic functions (well-defined for all real inputs)
    x_coord = torch.cosh(v_norm)  # Temporal component
    y_coords = torch.sinh(v_norm) * v_unit  # Spatial components
    
    # Concatenate to form hyperboloid point
    result = torch.cat([x_coord, y_coords], dim=-1)
    
    # Fallback: if invalid values detected, return origin
    if torch.isnan(result).any() or torch.isinf(result).any():
        print("Warning: NaN/Inf in exp_map_at_origin, returning origin")
        safe_point = torch.zeros_like(result)
        safe_point[..., 0] = 1.0  # Origin of hyperboloid
        return safe_point
    
    return result


def euclidean_distance(
    x: torch.Tensor, 
    y: torch.Tensor
) -> torch.Tensor:
    """
    Compute Euclidean (L2) distance between points.
    
    Notes
    -----
    Used as baseline when use_euclidean_manifold=True.
    """
    return torch.norm(x - y, p=2, dim=-1)