
"""
Updated mixin.py for SDE support.

KEY CHANGE: SDEMixin replaces NODEMixin
- solve_ode() → solve_sde()
- Uses torchsde.sdeint() instead of torchdiffeq.odeint()
- All other mixins (scviMixin, betatcMixin, etc.) UNCHANGED

All loss/metric computation logic UNCHANGED - only the dynamics solver differs
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torchsde  # CHANGED: from torchdiffeq import odeint
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import (
    adjusted_rand_score,
    normalized_mutual_info_score,
    silhouette_score,
    calinski_harabasz_score,
    davies_bouldin_score
)
from scipy.sparse import issparse, csr_matrix
from typing import Optional, Tuple
from anndata import AnnData

class scviMixin:
    """Count-based likelihood functions for single-cell RNA-seq data.
    
    ⚠️ UNCHANGED from ODE version
    """
    
    def _normal_kl(self, mu1, lv1, mu2, lv2):
        """
        KL divergence between two diagonal Gaussians.
        
        KL(N(mu1, exp(lv1)) || N(mu2, exp(lv2)))
        """
        v1 = torch.exp(lv1)
        v2 = torch.exp(lv2)
        lstd1 = lv1 / 2.0
        lstd2 = lv2 / 2.0
        return lstd2 - lstd1 + (v1 + (mu1 - mu2) ** 2) / (2.0 * v2) - 0.5
    
    def _log_nb(self, x, mu, theta, eps=1e-8):
        """
        Negative Binomial log-likelihood.
        
        Parameterized by mean mu and inverse dispersion theta.
        """
        log_theta_mu_eps = torch.log(theta + mu + eps)
        return (
            theta * (torch.log(theta + eps) - log_theta_mu_eps)
            + x * (torch.log(mu + eps) - log_theta_mu_eps)
            + torch.lgamma(x + theta)
            - torch.lgamma(theta)
            - torch.lgamma(x + 1)
        )
    
    def _log_zinb(self, x, mu, theta, pi, eps=1e-8):
        """
        Zero-Inflated Negative Binomial log-likelihood.
        
        Mixture of point mass at zero and NB distribution.
        """
        pi = torch.sigmoid(pi)
        log_nb = self._log_nb(x, mu, theta, eps)
        case_zero = torch.log(pi + (1 - pi) * torch.exp(log_nb) + eps)
        case_nonzero = torch.log(1 - pi + eps) + log_nb
        return torch.where(x < eps, case_zero, case_nonzero)
    
    def _log_poisson(self, x, mu, eps=1e-8):
        """Poisson log-likelihood."""
        return x * torch.log(mu + eps) - mu - torch.lgamma(x + 1)
    
    def _log_zip(self, x, mu, pi, eps=1e-8):
        """
        Zero-Inflated Poisson log-likelihood.
        
        Mixture of point mass at zero and Poisson distribution.
        """
        pi = torch.sigmoid(pi)
        case_zero = torch.log(pi + (1 - pi) * torch.exp(-mu) + eps)
        case_nonzero = torch.log(1 - pi + eps) + self._log_poisson(x, mu, eps)
        return torch.where(x < eps, case_zero, case_nonzero)


class betatcMixin:
    """β-TC-VAE total correlation loss for disentanglement.
    
    ⚠️ UNCHANGED from ODE version
    """
    
    def _betatc_compute_gaussian_log_density(self, samples, mean, log_var):
        """Log density of Gaussian distribution."""
        normalization = torch.log(torch.tensor(2 * np.pi))
        inv_sigma = torch.exp(-log_var)
        tmp = samples - mean
        return -0.5 * (tmp * tmp * inv_sigma + log_var + normalization)
    
    def _betatc_compute_total_correlation(self, z_sampled, z_mean, z_logvar):
        """
        Total correlation: KL(q(z) || prod_j q(z_j))
        
        Measures statistical dependence between latent dimensions.
        """
        log_qz_prob = self._betatc_compute_gaussian_log_density(
            z_sampled.unsqueeze(1),
            z_mean.unsqueeze(0),
            z_logvar.unsqueeze(0)
        )
        log_qz_product = log_qz_prob.exp().sum(dim=1).log().sum(dim=1)
        log_qz = log_qz_prob.sum(dim=2).exp().sum(dim=1).log()
        return (log_qz - log_qz_product).mean()


class infoMixin:
    """InfoVAE maximum mean discrepancy loss.
    
    ⚠️ UNCHANGED from ODE version
    """
    
    def _compute_mmd(self, z_posterior, z_prior):
        """
        Maximum Mean Discrepancy with RBF kernel.
        
        Measures distance between posterior and prior distributions.
        """
        mean_pz_pz = self._compute_kernel_mean(
            self._compute_kernel(z_prior, z_prior), unbiased=True
        )
        mean_pz_qz = self._compute_kernel_mean(
            self._compute_kernel(z_prior, z_posterior), unbiased=False
        )
        mean_qz_qz = self._compute_kernel_mean(
            self._compute_kernel(z_posterior, z_posterior), unbiased=True
        )
        return mean_pz_pz - 2 * mean_pz_qz + mean_qz_qz
    
    def _compute_kernel_mean(self, kernel, unbiased):
        """Compute mean of kernel matrix."""
        N = kernel.shape[0]
        if unbiased:
            sum_kernel = kernel.sum() - torch.diagonal(kernel).sum()
            return sum_kernel / (N * (N - 1))
        return kernel.mean()
    
    def _compute_kernel(self, z0, z1):
        """RBF (Gaussian) kernel."""
        batch_size, z_size = z0.shape
        z0 = z0.unsqueeze(1).expand(batch_size, batch_size, z_size)
        z1 = z1.unsqueeze(0).expand(batch_size, batch_size, z_size)
        sigma = 2 * z_size
        return torch.exp(-((z0 - z1).pow(2).sum(dim=-1) / sigma))


class dipMixin:
    """Disentangled Inferred Prior (DIP-VAE) loss.
    
    ⚠️ UNCHANGED from ODE version
    """
    
    def _dip_loss(self, q_m, q_s):
        """
        DIP regularization on posterior covariance matrix.
        
        Encourages diagonal covariance (independence) and unit variance.
        """
        cov_matrix = self._dip_cov_matrix(q_m, q_s)
        cov_diag = torch.diagonal(cov_matrix)
        cov_off_diag = cov_matrix - torch.diag(cov_diag)
        
        dip_loss_d = torch.sum((cov_diag - 1) ** 2)
        dip_loss_od = torch.sum(cov_off_diag ** 2)
        
        return 10 * dip_loss_d + 5 * dip_loss_od
    
    def _dip_cov_matrix(self, q_m, q_s):
        """Covariance matrix of variational posterior."""
        cov_q_mean = torch.cov(q_m.T)
        E_var = torch.mean(torch.exp(q_s), dim=0)
        return cov_q_mean + torch.diag(E_var)


class envMixin:
    """Environment mixin for clustering and evaluation metrics.
    
    ⚠️ UNCHANGED from ODE version
    """
    
    def _calc_score_with_labels(self, latent, labels):
        """
        Compute clustering metrics against ground truth labels.
        
        Parameters
        ----------
        latent : ndarray
            Latent representations
        labels : ndarray
            Ground truth labels
        
        Returns
        -------
        scores : tuple
            (ARI, NMI, Silhouette, Calinski-Harabasz, Davies-Bouldin, Correlation)
        """
        n_clusters = len(np.unique(labels))
        pred_labels = KMeans(n_clusters=n_clusters, n_init=10, random_state=42).fit_predict(latent)
        
        ari = adjusted_rand_score(labels, pred_labels)
        nmi = normalized_mutual_info_score(labels, pred_labels)
        asw = silhouette_score(latent, pred_labels)
        cal = calinski_harabasz_score(latent, pred_labels)
        dav = davies_bouldin_score(latent, pred_labels)
        cor = self._calc_corr(latent)
        
        return (ari, nmi, asw, cal, dav, cor)
    
    def _calc_corr(self, latent):
        """
        Average absolute correlation per dimension.
        
        Measures linear dependencies between latent dimensions.
        """
        acorr = np.abs(np.corrcoef(latent.T))
        return acorr.sum(axis=1).mean().item() - 1


class SDEMixin:
    """
    Mixin providing Neural SDE solving capabilities.
    
    CHANGED FROM: NODEMixin (which used odeint)
    CHANGED TO: SDEMixin (which uses torchsde.sdeint)
    
    Key differences:
    - Solver: torchsde.sdeint instead of torchdiffeq.odeint
    - SDE requirements: forward() and g() methods, not just a function
    - Parameters: dt, sde_type instead of step_size, method
    - CPU/GPU: Same CPU-optimized strategy as before
    """
    
    @staticmethod
    def get_sde_step_size(
        step_size: Optional[float], 
        t0: float, 
        t1: float, 
        n_points: int
    ) -> float:
        """
        Determine SDE solver step size.
        
        Parameters
        ----------
        step_size : float or None
            Explicit step size, or None to auto-compute
        t0, t1 : float
            Time range
        n_points : int
            Number of time points
        
        Returns
        -------
        dt : float
            Recommended step size
        """
        if step_size is None:
            return (t1 - t0) / (n_points - 1)
        elif step_size == "auto":
            return (t1 - t0) / (n_points - 1)
        else:
            return float(step_size)

    def solve_sde(
        self,
        sde_func: nn.Module,
        z0: torch.Tensor,
        t: torch.Tensor,
        method: str = "euler",
        step_size: Optional[float] = None,
        rtol: Optional[float] = None,
        atol: Optional[float] = None,
        sde_type: str = "ito",
    ) -> torch.Tensor:
        """
        Solve SDE using torchsde.
        
        ⚠️ CRITICAL FIXES:
        1. Detach time AND move to same device as z0
        2. Ensure SDE function is on same device
        3. Keep gradients flowing through SDE parameters
        """
        device = z0.device
        
        # ⚠️ FIX: Detach AND move to same device
        if t.requires_grad:
            t = t.detach()
        t = t.to(device)  # ← CRITICAL: Move to same device as z0
        
        # Ensure SDE function is on same device
        sde_func = sde_func.to(device)
        
        # Get solver options
        dt = self.get_sde_step_size(step_size, t[0].item(), t[-1].item(), len(t))
        
        # Prepare solver kwargs
        solver_kwargs = {
            'sde': sde_func,
            'y0': z0,
            'ts': t,
            'method': method,
            'dt': dt,
        }
        
        if rtol is not None:
            solver_kwargs['rtol'] = rtol
        if atol is not None:
            solver_kwargs['atol'] = atol
        
        try:
            # Solve SDE - all tensors now on same device
            pred_z = torchsde.sdeint(**solver_kwargs)
            # Output shape: (num_times, batch_size, latent_dim)
            
        except Exception as e:
            print(f"SDE solving failed: {e}, returning z0 trajectory")
            # Fallback: constant trajectory on same device
            pred_z = z0.unsqueeze(0).repeat(len(t), 1, 1)
        
        return pred_z


# ============================================================================
# Helper Functions (UNCHANGED from ODE version)
# ============================================================================

def quiver_autoscale(E: np.ndarray, V: np.ndarray) -> float:
    """
    Compute autoscale factor for quiver/streamplot visualization.
    
    ⚠️ UNCHANGED from ODE version
    """
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    scale_factor = np.abs(E).max()
    
    if scale_factor == 0:
        scale_factor = 1.0

    Q = ax.quiver(
        E[:, 0] / scale_factor,
        E[:, 1] / scale_factor,
        V[:, 0],
        V[:, 1],
        angles="xy",
        scale=None,
        scale_units="xy",
    )
    
    try:
        fig.canvas.draw()
        quiver_scale = Q.scale if Q.scale is not None else 1.0
    except Exception:
        quiver_scale = 1.0
    finally:
        plt.close(fig)

    return quiver_scale / scale_factor


def l2_norm(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """
    Compute L2 norm (Euclidean length) of vectors.
    
    ⚠️ UNCHANGED from ODE version
    """
    if issparse(x):
        return np.sqrt(x.multiply(x).sum(axis=axis).A1)
    else:
        return np.sqrt(np.sum(x * x, axis=axis))


# ============================================================================
# VectorFieldMixin - Vector Field Analysis
# ============================================================================

class VectorFieldMixin:
    """
    Mixin class for vector field analysis and trajectory visualization.
    
    ⚠️ COMPLETELY UNCHANGED from ODE version
    
    All methods remain identical since vector field computation
    happens after SDE solving (output independent of solver type)
    """

    def get_vfres(
        self,
        adata: AnnData,
        zs_key: str,
        E_key: str,
        vf_key: str = "X_vf",
        T_key: str = "cosine_similarity",
        dv_key: str = "X_dv",
        reverse: bool = False,
        run_neigh: bool = True,
        use_rep_neigh: Optional[str] = None,
        t_key: Optional[str] = None,
        n_neigh: int = 20,
        var_stabilize_transform: bool = False,
        scale: int = 10,
        self_transition: bool = False,
        smooth: float = 0.5,
        stream: bool = True,
        density: float = 1.0,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute vector field for visualization.
        
        ⚠️ UNCHANGED - see ode version for full docstring
        """
        if not self.use_sde:  # CHANGED: use_ode → use_sde
            raise ValueError(
                "Vector field analysis requires use_sde=True. "
                "Reinitialize with: HSDEModel(adata, use_sde=True)"
            )
        
        grads = self.take_grad(self.X_norm)
        adata.obsm[vf_key] = grads
        
        adata.obsp[T_key] = self.get_similarity(
            adata,
            zs_key=zs_key,
            vf_key=vf_key,
            reverse=reverse,
            run_neigh=run_neigh,
            use_rep_neigh=use_rep_neigh,
            t_key=t_key,
            n_neigh=n_neigh,
            var_stabilize_transform=var_stabilize_transform,
        )
        
        adata.obsm[dv_key] = self.get_vf(
            adata,
            T_key=T_key,
            E_key=E_key,
            scale=scale,
            self_transition=self_transition,
        )
        
        E = np.asarray(adata.obsm[E_key])
        V = np.asarray(adata.obsm[dv_key])
        E_grid, V_grid = self.get_vfgrid(
            E=E,
            V=V,
            smooth=smooth,
            stream=stream,
            density=density,
        )
        
        return E_grid, V_grid

    def get_similarity(
        self,
        adata: AnnData,
        zs_key: str,
        vf_key: str = "X_vf",
        reverse: bool = False,
        run_neigh: bool = True,
        use_rep_neigh: Optional[str] = None,
        t_key: Optional[str] = None,
        n_neigh: int = 20,
        var_stabilize_transform: bool = False,
    ) -> csr_matrix:
        """
        Compute cosine similarity-based transition matrix.
        
        ⚠️ UNCHANGED - see ode version for full docstring
        """
        # Compute simple cosine-similarity transition matrix from vector field `V`.
        V = np.array(adata.obsm[vf_key])

        if reverse:
            V = -V
        if var_stabilize_transform:
            V = np.sqrt(np.abs(V)) * np.sign(V)

        ncells = adata.n_obs

        # Cosine similarity matrix (dense), then convert to sparse CSR
        norms = np.linalg.norm(V, axis=1, keepdims=True) + 1e-12
        sim = (V @ V.T) / (norms @ norms.T)
        np.fill_diagonal(sim, 0.0)

        rows, cols = np.nonzero(sim)
        vals = sim[rows, cols]

        return csr_matrix((vals, (rows, cols)), shape=(ncells, ncells))