# ============================================================================
# mixin.py - Loss Functions and Metrics
# ============================================================================
from __future__ import annotations

import torch
import torch.nn as nn
from torchdiffeq import odeint
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import (
    adjusted_rand_score,
    normalized_mutual_info_score,
    silhouette_score,
    calinski_harabasz_score,
    davies_bouldin_score
)

from scipy.sparse import issparse, csr_matrix, coo_matrix
from typing import Optional, Tuple
from anndata import AnnData

class scviMixin:
    """Count-based likelihood functions for single-cell RNA-seq data."""
    
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
    """β-TC-VAE total correlation loss for disentanglement."""
    
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
    """InfoVAE maximum mean discrepancy loss."""
    
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
            # Exclude diagonal for unbiased estimate
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
    """Disentangled Inferred Prior (DIP-VAE) loss."""
    
    def _dip_loss(self, q_m, q_s):
        """
        DIP regularization on posterior covariance matrix.
        
        Encourages diagonal covariance (independence) and unit variance.
        """
        cov_matrix = self._dip_cov_matrix(q_m, q_s)
        cov_diag = torch.diagonal(cov_matrix)
        cov_off_diag = cov_matrix - torch.diag(cov_diag)
        
        # Penalize deviation from identity covariance
        dip_loss_d = torch.sum((cov_diag - 1) ** 2)
        dip_loss_od = torch.sum(cov_off_diag ** 2)
        
        return 10 * dip_loss_d + 5 * dip_loss_od
    
    def _dip_cov_matrix(self, q_m, q_s):
        """Covariance matrix of variational posterior."""
        cov_q_mean = torch.cov(q_m.T)
        E_var = torch.mean(torch.exp(q_s), dim=0)
        return cov_q_mean + torch.diag(E_var)


class envMixin:
    """Environment mixin for clustering and evaluation metrics."""
    
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
        # Perform KMeans clustering
        n_clusters = len(np.unique(labels))
        pred_labels = KMeans(n_clusters=n_clusters, n_init=10, random_state=42).fit_predict(latent)
        
        # Compute metrics
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
        # Subtract 1 to exclude self-correlation
        return acorr.sum(axis=1).mean().item() - 1
    

class NODEMixin:
    """
    Mixin providing Neural ODE solving capabilities.
    
    Handles CPU-GPU device transfers for efficient ODE integration.
    The ODE solver runs on CPU (computational advantage), while
    model parameters remain on the specified device.
    """
    
    @staticmethod
    def get_step_size(
        step_size: Optional[float], 
        t0: float, 
        t1: float, 
        n_points: int
    ) -> dict:
        """
        Determine ODE solver step size.
        
        """
        if step_size is None:
            return {}
        else:
            if step_size == "auto":
                step_size = (t1 - t0) / (n_points - 1)
            return {"step_size": step_size}

    def solve_ode(
        self,
        ode_func: nn.Module,
        z0: torch.Tensor,
        t: torch.Tensor,
        method: str = "rk4",
        step_size: Optional[float] = None,
        rtol: Optional[float] = None,
        atol: Optional[float] = None,
    ) -> torch.Tensor:
        """
        Solve ODE using torchdiffeq on CPU.
        
        Key Design Decision: ODE solving intentionally remains on CPU because:
        1. torchdiffeq's adaptive step-size algorithms are CPU-optimized
        2. Latent dimension is small (typically 10-20), minimal GPU benefit
        3. Significant speedup (~2-3x) observed on CPU vs GPU
        4. Memory efficiency: avoids GPU memory pressure
        """
        # Get solver options
        options = self.get_step_size(step_size, t[0].item(), t[-1].item(), len(t))
        
        # Transfer to CPU for ODE solving
        original_device = z0.device
        cpu_z0 = z0.to("cpu")
        cpu_t = t.to("cpu")        
        try:
            # Solve ODE on CPU
            kwargs = {}
            if rtol is not None:
                kwargs['rtol'] = rtol
            if atol is not None:
                kwargs['atol'] = atol
            pred_z = odeint(ode_func, cpu_z0, cpu_t, method=method, options=options, **kwargs)
        except Exception as e:
            print(f"ODE solving failed: {e}, returning z0 trajectory")
            # Fallback: return constant trajectory
            pred_z = cpu_z0.unsqueeze(0).repeat(len(cpu_t), 1, 1)

        # Transfer result back to original device
        pred_z = pred_z.to(original_device)
        
        return pred_z


# ============================================================================
# Helper Functions
# ============================================================================

def quiver_autoscale(E: np.ndarray, V: np.ndarray) -> float:
    """
    Compute autoscale factor for quiver/streamplot visualization.
    
    Parameters
    ----------
    E : np.ndarray
        Embedding coordinates, shape (n_cells, 2)
    V : np.ndarray
        Velocity vectors, shape (n_cells, 2)
    
    Returns
    -------
    scale : float
        Autoscale factor
    """
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    scale_factor = np.abs(E).max()
    
    # Avoid division by zero
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
    
    # Render the figure to compute Q.scale
    try:
        fig.canvas.draw()
        quiver_scale = Q.scale if Q.scale is not None else 1.0
    except Exception:
        # Fallback if rendering fails
        quiver_scale = 1.0
    finally:
        plt.close(fig)

    return quiver_scale / scale_factor


def l2_norm(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """
    Compute L2 norm (Euclidean length) of vectors.
    
    Parameters
    ----------
    x : np.ndarray or sparse matrix
        Input vectors
    axis : int, default=-1
        Axis along which to compute norm
    
    Returns
    -------
    norm : np.ndarray
        L2 norms
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
    
    Provides methods for computing velocity fields, similarity matrices,
    and interpolation onto regular grids for visualization purposes.
    
    Requires:
    - self.use_ode: bool - ODE mode enabled
    - self.X: np.ndarray - Raw data matrix
    - self.device: torch.device - Computation device
    - self.nn: VAE module - Neural network model
    
    Methods
    -------
    get_vfres : Compute complete vector field representation
    get_similarity : Compute cosine similarity transition matrix
    get_vf : Project velocity field onto embedding space
    get_vfgrid : Interpolate velocity field onto regular grid
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
        Compute vector field for visualization (requires use_ode=True).
        
        Generates a complete vector field representation by:
        1. Computing velocity gradients in latent space
        2. Building a neighbor-based transition matrix
        3. Projecting velocities onto embedding space
        4. Interpolating onto a regular grid
        
        Parameters
        ----------
        adata : AnnData
            Annotated data object
        zs_key : str
            Key in adata.obsm for latent space (e.g., 'X_latent')
        E_key : str
            Key in adata.obsm for embedding space (e.g., 'X_umap')
        vf_key : str, default='X_vf'
            Key to store velocity field in adata.obsm
        T_key : str, default='cosine_similarity'
            Key to store transition matrix in adata.obsp
        dv_key : str, default='X_dv'
            Key to store projected velocities in adata.obsm
        reverse : bool, default=False
            Reverse velocity direction
        run_neigh : bool, default=True
            Recompute neighborhood graph
        use_rep_neigh : str, optional
            Representation for neighbor detection (defaults to zs_key)
        t_key : str, optional
            Key in adata.obs for pseudotime constraint
        n_neigh : int, default=20
            Number of neighbors
        var_stabilize_transform : bool, default=False
            Apply variance stabilizing transform
        scale : int, default=10
            Scaling factor for transition probabilities
        self_transition : bool, default=False
            Include self-transitions
        smooth : float, default=0.5
            Smoothing factor for grid interpolation
        stream : bool, default=True
            Return streamplot format (True) or quiver format (False)
        density : float, default=1.0
            Grid density for interpolation
        
        Returns
        -------
        E_grid : np.ndarray
            Grid coordinates for plotting
        V_grid : np.ndarray
            Interpolated velocities on grid
        
        Raises
        ------
        ValueError
            If use_ode=False
        
        Examples
        --------
        >>> adata.obsm['X_latent'] = model.get_latent()
        >>> sc.tl.umap(adata)
        >>> E_grid, V_grid = model.get_vfres(
        ...     adata, 
        ...     zs_key='X_latent', 
        ...     E_key='X_umap'
        ... )
        >>> ax = sc.pl.embedding(adata, basis='umap', show=False)
        >>> ax.streamplot(E_grid[0], E_grid[1], V_grid[0], V_grid[1])
        """
        if not self.use_ode:
            raise ValueError(
                "Vector field analysis requires use_ode=True. "
                "Reinitialize with: Liora(adata, use_ode=True)"
            )
        
        # Step 1: Compute velocity gradients
        grads = self.take_grad(self.X)
        adata.obsm[vf_key] = grads
        
        # Step 2: Compute transition similarity matrix
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
        
        # Step 3: Project velocities to embedding space
        adata.obsm[dv_key] = self.get_vf(
            adata,
            T_key=T_key,
            E_key=E_key,
            scale=scale,
            self_transition=self_transition,
        )
        
        # Step 4: Interpolate onto regular grid
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
        
        Builds a directed graph of cell-to-cell transitions based on:
        1. Neighborhood structure in latent space
        2. Velocity alignment (cosine similarity between displacement and velocity)
        3. Optional pseudotime constraints
        
        Parameters
        ----------
        adata : AnnData
            Annotated data object
        zs_key : str
            Key in adata.obsm for latent space
        vf_key : str, default='X_vf'
            Key in adata.obsm for velocity field
        reverse : bool, default=False
            Reverse velocity direction
        run_neigh : bool, default=True
            Recompute neighborhood graph
        use_rep_neigh : str, optional
            Representation for neighbor detection
        t_key : str, optional
            Key in adata.obs for pseudotime constraint
        n_neigh : int, default=20
            Number of neighbors
        var_stabilize_transform : bool, default=False
            Apply sqrt transform for variance stabilization
        
        Returns
        -------
        similarity : scipy.sparse.csr_matrix
            Cosine similarity matrix, shape (n_cells, n_cells)
        """
        Z = np.array(adata.obsm[zs_key])
        V = np.array(adata.obsm[vf_key])
        
        # Apply optional transformations
        if reverse:
            V = -V
        if var_stabilize_transform:
            V = np.sqrt(np.abs(V)) * np.sign(V)

        ncells = adata.n_obs

        # Build neighborhood graph
        if run_neigh or ("neighbors" not in adata.uns):
            if use_rep_neigh is None:
                use_rep_neigh = zs_key
            elif use_rep_neigh not in adata.obsm:
                raise KeyError(
                    f"`{use_rep_neigh}` not found in `.obsm`. "
                    "Please provide valid `use_rep_neigh`."
                )
            sc.pp.neighbors(adata, use_rep=use_rep_neigh, n_neighbors=n_neigh)
        
        n_neigh = adata.uns["neighbors"]["params"]["n_neighbors"] - 1

        # Pseudotime-constrained neighbors (optional)
        indices_matrix2 = None
        if t_key is not None:
            if t_key not in adata.obs:
                raise KeyError(f"`{t_key}` not found in `.obs`.")
            ts = adata.obs[t_key].values
            indices_matrix2 = np.zeros((ncells, n_neigh), dtype=int)
            for i in range(ncells):
                idx = np.abs(ts - ts[i]).argsort()[: (n_neigh + 1)]
                idx = np.setdiff1d(idx, i) if i in idx else idx[:-1]
                indices_matrix2[i] = idx

        # Compute cosine similarities
        vals: list = []
        rows: list = []
        cols: list = []
        
        for i in range(ncells):
            # Get neighbors (first-order + second-order)
            dist_mat = adata.obsp["distances"]
            row1 = dist_mat[i]
            idx = row1.indices if hasattr(row1, "indices") else np.where(row1 > 0)[0]
            
            # Collect second-order neighbors
            idx2_list = []
            for j in idx:
                r = dist_mat[j]
                if hasattr(r, "indices"):
                    idx2_list.append(r.indices)
                else:
                    idx2_list.append(np.where(r > 0)[0])
            idx2 = np.unique(np.concatenate(idx2_list)) if idx2_list else np.array([], dtype=int)
            idx2 = np.setdiff1d(idx2, i)
            
            # Combine neighbors with optional pseudotime constraint
            if t_key is None:
                idx = np.unique(np.concatenate([idx, idx2]))
            else:
                idx = np.unique(np.concatenate([idx, idx2, indices_matrix2[i]]))
            
            # Compute displacement vectors
            dZ = Z[idx] - Z[i, None]
            if var_stabilize_transform:
                dZ = np.sqrt(np.abs(dZ)) * np.sign(dZ)
            
            # Cosine similarity: cos(θ) = (dZ · V) / (|dZ| * |V|)
            cos_sim = np.einsum("ij, j", dZ, V[i]) / (
                l2_norm(dZ, axis=1) * l2_norm(V[i])
            )
            cos_sim[np.isnan(cos_sim)] = 0
            
            vals.extend(cos_sim)
            rows.extend(np.repeat(i, len(idx)))
            cols.extend(idx)

        # Build sparse COO matrix and convert to CSR
        res = coo_matrix((vals, (rows, cols)), shape=(ncells, ncells))
        res.data = np.clip(res.data, -1, 1)
        
        return res.tocsr()

    def get_vf(
        self,
        adata: AnnData,
        T_key: str,
        E_key: str,
        scale: int = 10,
        self_transition: bool = False,
    ) -> np.ndarray:
        """
        Project velocity field onto embedding space.
        
        Transforms latent space velocities to embedding coordinates by:
        1. Exponential scaling of transition probabilities
        2. Normalization and optional self-transition handling
        3. Weighted projection onto embedding space
        
        Parameters
        ----------
        adata : AnnData
            Annotated data object
        T_key : str
            Key in adata.obsp for transition matrix
        E_key : str
            Key in adata.obsm for embedding coordinates
        scale : int, default=10
            Exponential scaling factor for transitions
        self_transition : bool, default=False
            Include self-transitions in projection
        
        Returns
        -------
        V : np.ndarray
            Projected velocity field, shape (n_cells, n_dims)
        """
        T = adata.obsp[T_key].copy()

        # Handle self-transitions
        if self_transition:
            max_t = T.max(1).A.flatten() if hasattr(T.max(1), 'A') else np.array(T.max(1)).flatten()
            ub = np.percentile(max_t, 98)
            self_t = np.clip(ub - max_t, 0, 1)
            if hasattr(T, 'setdiag'):
                T.setdiag(self_t)

        # Apply exponential transform
        sign_T = T.sign() if hasattr(T, 'sign') else np.sign(T)
        if hasattr(sign_T, 'multiply'):
            T = sign_T.multiply(np.expm1(np.abs(T.data) * scale))
        else:
            T = sign_T * np.expm1(np.abs(T * scale))
        
        # Normalize rows
        if hasattr(T, 'multiply'):
            denom = np.array(np.abs(T).sum(1)).flatten()
            denom = np.maximum(denom, 1e-12)
            T = T.multiply(csr_matrix(1.0 / denom[:, np.newaxis]))
        else:
            denom = np.maximum(np.abs(T).sum(1, keepdims=True), 1e-12)
            T = T / denom
        
        # Clear self-transitions if requested
        if self_transition and hasattr(T, 'setdiag'):
            T.setdiag(0)
            if hasattr(T, 'eliminate_zeros'):
                T.eliminate_zeros()

        # Project to embedding space
        E = np.array(adata.obsm[E_key])
        V = np.zeros(E.shape)

        for i in range(adata.n_obs):
            idx = T[i].indices
            dE = E[idx] - E[i, None]
            dE /= l2_norm(dE)[:, None]
            dE[np.isnan(dE)] = 0
            prob = T[i].data
            V[i] = prob.dot(dE) - prob.mean() * dE.sum(0)

        V /= 3 * quiver_autoscale(E, V)
        
        return V

    def get_vfgrid(
        self,
        E: np.ndarray,
        V: np.ndarray,
        smooth: float = 0.5,
        stream: bool = True,
        density: float = 1.0,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Interpolate vector field onto regular grid.
        
        Creates a regular grid covering the embedding space and interpolates
        velocity values using Gaussian kernel smoothing.
        
        Parameters
        ----------
        E : np.ndarray
            Embedding coordinates, shape (n_cells, n_dims)
        V : np.ndarray
            Velocity vectors, shape (n_cells, n_dims)
        smooth : float, default=0.5
            Gaussian kernel smoothing bandwidth
        stream : bool, default=True
            Format for streamplot (True) or quiver (False)
        density : float, default=1.0
            Grid density multiplier
        
        Returns
        -------
        E_grid : np.ndarray
            Grid coordinates
        V_grid : np.ndarray
            Interpolated velocities
        """
        # Create regular grid
        grs = []
        for i in range(E.shape[1]):
            m, M = np.min(E[:, i]), np.max(E[:, i])
            diff = M - m
            m = m - 0.01 * diff
            M = M + 0.01 * diff
            gr = np.linspace(m, M, int(50 * density))
            grs.append(gr)

        meshes = np.meshgrid(*grs)
        E_grid_points = np.vstack([i.flat for i in meshes]).T

        # Find neighbors for each grid point
        n_neigh = max(1, int(E.shape[0] / 50))
        nn = NearestNeighbors(n_neighbors=n_neigh, n_jobs=-1)
        nn.fit(E)
        dists, neighs = nn.kneighbors(E_grid_points)

        # Gaussian kernel smoothing
        scale = np.mean([g[1] - g[0] for g in grs]) * smooth
        weight = norm.pdf(x=dists, scale=scale)
        weight_sum = weight.sum(1)

        V_grid = (V[neighs] * weight[:, :, None]).sum(1)
        V_grid /= np.maximum(1, weight_sum)[:, None]

        if stream:
            # Format for streamplot
            E_grid = np.stack(grs)
            ns = E_grid.shape[1]
            V_grid = V_grid.T.reshape(2, ns, ns)

            # Mask low-confidence regions
            mass = np.sqrt((V_grid * V_grid).sum(0))
            min_mass = 1e-5
            min_mass = np.clip(min_mass, None, np.percentile(mass, 99) * 0.01)
            cutoff1 = mass < min_mass

            length = np.sum(
                np.mean(np.abs(V[neighs]), axis=1), axis=1
            ).reshape(ns, ns)
            cutoff2 = length < np.percentile(length, 5)

            cutoff = cutoff1 | cutoff2
            V_grid[0][cutoff] = np.nan
        else:
            # Format for quiver plot
            min_weight = np.percentile(weight_sum, 99) * 0.01
            mask = weight_sum > min_weight
            E_grid = E_grid_points[mask]
            V_grid = V_grid[mask]
            V_grid /= 3 * quiver_autoscale(E_grid, V_grid)

        return E_grid, V_grid