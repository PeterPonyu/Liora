
"""Mixin classes for different loss functions and metrics."""

import torch
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import (
    adjusted_mutual_info_score,
    normalized_mutual_info_score,
    silhouette_score,
    calinski_harabasz_score,
    davies_bouldin_score
)


class scviMixin:
    """Count-based likelihood functions for scRNA-seq."""
    
    def _normal_kl(self, mu1, lv1, mu2, lv2):
        """KL divergence between two Gaussians."""
        v1 = torch.exp(lv1)
        v2 = torch.exp(lv2)
        lstd1 = lv1 / 2.0
        lstd2 = lv2 / 2.0
        return lstd2 - lstd1 + (v1 + (mu1 - mu2) ** 2) / (2.0 * v2) - 0.5
    
    def _log_nb(self, x, mu, theta, eps=1e-8):
        """Negative Binomial log-likelihood."""
        log_theta_mu_eps = torch.log(theta + mu + eps)
        return (
            theta * (torch.log(theta + eps) - log_theta_mu_eps)
            + x * (torch.log(mu + eps) - log_theta_mu_eps)
            + torch.lgamma(x + theta)
            - torch.lgamma(theta)
            - torch.lgamma(x + 1)
        )
    
    def _log_zinb(self, x, mu, theta, pi, eps=1e-8):
        """Zero-Inflated Negative Binomial log-likelihood."""
        pi = torch.sigmoid(pi)
        log_nb = self._log_nb(x, mu, theta, eps)
        case_zero = torch.log(pi + (1 - pi) * torch.exp(log_nb) + eps)
        case_nonzero = torch.log(1 - pi + eps) + log_nb
        return torch.where(x < eps, case_zero, case_nonzero)
    
    def _log_poisson(self, x, mu, eps=1e-8):
        """Poisson log-likelihood."""
        return x * torch.log(mu + eps) - mu - torch.lgamma(x + 1)
    
    def _log_zip(self, x, mu, pi, eps=1e-8):
        """Zero-Inflated Poisson log-likelihood."""
        pi = torch.sigmoid(pi)
        case_zero = torch.log(pi + (1 - pi) * torch.exp(-mu) + eps)
        case_nonzero = torch.log(1 - pi + eps) + self._log_poisson(x, mu, eps)
        return torch.where(x < eps, case_zero, case_nonzero)


class betatcMixin:
    """β-TC-VAE total correlation loss."""
    
    def _betatc_compute_gaussian_log_density(self, samples, mean, log_var):
        """Gaussian log density."""
        normalization = torch.log(torch.tensor(2 * np.pi))
        inv_sigma = torch.exp(-log_var)
        tmp = samples - mean
        return -0.5 * (tmp * tmp * inv_sigma + log_var + normalization)
    
    def _betatc_compute_total_correlation(self, z_sampled, z_mean, z_logvar):
        """Total correlation term."""
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
        """Maximum mean discrepancy."""
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
        """RBF kernel."""
        batch_size, z_size = z0.shape
        z0 = z0.unsqueeze(1).expand(batch_size, batch_size, z_size)
        z1 = z1.unsqueeze(0).expand(batch_size, batch_size, z_size)
        sigma = 2 * z_size
        return torch.exp(-((z0 - z1).pow(2).sum(dim=-1) / sigma))


class dipMixin:
    """Disentangled Inferred Prior (DIP-VAE) loss."""
    
    def _dip_loss(self, q_m, q_s):
        """DIP regularization on covariance matrix."""
        cov_matrix = self._dip_cov_matrix(q_m, q_s)
        cov_diag = torch.diagonal(cov_matrix)
        cov_off_diag = cov_matrix - torch.diag(cov_diag)
        dip_loss_d = torch.sum((cov_diag - 1) ** 2)
        dip_loss_od = torch.sum(cov_off_diag ** 2)
        return 10 * dip_loss_d + 5 * dip_loss_od
    
    def _dip_cov_matrix(self, q_m, q_s):
        """Covariance matrix of approximate posterior."""
        cov_q_mean = torch.cov(q_m.T)
        E_var = torch.mean(torch.exp(q_s), dim=0)
        return cov_q_mean + torch.diag(E_var)


class envMixin:
    """Environment mixin for clustering and metrics."""
    
    def _calc_score(self, latent):
        """Calculate clustering metrics."""
        labels = KMeans(n_clusters=latent.shape[1]).fit_predict(latent)
        return self._metrics(latent, labels)
    
    def _metrics(self, latent, labels):
        """Compute all metrics."""
        return (
            adjusted_mutual_info_score(self.labels[self.idx], labels),
            normalized_mutual_info_score(self.labels[self.idx], labels),
            silhouette_score(latent, labels),
            calinski_harabasz_score(latent, labels),
            davies_bouldin_score(latent, labels),
            self._calc_corr(latent)
        )
    
    def _calc_corr(self, latent):
        """Average correlation per dimension (original metric)."""
        acorr = abs(np.corrcoef(latent.T))
        return acorr.sum(axis=1).mean().item() - 1
