
"""Environment for LiVAE model training and evaluation."""

from .model import LiVAEModel
from .mixin import envMixin
import numpy as np
from scipy.sparse import issparse
from sklearn.cluster import KMeans


def is_raw_counts(X, threshold=0.5):
    """
    Check if data matrix contains raw integer counts.
    
    Parameters
    ----------
    X : array-like
        Data matrix (sparse or dense)
    threshold : float
        Minimum proportion of integer-like values
        
    Returns
    -------
    bool
        True if data appears to be raw counts
    """
    # Sample data
    if issparse(X):
        sample_data = X.data[:min(10000, len(X.data))]
    else:
        flat_data = X.flatten()
        sample_data = flat_data[np.random.choice(
            len(flat_data), min(10000, len(flat_data)), replace=False
        )]
    
    sample_data = sample_data[sample_data > 0]
    if len(sample_data) == 0:
        return False
    
    # Check for normalized/log-transformed data indicators
    if np.mean((sample_data > 0) & (sample_data < 1)) > 0.1:
        return False
    if np.any(sample_data < 0):
        return False
    
    # Check for integer-like values
    integer_like = np.abs(sample_data - np.round(sample_data)) < 1e-6
    return np.mean(integer_like) >= threshold


def compute_dataset_stats(X):
    """Compute statistics for adaptive normalization."""
    X_dense = X.toarray() if issparse(X) else X
    
    return {
        'sparsity': np.mean(X_dense == 0),
        'lib_size_mean': X_dense.sum(axis=1).mean(),
        'lib_size_std': X_dense.sum(axis=1).std(),
        'max_val': X_dense.max()
    }


class Env(LiVAEModel, envMixin):
    """Environment handling data loading and preprocessing."""
    
    def __init__(
        self,
        adata,
        layer,
        percent,
        recon,
        irecon,
        lorentz,
        beta,
        dip,
        tc,
        info,
        hidden_dim,
        latent_dim,
        i_dim,
        lr,
        use_bottleneck_lorentz,
        loss_type,
        device,
        grad_clip=1.0,
        adaptive_norm=True,
        use_layer_norm=True,
        use_euclidean_manifold=False,
        **kwargs
    ):
        # Register data first
        self.loss_type = loss_type
        self.adaptive_norm = adaptive_norm
        self._register_anndata(adata, layer, latent_dim)
        
        # Set batch size
        self.batch_size = max(32, int(percent * self.n_obs))
        if self.batch_size != int(percent * self.n_obs):
            print(f"  Adjusted batch size to {self.batch_size}")
        
        # Initialize model
        super().__init__(
            recon=recon,
            irecon=irecon,
            lorentz=lorentz,
            beta=beta,
            dip=dip,
            tc=tc,
            info=info,
            state_dim=self.n_var,
            hidden_dim=hidden_dim,
            latent_dim=latent_dim,
            i_dim=i_dim,
            lr=lr,
            use_bottleneck_lorentz=use_bottleneck_lorentz,
            loss_type=loss_type,
            device=device,
            grad_clip=grad_clip,
            use_layer_norm=use_layer_norm,
            use_euclidean_manifold=use_euclidean_manifold
        )
        self.score = []
    
    def load_data(self):
        """Load a random batch of data."""
        idx = np.random.choice(self.n_obs, self.batch_size, replace=False)
        self.idx = idx
        return self.X_norm[idx], self.X_raw[idx]
        
    def step(self, data):
        """Perform one training step."""
        data_norm, data_raw = data
        self.update(data_norm, data_raw)
        latent = self.take_latent(data_norm)
        score = self._calc_score(latent)
        self.score.append(score)

    def _register_anndata(self, adata, layer: str, latent_dim: int):
        """Register and preprocess AnnData object."""
        # Get raw counts
        X = adata.layers[layer]
        
        if not is_raw_counts(X):
            raise ValueError(
                f"Layer '{layer}' does not contain raw counts. "
                f"Loss type '{self.loss_type}' requires unnormalized integer counts."
            )
        
        X = X.toarray() if issparse(X) else np.asarray(X)
        self.X_raw = X.astype(np.float32)
        
        # Compute and display statistics
        stats = compute_dataset_stats(X)
        print(f"  Sparsity: {stats['sparsity']:.2f}, "
              f"Lib size: {stats['lib_size_mean']:.0f}±{stats['lib_size_std']:.0f}, "
              f"Max value: {stats['max_val']:.0f}")
        
        # Adaptive normalization
        X_log = np.log1p(X)
        
        if self.adaptive_norm:
            if stats['sparsity'] > 0.95:
                print("  → Aggressive clipping for sparse data")
                self.X_norm = np.clip(X_log, -5, 5).astype(np.float32)
            elif stats['lib_size_std'] / stats['lib_size_mean'] > 2.0:
                print("  → Per-cell standardization for high variance")
                cell_means = X_log.mean(axis=1, keepdims=True)
                cell_stds = X_log.std(axis=1, keepdims=True) + 1e-6
                self.X_norm = np.clip((X_log - cell_means) / cell_stds, -10, 10).astype(np.float32)
            elif stats['max_val'] > 10000:
                print("  → Scaled normalization for extreme values")
                scale = min(1.0, 10.0 / X_log.max())
                self.X_norm = np.clip(X_log * scale, -10, 10).astype(np.float32)
            else:
                self.X_norm = np.clip(X_log, -10, 10).astype(np.float32)
        else:
            self.X_norm = np.clip(X_log, -10, 10).astype(np.float32)
        
        # Validate
        if np.isnan(self.X_norm).any():
            raise ValueError(f"NaN detected in normalized data")
        if np.isinf(self.X_norm).any():
            raise ValueError(f"Inf detected in normalized data")
        
        self.n_obs, self.n_var = adata.shape
        
        # Initialize clustering labels
        try:
            self.labels = KMeans(
                n_clusters=latent_dim,
                n_init=10,
                max_iter=300,
                random_state=0
            ).fit_predict(self.X_norm)
        except Exception as e:
            print(f"  Warning: KMeans failed, using random labels")
            self.labels = np.random.randint(0, latent_dim, size=self.n_obs)
