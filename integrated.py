
"""LiVAE: Lorentz Information-regularized Variational AutoEncoder for single-cell transcriptomics."""

from .environment import Env
from anndata import AnnData
import torch
import tqdm


class Liora(Env):
    """
    Liora model for single-cell RNA-seq analysis.
    
    Combines Variational Autoencoder with:
    - Lorentz manifold regularization for geometric structure
    - Information bottleneck for representation learning
    - Multiple count-based likelihood functions
    
    Parameters
    ----------
    adata : AnnData
        Annotated data matrix containing raw counts
    layer : str, default='counts'
        Layer in adata.layers containing raw count data
    recon : float, default=1.0
        Weight for primary reconstruction loss        
    irecon : float, default=0.0
        Weight for information bottleneck reconstruction loss
    lorentz : float, default=0.0
        Weight for Lorentz manifold regularization
    beta : float, default=1.0
        Weight for KL divergence (β-VAE)
    dip : float, default=0.0
        Weight for disentangled inferred prior (DIP) loss
    tc : float, default=0.0
        Weight for total correlation (β-TC-VAE) loss
    info : float, default=0.0
        Weight for maximum mean discrepancy (InfoVAE) loss
    hidden_dim : int, default=128
        Hidden layer dimension
    latent_dim : int, default=10
        Latent space dimension
    i_dim : int, default=2
        Information bottleneck dimension
    lr : float, default=1e-4
        Learning rate
    use_bottleneck_lorentz : bool, default=True
        If True, use bottleneck for Lorentz pairing; if False, use resampling
    loss_type : str, default='nb'
        Count likelihood: 'nb' (Negative Binomial), 'zinb' (Zero-Inflated NB),
        'poisson', or 'zip' (Zero-Inflated Poisson)
    grad_clip : float, default=1.0
        Gradient clipping threshold
    adaptive_norm : bool, default=True
        Use adaptive normalization based on dataset statistics
    use_layer_norm : bool, default=True
        Use layer normalization for training stability
    use_euclidean_manifold : bool, default=False
        Use Euclidean manifold instead of Lorentz
    use_ode : bool, default=False
        Use Neural ODE regularization
    vae_reg : float, default=0.5
        Weight for VAE path in ODE mode
    ode_reg : float, default=0.5
        Weight for ODE path in ODE mode
    train_size : float, default=0.7
        Proportion of data for training
    val_size : float, default=0.15
        Proportion of data for validation
    test_size : float, default=0.15
        Proportion of data for testing
    batch_size : int, default=128
        Batch size for training
    random_seed : int, default=42
        Random seed for reproducibility
    device : torch.device, optional
        Device for training (defaults to CUDA if available)
    """
    
    def __init__(
        self,
        adata: AnnData,
        layer: str = 'counts',
        recon: float = 1.0,
        irecon: float = 0.0,
        lorentz: float = 0.0,
        beta: float = 1.0,
        dip: float = 0.0,
        tc: float = 0.0,
        info: float = 0.0,
        hidden_dim: int = 128,
        latent_dim: int = 10,
        i_dim: int = 2,
        lr: float = 1e-4,
        use_bottleneck_lorentz: bool = True,
        loss_type: str = 'nb',
        grad_clip: float = 1.0,
        adaptive_norm: bool = True,
        use_layer_norm: bool = True,
        use_euclidean_manifold: bool = False,
        use_ode: bool = False,
        vae_reg: float = 0.5,
        ode_reg: float = 0.5,
        train_size: float = 0.7,
        val_size: float = 0.15,
        test_size: float = 0.15,
        batch_size: int = 128,
        random_seed: int = 42,
        device: torch.device = None
    ):
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            
        super().__init__(
            adata=adata,
            layer=layer,
            recon=recon,
            irecon=irecon,
            lorentz=lorentz,
            beta=beta,
            dip=dip,
            tc=tc,
            info=info,
            hidden_dim=hidden_dim,
            latent_dim=latent_dim,
            i_dim=i_dim,
            lr=lr,
            use_bottleneck_lorentz=use_bottleneck_lorentz,
            loss_type=loss_type,
            grad_clip=grad_clip,
            adaptive_norm=adaptive_norm,
            use_layer_norm=use_layer_norm,
            use_euclidean_manifold=use_euclidean_manifold,
            use_ode=use_ode,
            vae_reg=vae_reg,
            ode_reg=ode_reg,
            train_size=train_size,
            val_size=val_size,
            test_size=test_size,
            batch_size=batch_size,
            random_seed=random_seed,
            device=device
        )
    
    def fit(
        self, 
        epochs: int = 400,
        patience: int = 25,
        val_every: int = 5,
        early_stop: bool = True,
    ):
        """
        Train model with epoch-based training and early stopping.
        
        Parameters
        ----------
        epochs : int, default=100
            Maximum number of epochs
        patience : int, default=20
            Early stopping patience (epochs without improvement)
        val_every : int, default=5
            Validate every N epochs
        early_stop : bool, default=True
            Enable early stopping
            
        Returns
        -------
        self : LiVAE
            Trained model
        """
        with tqdm.tqdm(total=epochs, desc="Training", ncols=200) as pbar:
            for epoch in range(epochs):
                
                # Train for one epoch
                train_loss = self.train_epoch()
                
                # Validate periodically
                if (epoch + 1) % val_every == 0 or epoch == 0:
                    val_loss, val_score = self.validate()
                    if early_stop:
                        # Check early stopping
                        should_stop, improved = self.check_early_stopping(
                            val_loss, patience
                        )
                        
                        # Update progress bar
                        pbar.set_postfix({
                            "Train": f"{train_loss:.2f}",
                            "Val": f"{val_loss:.2f}",
                            "ARI": f"{val_score[0]:.2f}",
                            "NMI": f"{val_score[1]:.2f}",
                            "ASW": f"{val_score[2]:.2f}",
                            "CAL": f"{val_score[3]:.2f}",
                            "DAV": f"{val_score[4]:.2f}",
                            "COR": f"{val_score[5]:.2f}",
                            "Best": f"{self.best_val_loss:.2f}",
                            "Pat": f"{self.patience_counter}/{patience}",
                            "✓" if improved else "✗": ""
                        })
                        
                        if should_stop:
                            print(f"\n\nEarly stopping triggered at epoch {epoch + 1}")
                            print(f"Best validation loss: {self.best_val_loss:.4f}")
                            self.load_best_model()
                            break
                    else:
                        # Update progress bar without early stopping
                        pbar.set_postfix({
                            "Train": f"{train_loss:.2f}",
                            "Val": f"{val_loss:.2f}",
                            "ARI": f"{val_score[0]:.2f}",
                            "NMI": f"{val_score[1]:.2f}",
                            "ASW": f"{val_score[2]:.2f}",
                            "CAL": f"{val_score[3]:.2f}",
                            "DAV": f"{val_score[4]:.2f}",
                            "COR": f"{val_score[5]:.2f}",
                        })
                pbar.update(1)
        
        return self
    
    def get_latent(self):
        """
        Get latent representations for all cells.
        
        Returns
        -------
        latent : ndarray
            Latent representations of shape (n_cells, latent_dim)
        """
        return self.take_latent(self.X_norm)
    
    def get_test_latent(self):
        """
        Get latent representation from test set only.
        
        Returns
        -------
        latent : ndarray
            Test set latent representations
        """
        return self.take_latent(self.X_test_norm)
    
    def get_bottleneck(self):
        """
        Get information bottleneck representations.
        
        Returns
        -------
        bottleneck : ndarray
            Bottleneck representations of shape (n_cells, i_dim)
        """
        x = torch.tensor(self.X_norm, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            outputs = self.nn(x)
            le = outputs[4]  # Information bottleneck encoding
        return le.cpu().numpy()

"""Environment for LiVAE model training and evaluation."""

from .model import LioraModel
from .mixin import envMixin
import numpy as np
from scipy.sparse import issparse
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder
import torch
from torch.utils.data import DataLoader, TensorDataset


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


class Env(LioraModel, envMixin):
    """Environment handling data loading and preprocessing."""
    
    def __init__(
        self,
        adata,
        layer,
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
        use_ode=False,
        vae_reg=0.5,
        ode_reg=0.5,
        train_size=0.7,
        val_size=0.15,
        test_size=0.15,
        batch_size=128,
        random_seed=42,
        **kwargs
    ):
        # Store split parameters
        self.train_size = train_size
        self.val_size = val_size
        self.test_size = test_size
        self.batch_size = batch_size
        self.random_seed = random_seed
        
        # Register data first
        self.loss_type = loss_type
        self.adaptive_norm = adaptive_norm
        self._register_anndata(adata, layer, latent_dim)
        
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
            use_euclidean_manifold=use_euclidean_manifold,
            use_ode=use_ode,
            vae_reg=vae_reg,
            ode_reg=ode_reg
        )
        
        # Initialize tracking
        self.score = []
        self.train_losses = []
        self.val_losses = []
        self.val_scores = []
        
        # Early stopping parameters
        self.best_val_loss = float('inf')
        self.best_model_state = None
        self.patience_counter = 0
    
    def _register_anndata(self, adata, layer: str, latent_dim: int):
        """Register and preprocess AnnData object with train/val/test splits."""
        # Get raw counts
        X = adata.layers[layer]
        
        if not is_raw_counts(X):
            raise ValueError(
                f"Layer '{layer}' does not contain raw counts. "
                f"Loss type '{self.loss_type}' requires unnormalized integer counts."
            )
        
        X = X.toarray() if issparse(X) else np.asarray(X)
        X_raw = X.astype(np.float32)
        
        # Compute and display statistics
        stats = compute_dataset_stats(X)
        print(f"Dataset statistics:")
        print(f"  Cells: {X.shape[0]:,}, Genes: {X.shape[1]:,}")
        print(f"  Sparsity: {stats['sparsity']:.2f}, "
              f"Lib size: {stats['lib_size_mean']:.0f}±{stats['lib_size_std']:.0f}, "
              f"Max value: {stats['max_val']:.0f}")
        
        # Adaptive normalization
        X_log = np.log1p(X)
        
        if self.adaptive_norm:
            if stats['sparsity'] > 0.95:
                print("  → Aggressive clipping for sparse data")
                X_norm = np.clip(X_log, -5, 5).astype(np.float32)
            elif stats['lib_size_std'] / stats['lib_size_mean'] > 2.0:
                print("  → Per-cell standardization for high variance")
                cell_means = X_log.mean(axis=1, keepdims=True)
                cell_stds = X_log.std(axis=1, keepdims=True) + 1e-6
                X_norm = np.clip((X_log - cell_means) / cell_stds, -10, 10).astype(np.float32)
            elif stats['max_val'] > 10000:
                print("  → Scaled normalization for extreme values")
                scale = min(1.0, 10.0 / X_log.max())
                X_norm = np.clip(X_log * scale, -10, 10).astype(np.float32)
            else:
                X_norm = np.clip(X_log, -10, 10).astype(np.float32)
        else:
            X_norm = np.clip(X_log, -10, 10).astype(np.float32)
        
        # Validate
        if np.isnan(X_norm).any():
            raise ValueError(f"NaN detected in normalized data")
        if np.isinf(X_norm).any():
            raise ValueError(f"Inf detected in normalized data")
        
        self.n_obs, self.n_var = adata.shape
        
        # Generate labels for evaluation
        if 'cell_type' in adata.obs.columns:
            # Use actual cell type labels if available
            self.labels = LabelEncoder().fit_transform(adata.obs['cell_type'])
            print(f"  Using cell_type labels: {len(np.unique(self.labels))} types")
        else:
            # Use KMeans as pseudo-labels for evaluation
            try:
                self.labels = KMeans(
                    n_clusters=latent_dim,
                    n_init=10,
                    max_iter=300,
                    random_state=self.random_seed
                ).fit_predict(X_norm)
                print(f"  Generated KMeans pseudo-labels: {latent_dim} clusters")
            except Exception as e:
                print(f"  Warning: KMeans failed, using random labels")
                self.labels = np.random.randint(0, latent_dim, size=self.n_obs)
        
        # Create train/val/test splits
        np.random.seed(self.random_seed)
        indices = np.random.permutation(self.n_obs)
        
        n_train = int(self.train_size * self.n_obs)
        n_val = int(self.val_size * self.n_obs)
        
        self.train_idx = indices[:n_train]
        self.val_idx = indices[n_train:n_train + n_val]
        self.test_idx = indices[n_train + n_val:]
        
        # Split data
        self.X_train_norm = X_norm[self.train_idx]
        self.X_train_raw = X_raw[self.train_idx]
        self.X_val_norm = X_norm[self.val_idx]
        self.X_val_raw = X_raw[self.val_idx]
        self.X_test_norm = X_norm[self.test_idx]
        self.X_test_raw = X_raw[self.test_idx]
        
        # Store full data for convenience
        self.X_norm = X_norm
        self.X_raw = X_raw
        
        self.labels_train = self.labels[self.train_idx]
        self.labels_val = self.labels[self.val_idx]
        self.labels_test = self.labels[self.test_idx]
        
        print(f"\nData split:")
        print(f"  Train: {len(self.train_idx):,} cells ({len(self.train_idx)/self.n_obs*100:.1f}%)")
        print(f"  Val:   {len(self.val_idx):,} cells ({len(self.val_idx)/self.n_obs*100:.1f}%)")
        print(f"  Test:  {len(self.test_idx):,} cells ({len(self.test_idx)/self.n_obs*100:.1f}%)")
        
        # Create PyTorch DataLoaders
        self._create_dataloaders()
    
    def _create_dataloaders(self):
        """Create PyTorch DataLoaders for train/val/test sets."""
        # Convert to tensors
        X_train_norm_tensor = torch.FloatTensor(self.X_train_norm)
        X_train_raw_tensor = torch.FloatTensor(self.X_train_raw)
        X_val_norm_tensor = torch.FloatTensor(self.X_val_norm)
        X_val_raw_tensor = torch.FloatTensor(self.X_val_raw)
        X_test_norm_tensor = torch.FloatTensor(self.X_test_norm)
        X_test_raw_tensor = torch.FloatTensor(self.X_test_raw)
        
        # Create datasets
        train_dataset = TensorDataset(X_train_norm_tensor, X_train_raw_tensor)
        val_dataset = TensorDataset(X_val_norm_tensor, X_val_raw_tensor)
        test_dataset = TensorDataset(X_test_norm_tensor, X_test_raw_tensor)
        
        # Create dataloaders
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=True  # Drop last incomplete batch
        )
        
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            drop_last=False
        )
        
        self.test_loader = DataLoader(
            test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            drop_last=False
        )
        
        print(f"  Batch size: {self.batch_size}, Batches per epoch: {len(self.train_loader)}")
    
    def train_epoch(self):
        """Train for one complete epoch through training data."""
        self.nn.train()  # Set model to training mode
        epoch_losses = []
        
        for batch_norm, batch_raw in self.train_loader:
            batch_norm = batch_norm.to(self.device)
            batch_raw = batch_raw.to(self.device)
            self.update(batch_norm.cpu().numpy(), batch_raw.cpu().numpy())
            epoch_losses.append(self.loss[-1][0])  # Get last total loss
        
        avg_train_loss = np.mean(epoch_losses)
        self.train_losses.append(avg_train_loss)
        
        return avg_train_loss
    
    def validate(self):
        """Evaluate on validation set."""
        self.nn.eval()  # Set model to evaluation mode
        val_losses = []
        all_latents = []
        
        with torch.no_grad():
            for batch_norm, batch_raw in self.val_loader:
                batch_norm = batch_norm.to(self.device)
                batch_raw = batch_raw.to(self.device)
                
                # Forward pass (compute loss without updating)
                loss_value = self._compute_loss_only(batch_norm, batch_raw)
                val_losses.append(loss_value)
                
                # Get latent representations
                latent = self.take_latent(batch_norm.cpu().numpy())
                all_latents.append(latent)
        
        # Average validation loss
        avg_val_loss = np.mean(val_losses)
        self.val_losses.append(avg_val_loss)
        
        # Compute metrics on validation latents
        all_latents = np.concatenate(all_latents, axis=0)
        val_score = self._calc_score_with_labels(all_latents, self.labels_val)
        self.val_scores.append(val_score)
        
        return avg_val_loss, val_score
    
    def _compute_loss_only(self, states_norm, states_raw):
        """Compute loss without gradient updates (for validation)."""
        states_norm = states_norm.to(self.device)
        states_raw = states_raw.to(self.device)
        
        # Forward pass
        if self.use_ode:
            (q_z, q_m, q_s, pred_x, le, ld, pred_xl, z_manifold, ld_manifold,
             dropout_x, dropout_xl, q_z_ode, pred_x_ode, dropout_x_ode,
             x_sorted, t) = self.nn(states_norm)
            
            # ODE divergence loss
            import torch.nn.functional as F
            qz_div = F.mse_loss(q_z, q_z_ode, reduction="none").sum(-1).mean()
            
            # Reconstruction losses (both paths)
            recon_loss = self.recon * self._compute_reconstruction_loss(
                x_sorted, pred_x, dropout_x
            )
            recon_loss += self.recon * self._compute_reconstruction_loss(
                x_sorted, pred_x_ode, dropout_x_ode
            )
        else:
            q_z, q_m, q_s, pred_x, le, ld, pred_xl, z_manifold, ld_manifold, dropout_x, dropout_xl = \
                self.nn(states_norm)
            
            qz_div = torch.tensor(0.0, device=self.device)
            recon_loss = self.recon * self._compute_reconstruction_loss(
                states_raw, pred_x, dropout_x
            )
        
        # Geometric regularization
        geometric_loss = torch.tensor(0.0, device=self.device)
        if self.lorentz > 0:
            if not (torch.isnan(z_manifold).any() or torch.isnan(ld_manifold).any()):
                if self.use_euclidean_manifold:
                    from .utils import euclidean_distance
                    dist = euclidean_distance(z_manifold, ld_manifold)
                else:
                    from .utils import lorentz_distance
                    dist = lorentz_distance(z_manifold, ld_manifold)
                
                if not torch.isnan(dist).any():
                    geometric_loss = self.lorentz * dist.mean()
        
        # Information bottleneck reconstruction
        irecon_loss = torch.tensor(0.0, device=self.device)
        if self.irecon > 0:
            if self.use_ode:
                irecon_loss = self.irecon * self._compute_reconstruction_loss(
                    x_sorted, pred_xl, dropout_xl
                )
            else:
                irecon_loss = self.irecon * self._compute_reconstruction_loss(
                    states_raw, pred_xl, dropout_xl
                )
        
        # KL divergence
        kl_div = self.beta * self._normal_kl(
            q_m, q_s, torch.zeros_like(q_m), torch.zeros_like(q_s)
        ).sum(dim=-1).mean()
        
        # Additional regularizations
        dip_loss = self.dip * self._dip_loss(q_m, q_s) if self.dip > 0 else torch.tensor(0.0, device=self.device)
        tc_loss = self.tc * self._betatc_compute_total_correlation(q_z, q_m, q_s) if self.tc > 0 else torch.tensor(0.0, device=self.device)
        mmd_loss = self.info * self._compute_mmd(q_z, torch.randn_like(q_z)) if self.info > 0 else torch.tensor(0.0, device=self.device)
        
        # Total loss
        total_loss = (
            recon_loss + irecon_loss + geometric_loss + qz_div + 
            kl_div + dip_loss + tc_loss + mmd_loss
        )
        
        return total_loss.item()
    
    def check_early_stopping(self, val_loss, patience=25):
        """
        Check if training should stop early.
        
        Parameters
        ----------
        val_loss : float
            Current validation loss
        patience : int
            Number of epochs to wait before stopping
            
        Returns
        -------
        should_stop : bool
            Whether training should stop
        improved : bool
            Whether validation loss improved
        """
        if val_loss < self.best_val_loss:
            # Improvement
            self.best_val_loss = val_loss
            self.best_model_state = {
                k: v.cpu().clone() for k, v in self.nn.state_dict().items()
            }
            self.patience_counter = 0
            return False, True  # Continue training, improved
        else:
            # No improvement
            self.patience_counter += 1
            
            if self.patience_counter >= patience:
                return True, False  # Stop training, not improved
            else:
                return False, False  # Continue training, not improved
    
    def load_best_model(self):
        """Load the best model from early stopping checkpoint."""
        if self.best_model_state is not None:
            self.nn.load_state_dict(self.best_model_state)
            print(f"Loaded best model with validation loss: {self.best_val_loss:.4f}")
        else:
            print("Warning: No best model state found!")
from .agent import Liora

__all__ = ['Liora']

__version__ = '0.3.0'

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

"""Core Liora model with loss computation and ODE support."""

import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from sklearn.metrics.pairwise import pairwise_distances
from .mixin import scviMixin, dipMixin, betatcMixin, infoMixin
from .module import VAE
from .utils import lorentz_distance


class LioraModel(scviMixin, dipMixin, betatcMixin, infoMixin):
    """
    Liora model with multiple regularization losses and optional ODE regularization.
    
    Combines standard VAE with:
    - Lorentz manifold regularization
    - Information bottleneck
    - Disentanglement losses (DIP, β-TC)
    - MMD regularization
    - Optional Neural ODE dynamics
    """
    
    def __init__(
        self,
        recon,
        irecon,
        lorentz,
        beta,
        dip,
        tc,
        info,
        state_dim,
        hidden_dim,
        latent_dim,
        i_dim,
        lr,
        device,
        use_bottleneck_lorentz=True,
        loss_type='nb',
        grad_clip=1.0,
        use_layer_norm=True,
        use_euclidean_manifold=False,
        use_ode=False,
        vae_reg=0.5,
        ode_reg=0.5,
        **kwargs
    ):
        self.recon = recon
        self.irecon = irecon
        self.lorentz = lorentz
        self.beta = beta
        self.dip = dip
        self.tc = tc
        self.info = info
        self.loss_type = loss_type
        self.grad_clip = grad_clip
        self.use_euclidean_manifold = use_euclidean_manifold
        self.use_ode = use_ode
        self.vae_reg = vae_reg
        self.ode_reg = ode_reg
        self.device = device
        
        self.nn = VAE(
            state_dim,
            hidden_dim,
            latent_dim,
            i_dim,
            use_bottleneck_lorentz=use_bottleneck_lorentz,
            loss_type=loss_type,
            use_layer_norm=use_layer_norm,
            use_euclidean_manifold=use_euclidean_manifold,
            use_ode=use_ode
        ).to(device)
        
        self.nn_optimizer = optim.Adam(self.nn.parameters(), lr=lr)
        self.loss = []
    
    @torch.no_grad()
    def take_latent(self, state):
        """Extract latent representation (combined VAE + ODE if applicable)."""
        state = torch.tensor(state, dtype=torch.float32).to(self.device)
        
        if self.use_ode:
            q_z, q_m, q_s, n, t = self.nn.encoder(state)
            t = t.cpu()
            t_sorted, sort_idx, sort_idxr = np.unique(
                t, return_index=True, return_inverse=True
            )
            t_sorted = torch.tensor(t_sorted)
            q_z_sorted = q_z[sort_idx]
            z0 = q_z_sorted[0]
            q_z_ode = self.nn.solve_ode(self.nn.ode_solver, z0, t_sorted)
            q_z_ode = q_z_ode[sort_idxr]
            return (self.vae_reg * q_z + self.ode_reg * q_z_ode).cpu().numpy()
        else:
            q_z, _, _, _ = self.nn.encoder(state)
            return q_z.cpu().numpy()
    
    @torch.no_grad()
    def take_iembed(self, state):
        """Extract information bottleneck embedding."""
        state = torch.tensor(state, dtype=torch.float32).to(self.device)
        
        if self.use_ode:
            q_z, q_m, q_s, n, t = self.nn.encoder(state)
            t = t.cpu()
            t_sorted, sort_idx, sort_idxr = np.unique(
                t, return_index=True, return_inverse=True
            )
            t_sorted = torch.tensor(t_sorted)
            q_z_sorted = q_z[sort_idx]
            z0 = q_z_sorted[0]
            q_z_ode = self.nn.solve_ode(self.nn.ode_solver, z0, t_sorted)
            q_z_ode = q_z_ode[sort_idxr]
            
            le = self.nn.latent_encoder(q_z)
            le_ode = self.nn.latent_encoder(q_z_ode)
            return (self.vae_reg * le + self.ode_reg * le_ode).cpu().numpy()
        else:
            outputs = self.nn(state)
            le = outputs[4]  # Information bottleneck encoding
            return le.cpu().numpy()
    
    @torch.no_grad()
    def take_time(self, state):
        """Extract predicted time values (ODE mode only)."""
        if not self.use_ode:
            raise ValueError("take_time() is only available in ODE mode")
        
        state = torch.tensor(state, dtype=torch.float32).to(self.device)
        _, _, _, _, t = self.nn.encoder(state)
        return t.cpu().numpy()
    
    @torch.no_grad()
    def take_grad(self, state):
        """Extract ODE gradients (ODE mode only)."""
        if not self.use_ode:
            raise ValueError("take_grad() is only available in ODE mode")
        
        state = torch.tensor(state, dtype=torch.float32).to(self.device)
        q_z, q_m, q_s, n, t = self.nn.encoder(state)
        grads = self.nn.ode_solver(t, q_z.cpu()).numpy()
        return grads
    
    @torch.no_grad()
    def take_transition(self, state, top_k: int = 30):
        """Compute transition matrix based on ODE dynamics."""
        if not self.use_ode:
            raise ValueError("take_transition() is only available in ODE mode")
        
        state = torch.tensor(state, dtype=torch.float32).to(self.device)
        q_z, q_m, q_s, n, t = self.nn.encoder(state)
        grads = self.nn.ode_solver(t, q_z.cpu()).numpy()
        z_latent = q_z.cpu().numpy()
        z_future = z_latent + 1e-2 * grads
        distances = pairwise_distances(z_latent, z_future)
        sigma = np.median(distances)
        similarity = np.exp(-(distances**2) / (2 * sigma**2))
        transition_matrix = similarity / similarity.sum(axis=1, keepdims=True)

        def sparsify_transitions(trans_matrix, top_k=top_k):
            n_cells = trans_matrix.shape[0]
            sparse_trans = np.zeros_like(trans_matrix)
            for i in range(n_cells):
                top_indices = np.argsort(trans_matrix[i])[::-1][:top_k]
                sparse_trans[i, top_indices] = trans_matrix[i, top_indices]
                sparse_trans[i] /= sparse_trans[i].sum()
            return sparse_trans

        transition_matrix = sparsify_transitions(transition_matrix)
        return transition_matrix
    
    def _compute_reconstruction_loss(self, x_raw, pred_x, dropout_x):
        """Compute reconstruction loss based on likelihood type."""
        # Scale predictions by library size
        lib_size = torch.clamp(x_raw.sum(dim=-1, keepdim=True), min=1.0)
        pred_x = pred_x * lib_size
        
        if self.loss_type == 'nb':
            disp = torch.exp(self.nn.decoder.disp)
            return -self._log_nb(x_raw, pred_x, disp).sum(dim=-1).mean()
        elif self.loss_type == 'zinb':
            disp = torch.exp(self.nn.decoder.disp)
            return -self._log_zinb(x_raw, pred_x, disp, dropout_x).sum(dim=-1).mean()
        elif self.loss_type == 'poisson':
            return -self._log_poisson(x_raw, pred_x).sum(dim=-1).mean()
        elif self.loss_type == 'zip':
            return -self._log_zip(x_raw, pred_x, dropout_x).sum(dim=-1).mean()
        else:
            raise ValueError(f"Unknown loss_type: {self.loss_type}")
    
    def update(self, states_norm, states_raw):
        """Perform one gradient update."""
        states_norm = torch.tensor(states_norm, dtype=torch.float32).to(self.device)
        states_raw = torch.tensor(states_raw, dtype=torch.float32).to(self.device)
        
        # Validate inputs
        if torch.isnan(states_norm).any() or torch.isinf(states_norm).any():
            print("Warning: Invalid input data, skipping batch")
            return
        
        # Forward pass (different unpacking for ODE vs non-ODE)
        if self.use_ode:
            (q_z, q_m, q_s, pred_x, le, ld, pred_xl, z_manifold, ld_manifold,
             dropout_x, dropout_xl, q_z_ode, pred_x_ode, dropout_x_ode,
             x_sorted, t) = self.nn(states_norm)
            
            # ODE divergence loss (key ODE regularization)
            qz_div = F.mse_loss(q_z, q_z_ode, reduction="none").sum(-1).mean()
            
            # Reconstruction losses (both VAE and ODE paths)
            recon_loss = self.recon * self._compute_reconstruction_loss(
                x_sorted, pred_x, dropout_x
            )
            recon_loss += self.recon * self._compute_reconstruction_loss(
                x_sorted, pred_x_ode, dropout_x_ode
            )
            
            # Geometric regularization (only for non-ODE paths)
            geometric_loss = torch.tensor(0.0, device=self.device)
            if self.lorentz > 0:
                if not (torch.isnan(z_manifold).any() or torch.isnan(ld_manifold).any()):
                    if self.use_euclidean_manifold:
                        from .utils import euclidean_distance
                        dist = euclidean_distance(z_manifold, ld_manifold)
                    else:
                        dist = lorentz_distance(z_manifold, ld_manifold)
                    
                    if not torch.isnan(dist).any():
                        geometric_loss = self.lorentz * dist.mean()
            
            # Information bottleneck reconstruction (both paths)
            irecon_loss = torch.tensor(0.0, device=self.device)
            if self.irecon > 0:
                irecon_loss = self.irecon * self._compute_reconstruction_loss(
                    x_sorted, pred_xl, dropout_xl
                )
        
        else:
            # Non-ODE forward pass
            q_z, q_m, q_s, pred_x, le, ld, pred_xl, z_manifold, ld_manifold, dropout_x, dropout_xl = \
                self.nn(states_norm)
            
            qz_div = torch.tensor(0.0, device=self.device)
            
            # Reconstruction loss
            recon_loss = self.recon * self._compute_reconstruction_loss(
                states_raw, pred_x, dropout_x
            )
            
            # Geometric regularization
            geometric_loss = torch.tensor(0.0, device=self.device)
            if self.lorentz > 0:
                if not (torch.isnan(z_manifold).any() or torch.isnan(ld_manifold).any()):
                    if self.use_euclidean_manifold:
                        from .utils import euclidean_distance
                        dist = euclidean_distance(z_manifold, ld_manifold)
                    else:
                        dist = lorentz_distance(z_manifold, ld_manifold)
                    
                    if not torch.isnan(dist).any():
                        geometric_loss = self.lorentz * dist.mean()
            
            # Information bottleneck reconstruction
            irecon_loss = torch.tensor(0.0, device=self.device)
            if self.irecon > 0:
                irecon_loss = self.irecon * self._compute_reconstruction_loss(
                    states_raw, pred_xl, dropout_xl
                )
        
        if torch.isnan(q_m).any() or torch.isnan(q_s).any():
            print("Warning: NaN in encoder output, skipping batch")
            return
        
        # KL divergence (standard VAE)
        kl_div = self.beta * self._normal_kl(
            q_m, q_s, torch.zeros_like(q_m), torch.zeros_like(q_s)
        ).sum(dim=-1).mean()
        
        # Additional regularizations
        dip_loss = self.dip * self._dip_loss(q_m, q_s) if self.dip > 0 else torch.tensor(0.0, device=self.device)
        tc_loss = self.tc * self._betatc_compute_total_correlation(q_z, q_m, q_s) if self.tc > 0 else torch.tensor(0.0, device=self.device)
        mmd_loss = self.info * self._compute_mmd(q_z, torch.randn_like(q_z)) if self.info > 0 else torch.tensor(0.0, device=self.device)
        
        # Total loss (includes qz_div for ODE mode)
        total_loss = (
            recon_loss + irecon_loss + geometric_loss + qz_div + 
            kl_div + dip_loss + tc_loss + mmd_loss
        )
        
        if torch.isnan(total_loss) or torch.isinf(total_loss):
            print(f"Warning: Invalid loss - skipping batch")
            return
        
        # Backpropagation
        self.nn_optimizer.zero_grad()
        total_loss.backward()
        
        if self.grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(self.nn.parameters(), self.grad_clip)
        
        self.nn_optimizer.step()
        
        # Log losses (now includes qz_div)
        self.loss.append((
            total_loss.item(),
            recon_loss.item(),
            irecon_loss.item(),
            geometric_loss.item(),
            qz_div.item(),
            kl_div.item(),
            dip_loss.item(),
            tc_loss.item(),
            mmd_loss.item()
        ))

"""Neural network modules for Liora."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from typing import Optional
from torchdiffeq import odeint
from .utils import exp_map_at_origin


def weight_init(m):
    """Xavier initialization for linear layers."""
    if isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight)
        nn.init.constant_(m.bias, 0.01)


class Encoder(nn.Module):
    """Encoder network with optional layer normalization and ODE support."""
    
    def __init__(self, state_dim, hidden_dim, action_dim, use_layer_norm=True, use_ode=False):
        super().__init__()
        self.use_layer_norm = use_layer_norm
        self.use_ode = use_ode
        
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim * 2)
        
        if use_layer_norm:
            self.ln1 = nn.LayerNorm(hidden_dim)
            self.ln2 = nn.LayerNorm(hidden_dim)
        
        # Time encoder for ODE mode
        if use_ode:
            self.time_encoder = nn.Sequential(
                nn.Linear(hidden_dim, 1),
                nn.Sigmoid(),  # Time values in [0, 1]
            )
        
        self.apply(weight_init)

    def forward(self, x):
        h1 = F.relu(self.ln1(self.fc1(x)) if self.use_layer_norm else self.fc1(x))
        h2 = F.relu(self.ln2(self.fc2(h1)) if self.use_layer_norm else self.fc2(h1))
        output = self.fc3(h2)
        
        q_m, q_s = torch.chunk(output, 2, dim=-1)
        q_m = torch.clamp(q_m, -10, 10)
        q_s = torch.clamp(q_s, -10, 10)
        
        s = torch.clamp(F.softplus(q_s) + 1e-6, min=1e-6, max=5.0)
        n = Normal(q_m, s)
        q_z = n.rsample()
        
        if self.use_ode:
            t = self.time_encoder(h2).squeeze(-1)  # Shape: (batch_size,)
            return q_z, q_m, q_s, n, t
        
        return q_z, q_m, q_s, n


class Decoder(nn.Module):
    """Decoder network with optional layer normalization."""
    
    def __init__(self, state_dim, hidden_dim, action_dim, loss_type='nb', use_layer_norm=True):
        super().__init__()
        self.loss_type = loss_type
        self.use_layer_norm = use_layer_norm
        
        self.fc1 = nn.Linear(action_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, state_dim)
        
        if use_layer_norm:
            self.ln1 = nn.LayerNorm(hidden_dim)
            self.ln2 = nn.LayerNorm(hidden_dim)
        
        self.disp = nn.Parameter(torch.randn(state_dim))
        
        if loss_type in ['zinb', 'zip']:
            self.dropout = nn.Sequential(
                nn.Linear(action_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, state_dim)
            )
        
        self.apply(weight_init)

    def forward(self, x):
        h1 = F.relu(self.ln1(self.fc1(x)) if self.use_layer_norm else self.fc1(x))
        h2 = F.relu(self.ln2(self.fc2(h1)) if self.use_layer_norm else self.fc2(h1))
        output = F.softmax(self.fc3(h2), dim=-1)
        
        dropout = self.dropout(x) if self.loss_type in ['zinb', 'zip'] else None
        return output, dropout


class LatentODEfunc(nn.Module):
    """
    Latent space ODE function model.
    
    Parameters
    ----------
    n_latent : int
        Latent space dimension
    n_hidden : int
        Hidden layer dimension
    """
    
    def __init__(self, n_latent: int = 10, n_hidden: int = 25):
        super().__init__()
        self.elu = nn.ELU()
        self.fc1 = nn.Linear(n_latent, n_hidden)
        self.fc2 = nn.Linear(n_hidden, n_latent)
        
        self.apply(weight_init)

    def forward(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """
        Compute gradient at time t and state x.
        
        Parameters
        ----------
        t : torch.Tensor
            Time point
        x : torch.Tensor
            Latent state
            
        Returns
        -------
        torch.Tensor
            Gradient value
        """
        out = self.fc1(x)
        out = self.elu(out)
        out = self.fc2(out)
        return out


class NODEMixin:
    """Mixin class providing Neural ODE functionality."""
    
    @staticmethod
    def get_step_size(step_size, t0, t1, n_points):
        """Get ODE solver step size configuration."""
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
    ) -> torch.Tensor:
        """
        Solve ODE using torchdiffeq.
        
        Parameters
        ----------
        ode_func : nn.Module
            ODE function model
        z0 : torch.Tensor
            Initial state
        t : torch.Tensor
            Time points
        method : str
            Solution method
        step_size : Optional[float]
            Step size
            
        Returns
        -------
        torch.Tensor
            ODE solution
        """
        options = self.get_step_size(step_size, t[0], t[-1], len(t))
        
        # Move to CPU for ODE solver (if needed)
        cpu_z0 = z0.to("cpu")
        cpu_t = t.to("cpu")
        
        # Solve ODE
        pred_z = odeint(ode_func, cpu_z0, cpu_t, method=method, options=options)
        
        # Move back to original device
        pred_z = pred_z.to(z0.device)
        
        return pred_z


class VAE(nn.Module, NODEMixin):
    """Lorenzian Interpretable ODE Regularization variational Autoencoder"""
    
    def __init__(self, state_dim, hidden_dim, action_dim, i_dim,
                 use_bottleneck_lorentz=True, loss_type='nb', use_layer_norm=True, 
                 use_euclidean_manifold=False, use_ode=False):
        super().__init__()
        
        self.encoder = Encoder(state_dim, hidden_dim, action_dim, use_layer_norm, use_ode)
        self.decoder = Decoder(state_dim, hidden_dim, action_dim, loss_type, use_layer_norm)
        self.latent_encoder = nn.Linear(action_dim, i_dim)
        self.latent_decoder = nn.Linear(i_dim, action_dim)
        self.use_bottleneck_lorentz = use_bottleneck_lorentz
        self.use_euclidean_manifold = use_euclidean_manifold
        self.use_ode = use_ode
        
        # Initialize ODE solver if needed
        if use_ode:
            self.ode_solver = LatentODEfunc(action_dim, hidden_dim)
        
    def forward(self, x):
        # Encode
        if self.use_ode:
            q_z, q_m, q_s, n, t = self.encoder(x)
            
            # Sort by time
            idxs = torch.argsort(t)
            t = t[idxs]
            q_z = q_z[idxs]
            q_m = q_m[idxs]
            q_s = q_s[idxs]
            x = x[idxs]
            
            # Remove duplicate time points
            unique_mask = torch.ones_like(t, dtype=torch.bool)
            unique_mask[1:] = t[1:] != t[:-1]
            
            t = t[unique_mask]
            q_z = q_z[unique_mask]
            q_m = q_m[unique_mask]
            q_s = q_s[unique_mask]
            x = x[unique_mask]
            
            # Solve ODE
            z0 = q_z[0]
            q_z_ode = self.solve_ode(self.ode_solver, z0, t)
            
            # Primary path: q_z through bottleneck
            q_z_clipped = torch.clamp(q_z, -5, 5)
            
            if self.use_euclidean_manifold:
                z_manifold = q_z
            else:
                z_tangent = F.pad(q_z_clipped, (1, 0), value=0)
                z_manifold = exp_map_at_origin(z_tangent)
            
            # Information bottleneck (only for q_z, NOT for q_z_ode)
            le = self.latent_encoder(q_z)
            ld = self.latent_decoder(le)
            
            ld_clipped = torch.clamp(ld, -5, 5)
            
            if self.use_euclidean_manifold:
                ld_manifold = ld
            else:
                if self.use_bottleneck_lorentz:
                    ld_tangent = F.pad(ld_clipped, (1, 0), value=0)
                    ld_manifold = exp_map_at_origin(ld_tangent)
                else:
                    q_z2 = n.sample()
                    q_z2_clipped = torch.clamp(q_z2, -5, 5)
                    z2_tangent = F.pad(q_z2_clipped, (1, 0), value=0)
                    ld_manifold = exp_map_at_origin(z2_tangent)
            
            # Decode all paths
            pred_x, dropout_x = self.decoder(q_z)
            pred_xl, dropout_xl = self.decoder(ld)
            pred_x_ode, dropout_x_ode = self.decoder(q_z_ode)
            
            # Return with ODE outputs
            return (q_z, q_m, q_s, pred_x, le, ld, pred_xl, z_manifold, ld_manifold, 
                    dropout_x, dropout_xl, q_z_ode, pred_x_ode, dropout_x_ode, 
                    x, t)
        
        else:
            # Original non-ODE forward pass
            q_z, q_m, q_s, n = self.encoder(x)
            
            # Primary Lorentz embedding
            q_z_clipped = torch.clamp(q_z, -5, 5)
            
            # Conditional manifold mapping
            if self.use_euclidean_manifold:                    
                # Euclidean: use embeddings directly
                z_manifold = q_z
            else:
                # Lorentz: map to hyperboloid
                z_tangent = F.pad(q_z_clipped, (1, 0), value=0)
                z_manifold = exp_map_at_origin(z_tangent)
            
            # Information bottleneck
            le = self.latent_encoder(q_z)
            ld = self.latent_decoder(le)

            # Conditional manifold mapping for bottleneck path
            ld_clipped = torch.clamp(ld, -5, 5)
            
            if self.use_euclidean_manifold:                    
                # Euclidean: use embeddings directly
                ld_manifold = ld
            else:
                # Lorentz: map based on option
                if self.use_bottleneck_lorentz:
                    ld_tangent = F.pad(ld_clipped, (1, 0), value=0)
                    ld_manifold = exp_map_at_origin(ld_tangent)
                else:
                    q_z2 = n.sample()
                    q_z2_clipped = torch.clamp(q_z2, -5, 5)
                    z2_tangent = F.pad(q_z2_clipped, (1, 0), value=0)
                    ld_manifold = exp_map_at_origin(z2_tangent)
            
            pred_x, dropout_x = self.decoder(q_z)
            pred_xl, dropout_xl = self.decoder(ld)
            
            # Return manifold embeddings instead of z_lorentz/ld_lorentz
            return q_z, q_m, q_s, pred_x, le, ld, pred_xl, z_manifold, ld_manifold, dropout_x, dropout_xl

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
    