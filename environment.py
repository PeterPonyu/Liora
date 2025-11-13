
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
