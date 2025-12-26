"""
HSDE: Hyperbolic SDE-Regularized VAE for Single-Cell Omics Analysis
====================================================================

A unified deep learning framework for single-cell omics analysis combining:
- Variational Autoencoder (VAE) for dimensionality reduction
- Lorentz geometric regularization for hierarchical structure
- Dual-path information bottleneck for coordinated biological programs
- Neural SDE regularization for trajectory inference
- Transformer-based attention mechanisms for long-range dependencies
- Multiple count-based likelihood functions (NB, ZINB, Poisson, ZIP)

Supports both scRNA-seq and scATAC-seq modalities without architectural modification.
"""

# ============================================================================
# agent.py - Main User Interface
# ============================================================================
from .mixin import VectorFieldMixin
from .environment import Env
from anndata import AnnData
from typing import Optional
import torch
import tqdm
import time

class HSDE(Env, VectorFieldMixin):
    """
    HSDE: Hyperbolic SDE-Regularized VAE
    
    A unified framework for single-cell omics analysis (scRNA-seq and scATAC-seq) that 
    learns low-dimensional representations while preserving both local cell-state structure 
    and global hierarchical organization through Lorentz geometric regularization, 
    information bottleneck architecture, and neural SDE-based trajectory inference.
    
    Architecture Overview
    ---------------------
    1. **Encoder**: Maps high-dimensional gene expression to latent space
    2. **Information Bottleneck**: Optional compression layer for hierarchical features
    3. **Manifold Regularization**: Lorentz or Euclidean distance constraints
    4. **Decoder**: Reconstructs gene expression with count-appropriate likelihoods
    5. **SDE Solver** (optional): Models stochastic cell state trajectories
    
    Parameters
    ----------
    adata : AnnData
        Annotated data matrix with raw count data in `adata.layers[layer]`
    layer : str, default='counts'
        Layer name containing raw unnormalized count data
    recon : float, default=1.0
        Weight for reconstruction loss (primary objective)
    irecon : float, default=0.0
        Weight for information bottleneck reconstruction loss
    lorentz : float, default=0.0
        Weight for geometric manifold regularization
    beta : float, default=1.0
        Weight for KL divergence (β-VAE); >1 encourages disentanglement
    dip : float, default=0.0
        Weight for Disentangled Inferred Prior (DIP-VAE) loss
    tc : float, default=0.0
        Weight for Total Correlation (β-TC-VAE) loss
    info : float, default=0.0
        Weight for Maximum Mean Discrepancy (InfoVAE) loss
    hidden_dim : int, default=128
        Hidden layer dimension in encoder/decoder
    latent_dim : int, default=10
        Primary latent space dimensionality
    i_dim : int, default=2
        Information bottleneck dimension (should be < latent_dim)
    lr : float, default=1e-4
        Learning rate for Adam optimizer
    use_bottleneck_lorentz : bool, default=True
        If True, compute manifold distance on bottleneck; else on resampled latents
    loss_type : str, default='nb'
        Count likelihood model:
        - 'nb': Negative Binomial (recommended for UMI data)
        - 'zinb': Zero-Inflated Negative Binomial (high dropout)
        - 'poisson': Poisson (simple baseline)
        - 'zip': Zero-Inflated Poisson
    grad_clip : float, default=1.0
        Gradient clipping threshold for training stability
    adaptive_norm : bool, default=True
        Use dataset-specific normalization heuristics
    use_layer_norm : bool, default=True
        Apply layer normalization in encoder/decoder
    use_euclidean_manifold : bool, default=False
        Use Euclidean distance instead of Lorentz (hyperbolic) distance
    use_sde : bool, default=False
        Enable Neural SDE regularization for trajectory inference
    vae_reg : float, default=0.5
        Weight for VAE path in SDE mode (should sum to 1.0 with sde_reg)
    sde_reg : float, default=0.5
        Weight for SDE path in SDE mode
    train_size : float, default=0.7
        Proportion of cells for training set
    val_size : float, default=0.15
        Proportion of cells for validation set
    test_size : float, default=0.15
        Proportion of cells for test set (should sum to 1.0)
    batch_size : int, default=128
        Mini-batch size for stochastic gradient descent
    random_seed : int, default=42
        Random seed for reproducibility
    device : torch.device, optional
        Computation device (auto-detects CUDA if available)
    encoder_type : str, default='mlp'
        Encoder backbone: 'mlp' (default) or 'transformer'
    attn_embed_dim : int, default=64
        Embedding dimension for transformer-based encoder
    attn_num_heads : int, default=4
        Number of attention heads for transformer encoder
    attn_num_layers : int, default=2
        Number of transformer encoder layers
    attn_seq_len : int, default=32
        Pseudo-sequence length used in transformer encoder
    sde_type : str, default='time_mlp'
        SDE function type: 'time_mlp' or 'gru'
    sde_time_cond : str, default='concat'
        Time conditioning strategy for 'time_mlp' SDE: 'concat', 'film', or 'add'
    sde_hidden_dim : int, optional
        Hidden dimension for SDE function (defaults to hidden_dim when None)
    sde_solver_method : str, default='euler'
        SDE solver method: 'euler', 'milstein', 'heun', 'srk'
    sde_step_size : float, optional
        Step size for SDE solver; use None to auto-compute
    sde_rtol : float, optional
        Relative tolerance for adaptive solvers
    sde_atol : float, optional
        Absolute tolerance for adaptive solvers
    sde_sde_type : str, default='ito'
        SDE calculus type: 'ito' (default) or 'stratonovich'
    
    Attributes
    ----------
    nn : VAE
        The neural network model
    train_losses : list
        Training loss history
    val_losses : list
        Validation loss history
    best_val_loss : float
        Best validation loss achieved (for early stopping)
    
    Examples
    --------
    >>> import scanpy as sc
    >>> from liora import HSDE
    >>> 
    >>> # Load data
    >>> adata = sc.read_h5ad('data.h5ad')
    >>> 
    >>> # Basic usage with default parameters
    >>> model = HSDE(adata, layer='counts')
    >>> model.fit(epochs=100)
    >>> latent = model.get_latent()
    >>> 
    >>> # Advanced: with manifold regularization and SDE
    >>> model = HSDE(
    ...     adata, 
    ...     lorentz=5.0,       # Enable Lorentz regularization
    ...     use_sde=True,      # Enable trajectory inference
    ...     latent_dim=10,
    ...     i_dim=2
    ... )
    >>> model.fit(epochs=400, patience=25)
    >>> 
    >>> # Get embeddings and trajectories
    >>> latent = model.get_latent()
    >>> bottleneck = model.get_bottleneck()
    >>> pseudotime = model.get_time()        # SDE mode only
    >>> transitions = model.get_transition() # SDE mode only
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
        use_sde: bool = False,
        vae_reg: float = 0.5,
        sde_reg: float = 0.5,
        train_size: float = 0.7,
        val_size: float = 0.15,
        test_size: float = 0.15,
        batch_size: int = 128,
        random_seed: int = 42,
        device: torch.device = None,
        encoder_type: str = 'mlp',
        attn_embed_dim: int = 64,
        attn_num_heads: int = 4,
        attn_num_layers: int = 2,
        attn_seq_len: int = 32,
        sde_type: str = 'time_mlp',
        sde_time_cond: str = 'concat',
        sde_hidden_dim: Optional[int] = None,
        sde_solver_method: str = 'euler',
        sde_step_size: Optional[float] = None,
        sde_rtol: Optional[float] = None,
        sde_atol: Optional[float] = None,
        sde_sde_type: str = 'ito',
    ):
        # Auto-detect device
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Validate parameters
        if not (0.99 <= train_size + val_size + test_size <= 1.01):
            raise ValueError(f"Split sizes must sum to 1.0, got {train_size + val_size + test_size}")
        
        if use_sde and not (0.99 <= vae_reg + sde_reg <= 1.01):
            raise ValueError(f"SDE weights must sum to 1.0, got {vae_reg + sde_reg}")
        
        if i_dim >= latent_dim:
            raise ValueError(f"Information bottleneck dimension ({i_dim}) must be < latent dimension ({latent_dim})")
        
        # Initialize parent environment
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
            use_sde=use_sde,
            vae_reg=vae_reg,
            sde_reg=sde_reg,
            train_size=train_size,
            val_size=val_size,
            test_size=test_size,
            batch_size=batch_size,
            random_seed=random_seed,
            device=device,
            sde_type=sde_type,
            sde_time_cond=sde_time_cond,
            sde_hidden_dim=sde_hidden_dim,
            sde_solver_method=sde_solver_method,
            sde_step_size=sde_step_size,
            sde_rtol=sde_rtol,
            sde_atol=sde_atol,
            sde_sde_type=sde_sde_type,
            encoder_type=encoder_type,
            attn_embed_dim=attn_embed_dim,
            attn_num_heads=attn_num_heads,
            attn_num_layers=attn_num_layers,
            attn_seq_len=attn_seq_len
        )
        
        # Resource tracking
        self.train_time = 0.0
        self.peak_memory_gb = 0.0
        self.actual_epochs = 0
    
    def fit(
        self, 
        epochs: int = 400,
        patience: int = 25,
        val_every: int = 5,
        early_stop: bool = True,
    ):
        """
        Train the HSDE model with mini-batch gradient descent.
        
        Training uses:
        - Adam optimizer with configurable learning rate
        - Early stopping based on validation loss
        - Gradient clipping for stability
        - Periodic validation every `val_every` epochs
        
        Parameters
        ----------
        epochs : int, default=400
            Maximum number of training epochs
        patience : int, default=25
            Early stopping patience (epochs without validation improvement)
        val_every : int, default=5
            Validation frequency (every N epochs)
        early_stop : bool, default=True
            Enable early stopping mechanism
        
        Returns
        -------
        self : HSDE
            Returns self for method chaining
        """
        use_cuda = torch.cuda.is_available()
        if use_cuda:
            torch.cuda.reset_peak_memory_stats()
        start_time = time.time()

        with tqdm.tqdm(total=epochs, desc="Training", ncols=200) as pbar:
            for epoch in range(epochs):
                # Train for one complete epoch
                train_loss = self.train_epoch()
                
                # Periodic validation
                if (epoch + 1) % val_every == 0 or epoch == 0:
                    val_loss, val_score = self.validate()
                    
                    if early_stop:
                        should_stop, improved = self.check_early_stopping(
                            val_loss, patience
                        )
                        
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
                            "Imp": "✓" if improved else "✗"
                        })
                        
                        if should_stop:
                            self.actual_epochs = epoch + 1
                            print(f"\n\nEarly stopping at epoch {epoch + 1}")
                            print(f"Best validation loss: {self.best_val_loss:.4f}")
                            self.load_best_model()
                            break
                    else:
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
            else:
                self.actual_epochs = epochs
                
        # Record resource usage
        self.train_time = time.time() - start_time
        self.peak_memory_gb = torch.cuda.max_memory_allocated() / 1e9 if use_cuda else 0.0
        return self
    
    def get_latent(self):
        """
        Extract latent representations for all cells.
        
        In standard mode, returns encoder output.
        In SDE mode, returns weighted combination of VAE and SDE paths.
        
        Returns
        -------
        latent : ndarray of shape (n_cells, latent_dim)
            Low-dimensional cell embeddings
        """
        return self.take_latent(self.X_norm)
    
    def get_time(self):
        """
        Extract pseudotime for all cells (SDE mode only).
        
        Returns
        -------
        pseudotime : ndarray of shape (n_cells,)
            Predicted cell pseudotime values
        """
        return self.take_time(self.X_norm)
    
    def get_transition(self):
        """
        Extract transition probabilities for all cells (SDE mode only).
        
        Returns
        -------
        transition : ndarray of shape (n_cells, n_cells)
            Cell-to-cell transition probability matrix
        """
        return self.take_transition(self.X_norm)
    
    def get_test_latent(self):
        """
        Extract latent representations for test set only.
        
        Returns
        -------
        latent : ndarray of shape (n_test, latent_dim)
            Low-dimensional test cell embeddings
        """
        return self.take_latent(self.X_test_norm)
    
    def get_bottleneck(self):
        """
        Extract information bottleneck representations.
        
        The bottleneck layer compresses latent features to a lower-dimensional
        space (i_dim), capturing the most essential information for reconstruction.
        
        Returns
        -------
        bottleneck : ndarray of shape (n_cells, i_dim)
            Compressed hierarchical representations
        """
        x = torch.tensor(self.X_norm, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            outputs = self.nn(x)
            le = outputs[4]  # Information bottleneck encoding
        return le.cpu().numpy()

    def get_resource_metrics(self):
        """
        Get training resource usage metrics.
        
        Returns
        -------
        metrics : dict
            Dictionary with 'train_time', 'peak_memory_gb', 'actual_epochs'
        """
        return {
            'train_time': self.train_time,
            'peak_memory_gb': self.peak_memory_gb,
            'actual_epochs': self.actual_epochs
        }