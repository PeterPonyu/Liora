"""
Liora: Lorentz Information ODE Regularized Variational AutoEncoder
==================================================================

A deep learning framework for single-cell RNA-seq analysis combining:
- Variational Autoencoder (VAE) for dimensionality reduction
- Lorentz/Euclidean manifold regularization for geometric structure
- Information bottleneck for hierarchical representation learning
- Neural ODE for trajectory inference (optional)
- Multiple count-based likelihood functions (NB, ZINB, Poisson, ZIP)
"""

# ============================================================================
# agent.py - Main User Interface
# ============================================================================

from .environment import Env
from anndata import AnnData
import torch
import tqdm


class Liora(Env):
    """
    Liora: Lorentz Information ODE Regularized Variational AutoEncoder
    
    A unified framework for single-cell RNA-seq analysis that learns low-dimensional
    representations while preserving geometric and topological structure through
    manifold regularization and optional temporal dynamics via Neural ODEs.
    
    Architecture Overview
    ---------------------
    1. **Encoder**: Maps high-dimensional gene expression to latent space
    2. **Information Bottleneck**: Optional compression layer for hierarchical features
    3. **Manifold Regularization**: Lorentz or Euclidean distance constraints
    4. **Decoder**: Reconstructs gene expression with count-appropriate likelihoods
    5. **ODE Solver** (optional): Models continuous cell state trajectories
    
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
    use_ode : bool, default=False
        Enable Neural ODE regularization for trajectory inference
    vae_reg : float, default=0.5
        Weight for VAE path in ODE mode (should sum to 1.0 with ode_reg)
    ode_reg : float, default=0.5
        Weight for ODE path in ODE mode
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
    >>> from liora import Liora
    >>> 
    >>> # Load data
    >>> adata = sc.read_h5ad('data.h5ad')
    >>> 
    >>> # Basic usage with default parameters
    >>> model = Liora(adata, layer='counts')
    >>> model.fit(epochs=100)
    >>> latent = model.get_latent()
    >>> 
    >>> # Advanced: with manifold regularization and ODE
    >>> model = Liora(
    ...     adata, 
    ...     lorentz=5.0,      # Enable Lorentz regularization
    ...     use_ode=True,     # Enable trajectory inference
    ...     latent_dim=10,
    ...     i_dim=2
    ... )
    >>> model.fit(epochs=400, patience=25)
    >>> 
    >>> # Get embeddings and trajectories
    >>> latent = model.get_latent()
    >>> bottleneck = model.get_bottleneck()
    >>> pseudotime = model.get_time() # ODE mode only
    >>> transitions = model.get_transition()  # ODE mode only
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
        device : torch.device = None,
        # Encoder selection and attention hyperparameters
        encoder_type: str = 'mlp',
        attn_embed_dim: int = 64,
        attn_num_heads: int = 4,
        attn_num_layers: int = 2,
        attn_seq_len: int = 32,
        # ODE function and solver parameters
        ode_type: str = 'time_mlp',
        ode_time_cond: str = 'concat',
        ode_hidden_dim: int | None = None,
        ode_solver_method: str = 'rk4',
        ode_step_size: float | None = None,
        ode_rtol: float | None = None,
        ode_atol: float | None = None,
    ):
        # Auto-detect device
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Validate parameters
        if not (0.99 <= train_size + val_size + test_size <= 1.01):
            raise ValueError(f"Split sizes must sum to 1.0, got {train_size + val_size + test_size}")
        
        if use_ode and not (0.99 <= vae_reg + ode_reg <= 1.01):
            raise ValueError(f"ODE weights must sum to 1.0, got {vae_reg + ode_reg}")
        
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
            use_ode=use_ode,
            vae_reg=vae_reg,
            ode_reg=ode_reg,
            train_size=train_size,
            val_size=val_size,
            test_size=test_size,
            batch_size=batch_size,
            random_seed=random_seed,
            device=device,
            ode_type=ode_type,
            ode_time_cond=ode_time_cond,
            ode_hidden_dim=ode_hidden_dim,
            ode_solver_method=ode_solver_method,
            ode_step_size=ode_step_size,
            ode_rtol=ode_rtol,
            ode_atol=ode_atol,
            encoder_type=encoder_type,
            attn_embed_dim=attn_embed_dim,
            attn_num_heads=attn_num_heads,
            attn_num_layers=attn_num_layers,
            attn_seq_len=attn_seq_len
        )
    
    def fit(
        self, 
        epochs: int = 400,
        patience: int = 25,
        val_every: int = 5,
        early_stop: bool = True,
    ):
        """
        Train the Liora model with mini-batch gradient descent.
        
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
        self : Liora
            Returns self for method chaining
        
        Notes
        -----
        Progress bar displays:
        - Train: Training loss
        - Val: Validation loss
        - ARI: Adjusted Rand Index
        - NMI: Normalized Mutual Information
        - ASW: Average Silhouette Width
        - CAL: Calinski-Harabasz score
        - DAV: Davies-Bouldin score
        - COR: Average correlation per dimension
        - Best: Best validation loss seen
        - Pat: Patience counter
        """
        with tqdm.tqdm(total=epochs, desc="Training", ncols=200) as pbar:
            for epoch in range(epochs):
                # Train for one complete epoch
                train_loss = self.train_epoch()
                
                # Periodic validation
                if (epoch + 1) % val_every == 0 or epoch == 0:
                    val_loss, val_score = self.validate()
                    
                    if early_stop:
                        # Check if we should stop early
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
                            "Imp": "✓" if improved else "✗"
                        })
                        
                        if should_stop:
                            print(f"\n\nEarly stopping at epoch {epoch + 1}")
                            print(f"Best validation loss: {self.best_val_loss:.4f}")
                            self.load_best_model()
                            break
                    else:
                        # No early stopping - just display metrics
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
        Extract latent representations for all cells.
        
        In standard mode, returns encoder output.
        In ODE mode, returns weighted combination of VAE and ODE paths.
        
        Returns
        -------
        latent : ndarray of shape (n_cells, latent_dim)
            Low-dimensional cell embeddings
        """
        return self.take_latent(self.X_norm)
    
    def get_time(self):
        """
        Extract pseudotime for all cells.
        """
        return self.take_time(self.X_norm)
    
    def get_transition(self):
        """
        Extract transition probabilities for all cells.
        """
        return self.take_transition(self.X_norm)
    
    def get_test_latent(self):
        """
        Extract latent representations for test set only.
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