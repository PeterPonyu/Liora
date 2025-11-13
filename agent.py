
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
