
"""LiVAE: Lorentz Information-regularized Variational AutoEncoder for single-cell transcriptomics."""

from .environment import Env
from anndata import AnnData
import torch
import tqdm


class LiVAE(Env):
    """
    LiVAE model for single-cell RNA-seq analysis.
    
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
    percent : float, default=0.01
        Fraction of data to use per batch
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
    device : torch.device, optional
        Device for training (defaults to CUDA if available)
    """
    
    def __init__(
        self,
        adata: AnnData,
        layer: str = 'counts',
        percent: float = 0.01,
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
        device: torch.device = None
    ):
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            
        super().__init__(
            adata=adata,
            layer=layer,
            percent=percent,
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
            device=device
        )
        
    def fit(self, epochs: int = 1000):
        """
        Train the model.
        
        Parameters
        ----------
        epochs : int, default=1000
            Number of training epochs
            
        Returns
        -------
        self : LiVAE
            Trained model
        """
        with tqdm.tqdm(total=epochs, desc='Training', ncols=140) as pbar:
            for i in range(epochs):
                data = self.load_data()
                self.step(data)
                
                if (i + 1) % 10 == 0:
                    pbar.set_postfix({
                        'Loss': f'{self.loss[-1][0]:.2f}',
                        'ARI': f'{self.score[-1][0]:.2f}',
                        'NMI': f'{self.score[-1][1]:.2f}',
                        'ASW': f'{self.score[-1][2]:.2f}',
                        'C_H': f'{self.score[-1][3]:.2f}',
                        'D_B': f'{self.score[-1][4]:.2f}',
                        'P_C': f'{self.score[-1][5]:.2f}'
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
            _, _, _, _, le, _, _, _, _, _, _ = self.nn(x)
        return le.cpu().numpy()
