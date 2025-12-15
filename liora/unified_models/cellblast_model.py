"""
Cell BLAST的PyTorch实现
基于DIRECTi的单细胞注释模型

Based on: https://github.com/gao-lab/Cell_BLAST
Reference: Cao et al. (2020) Searching large-scale scRNA-seq databases via unbiased cell embedding with Cell BLAST
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple
import numpy as np
from .base_model import BaseModel


class Encoder(nn.Module):
    """
    DIRECTi编码器网络
    
    Uses ELU activation (not ReLU) as in original implementation
    """
    def __init__(self, input_dim: int, hidden_dims: list, latent_dim: int, 
                 dropout: float = 0.0, use_bn: bool = True):
        super().__init__()
        
        # Build encoder layers with ELU activation (key difference from standard VAE)
        self.layers = nn.ModuleList()
        prev_dim = input_dim
        
        for i, hidden_dim in enumerate(hidden_dims):
            layer_modules = []
            layer_modules.append(nn.Linear(prev_dim, hidden_dim))
            
            if use_bn:
                layer_modules.append(nn.BatchNorm1d(hidden_dim))
            
            layer_modules.append(nn.ELU())  # Original uses ELU, not ReLU
            
            if dropout > 0:
                layer_modules.append(nn.Dropout(dropout))
            
            self.layers.append(nn.Sequential(*layer_modules))
            prev_dim = hidden_dim
        
        # Output layers for Gaussian latent
        self.fc_mu = nn.Linear(prev_dim, latent_dim)
        self.fc_logvar = nn.Linear(prev_dim, latent_dim)
    
    def forward(self, x):
        """
        Args:
            x: [batch, input_dim] log-normalized expression
        Returns:
            mu: [batch, latent_dim]
            logvar: [batch, latent_dim]
        """
        h = x
        for layer in self.layers:
            h = layer(h)
        
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        
        return mu, logvar


class Decoder(nn.Module):
    """
    DIRECTi解码器网络
    
    Outputs ZINB parameters for count data
    """
    def __init__(self, latent_dim: int, hidden_dims: list, output_dim: int,
                 dropout: float = 0.0, use_bn: bool = True,
                 output_distribution: str = 'zinb'):
        super().__init__()
        
        self.output_distribution = output_distribution
        
        # Build decoder layers
        self.layers = nn.ModuleList()
        prev_dim = latent_dim
        
        for hidden_dim in reversed(hidden_dims):
            layer_modules = []
            layer_modules.append(nn.Linear(prev_dim, hidden_dim))
            
            if use_bn:
                layer_modules.append(nn.BatchNorm1d(hidden_dim))
            
            layer_modules.append(nn.ELU())
            
            if dropout > 0:
                layer_modules.append(nn.Dropout(dropout))
            
            self.layers.append(nn.Sequential(*layer_modules))
            prev_dim = hidden_dim
        
        # ZINB output layers
        if output_distribution == 'zinb':
            # Mean parameter (normalized, will be scaled by library size)
            self.mean_decoder = nn.Sequential(
                nn.Linear(prev_dim, output_dim),
                nn.Softmax(dim=-1)
            )
            # Dispersion parameter (positive)
            self.disp_decoder = nn.Sequential(
                nn.Linear(prev_dim, output_dim),
                nn.Softplus()
            )
            # Dropout logit (will be passed through sigmoid for probability)
            self.dropout_decoder = nn.Linear(prev_dim, output_dim)
        elif output_distribution == 'nb':
            self.mean_decoder = nn.Sequential(
                nn.Linear(prev_dim, output_dim),
                nn.Softmax(dim=-1)
            )
            self.disp_decoder = nn.Sequential(
                nn.Linear(prev_dim, output_dim),
                nn.Softplus()
            )
        else:  # gaussian
            self.mean_decoder = nn.Linear(prev_dim, output_dim)
    
    def forward(self, z):
        """
        Args:
            z: [batch, latent_dim]
        Returns:
            Dictionary with distribution parameters
        """
        h = z
        for layer in self.layers:
            h = layer(h)
        
        if self.output_distribution == 'zinb':
            mean = self.mean_decoder(h)
            disp = self.disp_decoder(h)
            dropout_logit = self.dropout_decoder(h)
            return {
                'mean': mean,
                'disp': disp,
                'dropout_logit': dropout_logit
            }
        elif self.output_distribution == 'nb':
            mean = self.mean_decoder(h)
            disp = self.disp_decoder(h)
            return {'mean': mean, 'disp': disp}
        else:
            mean = self.mean_decoder(h)
            return {'mean': mean}


class BatchDiscriminator(nn.Module):
    """
    Batch discriminator for adversarial batch correction
    
    Part of DIRECTi's batch effect removal strategy
    """
    def __init__(self, latent_dim: int, n_batches: int, hidden_dim: int = 128):
        super().__init__()
        
        self.discriminator = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, n_batches)
        )
    
    def forward(self, z):
        """
        Args:
            z: [batch, latent_dim]
        Returns:
            [batch, n_batches] logits
        """
        return self.discriminator(z)


class CellBLASTModel(BaseModel):
    """
    Cell BLAST模型的完整PyTorch实现
    
    Features:
    - ZINB/NB reconstruction for count data
    - Adversarial batch correction
    - Probabilistic latent space for cell type annotation
    - ELU activation (not ReLU)
    
    Reference:
        Cao et al. (2020) Searching large-scale scRNA-seq databases via 
        unbiased cell embedding with Cell BLAST. Nature Communications.
    """
    
    def __init__(self,
                 input_dim: int,
                 latent_dim: int = 10,
                 hidden_dims: list = None,
                 dropout: float = 0.0,
                 use_bn: bool = True,
                 output_distribution: str = 'zinb',
                 use_batch_correction: bool = False,
                 n_batches: int = 1,
                 adversarial_weight: float = 1.0,
                 model_name: str = "CellBLAST"):
        """
        Args:
            input_dim: Input feature dimension (number of genes)
            latent_dim: Latent space dimension
            hidden_dims: Hidden layer dimensions
            dropout: Dropout rate
            use_bn: Use batch normalization
            output_distribution: 'zinb', 'nb', or 'gaussian'
            use_batch_correction: Use adversarial batch correction
            n_batches: Number of batches (for batch correction)
            adversarial_weight: Weight for adversarial loss
            model_name: Model name
        """
        if hidden_dims is None:
            hidden_dims = [512, 256, 128]
        
        super().__init__(input_dim, latent_dim, hidden_dims, model_name)
        
        self.dropout = dropout
        self.use_bn = use_bn
        self.output_distribution = output_distribution
        self.use_batch_correction = use_batch_correction
        self.n_batches = n_batches
        self.adversarial_weight = adversarial_weight
        
        # Build encoder and decoder
        self.encoder_net = Encoder(
            input_dim, hidden_dims, latent_dim, dropout, use_bn
        )
        
        self.decoder_net = Decoder(
            latent_dim, hidden_dims, input_dim, dropout, use_bn, output_distribution
        )
        
        # Batch discriminator for adversarial training
        if use_batch_correction and n_batches > 1:
            self.batch_discriminator = BatchDiscriminator(latent_dim, n_batches)
        else:
            self.batch_discriminator = None
    
    def _prepare_batch(self, batch_data, device):
        """Prepare batch data and extract metadata"""
        if isinstance(batch_data, (list, tuple)):
            x = batch_data[0].to(device).float()
            metadata = {}
            
            if len(batch_data) >= 2:
                second_item = batch_data[1]
                if second_item.dtype in [torch.long, torch.int32, torch.int64]:
                    if self.use_batch_correction and second_item.max() < self.n_batches:
                        metadata['batch_id'] = second_item.to(device).long()
            
            # Set default if not provided
            if self.use_batch_correction and 'batch_id' not in metadata:
                metadata['batch_id'] = torch.zeros(x.size(0), dtype=torch.long, device=device)
            
            return x, metadata
        else:
            x = batch_data.to(device).float()
            metadata = {}
            if self.use_batch_correction:
                metadata['batch_id'] = torch.zeros(x.size(0), dtype=torch.long, device=device)
            return x, metadata
    
    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Reparameterization trick"""
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        else:
            return mu
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode to latent space"""
        mu, logvar = self.encoder_net(x)
        z = self.reparameterize(mu, logvar)
        return z
    
    def decode(self, z: torch.Tensor):
        """Decode from latent space"""
        decoder_output = self.decoder_net(z)
        if self.output_distribution in ['zinb', 'nb']:
            return decoder_output['mean']
        else:
            return decoder_output['mean']
    
    def forward(self, x: torch.Tensor, batch_id: Optional[torch.Tensor] = None,
                **kwargs) -> Dict[str, torch.Tensor]:
        """
        Forward pass
        
        Args:
            x: Input data [batch_size, input_dim] (raw counts or log-normalized)
            batch_id: Batch IDs [batch_size] (optional)
            
        Returns:
            Dictionary with outputs
        """
        # Compute library size (total count per cell)
        library_size = x.sum(dim=1, keepdim=True)
        
        # Encode
        mu, logvar = self.encoder_net(x)
        z = self.reparameterize(mu, logvar)
        
        # Decode
        decoder_output = self.decoder_net(z)
        
        # Batch discrimination (for adversarial training)
        if self.batch_discriminator is not None and batch_id is not None:
            batch_logits = self.batch_discriminator(z)
        else:
            batch_logits = None
        
        output = {
            'latent': z,
            'mu': mu,
            'logvar': logvar,
            'library_size': library_size,
            'batch_logits': batch_logits
        }
        
        # Add decoder outputs
        output.update(decoder_output)
        
        return output
    
    def _zinb_loss(self, x: torch.Tensor, mean: torch.Tensor,
                   disp: torch.Tensor, dropout_logit: torch.Tensor,
                   library_size: torch.Tensor) -> torch.Tensor:
        """
        Zero-Inflated Negative Binomial loss
        
        Args:
            x: True counts [batch, genes]
            mean: Normalized mean [batch, genes]
            disp: Dispersion [batch, genes]
            dropout_logit: Dropout logit [batch, genes]
            library_size: Library size [batch, 1]
        """
        eps = 1e-10
        mean_scaled = mean * library_size
        
        # Dropout probability
        pi = torch.sigmoid(dropout_logit)
        
        # NB log likelihood
        t1 = torch.lgamma(disp + eps) + torch.lgamma(x + 1.0) - torch.lgamma(x + disp + eps)
        t2 = (disp + x) * torch.log(1.0 + (mean_scaled / (disp + eps))) + \
             (x * (torch.log(disp + eps) - torch.log(mean_scaled + eps)))
        nb_log_likelihood = -(t1 + t2)
        
        # Zero-inflation
        zero_nb = torch.pow(disp / (disp + mean_scaled + eps), disp)
        zero_case_log_prob = torch.log(pi + (1.0 - pi) * zero_nb + eps)
        non_zero_case_log_prob = torch.log(1.0 - pi + eps) + nb_log_likelihood
        
        # Combine
        loss = torch.where(
            x < 1e-8,
            -zero_case_log_prob,
            -non_zero_case_log_prob
        )
        
        return loss.mean()
    
    def _nb_loss(self, x: torch.Tensor, mean: torch.Tensor,
                 disp: torch.Tensor, library_size: torch.Tensor) -> torch.Tensor:
        """Negative Binomial loss"""
        eps = 1e-10
        mean_scaled = mean * library_size
        
        t1 = torch.lgamma(disp + eps) + torch.lgamma(x + 1.0) - torch.lgamma(x + disp + eps)
        t2 = (disp + x) * torch.log(1.0 + (mean_scaled / (disp + eps))) + \
             (x * (torch.log(disp + eps) - torch.log(mean_scaled + eps)))
        
        return (t1 + t2).mean()
    
    def compute_loss(self, x: torch.Tensor, outputs: Dict[str, torch.Tensor],
                     beta: float = 1.0,
                     batch_id: Optional[torch.Tensor] = None,
                     **kwargs) -> Dict[str, torch.Tensor]:
        """
        Compute loss
        
        Args:
            x: Input data (raw counts for zinb/nb)
            outputs: forward() outputs
            beta: KL weight
            batch_id: Batch IDs (for adversarial loss)
            
        Returns:
            Loss dictionary
        """
        mu = outputs['mu']
        logvar = outputs['logvar']
        library_size = outputs['library_size']
        
        # KL divergence
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / x.size(0)
        kl_loss = torch.clamp(kl_loss, min=0.0)
        
        # Reconstruction loss
        if self.output_distribution == 'zinb':
            recon_loss = self._zinb_loss(
                x, outputs['mean'], outputs['disp'], 
                outputs['dropout_logit'], library_size
            )
        elif self.output_distribution == 'nb':
            recon_loss = self._nb_loss(
                x, outputs['mean'], outputs['disp'], library_size
            )
        else:  # gaussian
            recon_loss = F.mse_loss(outputs['mean'], x, reduction='mean')
        
        # Adversarial batch correction loss
        if self.batch_discriminator is not None and batch_id is not None and outputs['batch_logits'] is not None:
            # Discriminator tries to predict batch
            batch_disc_loss = F.cross_entropy(outputs['batch_logits'], batch_id)
            
            # Encoder tries to fool discriminator (gradient reversal)
            # In practice, we minimize negative cross-entropy for encoder
            batch_adv_loss = -batch_disc_loss
        else:
            batch_disc_loss = torch.tensor(0.0, device=x.device)
            batch_adv_loss = torch.tensor(0.0, device=x.device)
        
        # Total loss (for encoder/decoder)
        total_loss = recon_loss + beta * kl_loss + self.adversarial_weight * batch_adv_loss
        
        return {
            'total_loss': total_loss,
            'recon_loss': recon_loss,
            'kl_loss': kl_loss,
            'batch_disc_loss': batch_disc_loss,
            'batch_adv_loss': batch_adv_loss
        }
    
    def compute_posterior_distance(self, z1: torch.Tensor, z2: torch.Tensor,
                                  mu1: torch.Tensor, mu2: torch.Tensor,
                                  logvar1: torch.Tensor, logvar2: torch.Tensor) -> torch.Tensor:
        """
        Compute posterior distance between cells (for Cell BLAST annotation)
        
        Uses KL divergence between posterior distributions:
        KL(q(z|x1) || q(z|x2))
        
        Args:
            z1, z2: Sampled latent representations
            mu1, mu2: Posterior means
            logvar1, logvar2: Posterior log variances
        
        Returns:
            [batch1, batch2] pairwise KL divergences
        """
        # Expand for pairwise comparison
        mu1 = mu1.unsqueeze(1)  # [batch1, 1, latent_dim]
        mu2 = mu2.unsqueeze(0)  # [1, batch2, latent_dim]
        logvar1 = logvar1.unsqueeze(1)
        logvar2 = logvar2.unsqueeze(0)
        
        # KL divergence: KL(N(mu1, var1) || N(mu2, var2))
        var1 = torch.exp(logvar1)
        var2 = torch.exp(logvar2)
        
        kl = 0.5 * (
            logvar2 - logvar1 +
            (var1 + (mu1 - mu2).pow(2)) / (var2 + 1e-10) - 1
        )
        
        # Sum over latent dimensions
        kl = kl.sum(dim=-1)
        
        return kl
    
    def extract_latent(self, data_loader, device='cuda', return_reconstructions=False):
        """Extract latent representations"""
        self.eval()
        self.to(device)
        
        latents = []
        mus = []
        logvars = []
        reconstructions = [] if return_reconstructions else None
        
        with torch.no_grad():
            for batch_data in data_loader:
                x, metadata = self._prepare_batch(batch_data, device)
                
                # Encode
                mu, logvar = self.encoder_net(x)
                z = mu  # Use mean for deterministic representation
                
                latents.append(z.cpu().numpy())
                mus.append(mu.cpu().numpy())
                logvars.append(logvar.cpu().numpy())
                
                # Reconstruct if needed
                if return_reconstructions:
                    recon = self.decode(z)
                    reconstructions.append(recon.cpu().numpy())
        
        result = {
            'latent': np.concatenate(latents, axis=0),
            'mu': np.concatenate(mus, axis=0),
            'logvar': np.concatenate(logvars, axis=0)
        }
        
        if return_reconstructions:
            result['reconstruction'] = np.concatenate(reconstructions, axis=0)
        
        return result


def create_cellblast_model(input_dim: int, latent_dim: int = 10, **kwargs):
    """
    Create Cell BLAST model
    
    Args:
        input_dim: Number of genes
        latent_dim: Latent dimension
        **kwargs: Additional arguments
    
    Examples:
        >>> # Basic Cell BLAST
        >>> model = create_cellblast_model(2000, latent_dim=10)
        
        >>> # With ZINB reconstruction
        >>> model = create_cellblast_model(2000, latent_dim=10, output_distribution='zinb')
        
        >>> # With batch correction
        >>> model = create_cellblast_model(2000, latent_dim=10, 
        ...                               use_batch_correction=True, n_batches=5)
    """
    return CellBLASTModel(input_dim=input_dim, latent_dim=latent_dim, **kwargs)