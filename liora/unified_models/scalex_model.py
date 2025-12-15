"""
SCALEX的PyTorch实现
用于单细胞数据跨批次整合的VAE模型

Based on: https://github.com/jsxlei/SCALEX
Reference: Xiong et al. (2021) Online single-cell data integration through projecting heterogeneous datasets into a common cell-embedding space
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Union, Tuple
import numpy as np
from .base_model import BaseModel


class DSBatchNorm(nn.Module):
    """
    Domain-Specific Batch Normalization
    
    Key component of SCALEX for handling multiple batches/domains.
    Each domain has its own batch norm parameters (gamma, beta).
    """
    def __init__(self, num_features: int, n_domains: int = 1, eps: float = 1e-5, momentum: float = 0.1):
        super().__init__()
        self.n_domains = n_domains
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        
        # Domain-specific parameters
        self.bns = nn.ModuleList([
            nn.BatchNorm1d(num_features, eps=eps, momentum=momentum)
            for _ in range(n_domains)
        ])
    
    def forward(self, x: torch.Tensor, domain_id: Union[int, torch.Tensor]) -> torch.Tensor:
        """
        Args:
            x: [batch_size, num_features]
            domain_id: scalar or [batch_size] tensor
        """
        if self.n_domains == 1:
            return self.bns[0](x)
        
        # Handle scalar domain_id
        if isinstance(domain_id, int) or (isinstance(domain_id, torch.Tensor) and domain_id.dim() == 0):
            domain_id = int(domain_id) if isinstance(domain_id, torch.Tensor) else domain_id
            return self.bns[domain_id](x)
        
        # Handle vector domain_id (different domains in same batch)
        output = torch.zeros_like(x)
        for i in range(self.n_domains):
            mask = (domain_id == i)
            if mask.any():
                output[mask] = self.bns[i](x[mask])
        
        return output


class SCALEXEncoder(nn.Module):
    """SCALEX编码器（完全按照原始实现）"""
    def __init__(self, input_dim: int, hidden_dims: list, latent_dim: int, 
                 n_domains: int = 1, use_bn: bool = True):
        super().__init__()
        
        self.n_domains = n_domains
        self.use_bn = use_bn
        
        # Build encoder layers with DSBatchNorm
        self.encoder_layers = nn.ModuleList()
        prev_dim = input_dim
        
        for i, hidden_dim in enumerate(hidden_dims):
            # Linear layer
            self.encoder_layers.append(nn.Linear(prev_dim, hidden_dim))
            
            # Domain-specific batch norm
            if use_bn:
                self.encoder_layers.append(DSBatchNorm(hidden_dim, n_domains))
            
            # ReLU activation
            self.encoder_layers.append(nn.ReLU())
            
            prev_dim = hidden_dim
        
        # Output layer: mu and logvar
        self.fc_mu = nn.Linear(prev_dim, latent_dim)
        self.fc_var = nn.Linear(prev_dim, latent_dim)
    
    def forward(self, x: torch.Tensor, domain_id: Union[int, torch.Tensor] = 0):
        """
        Args:
            x: [batch_size, input_dim]
            domain_id: batch/domain ID
        
        Returns:
            z: [batch_size, latent_dim] - sampled latent
            mu: [batch_size, latent_dim] - mean
            logvar: [batch_size, latent_dim] - log variance
        """
        h = x
        
        # Pass through encoder layers
        for layer in self.encoder_layers:
            if isinstance(layer, DSBatchNorm):
                h = layer(h, domain_id)
            else:
                h = layer(h)
        
        # Get distribution parameters
        mu = self.fc_mu(h)
        logvar = self.fc_var(h)
        
        # Reparameterization trick
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            z = mu + eps * std
        else:
            z = mu
        
        return z, mu, logvar


class SCALEXDecoder(nn.Module):
    """
    SCALEX解码器（支持ZINB/NB likelihood和domain-specific decoders）
    """
    def __init__(self, latent_dim: int, hidden_dims: list, output_dim: int, 
                 n_domains: int = 1, recon_type: str = 'zinb', use_bn: bool = True):
        super().__init__()
        
        self.n_domains = n_domains
        self.recon_type = recon_type  # 'zinb', 'nb', or 'mse'
        self.use_bn = use_bn
        
        # Shared decoder layers with DSBatchNorm
        self.decoder_layers = nn.ModuleList()
        prev_dim = latent_dim
        
        for hidden_dim in reversed(hidden_dims):
            self.decoder_layers.append(nn.Linear(prev_dim, hidden_dim))
            
            if use_bn:
                self.decoder_layers.append(DSBatchNorm(hidden_dim, n_domains))
            
            self.decoder_layers.append(nn.ReLU())
            prev_dim = hidden_dim
        
        # Domain-specific output layers
        if recon_type in ['zinb', 'nb']:
            # ZINB requires: mean, dispersion, (and dropout for zinb)
            if recon_type == 'zinb':
                self.output_layers = nn.ModuleList([
                    nn.ModuleDict({
                        'mean': nn.Sequential(nn.Linear(prev_dim, output_dim), nn.Softmax(dim=-1)),
                        'disp': nn.Sequential(nn.Linear(prev_dim, output_dim), nn.Softmax(dim=-1)),
                        'dropout': nn.Sequential(nn.Linear(prev_dim, output_dim), nn.Sigmoid())
                    }) for _ in range(n_domains)
                ])
            else:  # nb
                self.output_layers = nn.ModuleList([
                    nn.ModuleDict({
                        'mean': nn.Sequential(nn.Linear(prev_dim, output_dim), nn.Softmax(dim=-1)),
                        'disp': nn.Sequential(nn.Linear(prev_dim, output_dim), nn.Softmax(dim=-1)),
                    }) for _ in range(n_domains)
                ])
        else:  # mse
            self.output_layers = nn.ModuleList([
                nn.Linear(prev_dim, output_dim) for _ in range(n_domains)
            ])
        
        # Scaling factor for reconstruction (from original SCALEX)
        self.recon_scaling = nn.Parameter(torch.tensor(1.0))
    
    def forward(self, z: torch.Tensor, domain_id: Union[int, torch.Tensor] = 0):
        """
        Args:
            z: [batch_size, latent_dim]
            domain_id: batch/domain ID
        
        Returns:
            For ZINB: dict with 'mean', 'disp', 'dropout'
            For NB: dict with 'mean', 'disp'
            For MSE: reconstructed x
        """
        h = z
        
        # Pass through shared decoder
        for layer in self.decoder_layers:
            if isinstance(layer, DSBatchNorm):
                h = layer(h, domain_id)
            else:
                h = layer(h)
        
        # Handle scalar domain_id
        if isinstance(domain_id, int) or (isinstance(domain_id, torch.Tensor) and domain_id.dim() == 0):
            domain_id = int(domain_id) if isinstance(domain_id, torch.Tensor) else domain_id
            
            if self.recon_type in ['zinb', 'nb']:
                output = {}
                for key in self.output_layers[domain_id].keys():
                    output[key] = self.output_layers[domain_id][key](h)
                return output
            else:
                return self.output_layers[domain_id](h)
        
        # Handle vector domain_id (mixed batch)
        if self.recon_type in ['zinb', 'nb']:
            batch_size = z.size(0)
            output_dim = list(self.output_layers[0].values())[0][-2].out_features
            
            output = {key: torch.zeros(batch_size, output_dim, device=z.device) 
                     for key in self.output_layers[0].keys()}
            
            for i in range(self.n_domains):
                mask = (domain_id == i)
                if mask.any():
                    for key in output.keys():
                        output[key][mask] = self.output_layers[i][key](h[mask])
            return output
        else:
            output = torch.zeros(z.size(0), self.output_layers[0].out_features, device=z.device)
            for i in range(self.n_domains):
                mask = (domain_id == i)
                if mask.any():
                    output[mask] = self.output_layers[i](h[mask])
            return output


class SCALEXModel(BaseModel):
    """
    SCALEX模型的完整PyTorch实现
    
    Features:
    - Domain-specific batch normalization for batch correction
    - ZINB/NB likelihood for count data
    - Multiple domain support
    - Online integration capability
    
    Reference:
        Xiong et al. (2021) Online single-cell data integration through 
        projecting heterogeneous datasets into a common cell-embedding space.
        Nature Communications.
    """
    
    def __init__(self,
                 input_dim: int,
                 latent_dim: int = 10,
                 hidden_dims: list = None,
                 n_domains: int = 1,
                 recon_type: str = 'mse',  # 'zinb', 'nb', or 'mse'
                 use_bn: bool = True,
                 model_name: str = "SCALEX"):
        """
        Args:
            input_dim: Input feature dimension (number of genes)
            latent_dim: Latent space dimension
            hidden_dims: Hidden layer dimensions for encoder/decoder
            n_domains: Number of batches/domains
            recon_type: Reconstruction loss type ('zinb', 'nb', 'mse')
            use_bn: Whether to use domain-specific batch normalization
            model_name: Model name
        """
        if hidden_dims is None:
            hidden_dims = [512, 256, 128]
        
        super().__init__(input_dim, latent_dim, hidden_dims, model_name)
        
        self.n_domains = n_domains
        self.recon_type = recon_type
        self.use_bn = use_bn
        
        # Build encoder and decoder
        self.encoder_net = SCALEXEncoder(
            input_dim, hidden_dims, latent_dim, n_domains, use_bn
        )
        
        self.decoder_net = SCALEXDecoder(
            latent_dim, hidden_dims, input_dim, n_domains, recon_type, use_bn
        )
    
    def _prepare_batch(self, batch_data, device):
        """
        Prepare batch data and extract domain information
        """
        if isinstance(batch_data, (list, tuple)):
            x = batch_data[0].to(device).float()
            
            if len(batch_data) >= 2:
                second_item = batch_data[1]
                
                if second_item.dim() == 1:
                    domain_id = second_item.to(device).long()
                    return x, {'domain_id': domain_id}
                else:
                    return x, {'domain_id': 0}
            else:
                return x, {'domain_id': 0}
        else:
            x = batch_data.to(device).float()
            return x, {'domain_id': 0}
    
    def encode(self, x: torch.Tensor, domain_id: Union[int, torch.Tensor] = 0) -> torch.Tensor:
        """Encode to latent space"""
        z, mu, logvar = self.encoder_net(x, domain_id)
        return z
    
    def decode(self, z: torch.Tensor, domain_id: Union[int, torch.Tensor] = 0):
        """Decode from latent space"""
        return self.decoder_net(z, domain_id)
    
    def forward(self, x: torch.Tensor, domain_id: Union[int, torch.Tensor] = 0, 
                **kwargs) -> Dict[str, torch.Tensor]:
        """
        Forward pass
        
        Args:
            x: Input data [batch_size, input_dim]
            domain_id: Batch/domain ID
            
        Returns:
            Dictionary with outputs depending on recon_type
        """
        # Encode
        z, mu, logvar = self.encoder_net(x, domain_id)
        
        # Decode
        decoder_output = self.decoder_net(z, domain_id)
        
        # Organize output
        output = {
            'latent': z,
            'mu': mu,
            'logvar': logvar,
        }
        
        if self.recon_type in ['zinb', 'nb']:
            output.update(decoder_output)
        else:
            output['reconstruction'] = decoder_output
        
        return output
    
    def _zinb_loss(self, x: torch.Tensor, mean: torch.Tensor, 
                   disp: torch.Tensor, pi: torch.Tensor, 
                   scale_factor: float = 1.0) -> torch.Tensor:
        """
        Zero-Inflated Negative Binomial loss
        
        Args:
            x: True counts [batch_size, n_genes]
            mean: Mean parameter [batch_size, n_genes]
            disp: Dispersion parameter [batch_size, n_genes]
            pi: Dropout probability [batch_size, n_genes]
            scale_factor: Scaling factor (library size)
        """
        eps = 1e-10
        scale_factor = scale_factor[:, None]
        mean = mean * scale_factor
        
        t1 = torch.lgamma(disp + eps) + torch.lgamma(x + 1.0) - torch.lgamma(x + disp + eps)
        t2 = (disp + x) * torch.log(1.0 + (mean / (disp + eps))) + (x * (torch.log(disp + eps) - torch.log(mean + eps)))
        nb_final = t1 + t2
        
        nb_case = nb_final - torch.log(1.0 - pi + eps)
        zero_nb = torch.pow(disp / (disp + mean + eps), disp)
        zero_case = -torch.log(pi + ((1.0 - pi) * zero_nb) + eps)
        result = torch.where(torch.le(x, 1e-8), zero_case, nb_case)
        
        return torch.mean(result)
    
    def _nb_loss(self, x: torch.Tensor, mean: torch.Tensor, 
                 disp: torch.Tensor, scale_factor: float = 1.0) -> torch.Tensor:
        """
        Negative Binomial loss
        
        Args:
            x: True counts [batch_size, n_genes]
            mean: Mean parameter [batch_size, n_genes]
            disp: Dispersion parameter [batch_size, n_genes]
            scale_factor: Scaling factor (library size)
        """
        eps = 1e-10
        scale_factor = scale_factor[:, None]
        mean = mean * scale_factor
        
        t1 = torch.lgamma(disp + eps) + torch.lgamma(x + 1.0) - torch.lgamma(x + disp + eps)
        t2 = (disp + x) * torch.log(1.0 + (mean / (disp + eps))) + (x * (torch.log(disp + eps) - torch.log(mean + eps)))
        final = t1 + t2
        
        return torch.mean(final)
    
    def compute_loss(self, x: torch.Tensor, outputs: Dict[str, torch.Tensor],
                     beta: float = 1.0, scale_factor: Optional[torch.Tensor] = None,
                     **kwargs) -> Dict[str, torch.Tensor]:
        """
        Compute loss
        
        Args:
            x: Input data (raw counts for zinb/nb, log-normalized for mse)
            outputs: forward() outputs
            beta: KL weight
            scale_factor: Library size for each cell [batch_size]
            
        Returns:
            Loss dictionary
        """
        mu = outputs['mu']
        logvar = outputs['logvar']
        
        # Reconstruction loss
        if self.recon_type == 'zinb':
            if scale_factor is None:
                scale_factor = x.sum(1)
            recon_loss = self._zinb_loss(
                x, outputs['mean'], outputs['disp'], outputs['dropout'], scale_factor
            )
        elif self.recon_type == 'nb':
            if scale_factor is None:
                scale_factor = x.sum(1)
            recon_loss = self._nb_loss(
                x, outputs['mean'], outputs['disp'], scale_factor
            )
        else:  # mse
            recon_x = outputs['reconstruction']
            recon_loss = F.mse_loss(recon_x, x, reduction='mean')
        
        # KL divergence
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / x.size(0)
        kl_loss = torch.clamp(kl_loss, min=0.0)
        
        # Total loss
        total_loss = recon_loss + beta * kl_loss
        
        return {
            'total_loss': total_loss,
            'recon_loss': recon_loss,
            'kl_loss': kl_loss
        }
    
    def extract_latent(self, data_loader, device='cuda', batch_id: int = 0,
                      return_reconstructions: bool = False):
        """
        Extract latent representations
        
        Args:
            data_loader: Data loader
            device: Computing device
            batch_id: Domain ID for reconstruction
            return_reconstructions: Whether to return reconstructions
            
        Returns:
            Dictionary with latent representations
        """
        self.eval()
        self.to(device)
        
        latents = []
        mus = []
        reconstructions = [] if return_reconstructions else None
        
        with torch.no_grad():
            for batch_data in data_loader:
                x, metadata = self._prepare_batch(batch_data, device)
                domain_id = metadata.get('domain_id', 0)
                
                # Encode
                z, mu, logvar = self.encoder_net(x, domain_id)
                latents.append(z.cpu().numpy())
                mus.append(mu.cpu().numpy())
                
                # Decode for reconstruction if needed
                if return_reconstructions:
                    decoder_output = self.decoder_net(z, batch_id)
                    
                    if self.recon_type in ['zinb', 'nb']:
                        recon = decoder_output['mean']
                    else:
                        recon = decoder_output
                    
                    reconstructions.append(recon.cpu().numpy())
        
        result = {
            'latent': np.concatenate(latents, axis=0),
            'mu': np.concatenate(mus, axis=0)
        }
        
        if return_reconstructions:
            result['reconstruction'] = np.concatenate(reconstructions, axis=0)
        
        return result


# Convenience functions
def create_scalex_model(input_dim: int, latent_dim: int = 10, 
                       n_domains: int = 1, recon_type: str = 'mse', **kwargs):
    """
    Create SCALEX model
    
    Args:
        input_dim: Number of genes
        latent_dim: Latent dimension
        n_domains: Number of batches/domains
        recon_type: 'zinb', 'nb', or 'mse'
        **kwargs: Additional arguments
    
    Examples:
        >>> # For batch correction with MSE loss
        >>> model = create_scalex_model(2000, latent_dim=10, n_domains=5, recon_type='mse')
        
        >>> # For count data with ZINB likelihood
        >>> model = create_scalex_model(2000, latent_dim=10, n_domains=5, recon_type='zinb')
    """
    return SCALEXModel(input_dim=input_dim, latent_dim=latent_dim,
                      n_domains=n_domains, recon_type=recon_type, **kwargs)