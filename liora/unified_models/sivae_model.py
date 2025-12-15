"""
siVAE的PyTorch实现
监督变分自编码器，用于基因调控网络推断

Based on: https://github.com/gcskoenig/siVAE
Reference: Kopf et al. (2021) Mixture-of-Experts Variational Autoencoder for clustering and generating from similarity-based representations on single cell data
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, List, Tuple
import numpy as np
from .base_model import BaseModel


class InterpretableLinearEncoder(nn.Module):
    """
    Interpretable Linear Encoder (ILE)
    
    Key component of siVAE that maps genes to interpretable latent factors.
    Uses constrained weights for biological interpretability.
    """
    def __init__(self, input_dim: int, latent_dim: int, 
                 constraint: str = 'l1', constraint_weight: float = 0.01):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.constraint = constraint
        self.constraint_weight = constraint_weight
        
        # Linear projection with interpretability constraint
        self.fc = nn.Linear(input_dim, latent_dim, bias=False)
        
        # Initialize with small random weights
        nn.init.xavier_normal_(self.fc.weight, gain=0.01)
    
    def forward(self, x):
        """
        Args:
            x: [batch, input_dim] normalized gene expression
        Returns:
            [batch, latent_dim] interpretable latent factors
        """
        return self.fc(x)
    
    def get_constraint_loss(self):
        """
        Compute constraint loss for interpretability
        
        L1: Encourages sparsity (few genes per factor)
        L2: Encourages smoothness
        """
        if self.constraint == 'l1':
            return self.constraint_weight * torch.abs(self.fc.weight).sum()
        elif self.constraint == 'l2':
            return self.constraint_weight * (self.fc.weight ** 2).sum()
        else:
            return 0.0


class siVAEEncoder(nn.Module):
    """
    siVAE编码器（完全按照原始实现）
    
    Architecture:
    1. Interpretable Linear Encoder (genes → latent factors)
    2. Optional nonlinear layers
    3. Output: mu, logvar for reparameterization
    """
    def __init__(self, input_dim: int, hidden_dims: list, latent_dim: int,
                 use_interpretable: bool = True,
                 constraint: str = 'l1',
                 constraint_weight: float = 0.01,
                 batch_norm: bool = True, 
                 dropout: float = 0.1):
        super().__init__()
        
        self.use_interpretable = use_interpretable
        self.batch_norm = batch_norm
        
        # Interpretable Linear Encoder
        if use_interpretable:
            self.ile = InterpretableLinearEncoder(
                input_dim, latent_dim, constraint, constraint_weight
            )
            encoder_input_dim = latent_dim
        else:
            self.ile = None
            encoder_input_dim = input_dim
        
        # Nonlinear encoder layers (optional)
        if len(hidden_dims) > 0:
            layers = []
            prev_dim = encoder_input_dim
            
            for hidden_dim in hidden_dims:
                layers.append(nn.Linear(prev_dim, hidden_dim))
                if batch_norm:
                    layers.append(nn.BatchNorm1d(hidden_dim))
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(dropout))
                prev_dim = hidden_dim
            
            self.encoder = nn.Sequential(*layers)
            self.fc_mu = nn.Linear(prev_dim, latent_dim)
            self.fc_logvar = nn.Linear(prev_dim, latent_dim)
        else:
            # Direct mapping from ILE to mu/logvar
            self.encoder = nn.Identity()
            self.fc_mu = nn.Identity()
            self.fc_logvar = nn.Linear(encoder_input_dim, latent_dim)
        
        # Initialize logvar to small values
        if isinstance(self.fc_logvar, nn.Linear):
            with torch.no_grad():
                self.fc_logvar.weight.fill_(0.0)
                self.fc_logvar.bias.fill_(-5.0)
    
    def forward(self, x):
        """
        Args:
            x: [batch, input_dim] normalized gene expression
        Returns:
            mu: [batch, latent_dim]
            logvar: [batch, latent_dim]
            ile_output: [batch, latent_dim] (if use_interpretable)
        """
        if self.use_interpretable:
            ile_output = self.ile(x)
            h = self.encoder(ile_output)
            
            if isinstance(self.fc_mu, nn.Identity):
                mu = ile_output
            else:
                mu = self.fc_mu(h)
            
            logvar = self.fc_logvar(h)
            return mu, logvar, ile_output
        else:
            h = self.encoder(x)
            mu = self.fc_mu(h)
            logvar = self.fc_logvar(h)
            return mu, logvar, None


class siVAEDecoder(nn.Module):
    """
    siVAE解码器（支持ZINB/NB/Gaussian输出）
    """
    def __init__(self, latent_dim: int, hidden_dims: list, output_dim: int,
                 batch_norm: bool = True, dropout: float = 0.1,
                 output_distribution: str = 'nb'):
        super().__init__()
        
        self.batch_norm = batch_norm
        self.output_distribution = output_distribution
        
        # Decoder layers
        layers = []
        prev_dim = latent_dim
        
        for hidden_dim in reversed(hidden_dims):
            layers.append(nn.Linear(prev_dim, hidden_dim))
            if batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim
        
        self.decoder = nn.Sequential(*layers)
        
        # Output layers based on distribution
        if output_distribution == 'zinb':
            # ZINB: mean (scaled by size factor), dispersion, dropout probability
            self.fc_mean = nn.Sequential(
                nn.Linear(prev_dim, output_dim),
                nn.Softmax(dim=-1)  # Normalized mean
            )
            self.fc_disp = nn.Sequential(
                nn.Linear(prev_dim, output_dim),
                nn.Softplus()  # Positive dispersion
            )
            self.fc_pi = nn.Sequential(
                nn.Linear(prev_dim, output_dim),
                nn.Sigmoid()  # Dropout probability [0, 1]
            )
        elif output_distribution == 'nb':
            # NB: mean (scaled by size factor), dispersion
            self.fc_mean = nn.Sequential(
                nn.Linear(prev_dim, output_dim),
                nn.Softmax(dim=-1)
            )
            self.fc_disp = nn.Sequential(
                nn.Linear(prev_dim, output_dim),
                nn.Softplus()
            )
        elif output_distribution == 'gaussian':
            self.fc_mu = nn.Linear(prev_dim, output_dim)
            self.fc_logvar = nn.Linear(prev_dim, output_dim)
        else:
            raise ValueError(f"Unknown distribution: {output_distribution}")
    
    def forward(self, z):
        """
        Args:
            z: [batch, latent_dim]
        Returns:
            Distribution parameters
        """
        h = self.decoder(z)
        
        if self.output_distribution == 'zinb':
            mean = self.fc_mean(h)
            disp = self.fc_disp(h)
            pi = self.fc_pi(h)
            return {'mean': mean, 'disp': disp, 'pi': pi}
        elif self.output_distribution == 'nb':
            mean = self.fc_mean(h)
            disp = self.fc_disp(h)
            return {'mean': mean, 'disp': disp}
        elif self.output_distribution == 'gaussian':
            mu = self.fc_mu(h)
            logvar = self.fc_logvar(h)
            return {'mu': mu, 'logvar': logvar}


class SupervisedClassifier(nn.Module):
    """
    Supervised classifier for siVAE
    
    Predicts cell type labels from latent representation.
    This is what makes siVAE "supervised".
    """
    def __init__(self, latent_dim: int, n_classes: int, hidden_dim: int = 128):
        super().__init__()
        
        self.classifier = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, n_classes)
        )
    
    def forward(self, z):
        """
        Args:
            z: [batch, latent_dim]
        Returns:
            [batch, n_classes] logits
        """
        return self.classifier(z)


class siVAEModel(BaseModel):
    """
    siVAE模型的完整PyTorch实现
    
    Features:
    - Interpretable Linear Encoder for gene-factor mapping
    - Supervised classification for cell type prediction
    - ZINB/NB/Gaussian reconstruction
    - Library size normalization
    - Gene relevance scoring for GRN inference
    
    Reference:
        Kopf et al. (2021) Mixture-of-Experts Variational Autoencoder 
        for clustering and generating from similarity-based representations 
        on single cell data.
    """
    
    def __init__(self,
                 input_dim: int,
                 latent_dim: int = 50,
                 hidden_dims: list = None,
                 n_classes: int = 0,
                 use_interpretable: bool = True,
                 constraint: str = 'l1',
                 constraint_weight: float = 0.01,
                 batch_norm: bool = True,
                 dropout: float = 0.1,
                 output_distribution: str = 'nb',
                 use_batch: bool = False,
                 n_batches: int = 1,
                 supervised_weight: float = 1.0,
                 model_name: str = "siVAE"):
        """
        Args:
            input_dim: Input feature dimension (number of genes)
            latent_dim: Latent space dimension
            hidden_dims: Hidden layer dimensions
            n_classes: Number of cell type classes (0 = unsupervised)
            use_interpretable: Use interpretable linear encoder
            constraint: Constraint type ('l1' or 'l2')
            constraint_weight: Weight for constraint loss
            batch_norm: Use batch normalization
            dropout: Dropout rate
            output_distribution: 'zinb', 'nb', or 'gaussian'
            use_batch: Use batch correction
            n_batches: Number of batches
            supervised_weight: Weight for supervised classification loss
            model_name: Model name
        """
        if hidden_dims is None:
            hidden_dims = [512, 256]
        
        super().__init__(input_dim, latent_dim, hidden_dims, model_name)
        
        self.n_classes = n_classes
        self.use_interpretable = use_interpretable
        self.constraint = constraint
        self.constraint_weight = constraint_weight
        self.batch_norm = batch_norm
        self.dropout = dropout
        self.output_distribution = output_distribution
        self.use_batch = use_batch
        self.n_batches = n_batches
        self.supervised_weight = supervised_weight
        
        # Encoder
        self.encoder_net = siVAEEncoder(
            input_dim, hidden_dims, latent_dim,
            use_interpretable, constraint, constraint_weight,
            batch_norm, dropout
        )
        
        # Decoder
        decoder_input_dim = latent_dim
        if use_batch:
            decoder_input_dim += n_batches
        
        self.decoder_net = siVAEDecoder(
            decoder_input_dim, hidden_dims, input_dim,
            batch_norm, dropout, output_distribution
        )
        
        # Supervised classifier
        if n_classes > 0:
            self.classifier = SupervisedClassifier(latent_dim, n_classes)
        else:
            self.classifier = None
        
        # Batch embedding
        if use_batch:
            self.batch_embedding = nn.Embedding(n_batches, n_batches)
            self.batch_embedding.weight.data = torch.eye(n_batches)
            self.batch_embedding.weight.requires_grad = False
    
    def _prepare_batch(self, batch_data, device):
        """Prepare batch data and extract metadata"""
        if isinstance(batch_data, (list, tuple)):
            x = batch_data[0].to(device).float()
            metadata = {}
            
            if len(batch_data) >= 2:
                second_item = batch_data[1]
                if second_item.dtype in [torch.long, torch.int32, torch.int64]:
                    # Could be batch_id or labels
                    if second_item.max() < self.n_batches and self.use_batch:
                        metadata['batch_id'] = second_item.to(device).long()
                    elif self.n_classes > 0 and second_item.max() < self.n_classes:
                        metadata['labels'] = second_item.to(device).long()
            
            if len(batch_data) >= 3:
                third_item = batch_data[2]
                if third_item.dtype in [torch.long, torch.int32, torch.int64]:
                    if self.n_classes > 0 and third_item.max() < self.n_classes:
                        metadata['labels'] = third_item.to(device).long()
            
            # Set defaults if not provided
            if self.use_batch and 'batch_id' not in metadata:
                metadata['batch_id'] = torch.zeros(x.size(0), dtype=torch.long, device=device)
            
            return x, metadata
        else:
            x = batch_data.to(device).float()
            metadata = {}
            if self.use_batch:
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
        mu, logvar, _ = self.encoder_net(x)
        z = self.reparameterize(mu, logvar)
        return z
    
    def decode(self, z: torch.Tensor, batch_id: Optional[torch.Tensor] = None):
        """Decode from latent space"""
        if self.use_batch and batch_id is not None:
            batch_emb = self.batch_embedding(batch_id)
            z = torch.cat([z, batch_emb], dim=-1)
        
        decoder_output = self.decoder_net(z)
        
        if self.output_distribution in ['zinb', 'nb']:
            return decoder_output['mean']
        else:
            return decoder_output['mu']
    
    def forward(self, x: torch.Tensor, 
                batch_id: Optional[torch.Tensor] = None,
                labels: Optional[torch.Tensor] = None,
                **kwargs) -> Dict[str, torch.Tensor]:
        """
        Forward pass
        
        Args:
            x: Input data [batch_size, input_dim] (raw counts)
            batch_id: Batch IDs [batch_size]
            labels: Cell type labels [batch_size] (for supervised learning)
        
        Returns:
            Dictionary with outputs
        """
        # Compute library size (total count per cell)
        library_size = x.sum(dim=1, keepdim=True)
        
        # Normalize by library size
        x_norm = x / (library_size + 1e-6)
        
        # Encode
        mu_z, logvar_z, ile_output = self.encoder_net(x_norm)
        z = self.reparameterize(mu_z, logvar_z)
        
        # Decode
        decoder_input = z
        if self.use_batch and batch_id is not None:
            batch_emb = self.batch_embedding(batch_id)
            decoder_input = torch.cat([z, batch_emb], dim=-1)
        
        decoder_output = self.decoder_net(decoder_input)
        
        # Supervised classification
        if self.classifier is not None and labels is not None:
            logits = self.classifier(z)
        else:
            logits = None
        
        output = {
            'latent': z,
            'mu': mu_z,
            'logvar': logvar_z,
            'library_size': library_size,
            'ile_output': ile_output,
            'logits': logits
        }
        
        # Add decoder outputs
        output.update(decoder_output)
        
        return output
    
    def _zinb_loss(self, x: torch.Tensor, mean: torch.Tensor,
                   disp: torch.Tensor, pi: torch.Tensor,
                   library_size: torch.Tensor) -> torch.Tensor:
        """
        Zero-Inflated Negative Binomial loss
        
        Args:
            x: True counts [batch, genes]
            mean: Normalized mean [batch, genes]
            disp: Dispersion [batch, genes]
            pi: Dropout probability [batch, genes]
            library_size: Library size [batch, 1]
        """
        eps = 1e-10
        mean_scaled = mean * library_size
        
        # NB part
        t1 = torch.lgamma(disp + eps) + torch.lgamma(x + 1.0) - torch.lgamma(x + disp + eps)
        t2 = (disp + x) * torch.log(1.0 + (mean_scaled / (disp + eps))) + \
             (x * (torch.log(disp + eps) - torch.log(mean_scaled + eps)))
        nb_case = t1 + t2 - torch.log(1.0 - pi + eps)
        
        # Zero-inflation part
        zero_nb = torch.pow(disp / (disp + mean_scaled + eps), disp)
        zero_case = -torch.log(pi + ((1.0 - pi) * zero_nb) + eps)
        
        result = torch.where(x < 1e-8, zero_case, nb_case)
        return result.mean()
    
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
                     beta: float = 1, 
                     labels: Optional[torch.Tensor] = None,
                     **kwargs) -> Dict[str, torch.Tensor]:
        """
        Compute loss
        
        Args:
            x: Input data (raw counts)
            outputs: forward() outputs
            beta: KL weight
            labels: Cell type labels (for supervised loss)
        
        Returns:
            Loss dictionary
        """
        mu_z = outputs['mu']
        logvar_z = outputs['logvar']
        library_size = outputs['library_size']
        
        # KL divergence
        logvar_z_clamped = torch.clamp(logvar_z, min=-5, max=5)
        kl_loss = -0.5 * torch.sum(1 + logvar_z_clamped - mu_z.pow(2) - logvar_z_clamped.exp()) / x.size(0)
        kl_loss = torch.clamp(kl_loss, min=0.0)
        
        # Reconstruction loss
        if self.output_distribution == 'zinb':
            recon_loss = self._zinb_loss(
                x, outputs['mean'], outputs['disp'], outputs['pi'], library_size
            )
        elif self.output_distribution == 'nb':
            recon_loss = self._nb_loss(
                x, outputs['mean'], outputs['disp'], library_size
            )
        else:  # gaussian
            mu_x = outputs['mu']
            recon_loss = F.mse_loss(mu_x, x, reduction='mean')
        
        # Supervised classification loss
        if self.classifier is not None and labels is not None and outputs['logits'] is not None:
            supervised_loss = F.cross_entropy(outputs['logits'], labels)
        else:
            supervised_loss = torch.tensor(0.0, device=x.device)
        
        # Interpretable encoder constraint
        if self.use_interpretable and self.encoder_net.ile is not None:
            constraint_loss = self.encoder_net.ile.get_constraint_loss()
        else:
            constraint_loss = torch.tensor(0.0, device=x.device)
        
        # Total loss
        total_loss = recon_loss + beta * kl_loss + \
                    self.supervised_weight * supervised_loss + constraint_loss
        
        return {
            'total_loss': total_loss,
            'recon_loss': recon_loss,
            'kl_loss': kl_loss,
            'supervised_loss': supervised_loss,
            'constraint_loss': constraint_loss
        }
    
    def compute_gene_relevance(self, x: torch.Tensor, 
                              latent_dim_idx: int = 0) -> torch.Tensor:
        """
        Compute gene relevance scores for a specific latent dimension
        
        This is key for GRN inference in siVAE.
        
        Args:
            x: Input data [batch, genes]
            latent_dim_idx: Which latent dimension to analyze
        
        Returns:
            [genes] relevance scores
        """
        x.requires_grad = True
        
        # Forward pass
        mu_z, logvar_z, ile_output = self.encoder_net(x)
        
        # Select specific latent dimension
        target_latent = mu_z[:, latent_dim_idx].sum()
        
        # Compute gradient
        target_latent.backward()
        
        # Gene relevance = absolute gradient
        relevance = torch.abs(x.grad).mean(dim=0)
        
        return relevance
    
    def extract_latent(self, data_loader, device='cuda', batch_id=None,
                      return_reconstructions=False):
        """Extract latent representations"""
        self.eval()
        self.to(device)
        
        latents = []
        mus = []
        ile_outputs = [] if self.use_interpretable else None
        reconstructions = [] if return_reconstructions else None
        predictions = [] if self.classifier is not None else None
        
        with torch.no_grad():
            for batch_data in data_loader:
                x, metadata = self._prepare_batch(batch_data, device)
                
                # Normalize
                library_size = x.sum(dim=1, keepdim=True)
                x_norm = x / (library_size + 1e-6)
                
                # Encode
                mu_z, logvar_z, ile_output = self.encoder_net(x_norm)
                z = mu_z
                
                latents.append(z.cpu().numpy())
                mus.append(mu_z.cpu().numpy())
                
                if self.use_interpretable and ile_output is not None:
                    ile_outputs.append(ile_output.cpu().numpy())
                
                # Predictions
                if self.classifier is not None:
                    logits = self.classifier(z)
                    preds = torch.argmax(logits, dim=1)
                    predictions.append(preds.cpu().numpy())
                
                # Reconstructions
                if return_reconstructions:
                    recon = self.decode(z, metadata.get('batch_id'))
                    reconstructions.append(recon.cpu().numpy())
        
        result = {
            'latent': np.concatenate(latents, axis=0),
            'mu': np.concatenate(mus, axis=0)
        }
        
        if self.use_interpretable and ile_outputs:
            result['ile_output'] = np.concatenate(ile_outputs, axis=0)
        
        if predictions:
            result['predictions'] = np.concatenate(predictions, axis=0)
        
        if return_reconstructions:
            result['reconstruction'] = np.concatenate(reconstructions, axis=0)
        
        return result


def create_sivae_model(input_dim: int, latent_dim: int = 50,
                      n_classes: int = 0, **kwargs):
    """
    Create siVAE model
    
    Args:
        input_dim: Number of genes
        latent_dim: Latent dimension
        n_classes: Number of cell type classes (0 = unsupervised)
        **kwargs: Additional arguments
    
    Examples:
        >>> # Unsupervised siVAE with interpretable encoder
        >>> model = create_sivae_model(2000, latent_dim=50, use_interpretable=True)
        
        >>> # Supervised siVAE for cell type classification
        >>> model = create_sivae_model(2000, latent_dim=50, n_classes=10)
        
        >>> # With ZINB reconstruction
        >>> model = create_sivae_model(2000, latent_dim=50, output_distribution='zinb')
    """
    return siVAEModel(input_dim=input_dim, latent_dim=latent_dim,
                     n_classes=n_classes, **kwargs)