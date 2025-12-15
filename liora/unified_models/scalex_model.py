"""
SCALEX的PyTorch实现
用于单细胞数据跨批次整合的VAE模型
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional
import numpy as np
from .base_model import BaseModel


class SCALEXEncoder(nn.Module):
    """SCALEX编码器（更接近原始实现）"""
    def __init__(self, input_dim: int, hidden_dims: list, latent_dim: int, dropout: float = 0.1):
        super().__init__()
        
        # 构建编码器层（更接近原始SCALEX实现）
        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim
        
        self.encoder = nn.Sequential(*layers)
        # SCALEX原始实现中，最后一层输出是2*latent_dim，分为mu和logvar
        self.fc_params = nn.Linear(prev_dim, latent_dim * 2)
    
    def forward(self, x):
        h = self.encoder(x)
        params = self.fc_params(h)
        mu, logvar = torch.chunk(params, 2, dim=-1)
        
        # 重参数化
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            z = mu + eps * std
        else:
            z = mu
        
        return z, mu, logvar


class SCALEXDecoder(nn.Module):
    """SCALEX解码器（支持多个批次）"""
    def __init__(self, latent_dim: int, hidden_dims: list, output_dim: int, n_domains: int = 1):
        super().__init__()
        
        self.n_domains = n_domains
        
        # 共享的解码器层
        layers = []
        prev_dim = latent_dim
        for hidden_dim in reversed(hidden_dims):
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
            ])
            prev_dim = hidden_dim
        
        self.shared_decoder = nn.Sequential(*layers)
        
        # 每个domain一个输出层
        self.output_layers = nn.ModuleList([
            nn.Linear(prev_dim, output_dim) for _ in range(n_domains)
        ])
    
    def forward(self, z, domain_id):
        """
        Args:
            z: 潜在表示 [batch_size, latent_dim]
            domain_id: 批次ID [batch_size] 或标量，或 None（默认为0）
        """
        h = self.shared_decoder(z)
        
        # Handle None domain_id
        if domain_id is None:
            domain_id = 0
        
        # Convert tensor to appropriate type if needed
        if isinstance(domain_id, torch.Tensor):
            if domain_id.dim() == 0:
                # Scalar tensor
                domain_id = domain_id.item()
            elif domain_id.dim() == 1:
                # 1D tensor - multiple domains per batch
                output = torch.zeros(z.size(0), self.output_layers[0].out_features, 
                                   device=z.device)
                for i in range(self.n_domains):
                    mask = (domain_id == i)
                    if mask.any():
                        output[mask] = self.output_layers[i](h[mask])
                return output
        
        # Scalar domain_id (all samples use the same decoder)
        return self.output_layers[domain_id](h)


class SCALEXModel(BaseModel):
    """
    SCALEX模型的PyTorch实现
    用于单细胞数据的批次整合和跨模态整合
    """
    
    def __init__(self,
                 input_dim: int,
                 latent_dim: int = 10,
                 hidden_dims: list = None,
                 n_domains: int = 1,
                 model_name: str = "SCALEX"):
        """
        Args:
            input_dim: 输入特征维度（基因数）
            latent_dim: 潜在空间维度
            hidden_dims: 编码器/解码器隐藏层维度
            n_domains: 批次/域的数量
            model_name: 模型名称
        """
        if hidden_dims is None:
            hidden_dims = [512, 256, 128]
        
        super().__init__(input_dim, latent_dim, hidden_dims, model_name)
        
        self.n_domains = n_domains
        self.encoder_net = SCALEXEncoder(input_dim, hidden_dims, latent_dim, dropout=0.1)
        self.decoder_net = SCALEXDecoder(latent_dim, hidden_dims, input_dim, n_domains)
    
    def _prepare_batch(self, batch_data, device):
        """
        Prepare batch data, extract batch/domain information
        
        Since Liora's Env returns (X_norm, X_raw), we need to handle this format
        and ignore the raw data for SCALEX which doesn't use it.
        """
        if isinstance(batch_data, (list, tuple)):
            x = batch_data[0].to(device).float()
            
            # Check if batch_data[1] is domain_id (1D tensor) or raw data (2D tensor)
            if len(batch_data) >= 2:
                second_item = batch_data[1]
                
                # If it's 1D, it's domain_id; if 2D, it's raw data (ignore it)
                if second_item.dim() == 1:
                    domain_id = second_item.to(device).long()
                    return x, {'domain_id': domain_id}
                else:
                    # It's raw data from Liora's loader, treat as single batch
                    return x, {'domain_id': 0}
            else:
                return x, {'domain_id': 0}
        else:
            x = batch_data.to(device).float()
            return x, {'domain_id': 0}
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """编码到潜在空间"""
        z, mu, logvar = self.encoder_net(x)
        return z
    
    def decode(self, z: torch.Tensor, domain_id: int = 0) -> torch.Tensor:
        """从潜在空间解码"""
        return self.decoder_net(z, domain_id)
    
    def forward(self, x: torch.Tensor, domain_id=None, **kwargs) -> Dict[str, torch.Tensor]:
        """
        Forward pass
        
        Args:
            x: Input data [batch_size, input_dim]
            domain_id: Batch ID [batch_size] or scalar, optional
            **kwargs: Additional arguments
            
        Returns:
            Dictionary with 'reconstruction', 'latent', 'mu', 'logvar'
        """
        # Encode
        z, mu, var = self.encoder_net(x)
        
        # Decode with domain-specific decoder
        if domain_id is None:
            domain_id = 0
        
        recon_x = self.decoder_net(z, domain_id)
        
        return {
            'reconstruction': recon_x,
            'latent': z,
            'mu': mu,
            'logvar': var
        }
    
    def compute_loss(self, x: torch.Tensor, outputs: Dict[str, torch.Tensor],
                    beta: float = 0.5, **kwargs) -> Dict[str, torch.Tensor]:
        """
        Compute loss
        
        Args:
            x: Input data (log-normalized, NOT in [0,1] range)
            outputs: forward() output
            beta: KL divergence weight
            
        Returns:
            Loss dictionary
        """
        recon_x = outputs['reconstruction']
        mu = outputs['mu']
        logvar = outputs['logvar']
        
        # Reconstruction loss - Use MSE for log-normalized data
        recon_loss = F.mse_loss(recon_x, x, reduction='sum') / x.size(0)
        
        # KL divergence: D_KL(N(mu, sigma) || N(0, 1))
        # Always non-negative by definition
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / x.size(0)
        kl_loss = torch.clamp(kl_loss, min=0.0)  # Ensure non-negative for numerical stability
        
        # Total loss
        total_loss = recon_loss + beta * kl_loss
        
        return {
            'total_loss': total_loss,
            'recon_loss': recon_loss,
            'kl_loss': kl_loss
        }
        
    def extract_latent(self, data_loader, device='cuda', batch_id=None, 
                    return_reconstructions=False):
        """
        提取潜在表示
        
        Args:
            data_loader: 数据加载器
            device: 计算设备
            batch_id: 用于重构的批次ID (integer)
            return_reconstructions: 是否返回重构
            
        Returns:
            包含潜在表示的字典
        """
        self.eval()
        self.to(device)
        
        latents = []
        mus = []
        reconstructions = [] if return_reconstructions else None
        
        with torch.no_grad():
            for batch_data in data_loader:
                # Extract input data
                if isinstance(batch_data, (list, tuple)):
                    x = batch_data[0]
                    # Don't extract domain from batch_data[1] - it's raw data, not domain_id
                else:
                    x = batch_data
                
                x = x.to(device).float()
                
                # Encode
                z, mu, var = self.encoder_net(x)
                latents.append(z.cpu().numpy())
                mus.append(mu.cpu().numpy())
                
                # Decode for reconstruction if needed
                if return_reconstructions:
                    # Use batch_id if provided, otherwise default to 0
                    domain_for_recon = batch_id if batch_id is not None else 0
                    recon = self.decoder_net(z, domain_for_recon)
                    reconstructions.append(recon.cpu().numpy())
        
        result = {
            'latent': np.concatenate(latents, axis=0),
            'mu': np.concatenate(mus, axis=0)
        }
        
        if return_reconstructions:
            result['reconstruction'] = np.concatenate(reconstructions, axis=0)
        
        return result


# 便捷的创建函数
def create_scalex_model(input_dim: int, latent_dim: int = 10, n_domains: int = 1, **kwargs):
    """创建SCALEX模型"""
    return SCALEXModel(input_dim=input_dim, latent_dim=latent_dim, 
                      n_domains=n_domains, **kwargs)
