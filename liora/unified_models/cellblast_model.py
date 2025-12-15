"""
Cell BLAST的PyTorch实现
基于DIRECTi的单细胞注释模型
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional
import numpy as np
from .base_model import BaseModel


class Encoder(nn.Module):
    """编码器网络（基于CellBLAST/DIRECTi的Gaussian latent实现）"""
    def __init__(self, input_dim: int, hidden_dims: list, latent_dim: int, dropout: float = 0.1):
        super().__init__()
        
        # 构建MLP编码器
        layers = []
        prev_dim = input_dim
        for i, hidden_dim in enumerate(hidden_dims):
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            # 第一层不加dropout（与原始实现一致）
            if i > 0 or dropout == 0.0:
                layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim
        
        self.encoder = nn.Sequential(*layers)
        # Gaussian VAE的参数层
        self.fc_mu = nn.Linear(prev_dim, latent_dim)
        self.fc_logvar = nn.Linear(prev_dim, latent_dim)
    
    def forward(self, x):
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar


class Decoder(nn.Module):
    """解码器网络（基于CellBLAST/DIRECTi实现）"""
    def __init__(self, latent_dim: int, hidden_dims: list, output_dim: int, dropout: float = 0.1):
        super().__init__()
        
        # 构建MLP解码器（与编码器结构对称）
        layers = []
        prev_dim = latent_dim
        reversed_dims = list(reversed(hidden_dims))
        for i, hidden_dim in enumerate(reversed_dims):
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            # 最后一层前不加dropout
            if i < len(reversed_dims) - 1:
                layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim
        
        # 输出层
        layers.append(nn.Linear(prev_dim, output_dim))
        self.decoder = nn.Sequential(*layers)
    
    def forward(self, z):
        return self.decoder(z)


class CellBLASTModel(BaseModel):
    """
    Cell BLAST模型的PyTorch实现
    用于单细胞数据的降维和细胞类型注释
    """
    
    def __init__(self,
                 input_dim: int,
                 latent_dim: int = 10,
                 hidden_dims: list = None,
                 dropout: float = 0.1,
                 model_name: str = "CellBLAST"):
        """
        Args:
            input_dim: 基因表达维度
            latent_dim: 潜在空间维度
            hidden_dims: 编码器/解码器隐藏层维度
            dropout: Dropout比例
            model_name: 模型名称
        """
        if hidden_dims is None:
            hidden_dims = [512, 256, 128]
        
        super().__init__(input_dim, latent_dim, hidden_dims, model_name)
        
        self.dropout = dropout
        self.encoder_net = Encoder(input_dim, hidden_dims, latent_dim, dropout)
        self.decoder_net = Decoder(latent_dim, hidden_dims, input_dim, dropout)
    
    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """重参数化技巧"""
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        else:
            return mu
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """编码到潜在空间"""
        mu, logvar = self.encoder_net(x)
        z = self.reparameterize(mu, logvar)
        return z
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """从潜在空间解码"""
        return self.decoder_net(z)
    
    def forward(self, x: torch.Tensor, **kwargs) -> Dict[str, torch.Tensor]:
        """
        前向传播
        
        Args:
            x: 输入基因表达 [batch_size, input_dim]
            
        Returns:
            包含重构、潜在表示、mu、logvar的字典
        """
        mu, logvar = self.encoder_net(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decoder_net(z)
        
        return {
            'reconstruction': recon,
            'latent': z,
            'mu': mu,
            'logvar': logvar
        }
    
    def compute_loss(self, x: torch.Tensor, outputs: Dict[str, torch.Tensor], 
                     beta: float = 1.0, **kwargs) -> Dict[str, torch.Tensor]:
        """
        计算VAE损失
        
        Args:
            x: 输入数据
            outputs: forward()的输出
            beta: KL散度的权重
            
        Returns:
            包含各种损失的字典
        """
        recon = outputs['reconstruction']
        mu = outputs['mu']
        logvar = outputs['logvar']
        
        # 重构损失（假设输入是log归一化的表达值）
        recon_loss = F.mse_loss(recon, x, reduction='sum') / x.size(0)
        
        # KL散度
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / x.size(0)
        
        # 总损失
        total_loss = recon_loss + beta * kl_loss
        
        return {
            'total_loss': total_loss,
            'recon_loss': recon_loss,
            'kl_loss': kl_loss
        }
    
    def extract_latent(self, data_loader, device='cuda', return_reconstructions=False):
        """
        提取潜在表示（返回均值，用于下游任务）
        """
        self.eval()
        self.to(device)
        
        latents = []
        mus = []
        reconstructions = [] if return_reconstructions else None
        
        with torch.no_grad():
            for batch_data in data_loader:
                if isinstance(batch_data, (list, tuple)):
                    x = batch_data[0]
                else:
                    x = batch_data
                    
                x = x.to(device).float()
                
                mu, logvar = self.encoder_net(x)
                z = mu  # 使用均值作为确定性的潜在表示
                
                latents.append(z.cpu().numpy())
                mus.append(mu.cpu().numpy())
                
                if return_reconstructions:
                    recon = self.decode(z)
                    reconstructions.append(recon.cpu().numpy())
        
        result = {
            'latent': np.concatenate(latents, axis=0),
            'mu': np.concatenate(mus, axis=0)
        }
        if return_reconstructions:
            result['reconstruction'] = np.concatenate(reconstructions, axis=0)
            
        return result


# 便捷的创建函数
def create_cellblast_model(input_dim: int, latent_dim: int = 10, **kwargs):
    """创建CellBLAST模型"""
    return CellBLASTModel(input_dim=input_dim, latent_dim=latent_dim, **kwargs)
