"""
siVAE的PyTorch实现
监督变分自编码器，用于基因调控网络推断
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, List
import numpy as np
from .base_model import BaseModel


class siVAEEncoder(nn.Module):
    """siVAE编码器"""
    def __init__(self, input_dim: int, hidden_dims: list, latent_dim: int, 
                 batch_norm: bool = True, dropout: float = 0.1):
        super().__init__()
        
        self.batch_norm = batch_norm
        layers = []
        prev_dim = input_dim
        
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
        
        # Initialize logvar to small values to prevent initial KL blow-up
        # log(variance) = log(std^2) = 2*log(std)
        # With bias = -5: log(std^2) = -5, so std = 0.0067 (very small)
        # This prevents encoder from being overconfident initially, reducing KL loss at start
        with torch.no_grad():
            self.fc_logvar.weight.fill_(0.0)
            self.fc_logvar.bias.fill_(-5.0)  # Start with small variance to keep KL loss small
    
    def forward(self, x):
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar


class siVAEDecoder(nn.Module):
    """siVAE解码器"""
    def __init__(self, latent_dim: int, hidden_dims: list, output_dim: int,
                 batch_norm: bool = True, dropout: float = 0.1, 
                 output_distribution: str = 'normal'):
        super().__init__()
        
        self.batch_norm = batch_norm
        self.output_distribution = output_distribution
        
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
        
        # 根据输出分布选择输出层
        if output_distribution == 'normal':
            self.fc_mu = nn.Linear(prev_dim, output_dim)
            self.fc_logvar = nn.Linear(prev_dim, output_dim)
        else:
            self.fc_out = nn.Linear(prev_dim, output_dim)
    
    def forward(self, z):
        h = self.decoder(z)
        
        if self.output_distribution == 'normal':
            mu = self.fc_mu(h)
            logvar = self.fc_logvar(h)
            return mu, logvar
        else:
            return self.fc_out(h)


class siVAEModel(BaseModel):
    """
    siVAE模型的PyTorch实现
    监督变分自编码器，用于单细胞数据分析和基因调控网络推断
    """
    
    def __init__(self,
                 input_dim: int,
                 latent_dim: int = 50,
                 hidden_dims: list = None,
                 batch_norm: bool = True,
                 dropout: float = 0.1,
                 output_distribution: str = 'normal',
                 use_batch: bool = False,
                 n_batches: int = 1,
                 model_name: str = "siVAE"):
        """
        Args:
            input_dim: 输入特征维度（基因数）
            latent_dim: 潜在空间维度
            hidden_dims: 隐藏层维度列表
            batch_norm: 是否使用批归一化
            dropout: Dropout比例
            output_distribution: 输出分布类型 ('normal', 'bernoulli')
            use_batch: 是否使用批次信息
            n_batches: 批次数量
            model_name: 模型名称
        """
        if hidden_dims is None:
            hidden_dims = [512, 256, 128]
        
        super().__init__(input_dim, latent_dim, hidden_dims, model_name)
        
        self.batch_norm = batch_norm
        self.dropout = dropout
        self.output_distribution = output_distribution
        self.use_batch = use_batch
        self.n_batches = n_batches
        
        # 编码器
        self.encoder_net = siVAEEncoder(input_dim, hidden_dims, latent_dim, 
                                       batch_norm, dropout)
        
        # 解码器
        decoder_input_dim = latent_dim
        if use_batch:
            decoder_input_dim += n_batches  # 添加batch embedding
        
        self.decoder_net = siVAEDecoder(decoder_input_dim, hidden_dims, input_dim,
                                       batch_norm, dropout, output_distribution)
        
        # 批次嵌入
        if use_batch:
            self.batch_embedding = nn.Embedding(n_batches, n_batches)
            # 初始化为one-hot
            self.batch_embedding.weight.data = torch.eye(n_batches)
            self.batch_embedding.weight.requires_grad = False
    
    def _prepare_batch(self, batch_data, device):
        """
        准备批次数据，提取batch信息
        
        如果没有提供batch信息，将所有细胞视为单个批次。
        当use_batch=True但未提供batch_id时，会使用batch_id=0进行批次修正
        当use_batch=False时，忽略batch信息
        这允许模型在没有批次注解的情况下正常工作
        """
        if isinstance(batch_data, (list, tuple)):
            x = batch_data[0].to(device).float()
            if len(batch_data) >= 2 and self.use_batch:
                batch_id = batch_data[1].to(device).long()
                return x, {'batch_id': batch_id}
            elif self.use_batch:
                # 如果启用了批次修正但没有提供batch_id，使用batch_id=0
                # 这将所有细胞视为单个批次
                batch_id = torch.zeros(x.size(0), dtype=torch.long, device=device)
                return x, {'batch_id': batch_id}
            else:
                return x, {}
        else:
            x = batch_data.to(device).float()
            if self.use_batch:
                # 如果启用了批次修正但没有提供batch数据，使用batch_id=0
                batch_id = torch.zeros(x.size(0), dtype=torch.long, device=device)
                return x, {'batch_id': batch_id}
            else:
                return x, {}
    
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
    
    def decode(self, z: torch.Tensor, batch_id: Optional[torch.Tensor] = None) -> torch.Tensor:
        """从潜在空间解码"""
        if self.use_batch and batch_id is not None:
            batch_emb = self.batch_embedding(batch_id)
            z = torch.cat([z, batch_emb], dim=-1)
        
        if self.output_distribution == 'normal':
            mu, logvar = self.decoder_net(z)
            return mu
        else:
            return self.decoder_net(z)
    
    def forward(self, x: torch.Tensor, batch_id: Optional[torch.Tensor] = None,
                **kwargs) -> Dict[str, torch.Tensor]:
        """
        前向传播
        
        Args:
            x: 输入数据 [batch_size, input_dim]
            batch_id: 批次ID [batch_size]
            
        Returns:
            包含重构、潜在表示等的字典
        """
        # 编码
        mu_z, logvar_z = self.encoder_net(x)
        z = self.reparameterize(mu_z, logvar_z)
        
        # 解码
        decoder_input = z
        if self.use_batch and batch_id is not None:
            batch_emb = self.batch_embedding(batch_id)
            decoder_input = torch.cat([z, batch_emb], dim=-1)
        
        if self.output_distribution == 'normal':
            mu_x, logvar_x = self.decoder_net(decoder_input)
            recon_x = mu_x
            output = {
                'reconstruction': recon_x,
                'recon_mu': mu_x,
                'recon_logvar': logvar_x,
                'latent': z,
                'mu': mu_z,
                'logvar': logvar_z
            }
        else:
            recon_x = self.decoder_net(decoder_input)
            output = {
                'reconstruction': recon_x,
                'latent': z,
                'mu': mu_z,
                'logvar': logvar_z
            }
        
        return output
    
    def compute_loss(self, x: torch.Tensor, outputs: Dict[str, torch.Tensor],
                     beta: float = 0.001, l2_scale: float = 0.0, **kwargs) -> Dict[str, torch.Tensor]:
        """
        计算损失
        
        Args:
            x: 输入数据 (log-normalized, range: [-10, 10])
            outputs: forward()的输出
            beta: KL散度权重 (default 0.001 to prevent KL collapse early in training)
            l2_scale: L2正则化权重
            
        Returns:
            损失字典
        """
        mu_z = outputs['mu']
        logvar_z = outputs['logvar']
        
        # Clamp log variance to prevent numerical instability and KL blow-up
        logvar_z_clamped = torch.clamp(logvar_z, min=-5, max=5)
        
        # KL散度: D_KL(N(mu, sigma) || N(0, 1))
        # This is always >= 0
        kl_loss = -0.5 * torch.sum(1 + logvar_z_clamped - mu_z.pow(2) - logvar_z_clamped.exp()) / x.size(0)
        
        # Ensure KL is non-negative (numerical stability)
        kl_loss = torch.clamp(kl_loss, min=0.0)
        
        # 重构损失 - Use MSE loss (always positive, works for log-normalized data)
        if self.output_distribution == 'normal':
            mu_x = outputs['recon_mu']
            # Use simple MSE loss instead of Gaussian NLL to avoid negative losses
            recon_loss = F.mse_loss(mu_x, x, reduction='sum') / x.size(0)
        else:
            recon_x = outputs['reconstruction']
            # Use MSE for non-normal distribution too (better than BCE for log-normalized data)
            recon_loss = F.mse_loss(recon_x, x, reduction='sum') / x.size(0)
        
        # L2正则化
        l2_reg = 0.0
        if l2_scale > 0:
            for param in self.parameters():
                l2_reg += torch.sum(param ** 2)
            l2_reg = l2_scale * l2_reg
        
        # 总损失：重构损失 + KL正则化
        total_loss = recon_loss + beta * kl_loss + l2_reg
        
        return {
            'total_loss': total_loss,
            'recon_loss': recon_loss,
            'kl_loss': kl_loss,
            'l2_reg': l2_reg
        }
    
    def extract_latent(self, data_loader, device='cuda', batch_id=None,
                      return_reconstructions=False):
        """
        提取潜在表示
        
        Args:
            data_loader: 数据加载器
            device: 计算设备
            batch_id: 批次ID（用于重构）
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
                if isinstance(batch_data, (list, tuple)):
                    if len(batch_data) >= 2:
                        x, batch = batch_data[0], batch_data[1]
                    else:
                        x = batch_data[0]
                        batch = None
                else:
                    x = batch_data
                    batch = None
                
                x = x.to(device).float()
                
                mu_z, logvar_z = self.encoder_net(x)
                z = mu_z  # 使用均值作为潜在表示
                
                latents.append(z.cpu().numpy())
                mus.append(mu_z.cpu().numpy())
                
                if return_reconstructions:
                    if self.use_batch:
                        if batch_id is not None:
                            batch = torch.full((x.size(0),), batch_id, 
                                             dtype=torch.long, device=device)
                        elif batch is None:
                            batch = torch.zeros(x.size(0), dtype=torch.long, device=device)
                        else:
                            batch = batch.to(device).long()
                    
                    recon = self.decode(z, batch)
                    reconstructions.append(recon.cpu().numpy())
        
        result = {
            'latent': np.concatenate(latents, axis=0),
            'mu': np.concatenate(mus, axis=0)
        }
        
        if return_reconstructions:
            result['reconstruction'] = np.concatenate(reconstructions, axis=0)
        
        return result


# 便捷的创建函数
def create_sivae_model(input_dim: int, latent_dim: int = 50, **kwargs):
    """创建siVAE模型"""
    return siVAEModel(input_dim=input_dim, latent_dim=latent_dim, **kwargs)
