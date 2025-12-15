"""
scDiffusion的PyTorch实现
基于扩散模型的单细胞数据生成模型
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional
import numpy as np
from .base_model import BaseModel


class TimeEmbedding(nn.Module):
    """时间步嵌入"""
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.time_embed = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
    
    def forward(self, t):
        """生成时间步嵌入"""
        # Sinusoidal position encoding
        half_dim = self.hidden_dim // 2
        emb = np.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=t.device) * -emb)
        emb = t[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        
        if self.hidden_dim % 2 == 1:
            emb = F.pad(emb, (0, 1))
        
        return self.time_embed(emb)


class ResidualBlock(nn.Module):
    """残差块"""
    def __init__(self, in_features: int, out_features: int, time_features: int):
        super().__init__()
        self.fc = nn.Linear(in_features, out_features)
        self.norm = nn.LayerNorm(out_features)
        self.emb_layer = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_features, out_features),
        )
        self.act = nn.SiLU()
        self.drop = nn.Dropout(0.1)
    
    def forward(self, x, emb):
        h = self.fc(x)
        h = h + self.emb_layer(emb)
        h = self.norm(h)
        h = self.act(h)
        h = self.drop(h)
        return h


class DenoisingUNet(nn.Module):
    """去噪U-Net"""
    def __init__(self, input_dim: int, hidden_dims: list = None, dropout: float = 0.1):
        super().__init__()
        
        if hidden_dims is None:
            hidden_dims = [2000, 1000, 500, 500]
        
        self.hidden_dims = hidden_dims
        self.time_embedding = TimeEmbedding(hidden_dims[0])
        
        # 编码器
        self.encoder_layers = nn.ModuleList()
        self.encoder_layers.append(ResidualBlock(input_dim, hidden_dims[0], hidden_dims[0]))
        for i in range(len(hidden_dims) - 1):
            self.encoder_layers.append(ResidualBlock(hidden_dims[i], hidden_dims[i+1], hidden_dims[0]))
        
        # 解码器
        self.decoder_layers = nn.ModuleList()
        for i in reversed(range(len(hidden_dims) - 1)):
            self.decoder_layers.append(ResidualBlock(hidden_dims[i+1], hidden_dims[i], hidden_dims[0]))
        
        # 输出层
        self.out1 = nn.Linear(hidden_dims[0], hidden_dims[1] * 2)
        self.norm_out = nn.LayerNorm(hidden_dims[1] * 2)
        self.out2 = nn.Linear(hidden_dims[1] * 2, input_dim)
        self.act = nn.SiLU()
        self.drop = nn.Dropout(dropout)
    
    def forward(self, x, t, y=None):
        """
        Args:
            x: 输入 [batch_size, input_dim]
            t: 时间步 [batch_size]
            y: 条件（可选）
        """
        emb = self.time_embedding(t)
        
        # 编码
        history = []
        for layer in self.encoder_layers:
            x = layer(x, emb)
            history.append(x)
        
        history.pop()
        
        # 解码（带跳跃连接）
        for layer in self.decoder_layers:
            x = layer(x, emb)
            x = x + history.pop()
        
        # 输出
        x = self.out1(x)
        x = self.norm_out(x)
        x = self.act(x)
        x = self.out2(x)
        
        return x


class scDiffusionModel(BaseModel):
    """
    scDiffusion模型的PyTorch实现
    使用扩散模型生成单细胞基因表达数据
    """
    
    def __init__(self,
                 input_dim: int,
                 latent_dim: int = 500,  # 潜在空间维度
                 hidden_dims: list = None,
                 n_timesteps: int = 1000,
                 beta_schedule: str = 'linear',
                 model_name: str = "scDiffusion"):
        """
        Args:
            input_dim: 输入特征维度
            latent_dim: 潜在空间维度（用户指定的最终latent维度）
            hidden_dims: U-Net隐藏层维度（瓶颈层是hidden_dims[-1]）
            n_timesteps: 扩散步数
            beta_schedule: beta调度方式
            model_name: 模型名称
        """
        if hidden_dims is None:
            hidden_dims = [2000, 1000, 500, 500]
        
        super().__init__(input_dim, latent_dim, hidden_dims, model_name)
        
        self.n_timesteps = n_timesteps
        self.bottleneck_dim = hidden_dims[-1]  # 瓶颈层的实际维度
        
        # 去噪网络
        self.denoising_net = DenoisingUNet(input_dim, hidden_dims)
        
        # ✅ 编码投影：bottleneck_dim -> latent_dim
        self.bottleneck_to_latent = nn.Sequential(
            nn.Linear(self.bottleneck_dim, latent_dim),
            nn.LayerNorm(latent_dim),
            nn.SiLU()
        )
        
        # ✅ 解码投影1：latent_dim -> bottleneck_dim
        self.latent_to_bottleneck = nn.Sequential(
            nn.Linear(latent_dim, self.bottleneck_dim),
            nn.LayerNorm(self.bottleneck_dim),
            nn.SiLU()
        )
        
        # ✅ 解码投影2：bottleneck_dim -> input_dim (用于生成去噪起点)
        self.bottleneck_to_input = nn.Sequential(
            nn.Linear(self.bottleneck_dim, hidden_dims[0]),
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dims[0], input_dim)
        )
        
        # 设置扩散参数
        self._setup_diffusion_schedule(beta_schedule)
    
    def _setup_diffusion_schedule(self, schedule_type: str = 'linear'):
        """设置扩散调度"""
        if schedule_type == 'linear':
            betas = torch.linspace(1e-4, 0.02, self.n_timesteps)
        elif schedule_type == 'cosine':
            steps = self.n_timesteps + 1
            x = torch.linspace(0, self.n_timesteps, steps)
            alphas_cumprod = torch.cos(((x / self.n_timesteps) + 0.008) / 1.008 * np.pi * 0.5) ** 2
            alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
            betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
            betas = torch.clip(betas, 0.0001, 0.9999)
        else:
            raise ValueError(f"Unknown schedule: {schedule_type}")
        
        alphas = 1 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
        
        # 注册为buffer（不作为参数训练，但会保存在模型中）
        self.register_buffer('betas', betas)
        self.register_buffer('alphas', alphas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1 - alphas_cumprod))
    
    def q_sample(self, x_0: torch.Tensor, t: torch.Tensor, noise: Optional[torch.Tensor] = None):
        """前向扩散过程：给x_0添加噪声"""
        if noise is None:
            noise = torch.randn_like(x_0)
        
        sqrt_alpha_cumprod_t = self.sqrt_alphas_cumprod[t][:, None]
        sqrt_one_minus_alpha_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t][:, None]
        
        return sqrt_alpha_cumprod_t * x_0 + sqrt_one_minus_alpha_cumprod_t * noise
    
    def get_bottleneck_features(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        ✅ 提取去噪网络瓶颈层的特征并投影到latent_dim
        
        Args:
            x: 输入数据 [batch_size, input_dim]
            t: 时间步 [batch_size]
            
        Returns:
            潜在表示 [batch_size, latent_dim]
        """
        emb = self.denoising_net.time_embedding(t)
        
        # 前向传播到瓶颈层
        h = x
        for layer in self.denoising_net.encoder_layers:
            h = layer(h, emb)
        
        # ✅ 投影到用户指定的latent_dim
        h = self.bottleneck_to_latent(h)
        
        return h  # [batch_size, latent_dim]
    
    def encode(self, x: torch.Tensor, use_learned_features: bool = True) -> torch.Tensor:
        """
        ✅ 编码到latent_dim维度的潜在空间
        
        Args:
            x: 输入数据 [batch_size, input_dim]
            use_learned_features: 是否使用学习特征
            
        Returns:
            潜在表示 [batch_size, latent_dim]
        """
        t = torch.ones(x.size(0), dtype=torch.long, device=x.device) * (self.n_timesteps // 2)
        
        if use_learned_features:
            # 1. 添加噪声
            x_noisy = self.q_sample(x, t)
            # 2. 通过网络提取特征并投影到latent_dim
            return self.get_bottleneck_features(x_noisy, t)
        else:
            # 旧行为：只返回加噪后的数据（维度仍是input_dim，不推荐）
            return self.q_sample(x, t)
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """
        ✅ 从latent_dim的潜在表示解码（使用去噪循环）
        
        Args:
            z: 潜在表示 [batch_size, latent_dim]
            
        Returns:
            重构数据 [batch_size, input_dim]
        """
        device = z.device
        
        # Step 1: latent_dim -> bottleneck_dim
        z_bottleneck = self.latent_to_bottleneck(z)  # [batch_size, bottleneck_dim]
        
        # Step 2: bottleneck_dim -> input_dim (作为去噪的起点)
        x_init = self.bottleneck_to_input(z_bottleneck)  # [batch_size, input_dim]
        
        # Step 3: 从部分噪声状态开始去噪
        # 不从完全噪声(t=n_timesteps)开始，而是从中间时间步开始
        start_t = self.n_timesteps // 3  # 从1/3处开始去噪
        
        x = x_init
        for t in reversed(range(start_t)):
            x = self.p_sample(x, t)
        
        return x
    
    def forward(self, x: torch.Tensor, **kwargs) -> Dict[str, torch.Tensor]:
        """
        训练时的前向传播
        
        Args:
            x: 输入数据 [batch_size, input_dim]
            
        Returns:
            预测的噪声和相关信息
        """
        batch_size = x.size(0)
        device = x.device
        
        # 随机采样时间步
        t = torch.randint(0, self.n_timesteps, (batch_size,), device=device)
        
        # 采样噪声
        noise = torch.randn_like(x)
        
        # 添加噪声
        x_noisy = self.q_sample(x, t, noise)
        
        # 预测噪声
        predicted_noise = self.denoising_net(x_noisy, t)
        
        return {
            'predicted_noise': predicted_noise,
            'true_noise': noise,
            'x_noisy': x_noisy,
            't': t
        }
    
    def compute_loss(self, x: torch.Tensor, outputs: Dict[str, torch.Tensor],
                     **kwargs) -> Dict[str, torch.Tensor]:
        """计算扩散损失"""
        predicted_noise = outputs['predicted_noise']
        true_noise = outputs['true_noise']
        
        # MSE损失
        loss = F.mse_loss(predicted_noise, true_noise)
        
        return {
            'total_loss': loss,
            'recon_loss': loss,
            'diffusion_loss': loss
        }
    
    @torch.no_grad()
    def p_sample(self, x: torch.Tensor, t: int):
        """单步去噪"""
        batch_size = x.size(0)
        t_tensor = torch.full((batch_size,), t, device=x.device, dtype=torch.long)
        
        # 预测噪声
        predicted_noise = self.denoising_net(x, t_tensor)
        
        # 计算均值
        alpha = self.alphas[t]
        alpha_cumprod = self.alphas_cumprod[t]
        beta = self.betas[t]
        
        mean = (x - beta / torch.sqrt(1 - alpha_cumprod) * predicted_noise) / torch.sqrt(alpha)
        
        if t > 0:
            noise = torch.randn_like(x)
            variance = beta
            return mean + torch.sqrt(variance) * noise
        else:
            return mean
    
    @torch.no_grad()
    def p_sample_loop(self, batch_size: int, device: str = 'cuda', 
                     init_x: Optional[torch.Tensor] = None):
        """完整的采样过程"""
        if init_x is None:
            x = torch.randn(batch_size, self.input_dim, device=device)
        else:
            x = init_x
        
        for t in reversed(range(self.n_timesteps)):
            x = self.p_sample(x, t)
        
        return x
    
    def extract_latent(self, data_loader, device='cuda', timestep=None,
                      return_reconstructions=False, use_learned_features=True):
        """
        ✅ 提取latent_dim维度的潜在表示
        
        Args:
            data_loader: 数据加载器
            device: 计算设备
            timestep: 扩散到哪个时间步（None表示中间步）
            return_reconstructions: 是否返回重构
            use_learned_features: 是否使用学习到的特征
            
        Returns:
            包含潜在表示和可选重构的字典
            - latent: [n_samples, latent_dim]  ✅ 现在是latent_dim维度
            - reconstruction: [n_samples, input_dim] (可选)
        """
        self.eval()
        self.to(device)
        
        if timestep is None:
            timestep = self.n_timesteps // 2
        
        latents = []
        reconstructions = [] if return_reconstructions else None
        
        with torch.no_grad():
            for batch_data in data_loader:
                if isinstance(batch_data, (list, tuple)):
                    x = batch_data[0]
                else:
                    x = batch_data
                
                x = x.to(device).float()
                t = torch.full((x.size(0),), timestep, device=device, dtype=torch.long)
                
                if use_learned_features:
                    # ✅ 使用encode方法，返回[batch_size, latent_dim]
                    z = self.encode(x, use_learned_features=True)
                else:
                    # 旧方法：只添加噪声
                    z = self.q_sample(x, t)
                
                latents.append(z.cpu().numpy())
                
                if return_reconstructions:
                    if use_learned_features:
                        # ✅ 使用decode方法从latent_dim重构
                        recon = self.decode(z)
                    else:
                        # 从噪声数据进行完整去噪
                        recon = self.p_sample_loop(x.size(0), device, init_x=z)
                    reconstructions.append(recon.cpu().numpy())
        
        result = {'latent': np.concatenate(latents, axis=0)}
        if return_reconstructions:
            result['reconstruction'] = np.concatenate(reconstructions, axis=0)
        
        return result


# 便捷的创建函数
def create_scdiffusion_model(input_dim: int, latent_dim: int = 500, **kwargs):
    """创建scDiffusion模型"""
    return scDiffusionModel(input_dim=input_dim, latent_dim=latent_dim, **kwargs)