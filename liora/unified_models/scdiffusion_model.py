"""
scDiffusion的PyTorch实现
基于扩散模型的单细胞数据生成模型

扩散概率模型（DPM）通过逐步向数据添加噪声（正向过程），然后学习逆向过程来撤销
该噪声，从而进行生成建模。这个实现使用 DDPM（去噪扩散概率模型）采样策略。

Features:
- ✓ 无条件和有条件生成
- ✓ 多个扩散时间表支持
- ✓ 灵活的 UNet 架构
- ✓ 支持 ZINB/NB 数据分布

Reference:
    Gong et al. (2023) Diffusion models for single-cell gene expression 
    generation. bioRxiv.
    
    Ho et al. (2020) Denoising Diffusion Probabilistic Models. NeurIPS.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from typing import Dict, Optional
from .base_model import BaseModel


# ============================
# Utilities
# ============================

def timestep_embedding(timesteps, dim, max_period=10000):
    """
    创建正弦时间步嵌入
    
    Args:
        timesteps: [batch_size] 时间步张量
        dim: 嵌入维度
        max_period: 最大周期（默认 10000）
        
    Returns:
        [batch_size, dim] 嵌入向量
    """
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period)
        * torch.arange(half, dtype=torch.float32, device=timesteps.device)
        / half
    )
    args = timesteps[:, None].float() * freqs[None]
    emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2 == 1:
        emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
    return emb


def get_num_groups(channels):
    """自动选择 GroupNorm 的组数"""
    for g in [32, 16, 8, 4, 2]:
        if channels % g == 0:
            return g
    return 1


# ============================
# Class Embedding
# ============================

class ClassEmbedder(nn.Module):
    """
    细胞类型嵌入层
    
    用于条件扩散模型中的类别信息编码。支持训练时的条件丢弃。
    """
    def __init__(self, embed_dim, n_classes, cond_drop_prob=0.1):
        """
        Args:
            embed_dim (int): 嵌入维度
            n_classes (int): 类别数量
            cond_drop_prob (float): 条件丢弃概率（0-1，用于无分类器引导）
        """
        super().__init__()
        self.embedding = nn.Embedding(n_classes + 1, embed_dim)
        self.n_classes = n_classes
        self.cond_drop_prob = cond_drop_prob

    def forward(self, y):
        """
        Args:
            y: [batch_size] 类别标签张量
            
        Returns:
            [batch_size, embed_dim] 嵌入向量
        """
        if self.training and self.cond_drop_prob > 0:
            drop = torch.rand(y.size(0), device=y.device) < self.cond_drop_prob
            y = torch.where(drop, torch.full_like(y, self.n_classes), y)
        return self.embedding(y)


# ============================
# Residual Block
# ============================

class ResBlock(nn.Module):
    """
    残差块，包含时间步和类别条件
    
    结构:
        Input → GroupNorm → Linear → TimeProj + input
               → GroupNorm → Dropout → Linear → Skip Connection
    """
    def __init__(self, in_channels, out_channels, time_channels, dropout):
        """
        Args:
            in_channels (int): 输入通道数
            out_channels (int): 输出通道数
            time_channels (int): 时间嵌入维度
            dropout (float): Dropout 概率
        """
        super().__init__()

        self.norm1 = nn.GroupNorm(get_num_groups(in_channels), in_channels)
        self.conv1 = nn.Linear(in_channels, out_channels)

        self.time_proj = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_channels, out_channels),
        )

        self.norm2 = nn.GroupNorm(get_num_groups(out_channels), out_channels)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Linear(out_channels, out_channels)

        self.skip = (
            nn.Linear(in_channels, out_channels)
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x, emb):
        """
        Args:
            x: [batch, channels]
            emb: [batch, time_channels]
            
        Returns:
            [batch, out_channels]
        """
        h = self.norm1(x.unsqueeze(-1)).squeeze(-1)
        h = F.silu(h)
        h = self.conv1(h)

        h = h + self.time_proj(emb)

        h = self.norm2(h.unsqueeze(-1)).squeeze(-1)
        h = F.silu(h)
        h = self.dropout(h)
        h = self.conv2(h)

        return h + self.skip(x)


# ============================
# Denoising UNet
# ============================

class DenoisingUNet(nn.Module):
    """
    去噪 UNet 架构
    
    用于学习扩散过程中的噪声预测。结构包括：
    - 输入投影层
    - 下采样路径（编码器）
    - 中间瓶颈
    - 上采样路径（解码器），带跳连接
    - 输出投影层
    
    支持无条件和有条件（细胞类型）生成。
    """
    def __init__(
        self,
        input_dim,
        model_channels,
        num_res_blocks,
        dropout,
        channel_mult,
        n_classes=0,
        cond_drop_prob=0.1,
    ):
        """
        Args:
            input_dim (int): 输入维度（基因数）
            model_channels (int): 基础通道数
            num_res_blocks (int): 每个级别的残差块数
            dropout (float): Dropout 概率
            channel_mult (tuple): 每个级别的通道倍数。
                例如 (1, 2, 3, 4) 表示 4 个下采样级别
            n_classes (int, optional): 细胞类型数（0=无条件）。默认为 0
            cond_drop_prob (float, optional): 条件丢弃概率。默认为 0.1
        """
        super().__init__()

        self.channel_mult = channel_mult
        self.num_res_blocks = num_res_blocks
        self.model_channels = model_channels

        time_dim = model_channels * 4
        self.time_embed = nn.Sequential(
            nn.Linear(model_channels, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )

        self.class_embedder = (
            ClassEmbedder(time_dim, n_classes, cond_drop_prob)
            if n_classes > 0
            else None
        )

        self.input_proj = nn.Linear(input_dim, model_channels)

        # 下采样路径
        self.down_blocks = nn.ModuleList()
        self.downsample = nn.ModuleList()
        self.skip_channels = []

        ch = model_channels
        for level, mult in enumerate(channel_mult):
            out_ch = model_channels * mult
            for _ in range(num_res_blocks):
                self.down_blocks.append(ResBlock(ch, out_ch, time_dim, dropout))
                ch = out_ch
                self.skip_channels.append(ch)
            self.downsample.append(nn.Linear(ch, ch) if level < len(channel_mult) - 1 else None)

        # 中间瓶颈
        self.mid1 = ResBlock(ch, ch, time_dim, dropout)
        self.mid2 = ResBlock(ch, ch, time_dim, dropout)

        # Expose bottleneck dimension for latent extraction
        self.bottleneck_dim = ch

        # 上采样路径
        self.up_blocks = nn.ModuleList()
        self.upsample = nn.ModuleList()

        for level, mult in reversed(list(enumerate(channel_mult))):
            out_ch = model_channels * mult
            for _ in range(num_res_blocks):
                skip_ch = self.skip_channels.pop()
                self.up_blocks.append(ResBlock(ch + skip_ch, out_ch, time_dim, dropout))
                ch = out_ch
            self.upsample.append(nn.Linear(ch, ch) if level > 0 else None)

        self.out_norm = nn.GroupNorm(get_num_groups(ch), ch)
        self.out_proj = nn.Linear(ch, input_dim)

    def _build_cond_embedding(self, t: torch.Tensor, y: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Build (time [+ class]) conditioning embedding."""
        emb = self.time_embed(timestep_embedding(t, self.time_embed[0].in_features))
        if self.class_embedder is not None and y is not None:
            emb = emb + self.class_embedder(y)
        return emb

    def extract_bottleneck(self, x: torch.Tensor, t: torch.Tensor, y: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Extract UNet bottleneck features for a noisy sample x_t.

        Args:
            x: [batch, input_dim] noisy sample x_t
            t: [batch] timesteps
            y: [batch] optional class labels

        Returns:
            [batch, bottleneck_dim] bottleneck representation
        """
        emb = self._build_cond_embedding(t, y)
        h = self.input_proj(x)

        down_idx = 0
        for level in range(len(self.channel_mult)):
            for _ in range(self.num_res_blocks):
                h = self.down_blocks[down_idx](h, emb)
                down_idx += 1
            if self.downsample[level] is not None:
                h = self.downsample[level](h)

        h = self.mid1(h, emb)
        h = self.mid2(h, emb)
        return h

    def forward(self, x, t, y=None):
        """
        Args:
            x: [batch, input_dim] 噪声样本
            t: [batch] 时间步
            y: [batch] 类别标签（可选）
            
        Returns:
            [batch, input_dim] 预测噪声
        """
        emb = self._build_cond_embedding(t, y)

        h = self.input_proj(x)
        hs = []

        # 下采样
        down_idx = 0
        for level in range(len(self.channel_mult)):
            for _ in range(self.num_res_blocks):
                h = self.down_blocks[down_idx](h, emb)
                hs.append(h)
                down_idx += 1
            if self.downsample[level] is not None:
                h = self.downsample[level](h)

        # 中间
        h = self.mid1(h, emb)
        h = self.mid2(h, emb)

        # 上采样（带跳连接）
        up_idx = 0
        for level in range(len(self.channel_mult)):
            for _ in range(self.num_res_blocks):
                h = torch.cat([h, hs.pop()], dim=-1)
                h = self.up_blocks[up_idx](h, emb)
                up_idx += 1
            if self.upsample[level] is not None:
                h = self.upsample[level](h)

        h = self.out_norm(h.unsqueeze(-1)).squeeze(-1)
        h = F.silu(h)
        return self.out_proj(h)


# ============================
# scDiffusion Model
# ============================

class scDiffusionModel(BaseModel):
    """
    scDiffusion 模型 - 基于扩散模型的单细胞生成模型
    
    这是一个实现了 DDPM（去噪扩散概率模型）的生成模型，专门用于
    单细胞 RNA-seq 数据生成。支持无条件和有条件（细胞类型）生成。
    
    **工作原理：**
    
    1. **正向过程（Forward）**：逐步向真实数据添加高斯噪声
       - x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * epsilon
       
    2. **反向过程（Reverse）**：学习预测并去除噪声
       - 使用 UNet 预测每一步的噪声
       - 通过迭代逆转正向过程来生成新样本
    
    3. **条件生成**：可以按细胞类型进行条件生成
    
    **关键参数：**
    - n_timesteps: 扩散步数（通常 1000）
    - beta_schedule: 噪声方差时间表（'linear' 或 'cosine'）
    - channel_mult: UNet 每级的通道倍数
    
    **典型用法：**
    
    .. code-block:: python
    
        # 创建模型
        model = scDiffusionModel(
            input_dim=2000,  # 基因数
            latent_dim=128,  # UNet 基础通道数
            n_timesteps=1000,
            n_classes=10,  # 细胞类型数
            beta_schedule='linear'
        )
        
        # 训练
        history = model.fit(
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=100,
            lr=1e-4,
            device='cuda',
            verbose=1  # 仅输出每个 epoch 的损失
        )
        
        # 生成新样本
        with torch.no_grad():
            # 无条件生成
            samples = model.p_sample_loop(batch_size=32)
            
            # 有条件生成（例如，生成 B 细胞）
            cell_types = torch.full((32,), 5, dtype=torch.long)
            samples = model.p_sample_loop(
                batch_size=32, 
                y=cell_types
            )
    
    References:
        - Gong et al. (2023): Diffusion models for single-cell...
        - Ho et al. (2020): Denoising Diffusion Probabilistic Models. NeurIPS.
    
    Attributes:
        denoising_net (DenoisingUNet): 去噪 UNet
        n_timesteps (int): 扩散步数
        n_classes (int): 细胞类型数（0=无条件）
        loss_type (str): 损失函数类型
    """
    
    def __init__(
        self,
        input_dim,
        latent_dim=128,
        embedding_dim: int = 10,
        hidden_dims=None,
        n_timesteps=1000,
        beta_schedule="linear",
        n_classes=0,
        cond_drop_prob=0.1,
        loss_type="mse",
        model_name="scDiffusion",
    ):
        """
        初始化 scDiffusion 模型
        
        Args:
            input_dim (int): 基因数量
            latent_dim (int, optional): UNet 基础通道数。默认为 128
            hidden_dims (list, optional): UNet 每级的通道倍数。
                例如 [1, 2, 3, 4] 表示 4 个下采样级别，
                通道数分别为 128, 256, 384, 512。
                默认为 [1, 2, 3, 4]
            n_timesteps (int, optional): 扩散步数。默认为 1000
            beta_schedule (str, optional): 噪声方差时间表。
                - 'linear': 线性时间表（简单但效果一般）
                - 'cosine': 余弦时间表（推荐，更稳定）
                默认为 'linear'
            n_classes (int, optional): 细胞类型数量。0 = 无条件生成。
                默认为 0
            cond_drop_prob (float, optional): 条件丢弃概率（用于改进无分类器引导）。
                默认为 0.1
            loss_type (str, optional): 损失函数类型。
                - 'mse': 均方误差（标准）
                - 'l1': L1 损失
                - 'hybrid': MSE + 0.1*L1
                默认为 'mse'
            model_name (str, optional): 模型名称。默认为 'scDiffusion'
        
        Examples:
            >>> # 创建基础扩散模型
            >>> model = scDiffusionModel(input_dim=2000)
            >>> 
            >>> # 创建条件扩散模型（带细胞类型）
            >>> model = scDiffusionModel(
            ...     input_dim=2000,
            ...     n_classes=15,  # 15 种细胞类型
            ...     n_timesteps=1000,
            ...     beta_schedule='cosine'
            ... )
            >>> 
            >>> # 自定义 UNet 架构
            >>> model = scDiffusionModel(
            ...     input_dim=2000,
            ...     latent_dim=256,  # 更大的通道数
            ...     hidden_dims=[1, 2, 4],  # 3 个级别
            ...     n_classes=10
            ... )
        """
        hidden_dims = hidden_dims or [1, 2, 3, 4]
        super().__init__(input_dim, latent_dim, hidden_dims, model_name)

        self.n_timesteps = n_timesteps
        self.loss_type = loss_type
        self.n_classes = n_classes
        self.embedding_dim = embedding_dim

        self.denoising_net = DenoisingUNet(
            input_dim=input_dim,
            model_channels=latent_dim,
            num_res_blocks=2,
            dropout=0.1,
            channel_mult=tuple(hidden_dims),
            n_classes=n_classes,
            cond_drop_prob=cond_drop_prob,
        )

        self.latent_head = nn.Linear(self.denoising_net.bottleneck_dim, embedding_dim)

        self._setup_diffusion_schedule(beta_schedule)

    def _prepare_batch(self, batch_data, device):
        """
        准备批次数据
        
        期望输入格式：(x, y) 或仅 x
        """
        if isinstance(batch_data, (list, tuple)) and len(batch_data) >= 2:
            x = batch_data[0].to(device).float()
            y = batch_data[1].to(device).long()
            return x, {"y": y}
        else:
            x = batch_data.to(device).float()
            return x, {}

    def _setup_diffusion_schedule(self, schedule_type: str = 'linear'):
        """
        设置扩散时间表
        
        Args:
            schedule_type (str): 时间表类型
                - 'linear': 线性
                - 'cosine': 余弦（推荐）
        """
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
        
        self.register_buffer('betas', betas)
        self.register_buffer('alphas', alphas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1 - alphas_cumprod))
        self.register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1.0 / alphas_cumprod))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', torch.sqrt(1.0 / alphas_cumprod - 1))
    
    def q_sample(self, x_0: torch.Tensor, t: torch.Tensor, noise: Optional[torch.Tensor] = None):
        """
        正向扩散过程：向数据添加噪声
        
        Args:
            x_0 (torch.Tensor): [batch, input_dim] 原始数据
            t (torch.Tensor): [batch] 时间步
            noise (torch.Tensor, optional): [batch, input_dim] 高斯噪声。
                如果为 None，则随机采样。
            
        Returns:
            torch.Tensor: [batch, input_dim] 噪声加入的样本
        
        Formula:
            x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * epsilon
        """
        if noise is None:
            noise = torch.randn_like(x_0)
        
        sqrt_alpha_cumprod_t = self.sqrt_alphas_cumprod[t][:, None]
        sqrt_one_minus_alpha_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t][:, None]
        
        return sqrt_alpha_cumprod_t * x_0 + sqrt_one_minus_alpha_cumprod_t * noise
    
    def encode(self, x: torch.Tensor, t: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        编码（添加噪声）
        
        Args:
            x (torch.Tensor): [batch, input_dim] 原始数据
            t (torch.Tensor, optional): [batch] 时间步。
                如果为 None，使用中间时间步。
            
        Returns:
            torch.Tensor: [batch, input_dim] 噪声样本
        """
        if t is None:
            t = torch.ones(x.size(0), dtype=torch.long, device=x.device) * (self.n_timesteps // 2)
        
        return self.q_sample(x, t)
    
    def decode(self, z: torch.Tensor, y: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        解码（去噪采样）
        
        Args:
            z (torch.Tensor): [batch, input_dim] 起始噪声或中间表示
            y (torch.Tensor, optional): [batch] 细胞类型标签（用于条件生成）
            
        Returns:
            torch.Tensor: [batch, input_dim] 生成样本
        """
        return self.p_sample_loop(z.size(0), z.device, init_x=z, y=y)
    
    def forward(self, x: torch.Tensor, y: Optional[torch.Tensor] = None, 
                **kwargs) -> Dict[str, torch.Tensor]:
        """
        前向传播（训练）
        
        Args:
            x (torch.Tensor): [batch, input_dim] 真实数据
            y (torch.Tensor, optional): [batch] 细胞类型标签
            **kwargs: 额外参数
            
        Returns:
            Dict[str, torch.Tensor]: 包含以下键：
                - 'predicted_noise': 模型预测的噪声
                - 'true_noise': 真实添加的噪声
                - 'x_noisy': 噪声样本
                - 't': 时间步
                - 'y': 细胞类型标签
        """
        batch_size = x.size(0)
        device = x.device
        
        # 随机时间步
        t = torch.randint(0, self.n_timesteps, (batch_size,), device=device)
        
        # 采样噪声
        noise = torch.randn_like(x)
        
        # 添加噪声
        x_noisy = self.q_sample(x, t, noise)
        
        # 预测噪声
        predicted_noise = self.denoising_net(x_noisy, t, y)
        
        return {
            'predicted_noise': predicted_noise,
            'true_noise': noise,
            'x_noisy': x_noisy,
            't': t,
            'y': y
        }
    
    def compute_loss(self, x: torch.Tensor, outputs: Dict[str, torch.Tensor],
                     **kwargs) -> Dict[str, torch.Tensor]:
        """
        计算扩散损失
        
        损失函数是真实噪声与模型预测噪声之间的距离。
        
        Args:
            x (torch.Tensor): [batch, input_dim] 真实数据（用于索引）
            outputs (Dict[str, torch.Tensor]): forward() 的输出
            **kwargs: 额外参数
            
        Returns:
            Dict[str, torch.Tensor]: 包含：
                - 'total_loss': 总损失
                - 'recon_loss': 重构损失（同 total_loss）
                - 'diffusion_loss': 扩散损失
        """
        predicted_noise = outputs['predicted_noise']
        true_noise = outputs['true_noise']
        
        # 计算损失
        if self.loss_type == 'mse':
            loss = F.mse_loss(predicted_noise, true_noise, reduction='mean')
        elif self.loss_type == 'l1':
            loss = F.l1_loss(predicted_noise, true_noise, reduction='mean')
        elif self.loss_type == 'hybrid':
            loss = F.mse_loss(predicted_noise, true_noise, reduction='mean') + \
                   0.1 * F.l1_loss(predicted_noise, true_noise, reduction='mean')
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")
        
        return {
            'total_loss': loss,
            'recon_loss': loss,
            'diffusion_loss': loss
        }
    
    @torch.no_grad()
    def p_sample(self, x: torch.Tensor, t: int, y: Optional[torch.Tensor] = None,
                 clip_denoised: bool = True):
        """
        单步去噪（DDPM）
        
        Args:
            x (torch.Tensor): [batch, input_dim] t 时刻的样本
            t (int): 当前时间步
            y (torch.Tensor, optional): [batch] 细胞类型标签
            clip_denoised (bool): 是否夹取去噪样本到 [-1, 1]
            
        Returns:
            torch.Tensor: [batch, input_dim] t-1 时刻的样本
        """
        batch_size = x.size(0)
        t_tensor = torch.full((batch_size,), t, device=x.device, dtype=torch.long)
        
        # 预测噪声
        predicted_noise = self.denoising_net(x, t_tensor, y)
        
        # 计算去噪样本
        alpha = self.alphas[t]
        alpha_cumprod = self.alphas_cumprod[t]
        beta = self.betas[t]
        
        # p(x_{t-1} | x_t) 的均值
        mean = (x - beta / torch.sqrt(1 - alpha_cumprod) * predicted_noise) / torch.sqrt(alpha)
        
        if clip_denoised:
            mean = torch.clamp(mean, -1, 1)
        
        if t > 0:
            noise = torch.randn_like(x)
            variance = beta
            return mean + torch.sqrt(variance) * noise
        else:
            return mean
    
    @torch.no_grad()
    def p_sample_loop(self, batch_size: int, device: str = 'cuda',
                     init_x: Optional[torch.Tensor] = None,
                     y: Optional[torch.Tensor] = None,
                     clip_denoised: bool = True):
        """
        完整采样循环（DDPM）
        
        从纯噪声开始，迭代地去噪以生成样本。
        
        Args:
            batch_size (int): 生成样本数
            device (str): 计算设备。默认为 'cuda'
            init_x (torch.Tensor, optional): [batch, input_dim] 初始样本。
                如果为 None，使用纯高斯噪声。
            y (torch.Tensor, optional): [batch] 细胞类型标签（用于条件生成）
            clip_denoised (bool): 是否夹取去噪样本
            
        Returns:
            torch.Tensor: [batch, input_dim] 生成样本
        
        Examples:
            >>> # 无条件生成
            >>> samples = model.p_sample_loop(batch_size=100, device='cuda')
            >>> 
            >>> # 生成特定细胞类型（例如细胞类型 5）
            >>> cell_type = torch.full((100,), 5, dtype=torch.long)
            >>> samples = model.p_sample_loop(
            ...     batch_size=100,
            ...     device='cuda',
            ...     y=cell_type
            ... )
        """
        if init_x is None:
            x = torch.randn(batch_size, self.input_dim, device=device)
        else:
            x = init_x
        
        # 从 T-1 到 0 逐步去噪
        for t in reversed(range(self.n_timesteps)):
            x = self.p_sample(x, t, y, clip_denoised)
        
        return x
    
    def extract_latent(self, data_loader, device='cuda', timestep=None,
                      return_reconstructions=False):
        """
        提取低维表示（在特定扩散时间步）
        
        Args:
            data_loader: 数据加载器
            device (str): 计算设备。默认为 'cuda'
            timestep (int, optional): 扩散时间步。
                如果为 None，使用中间时间步（n_timesteps//2）
            return_reconstructions (bool): 是否返回重构（完整去噪过程）
            
        Returns:
            Dict[str, np.ndarray]: 包含：
                - 'latent': 低维表示，形状 [n_samples, embedding_dim]
                - 'reconstruction': 去噪样本（如果 return_reconstructions=True）
                - 'labels': 细胞类型标签（如果提供）
        
        Examples:
            >>> # 在中间时间步提取表示
            >>> result = model.extract_latent(test_loader)
            >>> latent = result['latent']
            >>> 
            >>> # 完整去噪
            >>> result = model.extract_latent(
            ...     test_loader,
            ...     return_reconstructions=True
            ... )
        """
        self.eval()
        self.to(device)
        
        if timestep is None:
            timestep = self.n_timesteps // 2
        
        latents = []
        reconstructions = [] if return_reconstructions else None
        labels = []
        
        with torch.no_grad():
            for batch_data in data_loader:
                if isinstance(batch_data, (list, tuple)):
                    x = batch_data[0].to(device).float()
                    y = batch_data[1].to(device).long() if len(batch_data) > 1 else None
                else:
                    x = batch_data.to(device).float()
                    y = None
                
                t = torch.full((x.size(0),), timestep, device=device, dtype=torch.long)
                
                # 编码（添加噪声）
                z = self.q_sample(x, t)

                # UNet 瓶颈特征 → 低维嵌入
                h = self.denoising_net.extract_bottleneck(z, t, y=y)
                emb = self.latent_head(h)
                latents.append(emb.cpu().numpy())
                
                if y is not None:
                    labels.append(y.cpu().numpy())
                
                # 重构（如需）
                if return_reconstructions:
                    recon = self.p_sample_loop(x.size(0), device, init_x=z, y=y)
                    reconstructions.append(recon.cpu().numpy())
        
        result = {'latent': np.concatenate(latents, axis=0)}
        
        if len(labels) > 0:
            result['labels'] = np.concatenate(labels, axis=0)
        
        if return_reconstructions:
            result['reconstruction'] = np.concatenate(reconstructions, axis=0)
        
        return result


def create_scdiffusion_model(input_dim: int, latent_dim: int = 128,
                             n_classes: int = 0, embedding_dim: int = 10, **kwargs):
    """
    创建 scDiffusion 模型
    
    Args:
        input_dim (int): 基因数量
        latent_dim (int, optional): UNet 基础通道数。默认为 128
        n_classes (int, optional): 细胞类型数（0=无条件）。默认为 0
        **kwargs: 额外参数（n_timesteps, beta_schedule, loss_type 等）
    
    Returns:
        scDiffusionModel: 初始化好的模型
    
    Examples:
        >>> # 无条件扩散模型
        >>> model = create_scdiffusion_model(input_dim=2000)
        >>> 
        >>> # 条件扩散模型（15 种细胞类型）
        >>> model = create_scdiffusion_model(
        ...     input_dim=2000,
        ...     latent_dim=256,
        ...     n_classes=15,
        ...     n_timesteps=1000,
        ...     beta_schedule='cosine'
        ... )
        >>> 
        >>> # 训练
        >>> history = model.fit(
        ...     train_loader=train_loader,
        ...     val_loader=val_loader,
        ...     epochs=100,
        ...     lr=1e-4,
        ...     device='cuda',
        ...     verbose=1
        ... )
        >>> 
        >>> # 生成样本
        >>> with torch.no_grad():
        ...     samples = model.p_sample_loop(batch_size=100)
    """
    return scDiffusionModel(input_dim=input_dim, latent_dim=latent_dim,
                           n_classes=n_classes, embedding_dim=embedding_dim, **kwargs)