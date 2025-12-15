
"""
GM-VAE的完整PyTorch实现
支持5种几何分布的几何变分自编码器
Supports: Euclidean, Poincaré, PGM, LearnablePGM, and HW (Hyperboloid Wrapped) distributions
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Union, Tuple
import numpy as np
from .base_model import BaseModel

# Import distribution modules - now much cleaner
from liora.distributions.EuclideanNormal import (
    Distribution as EuclideanDistribution,
    VanillaEncoderLayer as EuclideanEncoderLayer,
    VanillaDecoderLayer as EuclideanDecoderLayer,
    get_prior as get_euclidean_prior
)

from liora.distributions.PoincareNormal import (
    Distribution as PoincareDistribution,
    VanillaEncoderLayer as PoincareEncoderLayer,
    VanillaDecoderLayer as PoincareDecoderLayer,
    get_prior as get_poincare_prior
)

from liora.distributions.PGMNormal import (
    Distribution as PGMDistribution,
    VanillaEncoderLayer as PGMVanillaEncoderLayer,
    GeoEncoderLayer as PGMGeoEncoderLayer,
    VanillaDecoderLayer as PGMVanillaDecoderLayer,
    GeoDecoderLayer as PGMGeoDecoderLayer,
    get_prior as get_pgm_prior
)

from liora.distributions.LearnablePGMNormal import (
    Distribution as LearnablePGMDistribution,
    VanillaEncoderLayer as LearnablePGMVanillaEncoderLayer,
    ExpEncoderLayer as LearnablePGMExpEncoderLayer,
    VanillaDecoderLayer as LearnablePGMVanillaDecoderLayer,
    LogDecoderLayer as LearnablePGMLogDecoderLayer,
    get_prior as get_learnable_pgm_prior
)

from liora.distributions.HWNormal import (
    Distribution as HWDistribution,
    VanillaEncoderLayer as HWEncoderLayer,
    VanillaDecoderLayer as HWDecoderLayer,
    get_prior as get_hw_prior
)

class SimpleArgs:
    """Simple argument container to interface with distribution modules"""
    def __init__(self, latent_dim, device, c=-1.0):
        self.latent_dim = latent_dim
        self.device = device
        self.c = c  # Curvature parameter


# Distribution configuration mapping
DISTRIBUTION_CONFIG = {
    'euclidean': {
        'distribution_class': EuclideanDistribution,
        'encoder_layers': {'Vanilla': EuclideanEncoderLayer},
        'decoder_layers': {'Vanilla': EuclideanDecoderLayer},
        'get_prior': get_euclidean_prior,
        'default_layer': 'Vanilla',
        'param_count': 2,  # mean, logvar (scalar each)
        'internal_multiplier': 2,  # Uses latent_dim * 2 internally
        'output_shape': 'flat',  # [batch, latent_dim]
    },
    'poincare': {
        'distribution_class': PoincareDistribution,
        'encoder_layers': {'Vanilla': PoincareEncoderLayer},
        'decoder_layers': {'Vanilla': PoincareDecoderLayer},
        'get_prior': get_poincare_prior,
        'default_layer': 'Vanilla',
        'param_count': 2,  # mean (2D point), sigma (scalar per point)
        'internal_multiplier': 2,  # latent_dim must be even
        'output_shape': '2d',  # [batch, latent_dim//2, 2]
    },
    'pgm': {
        'distribution_class': PGMDistribution,
        'encoder_layers': {
            'Vanilla': PGMVanillaEncoderLayer,
            'Geo': PGMGeoEncoderLayer
        },
        'decoder_layers': {
            'Vanilla': PGMVanillaDecoderLayer,
            'Geo': PGMGeoDecoderLayer
        },
        'get_prior': get_pgm_prior,
        'default_layer': 'Vanilla',
        'param_count': 2,  # mean (alpha, log_beta_square), log_gamma_square
        'internal_multiplier': 2,  # latent_dim must be even
        'output_shape': '2d',  # [batch, latent_dim//2, 2]
    },
    'learnable_pgm': {
        'distribution_class': LearnablePGMDistribution,
        'encoder_layers': {
            'Vanilla': LearnablePGMVanillaEncoderLayer,
            'Exp': LearnablePGMExpEncoderLayer
        },
        'decoder_layers': {
            'Vanilla': LearnablePGMVanillaDecoderLayer,
            'Log': LearnablePGMLogDecoderLayer
        },
        'get_prior': get_learnable_pgm_prior,
        'default_layer': 'Vanilla',
        'param_count': 2,  # mean (alpha, log_beta_square, log_c), log_gamma_square
        'internal_multiplier': 2,  # latent_dim must be even
        'output_shape': '2d_extended',  # [batch, latent_dim//2, 3] for mean, but output as flat
    },
    'hw': {
        'distribution_class': HWDistribution,
        'encoder_layers': {'Vanilla': HWEncoderLayer},
        'decoder_layers': {'Vanilla': HWDecoderLayer},
        'get_prior': get_hw_prior,
        'default_layer': 'Vanilla',
        'param_count': 2,  # mean (3D Lorentz), sigma (2D)
        'internal_multiplier': 2,  # latent_dim must be even
        'output_shape': 'lorentz',  # [batch, latent_dim//2, 3] internally
    },
}


class GMVAEEncoder(nn.Module):
    """GM-VAE编码器 - 支持5种几何分布"""
    def __init__(self, input_dim: int, hidden_dims: list, latent_dim: int,
                 distribution: str = 'euclidean', layer_type: str = None,
                 device='cuda', c=-1.0):
        super().__init__()
        
        if distribution not in DISTRIBUTION_CONFIG:
            raise ValueError(f"Unknown distribution: {distribution}. "
                           f"Choose from {list(DISTRIBUTION_CONFIG.keys())}")
        
        self.distribution = distribution
        self.config = DISTRIBUTION_CONFIG[distribution]
        
        # Use default layer if not specified
        if layer_type is None:
            layer_type = self.config['default_layer']
        
        if layer_type not in self.config['encoder_layers']:
            raise ValueError(f"Invalid layer_type '{layer_type}' for {distribution}. "
                           f"Choose from {list(self.config['encoder_layers'].keys())}")
        
        self.layer_type = layer_type
        self.latent_dim = latent_dim
        
        # Build shared feature extractor
        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_dim)
            ])
            prev_dim = hidden_dim
        
        self.feature_extractor = nn.Sequential(*layers)
        self.feature_dim = prev_dim
        
        # Create distribution-specific encoder layer
        args = SimpleArgs(latent_dim, device, c)
        
        # Adjust latent_dim for distributions that use paired dimensions
        if self.config['output_shape'] in ['2d', '2d_extended', 'lorentz']:
            args.latent_dim = latent_dim // 2
        elif self.config['output_shape'] == 'flat' and self.config['internal_multiplier'] == 2:
            # Euclidean uses latent_dim * 2 internally
            args.latent_dim = latent_dim // 2
        
        encoder_class = self.config['encoder_layers'][layer_type]
        self.variational_layer = encoder_class(args, self.feature_dim)
    
    def forward(self, x):
        """
        Returns distribution parameters based on the geometry type
        Returns: tuple of parameters specific to each distribution
        """
        features = self.feature_extractor(x)
        return self.variational_layer(features)


class GMVAEDecoder(nn.Module):
    """GM-VAE解码器 - 支持5种几何分布"""
    def __init__(self, latent_dim: int, hidden_dims: list, output_dim: int,
                 distribution: str = 'euclidean', layer_type: str = None,
                 device='cuda', c=-1.0, loss_type: str = 'MSE'):
        super().__init__()
        
        if distribution not in DISTRIBUTION_CONFIG:
            raise ValueError(f"Unknown distribution: {distribution}")
        
        self.distribution = distribution
        self.config = DISTRIBUTION_CONFIG[distribution]
        
        # Use default layer if not specified
        if layer_type is None:
            layer_type = self.config['default_layer']
        
        if layer_type not in self.config['decoder_layers']:
            raise ValueError(f"Invalid layer_type '{layer_type}' for {distribution}. "
                           f"Choose from {list(self.config['decoder_layers'].keys())}")
        
        self.layer_type = layer_type
        self.latent_dim = latent_dim
        
        # Create distribution-specific decoder layer
        args = SimpleArgs(latent_dim, device, c)
        
        # Adjust latent_dim for distributions that use paired dimensions
        if self.config['output_shape'] in ['2d', '2d_extended', 'lorentz']:
            args.latent_dim = latent_dim // 2
        elif self.config['output_shape'] == 'flat' and self.config['internal_multiplier'] == 2:
            args.latent_dim = latent_dim // 2
        
        decoder_class = self.config['decoder_layers'][layer_type]
        self.decode_layer = decoder_class(args)
        
        # Decoder output is always flattened to latent_dim
        decoder_input_dim = latent_dim
        
        # Build reconstruction network
        layers = []
        prev_dim = decoder_input_dim
        for hidden_dim in reversed(hidden_dims):
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_dim)
            ])
            prev_dim = hidden_dim
        
        # Output layer (double size for NLL loss to output mean and logvar)
        output_features = output_dim * 2 if loss_type == 'NLL' else output_dim
        layers.append(nn.Linear(prev_dim, output_features))
        
        self.decoder_net = nn.Sequential(*layers)
    
    def forward(self, z):
        """
        Args:
            z: Latent representation in geometry-specific format
        Returns:
            Reconstruction [batch, output_dim] or [batch, output_dim*2] for NLL
        """
        z_decoded = self.decode_layer(z)
        return self.decoder_net(z_decoded)


class GMVAEModel(BaseModel):
    """
    Geometric Manifold VAE的完整PyTorch实现
    支持5种几何空间中的变分推断：
    - Euclidean: 标准欧几里得空间
    - Poincaré: 庞加莱球模型（双曲几何）
    - PGM: 伪高斯流形
    - LearnablePGM: 可学习曲率的PGM
    - HW: 超球面包裹正态分布（Lorentz模型）
    """
    
    def __init__(self,
                 input_dim: int,
                 latent_dim: int = 10,
                 hidden_dims: list = None,
                 distribution: str = 'euclidean',
                 encoder_layer: str = None,
                 decoder_layer: str = None,
                 curvature: float = -1.0,
                 loss_type: str = 'MSE',
                 model_name: str = "GMVAE"):
        """
        Args:
            input_dim: 输入维度
            latent_dim: 潜在空间总维度（统一标准：10表示10维输出）
            hidden_dims: 隐藏层维度列表
            distribution: 分布类型
                - 'euclidean': 欧几里得正态分布
                - 'poincare': 庞加莱球正态分布
                - 'pgm': 伪高斯流形
                - 'learnable_pgm': 可学习曲率的PGM
                - 'hw': 超球面包裹正态分布
            encoder_layer: 编码器层类型 (None使用默认值)
                - Euclidean: 'Vanilla'
                - Poincaré: 'Vanilla'
                - PGM: 'Vanilla', 'Geo'
                - LearnablePGM: 'Vanilla', 'Exp'
                - HW: 'Vanilla'
            decoder_layer: 解码器层类型 (None使用默认值)
                - Euclidean: 'Vanilla'
                - Poincaré: 'Vanilla'
                - PGM: 'Vanilla', 'Geo'
                - LearnablePGM: 'Vanilla', 'Log'
                - HW: 'Vanilla'
            curvature: 曲率参数（用于PGM/LearnablePGM/HW）
            loss_type: 损失类型 ('BCE', 'MSE', 'NLL')
            model_name: 模型名称
        """
        if hidden_dims is None:
            hidden_dims = [512, 256]
        
        # Validate distribution
        if distribution not in DISTRIBUTION_CONFIG:
            raise ValueError(f"Unknown distribution: {distribution}. "
                           f"Choose from {list(DISTRIBUTION_CONFIG.keys())}")
        
        config = DISTRIBUTION_CONFIG[distribution]
        
        # Validate latent_dim for distributions requiring even dimensions
        if config['output_shape'] in ['2d', '2d_extended', 'lorentz']:
            if latent_dim % 2 != 0:
                raise ValueError(f"latent_dim must be even for {distribution} distribution "
                               f"(got {latent_dim})")
        
        super().__init__(input_dim, latent_dim, hidden_dims, model_name)
        
        self.distribution = distribution
        self.config = config
        self.encoder_layer = encoder_layer or config['default_layer']
        self.decoder_layer = decoder_layer or config['default_layer']
        self.curvature = curvature
        self.loss_type = loss_type
        
        # Get device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Build encoder and decoder
        self.encoder_net = GMVAEEncoder(
            input_dim, hidden_dims, latent_dim,
            distribution, self.encoder_layer, self.device, curvature
        )
        
        self.decoder_net = GMVAEDecoder(
            latent_dim, hidden_dims, input_dim,
            distribution, self.decoder_layer, self.device, curvature, loss_type
        )
        
        # Create prior distribution
        args = SimpleArgs(
            latent_dim // 2 if config['output_shape'] in ['2d', '2d_extended', 'lorentz'] 
            else latent_dim // 2,
            self.device,
            curvature
        )
        
        self.prior = config['get_prior'](args)
    
    def _create_distribution(self, *params):
        """Create distribution object from parameters"""
        return self.config['distribution_class'](*params)
    
    def _reshape_for_output(self, z):
        """Reshape latent representation to [batch, latent_dim]"""
        if z.dim() == 2:
            return z
        # For multi-dimensional representations, flatten
        return z.reshape(z.size(0), -1)
    
    def _reshape_for_decoder(self, z):
        """Reshape latent representation for decoder input"""
        batch_size = z.size(0)
        
        if self.config['output_shape'] == 'flat':
            return z
        elif self.config['output_shape'] == '2d':
            # [batch, latent_dim] -> [batch, latent_dim//2, 2]
            return z.reshape(batch_size, -1, 2)
        elif self.config['output_shape'] == '2d_extended':
            # For LearnablePGM: [batch, latent_dim] -> [batch, latent_dim//2, 2]
            # But samples come out as [batch, latent_dim//2, 2]
            if z.dim() == 2:
                return z.reshape(batch_size, -1, 2)
            return z
        elif self.config['output_shape'] == 'lorentz':
            # For HW: samples come as [batch, latent_dim//2, 3]
            return z
        
        return z
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        编码到潜在空间
        Returns: [batch, latent_dim] - 统一的latent_dim维输出
        """
        params = self.encoder_net(x)
        variational = self._create_distribution(*params)
        z = variational.rsample(1).squeeze(0)  # [batch, ...]
        
        return self._reshape_for_output(z)
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """
        从潜在空间解码
        Args: z with shape [batch, latent_dim]
        Returns: [batch, input_dim]
        """
        z = self._reshape_for_decoder(z)
        output = self.decoder_net(z)
        
        if self.loss_type == 'NLL':
            return output[..., :output.size(-1)//2]  # 返回均值部分
        return output
    
    def forward(self, x: torch.Tensor, n_samples: int = 1, beta: float = 1.0, 
                iwae: int = 0, **kwargs) -> Dict[str, torch.Tensor]:
        """
        前向传播
        
        Args:
            x: 输入数据 [batch_size, input_dim]
            n_samples: 采样数量
            beta: KL权重
            iwae: 是否使用IWAE
            
        Returns:
            包含重构、潜在表示等的字典
        """
        params = self.encoder_net(x)
        variational = self._create_distribution(*params)
        
        # 采样
        z = variational.rsample(n_samples)  # [n_samples, batch, ...]
        
        # 解码
        if n_samples > 1:
            # Flatten for decoding
            if z.dim() == 3:  # [n_samples, batch, latent_dim]
                z_flat = z.reshape(-1, z.size(-1))
            elif z.dim() == 4:  # [n_samples, batch, latent_dim//2, 2or3]
                original_shape = z.shape
                z_flat = z.reshape(-1, original_shape[-2], original_shape[-1])
            else:
                z_flat = z.reshape(n_samples * x.size(0), -1)
            
            x_generated = self.decoder_net(z_flat)
            
            # Reshape back
            x_generated = x_generated.view(n_samples, x.size(0), -1)
        else:
            z = z.squeeze(0)  # Remove n_samples dimension
            x_generated = self.decoder_net(z)
        
        return {
            'reconstruction': x_generated,
            'latent': z,
            'variational': variational,
            'params': params
        }
    
    def compute_loss(self, x: torch.Tensor, outputs: Dict[str, torch.Tensor],
                     beta: float = 1.0, n_samples: int = 1, iwae: int = 0,
                     **kwargs) -> Dict[str, torch.Tensor]:
        """
        计算损失
        
        Args:
            x: 输入数据
            outputs: forward()的输出
            beta: KL权重
            n_samples: 采样数量
            iwae: 是否使用IWAE
            
        Returns:
            损失字典
        """
        x_generated = outputs['reconstruction']
        z = outputs['latent']
        variational = outputs['variational']
        
        # ✅ 重构损失 - 使用mean保持尺度平衡
        if self.loss_type == 'BCE':
            if n_samples > 1:
                recon_loss = F.binary_cross_entropy_with_logits(
                    x_generated,
                    x.unsqueeze(0).expand(x_generated.size()),
                    reduction='mean'
                )
            else:
                recon_loss = F.binary_cross_entropy_with_logits(
                    x_generated, x, reduction='mean'
                )
        elif self.loss_type == 'MSE':
            if n_samples > 1:
                recon_loss = F.mse_loss(
                    x_generated,
                    x.unsqueeze(0).expand(x_generated.size()),
                    reduction='mean'
                )
            else:
                recon_loss = F.mse_loss(x_generated, x, reduction='mean')
        elif self.loss_type == 'NLL':
            # For NLL, x_generated has shape [..., 2*input_dim]
            if n_samples > 1:
                mu = x_generated[..., :x.size(-1)]
                logvar = x_generated[..., x.size(-1):]
                recon_loss = 0.5 * ((x.unsqueeze(0) - mu).pow(2) / logvar.exp() + logvar).mean()
            else:
                mu = x_generated[..., :x.size(-1)]
                logvar = x_generated[..., x.size(-1):]
                recon_loss = 0.5 * ((x - mu).pow(2) / logvar.exp() + logvar).mean()
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")
        
        # ✅ KL散度 - 使用mean保持与重构损失相同的尺度
        if iwae == 0 or n_samples == 1:
            # Standard VAE loss
            if hasattr(variational, 'kl_div') and variational.kl_div is not None:
                # Use closed-form KL if available
                kl_div = variational.kl_div
                if callable(kl_div):
                    kl_loss = kl_div(self.prior).mean()
                else:
                    kl_loss = kl_div.mean()
            else:
                # Compute KL from log probabilities
                log_q = variational.log_prob(z)
                log_p = self.prior.log_prob(z)
                
                if log_q.dim() > 1:
                    kl_loss = (log_q - log_p).sum(dim=-1).mean()
                else:
                    kl_loss = (log_q - log_p).mean()
            
            total_loss = recon_loss + beta * kl_loss
            recon_loss_sum = recon_loss
            kl_loss_sum = kl_loss
        else:
            # IWAE
            log_q = variational.log_prob(z)
            log_p = self.prior.log_prob(z)
            
            if log_q.dim() > 2:
                log_q = log_q.sum(dim=-1)
                log_p = log_p.sum(dim=-1)
            
            kl_loss = log_q - log_p
            total_loss_sum = -recon_loss - beta * kl_loss
            
            loss = total_loss_sum.logsumexp(dim=0)
            loss = loss - np.log(n_samples)
            total_loss = -loss.mean()
            
            recon_loss_sum = recon_loss.mean(dim=0).sum()
            kl_loss_sum = kl_loss.mean(dim=0).sum()
        
        return {
            'total_loss': total_loss,
            'recon_loss': recon_loss_sum,
            'kl_loss': kl_loss_sum
        }


# 便捷的创建函数
def create_gmvae_model(input_dim: int, latent_dim: int = 10, 
                       distribution: str = 'euclidean', **kwargs):
    """
    创建GM-VAE模型
    
    Args:
        input_dim: 输入维度
        latent_dim: 潜在空间总维度（10表示10维输出，与其他模型一致）
        distribution: 'euclidean', 'poincare', 'pgm', 'learnable_pgm', 'hw'
        **kwargs: 其他参数传递给GMVAEModel
            - encoder_layer: 编码器层类型
            - decoder_layer: 解码器层类型
            - curvature: 曲率参数
            - loss_type: 损失类型
            - hidden_dims: 隐藏层维度
    
    Examples:
        >>> # Euclidean VAE with 10D latent space
        >>> model = create_gmvae_model(2000, latent_dim=10, distribution='euclidean')
        
        >>> # Poincaré VAE with 10D latent space (5 points × 2 coords)
        >>> model = create_gmvae_model(2000, latent_dim=10, distribution='poincare')
        
        >>> # PGM VAE with Geo layers
        >>> model = create_gmvae_model(2000, latent_dim=10, distribution='pgm', 
        ...                           encoder_layer='Geo', decoder_layer='Geo', 
        ...                           curvature=-1.0)
        
        >>> # LearnablePGM VAE with Exp/Log layers
        >>> model = create_gmvae_model(2000, latent_dim=10, distribution='learnable_pgm',
        ...                           encoder_layer='Exp', decoder_layer='Log')
        
        >>> # HW (Hyperboloid Wrapped) VAE
        >>> model = create_gmvae_model(2000, latent_dim=10, distribution='hw')
    """
    return GMVAEModel(input_dim=input_dim, latent_dim=latent_dim, 
                     distribution=distribution, **kwargs)


def get_available_distributions():
    """返回所有可用的分布及其配置信息"""
    info = {}
    for dist_name, config in DISTRIBUTION_CONFIG.items():
        info[dist_name] = {
            'encoder_layers': list(config['encoder_layers'].keys()),
            'decoder_layers': list(config['decoder_layers'].keys()),
            'default_layer': config['default_layer'],
            'requires_even_latent_dim': config['output_shape'] in ['2d', '2d_extended', 'lorentz']
        }
    return info


def print_distribution_info():
    """打印所有可用分布的详细信息"""
    print("=" * 80)
    print("GM-VAE: Available Geometric Distributions")
    print("=" * 80)
    
    info = get_available_distributions()
    for dist_name, dist_info in info.items():
        print(f"\n📊 {dist_name.upper()}")
        print(f"   Encoder layers: {', '.join(dist_info['encoder_layers'])}")
        print(f"   Decoder layers: {', '.join(dist_info['decoder_layers'])}")
        print(f"   Default layer: {dist_info['default_layer']}")
        print(f"   Requires even latent_dim: {dist_info['requires_even_latent_dim']}")
    
    print("\n" + "=" * 80)


# 自动打印信息（可选）
if __name__ == "__main__":
    print_distribution_info()
