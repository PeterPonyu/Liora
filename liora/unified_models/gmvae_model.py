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


def get_learnable_pgm_prior_fixed(args):
    """
    Fixed prior for LearnablePGM distribution
    Creates proper 3D mean tensor [1, D, 3] for (alpha, log_beta_square, log_c)
    """
    mean = torch.zeros(
        [1, args.latent_dim, 3],
        device=args.device
    )
    mean[..., 0] = 0.0  # alpha
    mean[..., 1] = 0.0  # log_beta_square
    mean[..., 2] = torch.log(torch.tensor(abs(args.c), device=args.device))  # log_c (>0)

    # log_gamma_square (log variance of Gamma) – any finite value is OK with fixed denom>0
    covar = torch.full(
        [1, args.latent_dim],
        0.0,
        device=args.device
    )

    prior = LearnablePGMDistribution(mean, covar)
    return prior


# Distribution configuration mapping
DISTRIBUTION_CONFIG = {
    'euclidean': {
        'distribution_class': EuclideanDistribution,
        'encoder_layers': {'Vanilla': EuclideanEncoderLayer},
        'decoder_layers': {'Vanilla': EuclideanDecoderLayer},
        'get_prior': get_euclidean_prior,
        'default_layer': 'Vanilla',
        'param_count': 2,  # loc, scale
        'loc_shape': 'flat',  # [B, D]
        'scale_shape': 'flat',  # [B, D]
        'sample_shape': 'flat',  # [B, D]
        'requires_even': False,
        'internal_dim_factor': 1.0,  # Use latent_dim directly
        'decoder_output_shape': 'doubled_flat',  # VanillaDecoderLayer: [B, D*2]
    },
    'poincare': {
        'distribution_class': PoincareDistribution,
        'encoder_layers': {'Vanilla': PoincareEncoderLayer},
        'decoder_layers': {'Vanilla': PoincareDecoderLayer},
        'get_prior': get_poincare_prior,
        'default_layer': 'Vanilla',
        'param_count': 2,  # loc, scale
        'loc_shape': '2d',  # [B, D//2, 2]
        'scale_shape': 'vector',  # [B, D//2]
        'sample_shape': '2d',  # [B, D//2, 2]
        'requires_even': True,
        'internal_dim_factor': 0.5,  # latent_dim // 2 = number of points
        'decoder_output_shape': 'geometry_2d',  # VanillaDecoderLayer: [B, D//2, 2]
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
        'param_count': 2,  # loc, scale
        'loc_shape': '2d',  # [B, D//2, 2] (alpha, log_beta_square)
        'scale_shape': 'vector',  # [B, D//2]
        'sample_shape': '2d',  # [B, D//2, 2]
        'requires_even': True,
        'internal_dim_factor': 0.5,
        'decoder_output_shape': 'geometry_2d',  # GeoDecoderLayer: [B, D//2, 2]
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
        'get_prior': get_learnable_pgm_prior_fixed,  # ✅ Use fixed version
        'default_layer': 'Vanilla',
        'param_count': 2,  # loc, scale
        'loc_shape': '3d',  # [B, D//2, 3] (alpha, log_beta_square, log_c)
        'scale_shape': 'vector',  # [B, D//2]
        'sample_shape': '2d',  # [B, D//2, 2] (samples in 2D after projection)
        'requires_even': True,
        'internal_dim_factor': 0.5,
        'decoder_output_shape': 'flat',  # LogDecoderLayer: [B, D]
    },
    'hw': {
        'distribution_class': HWDistribution,
        'encoder_layers': {'Vanilla': HWEncoderLayer},
        'decoder_layers': {'Vanilla': HWDecoderLayer},
        'get_prior': get_hw_prior,
        'default_layer': 'Vanilla',
        'param_count': 2,  # loc, scale
        'loc_shape': '3d',  # [B, D//2, 3] (Lorentz coordinates)
        'scale_shape': '2d',  # [B, D//2, 2]
        'sample_shape': '3d',  # [B, D//2, 3]
        'requires_even': True,
        'internal_dim_factor': 0.5,
        'decoder_output_shape': 'geometry_3d',  # VanillaDecoderLayer: [B, D//2, 3]
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
        
        # Validate latent_dim requirements
        if self.config['requires_even'] and latent_dim % 2 != 0:
            raise ValueError(f"{distribution} requires even latent_dim (got {latent_dim})")
        
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
        internal_latent_dim = int(latent_dim * self.config['internal_dim_factor'])
        args = SimpleArgs(internal_latent_dim, device, c)
        
        encoder_class = self.config['encoder_layers'][layer_type]
        self.variational_layer = encoder_class(args, self.feature_dim)
    
    def forward(self, x):
        """
        Returns distribution parameters based on the geometry type
        Returns: tuple of (loc, scale) specific to each distribution
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
        
        # Validate latent_dim requirements
        if self.config['requires_even'] and latent_dim % 2 != 0:
            raise ValueError(f"{distribution} requires even latent_dim (got {latent_dim})")
        
        # Use default layer if not specified
        if layer_type is None:
            layer_type = self.config['default_layer']
        
        if layer_type not in self.config['decoder_layers']:
            raise ValueError(f"Invalid layer_type '{layer_type}' for {distribution}. "
                           f"Choose from {list(self.config['decoder_layers'].keys())}")
        
        self.layer_type = layer_type
        self.latent_dim = latent_dim
        self.loss_type = loss_type
        
        # Create distribution-specific decoder layer
        internal_latent_dim = int(latent_dim * self.config['internal_dim_factor'])
        args = SimpleArgs(internal_latent_dim, device, c)
        
        decoder_class = self.config['decoder_layers'][layer_type]
        self.decode_layer = decoder_class(args)
        
        # ✅ Determine decoder input dimension based on actual output shape
        decoder_output_shape = self.config.get('decoder_output_shape', 'flat')
        
        if decoder_output_shape == 'doubled_flat':
            # Euclidean VanillaDecoderLayer: [B, D*2] -> take first half -> [B, D]
            decoder_input_dim = latent_dim
        elif decoder_output_shape == 'geometry_2d':
            # Poincaré/PGM VanillaDecoderLayer: [B, D//2, 2] -> flatten -> [B, D]
            decoder_input_dim = latent_dim
        elif decoder_output_shape == 'geometry_3d':
            # HW VanillaDecoderLayer: [B, D//2, 3] -> flatten -> [B, D//2*3]
            # But we only use first 2 coords, so [B, D]
            decoder_input_dim = latent_dim
        elif decoder_output_shape == 'flat':
            # LearnablePGM LogDecoderLayer: [B, D] -> [B, D]
            decoder_input_dim = latent_dim
        else:
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
               - Euclidean: [B, D]
               - Poincaré: [B, D//2, 2]
               - PGM: [B, D//2, 2]
               - LearnablePGM: [B, D//2, 2]
               - HW: [B, D//2, 3]
        Returns:
            Reconstruction [batch, output_dim] or [batch, output_dim*2] for NLL
        """
        z_decoded = self.decode_layer(z)  # Outputs geometry-specific shape
        
        # ✅ Handle different decoder output shapes properly
        decoder_output_shape = self.config.get('decoder_output_shape', 'flat')
        batch_size = z_decoded.size(0) if z_decoded.dim() > 0 else 1
        
        if decoder_output_shape == 'doubled_flat':
            # Euclidean: [B, D*2] -> take first half -> [B, D]
            if z_decoded.dim() == 2:
                z_decoded = z_decoded[..., :z_decoded.size(-1)//2]
        
        elif decoder_output_shape == 'geometry_2d':
            # Poincaré/PGM: [B, D//2, 2] -> flatten -> [B, D]
            if z_decoded.dim() == 3:
                z_decoded = z_decoded.reshape(batch_size, -1)
        
        elif decoder_output_shape == 'geometry_3d':
            # HW: [B, D//2, 3] -> take first 2 coords -> [B, D//2, 2] -> flatten -> [B, D]
            if z_decoded.dim() == 3:
                z_decoded = z_decoded[..., :2]  # Take first 2 coordinates
                z_decoded = z_decoded.reshape(batch_size, -1)
        
        elif decoder_output_shape == 'flat':
            # LearnablePGM: [B, D] -> already flat
            if z_decoded.dim() > 2:
                z_decoded = z_decoded.reshape(batch_size, -1)
        
        # Final flatten if still multi-dimensional
        if z_decoded.dim() > 2:
            z_decoded = z_decoded.reshape(batch_size, -1)
        
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
                - Euclidean: 直接使用10维
                - Poincaré: 10维 = 5个2D点 (latent_dim必须是偶数)
                - PGM: 10维 = 5个2D极坐标点 (latent_dim必须是偶数)
                - LearnablePGM: 10维 = 5个2D点 (latent_dim必须是偶数)
                - HW: 10维 = 对应内部使用5个3D Lorentz点 (latent_dim必须是偶数)
            hidden_dims: 隐藏层维度列表
            distribution: 分布类型
                - 'euclidean': 欧几里得正态分布
                - 'poincare': 庞加莱球正态分布
                - 'pgm': 伪高斯流形
                - 'learnable_pgm': 可学习曲率的PGM
                - 'hw': 超球面包裹正态分布
            encoder_layer: 编码器层类型 (None使用默认值)
            decoder_layer: 解码器层类型 (None使用默认值)
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
        if config['requires_even'] and latent_dim % 2 != 0:
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
        internal_latent_dim = int(latent_dim * config['internal_dim_factor'])
        args = SimpleArgs(internal_latent_dim, self.device, curvature)
        
        self.prior = config['get_prior'](args)
    
    def _create_distribution(self, *params):
        """Create distribution object from parameters"""
        return self.config['distribution_class'](*params)
    
    def _reshape_for_output(self, z):
        """
        Reshape latent representation to [batch, latent_dim] for unified output
        
        Args:
            z: Distribution samples in geometry-specific format
        
        Returns:
            [B, latent_dim] - flattened representation
        """
        if z.dim() == 2:
            # Already flat (Euclidean)
            return z
        
        # For multi-dimensional representations, flatten to [B, D]
        batch_size = z.size(0)
        
        if self.config['sample_shape'] == '2d':
            # Poincaré, PGM, LearnablePGM: [B, D//2, 2] -> [B, D]
            return z.reshape(batch_size, -1)
        elif self.config['sample_shape'] == '3d':
            # HW: [B, D//2, 3] -> take first 2 coords -> [B, D]
            if self.distribution == 'hw':
                return z[..., :2].reshape(batch_size, -1)
            return z.reshape(batch_size, -1)
        
        return z.reshape(batch_size, -1)
    
    def _reshape_for_decoder(self, z):
        """
        Reshape latent representation from [B, latent_dim] to decoder input format
        
        Args:
            z: [B, latent_dim]
        
        Returns:
            Geometry-specific shape for decoder
        """
        batch_size = z.size(0)
        
        if self.config['sample_shape'] == 'flat':
            # Euclidean: [B, D] -> [B, D]
            return z
        elif self.config['sample_shape'] == '2d':
            # Poincaré, PGM, LearnablePGM: [B, D] -> [B, D//2, 2]
            return z.reshape(batch_size, -1, 2)
        elif self.config['sample_shape'] == '3d':
            # HW: [B, D] -> [B, D//2, 3]
            if self.distribution == 'hw':
                # Pad with zeros for the third coordinate
                z_2d = z.reshape(batch_size, -1, 2)  # [B, D//2, 2]
                z_3d = torch.cat([
                    z_2d,
                    torch.zeros(batch_size, z_2d.size(1), 1, device=z.device)
                ], dim=-1)  # [B, D//2, 3]
                return z_3d
            return z.reshape(batch_size, -1, 3)
        
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
            # Flatten samples for batch processing
            original_shape = z.shape
            
            if z.dim() == 3:  # [n_samples, batch, latent_dim] (Euclidean)
                z_flat = z.reshape(-1, z.size(-1))
                z_reshaped = z_flat
            elif z.dim() == 4:  # [n_samples, batch, D//2, 2or3]
                z_flat = z.reshape(-1, original_shape[-2], original_shape[-1])
                z_reshaped = z_flat
            else:
                z_flat = z.reshape(n_samples * x.size(0), -1)
                z_reshaped = self._reshape_for_decoder(z_flat)
            
            x_generated = self.decoder_net(z_reshaped)
            
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
        """
        x_generated = outputs['reconstruction']
        z = outputs['latent']
        variational = outputs['variational']
        
        # 重构损失
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
        
        # ---------- FIX: prepare z for KL / log_prob ----------
        # For manifold distributions (Poincaré, PGM, LearnablePGM, HW),
        # their log_prob implementations expect a sample dimension K:
        #   z: [K, B, ...]
        # Our forward() stores z as [B, ...] when n_samples == 1.
        z_kl = z
        if (self.distribution in ('poincare', 'pgm', 'learnable_pgm', 'hw')
                and n_samples == 1
                and z.dim() >= 2
                and z.shape[0] == x.shape[0]):  # z is [B, ...]
            z_kl = z.unsqueeze(0)              # -> [1, B, ...]
        # ------------------------------------------------------
        
        # KL散度
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
                log_q = variational.log_prob(z_kl)
                log_p = self.prior.log_prob(z_kl)
                
                if log_q.dim() > 1:
                    # Sum over latent dims, mean over samples & batch
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
            ⚠️ 注意：非欧几里得分布需要偶数维度
        distribution: 'euclidean', 'poincare', 'pgm', 'learnable_pgm', 'hw'
        **kwargs: 其他参数传递给GMVAEModel
    
    Examples:
        >>> # Euclidean VAE with 10D latent space
        >>> model = create_gmvae_model(2000, latent_dim=10, distribution='euclidean')
        
        >>> # Poincaré VAE with 10D latent space (5 points × 2 coords)
        >>> model = create_gmvae_model(2000, latent_dim=10, distribution='poincare')
        
        >>> # LearnablePGM VAE with Exp/Log layers
        >>> model = create_gmvae_model(2000, latent_dim=10, distribution='learnable_pgm',
        ...                           encoder_layer='Exp', decoder_layer='Log')
    """
    return GMVAEModel(input_dim=input_dim, latent_dim=latent_dim, 
                     distribution=distribution, **kwargs)

