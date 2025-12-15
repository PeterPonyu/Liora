"""
统一的基础模型接口
提供一致的训练、推理和潜在表示提取接口
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, Optional, Tuple, Any
from abc import ABC, abstractmethod
import numpy as np
from pathlib import Path
import json


class BaseModel(ABC, nn.Module):
    """
    所有模型的基类，定义统一的接口
    
    这是一个抽象基类，所有单细胞基因表达模型都应继承此类。
    它提供了统一的训练、验证、潜在表示提取等功能。
    
    Attributes:
        input_dim (int): 输入特征维度（基因数）
        latent_dim (int): 潜在空间维度
        hidden_dims (list): 隐藏层维度列表
        model_name (str): 模型名称
    
    Examples:
        >>> # 定义一个自定义模型
        >>> class MyModel(BaseModel):
        ...     def __init__(self, input_dim, latent_dim):
        ...         super().__init__(input_dim, latent_dim)
        ...         self.encoder = nn.Linear(input_dim, latent_dim)
        ...         self.decoder = nn.Linear(latent_dim, input_dim)
        ...     
        ...     def encode(self, x):
        ...         return self.encoder(x)
        ...     
        ...     def decode(self, z):
        ...         return self.decoder(z)
        ...     
        ...     def forward(self, x, **kwargs):
        ...         z = self.encode(x)
        ...         recon = self.decode(z)
        ...         return {'latent': z, 'reconstruction': recon}
        ...     
        ...     def compute_loss(self, x, outputs, **kwargs):
        ...         recon_loss = F.mse_loss(outputs['reconstruction'], x)
        ...         return {'total_loss': recon_loss, 'recon_loss': recon_loss}
        >>> 
        >>> model = MyModel(2000, 10)
        >>> history = model.fit(train_loader, val_loader, epochs=100)
    """
    
    def __init__(self, 
                 input_dim: int,
                 latent_dim: int,
                 hidden_dims: list = None,
                 model_name: str = "base_model"):
        """
        初始化基础模型
        
        Args:
            input_dim (int): 输入特征维度（通常是基因数）
            latent_dim (int): 潜在空间维度
            hidden_dims (list, optional): 隐藏层维度列表。默认为 [512, 256]
            model_name (str, optional): 模型名称。默认为 "base_model"
        
        Examples:
            >>> model = BaseModel(input_dim=2000, latent_dim=10, 
            ...                   hidden_dims=[512, 256, 128])
        """
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.hidden_dims = hidden_dims or [512, 256]
        self.model_name = model_name
        
    @abstractmethod
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        编码输入为潜在表示
        
        Args:
            x (torch.Tensor): 输入张量，形状为 [batch_size, input_dim]
            
        Returns:
            torch.Tensor: 潜在表示，形状为 [batch_size, latent_dim]
        
        Examples:
            >>> x = torch.randn(32, 2000)  # 32个细胞，2000个基因
            >>> z = model.encode(x)  # [32, 10]
        """
        pass
    
    @abstractmethod
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """
        从潜在表示解码
        
        Args:
            z (torch.Tensor): 潜在表示，形状为 [batch_size, latent_dim]
            
        Returns:
            torch.Tensor: 重构输出，形状为 [batch_size, input_dim]
        
        Examples:
            >>> z = torch.randn(32, 10)
            >>> recon = model.decode(z)  # [32, 2000]
        """
        pass
    
    @abstractmethod
    def forward(self, x: torch.Tensor, **kwargs) -> Dict[str, torch.Tensor]:
        """
        前向传播
        
        Args:
            x (torch.Tensor): 输入张量，形状为 [batch_size, input_dim]
            **kwargs: 额外参数（如 batch_id, labels 等）
            
        Returns:
            Dict[str, torch.Tensor]: 包含以下键的字典：
                - 'latent': 潜在表示
                - 'reconstruction' 或模型特定的输出
        
        Examples:
            >>> x = torch.randn(32, 2000)
            >>> outputs = model(x)
            >>> z = outputs['latent']
        """
        pass
    
    @abstractmethod
    def compute_loss(self, x: torch.Tensor, outputs: Dict[str, torch.Tensor], 
                     **kwargs) -> Dict[str, torch.Tensor]:
        """
        计算损失函数
        
        Args:
            x (torch.Tensor): 输入张量，形状为 [batch_size, input_dim]
            outputs (Dict[str, torch.Tensor]): forward() 的输出
            **kwargs: 额外参数
            
        Returns:
            Dict[str, torch.Tensor]: 包含以下键的字典：
                - 'total_loss': 总损失（必须）
                - 'recon_loss': 重构损失
                - 其他模型特定的损失（如 'kl_loss', 'adversarial_loss' 等）
        
        Examples:
            >>> x = torch.randn(32, 2000)
            >>> outputs = model(x)
            >>> losses = model.compute_loss(x, outputs)
            >>> print(losses['total_loss'].item())
        """
        pass
    
    def fit(self,
            train_loader: DataLoader,
            val_loader: Optional[DataLoader] = None,
            epochs: int = 100,
            lr: float = 1e-3,
            device: str = 'cuda',
            save_path: Optional[str] = None,
            patience: int = 10,
            verbose: int = 1,
            **kwargs) -> Dict[str, list]:
        """
        统一的训练接口
        
        Args:
            train_loader (DataLoader): 训练数据加载器
            val_loader (DataLoader, optional): 验证数据加载器。默认为 None
            epochs (int, optional): 训练轮数。默认为 100
            lr (float, optional): 学习率。默认为 1e-3
            device (str, optional): 训练设备（'cuda' 或 'cpu'）。默认为 'cuda'
            save_path (str, optional): 模型保存路径。默认为 None（不保存）
            patience (int, optional): 早停耐心值。默认为 10
            verbose (int, optional): 日志详细程度。
                - 0: 不输出任何信息
                - 1: 仅输出每个 epoch 的损失（默认）
                - 2: 输出详细的批次级别信息
            **kwargs: 额外的训练参数（如 beta, weight_decay 等）
            
        Returns:
            Dict[str, list]: 训练历史字典，包含：
                - 'train_loss': 训练损失列表
                - 'val_loss': 验证损失列表（如果提供了 val_loader）
                - 'train_recon_loss': 训练重构损失列表
                - 'val_recon_loss': 验证重构损失列表（如果提供了 val_loader）
        
        Examples:
            >>> # 基础训练
            >>> model = create_my_model(input_dim=2000, latent_dim=10)
            >>> history = model.fit(
            ...     train_loader=train_loader,
            ...     val_loader=val_loader,
            ...     epochs=100,
            ...     lr=1e-3,
            ...     device='cuda',
            ...     verbose=1  # 仅输出每个 epoch 的损失
            ... )
            >>> 
            >>> # 无验证集训练
            >>> history = model.fit(
            ...     train_loader=train_loader,
            ...     epochs=50,
            ...     lr=5e-4,
            ...     verbose=0  # 不输出任何信息
            ... )
            >>> 
            >>> # 保存最好的模型
            >>> history = model.fit(
            ...     train_loader=train_loader,
            ...     val_loader=val_loader,
            ...     epochs=200,
            ...     save_path='./checkpoints/best_model.pt',
            ...     patience=20,
            ...     verbose=2  # 详细输出
            ... )
        
        Notes:
            - 如果提供了 val_loader，会根据验证损失进行早停
            - 学习率会根据验证损失动态调整
            - 如果设置了 save_path，最好的模型会自动保存
        """
        self.to(device)
        optimizer = torch.optim.Adam(
            self.parameters(), 
            lr=lr, 
            weight_decay=kwargs.get('weight_decay', 0.0)
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5
        )
        
        history = {
            'train_loss': [],
            'val_loss': [],
            'train_recon_loss': [],
            'val_recon_loss': []
        }
        
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(epochs):
            # 训练
            train_metrics = self._train_epoch(train_loader, optimizer, device, verbose, **kwargs)
            history['train_loss'].append(train_metrics['total_loss'])
            history['train_recon_loss'].append(train_metrics['recon_loss'])
            
            # 验证
            if val_loader is not None:
                val_metrics = self._validate_epoch(val_loader, device, verbose, **kwargs)
                history['val_loss'].append(val_metrics['total_loss'])
                history['val_recon_loss'].append(val_metrics['recon_loss'])
                
                # 学习率调度
                scheduler.step(val_metrics['total_loss'])
                
                # 早停
                if val_metrics['total_loss'] < best_val_loss:
                    best_val_loss = val_metrics['total_loss']
                    patience_counter = 0
                    if save_path:
                        self.save_model(save_path)
                else:
                    patience_counter += 1
                    
                if patience_counter >= patience:
                    if verbose >= 1:
                        print(f"\n✓ 早停于第 {epoch+1} epoch")
                    break
                    
                # 根据 verbose 级别输出信息
                if verbose >= 1:
                    print(f"Epoch {epoch+1:3d}/{epochs} | "
                          f"Train Loss: {train_metrics['total_loss']:8.4f} | "
                          f"Val Loss: {val_metrics['total_loss']:8.4f}")
            else:
                # 无验证集时的输出
                if verbose >= 1:
                    print(f"Epoch {epoch+1:3d}/{epochs} | "
                          f"Train Loss: {train_metrics['total_loss']:8.4f}")
        
        if verbose >= 1:
            print("\n✓ 训练完成!")
        
        return history
    
    def _prepare_batch(self, batch_data: Any, device: str) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        将 DataLoader 输出标准化为 (x, batch_kwargs)
        
        子类可以重载该方法以处理特定的批次信息。
        
        Args:
            batch_data: DataLoader 的输出
            device (str): 目标设备
            
        Returns:
            Tuple[torch.Tensor, Dict[str, Any]]: 
                - x: 输入张量
                - batch_kwargs: 额外的批次信息字典
        
        Examples:
            >>> # 默认实现（仅提取 x）
            >>> x, batch_kwargs = model._prepare_batch(batch_data, 'cuda')
            >>> 
            >>> # 自定义实现示例
            >>> class MyModel(BaseModel):
            ...     def _prepare_batch(self, batch_data, device):
            ...         x, batch_id, labels = batch_data
            ...         x = x.to(device).float()
            ...         return x, {
            ...             'batch_id': batch_id.to(device),
            ...             'labels': labels.to(device)
            ...         }
        """
        if isinstance(batch_data, (list, tuple)):
            x = batch_data[0]
        else:
            x = batch_data

        x = x.to(device).float()
        batch_kwargs: Dict[str, Any] = {}
        return x, batch_kwargs
    
    def _train_epoch(self, train_loader: DataLoader, optimizer: torch.optim.Optimizer, 
                     device: str, verbose: int = 1, **kwargs) -> Dict[str, float]:
        """
        训练一个 epoch
        
        Args:
            train_loader (DataLoader): 训练数据加载器
            optimizer (torch.optim.Optimizer): 优化器
            device (str): 训练设备
            verbose (int): 日志详细程度
            **kwargs: 额外参数
            
        Returns:
            Dict[str, float]: 包含平均损失的字典
        """
        self.train()
        total_loss = 0
        total_recon_loss = 0
        n_batches = 0
        
        for batch_idx, batch_data in enumerate(train_loader):
            x, batch_kwargs = self._prepare_batch(batch_data, device)
            
            optimizer.zero_grad()
            outputs = self.forward(x, **batch_kwargs, **kwargs)
            losses = self.compute_loss(x, outputs, **batch_kwargs, **kwargs)
            
            loss = losses['total_loss']
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            total_recon_loss += losses.get('recon_loss', loss).item()
            n_batches += 1
            
            # 详细模式下输出批次信息
            if verbose >= 2:
                print(f"  Batch {batch_idx+1}/{len(train_loader)} | Loss: {loss.item():.4f}")
        
        return {
            'total_loss': total_loss / n_batches,
            'recon_loss': total_recon_loss / n_batches
        }
    
    def _validate_epoch(self, val_loader: DataLoader, device: str, 
                       verbose: int = 1, **kwargs) -> Dict[str, float]:
        """
        验证一个 epoch
        
        Args:
            val_loader (DataLoader): 验证数据加载器
            device (str): 计算设备
            verbose (int): 日志详细程度
            **kwargs: 额外参数
            
        Returns:
            Dict[str, float]: 包含平均损失的字典
        """
        self.eval()
        total_loss = 0
        total_recon_loss = 0
        n_batches = 0
        
        with torch.no_grad():
            for batch_data in val_loader:
                x, batch_kwargs = self._prepare_batch(batch_data, device)
                
                outputs = self.forward(x, **batch_kwargs, **kwargs)
                losses = self.compute_loss(x, outputs, **batch_kwargs, **kwargs)
                
                total_loss += losses['total_loss'].item()
                total_recon_loss += losses.get('recon_loss', losses['total_loss']).item()
                n_batches += 1
        
        return {
            'total_loss': total_loss / n_batches,
            'recon_loss': total_recon_loss / n_batches
        }
    
    def extract_latent(self,
                      data_loader: DataLoader,
                      device: str = 'cuda',
                      return_reconstructions: bool = False) -> Dict[str, np.ndarray]:
        """
        提取潜在表示（统一接口）
        
        Args:
            data_loader (DataLoader): 数据加载器
            device (str, optional): 计算设备。默认为 'cuda'
            return_reconstructions (bool, optional): 是否返回重构结果。默认为 False
            
        Returns:
            Dict[str, np.ndarray]: 包含以下键的字典：
                - 'latent': 潜在表示，形状为 [n_samples, latent_dim]
                - 'reconstruction': 重构结果（如果 return_reconstructions=True）
        
        Examples:
            >>> # 提取潜在表示
            >>> result = model.extract_latent(test_loader, device='cuda')
            >>> latent = result['latent']  # [n_samples, latent_dim]
            >>> 
            >>> # 同时获取重构结果
            >>> result = model.extract_latent(
            ...     test_loader,
            ...     device='cuda',
            ...     return_reconstructions=True
            ... )
            >>> latent = result['latent']
            >>> recon = result['reconstruction']
        
        Notes:
            - 在推理模式下执行（无梯度计算）
            - 返回的数据已转移到 CPU 并转换为 NumPy 数组
        """
        self.eval()
        self.to(device)
        
        latents = []
        reconstructions = [] if return_reconstructions else None
        
        with torch.no_grad():
            for batch_data in data_loader:
                if isinstance(batch_data, (list, tuple)):
                    x = batch_data[0]
                else:
                    x = batch_data
                    
                x = x.to(device).float()
                
                z = self.encode(x)
                latents.append(z.cpu().numpy())
                
                if return_reconstructions:
                    recon = self.decode(z)
                    reconstructions.append(recon.cpu().numpy())
        
        result = {'latent': np.concatenate(latents, axis=0)}
        if return_reconstructions:
            result['reconstruction'] = np.concatenate(reconstructions, axis=0)
            
        return result
    
    def save_model(self, path: str):
        """
        保存模型权重和配置
        
        Args:
            path (str): 保存路径
        
        Examples:
            >>> model.save_model('./checkpoints/my_model.pt')
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        torch.save({
            'model_state_dict': self.state_dict(),
            'config': {
                'input_dim': self.input_dim,
                'latent_dim': self.latent_dim,
                'hidden_dims': self.hidden_dims,
                'model_name': self.model_name
            }
        }, path)
    
    def load_model(self, path: str) -> Dict[str, Any]:
        """
        加载模型权重和配置
        
        Args:
            path (str): 模型路径
            
        Returns:
            Dict[str, Any]: 模型配置
        
        Examples:
            >>> config = model.load_model('./checkpoints/my_model.pt')
        """
        checkpoint = torch.load(path, map_location='cpu')
        self.load_state_dict(checkpoint['model_state_dict'])
        return checkpoint.get('config', {})