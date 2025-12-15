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
    """
    
    def __init__(self, 
                 input_dim: int,
                 latent_dim: int,
                 hidden_dims: list = None,
                 model_name: str = "base_model"):
        """
        初始化基础模型
        
        Args:
            input_dim: 输入特征维度
            latent_dim: 潜在空间维度
            hidden_dims: 隐藏层维度列表
            model_name: 模型名称
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
            x: 输入张量 [batch_size, input_dim]
            
        Returns:
            潜在表示 [batch_size, latent_dim]
        """
        pass
    
    @abstractmethod
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """
        从潜在表示解码
        
        Args:
            z: 潜在表示 [batch_size, latent_dim]
            
        Returns:
            重构输出 [batch_size, input_dim]
        """
        pass
    
    @abstractmethod
    def forward(self, x: torch.Tensor, **kwargs) -> Dict[str, torch.Tensor]:
        """
        前向传播
        
        Args:
            x: 输入张量
            **kwargs: 额外参数
            
        Returns:
            包含重构、潜在表示、损失等的字典
        """
        pass
    
    @abstractmethod
    def compute_loss(self, x: torch.Tensor, outputs: Dict[str, torch.Tensor], **kwargs) -> Dict[str, torch.Tensor]:
        """
        计算损失
        
        Args:
            x: 输入张量
            outputs: forward()的输出
            **kwargs: 额外参数
            
        Returns:
            包含各种损失的字典
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
            verbose: bool = True,
            **kwargs) -> Dict[str, list]:
        """
        统一的训练接口
        
        Args:
            train_loader: 训练数据加载器
            val_loader: 验证数据加载器
            epochs: 训练轮数
            lr: 学习率
            device: 训练设备
            save_path: 模型保存路径
            patience: 早停耐心值
            verbose: 是否打印训练信息
            **kwargs: 额外的训练参数
            
        Returns:
            训练历史记录字典
        """
        self.to(device)
        optimizer = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=kwargs.get('weight_decay', 0.0))
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
            train_metrics = self._train_epoch(train_loader, optimizer, device, **kwargs)
            history['train_loss'].append(train_metrics['total_loss'])
            history['train_recon_loss'].append(train_metrics['recon_loss'])
            
            # 验证
            if val_loader is not None:
                val_metrics = self._validate_epoch(val_loader, device, **kwargs)
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
                    if verbose:
                        print(f"Early stopping at epoch {epoch+1}")
                    break
                    
                if verbose:
                    print(f"Epoch {epoch+1}/{epochs} - "
                          f"Train Loss: {train_metrics['total_loss']:.4f}, "
                          f"Val Loss: {val_metrics['total_loss']:.4f}")
            else:
                if verbose:
                    print(f"Epoch {epoch+1}/{epochs} - "
                          f"Train Loss: {train_metrics['total_loss']:.4f}")
        
        return history
    
    def _prepare_batch(self, batch_data: Any, device: str):
        """将 DataLoader 输出标准化为 (x, batch_kwargs)。

        默认实现：
        - 如果 batch_data 是 (x, ...)，取第一个元素作为 x，其余信息忽略；
        - 子类可以重载该方法，将 batch 信息等通过 batch_kwargs 传入 forward/compute_loss。
        """
        if isinstance(batch_data, (list, tuple)):
            x = batch_data[0]
        else:
            x = batch_data

        x = x.to(device).float()
        batch_kwargs: Dict[str, Any] = {}
        return x, batch_kwargs
    
    def _train_epoch(self, train_loader: DataLoader, optimizer: torch.optim.Optimizer, 
                     device: str, **kwargs) -> Dict[str, float]:
        """训练一个epoch"""
        self.train()
        total_loss = 0
        total_recon_loss = 0
        n_batches = 0
        
        for batch_data in train_loader:
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
        
        return {
            'total_loss': total_loss / n_batches,
            'recon_loss': total_recon_loss / n_batches
        }
    
    def _validate_epoch(self, val_loader: DataLoader, device: str, **kwargs) -> Dict[str, float]:
        """验证一个epoch"""
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
            data_loader: 数据加载器
            device: 计算设备
            return_reconstructions: 是否返回重构结果
            
        Returns:
            包含潜在表示和可选重构的字典
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
        """保存模型"""
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
    
    def load_model(self, path: str):
        """加载模型"""
        checkpoint = torch.load(path, map_location='cpu')
        self.load_state_dict(checkpoint['model_state_dict'])
        return checkpoint.get('config', {})
