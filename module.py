
"""Neural network modules for Liora."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from typing import Optional
from torchdiffeq import odeint
from .utils import exp_map_at_origin


def weight_init(m):
    """Xavier initialization for linear layers."""
    if isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight)
        nn.init.constant_(m.bias, 0.01)


class Encoder(nn.Module):
    """Encoder network with optional layer normalization and ODE support."""
    
    def __init__(self, state_dim, hidden_dim, action_dim, use_layer_norm=True, use_ode=False):
        super().__init__()
        self.use_layer_norm = use_layer_norm
        self.use_ode = use_ode
        
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim * 2)
        
        if use_layer_norm:
            self.ln1 = nn.LayerNorm(hidden_dim)
            self.ln2 = nn.LayerNorm(hidden_dim)
        
        # Time encoder for ODE mode
        if use_ode:
            self.time_encoder = nn.Sequential(
                nn.Linear(hidden_dim, 1),
                nn.Sigmoid(),  # Time values in [0, 1]
            )
        
        self.apply(weight_init)

    def forward(self, x):
        h1 = F.relu(self.ln1(self.fc1(x)) if self.use_layer_norm else self.fc1(x))
        h2 = F.relu(self.ln2(self.fc2(h1)) if self.use_layer_norm else self.fc2(h1))
        output = self.fc3(h2)
        
        q_m, q_s = torch.chunk(output, 2, dim=-1)
        q_m = torch.clamp(q_m, -10, 10)
        q_s = torch.clamp(q_s, -10, 10)
        
        s = torch.clamp(F.softplus(q_s) + 1e-6, min=1e-6, max=5.0)
        n = Normal(q_m, s)
        q_z = n.rsample()
        
        if self.use_ode:
            t = self.time_encoder(h2).squeeze(-1)  # Shape: (batch_size,)
            return q_z, q_m, q_s, n, t
        
        return q_z, q_m, q_s, n


class Decoder(nn.Module):
    """Decoder network with optional layer normalization."""
    
    def __init__(self, state_dim, hidden_dim, action_dim, loss_type='nb', use_layer_norm=True):
        super().__init__()
        self.loss_type = loss_type
        self.use_layer_norm = use_layer_norm
        
        self.fc1 = nn.Linear(action_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, state_dim)
        
        if use_layer_norm:
            self.ln1 = nn.LayerNorm(hidden_dim)
            self.ln2 = nn.LayerNorm(hidden_dim)
        
        self.disp = nn.Parameter(torch.randn(state_dim))
        
        if loss_type in ['zinb', 'zip']:
            self.dropout = nn.Sequential(
                nn.Linear(action_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, state_dim)
            )
        
        self.apply(weight_init)

    def forward(self, x):
        h1 = F.relu(self.ln1(self.fc1(x)) if self.use_layer_norm else self.fc1(x))
        h2 = F.relu(self.ln2(self.fc2(h1)) if self.use_layer_norm else self.fc2(h1))
        output = F.softmax(self.fc3(h2), dim=-1)
        
        dropout = self.dropout(x) if self.loss_type in ['zinb', 'zip'] else None
        return output, dropout


class LatentODEfunc(nn.Module):
    """
    Latent space ODE function model.
    
    Parameters
    ----------
    n_latent : int
        Latent space dimension
    n_hidden : int
        Hidden layer dimension
    """
    
    def __init__(self, n_latent: int = 10, n_hidden: int = 25):
        super().__init__()
        self.elu = nn.ELU()
        self.fc1 = nn.Linear(n_latent, n_hidden)
        self.fc2 = nn.Linear(n_hidden, n_latent)
        
        self.apply(weight_init)

    def forward(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """
        Compute gradient at time t and state x.
        
        Parameters
        ----------
        t : torch.Tensor
            Time point
        x : torch.Tensor
            Latent state
            
        Returns
        -------
        torch.Tensor
            Gradient value
        """
        out = self.fc1(x)
        out = self.elu(out)
        out = self.fc2(out)
        return out


class NODEMixin:
    """Mixin class providing Neural ODE functionality."""
    
    @staticmethod
    def get_step_size(step_size, t0, t1, n_points):
        """Get ODE solver step size configuration."""
        if step_size is None:
            return {}
        else:
            if step_size == "auto":
                step_size = (t1 - t0) / (n_points - 1)
            return {"step_size": step_size}

    def solve_ode(
        self,
        ode_func: nn.Module,
        z0: torch.Tensor,
        t: torch.Tensor,
        method: str = "rk4",
        step_size: Optional[float] = None,
    ) -> torch.Tensor:
        """
        Solve ODE using torchdiffeq.
        
        Parameters
        ----------
        ode_func : nn.Module
            ODE function model
        z0 : torch.Tensor
            Initial state
        t : torch.Tensor
            Time points
        method : str
            Solution method
        step_size : Optional[float]
            Step size
            
        Returns
        -------
        torch.Tensor
            ODE solution
        """
        options = self.get_step_size(step_size, t[0], t[-1], len(t))
        
        # Move to CPU for ODE solver (if needed)
        cpu_z0 = z0.to("cpu")
        cpu_t = t.to("cpu")
        
        # Solve ODE
        pred_z = odeint(ode_func, cpu_z0, cpu_t, method=method, options=options)
        
        # Move back to original device
        pred_z = pred_z.to(z0.device)
        
        return pred_z


class VAE(nn.Module, NODEMixin):
    """Lorenzian Interpretable ODE Regularization variational Autoencoder"""
    
    def __init__(self, state_dim, hidden_dim, action_dim, i_dim,
                 use_bottleneck_lorentz=True, loss_type='nb', use_layer_norm=True, 
                 use_euclidean_manifold=False, use_ode=False):
        super().__init__()
        
        self.encoder = Encoder(state_dim, hidden_dim, action_dim, use_layer_norm, use_ode)
        self.decoder = Decoder(state_dim, hidden_dim, action_dim, loss_type, use_layer_norm)
        self.latent_encoder = nn.Linear(action_dim, i_dim)
        self.latent_decoder = nn.Linear(i_dim, action_dim)
        self.use_bottleneck_lorentz = use_bottleneck_lorentz
        self.use_euclidean_manifold = use_euclidean_manifold
        self.use_ode = use_ode
        
        # Initialize ODE solver if needed
        if use_ode:
            self.ode_solver = LatentODEfunc(action_dim, hidden_dim)
        
    def forward(self, x):
        # Encode
        if self.use_ode:
            q_z, q_m, q_s, n, t = self.encoder(x)
            
            # Sort by time
            idxs = torch.argsort(t)
            t = t[idxs]
            q_z = q_z[idxs]
            q_m = q_m[idxs]
            q_s = q_s[idxs]
            x = x[idxs]
            
            # Remove duplicate time points
            unique_mask = torch.ones_like(t, dtype=torch.bool)
            unique_mask[1:] = t[1:] != t[:-1]
            
            t = t[unique_mask]
            q_z = q_z[unique_mask]
            q_m = q_m[unique_mask]
            q_s = q_s[unique_mask]
            x = x[unique_mask]
            
            # Solve ODE
            z0 = q_z[0]
            q_z_ode = self.solve_ode(self.ode_solver, z0, t)
            
            # Primary path: q_z through bottleneck
            q_z_clipped = torch.clamp(q_z, -5, 5)
            
            if self.use_euclidean_manifold:
                z_manifold = q_z
            else:
                z_tangent = F.pad(q_z_clipped, (1, 0), value=0)
                z_manifold = exp_map_at_origin(z_tangent)
            
            # Information bottleneck (only for q_z, NOT for q_z_ode)
            le = self.latent_encoder(q_z)
            ld = self.latent_decoder(le)
            
            ld_clipped = torch.clamp(ld, -5, 5)
            
            if self.use_euclidean_manifold:
                ld_manifold = ld
            else:
                if self.use_bottleneck_lorentz:
                    ld_tangent = F.pad(ld_clipped, (1, 0), value=0)
                    ld_manifold = exp_map_at_origin(ld_tangent)
                else:
                    q_z2 = n.sample()
                    q_z2_clipped = torch.clamp(q_z2, -5, 5)
                    z2_tangent = F.pad(q_z2_clipped, (1, 0), value=0)
                    ld_manifold = exp_map_at_origin(z2_tangent)
            
            # Decode all paths
            pred_x, dropout_x = self.decoder(q_z)
            pred_xl, dropout_xl = self.decoder(ld)
            pred_x_ode, dropout_x_ode = self.decoder(q_z_ode)
            
            # Return with ODE outputs
            return (q_z, q_m, q_s, pred_x, le, ld, pred_xl, z_manifold, ld_manifold, 
                    dropout_x, dropout_xl, q_z_ode, pred_x_ode, dropout_x_ode, 
                    x, t)
        
        else:
            # Original non-ODE forward pass
            q_z, q_m, q_s, n = self.encoder(x)
            
            # Primary Lorentz embedding
            q_z_clipped = torch.clamp(q_z, -5, 5)
            
            # Conditional manifold mapping
            if self.use_euclidean_manifold:                    
                # Euclidean: use embeddings directly
                z_manifold = q_z
            else:
                # Lorentz: map to hyperboloid
                z_tangent = F.pad(q_z_clipped, (1, 0), value=0)
                z_manifold = exp_map_at_origin(z_tangent)
            
            # Information bottleneck
            le = self.latent_encoder(q_z)
            ld = self.latent_decoder(le)

            # Conditional manifold mapping for bottleneck path
            ld_clipped = torch.clamp(ld, -5, 5)
            
            if self.use_euclidean_manifold:                    
                # Euclidean: use embeddings directly
                ld_manifold = ld
            else:
                # Lorentz: map based on option
                if self.use_bottleneck_lorentz:
                    ld_tangent = F.pad(ld_clipped, (1, 0), value=0)
                    ld_manifold = exp_map_at_origin(ld_tangent)
                else:
                    q_z2 = n.sample()
                    q_z2_clipped = torch.clamp(q_z2, -5, 5)
                    z2_tangent = F.pad(q_z2_clipped, (1, 0), value=0)
                    ld_manifold = exp_map_at_origin(z2_tangent)
            
            pred_x, dropout_x = self.decoder(q_z)
            pred_xl, dropout_xl = self.decoder(ld)
            
            # Return manifold embeddings instead of z_lorentz/ld_lorentz
            return q_z, q_m, q_s, pred_x, le, ld, pred_xl, z_manifold, ld_manifold, dropout_x, dropout_xl
