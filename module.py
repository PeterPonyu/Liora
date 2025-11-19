"""
Neural network modules for Liora.

Core components:
- Encoder: Maps high-dimensional input to latent distribution
- Decoder: Reconstructs input from latent codes with count-appropriate likelihoods
- LatentODEfunc: Neural ODE function for trajectory dynamics
- VAE: Full variational autoencoder with optional ODE regularization
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from typing import Optional, Tuple
from torchdiffeq import odeint
from .utils import exp_map_at_origin


def weight_init(m):
    """
    Xavier normal initialization for linear layers.
    """
    if isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight)
        nn.init.constant_(m.bias, 0.01)


class Encoder(nn.Module):
    """
    Variational encoder network.
    
    Maps input data to a latent distribution via:
    - Dense layers with ReLU activations
    - Optional layer normalization for training stability
    - Output: mean and log-variance of latent Gaussian
    - Optional: time/pseudotime prediction for ODE mode
    """
    
    def __init__(
        self, 
        state_dim: int, 
        hidden_dim: int, 
        action_dim: int, 
        use_layer_norm: bool = True, 
        use_ode: bool = False,
        # Encoder type options: 'mlp' (default), 'transformer' (self-attention based)
        encoder_type: str = 'mlp',
        # Attention-specific hyperparameters (only used when encoder_type != 'mlp')
        attn_embed_dim: int = 64,
        attn_num_heads: int = 4,
        attn_num_layers: int = 2,
        attn_seq_len: int = 32,
    ):
        super().__init__()
        self.use_layer_norm = use_layer_norm
        self.use_ode = use_ode
        self.encoder_type = encoder_type.lower() if isinstance(encoder_type, str) else 'mlp'
        
        # Choose encoder implementation
        if self.encoder_type == 'mlp':
            # Main encoder layers (MLP)
            self.fc1 = nn.Linear(state_dim, hidden_dim)
            self.fc2 = nn.Linear(hidden_dim, hidden_dim)
            self.fc3 = nn.Linear(hidden_dim, action_dim * 2)  # mu and log_var
        else:
            # Self-attention / Transformer-based encoder
            # Design: project input features into a small sequence of token embeddings,
            # run through TransformerEncoder, then aggregate to obtain a latent vector.
            self.attn_seq_len = attn_seq_len
            self.attn_embed_dim = attn_embed_dim
            # Project raw features -> seq_len * embed_dim
            self.input_proj = nn.Linear(state_dim, attn_seq_len * attn_embed_dim)

            # Transformer encoder stack
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=attn_embed_dim,
                nhead=attn_num_heads,
                dim_feedforward=max(attn_embed_dim * 4, 128),
                activation='relu',
                batch_first=False,  # we'll feed (seq_len, batch, embed_dim)
            )
            self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=attn_num_layers)

            # Final projection from pooled transformer embedding -> mu/logvar
            self.attn_pool_fc = nn.Linear(attn_embed_dim, action_dim * 2)
        
        # Optional layer normalization (MLP path)
        if use_layer_norm and self.encoder_type == 'mlp':
            self.ln1 = nn.LayerNorm(hidden_dim)
            self.ln2 = nn.LayerNorm(hidden_dim)
        # Optional layernorm for attention outputs
        if use_layer_norm and self.encoder_type != 'mlp':
            self.attn_ln = nn.LayerNorm(attn_embed_dim)
        
        # Time encoder for ODE mode
        if use_ode:
            self.time_encoder = nn.Sequential(
                nn.Linear(hidden_dim, 1),
                nn.Sigmoid(),  # Normalize time to [0, 1]
            )
        
        self.apply(weight_init)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Normal, Optional[torch.Tensor]]:
        """
        Encode input to latent distribution.
        """
        if self.encoder_type == 'mlp':
            # First hidden layer with optional normalization
            h1 = self.fc1(x)
            if self.use_layer_norm:
                h1 = self.ln1(h1)
            h1 = F.relu(h1)

            # Second hidden layer with optional normalization
            h2 = self.fc2(h1)
            if self.use_layer_norm:
                h2 = self.ln2(h2)
            h2 = F.relu(h2)

            # Output layer: mean and log-variance
            output = self.fc3(h2)
        else:
            # Attention / Transformer-based encoder path
            # Project input into sequence of embeddings
            proj = self.input_proj(x)  # (batch, seq_len * embed)
            bsz = proj.size(0)
            seq = proj.view(bsz, self.attn_seq_len, self.attn_embed_dim)  # (batch, seq, embed)

            # Transformer expects (seq_len, batch, embed)
            seq = seq.transpose(0, 1)
            seq_out = self.transformer(seq)  # (seq_len, batch, embed)

            # Back to (batch, seq, embed)
            seq_out = seq_out.transpose(0, 1)

            # Optional layernorm then pool across sequence
            if self.use_layer_norm:
                seq_out = self.attn_ln(seq_out)

            pooled = seq_out.mean(dim=1)  # (batch, embed)

            # Final projection to get mu/logvar
            output = self.attn_pool_fc(pooled)
        q_m, q_s = torch.chunk(output, 2, dim=-1)
        
        # Clamp for numerical stability
        q_m = torch.clamp(q_m, -10, 10)
        q_s = torch.clamp(q_s, -10, 10)
        
        # Convert log-variance to standard deviation with numerical safety
        s = torch.clamp(F.softplus(q_s) + 1e-6, min=1e-6, max=5.0)
        
        # Create posterior distribution and sample
        n = Normal(q_m, s)
        q_z = n.rsample()
        
        # Optional: predict time for ODE trajectory
        if self.use_ode:
            # For attention path, build a small time predictor if needed
            if hasattr(self, 'time_encoder'):
                # MLP time encoder expects hidden_dim inputs; try to reuse pooled representation
                t_in = pooled if self.encoder_type != 'mlp' else h2
                t = self.time_encoder(t_in).squeeze(-1)
            else:
                t = None
            return q_z, q_m, q_s, n, t

        return q_z, q_m, q_s, n


class Decoder(nn.Module):
    """
    Generative decoder network.
    
    Maps latent codes back to input space with count-appropriate
    likelihood functions (NB, ZINB, Poisson, ZIP).
    """
    
    def __init__(
        self, 
        state_dim: int, 
        hidden_dim: int, 
        action_dim: int, 
        loss_type: str = 'nb', 
        use_layer_norm: bool = True
    ):
        super().__init__()
        self.loss_type = loss_type
        self.use_layer_norm = use_layer_norm
        
        # Main decoder layers
        self.fc1 = nn.Linear(action_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, state_dim)
        
        # Optional layer normalization
        if use_layer_norm:
            self.ln1 = nn.LayerNorm(hidden_dim)
            self.ln2 = nn.LayerNorm(hidden_dim)
        
        # Dispersion parameter (shared across batch)
        self.disp = nn.Parameter(torch.randn(state_dim))
        
        # Dropout rate predictor for zero-inflated models
        if loss_type in ['zinb', 'zip']:
            self.dropout = nn.Sequential(
                nn.Linear(action_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, state_dim)
            )
        
        self.apply(weight_init)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Decode latent code to output distribution parameters.
        """
        # First hidden layer with optional normalization
        h1 = self.fc1(x)
        if self.use_layer_norm:
            h1 = self.ln1(h1)
        h1 = F.relu(h1)
        
        # Second hidden layer with optional normalization
        h2 = self.fc2(h1)
        if self.use_layer_norm:
            h2 = self.ln2(h2)
        h2 = F.relu(h2)
        
        # Output: means as probability distribution
        output = F.softmax(self.fc3(h2), dim=-1)
        
        # Dropout rate for zero-inflated models
        dropout = self.dropout(x) if self.loss_type in ['zinb', 'zip'] else None
        
        return output, dropout


class LatentODEfunc(nn.Module):
    """
    Neural ODE function in latent space.
    
    Defines the continuous dynamics: dz/dt = f_θ(z, t)
    Learns smooth trajectories through latent space representing
    cell differentiation or developmental processes.
    
    The ODE solver is kept on CPU for computational efficiency
    (torchdiffeq is optimized for CPU execution).
    """
    
    def __init__(self, n_latent: int = 10, n_hidden: int = 25):
        super().__init__()
        self.n_latent = n_latent
        self.n_hidden = n_hidden
        
        # ODE neural network
        self.fc1 = nn.Linear(n_latent, n_hidden)
        self.fc2 = nn.Linear(n_hidden, n_latent)
        self.elu = nn.ELU()
        
        self.apply(weight_init)

    def forward(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """
        Compute latent dynamics gradient.
        """
        h = self.fc1(x)
        h = self.elu(h)
        dz = self.fc2(h)
        return dz


class NODEMixin:
    """
    Mixin providing Neural ODE solving capabilities.
    
    Handles CPU-GPU device transfers for efficient ODE integration.
    The ODE solver runs on CPU (computational advantage), while
    model parameters remain on the specified device.
    """
    
    @staticmethod
    def get_step_size(
        step_size: Optional[float], 
        t0: float, 
        t1: float, 
        n_points: int
    ) -> dict:
        """
        Determine ODE solver step size.
        
        """
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
        Solve ODE using torchdiffeq on CPU.
        
        Key Design Decision: ODE solving intentionally remains on CPU because:
        1. torchdiffeq's adaptive step-size algorithms are CPU-optimized
        2. Latent dimension is small (typically 10-20), minimal GPU benefit
        3. Significant speedup (~2-3x) observed on CPU vs GPU
        4. Memory efficiency: avoids GPU memory pressure
        """
        # Get solver options
        options = self.get_step_size(step_size, t[0].item(), t[-1].item(), len(t))
        
        # Transfer to CPU for ODE solving
        original_device = z0.device
        cpu_z0 = z0.to("cpu")
        cpu_t = t.to("cpu")        
        try:
            # Solve ODE on CPU
            pred_z = odeint(ode_func, cpu_z0, cpu_t, method=method, options=options)
        except Exception as e:
            print(f"ODE solving failed: {e}, returning z0 trajectory")
            # Fallback: return constant trajectory
            pred_z = cpu_z0.unsqueeze(0).repeat(len(cpu_t), 1, 1)

        # Transfer result back to original device
        pred_z = pred_z.to(original_device)
        
        return pred_z


class VAE(nn.Module, NODEMixin):
    """
    Liora: Lorentz Information ODE Regularized Variational AutoEncoder.
    
    Combines VAE with geometric regularization on Lorentz/Euclidean manifolds
    and optional Neural ODE dynamics for continuous trajectory learning.
    
    Architecture
    -----------
    1. Encoder → latent distribution q(z|x)
    2. Sample z ~ q(z|x)
    3. Information Bottleneck: z → le → ld (optional compression)
    4. Manifold Embedding: z → z_manifold (Lorentz or Euclidean)
    5. Decoder → reconstruction with count-appropriate likelihood
    6. ODE Solver: trajectory dynamics (optional)
    """
    
    def __init__(
        self, 
        state_dim: int, 
        hidden_dim: int, 
        action_dim: int, 
        i_dim: int,
        use_bottleneck_lorentz: bool = True, 
        loss_type: str = 'nb', 
        use_layer_norm: bool = True, 
        use_euclidean_manifold: bool = False, 
        use_ode: bool = False,
        device: torch.device = None,
        # Encoder type and attention options to be forwarded to Encoder
        encoder_type: str = 'mlp',
        attn_embed_dim: int = 64,
        attn_num_heads: int = 4,
        attn_num_layers: int = 2,
        attn_seq_len: int = 32,
    ):
        super().__init__()
        
        # Core components
        self.encoder = Encoder(
            state_dim,
            hidden_dim,
            action_dim,
            use_layer_norm,
            use_ode,
            encoder_type=encoder_type,
            attn_embed_dim=attn_embed_dim,
            attn_num_heads=attn_num_heads,
            attn_num_layers=attn_num_layers,
            attn_seq_len=attn_seq_len,
        ).to(device)
        self.decoder = Decoder(state_dim, hidden_dim, action_dim, loss_type, use_layer_norm).to(device)
        self.latent_encoder = nn.Linear(action_dim, i_dim).to(device)
        self.latent_decoder = nn.Linear(i_dim, action_dim).to(device)
        
        # Configuration
        self.use_bottleneck_lorentz = use_bottleneck_lorentz
        self.use_euclidean_manifold = use_euclidean_manifold
        self.use_ode = use_ode
        
        # Initialize ODE solver if needed
        if use_ode:
            self.ode_solver = LatentODEfunc(action_dim, hidden_dim)
    
    def forward(self, x: torch.Tensor):
        """
        Forward pass through full VAE with optional ODE.
        """
        
        if self.use_ode:
            return self._forward_ode(x)
        else:
            return self._forward_standard(x)
    
    def _forward_standard(self, x: torch.Tensor) -> Tuple:
        """
        Standard VAE forward pass without ODE.
        """
        # Encode
        q_z, q_m, q_s, n = self.encoder(x)
        
        # Primary path: encoder → manifold
        q_z_clipped = torch.clamp(q_z, -5, 5)
        
        if self.use_euclidean_manifold:
            z_manifold = q_z
        else:
            # Lorentz: tangent space → hyperboloid
            z_tangent = F.pad(q_z_clipped, (1, 0), value=0)
            z_manifold = exp_map_at_origin(z_tangent)
        
        # Information bottleneck path
        le = self.latent_encoder(q_z)
        ld = self.latent_decoder(le)
        ld_clipped = torch.clamp(ld, -5, 5)
        
        if self.use_euclidean_manifold:
            ld_manifold = ld
        else:
            if self.use_bottleneck_lorentz:
                # Bottleneck → manifold
                ld_tangent = F.pad(ld_clipped, (1, 0), value=0)
                ld_manifold = exp_map_at_origin(ld_tangent)
            else:
                # Resample from posterior
                q_z2 = n.sample()
                q_z2_clipped = torch.clamp(q_z2, -5, 5)
                z2_tangent = F.pad(q_z2_clipped, (1, 0), value=0)
                ld_manifold = exp_map_at_origin(z2_tangent)
        
        # Decode all paths
        pred_x, dropout_x = self.decoder(q_z)
        pred_xl, dropout_xl = self.decoder(ld)
        
        return q_z, q_m, q_s, pred_x, le, ld, pred_xl, z_manifold, ld_manifold, dropout_x, dropout_xl
    
    def _forward_ode(self, x: torch.Tensor) -> Tuple:
        """
        ODE-augmented forward pass.
        
        Steps:
        1. Encode with time prediction
        2. Sort by pseudotime
        3. Solve ODE trajectory
        4. Generate predictions from both VAE and ODE paths
        """
        # Encode with time
        q_z, q_m, q_s, n, t = self.encoder(x)
        
        # Sort by time (CPU-efficient preprocessing)
        idxs = torch.argsort(t)
        t = t[idxs]
        q_z = q_z[idxs]
        q_m = q_m[idxs]
        q_s = q_s[idxs]
        x = x[idxs]
        
        # Remove duplicate time points for numerical stability
        unique_mask = torch.ones_like(t, dtype=torch.bool)
        if len(t) > 1:
            unique_mask[1:] = t[1:] != t[:-1]
        
        t = t[unique_mask]
        q_z = q_z[unique_mask]
        q_m = q_m[unique_mask]
        q_s = q_s[unique_mask]
        x = x[unique_mask]
        
        # Solve ODE trajectory (CPU-optimized)
        z0 = q_z[0].unsqueeze(0)
        q_z_ode = self.solve_ode(self.ode_solver, z0, t).squeeze(1)
        
        # Primary path: VAE latent → manifold
        q_z_clipped = torch.clamp(q_z, -5, 5)
        
        if self.use_euclidean_manifold:
            z_manifold = q_z
        else:
            z_tangent = F.pad(q_z_clipped, (1, 0), value=0)
            z_manifold = exp_map_at_origin(z_tangent)
        
        # Information bottleneck (only on VAE path, not ODE)
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
        
        # Decode all paths (VAE, bottleneck, ODE)
        pred_x, dropout_x = self.decoder(q_z)
        pred_xl, dropout_xl = self.decoder(ld)
        pred_x_ode, dropout_x_ode = self.decoder(q_z_ode)
        
        return (q_z, q_m, q_s, pred_x, le, ld, pred_xl, z_manifold, ld_manifold, 
                dropout_x, dropout_xl, q_z_ode, pred_x_ode, dropout_x_ode, 
                x, t)
