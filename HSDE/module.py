
"""
Neural network modules for HSDE - SDE VERSION

Core components:
- Encoder: Maps high-dimensional input to latent distribution (UNCHANGED)
- Decoder: Reconstructs input from latent codes (UNCHANGED)
- LatentSDEfunc: Stochastic differential equation for trajectory dynamics (REPLACES LatentODEfunc)
- VAE: Full variational autoencoder with optional SDE regularization (MODIFIED)

=== CHANGES FROM ODE VERSION ===
1. ode_functions.py: ODE classes replaced with SDE classes
2. mixin.py: solve_ode() replaced with solve_sde() (uses torchsde.sdeint)
3. modules.py: VAE forward pass compatible with SDE solver output
4. All encoder/decoder logic remains identical
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from typing import Optional, Tuple
from .utils import exp_map_at_origin
from .mixin import SDEMixin  # CHANGED: NODEMixin → SDEMixin
from .sde_functions import create_sde_func  # CHANGED: ode_functions → sde_functions


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
    - Optional: time/pseudotime prediction for SDE mode
    
    ⚠️ UNCHANGED from ODE version
    """
    
    def __init__(
        self, 
        state_dim: int, 
        hidden_dim: int, 
        action_dim: int, 
        use_layer_norm: bool = True, 
        use_sde: bool = False,  # CHANGED: use_ode → use_sde
        encoder_type: str = 'mlp',
        attn_embed_dim: int = 64,
        attn_num_heads: int = 4,
        attn_num_layers: int = 2,
        attn_seq_len: int = 32,
    ):
        super().__init__()
        self.use_layer_norm = use_layer_norm
        self.use_sde = use_sde  # CHANGED: use_ode → use_sde
        self.encoder_type = encoder_type.lower() if isinstance(encoder_type, str) else 'mlp'
        
        # Choose encoder implementation
        if self.encoder_type == 'mlp':
            # Main encoder layers (MLP)
            self.fc1 = nn.Linear(state_dim, hidden_dim)
            self.fc2 = nn.Linear(hidden_dim, hidden_dim)
            self.fc3 = nn.Linear(hidden_dim, action_dim * 2)  # mu and log_var
        else:
            # Self-attention / Transformer-based encoder
            self.attn_seq_len = attn_seq_len
            self.attn_embed_dim = attn_embed_dim
            self.input_proj = nn.Linear(state_dim, attn_seq_len * attn_embed_dim)

            encoder_layer = nn.TransformerEncoderLayer(
                d_model=attn_embed_dim,
                nhead=attn_num_heads,
                dim_feedforward=max(attn_embed_dim * 4, 128),
                activation='relu',
                batch_first=False,
            )
            self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=attn_num_layers)
            self.attn_pool_fc = nn.Linear(attn_embed_dim, action_dim * 2)
        
        # Optional layer normalization (MLP path)
        if use_layer_norm and self.encoder_type == 'mlp':
            self.ln1 = nn.LayerNorm(hidden_dim)
            self.ln2 = nn.LayerNorm(hidden_dim)
        # Optional layernorm for attention outputs
        if use_layer_norm and self.encoder_type != 'mlp':
            self.attn_ln = nn.LayerNorm(attn_embed_dim)
        
        # Time encoder for SDE mode (unchanged logic)
        if use_sde:
            time_in_dim = hidden_dim if self.encoder_type == 'mlp' else attn_embed_dim
            self.time_encoder = nn.Sequential(
                nn.Linear(time_in_dim, 1),
                nn.Sigmoid(),  # Normalize time to [0, 1]
            )
        
        self.apply(weight_init)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Normal, Optional[torch.Tensor]]:
        """
        Encode input to latent distribution.
        
        Output format unchanged from ODE version
        """
        if self.encoder_type == 'mlp':
            h1 = self.fc1(x)
            if self.use_layer_norm:
                h1 = self.ln1(h1)
            h1 = F.relu(h1)

            h2 = self.fc2(h1)
            if self.use_layer_norm:
                h2 = self.ln2(h2)
            h2 = F.relu(h2)

            output = self.fc3(h2)
        else:
            proj = self.input_proj(x)
            bsz = proj.size(0)
            seq = proj.view(bsz, self.attn_seq_len, self.attn_embed_dim)
            seq = seq.transpose(0, 1)
            seq_out = self.transformer(seq)
            seq_out = seq_out.transpose(0, 1)

            if self.use_layer_norm:
                seq_out = self.attn_ln(seq_out)

            pooled = seq_out.mean(dim=1)
            output = self.attn_pool_fc(pooled)
        
        q_m, q_s = torch.chunk(output, 2, dim=-1)
        q_m = torch.clamp(q_m, -10, 10)
        q_s = torch.clamp(q_s, -10, 10)
        s = torch.clamp(F.softplus(q_s) + 1e-6, min=1e-6, max=5.0)
        
        n = Normal(q_m, s)
        q_z = n.rsample()
        
        if self.use_sde:
            if hasattr(self, 'time_encoder'):
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
    
    ⚠️ UNCHANGED from ODE version
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

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Decode latent code to output distribution parameters.
        
        ⚠️ UNCHANGED from ODE version
        """
        h1 = self.fc1(x)
        if self.use_layer_norm:
            h1 = self.ln1(h1)
        h1 = F.relu(h1)
        
        h2 = self.fc2(h1)
        if self.use_layer_norm:
            h2 = self.ln2(h2)
        h2 = F.relu(h2)
        
        output = F.softmax(self.fc3(h2), dim=-1)
        dropout = self.dropout(x) if self.loss_type in ['zinb', 'zip'] else None
        
        return output, dropout


class VAE(nn.Module, SDEMixin):  # CHANGED: NODEMixin → SDEMixin
    """
    HSDE: Hyperbolic-wrapped SDE VAE.

    Combines VAE with geometric regularization on Lorentz/Euclidean manifolds
    and optional Neural SDE dynamics for continuous trajectory learning.
    
    KEY CHANGES from ODE version:
    1. use_ode → use_sde parameter
    2. ode_type, ode_solver_method, etc. → sde_type, sde_solver_method
    3. create_ode_func → create_sde_func
    4. solve_ode() → solve_sde() in forward pass
    5. All other logic (encoder, decoder, manifolds, losses) UNCHANGED
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
        use_sde: bool = False,  # CHANGED: use_ode → use_sde
        device: torch.device = None,
        encoder_type: str = 'mlp',
        attn_embed_dim: int = 64,
        attn_num_heads: int = 4,
        attn_num_layers: int = 2,
        attn_seq_len: int = 32,
        # SDE-specific parameters (replaces ODE parameters)
        sde_type: str = 'time_mlp',  # CHANGED: ode_type → sde_type
        sde_time_cond: str = 'concat',  # CHANGED: ode_time_cond → sde_time_cond
        sde_hidden_dim: Optional[int] = None,  # CHANGED: ode_hidden_dim → sde_hidden_dim
        sde_solver_method: str = 'euler',  # CHANGED: ode_solver_method → sde_solver_method (default euler for SDE)
        sde_step_size: Optional[float] = None,  # CHANGED: ode_step_size → sde_step_size
        sde_rtol: Optional[float] = None,  # CHANGED: ode_rtol → sde_rtol
        sde_atol: Optional[float] = None,  # CHANGED: ode_atol → sde_atol
        sde_sde_type: str = 'ito',  # NEW: SDE calculus type ('ito' or 'stratonovich')
        **kwargs
    ):
        super().__init__()
        
        # Core components (UNCHANGED)
        self.encoder = Encoder(
            state_dim,
            hidden_dim,
            action_dim,
            use_layer_norm,
            use_sde,  # CHANGED
            encoder_type=encoder_type,
            attn_embed_dim=attn_embed_dim,
            attn_num_heads=attn_num_heads,
            attn_num_layers=attn_num_layers,
            attn_seq_len=attn_seq_len,
        ).to(device)
        self.decoder = Decoder(state_dim, hidden_dim, action_dim, loss_type, use_layer_norm).to(device)
        self.latent_encoder = nn.Linear(action_dim, i_dim).to(device)
        self.latent_decoder = nn.Linear(i_dim, action_dim).to(device)
        
        # Configuration (CHANGED)
        self.use_bottleneck_lorentz = use_bottleneck_lorentz
        self.use_euclidean_manifold = use_euclidean_manifold
        self.use_sde = use_sde  # CHANGED: use_ode → use_sde
        self.sde_solver_method = sde_solver_method  # CHANGED
        self.sde_step_size = sde_step_size  # CHANGED
        self.sde_rtol = sde_rtol  # CHANGED
        self.sde_atol = sde_atol  # CHANGED
        self.sde_sde_type = sde_sde_type  # NEW
        
        # Initialize SDE solver if needed (CHANGED)
        if use_sde:
            sde_n_hidden = sde_hidden_dim if sde_hidden_dim is not None else hidden_dim
            self.sde_solver = create_sde_func(
                sde_type=sde_type,  # CHANGED
                n_latent=action_dim,
                n_hidden=sde_n_hidden,
                time_cond=sde_time_cond  # CHANGED
                , sde_calculus=sde_sde_type
            ).to(device)
            self.sde_type = sde_type  # CHANGED
    
    def forward(self, x: torch.Tensor):
        """
        Forward pass through full VAE with optional SDE.
        
        ⚠️ Logic UNCHANGED from ODE version, only solver call differs
        """
        
        if self.use_sde:  # CHANGED
            return self._forward_sde(x)  # CHANGED
        else:
            return self._forward_standard(x)
    
    def _forward_standard(self, x: torch.Tensor) -> Tuple:
        """
        Standard VAE forward pass without SDE.
        
        ⚠️ UNCHANGED from ODE version
        """
        q_z, q_m, q_s, n = self.encoder(x)
        
        q_z_clipped = torch.clamp(q_z, -5, 5)
        
        if self.use_euclidean_manifold:
            z_manifold = q_z
        else:
            z_tangent = F.pad(q_z_clipped, (1, 0), value=0)
            z_manifold = exp_map_at_origin(z_tangent)
        
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
        
        pred_x, dropout_x = self.decoder(q_z)
        pred_xl, dropout_xl = self.decoder(ld)
        
        return q_z, q_m, q_s, pred_x, le, ld, pred_xl, z_manifold, ld_manifold, dropout_x, dropout_xl
    
    def _forward_sde(self, x: torch.Tensor) -> Tuple:  # CHANGED: _forward_ode → _forward_sde
        """
        SDE-augmented forward pass.
        
        Steps:
        1. Encode with time prediction
        2. Sort by pseudotime
        3. Solve SDE trajectory
        4. Generate predictions from both VAE and SDE paths
        
        ⚠️ Logic UNCHANGED from ODE version, only solver call differs (solve_sde vs solve_ode)
        """
        # Encode with time (UNCHANGED)
        q_z, q_m, q_s, n, t = self.encoder(x)
        
        # Sort by time (UNCHANGED)
        idxs = torch.argsort(t)
        t = t[idxs]
        q_z = q_z[idxs]
        q_m = q_m[idxs]
        q_s = q_s[idxs]
        x = x[idxs]
        
        # Remove duplicate time points (UNCHANGED)
        unique_mask = torch.ones_like(t, dtype=torch.bool)
        if len(t) > 1:
            unique_mask[1:] = t[1:] != t[:-1]
        
        t = t[unique_mask]
        q_z = q_z[unique_mask]
        q_m = q_m[unique_mask]
        q_s = q_s[unique_mask]
        x = x[unique_mask]
        
        # Solve SDE trajectory (CHANGED: solve_ode → solve_sde)
        z0 = q_z[0].unsqueeze(0)
        if hasattr(self.sde_solver, 'reset_hidden'):
            self.sde_solver.reset_hidden()
        q_z_sde = self.solve_sde(  # CHANGED: solve_ode → solve_sde
            self.sde_solver, z0, t,
            method=self.sde_solver_method,  # CHANGED
            step_size=self.sde_step_size,  # CHANGED
            rtol=self.sde_rtol,  # CHANGED
            atol=self.sde_atol,  # CHANGED
        ).squeeze(1)
        
        # Rest of forward pass (UNCHANGED)
        q_z_clipped = torch.clamp(q_z, -5, 5)
        
        if self.use_euclidean_manifold:
            z_manifold = q_z
        else:
            z_tangent = F.pad(q_z_clipped, (1, 0), value=0)
            z_manifold = exp_map_at_origin(z_tangent)
        
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
        pred_x_sde, dropout_x_sde = self.decoder(q_z_sde)  # CHANGED: q_z_ode → q_z_sde
        
        return (q_z, q_m, q_s, pred_x, le, ld, pred_xl, z_manifold, ld_manifold, 
                dropout_x, dropout_xl, q_z_sde, pred_x_sde, dropout_x_sde,  # CHANGED: q_z_ode → q_z_sde
                x, t)



