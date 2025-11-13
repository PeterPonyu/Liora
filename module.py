
"""Neural network modules for LiVAE."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from .utils import exp_map_at_origin


def weight_init(m):
    """Xavier initialization for linear layers."""
    if isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight)
        nn.init.constant_(m.bias, 0.01)


class Encoder(nn.Module):
    """Encoder network with optional layer normalization."""
    
    def __init__(self, state_dim, hidden_dim, action_dim, use_layer_norm=True):
        super().__init__()
        self.use_layer_norm = use_layer_norm
        
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim * 2)
        
        if use_layer_norm:
            self.ln1 = nn.LayerNorm(hidden_dim)
            self.ln2 = nn.LayerNorm(hidden_dim)
        
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


class VAE(nn.Module):
    """Variational Autoencoder with Lorentz regularization."""
    
    def __init__(self, state_dim, hidden_dim, action_dim, i_dim,
                 use_bottleneck_lorentz=True, loss_type='nb', use_layer_norm=True, use_euclidean_manifold=False):
        super().__init__()
        
        self.encoder = Encoder(state_dim, hidden_dim, action_dim, use_layer_norm)
        self.decoder = Decoder(state_dim, hidden_dim, action_dim, loss_type, use_layer_norm)
        self.latent_encoder = nn.Linear(action_dim, i_dim)
        self.latent_decoder = nn.Linear(i_dim, action_dim)
        self.use_bottleneck_lorentz = use_bottleneck_lorentz
        self.use_euclidean_manifold = use_euclidean_manifold
        
    def forward(self, x):
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
